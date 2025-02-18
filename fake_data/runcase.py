import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
#from examples.ZZget_args import get_name
from synthetic import CSBM, edge_homo_ratio
import math
import torch
import numpy as np
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import  add_self_loops
from torch_geometric.utils import get_laplacian
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2
class ChebnetII_prop(MessagePassing):
    def __init__(self, K, name, Init=False, bias=True, **kwargs):
        super(ChebnetII_prop, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.name = name
        self.Init = Init
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

    def forward(self, x, edge_index, edge_weight=None):
        # coe_tmp=F.relu(self.temp)
        coe_tmp = self.temp
        coe = coe_tmp.clone()

        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        if self.name == "fb100":
            edge_index_tilde, norm_tilde = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        else:
            # L=I-D^(-0.5)AD^(-0.5)
            edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                               num_nodes=x.size(self.node_dim))

            # L_tilde=L-I
            edge_index_tilde, norm_tilde = add_self_loops(edge_index1, norm1, fill_value=-1.0,
                                                          num_nodes=x.size(self.node_dim))

        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)
        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)



        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    print("our")
    return test_our(data, z.detach())


def test_our(data, embeds):
    results = []
    for i in range(data.train_mask.shape[1]):
        print('MASK:', i)
        train_mask = data.train_mask[:, i]
        val_mask = data.val_mask[:, i]
        test_mask = data.test_mask[:, i]

        train_idx_tmp = train_mask
        val_idx_tmp = val_mask
        test_idx_tmp = test_mask
        train_embs = embeds[train_idx_tmp]
        val_embs = embeds[val_idx_tmp]
        test_embs = embeds[test_idx_tmp]

        label = data.y.to(embeds.device)

        train_labels = label[train_idx_tmp]
        val_labels = label[val_idx_tmp]
        test_labels = label[test_idx_tmp]
        import torch.nn as nn
        class LogReg(nn.Module):
            def __init__(self, hid_dim, out_dim):
                super(LogReg, self).__init__()
                self.fc = nn.Linear(hid_dim, out_dim)

            def forward(self, x):
                ret = self.fc(x)
                return ret

        ''' Linear Evaluation '''
        logreg = LogReg(train_embs.shape[1], (torch.max(data.y) + 1).item())
        opt = torch.optim.Adam(logreg.parameters(), lr=1e-2, weight_decay=1e-5)

        logreg = logreg.to(embeds.device)
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0
        eval_acc = 0

        for epoch in range(600):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = torch.argmax(logits, dim=1)
            train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with torch.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = torch.argmax(val_logits, dim=1)
                test_preds = torch.argmax(test_logits, dim=1)

                val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    eval_acc = test_acc

                print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc,
                                                                                         test_acc))
        results.append(eval_acc.item())
        print(f'Validation Accuracy: {best_val_acc}, Test Accuracy: {eval_acc}')
    print("mean:")
    import statistics
    print(statistics.mean(results) * 100)
    print("std:")
    print(statistics.stdev(results) * 100)

    return round(statistics.mean(results) * 100,2), round(statistics.stdev(results) * 100,2)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,dropout=.5, is_bns=True):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.is_bns = is_bns
        if is_bns:
            self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if is_bns:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if is_bns:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.is_bns:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data, input_tensor=True):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        if self.is_bns:
            for i, lin in enumerate(self.lins[:-1]):
                x = lin(x)
                x = F.relu(x, inplace=True)
                x = self.bns[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
            return x
        else:
            for i, lin in enumerate(self.lins[:-1]):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = lin(x)
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
            return x


class ChebNetII(torch.nn.Module):
    def __init__(self, num_features, hidden_channels,K):
        super(ChebNetII, self).__init__()
        self.name = '11'
        self.mlp = MLP(num_features, hidden_channels, hidden_channels, 2, 0.5,
                       is_bns=False)
        self.prop1 = ChebnetII_prop(K, self.name)
        self.dprate = 0.5
        self.dropout = 0.5
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x,edge_index,edge_att):

        data = x
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)

        return x


def main():
    device = torch.device('cuda')
    dataname = 'Actor'
    # 6666
    test_sequence = ['0.00','0.10','0.20','0.30','0.40','0.50','0.60','0.70','0.80','0.90','1.00']
    # test_sequence = ['0.00']
    hidden = 1024
    experi_name = 'High-Pass'
    # experi_name = 'Low-Pass'
    # experi_name = 'ChebNet'
    # experi_name = 'GCN'
    if_record = True
    gconv = GConv(input_dim=3000, hidden_dim=hidden, activation=torch.nn.ReLU, num_layers=2).to(device)
    K = 6
    gconv = ChebNetII(3000, hidden, K).to(device)
    # gconv.prop1.temp = torch.nn.Parameter(torch.linspace(2, 0, steps=K + 1), requires_grad=False)
    gconv.prop1.temp = torch.nn.Parameter(torch.linspace(0, 2, steps=K + 1), requires_grad=False)

    ratio_map = {
        '0.00': 0.00001, '0.05': 0.05, '0.10': 0.10, '0.15': 0.15,
        '0.20': 0.20, '0.25': 0.25, '0.30': 0.30, '0.35': 0.35,
        '0.40': 0.40, '0.45': 0.45, '0.50': 0.50, '0.55': 0.55,
        '0.60': 0.60, '0.65': 0.65, '0.70': 0.70, '0.75': 0.75,
        '0.80': 0.80, '0.85': 0.85, '0.90': 0.90, '0.95': 0.95, '1.00': 0.99999
    }
    results = []
    for choose_ratio in test_sequence:
        ratio = ratio_map[choose_ratio]
        data = CSBM(n=3000, d=5, ratio=ratio, p=3000, mu=1,
                    train_prop=.025, valid_prop=.025, num_masks=5)
        print("Homophily Ratio: ", edge_homo_ratio(data))
        data = data.to(device)

        aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
        aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=hidden, proj_dim=hidden).to(device)
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

        optimizer = Adam(encoder_model.parameters(), lr=0.01)
        epochs = 100
        with tqdm(total=epochs, desc='(T)') as pbar:
            for epoch in range(1, epochs+1):
                loss = train(encoder_model, contrast_model, data, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()

        result = test(encoder_model, data)
        from colorama import Fore
        # 颜色

        print(Fore.GREEN + "Homophily Ratio: ", edge_homo_ratio(data))
        print(Fore.BLUE + "Result ", result)
        print(Fore.RESET)
        results.append(result)

    print(results)

    if if_record:
        import csv

        csv_file = 'results' + str(len(test_sequence)) + '.csv'

        # Prepare the row data
        row_data = [experi_name] + results

        # Check if the CSV file exists
        if not osp.exists(csv_file):
            # Create the CSV file and write the header
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Creating a header with 'Experiment Name' and then placeholders for each result
                header = ['Experiment Name'] + [i for i in test_sequence]
                writer.writerow(header)
                writer.writerow(row_data)
        else:
            # If the file exists, append the new data
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)

        sample_csv_path = csv_file
        import pandas as pd
        def calculate_average_rank_from_csv(csv_path):
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(csv_path)

            # Ranking the models within each dataset and storing the ranks in a new dataframe
            ranks_df = df.set_index('Experiment Name').rank(ascending=False, method='min', axis=0)

            # Calculating the average rank for each model and rounding to two decimal places
            df['Average Rank'] = ranks_df.mean(axis=1).round(2).values

            return df

        # Calculating average rank from the CSV file
        df_with_ranks = calculate_average_rank_from_csv(sample_csv_path)

        print(df_with_ranks)


# {\scriptstyle\pm0.56}


if __name__ == '__main__':
    main()
