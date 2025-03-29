import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient
from openfgl.model.gcn import *
from function import *

class FedAvgClient(BaseClient):
    """
    FedAvgClient implements the client-side logic for the Federated Averaging (FedAvg) algorithm,
    introduced in the paper "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    by McMahan et al. (2017). This class extends the BaseClient class and manages local training
    and communication with the server.

    The FedAvg algorithm allows clients to train models locally on their data and send the 
    updated model parameters to the server for aggregation, enabling efficient learning in 
    decentralized environments.

    Attributes:
        None (inherits attributes from BaseClient)
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedAvgClient.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedAvgClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device)

    def execute(self, round_id):
        """
        Executes the local training process. This method first synchronizes the local model
        with the global model parameters received from the server, and then trains the model
        on the client's local data.
        """
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)
        
        # self.task.model.eval()
        # and self.client_id in self.args.contam_clients
        # and (round_id + 1) % 10 == 0
        if self.args.aggregation_mode == "lap" and (round_id + 1) % 10 == 0:
            if self.client_id == 0:
                self.args.S_ano_list = []
                print("computing S_ano")
            self.args.S_ano_list.append(compute_S_ano(self.task.data, self.device))

        if self.args.graph_repair and self.client_id in self.args.repair_clients and (round_id + 1) % 10 == 0:
            print(f"client {self.client_id} is repairing...")
            input_dim = self.task.data.x.size(1)
            hid_dim = 64
            output_dim = self.args.classes
            gcn_model = GCN(input_dim, hid_dim, output_dim)
            gcn_model = gcn_model.to(self.device)

            with torch.no_grad():
                for (param, global_param) in zip(gcn_model.parameters(), self.message_pool["server"]["weight"]):
                    param.data.copy_(global_param)

            gcn_model.eval()  # 设置为评估模式
            with torch.no_grad():
                # self.task.data.edge_index = self.task.data.edge_index.to(self.device)
                # self.task.data.x = self.task.data.x.to(self.device)
                self.task.data = self.task.data.to(self.device)
                node_features, logits = gcn_model(self.task.data)
            
            # node_features = node_features.cpu()
            similarity_matrix = compute_client_similarity_matrix(node_features)

            self.task.data.edge_index = modify_edges(similarity_matrix, self.task.data.edge_index, self.task.data.num_nodes, device = self.device)

        # 全局模型在本地数据上推理，丢掉边
        # gcn_model(self.task.data) x,y,edge_index
        # 计算每条边for i in sim_matrix的特征相似度sim(node_features)，全局模型在edge相邻的两个节点推理出来的特征node_features
        # 所有边现在有一个相似度，相似度最小的后30%的边断掉
        # 断掉现有边之后，需要随机采样节点对（目前没有边相连的），看相似度是否大于阈值，大于则相连，补全那30%数量的边
        # self.task.model()
        # if(self.client_id==0):
        #     self.task.data.edge_index[0][0] = 249  # 修改源节点
        #     self.task.data.edge_index[1][0] = 27  # 修改目标节点
        # print("round id:", round_id)
        self.task.train()

    def send_message(self):
        """
        Sends a message to the server containing the model parameters after training
        and the number of samples in the client's dataset.
        """
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
