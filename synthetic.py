import os
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from load_data import *
from .load_data import rand_train_test_idx, index_to_mask
from torch_geometric.utils import get_laplacian, to_dense_adj


def edge_homo_ratio(data):
    sum = 0
    for i in range(len(data.edge_index[0])):
        if data.y[data.edge_index[0][i]] == data.y[data.edge_index[1][i]]:
            sum += 1
    return sum / len(data.edge_index[0])


def remove_homo_edges(data, remove_homo_ratio):
    edge_index_T = data.edge_index.T.cpu().numpy()
    homo_edge, hetero_edge = [], []

    for i in edge_index_T:
        if data.y[i[0]] == data.y[i[1]]:
            homo_edge.append(i)
        else:
            hetero_edge.append(i)

    hetero_edge = np.array(hetero_edge)
    homo_edge = np.array(homo_edge)
    np.random.shuffle(homo_edge)

    new_edge_index = np.concatenate((hetero_edge, homo_edge[:int(len(homo_edge) * (1 - remove_homo_ratio))]), 0)
    data.edge_index = torch.from_numpy(new_edge_index).T
    return data


def remove_hetero_edges(data, remove_hetero_ratio):
    edge_index_T = data.edge_index.T.cpu().numpy()
    homo_edge, hetero_edge = [], []

    for i in edge_index_T:
        if data.y[i[0]] == data.y[i[1]]:
            homo_edge.append(i)
        else:
            hetero_edge.append(i)

    hetero_edge = np.array(hetero_edge)
    homo_edge = np.array(homo_edge)
    np.random.shuffle(hetero_edge)

    new_edge_index = np.concatenate((homo_edge, hetero_edge[:int(len(hetero_edge) * (1 - remove_hetero_ratio))]), 0)
    data.edge_index = torch.from_numpy(new_edge_index).T
    return data


def synthetic_dataset(data, target_ratio):
    num_classes = data.y.max().item() + 1
    edge_index_T = data.edge_index.T.cpu().numpy()
    homo_edges, hetero_edges = [], []

    for i in edge_index_T:
        if data.y[i[0]] == data.y[i[1]]:
            homo_edges.append(i)
        else:
            hetero_edges.append(i)

    homo_edges, hetero_edges = np.array(homo_edges), np.array(hetero_edges)
    num_edges, num_homo_edges, num_hetero_edges = edge_index_T.shape[0], homo_edges.shape[0], hetero_edges.shape[0]
    ratio = num_homo_edges / num_edges

    if target_ratio > ratio:
        change = int(target_ratio * num_edges - num_homo_edges)

        np.random.shuffle(hetero_edges)
        new_hetero_edges = hetero_edges[:num_hetero_edges-change]

        nodes_each_class = [[] for _ in range(num_classes)]
        for i in range(len(data.y)):
            nodes_each_class[data.y[i]].append(i)

        new_homo_edges = []
        for i in nodes_each_class:
            for j in range(int(change / num_classes)):
                new_homo_edges.append(random.sample(i, 2))

        new_homo_edges = np.array(new_homo_edges)
        new_homo_edges = np.concatenate((new_homo_edges, homo_edges), 0)
        new_edge_index = np.concatenate((new_homo_edges, new_hetero_edges), 0)

    else:
        change = int((num_homo_edges - target_ratio * num_edges) / (1 - target_ratio))
        new_homo_edges = homo_edges[:num_homo_edges - change]
        new_edge_index = np.concatenate((new_homo_edges, hetero_edges), 0)

    new_edge_index = np.unique(new_edge_index, axis=0)
    data.edge_index = torch.from_numpy(new_edge_index).T
    return data


def CSBM(n, d, ratio, p, mu, train_prop=.6, valid_prop=.2, num_masks=5):
    Lambda = np.sqrt(d) * (2 * ratio - 1)
    c_in = d + np.sqrt(d) * Lambda
    c_out = d - np.sqrt(d)*Lambda
    print('c_in: ', c_in, 'c_out: ', c_out)
    y = np.ones(n)
    y[int(n/2)+1:] = -1
    y = np.asarray(y, dtype=int)

    # creating edge_index
    edge_index = [[], []]
    for i in range(n-1):
        for j in range(i+1, n):
            if y[i]*y[j] > 0:
                Flip = np.random.binomial(1, c_in/n)
            else:
                Flip = np.random.binomial(1, c_out/n)
            if Flip > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)

    # creating node features
    x = np.zeros([n, p])
    u = np.random.normal(0, 1/np.sqrt(p), [1, p])
    for i in range(n):
        Z = np.random.normal(0, 1, [1, p])
        x[i] = np.sqrt(mu/n)*y[i]*u + Z/np.sqrt(p)

    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index),
                y=torch.tensor((y + 1) // 2, dtype=torch.int64))

    data.coalesce()

    splits_lst = [rand_train_test_idx(data.y, train_prop=0.6, valid_prop=0.2, test_prop=0.2)
                  for _ in range(num_masks)]
    data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    return data


import torch
from torch_geometric.data import Data

import torch
from torch_geometric.data import Data

import torch
from torch_geometric.data import Data


def merge_graph_data(client_data_list):
    """
    合成一个大的图数据，每个客户端拥有单独的图数据对象，并处理掩码和节点标签。

    参数:
    - client_data_list: list of Data objects，每个对象代表一个客户端的图数据

    返回:
    - combined_data: Data 对象，合成后的大图数据
    - client_info: dict，记录每个客户端的节点编号范围
    """
    # 初始化节点特征、边索引、掩码和标签的列表
    x, edge_index, y = [], [], []
    train_mask, val_mask, test_mask = [], [], []
    node_offset = 0
    client_info = {}

    for i, data in enumerate(client_data_list):
        num_nodes = data.num_nodes

        # 更新节点特征
        x.append(data.x)

        # 更新边索引
        edge_index.append(data.edge_index + node_offset)

        # 更新节点标签，如果存在的话
        if hasattr(data, 'y') and data.y is not None:
            y.append(data.y)
        else:
            y.append(torch.full((num_nodes,), -1, dtype=torch.long))  # 使用-1作为缺失标签的占位符

        # 处理掩码
        train_mask.append(data.train_mask if hasattr(data, 'train_mask') else torch.zeros(num_nodes, dtype=torch.bool))
        val_mask.append(data.val_mask if hasattr(data, 'val_mask') else torch.zeros(num_nodes, dtype=torch.bool))
        test_mask.append(data.test_mask if hasattr(data, 'test_mask') else torch.zeros(num_nodes, dtype=torch.bool))

        # 记录客户端信息
        client_info[f"Client{i:02d}"] = {"data": list(range(node_offset, node_offset + num_nodes))}
        node_offset += num_nodes

    # 合并所有客户端的节点和边
    combined_data = Data(x=torch.cat(x, dim=0), edge_index=torch.cat(edge_index, dim=1),
                         y=torch.cat(y))

    # 转换掩码回torch.BoolTensor并设置到combined_data中
    combined_data.train_mask = torch.cat(train_mask)
    combined_data.val_mask = torch.cat(val_mask)
    combined_data.test_mask = torch.cat(test_mask)

    return combined_data, client_info


import json


def write_client_info_to_json(client_info, json_path="C:/Users/admin\Desktop\easyFL-FLGo\easyFL-FLGo\my_cora_louvain\data.json"):
    """
    将客户端信息写入JSON文件。

    参数:
    - client_info: dict，客户端信息，包括节点的编号
    - json_path: str，JSON文件的保存路径
    """
    # 初始化JSON数据结构
    json_data = {
        "client_names": list(client_info.keys()),
        "transductive": True
    }
    json_data.update(client_info)

    # 写入到JSON文件
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)


def calculate_homophily(edge_index, y):
    """
    计算同配度
    """
    num_edges = edge_index.size(1)
    same_class_edges = (y[edge_index[0]] == y[edge_index[1]]).sum().item()
    homophily = same_class_edges / num_edges
    return homophily

def generate_random_edges(nodes, num_edges, y, same_class):
    """
    生成随机边
    """
    edges = set()
    while len(edges) < num_edges:
        i, j = random.sample(range(nodes), 2)
        if same_class and y[i] == y[j]:
            edges.add((i, j))
        elif not same_class and y[i] != y[j]:
            edges.add((i, j))
    return edges

def reconstruct_edges(data, target_homophily, current_homophily, device='cuda'):
    """
    根据目标同配度重构边
    """
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    num_edges = edge_index.size(1)

    edges = set(map(tuple, edge_index.t().tolist()))
    nodes = len(y)

    # 计算需要增加或删除的同类别边数
    same_class_edges = int(target_homophily * num_edges)
    current_same_class_edges = int(current_homophily * num_edges)
    same_class_diff = same_class_edges - current_same_class_edges

    # 生成随机边
    if same_class_diff > 0:
        new_edges = generate_random_edges(nodes, same_class_diff, y, same_class=True)
        edges.update(new_edges)
    elif same_class_diff < 0:
        same_class_edges_to_remove = abs(same_class_diff)
        same_class_edges = {e for e in edges if y[e[0]] == y[e[1]]}
        for edge in random.sample(same_class_edges, same_class_edges_to_remove):
            edges.remove(edge)

    # 更新 edge_index
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous().to(device)
    data.edge_index = edge_index

    return data