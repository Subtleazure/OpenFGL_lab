import torch
from openfgl.model.gcn import *
from fake_data.synthetic import CSBM
from openfgl.data.processing import *
import os
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data


def load_client_list(client_data_dir, client_num=10):
    """
    加载客户端数据
    """
    client_data_list = []
    for num in range(client_num):
        filename = f"data_{num}.pt"
        file_path = os.path.join(client_data_dir, filename)
        if os.path.exists(file_path):
            client_data = torch.load(file_path)
            client_data_list.append(client_data)
    return client_data_list


def create_contaminated_client(contamination_ratio, args, processed_dir, client_data_dir, contaminated_client_dir):
    """
    创建并污染客户端数据。

    参数:
        contamination_ratio (float): 需要污染的客户端比例（例如 0.3 表示 30%）。
        args (argparse.Namespace): 包含噪声处理参数的命名空间对象。
        processed_dir (str): 处理后的数据保存目录。
        client_data_dir (str): 客户端数据文件所在的目录。

    返回:
        list: 包含所有客户端数据的列表（包括污染后的数据）。
    """
    # 确保处理后的目录存在
    os.makedirs(processed_dir, exist_ok=True)

    # 确保污染后的客户端目录存在
    os.makedirs(contaminated_client_dir, exist_ok=True)
    client_data_list = load_client_list(client_data_dir=client_data_dir)

    # 计算需要污染的客户端数量
    num_clients = len(client_data_list)
    num_contaminated_clients = int(num_clients * contamination_ratio)

    # 随机选择需要污染的客户端索引
    contaminated_client_indices = random.sample(
        range(num_clients), num_contaminated_clients)

    # 处理每个需要污染的客户端
    for client_id in contaminated_client_indices:
        splitted_data = client_data_list[client_id]
        # 调用 random_topology_noise 函数引入边噪声
        processed_data = random_topology_noise(
            args,
            splitted_data,
            processed_dir=processed_dir,
            client_id=client_id,
            noise_prob=args.processing_percentage
        )
        client_data_list[client_id] = processed_data

    # 保存污染后的客户端数据
    target_dir = os.path.join(args.root, 'distrib',
                              'subgraph_fl_louvain_1_Cora_client_10')
    for client_id, client_data in enumerate(client_data_list):
        torch.save(client_data, os.path.join(
            target_dir, f"data_{client_id}.pt"))

    return client_data_list


def generate_virtual_graph(num_nodes=500, feature_dim=1433, num_classes=5):
    """
    生成虚拟图数据
    :param num_nodes: 节点数量
    :param feature_dim: 节点特征维度
    :param num_classes: 类别数量
    :return: 虚拟图数据 (Data 对象)
    """
    data = CSBM(n=num_nodes, d=10, ratio=0.8, p=feature_dim, mu=1.0)
    return data


def compute_client_similarity_matrix(x):
    """
    客户端模型在服务器端计算相似度矩阵
    :param model: 客户端模型
    :x: 节点特征
    :return: 相似度矩阵
    """
    similarity_matrix = cosine_similarity(x.numpy())  # 计算相似度矩阵
    return torch.tensor(similarity_matrix, dtype=torch.float32)


def compute_mainstream_similarity(client_similarities):
    """
    计算主流相似度分布（平均相似度矩阵）
    :param client_similarities: 所有客户端的相似度矩阵列表
    :return: 主流相似度矩阵
    """
    mainstream_similarity = torch.mean(torch.stack(client_similarities), dim=0)
    return mainstream_similarity


def detect_corrupted_clients(client_similarities, mainstream_similarity, threshold=0.1):
    """
    检测污染客户端
    :param client_similarities: 所有客户端的相似度矩阵列表
    :param mainstream_similarity: 主流相似度矩阵
    :param threshold: 偏差阈值
    :return: 污染客户端的索引列表
    """
    corrupted_clients = []
    for i, client_sim in enumerate(client_similarities):
        # 计算均方误差
        mse = torch.mean((client_sim - mainstream_similarity) ** 2).item()
        if mse > threshold:
            corrupted_clients.append(i)
    return corrupted_clients


def adjust_client_weights(client_similarities, mainstream_similarity) -> list:
    """
    根据偏差动态调整客户端权重
    :param client_similarities: 所有客户端的相似度矩阵列表
    :param mainstream_similarity: 主流相似度矩阵
    :return: 客户端权重列表
    """
    weights = []
    for client_sim in client_similarities:
        # 计算均方误差
        mse = torch.mean((client_sim - mainstream_similarity) ** 2).item()
        # 偏差越大，权重越小
        weight = 1.0 / (1.0 + mse)
        weights.append(weight)
    # 归一化权重
    weights = torch.tensor(weights) / torch.sum(torch.tensor(weights))
    return weights


# def federated_anomaly_detection(client_data_list, hid_dim=64, num_layers=2, dropout=0.5, num_clients=10, num_nodes=500, feature_dim=128, num_classes=7, threshold=0.1):
    """
    联邦学习中的异常客户端检测
    :param num_clients: 客户端数量
    :param num_nodes: 虚拟图的节点数量
    :param feature_dim: 节点特征维度
    :param num_classes: 类别数量
    :param threshold: 偏差阈值
    :return: 
        - weights: 客户端权重列表（方案一）
        - remaining_clients: 剩余的客户端索引列表（方案二）
        - removed_clients: 被踢出的客户端索引列表（方案二）
    """

    # 服务器端生成虚拟图数据
    virtual_graph = generate_virtual_graph(
        num_nodes=num_nodes, feature_dim=feature_dim, num_classes=num_classes)

    gcn_model = GCN(input_dim=feature_dim, hid_dim=hid_dim, output_dim=num_classes,
                    num_layers=num_layers, dropout=dropout)

    client_data_x = []
    for index in range(num_clients):
        data = Data(x=client_data_list[index]['x'],
                    edge_index=client_data_list[index]['edge_index'], y=client_data_list[index]['y'])

        # 使用GCN进行推理
        with torch.no_grad():
            updated_x, _ = gcn_model(data)

        # 将更新后的x添加到client_data_x列表中
        client_data_x.append(updated_x)

        # 2. 模拟客户端模型上传到服务器
    # 使用client_data_x替换client_models
    client_similarities = [compute_client_similarity_matrix(
        x, virtual_graph) for x in client_data_x]
    mainstream_similarity = compute_mainstream_similarity(client_similarities)

    corrupted_clients = detect_corrupted_clients(
        client_similarities, mainstream_similarity, threshold)
    print("Detected corrupted clients:", corrupted_clients)

    # 对所有client, 都有weight
    weights = adjust_client_weights(client_similarities, mainstream_similarity)
    # print("Adjusted client weights:", weights)

    return weights
