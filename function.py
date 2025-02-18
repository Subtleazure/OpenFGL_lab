import torch
from fake_data.synthetic import CSBM
from openfgl.data.processing import *
import os
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity


def create_contaminated_client(contamination_ratio, args, processed_dir, client_data_dir):
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

    # 加载客户端数据
    client_data_list = []
    for num in range(10):  # 假设有 10 个客户端
        filename = f"data_{num}.pt"
        file_path = os.path.join(client_data_dir, filename)
        if os.path.exists(file_path):
            client_data = torch.load(file_path)
            client_data_list.append(client_data)

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

    return client_data_list


def generate_virtual_graph(num_nodes=500, feature_dim=128, num_classes=5):
    """
    生成虚拟图数据
    :param num_nodes: 节点数量
    :param feature_dim: 节点特征维度
    :param num_classes: 类别数量
    :return: 虚拟图数据 (Data 对象)
    """
    data = CSBM(n=num_nodes, d=10, ratio=0.8, p=feature_dim, mu=1.0)
    return data


def compute_client_similarity_matrix(data):
    """
    客户端模型在服务器端计算相似度矩阵
    :param model: 客户端模型
    :param data: 虚拟图数据
    :return: 相似度矩阵
    """
    similarity_matrix = cosine_similarity(data.x.numpy())  # 计算相似度矩阵
    return similarity_matrix


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


def adjust_client_weights(client_similarities, mainstream_similarity):
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


def federated_anomaly_detection(num_clients=10, num_nodes=500, feature_dim=128, num_classes=5, corruption_ratio=0.2, threshold=0.1):
    """
    联邦学习中的异常客户端检测
    :param num_clients: 客户端数量
    :param num_nodes: 虚拟图的节点数量
    :param feature_dim: 节点特征维度
    :param num_classes: 类别数量
    :param corruption_ratio: 污染客户端比例
    :param threshold: 偏差阈值
    :return: 
        - weights: 客户端权重列表（方案一）
        - remaining_clients: 剩余的客户端索引列表（方案二）
        - removed_clients: 被踢出的客户端索引列表（方案二）
    """

    # 服务器端生成虚拟图数据
    virtual_graph = generate_virtual_graph(
        num_nodes=num_nodes, feature_dim=feature_dim, num_classes=num_classes)

    return weights, remaining_clients, removed_clients
