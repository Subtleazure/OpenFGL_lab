import torch
from openfgl.model.gcn import *
from synthetic import CSBM
from openfgl.data.processing import *
import os
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
import community  # Louvain 算法的 Python 实现
import networkx as nx
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from torch_geometric.utils import degree, to_dense_adj, remove_isolated_nodes

def load_client_list(client_data_dir, client_num=200):
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
    print(num_contaminated_clients)
    # 随机选择需要污染的客户端索引
    contaminated_client_indices = random.sample(
        range(num_clients), num_contaminated_clients)
    print("contaminated_client_indices: ", contaminated_client_indices)
    with open(args.log_dir, 'a', encoding='utf-8') as file:
        file.write(
            f"contaminated_client_indices:{contaminated_client_indices}\n")
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
    for client_id, client_data in enumerate(client_data_list):
        torch.save(client_data, os.path.join(
            contaminated_client_dir, f"data_{client_id}.pt"))

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


def cosine_similarity_gpu(x):
    # 归一化
    x_norm = x / x.norm(dim=1)[:, None]
    # 计算相似度矩阵
    return torch.mm(x_norm, x_norm.transpose(0, 1))

def compute_client_similarity_matrix(x):
    device = x.device
    if device.type == 'cpu':
        return torch.tensor(cosine_similarity(x.numpy()), dtype=torch.float32, device=device)
    elif device.type == 'cuda':
        return cosine_similarity_gpu(x)
    else:
        raise ValueError(f"Unsupported device: {device}. Use 'cpu' or 'cuda'.")


def compute_mainstream_similarity(client_similarities):
    """
    计算主流相似度分布（平均相似度矩阵）
    :param client_similarities: 所有客户端的相似度矩阵列表
    :return: 主流相似度矩阵
    """
    mainstream_similarity = torch.mean(torch.stack(client_similarities), dim=0)
    return mainstream_similarity


# def detect_corrupted_clients(client_similarities, mainstream_similarity, threshold=0.1):
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


def adjust_client_weights(client_similarities, mainstream_similarity, gamma=4.0) -> list:
    """
    根据偏差动态调整客户端权重
    :param client_similarities: 所有客户端的相似度矩阵列表
    :param mainstream_similarity: 主流相似度矩阵
    :param gamma: 调整偏差影响的参数
    :return: 客户端权重列表
    """
    weights = []
    deltas = []
    for client_sim in client_similarities:
        # 计算偏差 Δ_i
        delta_i = torch.mean((client_sim - mainstream_similarity) ** 2).item()
        deltas.append(delta_i)

    # 归一化偏差
    delta_min = min(deltas)
    delta_max = max(deltas)
    normalized_deltas = [(delta - delta_min) /
                         (delta_max - delta_min) for delta in deltas]

    # 计算权重 w_i^f
    for delta_i in normalized_deltas:
        # 将 delta_i 转换为 PyTorch 张量
        delta_i_tensor = torch.tensor(delta_i, dtype=torch.float32)
        weight = torch.exp(gamma * delta_i_tensor)
        weights.append(weight.item())  # 将张量转换回 Python 浮点数

    # 归一化权重
    weights = torch.tensor(weights) / torch.sum(torch.tensor(weights))
    # print(deltas)
    return weights



def calculate_pairwise_weights(all_similarity_matrices_1, all_similarity_matrices_2, gamma=4.0) -> list:
    """
    计算两个相似度矩阵列表中每对矩阵的相似度，并返回权重列表
    :param all_similarity_matrices_1: 第一个相似度矩阵列表
    :param all_similarity_matrices_2: 第二个相似度矩阵列表
    :param gamma: 调整相似度影响的参数
    :return: 权重列表
    """
    weights = []
    similarities = []
    
    # 计算每对矩阵的相似度
    for mat1, mat2 in zip(all_similarity_matrices_1, all_similarity_matrices_2):
        # 计算相似度（例如，使用Frobenius范数）, similarity 越小相似度越大
        similarity = torch.norm(mat1 - mat2, p='fro').item()
        similarities.append(similarity)
    
    # 归一化相似度
    sim_min = min(similarities)
    sim_max = max(similarities)
    normalized_similarities = [(sim_max - sim) / (sim_max - sim_min) for sim in similarities]
    
    # 计算权重
    for sim in normalized_similarities:
        # 将相似度转换为 PyTorch 张量
        sim_tensor = torch.tensor(sim, dtype=torch.float32)
        weight = torch.exp(gamma * sim_tensor)
        weights.append(weight.item())  # 将张量转换回 Python 浮点数
    
    # 归一化权重
    weights = torch.tensor(weights) / torch.sum(torch.tensor(weights))
    
    return weights



def calculate_pairwise_weights_based_on_delta(all_similarity_matrices_1, all_similarity_matrices_2, gamma=4.0) -> list:
    """
    计算两个相似度矩阵列表中每对矩阵的偏差 Δ_i，并返回权重列表
    :param all_similarity_matrices_1: 第一个相似度矩阵列表
    :param all_similarity_matrices_2: 第二个相似度矩阵列表
    :param gamma: 调整偏差影响的参数
    :return: 权重列表
    """
    weights = []
    deltas = []
    
    # 计算每对矩阵的偏差 Δ_i
    for mat1, mat2 in zip(all_similarity_matrices_1, all_similarity_matrices_2):
        # 计算偏差 Δ_i（例如，使用均方误差）
        delta_i = torch.mean((mat1 - mat2) ** 2).item()
        deltas.append(delta_i)
    
    # 归一化偏差
    delta_min = min(deltas)
    delta_max = max(deltas)
    normalized_deltas = [(delta - delta_min) / (delta_max - delta_min) for delta in deltas]
    
    # 计算权重 w_i^f
    for delta_i in normalized_deltas:
        # 将 delta_i 转换为 PyTorch 张量
        delta_i_tensor = torch.tensor(delta_i, dtype=torch.float32)
        weight = torch.exp(-gamma * delta_i_tensor)  # 偏差越小，权重越大
        weights.append(weight.item())  # 将张量转换回 Python 浮点数
    
    # 归一化权重
    weights = torch.tensor(weights) / torch.sum(torch.tensor(weights))
    
    return weights


# 加载 ogbn-arxiv 数据集
def load_ogbn_arxiv():
    # 使用 pandas 加载 CSV 文件
    print("Loading node features...")
    node_features = pd.read_csv(
        r"D:\desk\WHU\Study\Paper\ogbn-arxiv\raw\node-feat.csv", header=None).values  # 加载为 numpy 数组
    print("Loading edge index...")
    # 加载为 [2, num_edges] 的 numpy 数组
    edge_index = pd.read_csv(
        r"D:\desk\WHU\Study\Paper\ogbn-arxiv\raw\edge.csv", header=None).values.T
    print("Loading labels...")
    labels = pd.read_csv(r"D:\desk\WHU\Study\Paper\ogbn-arxiv\raw\node-label.csv",
                         header=None).values       # 加载为 numpy 数组

    # 将 numpy 数组转换为 PyTorch 张量
    print("Converting to PyTorch tensors...")
    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return node_features, edge_index, labels


# 均衡调整客户端分配
def balance_clients(client_communities, community_sizes, max_iter=200):
    for _ in range(max_iter):
        # 计算每个客户端的节点数量
        sizes = [sum(community_sizes[comm_id] for comm_id in comms)
                 for comms in client_communities.values()]
        avg_size = sum(sizes) / len(sizes)
        # 找到过大和过小的客户端
        oversized = [cid for cid, size in enumerate(
            sizes) if size > avg_size * 1.2]
        undersized = [cid for cid, size in enumerate(
            sizes) if size < avg_size * 0.8]
        if not oversized or not undersized:
            break
        # 从过大的客户端中移动一个社区到过小的客户端
        moved = False
        for src_cid in oversized:
            if client_communities[src_cid]:  # Check if the list is not empty
                dst_cid = undersized[0]
                comm_to_move = client_communities[src_cid].pop(0)
                client_communities[dst_cid].append(comm_to_move)
                moved = True
                break  # Move only one community per iteration
        if not moved:
            break  # No more communities to move
    return client_communities


# 使用 Louvain 算法和分层聚类划分社区
def partition_graph(node_features, edge_index, labels, num_clients=150, resolution=1.0):
    # 将 edge_index 转换为 networkx 图对象
    print("Creating graph from edge index...")
    G = nx.Graph()
    for src, dst in edge_index.t():
        G.add_edge(src.item(), dst.item())

    # 使用 Louvain 算法生成社区
    print("Running Louvain algorithm...")
    partition = community.best_partition(G, resolution=resolution)
    num_communities = len(set(partition.values()))
    print(f"Generated {num_communities} communities.")

    # 提取社区特征
    print("Extracting community features...")
    community_features = []
    community_sizes = []
    for comm_id in set(partition.values()):
        nodes = [node for node, cid in partition.items() if cid == comm_id]
        features = node_features[nodes].mean(axis=0).numpy()  # 特征均值
        class_dist = np.bincount(
            labels[nodes].flatten(), minlength=40) / len(nodes)  # 类别分布
        density = len(G.subgraph(nodes).edges()) / (len(nodes) *
                                                    (len(nodes) - 1)) if len(nodes) > 1 else 0  # 边密度
        comm_feature = np.concatenate(
            [features, class_dist, [len(nodes), density]])
        community_features.append(comm_feature)
        community_sizes.append(len(nodes))

    community_features = np.array(community_features)

    # 标准化特征
    print("Standardizing features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(community_features)

    # 层次聚类
    print("Running hierarchical clustering...")
    agg_clustering = AgglomerativeClustering(
        n_clusters=num_clients,  # 目标客户端数量
        linkage='ward',  # 最小化合并后的方差
        metric='euclidean'
    )
    community_groups = agg_clustering.fit_predict(scaled_features)

    # 初始合并
    client_communities = defaultdict(list)
    for comm_id, client_id in enumerate(community_groups):
        client_communities[client_id].append(comm_id)

    # 均衡调整客户端分配
    print("Balancing client assignments...")
    balanced_clients = balance_clients(client_communities, community_sizes)

    # 生成客户端数据
    print("Generating client data...")
    client_data = {}
    for client_id, comm_ids in balanced_clients.items():
        print(client_id, comm_ids)
        nodes = []
        for comm_id in comm_ids:
            nodes += [node for node, cid in partition.items() if cid ==
                      comm_id]
        # 收集边
        edges = []
        for src, dst in edge_index.t():
            if src.item() in nodes and dst.item() in nodes:
                edges.append((src, dst))
        # 保存数据
        client_data[client_id] = {
            "num_global_classes": 40,
            "y": labels[nodes],
            "x": node_features[nodes],
            "edge_index": torch.tensor(edges).t(),
            "global_map": torch.tensor(nodes)
        }

    # 保存客户端数据
    target_dir = "D:\desk\WHU\Study\Paper\ogbn-arxiv\distrib\subgraph_fl_louvain_1_Cora_client_150"
    print(f"Creating target directory: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    for client_id, data in client_data.items():
        torch.save(data, os.path.join(target_dir, f"data_{client_id}.pt"))

    print("Partitioning completed!")


def sort_indices(weights):
    # 创建一个包含（值，原始索引）的元组列表
    indexed_list = list(enumerate(weights))

    # 根据列表的值进行排序，同时保持对原始索引的跟踪
    sorted_list = sorted(indexed_list, key=lambda x: x[1])

    # 提取排序后的索引
    sorted_indices = [index for (index, value) in sorted_list]

    return sorted_indices



def A_D(data, device):
    edge_index = data.edge_index
    
    # 确保无向图的边是双向的（添加反向边）
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 无向图 => 强制双向
    
    # 移除孤立节点（同时检查入度和出度）
    edge_index, _, mask = remove_isolated_nodes(edge_index, num_nodes=data.num_nodes)
    num_nodes = mask.sum().item()  # 有效节点数
    
    # 计算度矩阵 D（确保无零度节点）
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    assert not (deg == 0).any(), "仍有孤立节点未被移除！"  # 确保无零度
    
    D = torch.diag(deg).to(device)
    
    # 计算邻接矩阵 A（仅保留有效节点）
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].to(device)
    
    return A, D

def compute_S_ano(data, device):
    A, D = A_D(data, device)
    L = D - A  # 拉普拉斯矩阵 L = D - A
    
    # 计算 D^T L D 和 D^T D
    D_T_L_D = torch.matmul(D.T, torch.matmul(L, D))
    D_T_D = torch.matmul(D.T, D)
    
    # 标量除法：全局求和后计算比值
    numerator = torch.sum(D_T_L_D)   # 分子：D^T L D 的所有元素求和
    denominator = torch.sum(D_T_D)   # 分母：D^T D 的所有元素求和
    s = numerator / denominator     # 标量比值
    
    return s.item()  # 返回标量值


def S_client_weights_s(S_ano_list):
    weights = []
    
    # 根据 S_ano 值计算权重（S_ano 越大，权重越小）
    # 使用 1 / S_ano 作为权重的基础
    weights = [1 / s for s in S_ano_list]
    
    # 对权重进行归一化，确保总和为 1
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    return weights

def S_client_weights_data_list(data_list, device):
    weights = []
    S_ano_values = []
    
    # 计算每个客户端的 S_ano 值
    for data in data_list:
        s = compute_S_ano(data, device=device)
        S_ano_values.append(s)
    
    # 根据 S_ano 值计算权重（S_ano 越大，权重越小）
    # 使用 1 / S_ano 作为权重的基础
    weights = [1 / s for s in S_ano_values]
    
    # 对权重进行归一化，确保总和为 1
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    return weights


def modify_edges(similarity_matrix, edge_index, num_nodes, device=None):
    if device is None:
        device = similarity_matrix.device

    # Step 1: Extract edge similarities
    connected_edges = edge_index.t()  # Shape: [num_edges, 2]
    edge_similarities = similarity_matrix[connected_edges[:, 0], connected_edges[:, 1]]

    # Step 2: Remove lowest 30% edges (keep top 70%)
    threshold = torch.quantile(edge_similarities, 0.3)  # 30th percentile (keep 70%)
    mask = edge_similarities >= threshold  # Keep edges with similarity >= threshold
    edge_index_kept = edge_index[:, mask]
    num_edges_removed = edge_index.size(1) - edge_index_kept.size(1)

    # Early return if no edges need to be added
    if num_edges_removed == 0:
        return edge_index_kept

    # Step 3: Prepare existing edges (optimized using tensor operations)
    # 修正这里：正确构造稀疏张量
    existing_edges = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.size(1), device=device),  # 修正参数传递
        size=(num_nodes, num_nodes),
        device=device
    ).to_dense().bool()

    # Step 4: Generate all possible candidate edges (no self-loops, no existing edges)
    # Create upper triangular mask excluding diagonal
    triu_mask = torch.triu(torch.ones(num_nodes, num_nodes, device=device), diagonal=1).bool()
    # Mask out existing edges and their reverse
    candidate_mask = triu_mask & (~existing_edges) & (~existing_edges.t())
    
    # Get indices of candidate edges
    candidate_pairs = torch.nonzero(candidate_mask)

    # Step 5: Add top-k highest similarity edges (k = num_edges_removed)
    if num_edges_removed > 0 and len(candidate_pairs) > 0:
        candidate_similarities = similarity_matrix[candidate_pairs[:, 0], candidate_pairs[:, 1]]
        # Use torch.topk instead of sort for better performance
        num_edges_to_add = min(num_edges_removed, len(candidate_pairs))
        topk_values, topk_indices = torch.topk(candidate_similarities, k=num_edges_to_add)
        selected_pairs = candidate_pairs[topk_indices]
        
        # Concatenate kept edges with new edges
        edge_index = torch.cat([edge_index_kept, selected_pairs.t()], dim=1)
    else:
        edge_index = edge_index_kept

    return edge_index