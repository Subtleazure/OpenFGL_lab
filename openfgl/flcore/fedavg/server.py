import torch
from openfgl.flcore.base import BaseServer
from function import *
from openfgl.model.gcn import *
import numpy as np
import matplotlib.pyplot as plt

class FedAvgServer(BaseServer):
    """
    FedAvgServer implements the server-side logic for the Federated Averaging (FedAvg) algorithm,
    as introduced in the paper "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    by McMahan et al. (2017). This class is responsible for aggregating model updates from clients
    and broadcasting the updated global model to all participants in the federated learning process.

    Attributes:
        None (inherits attributes from BaseServer)
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedAvgServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(FedAvgServer, self).__init__(
            args, global_data, data_dir, message_pool, device)


    # benchmark
    def execute(self):
        """
        Executes the server-side operations. This method aggregates model updates from the 
        clients by computing a weighted average of the model parameters, based on the number 
        of samples each client used for training.
        """
        weights = []
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weights.append(self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples)
                
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weights[client_id] * local_param)
                    else:
                        global_param.data += weights[client_id] * local_param
            
        print("weights:", weights)
        sorted_indices1 = sort_indices(weights)
        if self.args.graph_repair:
            contam_num = int(self.args.num_clients * self.args.contamination_ratio)
            self.args.repair_clients = sorted_indices1[:contam_num]
        print("weights list(greater):", sorted_indices1)
        with open(self.args.log_dir, 'a', encoding='utf-8') as file:
            file.write(f"weights list(greater): {sorted_indices1}\n")


    # 单个虚拟图推理
    def execute_one_vg(self):
        """
        Executes the server-side operations. This method aggregates model updates from the 
        clients by computing a weighted average of the model parameters, based on the number 
        of samples each client used for training.
        """
        # 使用 torch.no_grad() 上下文管理器，确保在聚合过程中不会计算梯度
        # with torch.no_grad():
        # 计算所有参与训练的客户端的总样本数
        # 遍历所有被选中的客户端，从 message_pool 中获取每个客户端的样本数并求和
        # num_tot_samples = sum(
        #     [self.message_pool[f"client_{client_id}"]["num_samples"]
        #      for client_id in self.message_pool[f"sampled_clients"]])

        # 获取各个客户端的模型参数，所有客户端模型在虚拟图数据上推理得到节点特征列表

        # 生成虚拟图数据
        virtual_graph = generate_virtual_graph(feature_dim=self.args.features)

        # 初始化 GCN 模型
        input_dim = virtual_graph.x.size(1)  # 节点特征维度
        hid_dim = 64  # 隐藏层维度（可根据需要调整）
        output_dim = self.args.classes  # 输出维度（分类任务中的类别数）
        gcn_model = GCN(input_dim, hid_dim, output_dim)

        # 存储所有客户端的节点特征
        all_client_features = []

        # 遍历所有客户端
        for client_id in self.message_pool["sampled_clients"]:
            # 获取客户端的模型参数
            client_weights = self.message_pool[f"client_{client_id}"]["weight"]

            # 将客户端参数加载到 GCN 模型中
            with torch.no_grad():
                for param, client_param in zip(gcn_model.parameters(), client_weights):
                    param.data.copy_(client_param)

            # 在虚拟图数据上进行推理
            gcn_model.eval()  # 设置为评估模式
            with torch.no_grad():
                node_features, logits = gcn_model(virtual_graph)

            # 将节点特征添加到列表中
            all_client_features.append(node_features)

        # 节点特征使用相似度计算函数得到相似度矩阵
        # 计算相似度矩阵
        all_similarity_matrices = []
        for x in all_client_features:
            similarity_matrix = compute_client_similarity_matrix(x)
            all_similarity_matrices.append(similarity_matrix)

        # 计算主流相似度矩阵
        mainstream_similarity = compute_mainstream_similarity(
            all_similarity_matrices)

        # 调整客户端权重
        weights = adjust_client_weights(
            all_similarity_matrices, mainstream_similarity)
        print("weights:", weights)
        sorted_indices1 = sort_indices(weights)
        if self.args.graph_repair:
            contam_num = int(self.args.num_clients * self.args.contamination_ratio)
            self.args.repair_clients = sorted_indices1[:contam_num]
        print("weights list(greater):", sorted_indices1)
        with open(self.args.log_dir, 'a', encoding='utf-8') as file:
            file.write(f"weights list(greater): {sorted_indices1}\n")
        # 遍历所有被选中的客户端，更新全局模型参数(先不做聚合)
        for it, client_id in enumerate(self.message_pool["sampled_clients"]):
            # 计算当前客户端的权重，权重为该客户端的样本数占总样本数的比例(需要修改weight列表)
            weight = weights[it]
            # 遍历当前客户端的模型参数和全局模型的参数
            for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"],
                                                   self.task.model.parameters()):
                # 如果是第一个客户端，直接将全局模型参数设置为当前客户端的参数乘以权重
                if it == 0:
                    global_param.data.copy_(weight * local_param)
                else:
                    # 如果不是第一个客户端，将当前客户端的参数乘以权重并累加到全局模型参数中
                    global_param.data += weight * local_param


    # 两个虚拟图上推理
    def execute_two_vg(self):
        """
        Executes the server-side operations. This method aggregates model updates from the 
        clients by computing a weighted average of the model parameters, based on the number 
        of samples each client used for training.
        """
        # 使用 torch.no_grad() 上下文管理器，确保在聚合过程中不会计算梯度
        # with torch.no_grad():
        # 计算所有参与训练的客户端的总样本数
        # 遍历所有被选中的客户端，从 message_pool 中获取每个客户端的样本数并求和
        # num_tot_samples = sum(
        #     [self.message_pool[f"client_{client_id}"]["num_samples"]
        #      for client_id in self.message_pool[f"sampled_clients"]])

        # 获取各个客户端的模型参数，所有客户端模型在虚拟图数据上推理得到节点特征列表

        # 生成虚拟图数据
        virtual_graph = generate_virtual_graph(feature_dim=self.args.features)

        # 生成损坏的虚拟图
        virtual_noised_graph_1 = random_topology_noise(self.args, splitted_data=virtual_graph,processed_dir=None,client_id=None,noise_prob=self.args.processing_percentage)
        virtual_noised_graph_2 = random_topology_noise(self.args, splitted_data=virtual_graph,processed_dir=None,client_id=None,noise_prob=self.args.processing_percentage)

        # 初始化 GCN 模型
        input_dim = self.args.features  # 节点特征维度
        hid_dim = 64  # 隐藏层维度（可根据需要调整）
        output_dim = self.args.classes  # 输出维度（分类任务中的类别数）
        gcn_model = GCN(input_dim, hid_dim, output_dim)

        # 存储所有客户端的节点特征
        all_client_features_1 = []
        all_client_features_2 = []

        # 遍历所有客户端
        for client_id in self.message_pool["sampled_clients"]:
            # 获取客户端的模型参数
            client_weights = self.message_pool[f"client_{client_id}"]["weight"]

            # 将客户端参数加载到 GCN 模型中
            with torch.no_grad():
                for param, client_param in zip(gcn_model.parameters(), client_weights):
                    param.data.copy_(client_param)

            # 在虚拟图数据上进行推理
            gcn_model.eval()  # 设置为评估模式
            with torch.no_grad():
                node_features_1, logits_1 = gcn_model(virtual_noised_graph_1)
                node_features_2, logits_2 = gcn_model(virtual_noised_graph_2)

            # 将节点特征添加到列表中
            all_client_features_1.append(node_features_1)
            all_client_features_2.append(node_features_2)

        # 计算相似度矩阵
        all_similarity_matrices_1 = []
        all_similarity_matrices_2 = []
        for x in all_client_features_1:
            similarity_matrix = compute_client_similarity_matrix(x)
            all_similarity_matrices_1.append(similarity_matrix)

        for x in all_client_features_2:
            similarity_matrix = compute_client_similarity_matrix(x)
            all_similarity_matrices_2.append(similarity_matrix)
        

        # 调整客户端权重
        weights = calculate_pairwise_weights(
            all_similarity_matrices_1, all_similarity_matrices_2)
        print("weights:", weights)
        sorted_indices1 = sort_indices(weights)
        if self.args.graph_repair:
            contam_num = int(self.args.num_clients * self.args.contamination_ratio)
            self.args.repair_clients = sorted_indices1[:contam_num]
        print("weights list(greater):", sorted_indices1)
        with open(self.args.log_dir, 'a', encoding='utf-8') as file:
            file.write(f"weights list(greater): {sorted_indices1}\n")
        # 遍历所有被选中的客户端，更新全局模型参数(先不做聚合)
        for it, client_id in enumerate(self.message_pool["sampled_clients"]):
            # 计算当前客户端的权重，权重为该客户端的样本数占总样本数的比例(需要修改weight列表)
            weight = weights[it]
            # 遍历当前客户端的模型参数和全局模型的参数
            for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"],
                                                    self.task.model.parameters()):
                # 如果是第一个客户端，直接将全局模型参数设置为当前客户端的参数乘以权重
                if it == 0:
                    global_param.data.copy_(weight * local_param)
                else:
                    # 如果不是第一个客户端，将当前客户端的参数乘以权重并累加到全局模型参数中
                    global_param.data += weight * local_param


    # lap指标
    def execute_lap(self):
        """
        Executes the server-side operations. This method aggregates model updates from the 
        clients by computing a weighted average of the model parameters, based on the number 
        of samples each client used for training.
        """
        weights = S_client_weights_s(self.args.S_ano_list)

        if self.args.heatmap:
            S_ano_list = self.args.S_ano_list  # 原始数据列表
            N = len(S_ano_list)
            contaminated_indices = self.args.contaminated_client_indices  # 污染客户端编号（0~N-1）

            # 计算差值矩阵
            S_ano_array = np.array(S_ano_list)
            heatmap_data = np.abs(S_ano_array[:, None] - S_ano_array[None, :])

            # --- 指数变换（突出高值差异）---
            # 方法1：平方变换（gamma=2）
            heatmap_data_transformed = np.power(heatmap_data, 2)

            # 方法2：自然指数变换（更激进）
            # heatmap_data_transformed = np.exp(heatmap_data) - 1  # 减1避免0值变为1

            # 分离污染和非污染客户端索引
            clean_indices = [i for i in range(N) if i not in contaminated_indices]
            new_order = contaminated_indices + clean_indices  # 污染客户端在前，非污染在后

            # 重排矩阵行列
            heatmap_data_reordered = heatmap_data_transformed[new_order, :][:, new_order]

            # 设置颜色范围（基于变换后的数据）
            vmin, vmax = 0, np.max(heatmap_data_reordered)

            # 绘制热图
            plt.figure(figsize=(10, 8))
            heatmap = plt.imshow(
                heatmap_data_reordered,
                cmap='viridis_r',  # 反转颜色，高值更暗
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest'
            )

            # 添加颜色条（标注原始差值范围）
            cbar = plt.colorbar(heatmap, label='Transformed Absolute Difference')
            # 在颜色条上标注原始值（可选）
            original_ticks = np.linspace(0, np.max(heatmap_data), 5)  # 原始差值刻度
            transformed_ticks = np.power(original_ticks, 2)          # 对应的变换后刻度
            cbar.set_ticks(transformed_ticks)
            cbar.set_ticklabels([f'{x:.1f}' for x in original_ticks])  # 显示原始值

            # 设置坐标轴刻度（标注分组信息）
            tick_positions = np.arange(N)
            tick_labels = [f'C{i+1}' if i in contaminated_indices else f'N{i+1}' 
                        for i in new_order]  # C:污染, N:非污染

            plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=8)
            plt.yticks(tick_positions, tick_labels, fontsize=8)

            # 添加分组分隔线（红色虚线）
            num_contaminated = len(contaminated_indices)
            plt.axvline(x=num_contaminated - 0.5, color='red', linestyle='--', linewidth=1)
            plt.axhline(y=num_contaminated - 0.5, color='red', linestyle='--', linewidth=1)

            # 添加标题
            plt.title('Heatmap with Exponential Transformation\n(C: Contaminated, N: Clean)', fontsize=12)
            plt.xlabel('Client Index', fontsize=10)
            plt.ylabel('Client Index', fontsize=10)

            # 保存图片（dpi 设置分辨率，bbox_inches='tight' 去除白边）
            plt.savefig('/data2/liujiaqi/heatmap/heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图像，避免内存泄漏
        
        with torch.no_grad():
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):                
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weights[client_id] * local_param)
                    else:
                        global_param.data += weights[client_id] * local_param
            
        print("weights:", weights)
        sorted_indices1 = sort_indices(weights)
        if self.args.graph_repair:
            contam_num = int(self.args.num_clients * self.args.contamination_ratio)
            self.args.repair_clients = sorted_indices1[:contam_num]
        print("weights list(greater):", sorted_indices1)
        with open(self.args.log_dir, 'a', encoding='utf-8') as file:
            file.write(f"weights list(greater): {sorted_indices1}\n")
        # 遍历所有被选中的客户端，更新全局模型参数(先不做聚合)
        for it, client_id in enumerate(self.message_pool["sampled_clients"]):
            # 计算当前客户端的权重，权重为该客户端的样本数占总样本数的比例(需要修改weight列表)
            weight = weights[it]
            # 遍历当前客户端的模型参数和全局模型的参数
            for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"],
                                                    self.task.model.parameters()):
                # 如果是第一个客户端，直接将全局模型参数设置为当前客户端的参数乘以权重
                if it == 0:
                    global_param.data.copy_(weight * local_param)
                else:
                    # 如果不是第一个客户端，将当前客户端的参数乘以权重并累加到全局模型参数中
                    global_param.data += weight * local_param


    # rhfl
    def execute_rhfl(self):
        """
        Executes the server-side operations. This method aggregates model updates from the 
        clients by computing a weighted average of the model parameters, based on the number 
        of samples each client used for training.
        """
        weights = []
        quality_list = []
        amount_with_quality = [1 / (self.args.num_clients - 1) for i in range(self.args.num_clients)]
        amount_with_quality_exp = []
        beta = 0.5
        with torch.no_grad():
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                delta_loss = self.args.last_mean_loss_list[client_id] - self.args.current_mean_loss_list[client_id]
                quality_list.append(delta_loss / self.args.current_mean_loss_list[client_id])
            quality_sum = sum(quality_list)

            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                amount_with_quality[client_id] += beta * quality_list[client_id] / quality_sum
                amount_with_quality_exp.append(np.exp(amount_with_quality[client_id]))
            amount_with_quality_sum = sum(amount_with_quality_exp)

            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weights.append(amount_with_quality_exp[client_id] / amount_with_quality_sum)
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weights[client_id] * local_param)
                    else:
                        global_param.data += weights[client_id] * local_param
            
        print("weights:", weights)
        sorted_indices1 = sort_indices(weights)
        if self.args.graph_repair:
            contam_num = int(self.args.num_clients * self.args.contamination_ratio)
            self.args.repair_clients = sorted_indices1[:contam_num]
        print("weights list(greater):", sorted_indices1)
        with open(self.args.log_dir, 'a', encoding='utf-8') as file:
            file.write(f"weights list(greater): {sorted_indices1}\n")


    def send_message(self):
        """
        Sends a message to the clients containing the updated global model parameters after 
        aggregation.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }
