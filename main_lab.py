import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import os
import torch
from openfgl.data.processing import random_topology_noise
import argparse
import random

args = config.args

args.root = "your_data_root"
args.dataset = ["Cora"]
args.simulation_mode = "subgraph_fl_louvain"
args.num_clients = 10

if True:
    args.fl_algorithm = "fedavg"
    args.model = ["gcn"]
else:
    args.fl_algorithm = "fedproto"
    # choose multiple gnn models for model heterogeneity setting.
    args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"]

args.metrics = ["accuracy"]

# 定义参数
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="node_cls")
parser.add_argument("--processing", type=str, default="random_topology_noise")
parser.add_argument("--processing_percentage", type=float, default=0.2)  # 噪声强度

# args = parser.parse_args()

# 定义处理目录
processed_dir = "processed_data"
os.makedirs(processed_dir, exist_ok=True)

# 定义客户端数据目录
client_data_dir = "D:\desk\WHU\Study\Paper\OpenFGL-main\your_data_root\distrib\subgraph_fl_louvain_1_Cora_client_10"

# 加载客户端数据
client_data_list = []
for num in range(10):
    filename = f"data_{num}.pt"
    file_path = os.path.join(client_data_dir, filename)
    if os.path.exists(file_path):
        client_data = torch.load(file_path)
        client_data_list.append(client_data)

# 定义需要污染的客户端比例
contamination_ratio = 0.3  # 例如，30% 的客户端需要被污染

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

# 以下是原有的训练代码
trainer = FGLTrainer(args)
trainer.train()
