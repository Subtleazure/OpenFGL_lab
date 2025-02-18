import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import os
import torch
from openfgl.data.processing import random_topology_noise
import argparse
import random
from function import *

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


# 定义处理目录
processed_dir = "processed_data"

# 定义客户端数据目录
client_data_dir = "D:\desk\WHU\Study\Paper\OpenFGL-main\your_data_root\distrib\subgraph_fl_louvain_1_Cora_client_10"

# 创建污染客户端
create_contaminated_client(contamination_ratio=0.3, args=args,
                           processed_dir=processed_dir, client_data_dir=client_data_dir)


# 以下是原有的训练代码
trainer = FGLTrainer(args)
trainer.train()
