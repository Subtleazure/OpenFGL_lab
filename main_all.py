import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import os
import torch
from openfgl.data.processing import random_topology_noise
import argparse
import random
from function import *
from average_index import *
args = config.args
args.dataset_ = "Amz-Comp"

# modify the root path
args.root = f"/data2/liujiaqi/openfgl_dataset/contaminated/{args.dataset_}"
args.dataset = ["Computers"]
args.simulation_mode = "subgraph_fl_louvain"
args.num_clients = 50
args.features = 767
args.classes = 10
args.num_rounds = 3000
args.lr = 0.005
args.log_dir = f'/data2/liujiaqi/OpenFGL-main/log_{args.dataset_}.txt'
args.accuracy_curve_dir = f'/data2/liujiaqi/curves/{args.dataset_}/accuracy_curve_{args.num_rounds}_lr_0_005_{args.dataset_}_{args.aggregation_mode}.png'
args.accuracy_curve_html_dir = f'/data2/liujiaqi/curves/{args.dataset_}/accuracy_curve_{args.num_rounds}_lr_0_005_{args.dataset_}_{args.aggregation_mode}.html'
args.window_len = 1200

if True:
    args.fl_algorithm = "fedavg"
    args.model = ["gcn"]
else:
    args.fl_algorithm = "fedproto"
    # choose multiple gnn models for model heterogeneity setting.
    args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"]

args.metrics = ["accuracy"]

# 定义参数
# parser = argparse.ArgumentParser()
# parser.add_argument("--task", type=str, default="node_cls")
# parser.add_argument("--processing", type=str, default="random_topology_noise")
# parser.add_argument("--processing_percentage", type=float, default=0.2)  # 噪声强度
# args.processing = "random_topology_noise"
# args.processing_percentage = 0.2
args.task = "node_cls"
args.processing_percentage = 0.5
# 定义处理目录
processed_dir = "processed_data"
# 定义污染目录
contaminated_client_dir = f"/data2/liujiaqi/openfgl_dataset/contaminated/{args.dataset_}/distrib/subgraph_fl_louvain_1_{args.dataset[0]}_client_{args.num_clients}"
# 定义客户端数据目录
client_data_dir = f"/data2/liujiaqi/openfgl_dataset/source/{args.dataset_}/distrib/subgraph_fl_louvain_1_{args.dataset[0]}_client_{args.num_clients}"

# 创建污染客户端
client_data_list = create_contaminated_client(contamination_ratio=args.contamination_ratio, args=args,
                                              processed_dir=processed_dir, client_data_dir=client_data_dir,
                                              contaminated_client_dir=contaminated_client_dir)


# 以下是原有的训练代码
trainer = FGLTrainer(args)
trainer.train()

average_index_main(args.aggregation_mode, args.log_dir, f"/data2/liujiaqi/OpenFGL-main/weight_{args.dataset_}.txt")