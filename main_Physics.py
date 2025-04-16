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
args.fl_algorithm = "fedavg"
args.model = ["gcn"]
# modify the root path
args.root = "/data2/liujiaqi/openfgl_dataset/contaminated/Physics"
args.dataset = ["Physics"]
args.simulation_mode = "subgraph_fl_louvain"
args.num_clients = 100
args.features = 8415
args.classes = 5
args.num_rounds = 1000
args.lr = 0.01
args.log_dir = '/data2/liujiaqi/OpenFGL-main/log_Physics.txt'
# args.accuracy_curve_dir = f'/data2/liujiaqi/curves/Physics/accuracy_curve_{args.num_rounds}_lr_0_01_Physics_{args.aggregation_mode}_{args.graph_repair}.png'
# args.accuracy_curve_html_dir = f'/data2/liujiaqi/curves/Physics/accuracy_curve_{args.num_rounds}_lr_0_01_Physics_{args.aggregation_mode}_{args.graph_repair}.html'
args.accuracy_curve_dir = f'/data2/liujiaqi/curves/Physics/accuracy_curve_{args.num_rounds}_lr_0_01_Physics_{args.fl_algorithm}.png'
args.accuracy_curve_html_dir = f'/data2/liujiaqi/curves/Physics/accuracy_curve_{args.num_rounds}_lr_0_01_Physics_{args.fl_algorithm}.html'

# if True:
#     args.fl_algorithm = "fedavg"
#     args.model = ["gcn"]
# else:
#     args.fl_algorithm = "fedproto"
#     # choose multiple gnn models for model heterogeneity setting.
#     args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"]

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
contaminated_client_dir = "/data2/liujiaqi/openfgl_dataset/contaminated/Physics/distrib/subgraph_fl_louvain_1_Physics_client_100"
# 定义客户端数据目录
client_data_dir = "/data2/liujiaqi/openfgl_dataset/source/Physics/distrib/subgraph_fl_louvain_1_Physics_client_100"

# 创建污染客户端
client_data_list = create_contaminated_client(contamination_ratio=args.contamination_ratio, args=args,
                                              processed_dir=processed_dir, client_data_dir=client_data_dir,
                                              contaminated_client_dir=contaminated_client_dir)


# 以下是原有的训练代码
trainer = FGLTrainer(args)
trainer.train()


# average_index_main(args.aggregation_mode, args.log_dir, "/data2/liujiaqi/OpenFGL-main/weight_Physics.txt", args.graph_repair)