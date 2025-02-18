import torch

# 假设这里是加载数据的代码
file_path = "D:\desk\WHU\Study\Paper\OpenFGL-main\your_data_root\distrib\subgraph_fl_louvain_1_Cora_client_10\data_9.pt"
splitted_data = torch.load(file_path)
print(splitted_data.keys())  # 打印字典的键，检查是否有"data"键
