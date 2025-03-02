import torch
import pprint

# 指定.pt文件的路径
file_path = r'D:\desk\WHU\Study\Paper\OpenFGL-main\your_data_root\distrib\subgraph_fl_louvain_1_Cora_client_10\data_1.pt'

# 加载.pt文件
try:
    data = torch.load(file_path)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# 打印数据的基本信息
print(f"Data type: {type(data)}")

# 根据数据类型进行不同处理
if isinstance(data, dict):
    print("This is a dictionary. Here are the keys:")
    for key in data.keys():
        print(f"- {key}")

    # 如果你想查看某个特定键的值
    key_to_check = input(
        "Enter a key to check its value (or press Enter to skip): ").strip()
    if key_to_check and key_to_check in data:
        value = data[key_to_check]
        print(f"Value of '{key_to_check}':")
        pprint.pprint(value)
        print(f"Type of '{key_to_check}': {type(value)}")
        if isinstance(value, torch.Tensor):
            print(f"Shape of tensor: {value.shape}")
            print(f"First few elements: {value[:5]}")
        elif isinstance(value, (list, dict)):
            print(f"Length/Size: {len(value)}")
    else:
        print("Skipping detailed value inspection.")

elif isinstance(data, list):
    print("This is a list. Here are the first few items:")
    pprint.pprint(data[:5])
    print(f"Total length of list: {len(data)}")

elif isinstance(data, torch.Tensor):
    print("This is a tensor. Here are its details:")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Device: {data.device}")
    print(f"First few elements: {data.flatten()[:5]}")

else:
    print("The loaded data is of another type. Here is its content:")
    pprint.pprint(data)
