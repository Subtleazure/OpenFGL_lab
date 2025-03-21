import re


def parse_input(file_path):
    """
    从文件中读取数据并解析
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # 使用正则表达式提取 contaminated_client_indices 和 weights list(greater)
    pattern = r"contaminated_client_indices:\[(.*?)\]\n(weights list\(greater\): \[.*?\](?:\nweights list\(greater\): \[.*?\])*)"
    matches = re.findall(pattern, content)

    data = []
    for match in matches:
        indices = list(map(int, match[0].split(',')))
        weights_lists = []
        for line in match[1].split('\n'):
            if line.startswith("weights list(greater):"):
                weights = list(map(int, re.findall(r'\d+', line)))
                weights_lists.append(weights)
        data.append((indices, weights_lists))
    return data


def calculate_average_positions(indices, weights_lists):
    """
    计算 contaminated_client_indices 在 weights_lists 中的平均位置
    """
    positions = []
    for weights in weights_lists:
        positions_in_row = []
        for index in indices:
            pos = weights.index(index)
            positions_in_row.append(pos)
        positions.append(positions_in_row)

    average_positions = {}
    for i, index in enumerate(indices):
        index_positions = [row[i] for row in positions]
        average_positions[index] = sum(index_positions) / len(index_positions)
    
    # 计算平均位置的平均值
    avg_of_avg = sum(average_positions.values()) / len(average_positions)
    return average_positions, avg_of_avg


def write_output(file_path, aggregation_mode, data, average_positions, avg_of_avg):
    """
    将结果写入文件
    """
    with open(file_path, 'a') as file:
        file.write(f"{aggregation_mode}\n")
        file.write(f"contaminated_client_indices: {data[0]}\n")
        for index, avg_pos in average_positions.items():
            file.write(f"Index {index}: {avg_pos:.2f}\n")
        file.write(f"Average of average positions: {avg_of_avg:.2f}\n")
        file.write("\n")


def average_index_main(aggregation_mode, input_file, output_file):
    """
    主函数
    """
    # 解析输入文件
    data = parse_input(input_file)

    # 处理每组数据
    for indices, weights_lists in data:
        # 计算平均位置及其平均值
        average_positions, avg_of_avg = calculate_average_positions(indices, weights_lists)
        # 将结果写入输出文件
        write_output(output_file, aggregation_mode, (indices, weights_lists), average_positions, avg_of_avg)


# if __name__ == "__main__":
#     input_file = "/data2/liujiaqi/OpenFGL-main/log_CS.txt"  # 输入文件路径
#     output_file = "/data2/liujiaqi/OpenFGL-main/weight_CS.txt"  # 输出文件路径
#     aggregation_mode="benchmark"
#     average_index_main(aggregation_mode,input_file, output_file)