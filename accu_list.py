import re
from openfgl.flcore.curves import *

def extract_y_array(html_file_path, output_file_path):
    # 读取HTML文件内容
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用正则表达式匹配 "y": [...] 部分
    pattern = r'"y":\s*\[[\d.,\s]*\]'
    match = re.search(pattern, content)

    if match:
        # 提取匹配到的部分
        y_data = match.group(0)
        # 写入到新的txt文件
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(y_data)
        print(f"数据已保存到 {output_file_path}")
    else:
        print("未找到符合条件的'y'数组数据")


def read_y_data(html_file_path):
    # 读取txt文件内容
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # 使用正则表达式匹配 "y": [...] 部分
    pattern = r'"y":\s*\[[\d.,\s]*\]'
    match = re.search(pattern, content)

    if match:
        # 提取匹配到的部分
        y_data = match.group(0)
    else:
        print("未找到符合条件的'y'数组数据")

    # 提取数组部分，去掉"y": 和多余的空格
    y_data_str = y_data.split('[')[1].split(']')[0]
    # 转换为浮点数列表
    y_data_list = [float(x) for x in y_data_str.split(',')]
    return y_data_list


# 使用示例
html_file = "/data2/liujiaqi/curves/Amz-Comp/accuracy_curve_5000_lr_0_003_Amz-Comp_1.html"  # 替换为你的HTML文件路径
# output_file = "y_data.txt"  # 输出文件路径
# extract_y_array(html_file, output_file)
accu_list = read_y_data(html_file)
plot_smoothed_accuracy_plotly(accuracies=accu_list,save_path='11.html',window_len=800)
