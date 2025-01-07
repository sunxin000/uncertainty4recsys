import os

# 指定要处理的目录和文件顺序
directory = 'new_results'
order = ['raw', 'MC_Dropout', 'Deep_Ensembles', 'dfl', 'Platt_Scaling']
datasets = ['kuairand']
directions = ['ips', 'DR', 'MRDR']

# 用于存储所有结果的列表
all_results = []

# 循环处理每个数据集和方向的组合
for data in datasets:
    for dir_type in directions:
        prefix = f'mean_{data}_{dir_type}_'
        results = []

        # 遍历指定顺序的文件
        for name in order:
            filename = os.path.join(directory, f'{prefix}{name}.txt')
            try:
                with open(filename, 'r') as f:
                    # 读取第一行并获取第一个值
                    first_line = f.readline().strip()
                    first_value = first_line.split()[0]  # 假设值是用空格分隔的
                    results.append(first_value)
            except FileNotFoundError:
                print(f"警告: 文件 {filename} 未找到")
                results.append("N/A")
        
        # 添加标识行和结果到总结果列表
        all_results.append(f"=== {data}_{dir_type} ===")
        all_results.extend(results)
        all_results.append("")  # 添加空行分隔

# 将所有结果写入单个文件
output_file = 'all_extracted_values.txt'
with open(output_file, 'w') as f:
    for line in all_results:
        f.write(f"{line}\n")

print(f"所有结果已保存到 {output_file}")