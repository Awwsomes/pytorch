import os

def convert_labels(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取原标签文件
            with open(input_path, 'r') as f:
                lines = f.readlines()

            # 修改类别序号为1
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:  # 跳过空行
                    parts[0] = '0'  # 将类别序号改为0
                    new_lines.append(' '.join(parts) + '\n')

            # 写入新标签文件
            with open(output_path, 'w') as f:
                f.writelines(new_lines)

    print(f"处理完成！共修改 {len([f for f in os.listdir(input_folder) if f.endswith('.txt')])} 个标签文件")

# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 替换为你的输入和输出文件夹路径
    INPUT_LABELS_DIR = "D:\A_myData\RC26-Vision\dataset\juanZhou_obb1\labels"  # 原标签文件夹
    OUTPUT_LABELS_DIR = "D:\A_myData\RC26-Vision\dataset\juanZhou_obb1\labels_all_one"  # 新标签文件夹

    convert_labels(INPUT_LABELS_DIR, OUTPUT_LABELS_DIR)