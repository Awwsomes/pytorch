import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

path_txt = r"D:\A_myData\Pytorch\src\data\yolo11-cls_6000.txt"   # 存放数据的txt

def add_labels(bars, offset, color):
    """在每个条形的末端（柱顶）添加数值标签"""
    for bar in bars:
        # bar.get_height() 是柱子的高度 (即分数) -> 对应Y坐标
        # bar.get_x() 是柱子的X坐标起点
        # bar.get_width() 是柱子的宽度
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2  # X轴中心位置

        # 使用 ax.text() 添加标签
        # 注意：现在标签是垂直放置在柱子上方
        ax.text(x,  # X坐标 (柱子中心)
                height + offset,  # Y坐标 (柱子高度再加一个偏移量)
                f'{height:.3f}',  # 格式化数值，保留3位小数
                ha='center',  # 水平对齐方式：居中
                va='bottom',  # 垂直对齐方式：底部对齐
                fontsize=8,
                color=color,
                rotation=90)  # 【重要】旋转 90 度，避免标签重叠

categories = [i for i in range(31)]

precision = np.zeros(31)

# 读取数据
try:
    with open(path_txt, 'r') as txt_file:
        for idx, line in enumerate(txt_file):
            # 跳过首行
            if idx == 0:
                continue
            # 清理和分割数据
            parts = line.strip().split()
            if len(parts) >= 3:
                label, accuracy, recall_val = parts[0], parts[1], parts[2]

                try:
                    label_int = int(label)
                    if 0 <= label_int < 31:
                        precision[label_int] = float(accuracy)
                    else:
                        print(f"Warning: Category label {label} out of range (0-30).")
                except ValueError:
                    print(f"Error parsing label or score in line: {line.strip()}")
            else:
                print(f"Warning: Skipping line due to insufficient parts: {line.strip()}")

except FileNotFoundError:
    print(f"Error: File not found at {path_txt}. Using dummy data.")
    # 如果文件找不到，使用随机数据以便继续演示
    np.random.seed(42)
    precision = np.random.uniform(0.5, 0.95, 31)

# 将数据整理成 DataFrame
data = pd.DataFrame({
    'Category': categories,
    'Accuracy': precision,
})

# 1. 设置图表参数
# 增大图表宽度以容纳 31 个分组柱子
fig, ax = plt.subplots(figsize=(15, 7))
x_pos = np.arange(len(data['Category']))  # 类别的位置

# 设置柱子的宽度
bar_width = 0.35
offset = bar_width / 2  # 用于分组的偏移量

# 2. 绘制精确率（Precision）柱
# 绘制在 x_pos - offset 位置
bars_precision = ax.bar(x_pos - offset,
                        data['Accuracy'],
                        bar_width,
                        label='Accuracy',
                        color='#FF0000')

# 4. 调用函数分别添加标签
# 精确率标签 (向上偏移 0.005)
add_labels(bars_precision, 0.005, '#FF0000')

# 5. 设置轴标签和标题
ax.set_xticks(x_pos)
# X轴标签为类别编号，并旋转标签以避免重叠
ax.set_xticklabels(data['Category'], rotation=45, ha='right')
ax.set_ylabel('Score', fontsize=12)
txt_name = os.path.basename(path_txt)
txt_name,_ = os.path.splitext(txt_name)
ax.set_title(txt_name, fontsize=14)

# 6. 添加图例和限制
ax.legend()
# Y轴范围通常是 0 到 1
ax.set_ylim(0, data[['Accuracy']].values.max() * 1.05 + 0.05)

# 7. 添加网格线（可选，增强可读性）
ax.yaxis.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()  # 调整布局以适应所有元素和旋转的标签
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# def add_labels(bars, offset, color):
#     """在每个条形的末端添加数值标签"""
#     for bar in bars:
#         # bar.get_width() 是条形的X坐标 (即分数)
#         # bar.get_y() 是条形的Y坐标起点
#         # bar.get_height() 是条形的高度
#         width = bar.get_width()
#         y = bar.get_y() + bar.get_height() / 2 # Y轴中心位置
#
#         # 使用 ax.text() 添加标签
#         ax.text(width + offset,  # X坐标 (条形末端再加一个偏移量)
#                 y,               # Y坐标 (条形中心)
#                 f'{width:.3f}',  # 格式化数值，保留3位小数
#                 ha='left',       # 水平对齐方式：左对齐
#                 va='center',     # 垂直对齐方式：居中
#                 fontsize=8,
#                 color=color)
#
# path_txt = r"D:\A_myData\Pytorch\pytorch2\pytorch2\data\exp.txt"
#
# categories = [i for i in range(31)]
# # print(categories)
#
# # np.random.seed(42)
# # precision = np.random.uniform(0.5,0.95,31)
# # recall = np.random.uniform(0.5,0.95,31)
# precision = np.zeros(31)
# recalls = np.zeros(31)
#
# with open(path_txt,'r') as txt_file:
#     for idx,line in enumerate(txt_file):
#         # 跳过首行
#         if idx == 0:
#             continue
#         label,accuracy,recall = line.split(" ")
#         precision[int(label)] = float(accuracy)
#         recalls[int(label)] = float(recall)
#
# # 将数据整理成 DataFrame
# data = pd.DataFrame({
#     'Category': categories,
#     'Precision': precision,
#     'Recall': recall
# })
#
# # 1. 设置图表参数
# _, ax = plt.subplots(figsize=(15, 10)) # 增大高度以容纳 31 个类别
# y_pos = np.arange(len(data['Category'])) # 类别的位置
#
# # 设置条的宽度
# bar_height = 0.35
#
# # 2. 绘制精确率（Precision）条
# # 将第一个条向上平移 bar_height / 2
# bars_precision = ax.barh(y_pos - bar_height/2,
#         data['Precision'],
#         bar_height,
#         label='Precision',
#         color='#FF0000')
#
# # 4. 调用函数分别添加标签
# # 精确率标签 (向右偏移 0.005)
# add_labels(bars_precision, 0.005, '#4C72B0')
#
# # # 3. 绘制召回率（Recall）条
# # # 将第二个条向下平移 bar_height / 2
# # ax.barh(y_pos + bar_height/2,
# #         data['Recall'],
# #         bar_height,
# #         label='Recall',
# #         color='#55A868')
#
# # 4. 设置轴标签和标题
# ax.set_yticks(y_pos)
# ax.set_yticklabels(data['Category'])
# ax.invert_yaxis()  # 标签从上到下排列
# ax.set_xlabel('Score', fontsize=12)
# ax.set_title('31 Categories Performance: Precision vs. Recall', fontsize=14)
#
# # 5. 添加图例和限制
# ax.legend()
# ax.set_xlim(0, 1) # 分数范围通常是 0 到 1
#
# # 6. 添加网格线（可选，增强可读性）
# ax.xaxis.grid(True, linestyle='--', alpha=0.6)
#
# plt.tight_layout() # 调整布局以适应所有元素
# plt.show()
