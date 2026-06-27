import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def plot_timestamp_intervals(csv_file_path, column_index=0, save_fig=False):
    """
    读取CSV文件中的时间戳列，计算相邻时间差并绘制柱状图

    参数:
        csv_file_path (str): CSV文件路径
        column_index (int): 时间戳所在列的索引，默认为0（第一列）
        save_fig (bool): 是否保存图片，默认为False
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        # 提取时间戳列（第一列）
        timestamps = df.iloc[:, column_index].values

        # 检查数据是否有效
        if len(timestamps) < 2:
            print("错误：时间戳数据点少于2个，无法计算时间差")
            return

        # 计算相邻时间戳的差值
        time_diffs = np.diff(timestamps)

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制柱状图：每个柱子从t_i开始，宽度为时间差，高度为时间差
        # align='edge'表示柱子左边缘对齐x坐标
        bars = ax.bar(
            x=timestamps[:-1],  # 每个柱子的左边缘位置
            height=time_diffs,  # 柱子高度为时间差
            width=time_diffs,  # 柱子宽度也为时间差，正好填满两个时间点之间
            align='edge',  # 左边缘对齐
            color='skyblue',
            edgecolor='navy',
            alpha=0.7
        )

        # 添加统计信息
        avg_diff = np.mean(time_diffs)
        max_diff = np.max(time_diffs)
        min_diff = np.min(time_diffs)
        std_diff = np.std(time_diffs)

        stats_text = (f"平均间隔: {avg_diff:.4f} s\n"
                      f"最大间隔: {max_diff:.4f} s\n"
                      f"最小间隔: {min_diff:.4f} s\n"
                      f"标准差: {std_diff:.4f} s")

        # 在图上添加统计信息文本框
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # 设置图表标签和标题
        ax.set_xlabel('时间戳 (s)', fontsize=12)
        ax.set_ylabel('相邻时间戳差值 (s)', fontsize=12)
        ax.set_title('相邻时间戳间隔分布', fontsize=14, fontweight='bold')

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.6)

        # 自动调整x轴标签
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存图片（如果需要）
        if save_fig:
            plt.savefig('timestamp_intervals.png', dpi=300, bbox_inches='tight')
            print("图片已保存为: timestamp_intervals.png")

        # 显示图形
        plt.show()

        # 打印统计信息
        print("\n时间戳间隔统计信息:")
        print(f"总数据点数: {len(timestamps)}")
        print(f"时间间隔数: {len(time_diffs)}")
        print(f"平均间隔: {avg_diff:.4f} 秒")
        print(f"最大间隔: {max_diff:.4f} 秒 (出现在 {timestamps[np.argmax(time_diffs)]:.4f} s 附近)")
        print(f"最小间隔: {min_diff:.4f} 秒 (出现在 {timestamps[np.argmin(time_diffs)]:.4f} s 附近)")
        print(f"标准差: {std_diff:.4f} 秒")

    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_file_path}")
    except Exception as e:
        print(f"发生错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 请将下面的路径替换为您的CSV文件路径
    csv_file = "/home/awwsome/odom_data/odom_data_20260519_200149.csv"

    # 调用函数绘制图表
    plot_timestamp_intervals(csv_file, save_fig=False)