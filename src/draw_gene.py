import pandas as pd
import matplotlib.pyplot as plt


# 定义读取和调整平均值的函数
def calculate_and_adjust_means(file_path):
    # 读取 CSV 文件
    data = pd.read_csv(file_path)

    # 计算每列的平均值
    column_means = data.mean()

    # 找到最小值并调整
    min_value = column_means.min()
    adjusted_means = column_means + abs(min_value)

    return adjusted_means


# 标记前10个最大值的位置
def plot_top10(ax, x, y, color, title):
    # 找到前10个最大值的索引
    top10_indices = y.nlargest(10).index

    # 绘制折线图（无标记）
    ax.plot(x, y, linestyle='-', color=color, marker=None)

    # 标记前10个最大值的点
    ax.plot(top10_indices, y[top10_indices], 'o', color=color)

    # 添加标题和标签
    ax.set_title(title)
    # ax.set_xlabel('Column Index')
    ax.set_ylabel('weight')
    ax.grid(False)

    # 设置 X 轴标签的字体大小和倾斜角度
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=5.5, rotation=70)  # 字体大小为 8，旋转 45 度

# 标记前10个最大值的位置
def plot_top1012(ax, x, y, color, title):
    # 找到前10个最大值的索引
    top10_indices = y.nlargest(10).index

    # 绘制折线图（无标记）
    ax.plot(x, y, linestyle='-', color=color, marker=None)

    # 标记前10个最大值的点
    ax.plot(top10_indices, y[top10_indices], 'o', color=color)

    # 添加标题和标签
    ax.set_title(title)
    # ax.set_xlabel('Column Index')
    ax.set_ylabel('weight')
    ax.grid(False)

    # 隐藏X轴的刻度标签
    ax.tick_params(labelbottom=False, bottom=False)

    # 设置 X 轴标签的字体大小和倾斜角度
    # ax.set_xticks(x)
    # ax.set_xticklabels(x, fontsize=4, rotation=45)  # 字体大小为 8，旋转 45 度

# 读取三个文件并计算调整后的平均值
file1 = 'data/ad_cn/linear_g.csv'
file2 = 'data/ad_mci/linear_g.csv'
file3 = 'data/mci_cn/linear_g.csv'

adjusted_means_file1 = calculate_and_adjust_means(file1)
adjusted_means_file2 = calculate_and_adjust_means(file2)
adjusted_means_file3 = calculate_and_adjust_means(file3)


# 创建一个图形，并添加三个子图
fig, axs = plt.subplots(3, 1, figsize=(14, 8))  # 三行一列的子图，图像尺寸调整宽高

# 绘制第一个文件的折线图，标记前10个最大值
plot_top1012(axs[0], adjusted_means_file1.index, adjusted_means_file1, 'b', 'AD vs. CN')

# 绘制第二个文件的折线图，标记前10个最大值
plot_top1012(axs[1], adjusted_means_file2.index, adjusted_means_file2, 'g', 'AD vs. MCI')

# 绘制第三个文件的折线图，标记前10个最大值
plot_top10(axs[2], adjusted_means_file3.index, adjusted_means_file3, 'r', 'MCI vs. CN')

# 调整子图之间的间距
plt.tight_layout()

# 显示图表
plt.show()