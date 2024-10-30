import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pyinform.transferentropy import transfer_entropy

# 1. 读取数据，第一列作为索引列
file_path = "../data/MRI.csv"
data = pd.read_csv(file_path, index_col=0)

# 2. 提取第二行至第100行的数据，形成一个 (100, 140) 的矩阵
matrix = data.iloc[1:104, :].values  # 从第二行到第100行 (pandas从0开始计数)

# 3. 初始化传递熵矩阵
n_regions = matrix.shape[1]
te_matrix = np.zeros((n_regions, n_regions))

# 计算传递熵 (pairwise)
for i in range(n_regions):
    for j in range(n_regions):
        if i != j:  # 只计算不同区域之间的传递熵
            te_matrix[i, j] = transfer_entropy(matrix[:, i], matrix[:, j], 3)

# 4. 绘制热度图
plt.figure(figsize=(10, 8))
plt.imshow(te_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Transfer Entropy')
plt.title('Transfer Entropy Between Regions')
plt.xlabel('Region')
plt.ylabel('Region')
plt.show()

