from src.preprocess_data import ADNI_Dataset, ADNI_ALL_Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

train_data = ADNI_Dataset(random_seed=1111, mode='train')
valid_data = ADNI_Dataset(random_seed=1111, mode='val')
test_data = ADNI_Dataset(random_seed=1111, mode='test')

# 三分类
# train_data = ADNI_ALL_Dataset(random_seed=args.seed, mode='train')
# valid_data = ADNI_ALL_Dataset(random_seed=args.seed, mode='val')
# test_data = ADNI_ALL_Dataset(random_seed=args.seed, mode='test')

if torch.cuda.is_available():
    g_cuda = torch.Generator(device='cuda')
else:
    g_cuda = torch.Generator()

g_cpu = torch.Generator(device='cpu')

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, generator=g_cpu)
valid_loader = DataLoader(valid_data, batch_size=8, shuffle=True, generator=g_cpu)
test_loader = DataLoader(test_data, batch_size=8, shuffle=True, generator=g_cpu)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将数据转换为 NumPy 格式
def loader_to_numpy(train_loader):
    X = []
    Y = []
    for i_batch, data in enumerate(train_loader):
        x1 = data['x1']
        x2 = data['x2']
        x4 = data['x4']
        x3 = data['x3']
        labels = data['label']
        # 将数据从 GPU 转移到 CPU 并转换为 NumPy
        x1, x2, x3, x4 = x1.cpu().numpy(), x2.cpu().numpy(), x3.cpu().numpy(), x4.cpu().numpy()
        labels = torch.argmax(labels, dim=1).cpu().numpy()  # 将one-hot标签转换为类别
        # 拼接输入 x1, x2, x3, x4
        X.append(np.concatenate([x1, x2, x3, x4], axis=1))
        Y.append(labels)
    # 将列表转换为 NumPy 数组
    return np.vstack(X), np.hstack(Y)

# 加载训练和验证数据
X_train, Y_train = loader_to_numpy(train_loader)
X_valid, Y_valid = loader_to_numpy(valid_loader)

# 初始化 SVM 模型
svm_model = SVC(kernel='linear', probability=True)  # 使用线性核函数，并启用概率估计

# 训练 SVM 模型
svm_model.fit(X_train, Y_train)

# 验证 SVM 模型
def evaluate_svm(X, Y, model):
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]  # 提取属于类别1的概率
    acc = accuracy_score(Y, preds)
    auc = roc_auc_score(Y, probas)
    cm = confusion_matrix(Y, preds)
    TN, FP, FN, TP = cm.ravel()
    sen = TP / (TP + FN)  # 敏感度
    spe = TN / (TN + FP)  # 特异度
    print(f'ACC: {acc:.5f}, AUC: {auc:.5f}, SEN: {sen:.5f}, SPE: {spe:.5f}')

# 评估训练集
print("Training Set Evaluation:")
evaluate_svm(X_train, Y_train, svm_model)

# 评估验证集
print("Validation Set Evaluation:")
evaluate_svm(X_valid, Y_valid, svm_model)

# 评估测试集
X_test, Y_test = loader_to_numpy(test_loader)
print("Test Set Evaluation:")
evaluate_svm(X_test, Y_test, svm_model)

