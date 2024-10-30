from src.preprocess_data import ADNI_Dataset, ADNI_ALL_Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

train_data = ADNI_Dataset(random_seed=1111, mode='train')
valid_data = ADNI_Dataset(random_seed=1111, mode='val')
test_data = ADNI_Dataset(random_seed=1111, mode='test')

if torch.cuda.is_available():
    g_cuda = torch.Generator(device='cuda')
else:
    g_cuda = torch.Generator()

g_cpu = torch.Generator(device='cpu')

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, generator=g_cpu)
valid_loader = DataLoader(valid_data, batch_size=8, shuffle=True, generator=g_cpu)
test_loader = DataLoader(test_data, batch_size=8, shuffle=True, generator=g_cpu)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(140 + 140 + 140 + 100, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)  # 拼接输入
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def calc_sen_spe(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    return sen, spe

def train_and_evaluate(train_loader, valid_loader, model, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []

        for i_batch, data in enumerate(train_loader):
            x1 = data['x1']
            x2 = data['x2']
            x4 = data['x4']
            x3 = data['x3']
            labels = data['label']

            x1, x2, x3, x4, labels = x1.to(device), x2.to(device), x3.to(device), x4.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(x1, x2, x3, x4)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds).cpu().numpy()  # 转移到CPU计算
        all_labels = torch.cat(all_labels)
        all_labels = torch.argmax(all_labels, dim=1).cpu().numpy()  # 转换为类别索引
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)
        sen, spe = calc_sen_spe(all_labels, all_preds)

        print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, ACC: {acc:.4f}, AUC: {auc:.4f}, SEN: {sen:.4f}, SPE: {spe:.4f}')

        # 验证模型
        model.eval()
        valid_preds = []
        valid_labels = []
        with torch.no_grad():
            for i_batch, data in enumerate(valid_loader):
                x1 = data['x1']
                x2 = data['x2']
                x4 = data['x4']
                x3 = data['x3']
                labels = data['label']

                x1, x2, x3, x4, labels = x1.to(device), x2.to(device), x3.to(device), x4.to(device), labels.to(device)
                outputs = model(x1, x2, x3, x4)
                _, preds = torch.max(outputs, 1)
                valid_preds.append(preds)
                valid_labels.append(labels)

        valid_preds = torch.cat(valid_preds).cpu().numpy()
        valid_labels = torch.cat(valid_labels)
        valid_labels = torch.argmax(valid_labels, dim=1).cpu().numpy()  # 转换为类别索引
        acc = accuracy_score(valid_labels, valid_preds)
        auc = roc_auc_score(valid_labels, valid_preds)
        sen, spe = calc_sen_spe(valid_labels, valid_preds)

        print(f'Validation - ACC: {acc:.4f}, AUC: {auc:.4f}, SEN: {sen:.4f}, SPE: {spe:.4f}')


train_and_evaluate(train_loader, valid_loader, model, criterion, optimizer, epochs=20)


def evaluate_test(test_loader, model):
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
            x1 = data['x1']
            x2 = data['x2']
            x4 = data['x4']
            x3 = data['x3']
            labels = data['label']

            x1, x2, x3, x4, labels = x1.to(device), x2.to(device), x3.to(device), x4.to(device), labels.to(device)
            outputs = model(x1, x2, x3, x4)
            _, preds = torch.max(outputs, 1)
            test_preds.append(preds)
            test_labels.append(labels)

    test_preds = torch.cat(test_preds).cpu().numpy()
    test_labels = torch.cat(test_labels)
    test_labels = torch.argmax(test_labels, dim=1).cpu().numpy()  # 转换为类别索引
    acc = accuracy_score(test_labels, test_preds)
    auc = roc_auc_score(test_labels, test_preds)
    sen, spe = calc_sen_spe(test_labels, test_preds)

    print(f'Test Set - ACC: {acc:.5f}, AUC: {auc:.5f}, SEN: {sen:.5f}, SPE: {spe:.5f}')

evaluate_test(test_loader, model)
