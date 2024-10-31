from src.preprocess_data import ADNI_Dataset, ADNI_three_Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from src.eval_metrics import calculate_metric, compute_metrics


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(140 + 140 + 140 + 100, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, mri, av45, gene, fdg):
        x = torch.cat([mri, av45, gene, fdg], dim=1)  # 拼接输入
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_and_evaluate(train_loader, valid_loader, model, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for i_batch, data in enumerate(train_loader):
            mri = data['mri']
            av45 = data['av45']
            fdg = data['fdg']
            gene = data['gene']
            label = data['label']
            mri, av45, fdg, gene, label = mri.to(device), av45.to(device), fdg.to(device), gene.to(device), label.to(
                device)

            optimizer.zero_grad()
            outputs = model(mri, av45, fdg, gene)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证模型
        model.eval()
        with torch.no_grad():
            for i_batch, data in enumerate(valid_loader):
                mri = data['mri']
                av45 = data['av45']
                fdg = data['fdg']
                gene = data['gene']
                label = data['label']

                mri, av45, fdg, gene, label = mri.to(device), av45.to(device), fdg.to(device), gene.to(
                    device), label.to(device)
                outputs = model(mri, av45, fdg, gene)
                compute_metrics(outputs, label)


def evaluate_test(test_loader, model):
    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
            mri = data['mri']
            av45 = data['av45']
            fdg = data['fdg']
            gene = data['gene']
            label = data['label']

            mri, av45, fdg, gene, label = mri.to(device), av45.to(device), fdg.to(device), gene.to(device), label.to(
                device)
            outputs = model(mri, av45, fdg, gene)
            compute_metrics(outputs, label)


if __name__ == '__main__':

    # train_data = ADNI_Dataset(random_seed=1, mode='train')
    # valid_data = ADNI_Dataset(random_seed=1, mode='val')
    # test_data = ADNI_Dataset(random_seed=1, mode='test')

    # three class
    train_data = ADNI_three_Dataset(random_seed=2, mode='train')
    valid_data = ADNI_three_Dataset(random_seed=2, mode='val')
    test_data = ADNI_three_Dataset(random_seed=2, mode='test')

    if torch.cuda.is_available():
        g_cuda = torch.Generator(device='cuda')
    else:
        g_cuda = torch.Generator()

    g_cpu = torch.Generator(device='cpu')

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=g_cpu)
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=True, generator=g_cpu)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True, generator=g_cpu)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_and_evaluate(train_loader, valid_loader, model, criterion, optimizer, epochs=20)
    print("this is test result")
    evaluate_test(test_loader, model)
