from src.preprocess_data import ADNI_Dataset, ADNI_three_Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize


def loader_to_numpy(data_loader):
    X = []
    Y = []
    for i_batch, data in enumerate(data_loader):
        mri = data['mri']
        av45 = data['av45']
        fdg = data['fdg']
        gene = data['gene']
        label = data['label']
        mri, av45, fdg, gene = mri.cpu().numpy(), av45.cpu().numpy(), fdg.cpu().numpy(), gene.cpu().numpy()
        label = torch.argmax(label, dim=1).cpu().numpy()
        X.append(np.concatenate([mri, av45, fdg, gene], axis=1))
        Y.append(label)
    return np.vstack(X), np.hstack(Y)


def evaluate_svm(X, Y, model):
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    acc = accuracy_score(Y, preds)
    auc = roc_auc_score(Y, probas)
    cm = confusion_matrix(Y, preds)
    TN, FP, FN, TP = cm.ravel()
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    print(f'ACC: {acc:.5f}, AUC: {auc:.5f}, SEN: {sen:.5f}, SPE: {spe:.5f}')


def evaluate_three_svm(X, Y, model):
    Y_pred = model.predict(X)
    Y_prob = model.predict_proba(X)
    accuracy = accuracy_score(Y, Y_pred)
    Y_binarized = label_binarize(Y, classes=np.unique(Y))
    auc = roc_auc_score(Y_binarized, Y_prob, average="macro", multi_class="ovr")

    cm = confusion_matrix(Y, Y_pred)
    specificity, sensitivity = [], []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        fn = np.sum(cm[i, :]) - cm[i, i]
        tp = cm[i, i]  # 真阳性

        spe = tn / (tn + fp) if (tn + fp) != 0 else 0
        sen = tp / (tp + fn) if (tp + fn) != 0 else 0

        specificity.append(spe)
        sensitivity.append(sen)

    avg_specificity = np.mean(specificity)
    avg_sensitivity = np.mean(sensitivity)
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    print(f"Sensitivity (Recall): {avg_sensitivity}")
    print(f"Specificity: {avg_specificity}")


if __name__ == '__main__':
    # train_data = ADNI_Dataset(random_seed=1111, mode='train')
    # valid_data = ADNI_Dataset(random_seed=1111, mode='val')
    # test_data = ADNI_Dataset(random_seed=1111, mode='test')

    # three class
    train_data = ADNI_three_Dataset(random_seed=11, mode='train')
    valid_data = ADNI_three_Dataset(random_seed=11, mode='val')
    test_data = ADNI_three_Dataset(random_seed=11, mode='test')

    if torch.cuda.is_available():
        g_cuda = torch.Generator(device='cuda')
    else:
        g_cuda = torch.Generator()

    g_cpu = torch.Generator(device='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=g_cpu)
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=True, generator=g_cpu)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True, generator=g_cpu)
    X_train, Y_train = loader_to_numpy(train_loader)
    X_valid, Y_valid = loader_to_numpy(valid_loader)
    X_test, Y_test = loader_to_numpy(test_loader)

    svm_model = SVC(kernel='linear', probability=True, decision_function_shape='ovr')  # 使用线性核函数，并启用概率估计
    svm_model.fit(X_train, Y_train)

    print("Training Set Evaluation:")
    # evaluate_svm(X_train, Y_train, svm_model)
    evaluate_three_svm(X_train, Y_train, svm_model)

    print("Validation Set Evaluation:")
    # evaluate_svm(X_valid, Y_valid, svm_model)
    evaluate_three_svm(X_valid, Y_valid, svm_model)

    print("Test Set Evaluation:")
    # evaluate_svm(X_test, Y_test, svm_model)
    evaluate_three_svm(X_test, Y_test, svm_model)
