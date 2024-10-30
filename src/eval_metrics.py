import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, matthews_corrcoef


def calculate_metric(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.view(-1).cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.view(-1).cpu().numpy()
    fpr, tpr, thresholds = roc_curve(labels, preds)
    J = tpr - fpr
    optimal_idx = J.argmax()
    optimal_threshold = thresholds[optimal_idx]
    binary_predict = [1 if p >= optimal_threshold else 0 for p in preds]

    accuracy = accuracy_score(labels, binary_predict)  # ACC
    auc = roc_auc_score(labels, preds)  # AUC

    tn, fp, fn, tp = confusion_matrix(labels, binary_predict).ravel()
    sensitivity = tp / (tp + fn)  # SEN
    specificity = tn / (tn + fp)  # SPE

    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    print(f"Optimal Threshold: {optimal_threshold}")
    print(f"Sensitivity (Recall): {sensitivity}")
    print(f"Specificity: {specificity}")


def compute_metrics(preds, labels):
    pred_labels = torch.argmax(preds, dim=1)
    true_labels = torch.argmax(labels, dim=1)
    # ACC
    correct = (pred_labels == true_labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    # AUC
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    auc = roc_auc_score(labels_np, preds_np, average='macro', multi_class='ovr')

    tn, fp, fn, tp = confusion_matrix(true_labels.cpu().numpy(), pred_labels.cpu().numpy()).ravel()
    sen = tp / (tp + fn)  # SEN
    spe = tn / (tn + fp)  # SPE

    # # three class
    # cm = confusion_matrix(true_labels.cpu().numpy(), pred_labels.cpu().numpy())
    # # Specificity = TN / (TN + FP)
    # # Sensitivity (Recall) = TP / (TP + FN)
    # Calculate SPE and SEN one by one
    # specificity_list = []
    # sensitivity_list = []
    # for i in range(3):  # 对每个类别计算
    #     TP = cm[i, i]
    #     FN = cm[i, :].sum() - TP
    #     FP = cm[:, i].sum() - TP
    #     TN = cm.sum() - (TP + FN + FP)
    #     specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    #     sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    #     specificity_list.append(specificity)
    #     sensitivity_list.append(sensitivity)
    # spe = sum(specificity_list) / len(specificity_list)  # SPE
    # sen = sum(sensitivity_list) / len(sensitivity_list)  # SEN

    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    print(f"Sensitivity (Recall): {sen}")
    print(f"Specificity: {spe}")
