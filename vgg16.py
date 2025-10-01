import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
import time
import copy
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
import seaborn as sns
from itertools import cycle
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# 配置参数
class Config:
    data_dir = "dataset"  # 替换为你的数据集路径
    classes = ['UA', 'NUA', 'MIX']  # 类别标签
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.001
    step_size = 10
    gamma = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_dir = "Vgg16_experiment_results"
    os.makedirs(save_dir, exist_ok=True)

# 自定义数据集类
class StoneCTDataset(Dataset):
    def __init__(self, image_paths, labels, patient_ids, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.patient_ids = patient_ids
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path, patient_id

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集路径
def load_dataset_paths(data_dir, classes):
    image_paths = []
    labels = []
    patient_ids = []
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Loading dataset from: {data_dir}")
    
    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"Warning: Directory {cls_dir} does not exist. Skipping.")
            continue
            
        print(f"Processing class: {cls_name}")
        img_count = 0
        
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(cls_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_to_idx[cls_name])
                
                # 从文件名提取患者ID - 使用简单可靠的方法
                base_name = os.path.splitext(img_name)[0]
                
                # 简单方法：使用下划线分割，取第一部分
                if '_' in base_name:
                    patient_id = base_name.split('_')[0]
                else:
                    patient_id = base_name
                
                patient_ids.append(patient_id)
                img_count += 1
                
        print(f"  - Found {img_count} images in {cls_name}")
    
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Unique patients: {len(set(patient_ids))}")
    
    return image_paths, labels, patient_ids

# 训练模型
def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # 使用tqdm添加进度条
            for inputs, labels, _, _ in tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch+1}'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts)
    return model, best_acc

# 绘制并保存ROC曲线
def plot_roc_curve(y_true, y_prob, classes, save_path, timestamp, dpi=300):
    # 计算每个类别的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算宏平均ROC曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    
    # 保存图像
    roc_path = os.path.join(save_path, f'roc_curve_{timestamp}.png')
    plt.savefig(roc_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 保存ROC数据
    roc_data = pd.DataFrame()
    for i in range(n_classes):
        temp_df = pd.DataFrame({
            'Class': classes[i],
            'FPR': fpr[i],
            'TPR': tpr[i],
            'AUC': roc_auc[i]
        })
        roc_data = pd.concat([roc_data, temp_df], ignore_index=True)
    
    # 添加宏平均数据
    macro_df = pd.DataFrame({
        'Class': 'Macro',
        'FPR': fpr["macro"],
        'TPR': tpr["macro"],
        'AUC': roc_auc["macro"]
    })
    roc_data = pd.concat([roc_data, macro_df], ignore_index=True)
    
    roc_data_path = os.path.join(save_path, f'roc_data_{timestamp}.csv')
    roc_data.to_csv(roc_data_path, index=False)
    
    return roc_path, roc_data_path

# 绘制并保存PR曲线
def plot_pr_curve(y_true, y_prob, classes, save_path, timestamp, dpi=300):
    # 计算每个类别的PR曲线
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = len(classes)
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_prob[:, i])
    
    # 计算宏平均PR曲线
    all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
    mean_precision /= n_classes
    precision["macro"] = mean_precision
    recall["macro"] = all_recall
    average_precision["macro"] = np.mean(list(average_precision.values()))
    
    # 绘制PR曲线
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'PR curve of class {classes[i]} (AP = {average_precision[i]:.2f})')
    
    plt.plot(recall["macro"], precision["macro"],
             label=f'Macro-average PR curve (AP = {average_precision["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    
    # 保存图像
    pr_path = os.path.join(save_path, f'pr_curve_{timestamp}.png')
    plt.savefig(pr_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 保存PR数据
    pr_data = pd.DataFrame()
    for i in range(n_classes):
        temp_df = pd.DataFrame({
            'Class': classes[i],
            'Recall': recall[i],
            'Precision': precision[i],
            'AP': average_precision[i]
        })
        pr_data = pd.concat([pr_data, temp_df], ignore_index=True)
    
    # 添加宏平均数据
    macro_df = pd.DataFrame({
        'Class': 'Macro',
        'Recall': recall["macro"],
        'Precision': precision["macro"],
        'AP': average_precision["macro"]
    })
    pr_data = pd.concat([pr_data, macro_df], ignore_index=True)
    
    pr_data_path = os.path.join(save_path, f'pr_data_{timestamp}.csv')
    pr_data.to_csv(pr_data_path, index=False)
    
    return pr_path, pr_data_path

# 绘制并保存混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes, save_path, timestamp, dpi=300, normalize=True):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix (Counts)'
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt=fmt, 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    
    # 保存图像
    cm_path = os.path.join(save_path, f'confusion_matrix_{"norm" if normalize else "count"}_{timestamp}.png')
    plt.savefig(cm_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 保存混淆矩阵数据
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_data_path = os.path.join(save_path, f'confusion_matrix_{"norm" if normalize else "count"}_{timestamp}.csv')
    cm_df.to_csv(cm_data_path)
    
    return cm_path, cm_data_path

# 计算并保存评估指标
def calculate_and_save_metrics(y_true, y_pred, y_prob, classes, save_path, timestamp):
    # 计算多分类指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 计算每个类别的指标
    class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # 计算特异性和敏感性（敏感性就是召回率）
    # 特异性需要单独计算
    cm = confusion_matrix(y_true, y_pred)
    specificity = []
    for i in range(len(classes)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[i, :], i))
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    
    # 创建指标数据框
    metrics_data = pd.DataFrame({
        'Class': classes + ['Macro Average'],
        'Precision': list(class_precision) + [precision],
        'Recall/Sensitivity': list(class_recall) + [recall],
        'Specificity': specificity + [np.mean(specificity)],
        'F1-Score': list(class_f1) + [f1],
        'Accuracy': [accuracy] * (len(classes) + 1)
    })
    
    # 保存指标到CSV文件
    metrics_path = os.path.join(save_path, f'evaluation_metrics_{timestamp}.csv')
    metrics_data.to_csv(metrics_path, index=False)
    
    # 保存详细报告
    report_path = os.path.join(save_path, f'classification_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall/Sensitivity: {recall:.4f}\n")
        f.write(f"Macro F1-Score: {f1:.4f}\n")
        f.write(f"Average Specificity: {np.mean(specificity):.4f}\n\n")
        
        f.write("Per-class Metrics:\n")
        f.write("-" * 50 + "\n")
        for i, cls in enumerate(classes):
            f.write(f"{cls}:\n")
            f.write(f"  Precision: {class_precision[i]:.4f}\n")
            f.write(f"  Recall/Sensitivity: {class_recall[i]:.4f}\n")
            f.write(f"  Specificity: {specificity[i]:.4f}\n")
            f.write(f"  F1-Score: {class_f1[i]:.4f}\n\n")
    
    return metrics_path, report_path

# 绘制DCA曲线
def plot_dca_curve(y_true, y_prob, classes, save_path, timestamp, dpi=600):
    n_classes = len(classes)
    n_samples = len(y_true)
    
    # 为每个类别计算DCA曲线
    thresholds = np.linspace(0, 1, 100)
    net_benefit = np.zeros((n_classes, len(thresholds)))
    
    for cls_idx in range(n_classes):
        # 将当前类别视为阳性，其他类别视为阴性
        y_true_binary = (y_true == cls_idx).astype(int)
        y_prob_binary = y_prob[:, cls_idx]
        
        for i, threshold in enumerate(thresholds):
            # 应用阈值
            y_pred_binary = (y_prob_binary >= threshold).astype(int)
            
            # 计算真阳性、假阳性
            tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
            
            # 计算净收益
            net_benefit[cls_idx, i] = tp / n_samples - fp / n_samples * (threshold / (1 - threshold))
    
    # 绘制DCA曲线
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    
    for cls_idx, color in zip(range(n_classes), colors):
        plt.plot(thresholds, net_benefit[cls_idx], color=color, lw=2,
                 label=f'DCA curve of class {classes[cls_idx]}')
    
    # 添加所有治疗和无治疗的参考线
    plt.plot(thresholds, np.zeros(len(thresholds)), 'k--', lw=2, label='Treat None')
    plt.plot(thresholds, np.mean(y_true == 0) - (1 - np.mean(y_true == 0)) * thresholds / (1 - thresholds), 
             'k:', lw=2, label='Treat All')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, 0.3])  # 调整Y轴范围以适应净收益值
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend(loc="upper right")
    
    # 保存图像
    dca_path = os.path.join(save_path, f'dca_curve_{timestamp}.png')
    plt.savefig(dca_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 保存DCA数据
    dca_data = pd.DataFrame()
    for cls_idx in range(n_classes):
        temp_df = pd.DataFrame({
            'Class': classes[cls_idx],
            'Threshold': thresholds,
            'Net_Benefit': net_benefit[cls_idx]
        })
        dca_data = pd.concat([dca_data, temp_df], ignore_index=True)
    
    dca_data_path = os.path.join(save_path, f'dca_data_{timestamp}.csv')
    dca_data.to_csv(dca_data_path, index=False)
    
    return dca_path, dca_data_path

# 绘制CIC曲线
def plot_cic_curve(y_true, y_pred, y_prob, classes, save_path, timestamp, dpi=600):
    n_classes = len(classes)
    n_samples = len(y_true)
    
    # 为每个类别计算CIC曲线
    thresholds = np.linspace(0, 1, 100)
    high_risk_count = np.zeros((n_classes, len(thresholds)))
    true_positive_count = np.zeros((n_classes, len(thresholds)))
    
    for cls_idx in range(n_classes):
        # 将当前类别视为阳性，其他类别视为阴性
        y_true_binary = (y_true == cls_idx).astype(int)
        y_prob_binary = y_prob[:, cls_idx]
        
        for i, threshold in enumerate(thresholds):
            # 应用阈值
            high_risk = (y_prob_binary >= threshold).astype(int)
            high_risk_count[cls_idx, i] = np.sum(high_risk)
            true_positive_count[cls_idx, i] = np.sum(high_risk & y_true_binary)
    
    # 绘制CIC曲线
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    
    for cls_idx, color in zip(range(n_classes), colors):
        plt.plot(thresholds, high_risk_count[cls_idx] / n_samples * 100, color=color, lw=2,
                 label=f'High risk count - {classes[cls_idx]}')
        plt.plot(thresholds, true_positive_count[cls_idx] / n_samples * 100, color=color, linestyle='--', lw=2,
                 label=f'True positive count - {classes[cls_idx]}')
    
    plt.xlim([0.0, 1.0])
    plt.xlabel('Threshold Probability')
    plt.ylabel('Percentage of Patients (%)')
    plt.title('Clinical Impact Curve')
    plt.legend(loc="upper right")
    
    # 保存图像
    cic_path = os.path.join(save_path, f'cic_curve_{timestamp}.png')
    plt.savefig(cic_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 保存CIC数据
    cic_data = pd.DataFrame()
    for cls_idx in range(n_classes):
        temp_df = pd.DataFrame({
            'Class': classes[cls_idx],
            'Threshold': thresholds,
            'High_Risk_Count': high_risk_count[cls_idx],
            'True_Positive_Count': true_positive_count[cls_idx]
        })
        cic_data = pd.concat([cic_data, temp_df], ignore_index=True)
    
    cic_data_path = os.path.join(save_path, f'cic_data_{timestamp}.csv')
    cic_data.to_csv(cic_data_path, index=False)
    
    return cic_path, cic_data_path

# 主函数
def main():
    config = Config()
    
    # 加载数据集
    image_paths, labels, patient_ids = load_dataset_paths(config.data_dir, config.classes)
    
    # 检查数据是否加载成功
    if len(image_paths) == 0:
        raise ValueError("No images found in the dataset directory. Please check your data path.")
    
    # 确保所有数组长度一致
    assert len(image_paths) == len(labels) == len(patient_ids), (
        f"Mismatch in data arrays lengths! "
        f"image_paths: {len(image_paths)}, "
        f"labels: {len(labels)}, "
        f"patient_ids: {len(patient_ids)}"
    )
    
    # 转换为NumPy数组
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    
    # 打印数据集统计信息
    unique_patients, patient_counts = np.unique(patient_ids, return_counts=True)
    print(f"\nDataset Statistics:")
    print(f"Total images: {len(image_paths)}")
    print(f"Total patients: {len(unique_patients)}")
    print(f"Class distribution: {np.bincount(labels)}")
    print(f"Patients per class:")
    for i, cls in enumerate(config.classes):
        cls_patients = np.unique(patient_ids[labels == i])
        print(f"  {cls}: {len(cls_patients)} patients")
    
    # 使用患者ID进行分组留一法
    logo = LeaveOneGroupOut()
    groups = patient_ids
    
    # 初始化结果存储
    all_results = []
    model_paths = []
    fold_accuracies = []
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    
    print(f"\nStarting Leave-One-Out validation with {len(unique_patients)} groups...")
    
    # 创建用于调试的数据备份
    debug_dir = os.path.join(config.save_dir, "debug_data")
    os.makedirs(debug_dir, exist_ok=True)
    
    # 创建绘图目录
    plots_dir = os.path.join(config.save_dir, "evaluation_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 遍历每个分组
    for fold, (train_idx, test_idx) in enumerate(logo.split(image_paths, labels, groups)):
        print(f'\nFold {fold+1}/{len(unique_patients)}')
        print('=' * 50)
        
        # 打印索引信息用于调试
        print(f"Train indices count: {len(train_idx)}")
        print(f"Test indices count: {len(test_idx)}")
        print(f"Labels array size: {len(labels)}")
        
        # 确保测试集不为空
        if len(test_idx) == 0:
            print(f"Skipping fold {fold+1} as test set is empty.")
            continue
            
        # 检查索引是否在有效范围内
        if max(test_idx) >= len(labels):
            print(f"Error: Test index {max(test_idx)} is out of bounds for labels array (size {len(labels)}). Skipping fold.")
            continue
        
        # 获取测试患者信息
        test_patient = patient_ids[test_idx[0]]
        test_class_idx = labels[test_idx[0]]
        test_class = config.classes[test_class_idx]
        
        print(f"Testing on patient: {test_patient} (class: {test_class})")
        print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        # 保存当前分组信息用于调试
        fold_debug_dir = os.path.join(debug_dir, f"fold_{fold+1}")
        os.makedirs(fold_debug_dir, exist_ok=True)
        
        with open(os.path.join(fold_debug_dir, "split_info.txt"), "w") as f:
            f.write(f"Fold: {fold+1}\n")
            f.write(f"Test patient: {test_patient}\n")
            f.write(f"Test class: {test_class}\n")
            f.write(f"Test samples: {len(test_idx)}\n")
            f.write("\nTraining samples:\n")
            for idx in train_idx:
                f.write(f"{image_paths[idx]}, {patient_ids[idx]}, {config.classes[labels[idx]]}\n")
            f.write("\nTest samples:\n")
            for idx in test_idx:
                f.write(f"{image_paths[idx]}, {patient_ids[idx]}, {config.classes[labels[idx]]}\n")
        
        # 划分数据集 - 使用列表推导式避免索引问题
        train_paths = [image_paths[i] for i in train_idx]
        test_paths = [image_paths[i] for i in test_idx]
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]
        train_pids = [patient_ids[i] for i in train_idx]
        test_pids = [patient_ids[i] for i in test_idx]
        
        # 创建数据集和数据加载器
        train_dataset = StoneCTDataset(train_paths, train_labels, train_pids, data_transforms['train'])
        test_dataset = StoneCTDataset(test_paths, test_labels, test_pids, data_transforms['val'])
        
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=config.batch_size, 
                               shuffle=True, num_workers=4, pin_memory=True),
            'val': DataLoader(test_dataset, batch_size=config.batch_size, 
                             shuffle=False, num_workers=4, pin_memory=True)
        }
        
        # 初始化模型 - 修改为VGG16
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # 修改分类器最后一层
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, len(config.classes))
        model = model.to(config.device)
        
        # 设置优化器和学习率调度器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
        
        # 训练模型
        model, best_acc = train_model(
            model, criterion, optimizer, scheduler, 
            dataloaders, config.num_epochs, config.device
        )
        
        # 保存模型
        model_path = os.path.join(config.save_dir, f'model_fold_{fold+1}.pth')
        torch.save(model.state_dict(), model_path)
        model_paths.append(model_path)
        
        # 在测试集上进行最终评估
        model.eval()
        all_preds = []
        all_labels = []
        all_paths = []
        all_pids = []
        all_probs = []  # 存储预测概率
        
        with torch.no_grad():
            for inputs, labels_val, paths, pids in dataloaders['val']:
                inputs = inputs.to(config.device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_val.numpy())
                all_paths.extend(paths)
                all_pids.extend(pids)
                all_probs.extend(probs.cpu().numpy())
        
        # 保存结果
        fold_results = pd.DataFrame({
            'patient_id': all_pids,
            'image_path': all_paths,
            'true_label': [config.classes[i] for i in all_labels],
            'predicted_label': [config.classes[i] for i in all_preds],
            'correct': [t == p for t, p in zip(all_labels, all_preds)],
            'fold': fold+1
        })
        
        # 添加每个类别的概率
        for i, cls in enumerate(config.classes):
            fold_results[f'prob_{cls}'] = [p[i] for p in all_probs]
        
        all_results.append(fold_results)
        
        # 收集数据用于整体评估
        all_true_labels.extend(all_labels)
        all_pred_labels.extend(all_preds)
        all_pred_probs.extend(all_probs)
        
        fold_accuracy = np.mean(fold_results['correct'])
        fold_accuracies.append(fold_accuracy)
        print(f'Fold {fold+1} Test Accuracy: {fold_accuracy:.4f}')
        print(f'Test Patient ID: {test_patient} with {len(test_idx)} images')
        
        # 保存当前fold的结果
        fold_results.to_csv(os.path.join(fold_debug_dir, f"fold_{fold+1}_results.csv"), index=False)
    
    # 合并所有结果
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
    else:
        print("No results to save. Exiting.")
        return
    
    # 计算总体准确率
    overall_acc = final_results['correct'].mean()
    print(f'\nOverall Accuracy: {overall_acc:.4f}')
    print(f'Average Fold Accuracy: {np.mean(fold_accuracies):.4f}')
    
    # 保存结果和配置
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(config.save_dir, f'results_{timestamp}.csv')
    model_list_file = os.path.join(config.save_dir, f'model_paths_{timestamp}.txt')
    summary_file = os.path.join(config.save_dir, f'summary_{timestamp}.txt')
    
    final_results.to_csv(results_csv, index=False)
    
    with open(model_list_file, 'w') as f:
        f.write("\n".join(model_paths))
    
    with open(summary_file, 'w') as f:
        f.write(f"Experiment Summary ({timestamp})\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset Directory: {config.data_dir}\n")
        f.write(f"Total Samples: {len(image_paths)}\n")
        f.write(f"Patients: {len(unique_patients)}\n")
        f.write(f"Classes: {config.classes}\n")
        f.write(f"Class Distribution: {np.bincount(labels).tolist()}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Epochs: {config.num_epochs}\n")
        f.write(f"Learning Rate: {config.learning_rate}\n")
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n")
        f.write(f"Average Fold Accuracy: {np.mean(fold_accuracies):.4f}\n\n")
        f.write("Per-fold Accuracy:\n")
        for fold, acc in enumerate(fold_accuracies):
            f.write(f"  Fold {fold+1}: {acc:.4f}\n")
        
        # 添加混淆矩阵
        f.write("\nConfusion Matrix:\n")
        conf_matrix = pd.crosstab(final_results['true_label'], 
                                 final_results['predicted_label'], 
                                 rownames=['Actual'], 
                                 colnames=['Predicted'])
        f.write(conf_matrix.to_string())
    
    # 保存混淆矩阵
    conf_matrix.to_csv(os.path.join(config.save_dir, f'confusion_matrix_{timestamp}.csv'))
    
    # 绘制评估曲线和矩阵
    print("\nGenerating evaluation plots...")
    
    # 准备数据
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels)
    y_prob = np.array(all_pred_probs)
    
    # 将标签转换为one-hot编码
    y_true_onehot = np.zeros((len(y_true), len(config.classes)))
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1
    
    # 绘制ROC曲线
    roc_plot_path, roc_data_path = plot_roc_curve(
        y_true_onehot, y_prob, config.classes, plots_dir, timestamp, dpi=600
    )
    print(f"Saved ROC curve to: {roc_plot_path}")
    print(f"Saved ROC data to: {roc_data_path}")
    
    # 绘制PR曲线
    pr_plot_path, pr_data_path = plot_pr_curve(
        y_true_onehot, y_prob, config.classes, plots_dir, timestamp, dpi=600
    )
    print(f"Saved PR curve to: {pr_plot_path}")
    print(f"Saved PR data to: {pr_data_path}")
    
    # 绘制混淆矩阵（归一化和原始计数）
    cm_norm_plot_path, cm_norm_data_path = plot_confusion_matrix(
        y_true, y_pred, config.classes, plots_dir, timestamp, dpi=600, normalize=True
    )
    print(f"Saved normalized confusion matrix to: {cm_norm_plot_path}")
    print(f"Saved normalized confusion matrix data to: {cm_norm_data_path}")
    
    cm_count_plot_path, cm_count_data_path = plot_confusion_matrix(
        y_true, y_pred, config.classes, plots_dir, timestamp, dpi=600, normalize=False
    )
    print(f"Saved count confusion matrix to: {cm_count_plot_path}")
    print(f"Saved count confusion matrix data to: {cm_count_data_path}")
    
    # 计算并保存评估指标
    metrics_path, report_path = calculate_and_save_metrics(
        y_true, y_pred, y_prob, config.classes, config.save_dir, timestamp
    )
    print(f"Saved evaluation metrics to: {metrics_path}")
    print(f"Saved classification report to: {report_path}")
    
    # 绘制DCA曲线
    dca_plot_path, dca_data_path = plot_dca_curve(
        y_true, y_prob, config.classes, plots_dir, timestamp, dpi=600
    )
    print(f"Saved DCA curve to: {dca_plot_path}")
    print(f"Saved DCA data to: {dca_data_path}")
    
    # 绘制CIC曲线
    cic_plot_path, cic_data_path = plot_cic_curve(
        y_true, y_pred, y_prob, config.classes, plots_dir, timestamp, dpi=600
    )
    print(f"Saved CIC curve to: {cic_plot_path}")
    print(f"Saved CIC data to: {cic_data_path}")
    
    print(f"\nExperiment completed. Results saved to {config.save_dir}")
    print(f"Final results CSV: {results_csv}")
    print(f"Summary file: {summary_file}")

if __name__ == "__main__":
    main()