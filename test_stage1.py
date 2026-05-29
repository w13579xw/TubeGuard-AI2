import os
import argparse
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.topovarad import TopoVarAD, TopoVarADConfig
from data.dataset import CSVDataset, MVTecDataset
from utils.metrics import MetricsCalculator, compute_f1_max


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_datasets(config):
    """根据配置构建测试数据集。"""
    data_config = config.get('data', {})
    dataset_type = data_config.get('dataset_type', 'csv')

    if dataset_type == 'mvtec':
        test_dataset = MVTecDataset(
            root=data_config.get('dataset_path', 'data/mvtec'),
            category=data_config.get('category', 'bottle'),
            split='test',
            image_size=data_config.get('image_size', 512),
        )
    else:
        test_dataset = CSVDataset(
            csv_path=data_config.get('test_csv', 'data/test.csv'),
            images_dir=data_config.get('images_dir', 'data/images'),
            split='test',
            image_size=data_config.get('image_size', 512),
            augment=False,
        )

    return test_dataset


@torch.no_grad()
def evaluate_detailed(model, loader, device):
    """
    详细评估模型性能，包括：
    - AUROC, AU-PR, F1max (图像级和像素级)
    - 准确率、精确率、召回率、F1 (基于最佳阈值)
    - 混淆矩阵
    """
    model.eval()
    metrics_calc = MetricsCalculator()

    all_image_scores = []
    all_image_labels = []
    all_pixel_scores = []
    all_pixel_labels = []

    print("Evaluating model...")
    for batch in tqdm(loader, desc='Testing'):
        images = batch['image'].to(device)
        labels = batch['label']
        masks = batch['mask']

        image_scores, pixel_scores = model.predict(images)

        all_image_scores.append(image_scores.cpu().numpy())
        all_image_labels.append(labels.numpy())

        if masks.sum() > 0:
            all_pixel_scores.append(pixel_scores.cpu().numpy())
            all_pixel_labels.append(masks.numpy())

        metrics_calc.update(
            image_scores.cpu().numpy(),
            labels.numpy(),
            pixel_scores.cpu().numpy() if masks.sum() > 0 else None,
            masks.numpy() if masks.sum() > 0 else None,
        )

    # 合并所有结果
    image_scores = np.concatenate(all_image_scores)
    image_labels = np.concatenate(all_image_labels)

    # 计算基础指标 (AUROC, AU-PR, F1max)
    base_metrics = metrics_calc.compute()

    # 找到最佳阈值
    f1_max, best_threshold = compute_f1_max(image_scores, image_labels)

    # 基于最佳阈值计算分类指标
    image_preds = (image_scores >= best_threshold).astype(int)

    accuracy = accuracy_score(image_labels, image_preds)
    precision = precision_score(image_labels, image_preds, zero_division=0)
    recall = recall_score(image_labels, image_preds, zero_division=0)
    f1 = f1_score(image_labels, image_preds, zero_division=0)

    # 混淆矩阵
    cm = confusion_matrix(image_labels, image_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # 计算特异性 (Specificity)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 整合所有指标
    detailed_metrics = {
        # 基础指标
        'I-AUROC': base_metrics['I-AUROC'],
        'I-AU-PR': base_metrics['I-AU-PR'],
        'I-F1max': base_metrics['I-F1max'],

        # 分类指标 (基于最佳阈值)
        'Best_Threshold': best_threshold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity,

        # 混淆矩阵
        'True_Negative': int(tn),
        'False_Positive': int(fp),
        'False_Negative': int(fn),
        'True_Positive': int(tp),
    }

    # 如果有像素级指标
    if 'P-AUROC' in base_metrics:
        detailed_metrics['P-AUROC'] = base_metrics['P-AUROC']
        detailed_metrics['PRO'] = base_metrics['PRO']

    return detailed_metrics


def print_metrics(metrics):
    """格式化打印指标。"""
    print("\n" + "=" * 70)
    print(" " * 20 + "Stage1 Model Evaluation Results")
    print("=" * 70)

    print("\n📊 Image-Level Metrics (Threshold-Free)")
    print("-" * 70)
    print(f"  AUROC (Area Under ROC)        : {metrics['I-AUROC']:.4f}")
    print(f"  AU-PR (Area Under PR Curve)   : {metrics['I-AU-PR']:.4f}")
    print(f"  F1-max (Maximum F1 Score)     : {metrics['I-F1max']:.4f}")

    print("\n🎯 Classification Metrics (Threshold = {:.4f})".format(metrics['Best_Threshold']))
    print("-" * 70)
    print(f"  Accuracy                      : {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
    print(f"  Precision                     : {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)")
    print(f"  Recall (Sensitivity)          : {metrics['Recall']:.4f} ({metrics['Recall']*100:.2f}%)")
    print(f"  F1-Score                      : {metrics['F1-Score']:.4f}")
    print(f"  Specificity                   : {metrics['Specificity']:.4f} ({metrics['Specificity']*100:.2f}%)")

    print("\n📋 Confusion Matrix")
    print("-" * 70)
    print(f"                    Predicted Normal    Predicted Anomaly")
    print(f"  Actual Normal     {metrics['True_Negative']:>8d}            {metrics['False_Positive']:>8d}")
    print(f"  Actual Anomaly    {metrics['False_Negative']:>8d}            {metrics['True_Positive']:>8d}")

    if 'P-AUROC' in metrics:
        print("\n🔍 Pixel-Level Metrics")
        print("-" * 70)
        print(f"  P-AUROC (Pixel-level AUROC)   : {metrics['P-AUROC']:.4f}")
        print(f"  PRO (Per-Region Overlap)      : {metrics['PRO']:.4f}")

    print("\n" + "=" * 70)

    # 性能评估
    print("\n💡 Performance Assessment:")
    auroc = metrics['I-AUROC']
    if auroc >= 0.95:
        print("  ✅ Excellent performance (AUROC ≥ 0.95)")
    elif auroc >= 0.90:
        print("  ✅ Good performance (AUROC ≥ 0.90)")
    elif auroc >= 0.85:
        print("  ⚠️  Acceptable performance (AUROC ≥ 0.85)")
    else:
        print("  ❌ Poor performance (AUROC < 0.85)")

    print("\n")


def save_metrics_to_file(metrics, output_path):
    """保存指标到文件。"""
    with open(output_path, 'w') as f:
        f.write("Stage1 Model Evaluation Results\n")
        f.write("=" * 70 + "\n\n")

        f.write("Image-Level Metrics (Threshold-Free)\n")
        f.write("-" * 70 + "\n")
        f.write(f"AUROC: {metrics['I-AUROC']:.4f}\n")
        f.write(f"AU-PR: {metrics['I-AU-PR']:.4f}\n")
        f.write(f"F1-max: {metrics['I-F1max']:.4f}\n\n")

        f.write(f"Classification Metrics (Threshold = {metrics['Best_Threshold']:.4f})\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['Precision']:.4f}\n")
        f.write(f"Recall: {metrics['Recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['F1-Score']:.4f}\n")
        f.write(f"Specificity: {metrics['Specificity']:.4f}\n\n")

        f.write("Confusion Matrix\n")
        f.write("-" * 70 + "\n")
        f.write(f"True Negative: {metrics['True_Negative']}\n")
        f.write(f"False Positive: {metrics['False_Positive']}\n")
        f.write(f"False Negative: {metrics['False_Negative']}\n")
        f.write(f"True Positive: {metrics['True_Positive']}\n\n")

        if 'P-AUROC' in metrics:
            f.write("Pixel-Level Metrics\n")
            f.write("-" * 70 + "\n")
            f.write(f"P-AUROC: {metrics['P-AUROC']:.4f}\n")
            f.write(f"PRO: {metrics['PRO']:.4f}\n\n")

    print(f"✅ Metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Stage1 Model with Detailed Metrics')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/stage1_best.pth',
                        help='Path to Stage1 checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default='logs/stage1_test_results.txt',
                        help='Path to save results')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 构建数据集
    test_dataset = build_datasets(config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.get('data', {}).get('num_workers', 4),
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    # 构建模型
    model_config = config.get('model', {})
    topo_config = TopoVarADConfig(
        d_model=model_config.get('d_model', 256),
        n_tpm_layers=model_config.get('n_layers', 6),
        n_heads=model_config.get('n_heads', 8),
        superpixel_scales=tuple(model_config.get('superpixel_scales', [50, 100, 200])),
        rqvae_codebook_size=model_config.get('rqvae_codebook_size', 1024),
        rqvae_d_code=model_config.get('rqvae_d_code', 32),
        rqvae_n_layers=model_config.get('rqvae_n_layers', 8),
        tar_n_layers=model_config.get('tar_n_layers', 6),
        tar_n_heads=model_config.get('tar_n_heads', 8),
    )
    model = topo_config.build_model().to(device)

    # 加载checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Checkpoint loaded (trained for {ckpt['epoch']+1} epochs)")

    # 设置为Stage1模式
    model.set_stage(1)

    # 评估
    metrics = evaluate_detailed(model, test_loader, device)

    # 打印结果
    print_metrics(metrics)

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_metrics_to_file(metrics, args.output)


if __name__ == '__main__':
    main()