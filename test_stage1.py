import os
import argparse
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.topovarad import TopoVarAD, TopoVarADConfig
from data.dataset import CSVDataset, MVTecDataset
from utils.metrics import compute_auroc, compute_auprc, compute_f1_max


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_datasets(config):
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
def evaluate_stage1_reconstruction(model, loader, device):
    """
    Stage1 正确评估方式：用重建误差作为异常分数。

    流程:
      图像 x → T2M-Tokenizer → TPM Block ×L层 → GlobalPooling → PixelHead → 重建图 x_recon
      异常分数 = |x - x_recon|  (逐像素L1误差的均值)

    原理:
      Stage1 训练目标 = 最小化正常样本的重建误差
      → 正常样本：低重建误差（已学会重建）
      → 异常样本：高重建误差（未见过，重建失败）

    这与 predict() 不同——predict() 走 RQ-VAE + TAR Head 路径，
    在 Stage1 中 RQ-VAE 和 TAR Head 都是随机初始化的，不可用于评估。
    """
    model.eval()

    all_image_scores = []
    all_image_labels = []
    all_pixel_score_maps = []  # 存储每张图的像素级异常图
    all_pixel_label_maps = []  # 存储每张图的像素级GT mask

    print("Evaluating Stage1 via reconstruction error...")
    for batch in tqdm(loader, desc='Testing'):
        images = batch['image'].to(device)       # (1, 3, H, W)
        labels = batch['label']                    # (1,)  0=正常, 1=异常
        masks = batch['mask']                      # (1, H, W) 像素级GT

        # Stage1 前向：走重建路径，不使用 RQ-VAE / TAR Head
        outputs = model(images)  # model.stage=1 → forward_stage1()

        x_recon = outputs['reconstructed']   # (1, 3, H_recon, W_recon)
        x_resized = outputs['x_resized']     # (1, 3, H_recon, W_recon)

        # --- 图像级异常分数 ---
        # 逐像素 L1 误差 → 取所有像素和通道的均值 → 标量
        recon_error = F.l1_loss(x_recon, x_resized, reduction='none')  # (1, 3, H, W)
        image_score = recon_error.mean().item()  # 标量

        all_image_scores.append(image_score)
        all_image_labels.append(labels.item())

        # --- 像素级异常分数 ---
        # 逐像素 L1 误差 → 对通道取均值 → (H, W)
        pixel_error = recon_error.mean(dim=1).squeeze(0)  # (H_recon, W_recon)

        # 上采样回原始图像尺寸
        _, _, H_orig, W_orig = images.shape
        pixel_score_map = F.interpolate(
            pixel_error.unsqueeze(0).unsqueeze(0),   # (1, 1, H_recon, W_recon)
            size=(H_orig, W_orig),
            mode='bilinear',
            align_corners=False,
        ).squeeze()  # (H_orig, W_orig)

        all_pixel_score_maps.append(pixel_score_map.cpu().numpy())
        all_pixel_label_maps.append(masks.squeeze().cpu().numpy())

    # --- 合并为 numpy 数组 ---
    image_scores = np.array(all_image_scores)
    image_labels = np.array(all_image_labels)

    # --- 计算指标 ---
    auroc = compute_auroc(image_scores, image_labels)
    aupr = compute_auprc(image_scores, image_labels)
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
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 像素级 AUROC
    pixel_scores_flat = np.concatenate([p.flatten() for p in all_pixel_score_maps])
    pixel_labels_flat = np.concatenate([m.flatten() for m in all_pixel_label_maps])
    p_auroc = compute_auroc(pixel_scores_flat, pixel_labels_flat)

    metrics = {
        'I-AUROC': auroc,
        'I-AU-PR': aupr,
        'I-F1max': f1_max,
        'Best_Threshold': best_threshold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity,
        'True_Negative': int(tn),
        'False_Positive': int(fp),
        'False_Negative': int(fn),
        'True_Positive': int(tp),
        'P-AUROC': p_auroc,
    }

    return metrics


def print_metrics(metrics):
    """格式化打印指标。"""
    print("\n" + "=" * 70)
    print(" " * 15 + "Stage1 Evaluation (Reconstruction Error)")
    print(" " * 15 + "异常分数 = |原图 - 重建图| 的 L1 误差")
    print("=" * 70)

    print("\n📊 Image-Level Metrics (Threshold-Free)")
    print("-" * 70)
    print(f"  AUROC (Area Under ROC)        : {metrics['I-AUROC']:.4f}")
    print(f"  AU-PR (Area Under PR Curve)   : {metrics['I-AU-PR']:.4f}")
    print(f"  F1-max (Maximum F1 Score)     : {metrics['I-F1max']:.4f}")

    print(f"\n🎯 Classification Metrics (Threshold = {metrics['Best_Threshold']:.6f})")
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

    print(f"\n🔍 Pixel-Level Metrics")
    print("-" * 70)
    print(f"  P-AUROC (Pixel-level AUROC)   : {metrics['P-AUROC']:.4f}")

    print("\n" + "=" * 70)
    print("\n💡 评估原理:")
    print("  Stage1 训练目标 = 最小化正常样本的重建误差")
    print("  → 正常样本: 低重建误差 | 异常样本: 高重建误差")
    print("  → 重建误差 直接作为 异常分数，无需 RQ-VAE/TAR Head")

    auroc = metrics['I-AUROC']
    if auroc >= 0.95:
        print("\n  ✅ Excellent performance (AUROC ≥ 0.95)")
    elif auroc >= 0.90:
        print("\n  ✅ Good performance (AUROC ≥ 0.90)")
    elif auroc >= 0.85:
        print("\n  ⚠️  Acceptable performance (AUROC ≥ 0.85)")
    else:
        print("\n  ❌ Poor performance (AUROC < 0.85)")
    print()


def save_metrics_to_file(metrics, output_path):
    """保存指标到文件。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("Stage1 Evaluation Results (Reconstruction Error)\n")
        f.write("Anomaly Score = L1 |Original - Reconstructed|\n")
        f.write("=" * 70 + "\n\n")

        f.write("Image-Level Metrics (Threshold-Free)\n")
        f.write("-" * 70 + "\n")
        f.write(f"AUROC: {metrics['I-AUROC']:.4f}\n")
        f.write(f"AU-PR: {metrics['I-AU-PR']:.4f}\n")
        f.write(f"F1-max: {metrics['I-F1max']:.4f}\n\n")

        f.write(f"Classification Metrics (Threshold = {metrics['Best_Threshold']:.6f})\n")
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

        f.write("Pixel-Level Metrics\n")
        f.write("-" * 70 + "\n")
        f.write(f"P-AUROC: {metrics['P-AUROC']:.4f}\n")

    print(f"✅ Metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Stage1 via Reconstruction Error')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/stage1_best.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='logs/stage1_test_results.txt')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_dataset = build_datasets(config)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=config.get('data', {}).get('num_workers', 4),
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

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

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Checkpoint loaded (trained for {ckpt['epoch']+1} epochs)")

    # Stage1 模式：forward() 走重建路径
    model.set_stage(1)

    metrics = evaluate_stage1_reconstruction(model, test_loader, device)
    print_metrics(metrics)
    save_metrics_to_file(metrics, args.output)


if __name__ == '__main__':
    main()