"""
Train Stage 1 on NORMAL-ONLY samples for proper anomaly detection.

Standard anomaly detection principle: the model must only see normal data
during training, so reconstruction error naturally becomes an anomaly signal.

Usage:
  python train_stage1_normal.py --config configs/default.yaml --device cuda
"""

import os
import sys
import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from models.topovarad import TopoVarAD, TopoVarADConfig
from data.dataset import CSVDataset
from utils.losses import TopoVarADLoss
from utils.metrics import MetricsCalculator
from utils.logger import TrainingLogger


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_normal_only_dataset(config):
    """Build a dataset containing ONLY defect-free (normal) training samples."""
    data_config = config.get('data', {})
    train_config = config.get('train', {})

    # Load full train dataset
    full_dataset = CSVDataset(
        csv_path=data_config.get('train_csv', 'data/train.csv'),
        images_dir=data_config.get('images_dir', 'data/images'),
        split='train',
        image_size=data_config.get('image_size', 512),
        augment=train_config.get('augment', True),
    )

    # Filter to normal-only
    normal_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label == 0]
    normal_dataset = torch.utils.data.Subset(full_dataset, normal_indices)

    return normal_dataset, len(normal_indices)


def build_optimizer(model, lr, weight_decay):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, total_epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'normal_only': True,  # Mark as normal-only trained
    }, path)
    print(f"  Checkpoint saved: {path}")


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_pixel = 0.0
    total_lpips = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f'[Stage1-Normal] Epoch {epoch}', leave=False)
    for batch in pbar:
        images = batch['image'].to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(images)
            loss_dict = criterion(outputs, stage=1)
            loss = loss_dict['loss_total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_pixel += loss_dict['loss_pixel'].item()
        total_lpips += loss_dict['loss_lpips'].item()
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pixel': f'{loss_dict["loss_pixel"].item():.4f}',
        })

    return {
        'loss': total_loss / max(n_batches, 1),
        'loss_pixel': total_pixel / max(n_batches, 1),
        'loss_lpips': total_lpips / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_reconstruction(model, loader, device):
    """Evaluate Stage 1 via reconstruction error (normal-only training)."""
    model.eval()
    all_scores = []
    all_labels = []

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        images = batch['image'].to(device)
        labels = batch['label']

        outputs = model(images)
        x_recon = outputs['reconstructed']
        x_resized = outputs['x_resized']
        # Per-sample reconstruction error
        error = torch.abs(x_recon - x_resized).reshape(images.shape[0], -1).mean(dim=1)

        all_scores.append(error.cpu())
        all_labels.append(labels)

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    from utils.metrics import compute_auroc, compute_f1_max
    auroc = compute_auroc(scores, labels)
    f1max, _ = compute_f1_max(scores, labels)

    return auroc, f1max, scores, labels


def main():
    parser = argparse.ArgumentParser(description='Train Stage 1 on Normal-Only Samples')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_config = config.get('train', {})
    data_config = config.get('data', {})
    model_config = config.get('model', {})

    # Build normal-only training dataset
    train_dataset, n_normal = build_normal_only_dataset(config)
    print(f"Normal-only training samples: {n_normal}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 16),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    # Build full test dataset for evaluation
    test_dataset = CSVDataset(
        csv_path=data_config.get('test_csv', 'data/test.csv'),
        images_dir=data_config.get('images_dir', 'data/images'),
        split='test',
        image_size=data_config.get('image_size', 512),
        augment=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=data_config.get('num_workers', 4),
                             pin_memory=True)
    print(f"Test samples: {len(test_dataset)}")

    # Build model
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
    total_params = sum(p.numel() for p in model.parameters())
    model.set_stage(1)

    # Loss
    criterion = TopoVarADLoss(
        lambda_lpips=train_config.get('lambda_lpips', 0.1),
        lambda_rqvae=train_config.get('lambda_rqvae', 0.5),
        lambda_ar=train_config.get('lambda_ar', 1.0),
        lambda_diversity=train_config.get('lambda_diversity', 0.0),
        label_smoothing=train_config.get('label_smoothing', 0.1),
    ).to(device)

    epochs = args.epochs or train_config.get('stage1_epochs', 100)
    lr = train_config.get('lr_stage1', 1e-4)

    optimizer = build_optimizer(model, lr, train_config.get('weight_decay', 0.05))
    scheduler = build_scheduler(optimizer, epochs, train_config.get('warmup_epochs', 10))
    scaler = GradScaler(device='cuda' if device.type == 'cuda' else 'cpu')

    ckpt_dir = train_config.get('checkpoint_dir', 'checkpoints')
    log_dir = train_config.get('log_dir', 'logs')
    logger = TrainingLogger(log_dir=log_dir, stage=1)

    os.makedirs(ckpt_dir, exist_ok=True)
    best_auroc = 0.0

    print(f"\n{'='*60}")
    print(f"Stage 1 Normal-Only Training")
    print(f"  Training samples: {n_normal} (all defect-free)")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"  Model parameters: {total_params:,}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch)
        scheduler.step()

        lr_current = optimizer.param_groups[0]['lr']

        # Evaluate every 10 epochs
        eval_msg = ""
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            auroc, f1max, _, _ = evaluate_reconstruction(model, test_loader, device)
            eval_msg = f" | I-AUROC={auroc:.4f} I-F1max={f1max:.4f}"
            print(f"  [Eval] I-AUROC={auroc:.4f}  I-F1max={f1max:.4f}")

            if auroc > best_auroc:
                best_auroc = auroc
                save_checkpoint(model, optimizer, scheduler, epoch, train_metrics['loss'],
                                os.path.join(ckpt_dir, 'stage1_normal_best.pth'))

        elapsed = 0  # approximate
        logger.log_epoch(epoch, train_metrics,
                        {'I-AUROC': auroc, 'I-F1max': f1max} if eval_msg else None,
                        lr_current, elapsed)

        print(f"  [Epoch {epoch}] loss={train_metrics['loss']:.4f} "
              f"pixel={train_metrics['loss_pixel']:.4f} lr={lr_current:.2e}{eval_msg}")

    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, epochs - 1, train_metrics['loss'],
                    os.path.join(ckpt_dir, 'stage1_normal_final.pth'))

    logger.log_message(f"Stage 1 (normal-only) finished. Best I-AUROC: {best_auroc:.4f}")
    logger.close()
    print(f"\nBest I-AUROC: {best_auroc:.4f}")


if __name__ == '__main__':
    main()