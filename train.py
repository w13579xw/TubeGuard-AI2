import os
import sys
import time
import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.topovarad import TopoVarAD, TopoVarADConfig
from data.dataset import MVTecDataset
from utils.losses import TopoVarADLoss
from utils.metrics import MetricsCalculator


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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
    }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt['loss']


def train_one_epoch_stage1(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_pixel = 0.0
    total_lpips = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f'[Stage1] Epoch {epoch}', leave=False)
    for batch in pbar:
        images = batch['image'].to(device)

        optimizer.zero_grad()

        with autocast():
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


def train_one_epoch_stage2(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_pixel = 0.0
    total_rqvae = 0.0
    total_ar = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f'[Stage2] Epoch {epoch}', leave=False)
    for batch in pbar:
        images = batch['image'].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss_dict = criterion(outputs, stage=2)
            loss = loss_dict['loss_total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_pixel += loss_dict['loss_pixel'].item()
        total_rqvae += loss_dict['loss_rqvae'].item()
        total_ar += loss_dict['loss_ar'].item()
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ar': f'{loss_dict["loss_ar"].item():.4f}',
        })

    return {
        'loss': total_loss / max(n_batches, 1),
        'loss_pixel': total_pixel / max(n_batches, 1),
        'loss_rqvae': total_rqvae / max(n_batches, 1),
        'loss_ar': total_ar / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics = MetricsCalculator()

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        images = batch['image'].to(device)
        labels = batch['label']
        masks = batch['mask']

        image_scores, pixel_scores = model.predict(images)

        metrics.update(
            image_scores.cpu().numpy(),
            labels.numpy(),
            pixel_scores.cpu().numpy() if masks.sum() > 0 else None,
            masks.numpy() if masks.sum() > 0 else None,
        )

    return metrics.compute()


def main():
    parser = argparse.ArgumentParser(description='TopoVarAD Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_config = config.get('train', {})
    data_config = config.get('data', {})
    model_config = config.get('model', {})

    train_dataset = MVTecDataset(
        root=data_config.get('dataset_path', 'data/mvtec'),
        category=args.category,
        split='train',
        image_size=data_config.get('image_size', 512),
    )
    test_dataset = MVTecDataset(
        root=data_config.get('dataset_path', 'data/mvtec'),
        category=args.category,
        split='test',
        image_size=data_config.get('image_size', 512),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 4),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

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
    print(f"Model parameters: {total_params:,}")

    criterion = TopoVarADLoss(
        lambda_lpips=train_config.get('lambda_lpips', 0.1),
        lambda_rqvae=train_config.get('lambda_rqvae', 0.5),
        lambda_ar=train_config.get('lambda_ar', 1.0),
    )

    stage = args.stage
    model.set_stage(stage)

    if stage == 1:
        epochs = train_config.get('stage1_epochs', 200)
        lr = train_config.get('lr_stage1', 1e-4)
    else:
        epochs = train_config.get('stage2_epochs', 300)
        lr = train_config.get('lr_stage2', 5e-5)

    optimizer = build_optimizer(model, lr, train_config.get('weight_decay', 0.05))
    scheduler = build_scheduler(optimizer, epochs, train_config.get('warmup_epochs', 10))
    scaler = GradScaler()

    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}")

    ckpt_dir = train_config.get('checkpoint_dir', 'checkpoints')
    best_auroc = 0.0

    print(f"\n{'='*60}")
    print(f"Starting Stage {stage} Training")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Batch Size: {train_config.get('batch_size', 4)}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        if stage == 1:
            train_metrics = train_one_epoch_stage1(
                model, train_loader, optimizer, criterion, scaler, device, epoch
            )
        else:
            train_metrics = train_one_epoch_stage2(
                model, train_loader, optimizer, criterion, scaler, device, epoch
            )

        scheduler.step()

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs} [{elapsed:.1f}s] "
              f"lr={lr_current:.2e} loss={train_metrics['loss']:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            eval_metrics = evaluate(model, test_loader, device)
            print(f"  Eval: ", end="")
            for k, v in eval_metrics.items():
                print(f"{k}={v:.4f} ", end="")
            print()

            if eval_metrics.get('I-AUROC', 0) > best_auroc:
                best_auroc = eval_metrics['I-AUROC']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, train_metrics['loss'],
                    os.path.join(ckpt_dir, f'stage{stage}_best.pth')
                )
                print(f"  → Best model saved (I-AUROC={best_auroc:.4f})")

        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_metrics['loss'],
                os.path.join(ckpt_dir, f'stage{stage}_epoch{epoch+1}.pth')
            )

    print(f"\nStage {stage} training finished. Best I-AUROC: {best_auroc:.4f}")

    save_checkpoint(
        model, optimizer, scheduler, epochs - 1, train_metrics['loss'],
        os.path.join(ckpt_dir, f'stage{stage}_final.pth')
    )


if __name__ == '__main__':
    main()
