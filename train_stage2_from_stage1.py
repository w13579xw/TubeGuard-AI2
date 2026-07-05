import os
import sys
import time
import argparse
import yaml
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler, autocast
    AMP_BACKEND = 'torch.amp'
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AMP_BACKEND = 'torch.cuda.amp'
from tqdm import tqdm

from models.topovarad import TopoVarAD, TopoVarADConfig
from data.dataset import CSVDataset, MVTecDataset
from utils.losses import TopoVarADLoss
from utils.metrics import MetricsCalculator
from utils.logger import TrainingLogger


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_optimizer(model, lr, weight_decay):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


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


def load_stage1_checkpoint(model, path):
    """加载Stage1的模型权重，只加载模型参数。
    strict=False 以兼容新增的码本buffer（如 initialized），这些buffer会用默认值或随后由K-means初始化覆盖。
    """
    ckpt = torch.load(path, map_location='cpu')
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  [load] missing keys (use defaults): {missing}")
    if unexpected:
        print(f"  [load] unexpected keys (ignored): {unexpected}")
    print(f"Loaded Stage1 checkpoint from {path}")
    print(f"  Stage1 ended at epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f}")
    return ckpt['epoch']


STAGE1_MODULE_NAMES = [
    'input_proj', 'tokenizer', 'patch_embed', 'learned_pe',
    'tpm', 'pool_head', 'pixel_head'
]


def _set_stage1_requires_grad(model, requires_grad):
    n_params = 0
    for name in STAGE1_MODULE_NAMES:
        obj = getattr(model, name, None)
        if obj is None:
            continue
        if isinstance(obj, nn.Parameter):
            obj.requires_grad_(requires_grad)
            n_params += obj.numel()
        else:
            for p in obj.parameters():
                p.requires_grad_(requires_grad)
                n_params += p.numel()
    return n_params


def freeze_stage1_modules(model):
    return _set_stage1_requires_grad(model, False)


def unfreeze_stage1_modules(model):
    return _set_stage1_requires_grad(model, True)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_tar_requires_grad(model, requires_grad):
    n = 0
    if getattr(model, 'tar', None) is not None:
        for p in model.tar.parameters():
            p.requires_grad_(requires_grad)
            n += p.numel()
    return n


def make_grad_scaler(device):
    if AMP_BACKEND == 'torch.amp':
        return GradScaler(device='cuda' if device.type == 'cuda' else 'cpu')
    return GradScaler(enabled=(device.type == 'cuda'))


def autocast_context(device):
    if AMP_BACKEND == 'torch.amp':
        return autocast(device_type='cuda' if device.type == 'cuda' else 'cpu')
    return autocast(enabled=(device.type == 'cuda'))


def train_one_epoch_stage2(model, loader, optimizer, criterion, scaler, device, epoch,
                           contrastive_margin=1.0, lambda_contrastive=1.0):
    model.train()
    total_loss = 0.0
    total_pixel = 0.0
    total_rqvae = 0.0
    total_ar = 0.0
    total_diversity = 0.0
    total_contrastive = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f'[Stage2] Epoch {epoch}', leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device).float()  # (B,) 0=normal, 1=defect

        optimizer.zero_grad()

        with autocast_context(device):
            outputs = model(images)
            loss_dict = criterion(outputs, stage=2)
            loss = loss_dict['loss_total']

            # ========== 新增：弱监督对比学习 ==========
            # 强制正常样本的 rqvae_dist 小、异常样本的 rqvae_dist 大
            # 这样 codes 才会真正区分正常 vs 异常
            z_global = outputs['z_global']  # (B, D)
            # 重新过一次 RQ-VAE 拿 z_hat（因为 outputs['codes'] 是索引，不是量化重建）
            z_hat, codes, _, _ = model.rqvae(z_global)
            rqvae_dist = torch.nn.functional.mse_loss(z_hat, z_global, reduction='none').mean(dim=-1)  # (B,)

            # 弱监督：正常样本 rqvae_dist 应 < margin，异常样本 rqvae_dist 应 > margin
            # 使用 Margin Ranking Loss 思想：
            #   normal_dist 应尽量小（推向 0）
            #   defect_dist 应尽量大（推向 margin）
            normal_mask = (labels == 0)
            defect_mask = (labels == 1)

            loss_contrastive = torch.tensor(0.0, device=device)
            if normal_mask.sum() > 0 and defect_mask.sum() > 0:
                normal_dist = rqvae_dist[normal_mask].mean()
                defect_dist = rqvae_dist[defect_mask].mean()
                # 目标：defect_dist - normal_dist >= margin
                loss_contrastive = torch.clamp(contrastive_margin - (defect_dist - normal_dist), min=0.0)
            elif normal_mask.sum() > 0:
                # 只有正常样本，直接压低
                loss_contrastive = rqvae_dist[normal_mask].mean()

            loss = loss + lambda_contrastive * loss_contrastive

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_pixel += loss_dict['loss_pixel'].item()
        total_rqvae += loss_dict['loss_rqvae'].item()
        total_ar += loss_dict['loss_ar'].item()
        total_diversity += loss_dict.get('loss_diversity', torch.tensor(0.0)).item()
        total_contrastive += loss_contrastive.item() if isinstance(loss_contrastive, torch.Tensor) else 0.0
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ar': f'{loss_dict["loss_ar"].item():.4f}',
            'contra': f'{loss_contrastive.item() if isinstance(loss_contrastive, torch.Tensor) else 0:.4f}',
        })

    return {
        'loss': total_loss / max(n_batches, 1),
        'loss_pixel': total_pixel / max(n_batches, 1),
        'loss_rqvae': total_rqvae / max(n_batches, 1),
        'loss_ar': total_ar / max(n_batches, 1),
        'loss_diversity': total_diversity / max(n_batches, 1),
        'loss_contrastive': total_contrastive / max(n_batches, 1),
    }


@torch.no_grad()
def monitor_codebook(model, loader, device):
    """
    监控码本使用情况和分布熵
    """
    model.eval()
    all_codes = []

    for batch in tqdm(loader, desc='Monitoring codebook', leave=False):
        images = batch['image'].to(device)
        outputs = model(images)
        if 'codes' in outputs:
            all_codes.append(outputs['codes'].cpu())

    if len(all_codes) == 0:
        return None

    all_codes = torch.cat(all_codes, dim=0)
    B, D = all_codes.shape
    n_codes = 1024

    stats = {
        'usage': [],
        'entropy': [],
        'active_ratio': []
    }

    for d in range(D):
        layer_codes = all_codes[:, d]
        hist = torch.histc(layer_codes.float(), bins=n_codes, min=0, max=n_codes-1)

        # 使用率
        active_codes = (hist > 0).sum().item()
        active_ratio = active_codes / n_codes
        stats['active_ratio'].append(active_ratio)

        # 熵
        prob = hist / (hist.sum() + 1e-10)
        entropy = -(prob * torch.log(prob + 1e-10)).sum().item()
        max_entropy = math.log(n_codes)
        stats['entropy'].append(entropy / max_entropy)

        # 码本使用率（从模型获取）
        usage = model.rqvae.rq.get_codebook_usage()
        stats['usage'] = usage

    return stats


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


def build_datasets(config):
    """根据配置构建训练和测试数据集。"""
    data_config = config.get('data', {})
    train_config = config.get('train', {})
    dataset_type = data_config.get('dataset_type', 'csv')

    if dataset_type == 'mvtec':
        train_dataset = MVTecDataset(
            root=data_config.get('dataset_path', 'data/mvtec'),
            category=data_config.get('category', 'bottle'),
            split='train',
            image_size=data_config.get('image_size', 512),
        )
        test_dataset = MVTecDataset(
            root=data_config.get('dataset_path', 'data/mvtec'),
            category=data_config.get('category', 'bottle'),
            split='test',
            image_size=data_config.get('image_size', 512),
        )
    else:
        train_dataset = CSVDataset(
            csv_path=data_config.get('train_csv', 'data/train.csv'),
            images_dir=data_config.get('images_dir', 'data/images'),
            split='train',
            image_size=data_config.get('image_size', 512),
            augment=train_config.get('augment', True),
        )
        test_dataset = CSVDataset(
            csv_path=data_config.get('test_csv', 'data/test.csv'),
            images_dir=data_config.get('images_dir', 'data/images'),
            split='test',
            image_size=data_config.get('image_size', 512),
            augment=False,
        )

    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description='TopoVarAD Stage2 Training from Stage1')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                        help='Path to Stage1 checkpoint (e.g., checkpoints/stage1_best.pth)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--freeze_stage1_epochs', type=int, default=None,
                        help='Freeze Stage1 modules for N Stage2 epochs; 0=always frozen, -1=never freeze')
    parser.add_argument('--kmeans_init', action='store_true', default=None,
                        help='K-means init RQ-VAE codebook from Stage1 features before training')
    parser.add_argument('--no_kmeans_init', dest='kmeans_init', action='store_false',
                        help='Disable K-means codebook init')
    parser.add_argument('--tar_warmup_epochs', type=int, default=None,
                        help='Freeze TAR head for the first N epochs, training only RQ-VAE')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_config = config.get('train', {})
    data_config = config.get('data', {})
    model_config = config.get('model', {})

    train_dataset, test_dataset = build_datasets(config)

    use_sampler = train_config.get('use_sampler', True)
    sampler = None
    shuffle = True
    if use_sampler and hasattr(train_dataset, 'get_sampler'):
        sampler = train_dataset.get_sampler()
        shuffle = False
        n_normal, n_defect = train_dataset.get_class_counts()
        print(f"Using WeightedRandomSampler (Normal: {n_normal}, Defect: {n_defect})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 4),
        shuffle=shuffle,
        sampler=sampler,
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

    # 加载Stage1的权重
    stage1_epoch = load_stage1_checkpoint(model, args.stage1_checkpoint)

    # 切换到Stage2模式
    model.set_stage(2)
    print("Switched to Stage 2 mode")

    freeze_stage1_epochs = args.freeze_stage1_epochs
    if freeze_stage1_epochs is None:
        freeze_stage1_epochs = train_config.get('freeze_stage1_epochs', 10)
    frozen_params = 0
    if freeze_stage1_epochs >= 0:
        frozen_params = freeze_stage1_modules(model)
        print(f"Frozen Stage1 modules: {frozen_params:,} params")
        print(f"Trainable params after freeze: {count_trainable_params(model):,}")
    else:
        print("Stage1 freezing disabled")

    # ---- 码本K-means初始化（缓解codebook collapse）----
    kmeans_init = args.kmeans_init
    if kmeans_init is None:
        kmeans_init = train_config.get('kmeans_init', True)
    if kmeans_init:
        n_feat = model.init_codebook_from_loader(
            train_loader, device,
            max_batches=train_config.get('kmeans_init_batches', 50),
            n_iter=train_config.get('kmeans_iter', 10),
        )
        print(f"K-means codebook init done from {n_feat} Stage1 features")
        print(f"  Codebook usage after init: {[f'{u:.2%}' for u in model.rqvae.rq.get_codebook_usage()]}")

    # ---- TAR warmup：前N个epoch冻结TAR，只训练RQ-VAE让码本稳定 ----
    tar_warmup_epochs = args.tar_warmup_epochs
    if tar_warmup_epochs is None:
        tar_warmup_epochs = train_config.get('tar_warmup_epochs', 5)
    if tar_warmup_epochs > 0:
        frozen_tar = set_tar_requires_grad(model, False)
        print(f"TAR warmup: froze TAR head ({frozen_tar:,} params) for first {tar_warmup_epochs} epochs")

    criterion = TopoVarADLoss(
        lambda_lpips=train_config.get('lambda_lpips', 0.1),
        lambda_rqvae=train_config.get('lambda_rqvae', 0.5),
        lambda_ar=train_config.get('lambda_ar', 1.0),
        lambda_diversity=train_config.get('lambda_diversity', 0.0),
        label_smoothing=train_config.get('label_smoothing', 0.1),
    ).to(device)

    epochs = train_config.get('stage2_epochs', 300)
    lr = train_config.get('lr_stage2', 1e-5)

    optimizer = build_optimizer(model, lr, train_config.get('weight_decay', 0.05))
    scheduler = build_scheduler(optimizer, epochs, train_config.get('warmup_epochs', 10))
    scaler = make_grad_scaler(device)

    ckpt_dir = train_config.get('checkpoint_dir', 'checkpoints')
    best_auroc = 0.0

    log_dir = train_config.get('log_dir', 'logs')
    logger = TrainingLogger(log_dir=log_dir, stage=2)
    logger.log_message(f"Starting Stage 2 Training (from Stage1 checkpoint)")
    logger.log_message(f"  Stage1 checkpoint: {args.stage1_checkpoint}")
    logger.log_message(f"  Stage1 ended at epoch: {stage1_epoch}")
    logger.log_message(f"  Epochs: {epochs}, LR: {lr}, Batch Size: {train_config.get('batch_size', 4)}")
    logger.log_message(f"  Early stopping: disabled (train full {epochs} epochs), Device: {device}")
    logger.log_message(f"  Model parameters: {total_params:,}")
    logger.log_message(f"  Freeze Stage1 epochs: {freeze_stage1_epochs}")
    logger.log_message(f"  Frozen Stage1 params: {frozen_params:,}")
    logger.log_message(f"  Trainable params: {count_trainable_params(model):,}")
    logger.log_message(f"  AMP backend: {AMP_BACKEND}")
    logger.log_message(f"  Gradient clipping: 0.5 (stricter than Stage1)")

    print(f"\n{'='*60}")
    print(f"Starting Stage 2 Training")
    print(f"  Loaded from Stage1: {args.stage1_checkpoint}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr} (reduced from 5e-5)")
    print(f"  Batch Size: {train_config.get('batch_size', 4)}")
    print(f"  Early Stopping: disabled (full {epochs} epochs)")
    print(f"  Freeze Stage1 epochs: {freeze_stage1_epochs}")
    print(f"  Trainable Params: {count_trainable_params(model):,}")
    print(f"  AMP Backend: {AMP_BACKEND}")
    print(f"  Gradient Clipping: 0.5 (stricter)")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        t0 = time.time()

        if tar_warmup_epochs > 0 and epoch == tar_warmup_epochs:
            unfrozen_tar = set_tar_requires_grad(model, True)
            optimizer = build_optimizer(model, optimizer.param_groups[0]['lr'],
                                        train_config.get('weight_decay', 0.05))
            msg = (f"TAR warmup ended at epoch {epoch}: unfroze TAR ({unfrozen_tar:,} params), "
                   f"trainable={count_trainable_params(model):,}")
            logger.log_message(msg)
            print(msg)

        # ✅ 改进：全程冻结 Stage1 模块，不做解冻
        # 这样可以 100% 保证 Stage2 不会破坏 Stage1 学到的优秀特征
        # 只训练 RQVAE 和 TAR head，做增量改进
        if freeze_stage1_epochs > 0 and epoch == freeze_stage1_epochs:
            msg = (f"✅ Stage1 modules REMAIN FROZEN at epoch {epoch}: "
                   f"Stage1 特征全程保留，只训练 RQVAE + TAR")
            logger.log_message(msg)
            print(msg)

        train_metrics = train_one_epoch_stage2(
            model, train_loader, optimizer, criterion, scaler, device, epoch,
            contrastive_margin=train_config.get('contrastive_margin', 1.0),
            lambda_contrastive=train_config.get('lambda_contrastive', 1.0),
        )

        scheduler.step()

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]['lr']

        eval_metrics = None
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            eval_metrics = evaluate(model, test_loader, device)
            print(f"  Eval: ", end="")
            for k, v in eval_metrics.items():
                print(f"{k}={v:.4f} ", end="")
            print()

            # 监控码本状态（每10个epoch）
            codebook_stats = monitor_codebook(model, train_loader, device)
            if codebook_stats:
                print(f"  Codebook usage: {[f'{u:.2%}' for u in codebook_stats['usage']]}")
                print(f"  Codebook entropy: {[f'{e:.3f}' for e in codebook_stats['entropy']]}")
                print(f"  Active ratio: {[f'{r:.2%}' for r in codebook_stats['active_ratio']]}")

            if eval_metrics.get('I-AUROC', 0) > best_auroc:
                best_auroc = eval_metrics['I-AUROC']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, train_metrics['loss'],
                    os.path.join(ckpt_dir, 'stage2_best.pth')
                )
                print(f"  -> Best model saved (I-AUROC={best_auroc:.4f})")

        logger.log_epoch(epoch, train_metrics, eval_metrics, lr_current, elapsed)

        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_metrics['loss'],
                os.path.join(ckpt_dir, f'stage2_epoch{epoch+1}.pth')
            )

    logger.log_message(f"Stage 2 training finished. Best I-AUROC: {best_auroc:.4f}")
    print(f"\nStage 2 training finished. Best I-AUROC: {best_auroc:.4f}")

    logger.plot_training_curves()
    logger.close()

    save_checkpoint(
        model, optimizer, scheduler, epochs - 1, train_metrics['loss'],
        os.path.join(ckpt_dir, 'stage2_final.pth')
    )


if __name__ == '__main__':
    main()