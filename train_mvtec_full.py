"""
Train and evaluate TopoVarAD (full architecture with RQ-VAE + TAR Head) on MVTec AD.

Pipeline per category:
  1. Stage 1: Reconstruction pre-training (T2M-Tokenizer + TPM + PixelHead)
  2. Stage 2: Joint training with RQ-VAE + TAR Head (frozen backbone + K-means codebook init)
  3. Evaluation: image-level + pixel-level metrics using
       - recon score (Stage 1 signal)
       - AR log-likelihood (Stage 2 signal)
       - rqvae quantization distance (Stage 2 signal)
       - fused score (recon + rqvae, normalized)

Output structure:
  logs/mvtec_full/{category}/
      stage1_best.pth
      stage2_best.pth
      stage1_metrics.json
      stage2_metrics.json
      train.log

Usage:
  python train_mvtec_full.py --root /path/to/mvtec --category bottle --device cuda
  python train_mvtec_full.py --root /path/to/mvtec --category all --device cuda
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import MVTecDataset
from models.topovarad import TopoVarADConfig
from utils.losses import TopoVarADLoss
from utils.metrics import compute_auprc, compute_auroc, compute_f1_max, compute_pro

try:
    from torch.amp import GradScaler, autocast

    def autocast_ctx(device):
        return autocast(device_type=device.type)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    def autocast_ctx(device):
        return autocast()


ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_loaders(root, category, image_size=256, batch_size=8, num_workers=4):
    """MVTec train (normal-only) and test loaders."""
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = MVTecDataset(root=root, category=category, split='train',
                                  image_size=image_size, transform=train_tf)
    test_dataset = MVTecDataset(root=root, category=category, split='test',
                                image_size=image_size, transform=test_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def build_model(config, device):
    mc = config.get('model', {})
    topo_cfg = TopoVarADConfig(
        d_model=mc.get('d_model', 256),
        n_tpm_layers=mc.get('n_layers', 6),
        n_heads=mc.get('n_heads', 8),
        superpixel_scales=tuple(mc.get('superpixel_scales', [50, 100, 200])),
        rqvae_codebook_size=mc.get('rqvae_codebook_size', 1024),
        rqvae_d_code=mc.get('rqvae_d_code', 32),
        rqvae_n_layers=mc.get('rqvae_n_layers', 8),
        tar_n_layers=mc.get('tar_n_layers', 6),
        tar_n_heads=mc.get('tar_n_heads', 8),
        use_slic=True, use_topo_attn=True, use_glpe=True,
    )
    return topo_cfg.build_model().to(device)


# ============================================================================
# Custom predict for MVTec: training on normal-only, so recon has NORMAL direction
#   (defective samples yield higher recon error)
#   plus fuses AR score and rqvae distance from Stage 2.
# ============================================================================
@torch.no_grad()
def predict_mvtec(model, x, use_stage2=False):
    """
    MVTec-specific inference:
      recon_score: mean per-pixel L1 (normal direction: defective > normal)
      rqvae_score: quantization residual (Stage 2 signal, if available)
      ar_score:    AR NLL (Stage 2 signal, if available)
    Returns:
      scores: dict of image-level scores per method
      pixel_map: (B, H, W) recon-based pixel anomaly map
    """
    model.eval()
    tokens, sp_labels, M, N = model._tokenize(x)
    refined = model.tpm(tokens, sp_labels, M, N)
    B, C, H, W = x.shape

    # ---- Recon score (Stage 1 & final): normal direction for MVTec ----
    x_recon = model.pixel_head(refined, M, N)
    H_target, W_target = M * 16, N * 16
    x_resized = F.interpolate(x, size=(H_target, W_target), mode='bilinear', align_corners=False)
    recon_error_small = F.l1_loss(x_recon, x_resized, reduction='none').mean(dim=1)
    recon_score = recon_error_small.mean(dim=[1, 2])

    pixel_map = F.interpolate(recon_error_small.unsqueeze(1),
                              size=(H, W), mode='bilinear', align_corners=False).squeeze(1)

    scores = {'recon': recon_score}

    if use_stage2:
        # ---- Stage 2 signals ----
        z_global = model.pool_head(refined)
        z_hat, codes, _, _ = model.rqvae(z_global)
        rqvae_dist = F.mse_loss(z_hat, z_global, reduction='none').mean(dim=-1)
        scores['rqvae'] = rqvae_dist

        try:
            token_scores, ar_score = model.tar.compute_anomaly_score(codes, z_global)
            scores['ar'] = ar_score
        except Exception:
            pass

    return scores, pixel_map


@torch.no_grad()
def evaluate_mvtec(model, test_loader, device, use_stage2=False):
    """Evaluate a category. Returns per-scoring-method metrics."""
    model.eval()
    score_bag = {}
    labels_all = []
    pixel_maps_all = []
    masks_all = []

    for batch in tqdm(test_loader, desc='Eval', leave=False):
        images = batch['image'].to(device)
        labels = batch['label']
        masks = batch['mask']

        scores, pixel_map = predict_mvtec(model, images, use_stage2=use_stage2)

        for name, s in scores.items():
            score_bag.setdefault(name, []).extend(s.cpu().numpy().tolist())
        labels_all.extend(labels.numpy().tolist())

        # pixel-level for defect samples only (uses recon-based pmap)
        Hm, Wm = masks.shape[-2:]
        pmap_up = F.interpolate(pixel_map.unsqueeze(1), size=(Hm, Wm),
                                 mode='bilinear', align_corners=False).squeeze(1)
        pixel_maps_all.append(pmap_up.squeeze().cpu().numpy())
        masks_all.append(masks.squeeze().cpu().numpy())

    labels_all = np.array(labels_all)

    def metrics_for(score_name, score_arr):
        score_arr = np.array(score_arr)
        f1max, thresh = compute_f1_max(score_arr, labels_all)
        preds = (score_arr >= thresh).astype(int)
        cm = confusion_matrix(labels_all, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        return {
            'method': score_name,
            'I-AUROC': float(compute_auroc(score_arr, labels_all)),
            'I-F1max': float(f1max),
            'I-AU-PR': float(compute_auprc(score_arr, labels_all)),
            'Accuracy': float(accuracy_score(labels_all, preds)),
            'Precision': float(precision_score(labels_all, preds, zero_division=0)),
            'Recall': float(recall_score(labels_all, preds, zero_division=0)),
            'F1-Score': float(f1_score(labels_all, preds, zero_division=0)),
            'Specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'Best_Threshold': float(thresh),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        }

    results = {name: metrics_for(name, arr) for name, arr in score_bag.items()}

    # ---- Fused score (Stage 2): z-score normalize then average ----
    if 'recon' in score_bag and 'rqvae' in score_bag:
        r = np.array(score_bag['recon'])
        q = np.array(score_bag['rqvae'])
        r_z = (r - r.mean()) / (r.std() + 1e-8)
        q_z = (q - q.mean()) / (q.std() + 1e-8)
        fused = 0.5 * r_z + 0.5 * q_z
        results['fused_recon_rqvae'] = metrics_for('fused_recon_rqvae', fused)

    # ---- Pixel-level metrics (uses recon-based pmap) ----
    has_defect = any(m.sum() > 0 for m in masks_all)
    pixel_metrics = {}
    if has_defect:
        defect_pmaps = [p for p, m in zip(pixel_maps_all, masks_all) if m.sum() > 0]
        defect_masks = [m for m in masks_all if m.sum() > 0]
        pix_scores = np.concatenate([p.flatten() for p in defect_pmaps])
        pix_labels = np.concatenate([m.flatten() for m in defect_masks])
        pixel_metrics['P-AUROC'] = float(compute_auroc(pix_scores, pix_labels))
        try:
            pixel_metrics['PRO'] = float(compute_pro(defect_pmaps, defect_masks))
        except Exception as e:
            print(f"  PRO failed: {e}")
            pixel_metrics['PRO'] = 0.0

    for name in results:
        results[name].update(pixel_metrics)

    return results


# ============================================================================
# Training routines
# ============================================================================
def train_stage1(model, train_loader, test_loader, device, logger,
                 epochs, lr, patience, eval_every, ckpt_path):
    """Reconstruction pre-training."""
    criterion = TopoVarADLoss(lambda_lpips=0.1, lambda_rqvae=0, lambda_ar=0,
                              lambda_diversity=0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    model.set_stage(1)
    best_auroc, best_state, best_epoch, no_improve = 0.0, None, 0, 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'  [S1] Epoch {epoch+1}/{epochs}', leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            optimizer.zero_grad()
            with autocast_ctx(device):
                outputs = model(images)
                loss = criterion(outputs, stage=1)['loss_total']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = train_loss / max(len(train_loader), 1)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        if (epoch + 1) % eval_every == 0:
            metrics = evaluate_mvtec(model, test_loader, device, use_stage2=False)
            m = metrics['recon']
            val_auroc = m['I-AUROC']
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(best_state, ckpt_path)
            else:
                no_improve += eval_every
            logger.info(
                f"  [S1] Epoch {epoch+1:>3d} | loss={avg_loss:.4f} | lr={lr_now:.2e} | "
                f"I-AUROC={val_auroc:.4f} I-F1={m['I-F1max']:.4f} | "
                f"P-AUROC={m.get('P-AUROC',0):.4f} PRO={m.get('PRO',0):.4f} | "
                f"best={best_auroc:.4f} @ep{best_epoch} | wait={no_improve}/{patience}"
            )
        else:
            logger.info(f"  [S1] Epoch {epoch+1:>3d} | loss={avg_loss:.4f} | lr={lr_now:.2e}")

        if no_improve >= patience:
            logger.info(f"  [S1] Early stop at epoch {epoch+1}")
            break

    train_time = (time.time() - t0) / 3600
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auroc, best_epoch, train_time


def freeze_backbone(model):
    """Freeze T2M-Tokenizer, TPM, Pixel Head (Stage 1 modules)."""
    frozen = 0
    for name, module in [
        ('tokenizer', getattr(model, 'tokenizer', None)),
        ('input_proj', getattr(model, 'input_proj', None)),
        ('patch_embed', getattr(model, 'patch_embed', None)),
        ('tpm', getattr(model, 'tpm', None)),
        ('pixel_head', getattr(model, 'pixel_head', None)),
        ('pool_head', getattr(model, 'pool_head', None)),
    ]:
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def train_stage2(model, train_loader, test_loader, device, logger, config,
                 epochs, lr, patience, eval_every, ckpt_path,
                 kmeans_init=True, kmeans_batches=20):
    """RQ-VAE + TAR joint training with frozen backbone."""
    model.set_stage(2)

    # Freeze Stage 1 modules
    frozen = freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  [S2] Frozen: {frozen:,} | Trainable (RQ-VAE + TAR): {trainable:,}")

    # K-means init RQ-VAE codebook from Stage 1 features
    if kmeans_init:
        try:
            n_feat = model.init_codebook_from_loader(
                train_loader, device, max_batches=kmeans_batches, n_iter=10)
            usages = model.rqvae.rq.get_codebook_usage()
            logger.info(
                f"  [S2] K-means init: {n_feat} features | "
                f"codebook usage: {[f'{u:.2%}' for u in usages]}"
            )
        except Exception as e:
            logger.info(f"  [S2] K-means init failed: {e}")

    criterion = TopoVarADLoss(
        lambda_lpips=0.0,       # Stage 1 modules frozen; skip pixel/lpips loss
        lambda_rqvae=config.get('train', {}).get('lambda_rqvae', 0.5),
        lambda_ar=config.get('train', {}).get('lambda_ar', 1.0),
        lambda_diversity=config.get('train', {}).get('lambda_diversity', 0.0),
        label_smoothing=config.get('train', {}).get('label_smoothing', 0.1),
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_auroc, best_state, best_epoch, no_improve = 0.0, None, 0, 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        # Keep frozen modules in eval to preserve BN etc.
        for mod_name in ['tokenizer', 'tpm', 'pixel_head', 'pool_head', 'input_proj', 'patch_embed']:
            if hasattr(model, mod_name):
                getattr(model, mod_name).eval()

        train_loss = train_rqvae = train_ar = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f'  [S2] Epoch {epoch+1}/{epochs}', leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            optimizer.zero_grad()
            with autocast_ctx(device):
                outputs = model(images)
                # Manually build loss without pixel loss since backbone frozen
                loss_rqvae = outputs.get('loss_rqvae', torch.tensor(0.0, device=device))
                loss_ar = outputs.get('loss_ar', torch.tensor(0.0, device=device))
                loss = (criterion.lambda_rqvae * loss_rqvae
                        + criterion.lambda_ar * loss_ar)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_rqvae += loss_rqvae.item()
            train_ar += loss_ar.item()
            n_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ar': f'{loss_ar.item():.4f}',
            })

        avg_loss = train_loss / max(n_batches, 1)
        avg_rq = train_rqvae / max(n_batches, 1)
        avg_ar = train_ar / max(n_batches, 1)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        if (epoch + 1) % eval_every == 0:
            metrics = evaluate_mvtec(model, test_loader, device, use_stage2=True)
            # Track best on fused score, but also report recon/rqvae
            m_recon = metrics['recon']
            m_rqvae = metrics.get('rqvae', {})
            m_fused = metrics.get('fused_recon_rqvae', {})
            val_auroc = m_fused.get('I-AUROC', m_recon['I-AUROC'])
            usages = model.rqvae.rq.get_codebook_usage()

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(best_state, ckpt_path)
            else:
                no_improve += eval_every

            logger.info(
                f"  [S2] Epoch {epoch+1:>3d} | loss={avg_loss:.4f} rqvae={avg_rq:.4f} ar={avg_ar:.4f} | "
                f"recon-AUROC={m_recon['I-AUROC']:.4f} rqvae-AUROC={m_rqvae.get('I-AUROC',0):.4f} "
                f"fused-AUROC={m_fused.get('I-AUROC',0):.4f} | "
                f"codebook={[f'{u:.0%}' for u in usages]} | "
                f"best={best_auroc:.4f} @ep{best_epoch} | wait={no_improve}/{patience}"
            )
        else:
            logger.info(f"  [S2] Epoch {epoch+1:>3d} | loss={avg_loss:.4f} "
                        f"rqvae={avg_rq:.4f} ar={avg_ar:.4f} | lr={lr_now:.2e}")

        if no_improve >= patience:
            logger.info(f"  [S2] Early stop at epoch {epoch+1}")
            break

    train_time = (time.time() - t0) / 3600
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auroc, best_epoch, train_time


def train_one_category(category, root, config, device, output_dir,
                       stage1_epochs=200, stage2_epochs=50,
                       patience=30, eval_every=5,
                       image_size=256, batch_size=8,
                       skip_stage2=False):
    """Full pipeline: Stage 1 → Stage 2 → Evaluation."""
    cat_dir = os.path.join(output_dir, category)
    os.makedirs(cat_dir, exist_ok=True)

    # Per-category logger
    log_path = os.path.join(cat_dir, 'train.log')
    logger = logging.getLogger(f'mvtec_full_{category}')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())

    logger.info(f"{'='*70}")
    logger.info(f"  MVTec Category: {category}")
    logger.info(f"{'='*70}")

    train_loader, test_loader = build_loaders(root, category, image_size, batch_size)
    logger.info(f"  Train (normal-only): {len(train_loader.dataset)}")
    logger.info(f"  Test: {len(test_loader.dataset)}")

    model = build_model(config, device)
    logger.info(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Stage 1 ----
    stage1_ckpt = os.path.join(cat_dir, 'stage1_best.pth')
    train_cfg = config.get('train', {})
    logger.info(f"\n>>> Stage 1: Reconstruction Pre-training ({stage1_epochs} epochs)")
    s1_auroc, s1_epoch, s1_time = train_stage1(
        model, train_loader, test_loader, device, logger,
        epochs=stage1_epochs,
        lr=train_cfg.get('lr_stage1', 1e-4),
        patience=patience,
        eval_every=eval_every,
        ckpt_path=stage1_ckpt,
    )
    logger.info(f"  [S1] Best AUROC={s1_auroc:.4f} @ ep{s1_epoch} | time={s1_time:.2f}h")

    # Full Stage 1 evaluation
    metrics_s1 = evaluate_mvtec(model, test_loader, device, use_stage2=False)
    metrics_s1['recon']['best_epoch'] = s1_epoch
    metrics_s1['recon']['train_time_h'] = s1_time
    with open(os.path.join(cat_dir, 'stage1_metrics.json'), 'w') as f:
        json.dump(metrics_s1, f, indent=2, default=float)

    if skip_stage2:
        return {'stage1': metrics_s1, 'stage2': None}

    # ---- Stage 2 ----
    stage2_ckpt = os.path.join(cat_dir, 'stage2_best.pth')
    logger.info(f"\n>>> Stage 2: RQ-VAE + TAR Joint Training ({stage2_epochs} epochs)")
    s2_auroc, s2_epoch, s2_time = train_stage2(
        model, train_loader, test_loader, device, logger, config,
        epochs=stage2_epochs,
        lr=train_cfg.get('lr_stage2', 1e-5),
        patience=patience,
        eval_every=eval_every,
        ckpt_path=stage2_ckpt,
    )
    logger.info(f"  [S2] Best fused-AUROC={s2_auroc:.4f} @ ep{s2_epoch} | time={s2_time:.2f}h")

    # Full Stage 2 evaluation (all scoring methods)
    metrics_s2 = evaluate_mvtec(model, test_loader, device, use_stage2=True)
    for name in metrics_s2:
        metrics_s2[name]['best_epoch'] = s2_epoch
        metrics_s2[name]['train_time_h'] = s2_time
    with open(os.path.join(cat_dir, 'stage2_metrics.json'), 'w') as f:
        json.dump(metrics_s2, f, indent=2, default=float)

    # Log final summary
    logger.info(f"\n=== FINAL [{category}] ===")
    for name, m in metrics_s2.items():
        logger.info(
            f"  {name:>20s}: I-AUROC={m['I-AUROC']:.4f} I-F1={m['I-F1max']:.4f} "
            f"AU-PR={m['I-AU-PR']:.4f} | "
            f"P-AUROC={m.get('P-AUROC',0):.4f} PRO={m.get('PRO',0):.4f} | "
            f"CM(TN/FP/FN/TP)={m['TN']}/{m['FP']}/{m['FN']}/{m['TP']}"
        )

    return {'stage1': metrics_s1, 'stage2': metrics_s2}


def print_summary(all_results, output_dir):
    """Aggregate and print summary."""
    print(f"\n{'='*90}")
    print(f"  MVTec AD Summary ({len(all_results)} categories)")
    print(f"{'='*90}")

    # Stage 1 (recon) summary
    print(f"\n[Stage 1: Reconstruction]")
    print(f"{'Category':<14} {'I-AUROC':>10} {'I-F1max':>10} {'P-AUROC':>10} {'PRO':>10}")
    print(f"{'-'*90}")
    keys_img = ['I-AUROC', 'I-F1max']
    keys_pix = ['P-AUROC', 'PRO']
    s1_means = {k: [] for k in keys_img + keys_pix}
    for cat in ALL_CATEGORIES:
        if cat not in all_results:
            continue
        m = all_results[cat]['stage1']['recon']
        print(f"{cat:<14} {m['I-AUROC']:>10.4f} {m['I-F1max']:>10.4f} "
              f"{m.get('P-AUROC',0):>10.4f} {m.get('PRO',0):>10.4f}")
        for k in keys_img + keys_pix:
            s1_means[k].append(m.get(k, 0))
    print(f"{'-'*90}")
    print(f"{'MEAN':<14} "
          f"{np.mean(s1_means['I-AUROC']):>10.4f} "
          f"{np.mean(s1_means['I-F1max']):>10.4f} "
          f"{np.mean(s1_means['P-AUROC']):>10.4f} "
          f"{np.mean(s1_means['PRO']):>10.4f}")

    # Stage 2 summary (best per-category among {recon, rqvae, fused})
    if any(r.get('stage2') for r in all_results.values()):
        print(f"\n[Stage 2: Full architecture — best score per category]")
        print(f"{'Category':<14} {'method':>18} {'I-AUROC':>10} {'I-F1max':>10} "
              f"{'P-AUROC':>10} {'PRO':>10}")
        print(f"{'-'*90}")
        s2_means = {k: [] for k in keys_img + keys_pix}
        for cat in ALL_CATEGORIES:
            if cat not in all_results or not all_results[cat].get('stage2'):
                continue
            metrics = all_results[cat]['stage2']
            best_name = max(metrics.keys(), key=lambda n: metrics[n]['I-AUROC'])
            m = metrics[best_name]
            print(f"{cat:<14} {best_name:>18s} {m['I-AUROC']:>10.4f} {m['I-F1max']:>10.4f} "
                  f"{m.get('P-AUROC',0):>10.4f} {m.get('PRO',0):>10.4f}")
            for k in keys_img + keys_pix:
                s2_means[k].append(m.get(k, 0))
        if s2_means['I-AUROC']:
            print(f"{'-'*90}")
            print(f"{'MEAN':<14} {'':>18} "
                  f"{np.mean(s2_means['I-AUROC']):>10.4f} "
                  f"{np.mean(s2_means['I-F1max']):>10.4f} "
                  f"{np.mean(s2_means['P-AUROC']):>10.4f} "
                  f"{np.mean(s2_means['PRO']):>10.4f}")

    print(f"{'='*90}\n")

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../mvtec_ad')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--category', type=str, default='all')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='logs/mvtec_full')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--stage1_epochs', type=int, default=200)
    parser.add_argument('--stage2_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--skip_stage2', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    categories = ALL_CATEGORIES if args.category == 'all' else [args.category]

    all_results = {}
    for cat in categories:
        try:
            res = train_one_category(
                cat, args.root, config, device, args.output_dir,
                stage1_epochs=args.stage1_epochs,
                stage2_epochs=args.stage2_epochs,
                patience=args.patience,
                eval_every=args.eval_every,
                image_size=args.image_size,
                batch_size=args.batch_size,
                skip_stage2=args.skip_stage2,
            )
            all_results[cat] = res

            s1 = res['stage1']['recon']
            print(f"\n  >>> [{cat}] Stage1 recon: I-AUROC={s1['I-AUROC']:.4f} "
                  f"P-AUROC={s1.get('P-AUROC',0):.4f}")
            if res.get('stage2'):
                for name, m in res['stage2'].items():
                    print(f"       Stage2 {name:>18s}: I-AUROC={m['I-AUROC']:.4f} "
                          f"P-AUROC={m.get('P-AUROC',0):.4f}")
        except Exception as e:
            print(f"  ERROR [{cat}]: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) > 1:
        print_summary(all_results, args.output_dir)


if __name__ == '__main__':
    main()
