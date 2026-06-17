"""
Train and evaluate TopoVarAD on MVTec AD dataset (per-category).

MVTec AD has 15 categories. For each category:
  1. Train Stage 1 on normal samples only (with early stopping by val AUROC)
  2. Evaluate on test set with both image-level and pixel-level metrics
  3. Save checkpoints and per-category results

Output: logs/mvtec/{category}/ + logs/mvtec/summary.json

Usage:
  # Single category
  python train_mvtec.py --root /path/to/mvtec --category bottle --device cuda

  # All categories
  python train_mvtec.py --root /path/to/mvtec --category all --device cuda

  # Parallel (one process per category — recommended)
  nohup python train_mvtec.py --root /path/to/mvtec --category bottle --device cuda > logs/mvtec/bottle.log 2>&1 &
"""

import os, sys, argparse, yaml, json, time, logging
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from data.dataset import MVTecDataset
from models.topovarad import TopoVarAD, TopoVarADConfig
from utils.losses import TopoVarADLoss
from utils.metrics import compute_auroc, compute_f1_max, compute_auprc, compute_pro


# All 15 MVTec AD categories
ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_loaders(root, category, image_size=256, batch_size=8, num_workers=4):
    """Build MVTec train (normal-only) and test loaders for one category."""
    train_dataset = MVTecDataset(root=root, category=category, split='train', image_size=image_size)
    test_dataset = MVTecDataset(root=root, category=category, split='test', image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate_mvtec(model, test_loader, device):
    """Full evaluation: image-level + pixel-level + PRO."""
    model.eval()
    img_scores, img_labels = [], []
    pixel_maps_list, masks_list = [], []

    for batch in tqdm(test_loader, desc='Eval', leave=False):
        images = batch['image'].to(device)
        labels = batch['label']
        masks = batch['mask']

        outputs = model(images)
        x_recon = outputs['reconstructed']
        x_resized = outputs['x_resized']

        # Per-pixel L1 error
        error = torch.abs(x_recon - x_resized).mean(dim=1)  # (1, Hr, Wr)
        img_score = error.mean()
        # Upsample pixel error to mask size
        H, W = masks.shape[-2:]
        pmap = F.interpolate(error.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze()

        img_scores.append(img_score.item())
        img_labels.append(labels.item())
        pixel_maps_list.append(pmap.cpu().numpy())
        masks_list.append(masks.squeeze().cpu().numpy())

    img_scores = np.array(img_scores)
    img_labels = np.array(img_labels)

    results = {
        'I-AUROC': compute_auroc(img_scores, img_labels),
        'I-F1max': compute_f1_max(img_scores, img_labels)[0],
        'I-AU-PR': compute_auprc(img_scores, img_labels),
    }

    # Pixel-level (only if there are any defect masks)
    has_defect_mask = any(m.sum() > 0 for m in masks_list)
    if has_defect_mask:
        # Stack only defect samples for pixel-level eval
        defect_pmaps = [pmap for pmap, m in zip(pixel_maps_list, masks_list) if m.sum() > 0]
        defect_masks = [m for m in masks_list if m.sum() > 0]

        all_pixel_scores = np.concatenate([p.flatten() for p in defect_pmaps])
        all_pixel_labels = np.concatenate([m.flatten() for m in defect_masks])
        results['P-AUROC'] = compute_auroc(all_pixel_scores, all_pixel_labels)
        try:
            results['PRO'] = compute_pro(defect_pmaps, defect_masks)
        except Exception as e:
            print(f"  PRO computation failed: {e}")
            results['PRO'] = 0.0

    return results


def train_one_category(category, root, config, device, output_dir,
                       max_epochs=200, patience=30, eval_every=5):
    """Train Stage 1 on one MVTec category with early stopping."""
    train_cfg = config.get('train', {})
    model_cfg = config.get('model', {})

    image_size = 256  # MVTec standard
    batch_size = 8

    # Setup per-category logger
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f'{category}.log')
    logger = logging.getLogger(category)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())

    logger.info(f"{'='*60}\n  MVTec Category: {category}\n{'='*60}")
    print(f"\n{'='*60}\n  MVTec Category: {category}\n{'='*60}")

    train_loader, test_loader = build_loaders(root, category, image_size, batch_size)
    logger.info(f"  Train: {len(train_loader.dataset)} normal | Test: {len(test_loader.dataset)}")

    # Build model
    topo_cfg = TopoVarADConfig(
        d_model=model_cfg.get('d_model', 256),
        n_tpm_layers=model_cfg.get('n_layers', 6),
        n_heads=model_cfg.get('n_heads', 8),
        superpixel_scales=tuple(model_cfg.get('superpixel_scales', [50, 100, 200])),
        rqvae_codebook_size=model_cfg.get('rqvae_codebook_size', 1024),
        rqvae_d_code=model_cfg.get('rqvae_d_code', 32),
        rqvae_n_layers=model_cfg.get('rqvae_n_layers', 8),
        tar_n_layers=model_cfg.get('tar_n_layers', 6),
        tar_n_heads=model_cfg.get('tar_n_heads', 8),
        use_slic=True, use_topo_attn=True, use_glpe=True,
    )
    model = topo_cfg.build_model().to(device)
    model.set_stage(1)
    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = TopoVarADLoss(
        lambda_lpips=train_cfg.get('lambda_lpips', 0.1),
        lambda_rqvae=0, lambda_ar=0,
        lambda_diversity=0,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.get('lr_stage1', 1e-4),
                             weight_decay=0.05)
    scaler = GradScaler()

    best_auroc = 0.0
    best_state = None
    best_epoch = 0
    no_improve = 0
    t_start = time.time()

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'  [{category}] Epoch {epoch+1}/{max_epochs}', leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, stage=1)['loss_total']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = train_loss / len(train_loader)

        if (epoch + 1) % eval_every == 0:
            metrics = evaluate_mvtec(model, test_loader, device)
            val_auroc = metrics['I-AUROC']
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += eval_every
            logger.info(f"  Epoch {epoch+1:>3d} | loss={avg_loss:.4f} | "
                        f"I-AUROC={val_auroc:.4f} | P-AUROC={metrics.get('P-AUROC', 0):.4f} | "
                        f"best={best_auroc:.4f} | wait={no_improve}/{patience}")
        else:
            logger.info(f"  Epoch {epoch+1:>3d} | loss={avg_loss:.4f}")

        if no_improve >= patience:
            logger.info(f"  Early stop at epoch {epoch+1}, best={best_auroc:.4f} @ epoch {best_epoch}")
            break

    # Restore best, do final full eval
    if best_state is not None:
        model.load_state_dict(best_state)
        ckpt_path = os.path.join(output_dir, f'{category}_best.pth')
        torch.save(best_state, ckpt_path)
        print(f"  [{category}] Saved: {ckpt_path}")

    final = evaluate_mvtec(model, test_loader, device)
    final['best_epoch'] = best_epoch
    final['train_time_h'] = (time.time() - t_start) / 3600

    # Save per-category JSON
    result_path = os.path.join(output_dir, f'{category}.json')
    with open(result_path, 'w') as f:
        json.dump(final, f, indent=2, default=float)

    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../mvtec_ad', help='MVTec AD root dir')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--category', type=str, default='all',
                        help='Category name or "all"')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='logs/mvtec')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.category == 'all':
        categories = ALL_CATEGORIES
    else:
        categories = [args.category]

    all_results = {}
    for cat in categories:
        try:
            res = train_one_category(cat, args.root, config, device, args.output_dir,
                                     max_epochs=args.max_epochs, patience=args.patience)
            all_results[cat] = res
            print(f"  >>> {cat}: I-AUROC={res['I-AUROC']:.4f} P-AUROC={res.get('P-AUROC', 0):.4f}")
        except Exception as e:
            print(f"  ERROR {cat}: {e}")
            import traceback; traceback.print_exc()

    # Aggregate summary
    if len(all_results) > 1:
        with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=float)

        # Compute means
        keys = ['I-AUROC', 'I-F1max', 'I-AU-PR', 'P-AUROC', 'PRO']
        means = {}
        for k in keys:
            vals = [r.get(k, 0) for r in all_results.values() if k in r]
            if vals:
                means[k] = sum(vals) / len(vals)

        print(f"\n{'='*70}\n  MVTec AD SUMMARY ({len(all_results)} categories)\n{'='*70}")
        print(f"{'Category':<14} {'I-AUROC':>10} {'I-F1max':>10} {'P-AUROC':>10} {'PRO':>10}")
        print(f"{'-'*70}")
        for cat in ALL_CATEGORIES:
            if cat not in all_results:
                continue
            r = all_results[cat]
            print(f"{cat:<14} {r['I-AUROC']:>10.4f} {r.get('I-F1max',0):>10.4f} "
                  f"{r.get('P-AUROC',0):>10.4f} {r.get('PRO',0):>10.4f}")
        print(f"{'-'*70}")
        print(f"{'MEAN':<14} {means.get('I-AUROC',0):>10.4f} {means.get('I-F1max',0):>10.4f} "
              f"{means.get('P-AUROC',0):>10.4f} {means.get('PRO',0):>10.4f}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
