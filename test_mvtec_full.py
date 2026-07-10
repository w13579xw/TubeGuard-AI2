"""
Evaluate a trained TopoVarAD model (Stage 1 or Stage 2) on MVTec AD.

Loads a checkpoint and produces:
  - Image-level metrics (I-AUROC, I-F1max, I-AU-PR, Accuracy, Precision, Recall, Specificity)
  - Confusion matrix (TN/FP/FN/TP)
  - Pixel-level metrics (P-AUROC, PRO) when defect masks are available
  - Score-method breakdown (recon / rqvae / ar / fused) for Stage 2 checkpoints

Output:
  logs/mvtec_eval/{category}_stage{1|2}_metrics.json
  logs/mvtec_eval/{category}_stage{1|2}_scores.csv

Checkpoints are expected under:
  checkpoints/mvtec_full/{category}/stage{1|2}_best.pth
(set by train_mvtec_full.py --checkpoint_dir)

Usage:
  # Single category, Stage 1 checkpoint
  python test_mvtec_full.py --root /path/to/mvtec --category bottle \
      --checkpoint checkpoints/mvtec_full/bottle/stage1_best.pth --stage 1

  # Single category, Stage 2 checkpoint (full model with RQ-VAE + TAR)
  python test_mvtec_full.py --root /path/to/mvtec --category bottle \
      --checkpoint checkpoints/mvtec_full/bottle/stage2_best.pth --stage 2

  # All categories, auto-locate checkpoints under a common root
  python test_mvtec_full.py --root /path/to/mvtec --category all \
      --checkpoint_root checkpoints/mvtec_full --stage 2
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import MVTecDataset
from models.topovarad import TopoVarADConfig
from utils.metrics import compute_auprc, compute_auroc, compute_f1_max, compute_pro


ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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


def build_test_loader(root, category, image_size=256, num_workers=4):
    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    test_dataset = MVTecDataset(root=root, category=category, split='test',
                                image_size=image_size, transform=test_tf)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, test_dataset


@torch.no_grad()
def predict_full(model, x, use_stage2=False):
    """
    Compute all anomaly scores for one batch.
    Returns dict {name: (B,)} + pixel_map (B, H, W) from recon.
    """
    model.eval()
    tokens, sp_labels, M, N = model._tokenize(x)
    refined = model.tpm(tokens, sp_labels, M, N)
    B, C, H, W = x.shape

    # Recon score (MVTec: normal direction, defect > normal)
    x_recon = model.pixel_head(refined, M, N)
    H_t, W_t = M * 16, N * 16
    x_resized = F.interpolate(x, size=(H_t, W_t), mode='bilinear', align_corners=False)
    recon_err = F.l1_loss(x_recon, x_resized, reduction='none').mean(dim=1)
    recon_score = recon_err.mean(dim=[1, 2])

    pixel_map = F.interpolate(recon_err.unsqueeze(1),
                              size=(H, W), mode='bilinear', align_corners=False).squeeze(1)

    scores = {'recon': recon_score}

    if use_stage2:
        z_global = model.pool_head(refined)
        z_hat, codes, _, _ = model.rqvae(z_global)
        rqvae_dist = F.mse_loss(z_hat, z_global, reduction='none').mean(dim=-1)
        scores['rqvae'] = rqvae_dist

        try:
            _, ar_score = model.tar.compute_anomaly_score(codes, z_global)
            scores['ar'] = ar_score
        except Exception:
            pass

    return scores, pixel_map


@torch.no_grad()
def score_dataset(model, loader, device, use_stage2=False):
    """Run all test samples through the model and collect scores + labels + masks + paths."""
    model.eval()
    bag = {}
    labels_all, pixel_maps_all, masks_all, paths_all = [], [], [], []
    defect_types = []

    for batch in tqdm(loader, desc='Scoring'):
        images = batch['image'].to(device)
        scores, pixel_map = predict_full(model, images, use_stage2=use_stage2)

        for name, s in scores.items():
            bag.setdefault(name, []).extend(s.cpu().numpy().tolist())
        labels_all.extend(batch['label'].numpy().astype(int).tolist())
        paths_all.extend(batch.get('image_path', [''] * images.shape[0]))
        defect_types.extend(batch.get('defect_type', [''] * images.shape[0]))

        Hm, Wm = batch['mask'].shape[-2:]
        pmap_up = F.interpolate(pixel_map.unsqueeze(1), size=(Hm, Wm),
                                 mode='bilinear', align_corners=False).squeeze(1)
        pixel_maps_all.append(pmap_up.squeeze().cpu().numpy())
        masks_all.append(batch['mask'].squeeze().cpu().numpy())

    labels_all = np.array(labels_all)
    for name in bag:
        bag[name] = np.array(bag[name])

    return {
        'scores': bag,
        'labels': labels_all,
        'pixel_maps': pixel_maps_all,
        'masks': masks_all,
        'paths': paths_all,
        'defect_types': defect_types,
    }


def compute_image_metrics(score_arr, labels):
    """Full image-level metrics including confusion matrix at F1-max threshold."""
    f1max, thresh = compute_f1_max(score_arr, labels)
    preds = (score_arr >= thresh).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        'I-AUROC': float(compute_auroc(score_arr, labels)),
        'I-F1max': float(f1max),
        'I-AU-PR': float(compute_auprc(score_arr, labels)),
        'Accuracy': float(accuracy_score(labels, preds)),
        'Precision': float(precision_score(labels, preds, zero_division=0)),
        'Recall': float(recall_score(labels, preds, zero_division=0)),
        'F1-Score': float(f1_score(labels, preds, zero_division=0)),
        'Specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'Best_Threshold': float(thresh),
        'True_Negative': int(tn), 'False_Positive': int(fp),
        'False_Negative': int(fn), 'True_Positive': int(tp),
    }


def compute_pixel_metrics(pixel_maps, masks):
    """P-AUROC and PRO on defect samples only."""
    defect_maps = [p for p, m in zip(pixel_maps, masks) if m.sum() > 0]
    defect_masks = [m for m in masks if m.sum() > 0]
    if not defect_maps:
        return {}
    pix_scores = np.concatenate([p.flatten() for p in defect_maps])
    pix_labels = np.concatenate([m.flatten() for m in defect_masks])
    out = {'P-AUROC': float(compute_auroc(pix_scores, pix_labels))}
    try:
        out['PRO'] = float(compute_pro(defect_maps, defect_masks))
    except Exception as e:
        print(f"  PRO failed: {e}")
        out['PRO'] = 0.0
    return out


def build_all_results(scored, stage):
    """Compute metrics for each scoring method + fusion."""
    labels = scored['labels']
    scores = scored['scores']

    results = {}
    for name, arr in scores.items():
        results[name] = compute_image_metrics(arr, labels)

    # Fusion (Stage 2 only)
    if stage == 2 and 'recon' in scores and 'rqvae' in scores:
        r = scores['recon']
        q = scores['rqvae']
        r_z = (r - r.mean()) / (r.std() + 1e-8)
        q_z = (q - q.mean()) / (q.std() + 1e-8)
        fused = 0.5 * r_z + 0.5 * q_z
        results['fused_recon_rqvae'] = compute_image_metrics(fused, labels)
        scores['fused_recon_rqvae'] = fused  # save for CSV export

    # Attach pixel metrics to every method (same recon pmap)
    pixel_metrics = compute_pixel_metrics(scored['pixel_maps'], scored['masks'])
    for name in results:
        results[name].update(pixel_metrics)

    return results


def save_scores_csv(scored, results, out_path):
    """Per-sample scores CSV: path, label, defect_type, recon, rqvae, ar, fused."""
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        methods = sorted(scored['scores'].keys())
        header = ['image_path', 'label', 'defect_type'] + methods
        writer.writerow(header)
        n = len(scored['labels'])
        for i in range(n):
            row = [
                scored['paths'][i] if i < len(scored['paths']) else '',
                int(scored['labels'][i]),
                scored['defect_types'][i] if i < len(scored['defect_types']) else '',
            ]
            for m in methods:
                row.append(float(scored['scores'][m][i]))
            writer.writerow(row)


def print_metrics(category, stage, results):
    print(f"\n{'='*90}")
    print(f"  [{category}] Stage {stage} Evaluation")
    print(f"{'='*90}")
    header = f"{'Method':<20} {'I-AUROC':>9} {'I-F1max':>9} {'AU-PR':>9} " \
             f"{'Acc':>8} {'Prec':>8} {'Rec':>8} {'Spec':>8} " \
             f"{'P-AUROC':>9} {'PRO':>7} {'TN/FP/FN/TP':>20}"
    print(header)
    print('-' * 90)
    for name, m in results.items():
        cm_str = f"{m['True_Negative']}/{m['False_Positive']}/" \
                 f"{m['False_Negative']}/{m['True_Positive']}"
        print(f"{name:<20} "
              f"{m['I-AUROC']:>9.4f} {m['I-F1max']:>9.4f} {m['I-AU-PR']:>9.4f} "
              f"{m['Accuracy']:>8.4f} {m['Precision']:>8.4f} "
              f"{m['Recall']:>8.4f} {m['Specificity']:>8.4f} "
              f"{m.get('P-AUROC', 0):>9.4f} {m.get('PRO', 0):>7.4f} "
              f"{cm_str:>20}")


def eval_one_category(category, root, checkpoint, config, device, stage, output_dir,
                      image_size=256, num_workers=4):
    """Evaluate a single category."""
    print(f"\n>>> Evaluating [{category}] with checkpoint {checkpoint}")
    if not os.path.exists(checkpoint):
        print(f"  Checkpoint not found: {checkpoint}")
        return None

    loader, dataset = build_test_loader(root, category, image_size, num_workers)
    print(f"  Test samples: {len(dataset)}")

    model = build_model(config, device)
    ckpt = torch.load(checkpoint, map_location=device)
    # Handle both raw state_dict and wrapped ckpt {'model_state_dict': ...}
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys (using defaults): {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {len(unexpected)}")
    model.set_stage(stage)

    t0 = time.time()
    scored = score_dataset(model, loader, device, use_stage2=(stage == 2))
    inf_time = time.time() - t0
    print(f"  Inference: {inf_time:.1f}s ({inf_time/max(len(dataset),1)*1000:.1f} ms/img)")

    results = build_all_results(scored, stage)
    for name in results:
        results[name]['inference_time_s'] = inf_time
        results[name]['inference_per_image_ms'] = inf_time / max(len(dataset), 1) * 1000

    print_metrics(category, stage, results)

    # Save
    out_json = os.path.join(output_dir, f'{category}_stage{stage}_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    out_csv = os.path.join(output_dir, f'{category}_stage{stage}_scores.csv')
    save_scores_csv(scored, results, out_csv)
    print(f"  Saved: {out_json}\n         {out_csv}")

    return results


def print_summary(all_results, stage):
    """Cross-category summary."""
    print(f"\n{'='*90}")
    print(f"  MVTec AD Cross-Category Summary — Stage {stage}")
    print(f"{'='*90}")

    method_names = set()
    for cat_results in all_results.values():
        if cat_results:
            method_names.update(cat_results.keys())
    method_names = sorted(method_names)

    for method in method_names:
        print(f"\n[Scoring method: {method}]")
        print(f"{'Category':<14} {'I-AUROC':>9} {'I-F1max':>9} {'AU-PR':>9} "
              f"{'P-AUROC':>9} {'PRO':>7}")
        print('-' * 90)
        aurocs = []
        f1s = []
        auprs = []
        pauroc = []
        pros = []
        for cat in ALL_CATEGORIES:
            if cat not in all_results or not all_results[cat]:
                continue
            if method not in all_results[cat]:
                continue
            m = all_results[cat][method]
            print(f"{cat:<14} {m['I-AUROC']:>9.4f} {m['I-F1max']:>9.4f} "
                  f"{m['I-AU-PR']:>9.4f} {m.get('P-AUROC', 0):>9.4f} "
                  f"{m.get('PRO', 0):>7.4f}")
            aurocs.append(m['I-AUROC'])
            f1s.append(m['I-F1max'])
            auprs.append(m['I-AU-PR'])
            pauroc.append(m.get('P-AUROC', 0))
            pros.append(m.get('PRO', 0))
        if aurocs:
            print('-' * 90)
            print(f"{'MEAN':<14} {np.mean(aurocs):>9.4f} {np.mean(f1s):>9.4f} "
                  f"{np.mean(auprs):>9.4f} {np.mean(pauroc):>9.4f} "
                  f"{np.mean(pros):>7.4f}")
    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../mvtec_ad',
                        help='MVTec AD root directory')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--category', type=str, default='all',
                        help='Category name or "all"')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Single-category checkpoint path')
    parser.add_argument('--checkpoint_root', type=str, default='checkpoints/mvtec_full',
                        help='For --category all: root dir with '
                             '{category}/stage{1|2}_best.pth structure')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                        help='1 = reconstruction only, 2 = full model with RQ-VAE + TAR')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='logs/mvtec_eval')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate MVTec root up-front.
    if not os.path.isdir(args.root):
        raise SystemExit(
            f"\n[ERROR] MVTec root does not exist: {args.root}\n"
            f"        Pass --root <path_to_mvtec_ad> where the directory contains\n"
            f"        subfolders like bottle/, cable/, capsule/, ...\n"
        )
    missing_cats = [c for c in ALL_CATEGORIES
                    if not os.path.isdir(os.path.join(args.root, c))]
    if len(missing_cats) == len(ALL_CATEGORIES):
        raise SystemExit(
            f"\n[ERROR] --root points to {os.path.abspath(args.root)}, but none of the\n"
            f"        expected MVTec category subfolders were found.\n"
            f"        Contents observed: {sorted(os.listdir(args.root))[:15]}\n"
        )

    categories = ALL_CATEGORIES if args.category == 'all' else [args.category]

    all_results = {}
    for cat in categories:
        if args.category == 'all':
            ckpt_path = os.path.join(args.checkpoint_root, cat,
                                      f'stage{args.stage}_best.pth')
        else:
            ckpt_path = args.checkpoint or os.path.join(
                args.checkpoint_root, cat, f'stage{args.stage}_best.pth')

        try:
            res = eval_one_category(
                cat, args.root, ckpt_path, config, device, args.stage,
                args.output_dir, image_size=args.image_size,
                num_workers=args.num_workers,
            )
            all_results[cat] = res
        except Exception as e:
            print(f"  ERROR [{cat}]: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) > 1:
        print_summary(all_results, args.stage)
        with open(os.path.join(args.output_dir, f'summary_stage{args.stage}.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=float)


if __name__ == '__main__':
    main()
