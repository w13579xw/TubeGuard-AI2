"""
Run SOTA anomaly detection benchmarks for comparison with TopoVarAD.

Methods:
  1. PaDiM   - Patch Distribution Modeling (ICPR 2021)
  2. PatchCore - Coreset nearest-neighbor (CVPR 2022)
  3. AE      - Autoencoder reconstruction baseline
  4. TopoVarAD Stage 1 - Our method (reconstruction)

Usage:
  python benchmark/run_all.py --config configs/default.yaml --device cuda
"""

import os
import sys
import argparse
import yaml
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import CSVDataset
from utils.metrics import compute_auroc, compute_f1_max, compute_auprc
from benchmark.methods import (
    PaDiM, PatchCore, AutoencoderBaseline,
    RD4AD, EfficientAD,
    build_train_loader, build_test_loader, evaluate_method
)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# Autoencoder training
# ============================================================

def train_autoencoder(model, train_loader, device, epochs=100, lr=1e-3):
    """Train autoencoder on normal samples only (reconstruction)."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nTraining Autoencoder: {epochs} epochs, lr={lr}")
    pbar = tqdm(range(epochs), desc='AE training')
    best_loss = float('inf')

    for epoch in pbar:
        total_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            x_hat, _ = model(images)
            loss = F.l1_loss(x_hat, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'best': f'{best_loss:.4f}'})

    return model


# ============================================================
# Benchmark runner
# ============================================================

def run_padim(config, train_loader, test_loader, device):
    """Run PaDiM benchmark."""
    print("\n" + "=" * 60)
    print("  Running PaDiM Benchmark")
    print("=" * 60)

    padim = PaDiM(backbone='resnet18', device=device)
    t0 = time.time()
    padim.fit(train_loader)
    fit_time = time.time() - t0

    t0 = time.time()
    image_scores, pixel_maps, labels = padim.predict(test_loader)
    inf_time = time.time() - t0

    masks_list = [batch['mask'].squeeze().numpy() for batch in test_loader if batch['mask'].sum() > 0]
    results = evaluate_method(image_scores, pixel_maps, labels, masks_list, 'PaDiM')
    results['fit_time'] = fit_time
    results['inference_time'] = inf_time
    return results


def run_patchcore(config, train_loader, test_loader, device):
    """Run PatchCore benchmark."""
    print("\n" + "=" * 60)
    print("  Running PatchCore Benchmark")
    print("=" * 60)

    # Use resnet18 for stability (wideresnet50 may not be downloadable on offline servers)
    patchcore = PatchCore(backbone='resnet18', coreset_ratio=0.01, device=device)
    t0 = time.time()
    patchcore.fit(train_loader)
    fit_time = time.time() - t0

    t0 = time.time()
    image_scores, pixel_maps, labels = patchcore.predict(test_loader)
    inf_time = time.time() - t0

    masks_list = [batch['mask'].squeeze().numpy() for batch in test_loader if batch['mask'].sum() > 0]
    results = evaluate_method(image_scores, pixel_maps, labels, masks_list, 'PatchCore')
    results['fit_time'] = fit_time
    results['inference_time'] = inf_time
    return results


def run_autoencoder(config, train_loader, test_loader, device):
    """Run Autoencoder baseline."""
    print("\n" + "=" * 60)
    print("  Running Autoencoder Benchmark")
    print("=" * 60)

    model = AutoencoderBaseline(latent_dim=256).to(device)
    model = train_autoencoder(model, train_loader, device, epochs=100, lr=1e-3)
    model.eval()

    image_scores = []
    pixel_maps = []
    all_labels = []
    all_masks = []

    t0 = time.time()
    for batch in tqdm(test_loader, desc='AE: predicting'):
        images = batch['image'].to(device)
        labels = batch['label']
        img_score, error_map = model.anomaly_score(images)

        image_scores.extend(img_score.cpu().tolist())
        all_labels.extend(labels.tolist())

        for j in range(images.shape[0]):
            pmap = error_map[j].unsqueeze(0).unsqueeze(0)
            pmap = F.interpolate(pmap.float(), size=(512, 512), mode='bilinear', align_corners=False)
            pixel_maps.append(pmap.squeeze().cpu().numpy())

        if batch['mask'].sum() > 0:
            all_masks.append(batch['mask'].squeeze().cpu().numpy())

    inf_time = time.time() - t0

    results = evaluate_method(np.array(image_scores), pixel_maps, np.array(all_labels),
                              all_masks if all_masks else [], 'Autoencoder')
    results['inference_time'] = inf_time
    return results


def run_rd4ad(config, train_loader, test_loader, device):
    """Run RD4AD (Reverse Distillation) benchmark."""
    print("\n" + "=" * 60)
    print("  Running RD4AD Benchmark")
    print("=" * 60)

    rd4ad = RD4AD(device=device)
    rd4ad.fit(train_loader, epochs=60, lr=0.005)

    image_scores, pixel_maps, labels = rd4ad.predict(test_loader)
    masks_list = [batch['mask'].squeeze().numpy() for batch in test_loader if batch['mask'].sum() > 0]
    results = evaluate_method(image_scores, pixel_maps, labels, masks_list, 'RD4AD')
    return results


def run_efficientad(config, train_loader, test_loader, device):
    """Run EfficientAD (simplified teacher-student) benchmark."""
    print("\n" + "=" * 60)
    print("  Running EfficientAD Benchmark")
    print("=" * 60)

    ead = EfficientAD(device=device)
    ead.fit(train_loader, epochs=60, lr=1e-3)

    image_scores, pixel_maps, labels = ead.predict(test_loader)
    masks_list = [batch['mask'].squeeze().numpy() for batch in test_loader if batch['mask'].sum() > 0]
    results = evaluate_method(image_scores, pixel_maps, labels, masks_list, 'EfficientAD')
    return results


def run_topovarad_stage1(config, test_loader, device):
    """Run TopoVarAD Stage 1 benchmark (reconstruction error)."""
    print("\n" + "=" * 60)
    print("  Running TopoVarAD Stage 1 Benchmark")
    print("=" * 60)

    from models.topovarad import TopoVarAD, TopoVarADConfig

    # Build batch_size=1 test loader: SLIC tokenizer treats the batch as a
    # single pooled set of superpixels, so multi-image batches mix tokens
    # from different images, producing meaningless reconstruction.
    data_config = config.get('data', {})
    test_dataset = CSVDataset(
        csv_path=data_config.get('test_csv', 'data/test.csv'),
        images_dir=data_config.get('images_dir', 'data/images'),
        split='test',
        image_size=data_config.get('image_size', 512),
        augment=False,
    )
    test_loader_bs1 = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=data_config.get('num_workers', 2),
                                 pin_memory=True)

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

    # Prefer normal-only checkpoint; fall back to stage1_best
    ckpt_dir = config.get('train', {}).get('checkpoint_dir', 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, 'stage1_normal_best.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, 'stage1_best.pth')
        print("  (using stage1_best.pth — mixed normal+anomaly training)")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.set_stage(1)
    model.eval()
    print(f"  Loaded Stage 1 checkpoint: {ckpt_path} (epoch {ckpt['epoch']+1})")

    image_scores = []
    pixel_maps = []
    all_labels = []
    all_masks = []

    t0 = time.time()
    for batch in tqdm(test_loader_bs1, desc='TopoVarAD-S1: predicting'):
        images = batch['image'].to(device)
        labels = batch['label']

        with torch.no_grad():
            outputs = model(images)
            x_recon = outputs['reconstructed']  # (1, 3, Hr, Wr) per-image
            x_resized = outputs['x_resized']    # (1, 3, Hr, Wr)
            # Per-pixel L1 → mean over channels and spatial dims
            error = torch.abs(x_recon - x_resized)
            img_score = error.mean()
            pixel_error = error.mean(dim=1)  # (1, Hr, Wr)

        image_scores.append(img_score.item())
        all_labels.append(labels.item())

        pmap = F.interpolate(pixel_error.unsqueeze(0), size=(512, 512),
                            mode='bilinear', align_corners=False).squeeze()
        pixel_maps.append(pmap.cpu().numpy())

        if batch['mask'].sum() > 0:
            all_masks.append(batch['mask'].squeeze().cpu().numpy())

    inf_time = time.time() - t0

    results = evaluate_method(np.array(image_scores), pixel_maps, np.array(all_labels),
                              all_masks if all_masks else [], 'TopoVarAD-Stage1')
    results['inference_time'] = inf_time
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SOTA Benchmark for Anomaly Detection')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['padim', 'patchcore', 'ae', 'rd4ad', 'efficientad', 'topovarad_s1'],
                        choices=['padim', 'patchcore', 'ae', 'rd4ad', 'efficientad', 'topovarad_s1', 'all'],
                        help='Methods to benchmark')
    parser.add_argument('--output', type=str, default='logs/benchmark_results.json')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build data loaders
    train_loader = build_train_loader(config)
    test_loader = build_test_loader(config)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    if 'all' in args.methods:
        args.methods = ['padim', 'patchcore', 'ae', 'rd4ad', 'efficientad', 'topovarad_s1']

    all_results = {}

    for method in args.methods:
        try:
            if method == 'padim':
                results = run_padim(config, train_loader, test_loader, device)
            elif method == 'patchcore':
                results = run_patchcore(config, train_loader, test_loader, device)
            elif method == 'ae':
                results = run_autoencoder(config, train_loader, test_loader, device)
            elif method == 'rd4ad':
                results = run_rd4ad(config, train_loader, test_loader, device)
            elif method == 'efficientad':
                results = run_efficientad(config, train_loader, test_loader, device)
            elif method == 'topovarad_s1':
                results = run_topovarad_stage1(config, test_loader, device)
            else:
                print(f"Unknown method: {method}")
                continue

            all_results[method] = results
        except Exception as e:
            print(f"  ERROR running {method}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'I-AUROC':>10} {'P-AUROC':>10} {'I-F1max':>10} {'Inf Time':>10}")
    print("-" * 70)
    for method, res in all_results.items():
        iauroc = res.get('I-AUROC', float('nan'))
        pauroc = res.get('P-AUROC', float('nan'))
        f1max = res.get('I-F1max', float('nan'))
        inft = res.get('inference_time', float('nan'))
        print(f"{method:<25} {iauroc:>10.4f} {pauroc:>10.4f} {f1max:>10.4f} {inft:>9.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
