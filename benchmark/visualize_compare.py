"""
Qualitative comparison: AE vs TopoVarAD anomaly heatmaps side by side.
"""

import os, sys, argparse, yaml
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import CSVDataset
from benchmark.methods import AutoencoderBaseline


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


@torch.no_grad()
def get_topovarad_heatmaps(model, loader, device):
    """Reconstruction error heatmaps from TopoVarAD Stage 1."""
    model.eval()
    results = []
    for batch in tqdm(loader, desc='TopoVarAD'):
        images = batch['image'].to(device)
        outputs = model(images)
        x_recon = outputs['reconstructed']
        x_resized = outputs['x_resized']
        error = torch.abs(x_recon - x_resized).mean(dim=1)  # (B, Hr, Wr)
        for j in range(images.shape[0]):
            pmap = F.interpolate(error[j].unsqueeze(0).unsqueeze(0),
                                 size=(512, 512), mode='bilinear').squeeze().cpu().numpy()
            results.append({
                'image': images[j].cpu(),
                'heatmap': pmap,
                'label': batch['label'][j].item(),
                'score': error[j].mean().item(),
            })
    return results


@torch.no_grad()
def get_ae_heatmaps(model, loader, device):
    """Reconstruction error heatmaps from Autoencoder."""
    model.eval()
    results = []
    for batch in tqdm(loader, desc='AE'):
        images = batch['image'].to(device)
        _, error_maps = model.anomaly_score(images)
        for j in range(images.shape[0]):
            pmap = F.interpolate(error_maps[j].unsqueeze(0).unsqueeze(0),
                                 size=(512, 512), mode='bilinear').squeeze().cpu().numpy()
            results.append({
                'image': images[j].cpu(),
                'heatmap': pmap,
                'label': batch['label'][j].item(),
                'score': error_maps[j].mean().item(),
            })
    return results


def create_comparison_figure(topo_results, ae_results, save_dir, num_samples=6):
    """Create side-by-side comparison figure."""
    os.makedirs(save_dir, exist_ok=True)

    # Pick samples: 3 normal, 3 anomaly
    normal_idx = [i for i, r in enumerate(topo_results) if r['label'] == 0][:3]
    anomaly_idx = [i for i, r in enumerate(topo_results) if r['label'] == 1][:3]
    selected = normal_idx + anomaly_idx

    fig, axes = plt.subplots(len(selected), 4, figsize=(16, 4 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]

    col_labels = ['Input', 'AE Heatmap', 'TopoVarAD Heatmap', 'GT Mask']
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=14, fontweight='bold')

    for row, idx in enumerate(selected):
        t = topo_results[idx]
        a = ae_results[idx]

        # Input image
        img = t['image'].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[row, 0].imshow(img)
        axes[row, 0].axis('off')
        axes[row, 0].set_ylabel(f"{'Normal' if t['label']==0 else 'Anomaly'}\n#{idx}",
                                fontsize=12, fontweight='bold')

        # AE heatmap
        axes[row, 1].imshow(a['heatmap'], cmap='jet')
        axes[row, 1].axis('off')

        # TopoVarAD heatmap
        axes[row, 2].imshow(t['heatmap'], cmap='jet')
        axes[row, 2].axis('off')

        # Placeholder for GT mask (if available)
        axes[row, 3].axis('off')

    plt.tight_layout()
    path = os.path.join(save_dir, 'comparison_heatmaps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--topo_ckpt', type=str, default='checkpoints/stage1_best.pth')
    parser.add_argument('--ae_ckpt', type=str, default='checkpoints/autoencoder.pth')
    parser.add_argument('--output_dir', type=str, default='logs/visualization')
    parser.add_argument('--num_samples', type=int, default=6)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build test loader
    data_cfg = config.get('data', {})
    test_dataset = CSVDataset(
        csv_path=data_cfg.get('test_csv', 'data/test.csv'),
        images_dir=data_cfg.get('images_dir', 'data/images'),
        split='test', image_size=data_cfg.get('image_size', 512), augment=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Load TopoVarAD Stage 1
    from models.topovarad import TopoVarAD, TopoVarADConfig
    model_cfg = config.get('model', {})
    topo = TopoVarADConfig(
        d_model=model_cfg.get('d_model', 256),
        n_tpm_layers=model_cfg.get('n_layers', 6),
        n_heads=model_cfg.get('n_heads', 8),
        superpixel_scales=tuple(model_cfg.get('superpixel_scales', [50, 100, 200])),
        rqvae_codebook_size=model_cfg.get('rqvae_codebook_size', 1024),
        rqvae_d_code=model_cfg.get('rqvae_d_code', 32),
        rqvae_n_layers=model_cfg.get('rqvae_n_layers', 8),
        tar_n_layers=model_cfg.get('tar_n_layers', 6),
        tar_n_heads=model_cfg.get('tar_n_heads', 8),
    ).build_model().to(device)
    ckpt = torch.load(args.topo_ckpt, map_location=device)
    topo.load_state_dict(ckpt['model_state_dict'])
    topo.set_stage(1)
    topo.eval()
    print(f"Loaded TopoVarAD from {args.topo_ckpt}")

    # Load AE
    ae = AutoencoderBaseline(latent_dim=256).to(device)
    if os.path.exists(args.ae_ckpt):
        ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device))
        ae.eval()
        print(f"Loaded AE from {args.ae_ckpt}")
    else:
        print(f"AE checkpoint not found at {args.ae_ckpt}, skipping AE heatmaps")
        ae = None

    topo_results = get_topovarad_heatmaps(topo, test_loader, device)
    ae_results = get_ae_heatmaps(ae, test_loader, device) if ae else None

    if ae_results:
        create_comparison_figure(topo_results, ae_results, args.output_dir, args.num_samples)

    # Also save individual TopoVarAD heatmaps
    print(f"\nTopoVarAD mean scores: Normal={np.mean([r['score'] for r in topo_results if r['label']==0]):.4f}, "
          f"Anomaly={np.mean([r['score'] for r in topo_results if r['label']==1]):.4f}")
    if ae_results:
        print(f"AE mean scores: Normal={np.mean([r['score'] for r in ae_results if r['label']==0]):.4f}, "
              f"Anomaly={np.mean([r['score'] for r in ae_results if r['label']==1]):.4f}")


if __name__ == '__main__':
    main()