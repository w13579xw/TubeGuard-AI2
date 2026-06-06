"""
Ablation study variants of TopoVarAD Stage 1.

Variants:
  1. Full TopoVarAD S1 (baseline)
  2. w/o SLIC: replace T2M-Tokenizer with fixed-grid 16x16 patch embedding
  3. w/o TPM: replace TPM Block with plain bidirectional Mamba (no TopoAttn)
  4. w/o Graph Laplacian PE: use learned position embedding instead

Each variant trains on normal-only data for 50 epochs (fast ablation).
"""

import os
import sys
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from data.dataset import CSVDataset
from utils.metrics import compute_auroc, compute_f1_max, compute_auprc
from utils.logger import TrainingLogger


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_normal_loader(config):
    data_cfg = config.get('data', {})
    train_cfg = config.get('train', {})
    dataset = CSVDataset(
        csv_path=data_cfg.get('train_csv', 'data/train.csv'),
        images_dir=data_cfg.get('images_dir', 'data/images'),
        split='train', image_size=data_cfg.get('image_size', 512),
        augment=train_cfg.get('augment', True),
    )
    normal_idx = [i for i, s in enumerate(dataset.samples) if s['label'] == 0]
    dataset = torch.utils.data.Subset(dataset, normal_idx)
    return DataLoader(dataset, batch_size=train_cfg.get('batch_size', 16),
                      shuffle=True, num_workers=data_cfg.get('num_workers', 4),
                      pin_memory=True, drop_last=True)


def build_test_loader(config):
    data_cfg = config.get('data', {})
    dataset = CSVDataset(
        csv_path=data_cfg.get('test_csv', 'data/test.csv'),
        images_dir=data_cfg.get('images_dir', 'data/images'),
        split='test', image_size=data_cfg.get('image_size', 512), augment=False,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False,
                      num_workers=data_cfg.get('num_workers', 4), pin_memory=True)


# ---- Ablation Model Variants ----

class PatchEmbedding(nn.Module):
    """Fixed-grid 16×16 patch embedding (ViT-style, replaces SLIC)."""
    def __init__(self, d_model=256):
        super().__init__()
        self.proj = nn.Conv2d(3, d_model, kernel_size=16, stride=16)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, d_model)


class SimpleBidirectionalMamba(nn.Module):
    """Plain bidirectional Mamba without topology-constrained attention."""
    def __init__(self, d_model=256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # Simplified: use a 1D conv as SSM proxy
        self.scan_fwd = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU(),
        )
        self.scan_bwd = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU(),
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, L, D)
        residual = x
        x_norm = self.norm(x)

        # Forward scan
        out_fwd = self.scan_fwd(x_norm.transpose(1, 2)).transpose(1, 2)
        # Backward scan
        out_bwd = self.scan_bwd(x_norm.flip([1]).transpose(1, 2)).transpose(1, 2).flip([1])

        return residual + self.proj(out_fwd + out_bwd)


class SimplePixelHead(nn.Module):
    """Simple reconstruction head."""
    def __init__(self, d_model=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, 512), nn.GELU(),
            nn.Linear(512, 1024), nn.GELU(),
            nn.Linear(1024, 16 * 16 * 3),
        )

    def forward(self, tokens, M, N):
        B, L, D = tokens.shape
        pixels = self.proj(tokens)
        pixels = pixels.reshape(B, M, N, 16, 16, 3).permute(0, 5, 1, 3, 2, 4)
        return pixels.reshape(B, 3, M * 16, N * 16)


class AblationModel(nn.Module):
    """TopoVarAD ablation variant."""
    def __init__(self, variant='full', d_model=256):
        super().__init__()
        self.variant = variant
        self.d_model = d_model

        # Tokenizer
        if variant == 'no_slic':
            self.use_slic = False
            self.patch_embed = PatchEmbedding(d_model)
            self.token_proj = nn.Identity()
        else:
            self.use_slic = True
            self.input_proj = nn.Conv2d(3, d_model, 3, 1, 1)
            from models.t2m_tokenizer import T2MTokenizer
            self.tokenizer = T2MTokenizer(d_model, (50, 100, 200))

        # TPM
        if variant == 'no_tpm':
            self.tpm = SimpleBidirectionalMamba(d_model)
        else:
            from models.tpm_block import TPMBlock
            self.tpm = TPMBlock(d_model=d_model, n_layers=3, n_heads=8)

        # Position encoding
        self.use_glpe = (variant != 'no_glpe')
        if not self.use_glpe:
            self.learned_pe = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)

        self.pixel_head = SimplePixelHead(d_model)

    def forward(self, x):
        B = x.shape[0]

        if self.use_slic:
            feat = self.input_proj(x)
            tokens, masks, counts = self.tokenizer(feat)
            L = tokens.shape[1]
        else:
            tokens = self.patch_embed(x)
            L = tokens.shape[1]
            counts = [L]

        total_tokens = tokens.shape[1]
        M = int(np.ceil(np.sqrt(total_tokens)))
        N = int(np.ceil(total_tokens / M))
        pad_len = M * N - total_tokens
        if pad_len > 0:
            tokens = torch.cat([tokens, torch.zeros(B, pad_len, self.d_model, device=tokens.device)], dim=1)

        # Position encoding
        if not self.use_glpe:
            tokens = tokens + self.learned_pe[:, :tokens.shape[1], :]

        # TPM
        sp_labels = np.arange(M * N)
        refined = self.tpm(tokens, sp_labels, M, N)

        # Reconstruction
        x_recon = self.pixel_head(refined, M, N)
        x_resized = F.interpolate(x, size=x_recon.shape[2:], mode='bilinear', align_corners=False)

        return x_recon, x_resized


def train_variant(model, loader, device, epochs=50, lr=1e-4):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    pbar = tqdm(range(epochs), desc=f'Training {model.variant}')
    for epoch in pbar:
        total_loss = 0.0
        for batch in loader:
            images = batch['image'].to(device)
            with autocast('cuda'):
                x_recon, x_resized = model(images)
                loss = F.l1_loss(x_recon, x_resized)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
        scheduler.step()
        pbar.set_postfix({'loss': f'{total_loss/len(loader):.4f}'})
    return model


@torch.no_grad()
def evaluate_variant(model, loader, device):
    model.eval()
    scores, labels = [], []
    for batch in tqdm(loader, desc='Evaluating'):
        images = batch['image'].to(device)
        x_recon, x_resized = model(images)
        error = torch.abs(x_recon - x_resized).reshape(images.shape[0], -1).mean(dim=1)
        scores.extend(error.cpu().tolist())
        labels.extend(batch['label'].tolist())
    scores, labels = np.array(scores), np.array(labels)
    return {
        'I-AUROC': compute_auroc(scores, labels),
        'I-F1max': compute_f1_max(scores, labels)[0],
        'I-AU-PR': compute_auprc(scores, labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--variants', nargs='+',
                        default=['full', 'no_slic', 'no_tpm', 'no_glpe'],
                        help='Ablation variants to test')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output', type=str, default='logs/ablation_results.json')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_loader = build_normal_loader(config)
    test_loader = build_test_loader(config)

    all_results = {}
    for variant in args.variants:
        print(f"\n{'='*50}\n  Ablation: {variant}\n{'='*50}")
        model = AblationModel(variant=variant).to(device)
        model = train_variant(model, train_loader, device, epochs=args.epochs)
        results = evaluate_variant(model, test_loader, device)
        all_results[variant] = results
        print(f"  {variant}: AUROC={results['I-AUROC']:.4f}  F1max={results['I-F1max']:.4f}")

    import json
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'AUROC':>10} {'F1max':>10} {'AU-PR':>10}")
    for v, r in all_results.items():
        print(f"{v:<20} {r['I-AUROC']:>10.4f} {r['I-F1max']:>10.4f} {r['I-AU-PR']:>10.4f}")


if __name__ == '__main__':
    main()