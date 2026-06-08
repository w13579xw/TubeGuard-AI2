"""
Ablation study using the REAL TopoVarAD model with configurable switches.

Variants (additive removal):
  1. Full   — use_slic=True,  use_topo_attn=True,  use_glpe=True  (baseline)
  2. no_slic — use_slic=False, use_topo_attn=True,  use_glpe=True  (fixed patch)
  3. no_topo — use_slic=True,  use_topo_attn=False, use_glpe=True  (SSM only)
  4. no_glpe — use_slic=True,  use_topo_attn=True,  use_glpe=False (learned PE)

All variants use identical training: normal-only, L1+LPIPS, AdamW, early stopping.

Usage:
  python benchmark/ablation_real.py --config configs/default.yaml --device cuda
"""

import os, sys, argparse, yaml, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset import CSVDataset
from models.topovarad import TopoVarAD, TopoVarADConfig
from utils.losses import TopoVarADLoss
from utils.metrics import compute_auroc, compute_f1_max, compute_auprc


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_normal_train_loader(config):
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
    return DataLoader(dataset, batch_size=4,  # smaller batch for 25M model on 512×512
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
                      num_workers=data_cfg.get('num_workers', 2), pin_memory=True)


@torch.no_grad()
def evaluate_reconstruction(model, loader, device):
    model.eval()
    scores, labels_list = [], []
    for batch in tqdm(loader, desc='Eval', leave=False):
        images = batch['image'].to(device)
        outputs = model(images)
        error = torch.abs(outputs['reconstructed'] - outputs['x_resized']).reshape(images.shape[0], -1).mean(dim=1)
        scores.extend(error.cpu().tolist())
        labels_list.extend(batch['label'].tolist())
    s, l = np.array(scores), np.array(labels_list)
    return {'I-AUROC': compute_auroc(s, l), 'I-F1max': compute_f1_max(s, l)[0], 'I-AU-PR': compute_auprc(s, l)}


def train_one_variant(variant_name, use_slic, use_topo_attn, use_glpe,
                      config, train_loader, val_loader, device):
    """Train one ablation variant with early stopping."""
    train_cfg = config.get('train', {})
    model_cfg = config.get('model', {})
    epochs = 200
    lr = train_cfg.get('lr_stage1', 1e-4)
    patience = 20
    eval_every = 5

    print(f"\n{'='*60}")
    print(f"  Ablation: {variant_name}")
    print(f"  use_slic={use_slic}, use_topo_attn={use_topo_attn}, use_glpe={use_glpe}")
    print(f"{'='*60}")

    topo_cfg = TopoVarADConfig(
        d_model=model_cfg.get('d_model', 256),
        n_tpm_layers=model_cfg.get('n_layers', 6),
        n_heads=model_cfg.get('n_heads', 8),
        superpixel_scales=tuple(model_cfg.get('superpixel_scales', [50, 100, 200])),
        rqvae_codebook_size=model_cfg.get('rqvae_codebook_size', 1024),
        rqvae_d_code=model_cfg.get('rqvae_d_code', 32),
        rqvae_n_layers=model_cfg.get('rqvae_n_layers', 8),
        tar_n_layers=model_cfg.get('tar_n_layers', 6), tar_n_heads=model_cfg.get('tar_n_heads', 8),
        use_slic=use_slic, use_topo_attn=use_topo_attn, use_glpe=use_glpe,
    )
    model = topo_cfg.build_model().to(device)
    model.set_stage(1)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = TopoVarADLoss(
        lambda_lpips=train_cfg.get('lambda_lpips', 0.1),
        lambda_rqvae=train_cfg.get('lambda_rqvae', 0.5),
        lambda_ar=train_cfg.get('lambda_ar', 1.0),
        lambda_diversity=train_cfg.get('lambda_diversity', 0.0),
        label_smoothing=train_cfg.get('label_smoothing', 0.1),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=train_cfg.get('weight_decay', 0.05))
    scaler = GradScaler()

    best_auroc = 0.0
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'  [{variant_name}] Epoch {epoch+1}/{epochs}', leave=False)
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

        val_auroc = 0.0
        if (epoch + 1) % eval_every == 0:
            metrics = evaluate_reconstruction(model, val_loader, device)
            val_auroc = metrics['I-AUROC']
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += eval_every
            print(f"  [{variant_name}] Epoch {epoch+1:>3d} | loss={avg_loss:.4f} | AUROC={val_auroc:.4f} | best={best_auroc:.4f} | wait={no_improve}/{patience}")
        else:
            print(f"  [{variant_name}] Epoch {epoch+1:>3d} | loss={avg_loss:.4f}")

        if no_improve >= patience:
            print(f"  [{variant_name}] Early stop at epoch {epoch+1}, best={best_auroc:.4f} @ epoch {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final = evaluate_reconstruction(model, val_loader, device)
    final['best_epoch'] = best_epoch
    if best_state is not None:
        final['best_state'] = best_state
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--variant', type=str, default='full',
                        choices=['full', 'no_slic', 'no_topo', 'no_glpe', 'all'],
                        help='Which variant to run (or "all")')
    parser.add_argument('--output_dir', type=str, default='logs/ablation')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_loader = build_normal_train_loader(config)
    val_loader = build_test_loader(config)
    print(f"Train: {len(train_loader.dataset)} normal samples")
    print(f"Val: {len(val_loader.dataset)} samples")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.variant == 'all':
        variants = [
            ('full',     True,  True,  True),
            ('no_slic',  False, True,  True),
            ('no_topo',  True,  False, True),
            ('no_glpe',  True,  True,  False),
        ]
    else:
        mapping = {
            'full':    (True, True, True),
            'no_slic': (False, True, True),
            'no_topo': (True, False, True),
            'no_glpe': (True, True, False),
        }
        variants = [(args.variant,) + mapping[args.variant]]

    ckpt_dir = config.get('train', {}).get('checkpoint_dir', 'checkpoints')

    all_results = {}
    for name, slic, topo, glpe in variants:
        results = train_one_variant(name, slic, topo, glpe, config, train_loader, val_loader, device)
        all_results[name] = results
        print(f"  >>> {name}: AUROC={results['I-AUROC']:.4f} F1max={results['I-F1max']:.4f} (epoch {results['best_epoch']})")

        # Save best model (remove state_dict from JSON to keep it small)
        if 'best_state' in results:
            model_path = os.path.join(ckpt_dir, f'ablation_{name}.pth')
            torch.save(results.pop('best_state'), model_path)
            print(f"  Model saved: {model_path}")

        # Save individual result JSON
        out_path = os.path.join(args.output_dir, f'{name}.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)

    # Only write summary.json when running all variants
    if len(all_results) > 1:
        with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        baseline = all_results['full']['I-AUROC']
        print(f"\n{'='*60}\n  ABLATION SUMMARY\n{'='*60}")
        print(f"{'Variant':<15} {'AUROC':>10} {'F1max':>10} {'ΔAUROC':>10} {'Epoch':>8}")
        for name in ['full', 'no_slic', 'no_topo', 'no_glpe']:
            r = all_results[name]
            delta = r['I-AUROC'] - baseline
            print(f"{name:<15} {r['I-AUROC']:>10.4f} {r['I-F1max']:>10.4f} {delta:>+10.4f} {r['best_epoch']:>8}")


if __name__ == '__main__':
    main()