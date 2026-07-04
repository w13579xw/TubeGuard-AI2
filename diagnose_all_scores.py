"""
综合诊断：一次前向传播，测试所有可能的 anomaly score 方案
避免多次跑 600 样本（每次 70 分钟）
"""
import os, sys, yaml, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import CSVDataset
from models.topovarad import TopoVarADConfig
from utils.metrics import compute_auroc, compute_f1_max, compute_auprc


def build_model(config, device):
    mc = config.get('model', {})
    return TopoVarADConfig(
        d_model=mc.get('d_model', 256),
        n_tpm_layers=mc.get('n_layers', 6),
        n_heads=mc.get('n_heads', 8),
        superpixel_scales=tuple(mc.get('superpixel_scales', [50, 100, 200])),
        rqvae_codebook_size=mc.get('rqvae_codebook_size', 1024),
        rqvae_d_code=mc.get('rqvae_d_code', 32),
        rqvae_n_layers=mc.get('rqvae_n_layers', 8),
        tar_n_layers=mc.get('tar_n_layers', 6),
        tar_n_heads=mc.get('tar_n_heads', 8),
    ).build_model().to(device)


@torch.no_grad()
def score_all_methods(model, loader, device):
    """一次前向传播，计算所有分数方案"""
    model.eval()
    all_scores = {
        'recon': [],           # 原始重建误差
        'neg_recon': [],       # 翻转的重建误差
        'ar': [],              # AR 似然分数
        'rqvae_dist': [],      # RQ-VAE 量化距离
        'combined_v1': [],     # 融合方案 1：AR + 翻转 recon
        'combined_v2': [],     # 融合方案 2：仅归一化后融合
    }
    labels = []

    for batch in tqdm(loader, desc='Computing all scores'):
        x = batch['image'].to(device)
        y = batch['label'].numpy().astype(int)
        B = x.shape[0]

        # 前向传播
        tokens, sp_labels, M, N = model._tokenize(x)
        refined = model.tpm(tokens, sp_labels, M, N)
        z_global = model.pool_head(refined)

        # 1. 重建分数
        x_recon = model.pixel_head(refined, M, N)
        H_target = M * 16
        W_target = N * 16
        x_resized = F.interpolate(x, size=(H_target, W_target), mode='bilinear', align_corners=False)
        recon_error = F.l1_loss(x_recon, x_resized, reduction='none').mean(dim=1)
        recon_score = recon_error.mean(dim=[1, 2])

        # 2. RQ-VAE 量化距离（重建 z 的误差）
        z_hat, codes, rqvae_loss, embeddings = model.rqvae(z_global)
        rqvae_dist = F.mse_loss(z_hat, z_global, reduction='none').mean(dim=-1)

        # 3. AR 似然分数
        token_scores, ar_score = model.tar.compute_anomaly_score(codes, z_global)

        # 存储各种分数
        all_scores['recon'].extend(recon_score.cpu().numpy().tolist())
        all_scores['neg_recon'].extend((-recon_score).cpu().numpy().tolist())
        all_scores['ar'].extend(ar_score.cpu().numpy().tolist())
        all_scores['rqvae_dist'].extend(rqvae_dist.cpu().numpy().tolist())
        # 融合分数（各自 z-score 归一化后求和）
        combined_v1 = -recon_score + ar_score  # 假设两者尺度差不多
        combined_v2 = ar_score  # 只用 AR
        all_scores['combined_v1'].extend(combined_v1.cpu().numpy().tolist())
        all_scores['combined_v2'].extend(combined_v2.cpu().numpy().tolist())

        labels.extend(y.tolist())

    labels = np.array(labels)
    return {k: np.array(v) for k, v in all_scores.items()}, labels


def summarize(scores, labels, tag):
    auroc = compute_auroc(scores, labels)
    aupr = compute_auprc(scores, labels)
    f1max, thresh = compute_f1_max(scores, labels)
    m0 = scores[labels == 0].mean()
    m1 = scores[labels == 1].mean()
    direction = '✅ 正常' if m1 > m0 else '⚠️ 反了'
    print(f'{tag:20s} | AUROC={auroc:.4f} | AU-PR={aupr:.4f} | F1max={f1max:.4f} | '
          f'normal={m0:.4f} defect={m1:.4f} | {direction}')


def compute_fusion(scores_dict, labels):
    """z-score 归一化后融合各种分数"""
    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    # 各种融合权重组合
    configs = {
        'neg_recon + ar (0.5:0.5)': (0.5, 0.5),
        'neg_recon + ar (0.7:0.3)': (0.7, 0.3),
        'neg_recon + ar (0.3:0.7)': (0.3, 0.7),
        'neg_recon + rqvae (0.5:0.5)': (0.5, 0.5, 'rqvae_dist'),
        'ar + rqvae (0.5:0.5)': (0.5, 0.5, 'ar', 'rqvae_dist'),
    }
    print('\n===== 融合分数（z-score 归一化后加权）=====')
    z_neg_recon = zscore(scores_dict['neg_recon'])
    z_ar = zscore(scores_dict['ar'])
    z_rqvae = zscore(scores_dict['rqvae_dist'])

    fusions = {
        'neg_recon+ar (0.5:0.5)': 0.5 * z_neg_recon + 0.5 * z_ar,
        'neg_recon+ar (0.7:0.3)': 0.7 * z_neg_recon + 0.3 * z_ar,
        'neg_recon+ar (0.3:0.7)': 0.3 * z_neg_recon + 0.7 * z_ar,
        'neg_recon+ar (0.9:0.1)': 0.9 * z_neg_recon + 0.1 * z_ar,
        'neg_recon+rqvae (0.5:0.5)': 0.5 * z_neg_recon + 0.5 * z_rqvae,
        'ar+rqvae (0.5:0.5)': 0.5 * z_ar + 0.5 * z_rqvae,
        'all 3 (0.33 each)': (z_neg_recon + z_ar + z_rqvae) / 3,
    }
    for name, s in fusions.items():
        summarize(s, labels, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset = CSVDataset(
        csv_path=config['data']['test_csv'],
        images_dir=config['data']['images_dir'],
        split='test', image_size=config['data'].get('image_size', 512),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=config['data'].get('num_workers', 4), pin_memory=True)
    print(f'Test samples: {len(dataset)}, batch_size={args.batch_size}')

    model = build_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.set_stage(2)
    print(f'Loaded {args.checkpoint}, epoch={ckpt.get("epoch", "?")}')

    scores_dict, labels = score_all_methods(model, loader, device)

    print('\n===== 单一分数对比 =====')
    print(f'{"Method":20s} | {"AUROC":8s} | {"AU-PR":8s} | {"F1max":8s} | Direction')
    print('-' * 100)
    for name in ['recon', 'neg_recon', 'ar', 'rqvae_dist']:
        summarize(scores_dict[name], labels, name)

    compute_fusion(scores_dict, labels)

    # 保存分数供后续分析
    import json
    output = {
        'checkpoint': args.checkpoint,
        'epoch': ckpt.get('epoch', '?'),
        'results': {}
    }
    for name in ['recon', 'neg_recon', 'ar', 'rqvae_dist']:
        s = scores_dict[name]
        output['results'][name] = {
            'AUROC': float(compute_auroc(s, labels)),
            'AU-PR': float(compute_auprc(s, labels)),
            'F1max': float(compute_f1_max(s, labels)[0]),
            'normal_mean': float(s[labels == 0].mean()),
            'defect_mean': float(s[labels == 1].mean()),
        }
    out_path = 'logs/diagnose_all_scores.json'
    os.makedirs('logs', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n💾 保存到 {out_path}')


if __name__ == '__main__':
    main()
