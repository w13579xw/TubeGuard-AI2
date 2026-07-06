"""
Token-level RQ-VAE 诊断
测试：现有 codebook 用于 token-level 量化的效果
包括：AUROC、聚类质量、可视化定位能力
"""
import os, sys, argparse, json, yaml
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
def extract_token_features(model, loader, device):
    """提取每张图的 token-level 特征、codes 和量化距离"""
    model.eval()
    all_token_dists = []       # (N, L)
    all_token_codes = []       # (N, L, n_q)
    all_z_global = []          # (N, D)
    all_image_scores_mean = [] # (N,)
    all_image_scores_max = []
    all_image_scores_topk = []
    all_labels = []
    all_paths = []
    M_ref, N_ref = None, None

    for batch in tqdm(loader, desc='Extracting'):
        x = batch['image'].to(device)
        tokens, sp_labels, M, N = model._tokenize(x)
        refined = model.tpm(tokens, sp_labels, M, N)  # (B, L, D)
        B, L, D = refined.shape
        if M_ref is None:
            M_ref, N_ref = M, N

        # Token-level 量化
        refined_flat = refined.reshape(B * L, D)
        z_hat_flat, codes_flat, _, _ = model.rqvae(refined_flat)
        token_dist = F.mse_loss(z_hat_flat, refined_flat, reduction='none').mean(dim=-1)
        token_dist = token_dist.reshape(B, L)  # (B, L)
        codes_bl = codes_flat.reshape(B, L, -1)  # (B, L, n_q)

        # 三种聚合方式
        score_mean = token_dist.mean(dim=1)
        score_max = token_dist.max(dim=1)[0]
        k = max(1, int(L * 0.2))
        topk_vals = token_dist.topk(k, dim=1)[0]
        score_topk = topk_vals.mean(dim=1)

        # Image-level 参考
        z_global = model.pool_head(refined)

        all_token_dists.append(token_dist.cpu().numpy())
        all_token_codes.append(codes_bl.cpu().numpy())
        all_z_global.append(z_global.cpu().numpy())
        all_image_scores_mean.extend(score_mean.cpu().numpy().tolist())
        all_image_scores_max.extend(score_max.cpu().numpy().tolist())
        all_image_scores_topk.extend(score_topk.cpu().numpy().tolist())
        all_labels.extend(batch['label'].numpy().tolist())
        all_paths.extend(batch['image_path'])

    return {
        'token_dists': np.concatenate(all_token_dists, axis=0),  # (N, L)
        'token_codes': np.concatenate(all_token_codes, axis=0),  # (N, L, n_q)
        'z_global': np.concatenate(all_z_global, axis=0),
        'image_scores_mean': np.array(all_image_scores_mean),
        'image_scores_max': np.array(all_image_scores_max),
        'image_scores_topk': np.array(all_image_scores_topk),
        'labels': np.array(all_labels),
        'paths': all_paths,
        'M': M_ref, 'N': N_ref,
    }


def compare_image_level(data):
    print("\n" + "="*80)
    print("对比 1：Token-level vs Global-level 图像分数")
    print("="*80)

    labels = data['labels']
    for name in ['image_scores_mean', 'image_scores_max', 'image_scores_topk']:
        s = data[name]
        auroc = compute_auroc(s, labels)
        aupr = compute_auprc(s, labels)
        f1max, _ = compute_f1_max(s, labels)
        m0 = s[labels == 0].mean()
        m1 = s[labels == 1].mean()
        direction = '✅' if m1 > m0 else '⚠️反了'
        print(f'{name:25s} | AUROC={auroc:.4f} | AU-PR={aupr:.4f} | F1max={f1max:.4f} | '
              f'normal={m0:.4f} defect={m1:.4f} | {direction}')


def analyze_token_diversity(data):
    """分析 token codes 的多样性和判别力"""
    print("\n" + "="*80)
    print("对比 2：Token codes 多样性 & z_global 判别力")
    print("="*80)

    from sklearn.metrics.pairwise import cosine_similarity

    labels = data['labels']
    z = data['z_global']
    codes = data['token_codes']  # (N, L, n_q)
    N, L, n_q = codes.shape

    # z_global 的相似度分布
    sample_idx = np.random.RandomState(42).choice(N, min(50, N), replace=False)
    sims = cosine_similarity(z[sample_idx])
    # 去对角
    mask = ~np.eye(len(sample_idx), dtype=bool)
    print(f"z_global 相似度：mean={sims[mask].mean():.4f}, std={sims[mask].std():.4f}, "
          f"min={sims[mask].min():.4f}, max={sims[mask].max():.4f}")

    # Token codes 多样性
    codes_flat = codes.reshape(-1, n_q)  # (N*L, n_q)
    unique_codes = set(map(tuple, codes_flat.tolist()))
    print(f"Token codes 独特组合数：{len(unique_codes)} / 总 tokens {len(codes_flat)}")

    # 各量化层的 code 使用数
    for q in range(n_q):
        unique_at_q = len(set(codes_flat[:, q].tolist()))
        print(f"  Q{q}: 使用了 {unique_at_q} 个不同 codes")


def spatial_localization_analysis(data, out_dir):
    """空间定位能力：查看异常样本的 token dist 是否集中在某些区域"""
    print("\n" + "="*80)
    print("对比 3：空间定位能力（Token-level 独有）")
    print("="*80)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = data['labels']
    token_dists = data['token_dists']  # (N, L)
    M, N = data['M'], data['N']

    # 找出几个高分异常样本，可视化其 token dist 热图
    defect_idx = np.where(labels == 1)[0]
    normal_idx = np.where(labels == 0)[0]

    # 按 image_scores_topk 排序，选出 4 个最异常的样本
    scores = data['image_scores_topk']
    top_defects = defect_idx[np.argsort(-scores[defect_idx])[:4]]
    top_normals = normal_idx[np.argsort(-scores[normal_idx])[:2]]  # 高分正常（可能误检）
    low_normals = normal_idx[np.argsort(scores[normal_idx])[:2]]   # 低分正常（正确识别）

    samples = list(top_defects) + list(top_normals) + list(low_normals)
    titles = (['High-score Defect'] * 4 + ['High-score Normal'] * 2 + ['Low-score Normal'] * 2)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, idx, title in zip(axes.flatten(), samples, titles):
        heatmap = token_dists[idx].reshape(M, N)
        im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
        ax.set_title(f'{title}\n{os.path.basename(data["paths"][idx])}\nscore={scores[idx]:.4f}',
                     fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'token_dist_heatmaps.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 热图保存到 {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', default='logs/stage2_token_level')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
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
    print(f'Test samples: {len(dataset)}')

    model = build_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.set_stage(2)
    print(f'Loaded {args.checkpoint}')

    data = extract_token_features(model, loader, device)
    print(f"\nToken dists shape: {data['token_dists'].shape}")
    print(f"Token codes shape: {data['token_codes'].shape}")
    print(f"Spatial grid: M={data['M']}, N={data['N']}, L={data['M']*data['N']}")

    compare_image_level(data)
    analyze_token_diversity(data)
    spatial_localization_analysis(data, args.output_dir)

    # 保存数据
    np.savez(os.path.join(args.output_dir, 'token_features.npz'),
             token_dists=data['token_dists'], token_codes=data['token_codes'],
             z_global=data['z_global'], labels=data['labels'])

    # 保存 image-level 结果
    labels = data['labels']
    results = {}
    for name in ['image_scores_mean', 'image_scores_max', 'image_scores_topk']:
        s = data[name]
        results[name] = {
            'AUROC': float(compute_auroc(s, labels)),
            'AU-PR': float(compute_auprc(s, labels)),
            'F1max': float(compute_f1_max(s, labels)[0]),
        }
    with open(os.path.join(args.output_dir, 'token_level_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 完成！结果保存在 {args.output_dir}")

if __name__ == '__main__':
    main()
