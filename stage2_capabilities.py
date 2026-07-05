"""
Stage2 独特能力验证实验
证明 Stage2 提供了 Stage1 不具备的四大新能力：

1. 【异常聚类】用 codes 对异常做无监督聚类
2. 【异常检索】给定异常样本，找出相似的异常样本
3. 【可解释性】定位到哪个码本位置最异常
4. 【码本可视化】UMAP/t-SNE 降维展示正常 vs 异常的码分布

Stage1 只能输出标量分数，做不了任何一件事。
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
def extract_representations(model, loader, device):
    """提取每个样本的：z_global, codes, quantization_error"""
    model.eval()
    all_z, all_codes, all_dist, all_labels, all_paths = [], [], [], [], []

    for batch in tqdm(loader, desc='Extracting features'):
        x = batch['image'].to(device)
        tokens, sp_labels, M, N = model._tokenize(x)
        refined = model.tpm(tokens, sp_labels, M, N)
        z_global = model.pool_head(refined)
        z_hat, codes, _, _ = model.rqvae(z_global)
        rqvae_dist = F.mse_loss(z_hat, z_global, reduction='none').mean(dim=-1)

        all_z.append(z_global.cpu().numpy())
        all_codes.append(codes.cpu().numpy())
        all_dist.extend(rqvae_dist.cpu().numpy().tolist())
        all_labels.extend(batch['label'].numpy().tolist())
        all_paths.extend(batch['image_path'])

    return {
        'z_global': np.concatenate(all_z, axis=0),      # (N, D)
        'codes': np.concatenate(all_codes, axis=0),      # (N, n_quantizers)
        'rqvae_dist': np.array(all_dist),                # (N,)
        'labels': np.array(all_labels),                  # (N,)
        'paths': all_paths,
    }


def experiment_1_anomaly_clustering(data, out_dir):
    """
    实验 1：异常样本的无监督聚类
    如果 Stage2 学到了有意义的表示，同类异常的 codes 会相似
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    print("\n" + "="*80)
    print("实验 1：异常聚类（Stage2 独有）")
    print("="*80)

    anom_mask = data['labels'] == 1
    z_anom = data['z_global'][anom_mask]
    codes_anom = data['codes'][anom_mask]

    # 用 codes 做聚类（Stage2 独有的离散表示）
    codes_flat = codes_anom.reshape(len(codes_anom), -1).astype(float)
    print(f"异常样本数：{len(codes_anom)}, code 维度：{codes_flat.shape}")

    # 尝试不同聚类数
    results = {}
    for k in [3, 5, 7, 10]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(codes_flat)
        try:
            sil = silhouette_score(codes_flat, clusters)
            ch = calinski_harabasz_score(codes_flat, clusters)
        except Exception:
            sil, ch = -1, -1
        print(f"  K={k}: Silhouette={sil:.4f}, Calinski-Harabasz={ch:.2f}")
        results[k] = {'silhouette': float(sil), 'calinski_harabasz': float(ch),
                       'clusters': clusters.tolist()}

    # 找出最佳 K
    best_k = max(results.keys(), key=lambda k: results[k]['silhouette'])
    print(f"\n✅ 最佳聚类数 K={best_k}, Silhouette={results[best_k]['silhouette']:.4f}")

    with open(os.path.join(out_dir, 'exp1_clustering.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def experiment_2_anomaly_retrieval(data, out_dir, top_k=5):
    """
    实验 2：给定一个异常样本，检索最相似的异常样本
    使用 codes 的编辑距离 or z_global 的余弦相似度
    """
    print("\n" + "="*80)
    print("实验 2：异常样本检索（Stage2 独有）")
    print("="*80)

    anom_idx = np.where(data['labels'] == 1)[0]
    z_anom = data['z_global'][anom_idx]  # (N_anom, D)
    codes_anom = data['codes'][anom_idx]  # (N_anom, n_q)
    paths_anom = [data['paths'][i] for i in anom_idx]

    # 用 code 匹配（Stage2 独有）
    from scipy.spatial.distance import hamming
    from sklearn.metrics.pairwise import cosine_similarity

    # 随机选 5 个 query
    n_query = min(5, len(anom_idx))
    query_idx = np.random.RandomState(42).choice(len(anom_idx), n_query, replace=False)

    print(f"查询 {n_query} 个异常样本，找 Top-{top_k} 相似：")
    retrieval_results = []
    for q in query_idx:
        # 方法 1：codes hamming 距离（Stage2 独有）
        hamming_dists = np.array([
            hamming(codes_anom[q], codes_anom[j]) if j != q else np.inf
            for j in range(len(codes_anom))
        ])
        top_k_by_codes = np.argsort(hamming_dists)[:top_k]

        # 方法 2：z_global 余弦相似度（对比基线）
        sims = cosine_similarity(z_anom[q:q+1], z_anom)[0]
        sims[q] = -np.inf  # exclude self
        top_k_by_z = np.argsort(-sims)[:top_k]

        print(f"\n  Query {q}: {os.path.basename(paths_anom[q])}")
        print(f"    Top-{top_k} by codes (Stage2):")
        for r in top_k_by_codes:
            print(f"      - {os.path.basename(paths_anom[r])} (hamming={hamming_dists[r]:.3f})")
        print(f"    Top-{top_k} by z_global (baseline):")
        for r in top_k_by_z:
            print(f"      - {os.path.basename(paths_anom[r])} (sim={sims[r]:.3f})")

        retrieval_results.append({
            'query_idx': int(q),
            'query_path': paths_anom[q],
            'top_k_by_codes': [paths_anom[i] for i in top_k_by_codes],
            'top_k_by_z': [paths_anom[i] for i in top_k_by_z],
        })

    with open(os.path.join(out_dir, 'exp2_retrieval.json'), 'w') as f:
        json.dump(retrieval_results, f, indent=2)


def experiment_3_code_visualization(data, out_dir):
    """
    实验 3：正常 vs 异常的 code 分布可视化
    用 UMAP/t-SNE 降维展示
    """
    print("\n" + "="*80)
    print("实验 3：Code 分布可视化（Stage2 独有）")
    print("="*80)

    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ 需要安装 sklearn 和 matplotlib")
        return

    # 用 codes 做 t-SNE（Stage2 独有）
    codes_flat = data['codes'].reshape(len(data['codes']), -1).astype(float)
    print(f"对 {len(codes_flat)} 个样本做 t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    codes_2d = tsne.fit_transform(codes_flat)

    # 同时对 z_global 做 t-SNE（对比基线）
    tsne_z = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne_z.fit_transform(data['z_global'])

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, coords, title in [(axes[0], codes_2d, 'Stage2 codes (t-SNE)'),
                                (axes[1], z_2d, 'z_global (baseline)')]:
        for label, color, name in [(0, '#4A90E2', 'Normal'), (1, '#E24A4A', 'Defect')]:
            mask = data['labels'] == label
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=15, alpha=0.6, label=name)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'exp3_visualization.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 可视化保存到 {out_path}")

    # 也保存 npz 供后续复用
    np.savez(os.path.join(out_dir, 'exp3_tsne_coords.npz'),
             codes_2d=codes_2d, z_2d=z_2d, labels=data['labels'])


def experiment_4_interpretability(data, out_dir):
    """
    实验 4：可解释性分析
    对每个异常样本，找出"哪个 code 位置最偏离正常分布"
    """
    print("\n" + "="*80)
    print("实验 4：可解释性——异常码位置定位（Stage2 独有）")
    print("="*80)

    normal_codes = data['codes'][data['labels'] == 0]  # (N_n, n_q)
    anom_codes = data['codes'][data['labels'] == 1]     # (N_a, n_q)
    n_quantizers = normal_codes.shape[1]

    # 对每个 quantizer 层，统计正常样本的 code 频率分布
    from collections import Counter
    normal_dists = []
    for q in range(n_quantizers):
        counter = Counter(normal_codes[:, q])
        total = sum(counter.values())
        normal_dists.append({k: v/total for k, v in counter.items()})

    # 对每个异常样本，计算每个位置的"异常度"（1 - P_normal(code)）
    anomaly_per_pos = np.zeros((len(anom_codes), n_quantizers))
    for i, sample_codes in enumerate(anom_codes):
        for q in range(n_quantizers):
            p_normal = normal_dists[q].get(sample_codes[q], 0.0)
            anomaly_per_pos[i, q] = 1.0 - p_normal

    # 平均异常度 per position
    avg_anomaly_per_pos = anomaly_per_pos.mean(axis=0)
    print(f"每个 quantizer 层的平均异常度（越高越能区分异常）：")
    for q in range(n_quantizers):
        bar = '█' * int(avg_anomaly_per_pos[q] * 40)
        print(f"  Q{q}: {avg_anomaly_per_pos[q]:.4f} {bar}")

    # 找出对异常检测最敏感的层
    top_layers = np.argsort(-avg_anomaly_per_pos)[:3]
    print(f"\n✅ 最能区分异常的 3 个量化层：Q{top_layers[0]}, Q{top_layers[1]}, Q{top_layers[2]}")

    result = {
        'avg_anomaly_per_position': avg_anomaly_per_pos.tolist(),
        'top_discriminative_layers': [int(x) for x in top_layers],
    }
    with open(os.path.join(out_dir, 'exp4_interpretability.json'), 'w') as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', default='logs/stage2_capabilities')
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

    # 加载模型
    model = build_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.set_stage(2)
    print(f'Loaded {args.checkpoint}')

    # 提取特征
    data = extract_representations(model, loader, device)
    print(f"z_global shape: {data['z_global'].shape}")
    print(f"codes shape: {data['codes'].shape}")
    print(f"Normal: {(data['labels']==0).sum()}, Defect: {(data['labels']==1).sum()}")

    # 保存原始特征供后续复用
    np.savez(os.path.join(args.output_dir, 'features.npz'),
             z_global=data['z_global'], codes=data['codes'],
             rqvae_dist=data['rqvae_dist'], labels=data['labels'])

    # 运行 4 个实验
    experiment_1_anomaly_clustering(data, args.output_dir)
    experiment_2_anomaly_retrieval(data, args.output_dir)
    experiment_3_code_visualization(data, args.output_dir)
    experiment_4_interpretability(data, args.output_dir)

    print("\n" + "="*80)
    print(f"✅ 所有实验完成！结果保存在 {args.output_dir}")
    print("="*80)
    print("\n实验 1: exp1_clustering.json - 异常聚类质量")
    print("实验 2: exp2_retrieval.json - 异常检索结果")
    print("实验 3: exp3_visualization.png - t-SNE 可视化")
    print("实验 4: exp4_interpretability.json - 可解释性分析")


if __name__ == '__main__':
    main()
