"""
诊断 Stage2 训练中评估异常问题
比较 3 种评估方式：
1. Stage1 test 用 predict()（stage=1）
2. 加载 Stage1 ckpt + predict(stage=2)
3. 加载 stage2_best ckpt + predict(stage=2)
"""
import os, sys, yaml, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import CSVDataset
from models.topovarad import TopoVarADConfig
from utils.metrics import compute_auroc, compute_f1_max, compute_auprc


def build_model(config, device):
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
    return topo_config.build_model().to(device)


def score_dataset(model, loader, device, stage):
    model.set_stage(stage)
    model.eval()
    scores, labels = [], []
    for batch in tqdm(loader, desc=f'stage={stage}'):
        images = batch['image'].to(device)
        image_scores, _ = model.predict(images)
        scores.extend(image_scores.cpu().numpy().tolist())
        labels.extend(batch['label'].numpy().astype(int).tolist())
    return np.asarray(scores), np.asarray(labels)


def summarize(scores, labels, tag):
    auroc = compute_auroc(scores, labels)
    aupr = compute_auprc(scores, labels)
    f1max, thresh = compute_f1_max(scores, labels)
    print(f'\n[{tag}]')
    print(f'  Scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}')
    print(f'  Normal scores (label=0): mean={scores[labels==0].mean():.4f}, std={scores[labels==0].std():.4f}')
    print(f'  Defect scores (label=1): mean={scores[labels==1].mean():.4f}, std={scores[labels==1].std():.4f}')
    print(f'  AUROC: {auroc:.4f}, AU-PR: {aupr:.4f}, F1max: {f1max:.4f} @ threshold={thresh:.4f}')
    # 如果 defect < normal，方向反了
    if scores[labels==1].mean() < scores[labels==0].mean():
        print('  ⚠️ WARNING: 异常样本分数 < 正常样本分数（方向反了！）')
        print(f'  Reversed AUROC (1-AUROC): {1-auroc:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--stage1_ckpt', default='checkpoints/stage1_best.pth')
    parser.add_argument('--stage2_ckpt', default='checkpoints/stage2_best.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config.get('data', {})
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset = CSVDataset(
        csv_path=data_cfg.get('test_csv'),
        images_dir=data_cfg.get('images_dir'),
        split='test', image_size=data_cfg.get('image_size', 512),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=data_cfg.get('num_workers', 4), pin_memory=True)
    print(f'Test samples: {len(dataset)}')

    # ---- 测试 1：加载 stage1_best.pth，用 predict(stage=1) ----
    print('\n========== 测试 1：Stage1 ckpt + predict(stage=1) ==========')
    model = build_model(config, device)
    ckpt = torch.load(args.stage1_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f'Loaded {args.stage1_ckpt}, epoch={ckpt.get("epoch", "?")}')
    scores, labels = score_dataset(model, loader, device, stage=1)
    summarize(scores, labels, 'Stage1 ckpt + predict(stage=1)')

    # ---- 测试 2：加载 stage1_best.pth，用 predict(stage=2) ----
    print('\n========== 测试 2：Stage1 ckpt + predict(stage=2) ==========')
    scores, labels = score_dataset(model, loader, device, stage=2)
    summarize(scores, labels, 'Stage1 ckpt + predict(stage=2)')

    # ---- 测试 3：加载 stage2_best.pth（如存在），用 predict(stage=2) ----
    if os.path.exists(args.stage2_ckpt):
        print('\n========== 测试 3：Stage2 ckpt + predict(stage=2) ==========')
        model2 = build_model(config, device)
        ckpt = torch.load(args.stage2_ckpt, map_location=device)
        model2.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f'Loaded {args.stage2_ckpt}, epoch={ckpt.get("epoch", "?")}')
        scores, labels = score_dataset(model2, loader, device, stage=2)
        summarize(scores, labels, 'Stage2 ckpt + predict(stage=2)')

    print('\n完成。')

if __name__ == '__main__':
    main()
