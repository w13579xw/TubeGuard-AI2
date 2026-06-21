"""
Validate whether TopoVarAD Stage1's 100% precision transfers beyond the
same-test-label F1-max threshold.

Modes:
  - normal_quantile: choose threshold from normal training scores only.
  - split_test: choose threshold on a stratified calibration split, evaluate on held-out split.
  - kfold: stratified K-fold threshold transfer; each fold is held out once.
  - external: choose threshold on calib_csv, evaluate on eval_csv.

Outputs: metrics.json, score CSV files, and run.log under --output_dir.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import CSVDataset, MVTecDataset
from models.topovarad import TopoVarADConfig
from utils.metrics import compute_f1_max


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(config, checkpoint, device):
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
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.set_stage(1)
    model.eval()
    print(f"Loaded Stage1 checkpoint: {checkpoint}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}, loss: {ckpt.get('loss', '?')}")
    return model


def build_dataset(config, csv_path, split='test'):
    data_config = config.get('data', {})
    if data_config.get('dataset_type', 'csv') == 'mvtec':
        return MVTecDataset(
            root=data_config.get('dataset_path', 'data/mvtec'),
            category=data_config.get('category', 'bottle'),
            split=split,
            image_size=data_config.get('image_size', 512),
        )
    return CSVDataset(
        csv_path=csv_path,
        images_dir=data_config.get('images_dir', 'data/images'),
        split=split,
        image_size=data_config.get('image_size', 512),
        augment=False,
    )


def build_train_normal_dataset(config):
    data_config = config.get('data', {})
    dataset = CSVDataset(
        csv_path=data_config.get('train_csv', 'data/train.csv'),
        images_dir=data_config.get('images_dir', 'data/images'),
        split='train',
        image_size=data_config.get('image_size', 512),
        augment=False,
    )
    normal_idx = [i for i, s in enumerate(dataset.samples) if s['label'] == 0]
    print(f"Normal calibration samples from train_csv: {len(normal_idx)} / {len(dataset)}")
    return Subset(dataset, normal_idx)


@torch.no_grad()
def score_dataset(model, dataset, device, batch_size=1, num_workers=4, score_method='predict'):
    # batch_size=1 is intentional. In this codebase the superpixel tokenizer can
    # mix token pools across images for larger batches, which changes scores.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    scores, labels, paths = [], [], []
    for batch in tqdm(loader, desc=f'Scoring ({score_method})'):
        images = batch['image'].to(device)
        if score_method == 'predict':
            # Autoregressive negative-log-likelihood score (matches the
            # 0.9788 headline produced by test_stage1.py / model.predict()).
            image_scores, _ = model.predict(images)
            img_scores = image_scores
        else:
            # Reconstruction L1 error score.
            outputs = model(images)
            error = torch.abs(outputs['reconstructed'] - outputs['x_resized'])
            img_scores = error.reshape(images.shape[0], -1).mean(dim=1)
        scores.extend(img_scores.detach().cpu().numpy().tolist())
        labels.extend(batch['label'].numpy().astype(int).tolist())
        paths.extend(batch['image_path'])
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    print(f"Score stats ({score_method}): min={scores.min():.6f}, max={scores.max():.6f}, "
          f"mean={scores.mean():.6f}, AUROC={roc_auc_score(labels, scores):.4f}"
          if len(np.unique(labels)) == 2 else f"Score stats ({score_method}): single-class labels")
    return scores, labels, paths


def compute_metrics(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold),
        'I-AUROC': float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float('nan'),
        'I-AU-PR': float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else float('nan'),
        'Accuracy': float(accuracy_score(labels, preds)),
        'Precision': float(precision_score(labels, preds, zero_division=0)),
        'Recall': float(recall_score(labels, preds, zero_division=0)),
        'F1-Score': float(f1_score(labels, preds, zero_division=0)),
        'Specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'True_Negative': int(tn),
        'False_Positive': int(fp),
        'False_Negative': int(fn),
        'True_Positive': int(tp),
    }


def save_scores(path, scores, labels, image_paths):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label', 'score'])
        for p, y, s in zip(image_paths, labels, scores):
            writer.writerow([p, int(y), float(s)])


def stratified_split_indices(labels, calib_ratio, seed):
    rng = random.Random(seed)
    labels = np.asarray(labels)
    calib_idx, eval_idx = [], []
    for cls in [0, 1]:
        idx = np.where(labels == cls)[0].tolist()
        rng.shuffle(idx)
        n_calib = max(1, int(round(len(idx) * calib_ratio)))
        calib_idx.extend(idx[:n_calib])
        eval_idx.extend(idx[n_calib:])
    rng.shuffle(calib_idx)
    rng.shuffle(eval_idx)
    return np.asarray(calib_idx), np.asarray(eval_idx)


def summarize(result_list):
    summary = {}
    keys = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity',
            'False_Positive', 'False_Negative', 'True_Positive', 'True_Negative']
    for key in keys:
        vals = np.asarray([r[key] for r in result_list], dtype=np.float64)
        summary[f'{key}_mean'] = float(vals.mean())
        summary[f'{key}_std'] = float(vals.std(ddof=0))
        summary[f'{key}_min'] = float(vals.min())
        summary[f'{key}_max'] = float(vals.max())
    return summary


def main_impl(args):
    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    data_config = config.get('data', {})
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")

    model = build_model(config, args.checkpoint, device)
    num_workers = data_config.get('num_workers', 4)
    results = {
        'mode': args.mode,
        'score_method': args.score_method,
        'checkpoint': args.checkpoint,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    if args.mode == 'normal_quantile':
        calib_dataset = build_train_normal_dataset(config)
        calib_scores, calib_labels, calib_paths = score_dataset(
            model, calib_dataset, device, args.batch_size, num_workers, args.score_method)
        threshold = float(np.quantile(calib_scores, args.normal_quantile))
        print(f"Normal-only threshold q={args.normal_quantile}: {threshold:.8f}")

        eval_csv = args.eval_csv or data_config.get('test_csv', 'data/test.csv')
        eval_dataset = build_dataset(config, eval_csv, split='test')
        eval_scores, eval_labels, eval_paths = score_dataset(
            model, eval_dataset, device, args.batch_size, num_workers, args.score_method)
        metrics = compute_metrics(eval_scores, eval_labels, threshold)
        save_scores(os.path.join(args.output_dir, 'calib_normal_scores.csv'), calib_scores, calib_labels, calib_paths)
        save_scores(os.path.join(args.output_dir, 'eval_scores.csv'), eval_scores, eval_labels, eval_paths)
        results['normal_quantile'] = args.normal_quantile
        results['eval_metrics'] = metrics

    elif args.mode == 'split_test':
        eval_csv = args.eval_csv or data_config.get('test_csv', 'data/test.csv')
        dataset = build_dataset(config, eval_csv, split='test')
        scores, labels, paths = score_dataset(model, dataset, device, args.batch_size, num_workers, args.score_method)
        split_results = []
        for i in range(args.repeats):
            seed = args.seed + i
            calib_idx, eval_idx = stratified_split_indices(labels, args.calib_ratio, seed)
            f1max, threshold = compute_f1_max(scores[calib_idx], labels[calib_idx])
            metrics = compute_metrics(scores[eval_idx], labels[eval_idx], threshold)
            metrics.update({'seed': seed, 'calib_f1max': float(f1max),
                            'n_calib': int(len(calib_idx)), 'n_eval': int(len(eval_idx))})
            split_results.append(metrics)
            print(f"Seed {seed}: thr={threshold:.8f}, Precision={metrics['Precision']:.4f}, "
                  f"Recall={metrics['Recall']:.4f}, FP={metrics['False_Positive']}, FN={metrics['False_Negative']}")
        save_scores(os.path.join(args.output_dir, 'all_test_scores.csv'), scores, labels, paths)
        results['split_results'] = split_results
        results.update(summarize(split_results))

    elif args.mode == 'kfold':
        eval_csv = args.eval_csv or data_config.get('test_csv', 'data/test.csv')
        dataset = build_dataset(config, eval_csv, split='test')
        scores, labels, paths = score_dataset(model, dataset, device, args.batch_size, num_workers, args.score_method)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_results = []
        for fold, (calib_idx, eval_idx) in enumerate(skf.split(scores, labels), start=1):
            f1max, threshold = compute_f1_max(scores[calib_idx], labels[calib_idx])
            metrics = compute_metrics(scores[eval_idx], labels[eval_idx], threshold)
            metrics.update({'fold': fold, 'calib_f1max': float(f1max),
                            'n_calib': int(len(calib_idx)), 'n_eval': int(len(eval_idx))})
            fold_results.append(metrics)
            print(f"Fold {fold}/{args.folds}: thr={threshold:.8f}, Precision={metrics['Precision']:.4f}, "
                  f"Recall={metrics['Recall']:.4f}, FP={metrics['False_Positive']}, FN={metrics['False_Negative']}")
        save_scores(os.path.join(args.output_dir, 'all_test_scores.csv'), scores, labels, paths)
        results['fold_results'] = fold_results
        results.update(summarize(fold_results))

    elif args.mode == 'external':
        if not args.calib_csv or not args.eval_csv:
            raise ValueError('--calib_csv and --eval_csv are required for external mode')
        calib_dataset = build_dataset(config, args.calib_csv, split='test')
        eval_dataset = build_dataset(config, args.eval_csv, split='test')
        calib_scores, calib_labels, calib_paths = score_dataset(model, calib_dataset, device, args.batch_size, num_workers, args.score_method)
        eval_scores, eval_labels, eval_paths = score_dataset(model, eval_dataset, device, args.batch_size, num_workers, args.score_method)
        f1max, threshold = compute_f1_max(calib_scores, calib_labels)
        metrics = compute_metrics(eval_scores, eval_labels, threshold)
        save_scores(os.path.join(args.output_dir, 'calib_scores.csv'), calib_scores, calib_labels, calib_paths)
        save_scores(os.path.join(args.output_dir, 'eval_scores.csv'), eval_scores, eval_labels, eval_paths)
        results['calib_f1max'] = float(f1max)
        results['eval_metrics'] = metrics

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True))
    print(f"Saved metrics: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='TopoVarAD Stage1 threshold transfer validation')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--mode', choices=['normal_quantile', 'split_test', 'kfold', 'external'], default='kfold')
    parser.add_argument('--score_method', choices=['predict', 'reconstruction'], default='predict',
                        help='predict = autoregressive NLL (matches 0.9788 headline); reconstruction = L1 error')
    parser.add_argument('--output_dir', type=str, default='logs/stage1_threshold_validation')
    parser.add_argument('--calib_csv', type=str, default=None)
    parser.add_argument('--eval_csv', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--normal_quantile', type=float, default=0.995)
    parser.add_argument('--calib_ratio', type=float, default=0.5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--log_file', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(args.output_dir, 'run.log')
    with open(log_file, 'w', encoding='utf-8') as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee), redirect_stderr(tee):
            main_impl(args)


if __name__ == '__main__':
    main()
