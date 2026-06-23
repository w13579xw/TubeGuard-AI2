"""Evaluate a Stage2 TopoVarAD checkpoint on the configured test set."""

import argparse
import csv
import json
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
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
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.set_stage(2)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}, loss: {ckpt.get('loss', '?')}")
    return model


def build_dataset(config, split, csv_path=None):
    data_config = config.get('data', {})
    if data_config.get('dataset_type', 'csv') == 'mvtec':
        return MVTecDataset(
            root=data_config.get('dataset_path', 'data/mvtec'),
            category=data_config.get('category', 'bottle'),
            split=split,
            image_size=data_config.get('image_size', 512),
        )
    return CSVDataset(
        csv_path=csv_path or data_config.get(f'{split}_csv', f'data/{split}.csv'),
        images_dir=data_config.get('images_dir', 'data/images'),
        split=split,
        image_size=data_config.get('image_size', 512),
        augment=False,
    )


@torch.no_grad()
def score_dataset(model, loader, device):
    scores, labels, paths = [], [], []
    for batch in tqdm(loader, desc='Stage2 eval'):
        images = batch['image'].to(device)
        image_scores, _ = model.predict(images)
        scores.extend(image_scores.cpu().numpy().tolist())
        labels.extend(batch['label'].numpy().astype(int).tolist())
        paths.extend(batch['image_path'])
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64), paths


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


def main_impl(args):
    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = build_model(config, args.checkpoint, device)
    dataset = build_dataset(config, 'test', args.test_csv)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=config.get('data', {}).get('num_workers', 4), pin_memory=True)
    print(f"Test samples: {len(dataset)}")
    scores, labels, paths = score_dataset(model, loader, device)
    if args.threshold is None:
        f1max, threshold = compute_f1_max(scores, labels)
        print(f"F1-max threshold: {threshold:.8f}, F1max={f1max:.4f}")
    else:
        threshold = args.threshold
        print(f"Fixed threshold: {threshold:.8f}")
    metrics = compute_metrics(scores, labels, threshold)
    result = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'checkpoint': args.checkpoint, 'metrics': metrics}
    save_scores(os.path.join(args.output_dir, 'stage2_scores.csv'), scores, labels, paths)
    metrics_path = os.path.join(args.output_dir, 'stage2_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    print(f"Saved metrics: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Stage2 TopoVarAD')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='logs/stage2_eval')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_file', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(args.output_dir, 'eval.log')
    with open(log_file, 'w', encoding='utf-8') as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee), redirect_stderr(tee):
            main_impl(args)


if __name__ == '__main__':
    main()
