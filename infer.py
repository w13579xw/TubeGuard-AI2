import os
import sys
import argparse
import yaml
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.topovarad import TopoVarAD, TopoVarADConfig
from data.dataset import MVTecDataset
from utils.metrics import MetricsCalculator
from utils.visualize import (
    normalize_score_map,
    overlay_heatmap,
    draw_defect_contours,
    visualize_prediction,
    batch_visualize,
)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config, checkpoint_path, device):
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

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        print("Warning: No checkpoint loaded, using random weights")

    model.set_stage(2)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device, output_dir=None):
    """
    运行推理，返回所有样本的异常分数和可视化结果。
    """
    metrics = MetricsCalculator()
    all_results = []

    for batch in tqdm(loader, desc='Inference'):
        images = batch['image'].to(device)
        labels = batch['label'].numpy()
        masks = batch['mask'].numpy()
        paths = batch['image_path']

        image_scores, pixel_scores = model.predict(images)
        img_scores_np = image_scores.cpu().numpy()
        px_scores_np = pixel_scores.cpu().numpy()

        metrics.update(img_scores_np, labels, px_scores_np, masks)

        for i in range(images.shape[0]):
            result = {
                'image_path': paths[i],
                'image_score': float(img_scores_np[i]),
                'label': int(labels[i]),
                'pixel_score': px_scores_np[i],
                'gt_mask': masks[i],
            }
            all_results.append(result)

    eval_results = metrics.compute()
    return all_results, eval_results


def save_results(all_results, eval_results, output_dir, original_size=None):
    """保存推理结果：异常分数CSV + 可视化图。"""
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'scores.csv')
    with open(csv_path, 'w') as f:
        f.write('image_path,image_score,label\n')
        for r in all_results:
            f.write(f"{r['image_path']},{r['image_score']:.6f},{r['label']}\n")
    print(f"Scores saved to: {csv_path}")

    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        for k, v in eval_results.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Metrics saved to: {metrics_path}")

    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    for k, v in eval_results.items():
        print(f"  {k:>12s}: {v:.4f}")
    print(f"{'='*50}")


def visualize_single(image_path, pixel_score, gt_mask=None, save_path=None):
    """可视化单张图像的异常检测结果。"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Cannot read image {image_path}")
        return

    score_norm = normalize_score_map(pixel_score)

    if score_norm.shape[:2] != image.shape[:2]:
        score_norm = cv2.resize(score_norm, (image.shape[1], image.shape[0]))

    overlay = overlay_heatmap(image, score_norm, alpha=0.4)
    contour_img = draw_defect_contours(image, score_norm, threshold=0.5)

    n_cols = 3 if gt_mask is None else 4
    canvas = np.concatenate([image, overlay, contour_img], axis=1)

    if gt_mask is not None:
        gt_colored = cv2.cvtColor((gt_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        canvas = np.concatenate([canvas, gt_colored], axis=1)

    if save_path:
        cv2.imwrite(save_path, canvas)

    return canvas


def run_demo(model, image_path, device, output_path=None):
    """单张图像推理demo。"""
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    image_scores, pixel_scores = model.predict(image_tensor)

    img_score = image_scores[0].item()
    px_score = pixel_scores[0].cpu().numpy()

    print(f"Image: {image_path}")
    print(f"Anomaly Score: {img_score:.4f}")

    canvas = visualize_single(image_path, px_score, save_path=output_path)
    if output_path:
        print(f"Visualization saved to: {output_path}")

    return img_score, px_score


def main():
    parser = argparse.ArgumentParser(description='TopoVarAD Inference')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'demo'])
    parser.add_argument('--image', type=str, default=None, help='Single image path for demo mode')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(config, args.checkpoint, device)

    if args.mode == 'demo':
        if args.image is None:
            print("Error: --image is required in demo mode")
            sys.exit(1)
        output_path = os.path.join(args.output, 'demo_result.png')
        os.makedirs(args.output, exist_ok=True)
        run_demo(model, args.image, device, output_path)

    else:
        data_config = config.get('data', {})
        test_dataset = MVTecDataset(
            root=data_config.get('dataset_path', 'data/mvtec'),
            category=args.category,
            split='test',
            image_size=data_config.get('image_size', 512),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
        )

        print(f"Test samples: {len(test_dataset)}")

        all_results, eval_results = run_inference(model, test_loader, device, args.output)

        output_dir = os.path.join(args.output, args.category)
        save_results(all_results, eval_results, output_dir)

        for r in tqdm(all_results[:10], desc='Saving visualizations'):
            fname = os.path.splitext(os.path.basename(r['image_path']))[0]
            save_path = os.path.join(output_dir, 'visualizations', f'{fname}.png')
            visualize_single(
                r['image_path'],
                r['pixel_score'],
                r['gt_mask'] if r['gt_mask'].sum() > 0 else None,
                save_path,
            )

        print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
