import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def normalize_score_map(score_map):
    """
    归一化异常分数图到 [0, 1]。
    score_map: ndarray (H, W)
    """
    s_min = score_map.min()
    s_max = score_map.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(score_map)
    return (score_map - s_min) / (s_max - s_min)


def score_to_heatmap(score_map, colormap='jet'):
    """
    将异常分数图转为热力图 (H, W, 3) uint8 BGR。
    score_map: ndarray (H, W) [0,1]
    """
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(score_map)[:, :, :3]
    return (heatmap * 255).astype(np.uint8)


def overlay_heatmap(image, score_map, alpha=0.4, colormap='jet'):
    """
    在原图上叠加异常热力图。
    image: ndarray (H, W, 3) BGR uint8
    score_map: ndarray (H, W) [0,1]
    alpha: 热力图透明度
    返回: 叠加后的图像 (H, W, 3) BGR uint8
    """
    heatmap = score_to_heatmap(score_map, colormap)
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    if image.shape[:2] != heatmap_bgr.shape[:2]:
        heatmap_bgr = cv2.resize(heatmap_bgr, (image.shape[1], image.shape[0]))

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlay


def draw_defect_contours(image, score_map, threshold=0.5, color=(0, 0, 255), thickness=2):
    """
    在异常区域绘制轮廓。
    image: ndarray (H, W, 3) BGR uint8
    score_map: ndarray (H, W) [0,1]
    threshold: 二值化阈值
    返回: 带轮廓的图像
    """
    result = image.copy()
    binary = (score_map >= threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, thickness)

    return result


def visualize_prediction(image, score_map, gt_mask=None, save_path=None, threshold=0.5):
    """
    可视化异常检测结果：原图 + 热力图 + 叠加图 + (可选)GT对比。
    image: ndarray (H, W, 3) BGR uint8 或 RGB uint8
    score_map: ndarray (H, W) [0,1]
    gt_mask: ndarray (H, W) 二值掩码, optional
    save_path: 保存路径, optional
    """
    n_cols = 3 if gt_mask is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    if image.shape[-1] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.max() > 1 else image
    else:
        img_rgb = image

    axes[0].imshow(img_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')

    score_norm = normalize_score_map(score_map)
    axes[1].imshow(score_norm, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Anomaly Map')
    axes[1].axis('off')

    overlay = overlay_heatmap(image, score_norm, alpha=0.4)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    axes[2].imshow(overlay_rgb)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap='gray')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def batch_visualize(images, score_maps, gt_masks=None, save_dir=None, prefix='result'):
    """
    批量可视化。
    images: list of ndarray (H, W, 3)
    score_maps: list of ndarray (H, W)
    gt_masks: list of ndarray (H, W), optional
    save_dir: 保存目录, optional
    """
    for i, (img, score) in enumerate(zip(images, score_maps)):
        gt = gt_masks[i] if gt_masks is not None else None
        save_path = f"{save_dir}/{prefix}_{i:04d}.png" if save_dir else None
        visualize_prediction(img, score, gt, save_path)


if __name__ == "__main__":
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    score_map = np.random.rand(256, 256)
    gt_mask = (np.random.rand(256, 256) > 0.9).astype(float)

    visualize_prediction(image, score_map, gt_mask)
