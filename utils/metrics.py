import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score


def compute_auroc(scores, labels):
    """
    计算AUROC。
    scores: (N,) 异常分数
    labels: (N,) 二值标签 (0=正常, 1=异常)
    返回: float AUROC
    """
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels).flatten()
    return roc_auc_score(labels, scores)


def compute_auprc(scores, labels):
    """
    计算精确率-召回率曲线下面积 (AU-PR)。
    scores: (N,) 异常分数
    labels: (N,) 二值标签
    返回: float AU-PR
    """
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels).flatten()
    return average_precision_score(labels, scores)


def compute_f1_max(scores, labels, n_thresholds=200):
    """
    计算最大F1分数 (I-F1max)。
    遍历多个阈值，取F1的最大值。
    scores: (N,) 异常分数
    labels: (N,) 二值标签
    返回: (float 最佳F1, float 最佳阈值)
    """
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels).flatten()

    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_f1 = 0.0
    best_thresh = thresholds[0]

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_f1, best_thresh


def compute_pro(pixel_scores, gt_masks, n_fprs=100):
    """
    计算 Per-Region Overlap (PRO)。
    在多个FPR阈值下计算缺陷区域的重叠度，取AUC。

    pixel_scores: list of (H, W) ndarray 像素级异常分数
    gt_masks: list of (H, W) ndarray 二值缺陷掩码
    返回: float PRO-AUC
    """
    all_scores = np.concatenate([s.flatten() for s in pixel_scores])
    all_labels = np.concatenate([m.flatten() for m in gt_masks])

    fpr_thresholds = np.linspace(0, 0.3, n_fprs)
    pro_values = []

    for fpr_target in fpr_thresholds:
        normal_scores = all_scores[all_labels == 0]
        if len(normal_scores) == 0:
            threshold = all_scores.max()
        else:
            threshold = np.percentile(normal_scores, (1 - fpr_target) * 100)

        pro_sum = 0.0
        n_regions = 0

        for score_map, gt_mask in zip(pixel_scores, gt_masks):
            from skimage.measure import label as connected_components
            labeled = connected_components(gt_mask.astype(int))
            n_components = labeled.max()

            if n_components == 0:
                continue

            pred_mask = score_map >= threshold

            for region_id in range(1, n_components + 1):
                region_mask = labeled == region_id
                overlap = (pred_mask & region_mask).sum() / region_mask.sum()
                pro_sum += overlap
                n_regions += 1

        if n_regions > 0:
            pro_values.append(pro_sum / n_regions)
        else:
            pro_values.append(0.0)

    pro_auc = np.trapz(pro_values, fpr_thresholds) / (fpr_thresholds[-1] - fpr_thresholds[0])
    return pro_auc


class MetricsCalculator:
    """
    异常检测评估指标计算器。
    支持图像级和像素级指标。
    """

    def __init__(self):
        self.image_scores = []
        self.image_labels = []
        self.pixel_scores = []
        self.pixel_labels = []

    def update(self, image_scores, image_labels, pixel_scores=None, pixel_labels=None):
        """
        累积一批数据的预测结果。
        image_scores: (B,) tensor/array
        image_labels: (B,) tensor/array
        pixel_scores: (B, H, W) tensor/array, optional
        pixel_labels: (B, H, W) tensor/array, optional
        """
        if isinstance(image_scores, torch.Tensor):
            image_scores = image_scores.cpu().numpy()
        if isinstance(image_labels, torch.Tensor):
            image_labels = image_labels.cpu().numpy()

        self.image_scores.append(image_scores.flatten())
        self.image_labels.append(image_labels.flatten())

        if pixel_scores is not None:
            if isinstance(pixel_scores, torch.Tensor):
                pixel_scores = pixel_scores.cpu().numpy()
            if isinstance(pixel_labels, torch.Tensor):
                pixel_labels = pixel_labels.cpu().numpy()
            self.pixel_scores.extend([s for s in pixel_scores])
            self.pixel_labels.extend([m for m in pixel_labels])

    def compute(self):
        """
        计算所有指标。
        返回: dict
        """
        img_scores = np.concatenate(self.image_scores)
        img_labels = np.concatenate(self.image_labels)

        results = {
            'I-AUROC': compute_auroc(img_scores, img_labels),
            'I-F1max': compute_f1_max(img_scores, img_labels)[0],
            'I-AU-PR': compute_auprc(img_scores, img_labels),
        }

        if self.pixel_scores:
            results['P-AUROC'] = compute_auroc(
                np.concatenate([s.flatten() for s in self.pixel_scores]),
                np.concatenate([m.flatten() for m in self.pixel_labels]),
            )
            results['PRO'] = compute_pro(self.pixel_scores, self.pixel_labels)

        return results

    def reset(self):
        self.image_scores = []
        self.image_labels = []
        self.pixel_scores = []
        self.pixel_labels = []

    def summary(self):
        """打印格式化的指标摘要。"""
        results = self.compute()
        print("=" * 50)
        print("Evaluation Metrics")
        print("=" * 50)
        for name, value in results.items():
            print(f"  {name:>12s}: {value:.4f}")
        print("=" * 50)
        return results


if __name__ == "__main__":
    np.random.seed(42)

    n_normal = 100
    n_anomaly = 20

    normal_scores = np.random.beta(2, 5, n_normal)
    anomaly_scores = np.random.beta(5, 2, n_anomaly)

    image_scores = np.concatenate([normal_scores, anomaly_scores])
    image_labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

    print(f"I-AUROC: {compute_auroc(image_scores, image_labels):.4f}")
    print(f"I-AU-PR:  {compute_auprc(image_scores, image_labels):.4f}")
    f1, thresh = compute_f1_max(image_scores, image_labels)
    print(f"I-F1max: {f1:.4f} (threshold={thresh:.4f})")

    pixel_scores = [np.random.rand(64, 64) for _ in range(5)]
    pixel_labels = [(np.random.rand(64, 64) > 0.9).astype(float) for _ in range(5)]
    print(f"P-AUROC: {compute_auroc(np.concatenate([s.flatten() for s in pixel_scores]), np.concatenate([m.flatten() for m in pixel_labels])):.4f}")

    calc = MetricsCalculator()
    calc.update(image_scores[:50], image_labels[:50])
    calc.update(image_scores[50:], image_labels[50:])
    calc.summary()
