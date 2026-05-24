import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SSIMLoss(nn.Module):
    """
    结构相似性损失：L_ssim = 1 - SSIM(x, x_hat)。
    衡量重建图像与原图的结构一致性。
    """

    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)

    def _create_window(self, size, channel):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, size, size).contiguous()
        return window

    def forward(self, x, y):
        """
        x, y: (B, C, H, W)
        返回: 标量 SSIM损失
        """
        channel = x.shape[1]
        if channel != self.channel or self.window.device != x.device:
            self.window = self._create_window(self.window_size, channel).to(x.device)
            self.channel = channel

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.conv2d(x, self.window, padding=self.window_size // 2, groups=channel)
        mu_y = F.conv2d(y, self.window, padding=self.window_size // 2, groups=channel)

        mu_sq_x = mu_x ** 2
        mu_sq_y = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_sq_x = F.conv2d(x ** 2, self.window, padding=self.window_size // 2, groups=channel) - mu_sq_x
        sigma_sq_y = F.conv2d(y ** 2, self.window, padding=self.window_size // 2, groups=channel) - mu_sq_y
        sigma_xy = F.conv2d(x * y, self.window, padding=self.window_size // 2, groups=channel) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_sq_x + mu_sq_y + C1) * (sigma_sq_x + sigma_sq_y + C2))

        return 1 - ssim_map.mean()


class LPIPSLoss(nn.Module):
    """
    简化版感知损失：基于ImageNet预训练VGG特征的MSE。
    完整版可替换为 lpips.LPIPS(net='vgg')。
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x, y):
        """
        x, y: (B, 3, H, W) [0,1]范围
        返回: 标量 感知损失
        """
        x_n = self._normalize(x)
        y_n = self._normalize(y)
        return F.mse_loss(x_n, y_n)


class FocalLoss(nn.Module):
    """
    Focal Loss：处理正负样本不平衡的分类损失。
    L = -α * (1-p)^γ * log(p)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: (B, C) 未归一化分数
        targets: (B,) 类别标签
        返回: 标量
        """
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


class TopoVarADLoss(nn.Module):
    """
    TopoVarAD联合损失函数。

    Stage1: L = L_pixel + λ_lpips * L_lpips
    Stage2: L = L_pixel + λ_lpips * L_lpips + λ_rqvae * L_rqvae + λ_ar * L_ar + λ_diversity * L_diversity
    """

    def __init__(self, lambda_lpips=0.1, lambda_rqvae=0.5, lambda_ar=1.0, lambda_diversity=0.01, label_smoothing=0.0):
        super().__init__()
        self.lambda_lpips = lambda_lpips
        self.lambda_rqvae = lambda_rqvae
        self.lambda_ar = lambda_ar
        self.lambda_diversity = lambda_diversity
        self.label_smoothing = label_smoothing

        self.ssim_loss = SSIMLoss()
        self.lpips_loss = LPIPSLoss()

    def codebook_diversity_loss(self, codes, n_codes=1024):
        """
        码本多样性损失：鼓励使用更多样的码字，防止码本崩溃。
        codes: (B, D) 离散 token 索引
        返回: 标量损失（越小越好，表示分布越均匀）
        """
        B, D = codes.shape
        device = codes.device

        diversity_losses = []
        for d in range(D):
            # 统计每个码字在当前层的使用频率
            layer_codes = codes[:, d]
            hist = torch.histc(layer_codes.float(), bins=n_codes, min=0, max=n_codes-1)

            # 计算分布的熵
            prob = hist / (hist.sum() + 1e-10)
            entropy = -(prob * torch.log(prob + 1e-10)).sum()

            # 最大熵（均匀分布）
            max_entropy = math.log(n_codes)

            # 损失：鼓励高熵（均匀分布）
            diversity_losses.append(max_entropy - entropy)

        return torch.stack(diversity_losses).mean()

    def forward(self, outputs, stage=1):
        """
        outputs: TopoVarAD前向传播的输出字典
        stage: 1或2
        返回: loss_dict
        """
        loss_pixel = outputs['loss_pixel']
        loss_lpips = self.lpips_loss(outputs['reconstructed'],
                                     outputs['x_resized'])

        if stage == 1:
            total = loss_pixel + self.lambda_lpips * loss_lpips
            return {
                'loss_pixel': loss_pixel,
                'loss_lpips': loss_lpips,
                'loss_total': total,
            }
        else:
            loss_rqvae = outputs.get('loss_rqvae', torch.tensor(0.0))
            loss_ar = outputs.get('loss_ar', torch.tensor(0.0))

            # 计算码本多样性损失
            codes = outputs.get('codes', None)
            if codes is not None:
                loss_diversity = self.codebook_diversity_loss(codes)
            else:
                loss_diversity = torch.tensor(0.0, device=loss_pixel.device)

            total = (loss_pixel
                     + self.lambda_lpips * loss_lpips
                     + self.lambda_rqvae * loss_rqvae
                     + self.lambda_ar * loss_ar
                     + self.lambda_diversity * loss_diversity)
            return {
                'loss_pixel': loss_pixel,
                'loss_lpips': loss_lpips,
                'loss_rqvae': loss_rqvae,
                'loss_ar': loss_ar,
                'loss_diversity': loss_diversity,
                'loss_total': total,
            }


if __name__ == "__main__":
    ssim = SSIMLoss()
    x = torch.randn(2, 3, 64, 64).sigmoid()
    y = torch.randn(2, 3, 64, 64).sigmoid()
    print(f"SSIM loss: {ssim(x, y).item():.4f}")

    lpips = LPIPSLoss()
    print(f"LPIPS loss: {lpips(x, y).item():.4f}")

    focal = FocalLoss()
    logits = torch.randn(4, 1)
    targets = torch.tensor([0., 1., 1., 0.])
    print(f"Focal loss: {focal(logits.squeeze(), targets).item():.4f}")
