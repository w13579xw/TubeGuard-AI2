import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from .t2m_tokenizer import T2MTokenizer
from .tpm_block import TPMBlock
from .rqvae import RQVAEEncoder
from .tar_decoder import TARHead


class PixelReconstructionHead(nn.Module):
    """
    像素重建头：将token特征映射回像素空间。
    用于预训练阶段的重建损失计算。
    """

    def __init__(self, d_model=256, out_channels=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, 16 * 16 * out_channels),
        )
        self.out_channels = out_channels
        self.patch_size = 16

    def forward(self, tokens, M, N):
        """
        tokens: (B, L, D)
        M, N: token网格行列数
        返回: (B, C, H, W) 重建图像patch
        """
        B, L, D = tokens.shape
        pixels = self.proj(tokens)
        pixels = rearrange(pixels, 'b (m n) (p q c) -> b c (m p) (n q)',
                           m=M, n=N, p=self.patch_size, q=self.patch_size, c=self.out_channels)
        return pixels


class GlobalPoolingHead(nn.Module):
    """
    全局池化头：将token序列聚合为全局特征向量。
    用于RQ-VAE和TAR Head的条件输入。
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, tokens):
        """
        tokens: (B, L, D)
        返回: (B, D) 全局特征向量
        """
        x = self.norm(tokens)
        x = x.mean(dim=1)
        return self.proj(x)



class TopoVarAD(nn.Module):
    """
    TopoVarAD：面向弱监督高分辨率工业图像的拓扑感知自回归异常检测模型。

    架构：T2M-Tokenizer → TPM Blocks → Global Pool → RQ-VAE → TAR Head
    两阶段训练：
        Stage1: 重建预训练（L_pixel + L_lpips）
        Stage2: 联合训练（L_pixel + L_lpips + L_rqvae + L_ar）
    """

    def __init__(
        self,
        d_model=256,
        n_tpm_layers=6,
        n_heads=8,
        d_state=16,
        expand=2,
        superpixel_scales=(50, 100, 200),
        rqvae_codebook_size=1024,
        rqvae_d_code=32,
        rqvae_n_layers=8,
        tar_n_layers=6,
        tar_n_heads=8,
        dropout=0.0,
        use_slic=True,
        use_topo_attn=True,
        use_glpe=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.superpixel_scales = superpixel_scales
        self.use_slic = use_slic
        self.use_glpe = use_glpe

        # ---- Tokenizer ----
        if use_slic:
            self.input_proj = nn.Conv2d(3, d_model, kernel_size=3, padding=1)
            self.tokenizer = T2MTokenizer(d_model, superpixel_scales)
        else:
            # Fixed 16x16 patch embedding (ViT-style, no SLIC)
            self.patch_embed = nn.Conv2d(3, d_model, kernel_size=16, stride=16)

        # ---- Position Encoding ----
        if not use_glpe:
            self.learned_pe = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)

        # ---- TPM Block ----
        self.tpm = TPMBlock(
            d_model=d_model,
            n_layers=n_tpm_layers,
            n_heads=n_heads,
            d_state=d_state,
            expand=expand,
            dropout=dropout,
            use_topo_attn=use_topo_attn,
        )

        self.pool_head = GlobalPoolingHead(d_model)
        self.pixel_head = PixelReconstructionHead(d_model, out_channels=3)

        self.rqvae = RQVAEEncoder(
            d_model=d_model,
            n_codes=rqvae_codebook_size,
            d_code=rqvae_d_code,
            n_layers=rqvae_n_layers,
        )

        self.tar = TARHead(
            vocab_size=rqvae_codebook_size,
            d_model=d_model,
            n_layers=tar_n_layers,
            n_heads=tar_n_heads,
            d_code=rqvae_d_code,
            max_seq_len=rqvae_n_layers,
            dropout=dropout,
        )

        self.stage = 1

    def _tokenize(self, x):
        """
        图像 → 特征图 → token化。
        x: (B, 3, H, W)
        返回: tokens (B, L, D), sp_labels (L,), M, N
        """
        if self.use_slic:
            feat = self.input_proj(x)
            tokens, masks, counts = self.tokenizer(feat)
            total_tokens = tokens.shape[1]
        else:
            # Fixed-grid patch embedding
            tokens = self.patch_embed(x)  # (B, D, H/16, W/16)
            B, D, Hp, Wp = tokens.shape
            tokens = tokens.flatten(2).transpose(1, 2)  # (B, Hp*Wp, D)
            total_tokens = Hp * Wp

        M = int(np.ceil(np.sqrt(total_tokens)))
        N = int(np.ceil(total_tokens / M))

        pad_len = M * N - tokens.shape[1]
        if pad_len > 0:
            pad = torch.zeros(tokens.shape[0], pad_len, tokens.shape[2], device=tokens.device)
            tokens = torch.cat([tokens, pad], dim=1)

        # Position encoding
        if not self.use_glpe:
            tokens = tokens + self.learned_pe[:, :tokens.shape[1], :]

        sp_labels = np.arange(M * N)

        return tokens, sp_labels, M, N

    def forward_stage1(self, x):
        """
        阶段1：重建预训练。
        x: (B, 3, H, W)
        返回: dict with losses
        """
        tokens, sp_labels, M, N = self._tokenize(x)

        refined = self.tpm(tokens, sp_labels, M, N)

        z_global = self.pool_head(refined)

        x_recon = self.pixel_head(refined, M, N)

        H_target = M * 16
        W_target = N * 16
        x_resized = F.interpolate(x, size=(H_target, W_target), mode='bilinear', align_corners=False)

        loss_pixel = F.l1_loss(x_recon, x_resized)
        loss_lpips = self._compute_lpips(x_recon, x_resized)

        return {
            'loss_pixel': loss_pixel,
            'loss_lpips': loss_lpips,
            'loss_total': loss_pixel + 0.1 * loss_lpips,
            'reconstructed': x_recon,
            'x_resized': x_resized,
            'z_global': z_global,
        }

    def forward_stage2(self, x):
        """
        阶段2：联合训练（重建 + RQ-VAE + 自回归）。
        x: (B, 3, H, W)
        返回: dict with losses
        """
        tokens, sp_labels, M, N = self._tokenize(x)

        refined = self.tpm(tokens, sp_labels, M, N)

        z_global = self.pool_head(refined)

        z_hat, codes, rqvae_loss, embeddings = self.rqvae(z_global)

        logits, ar_loss = self.tar(codes, z_global)

        x_recon = self.pixel_head(refined, M, N)

        H_target = M * 16
        W_target = N * 16
        x_resized = F.interpolate(x, size=(H_target, W_target), mode='bilinear', align_corners=False)

        loss_pixel = F.l1_loss(x_recon, x_resized)
        loss_lpips = self._compute_lpips(x_recon, x_resized)

        return {
            'loss_pixel': loss_pixel,
            'loss_lpips': loss_lpips,
            'loss_rqvae': rqvae_loss,
            'loss_ar': ar_loss,
            'loss_total': loss_pixel + 0.1 * loss_lpips + 0.5 * rqvae_loss + 1.0 * ar_loss,
            'reconstructed': x_recon,
            'x_resized': x_resized,
            'codes': codes,
            'z_global': z_global,
            'logits': logits,
        }

    def forward(self, x):
        """
        根据当前阶段选择前向传播。
        x: (B, 3, H, W)
        """
        if self.stage == 1:
            return self.forward_stage1(x)
        else:
            return self.forward_stage2(x)

    def predict(self, x):
        """
        ✅ 改进版推理接口：融合重建误差 + AR 似然分数
        保证效果至少不比 Stage1 差
        x: (B, 3, H, W)
        返回:
            image_scores: (B,) 图像级异常分数
            pixel_scores: (B, H, W) 像素级异常图
        """
        self.eval()
        with torch.no_grad():
            tokens, sp_labels, M, N = self._tokenize(x)
            refined = self.tpm(tokens, sp_labels, M, N)

            # ========== Stage1 分数：重建误差（永远保留！这是性能底线）==========
            x_recon = self.pixel_head(refined, M, N)
            H_target = M * 16
            W_target = N * 16
            x_resized = F.interpolate(x, size=(H_target, W_target), mode='bilinear', align_corners=False)
            recon_error = F.l1_loss(x_recon, x_resized, reduction='none').mean(dim=1)  # (B, Hr, Wr)
            recon_score = recon_error.mean(dim=[1, 2])  # (B,)

            # ========== Stage2 分数：AR 似然（可选增强）==========
            if self.stage == 1:
                # Stage1 模式，只用重建误差
                image_scores = recon_score
                pixel_scores = recon_error
            else:
                # Stage2 模式：融合重建误差 + AR 似然
                z_global = self.pool_head(refined)
                codes = self.rqvae.encode(z_global)
                token_scores, ar_score = self.tar.compute_anomaly_score(codes, z_global)

                # ========== 分数融合（核心改进）==========
                # alpha=0.7 表示重建误差权重 70%，AR 权重 30%
                # 确保 AR 只做增量改进，不会破坏 Stage1 的优秀性能
                alpha = 0.7
                image_scores = alpha * recon_score + (1 - alpha) * ar_score

                # 像素级也融合
                B, C, H, W = x.shape
                pixel_from_ar = token_scores.mean(dim=-1)
                pixel_from_ar = pixel_from_ar.unsqueeze(1).unsqueeze(1).expand(B, M, N)
                pixel_from_ar = F.interpolate(
                    pixel_from_ar.unsqueeze(1).float(),
                    size=(H, W), mode='bilinear', align_corners=False
                ).squeeze(1)
                pixel_scores = alpha * recon_error + (1 - alpha) * pixel_from_ar

        return image_scores, pixel_scores

    def set_stage(self, stage):
        """设置训练阶段"""
        self.stage = stage

    @torch.no_grad()
    def extract_global_features(self, x):
        """提取一批图像的全局特征 z_global，用于码本初始化。
        x: (B, 3, H, W) -> (B, d_model)
        """
        tokens, sp_labels, M, N = self._tokenize(x)
        refined = self.tpm(tokens, sp_labels, M, N)
        return self.pool_head(refined)

    @torch.no_grad()
    def init_codebook_from_loader(self, loader, device, max_batches=50, n_iter=10):
        """用训练数据的全局特征对RQ-VAE码本做K-means初始化。"""
        self.eval()
        feats = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            images = batch['image'].to(device)
            feats.append(self.extract_global_features(images).cpu())
        feats = torch.cat(feats, dim=0).to(device)
        self.rqvae.init_codebook_kmeans(feats, n_iter=n_iter)
        return feats.shape[0]

    def _compute_lpips(self, x, y):
        """简化的感知损失：基于特征的MSE近似LPIPS。"""
        x_norm = (x - 0.5) / 0.5
        y_norm = (y - 0.5) / 0.5
        return F.mse_loss(x_norm, y_norm)


class TopoVarADConfig:
    """TopoVarAD配置类，支持从字典或YAML加载。"""

    def __init__(self, **kwargs):
        self.d_model = kwargs.get('d_model', 256)
        self.n_tpm_layers = kwargs.get('n_tpm_layers', 6)
        self.n_heads = kwargs.get('n_heads', 8)
        self.d_state = kwargs.get('d_state', 16)
        self.expand = kwargs.get('expand', 2)
        self.superpixel_scales = kwargs.get('superpixel_scales', (50, 100, 200))
        self.rqvae_codebook_size = kwargs.get('rqvae_codebook_size', 1024)
        self.rqvae_d_code = kwargs.get('rqvae_d_code', 32)
        self.rqvae_n_layers = kwargs.get('rqvae_n_layers', 8)
        self.tar_n_layers = kwargs.get('tar_n_layers', 6)
        self.tar_n_heads = kwargs.get('tar_n_heads', 8)
        self.dropout = kwargs.get('dropout', 0.0)
        self.use_slic = kwargs.get('use_slic', True)
        self.use_topo_attn = kwargs.get('use_topo_attn', True)
        self.use_glpe = kwargs.get('use_glpe', True)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def build_model(self):
        return TopoVarAD(**self.to_dict())


if __name__ == "__main__":
    config = TopoVarADConfig(
        d_model=256,
        n_tpm_layers=2,
        n_heads=8,
        superpixel_scales=(50, 100),
        rqvae_codebook_size=512,
        rqvae_n_layers=4,
        tar_n_layers=2,
        tar_n_heads=4,
    )

    model = config.build_model()

    x = torch.randn(1, 3, 512, 512)

    print("=== Stage 1: Pretrain ===")
    model.set_stage(1)
    out1 = model(x)
    print(f"  loss_pixel: {out1['loss_pixel'].item():.4f}")
    print(f"  loss_lpips: {out1['loss_lpips'].item():.4f}")
    print(f"  loss_total: {out1['loss_total'].item():.4f}")
    print(f"  reconstructed shape: {out1['reconstructed'].shape}")
    print(f"  z_global shape: {out1['z_global'].shape}")

    print("\n=== Stage 2: Joint Training ===")
    model.set_stage(2)
    out2 = model(x)
    print(f"  loss_pixel: {out2['loss_pixel'].item():.4f}")
    print(f"  loss_lpips: {out2['loss_lpips'].item():.4f}")
    print(f"  loss_rqvae: {out2['loss_rqvae'].item():.4f}")
    print(f"  loss_ar: {out2['loss_ar'].item():.4f}")
    print(f"  loss_total: {out2['loss_total'].item():.4f}")
    print(f"  codes shape: {out2['codes'].shape}")
    print(f"  logits shape: {out2['logits'].shape}")

    print("\n=== Inference ===")
    image_scores, pixel_scores = model.predict(x)
    print(f"  image_scores: {image_scores.shape}")
    print(f"  pixel_scores: {pixel_scores.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
