import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    向量量化器：维护码本，执行最近邻查找 + EMA码本更新。
    支持动态调整 decay 以适应训练不同阶段。
    """

    def __init__(self, n_codes=1024, d_code=32, commitment_cost=0.25, decay=0.99, min_decay=0.9):
        super().__init__()
        self.n_codes = n_codes
        self.d_code = d_code
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.min_decay = min_decay
        self.training_step = 0

        embedding = torch.randn(n_codes, d_code) * 0.02
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_count', torch.zeros(n_codes))
        self.register_buffer('ema_weight', embedding.clone())
        self.register_buffer('initialized', torch.zeros(1))

    @torch.no_grad()
    def init_codebook_kmeans(self, residual, n_iter=10):
        """用一批残差向量做K-means初始化码本，缓解codebook collapse。
        residual: (N, d_code) 输入残差样本
        """
        n_samples = residual.shape[0]
        if n_samples < self.n_codes:
            # 样本不足，重复采样填满
            reps = (self.n_codes // n_samples) + 1
            residual = residual.repeat(reps, 1)
            n_samples = residual.shape[0]

        # 随机选 n_codes 个点作为初始中心
        perm = torch.randperm(n_samples, device=residual.device)[:self.n_codes]
        centers = residual[perm].clone()

        for _ in range(n_iter):
            dist = (
                residual.pow(2).sum(dim=-1, keepdim=True)
                - 2 * residual @ centers.t()
                + centers.pow(2).sum(dim=-1).unsqueeze(0)
            )
            assign = dist.argmin(dim=-1)
            for k in range(self.n_codes):
                mask = assign == k
                if mask.any():
                    centers[k] = residual[mask].mean(dim=0)

        self.embedding.copy_(centers)
        self.ema_weight.copy_(centers)
        self.ema_count.fill_(1.0)
        self.initialized.fill_(1.0)

    def forward(self, z):
        """
        z: (B, d_code) 输入向量
        返回:
            z_q: (B, d_code) 量化后向量
            codes: (B,) 码本索引
            vq_loss: 标量 VQ损失
        """
        dist = (
            z.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=-1).unsqueeze(0)
        )

        codes = dist.argmin(dim=-1)
        z_q = F.embedding(codes, self.embedding)

        if self.training:
            # 动态调整 decay：训练早期使用较低的 decay 以快速适应
            # 随着训练进行，逐渐提高 decay 以稳定码本
            self.training_step += 1
            warmup_steps = 5000  # 前5000步逐渐从 min_decay 增加到 decay
            if self.training_step < warmup_steps:
                current_decay = self.min_decay + (self.decay - self.min_decay) * (self.training_step / warmup_steps)
            else:
                current_decay = self.decay

            with torch.no_grad():
                one_hot = F.one_hot(codes, self.n_codes).float()
                count = one_hot.sum(dim=0)
                self.ema_count.mul_(current_decay).add_(count, alpha=1 - current_decay)

                weight_sum = one_hot.t() @ z
                self.ema_weight.mul_(current_decay).add_(weight_sum, alpha=1 - current_decay)

                n = self.ema_count.sum()
                self.ema_count.clamp_(min=1e-5)
                self.embedding.copy_(
                    self.ema_weight / self.ema_count.unsqueeze(-1)
                )

                # 死码重激活：将长期未使用的码本随机替换为当前batch中的活跃向量，
                # 缓解chronic codebook collapse。
                if self.training_step % 100 == 0:
                    dead = self.ema_count < 1e-3
                    n_dead = int(dead.sum().item())
                    if n_dead > 0 and z.shape[0] > 0:
                        idx = torch.randint(0, z.shape[0], (n_dead,), device=z.device)
                        self.embedding[dead] = z[idx].detach()
                        self.ema_weight[dead] = z[idx].detach()
                        self.ema_count[dead] = 1.0

        commitment_loss = F.mse_loss(z, z_q.detach())
        vq_loss = self.commitment_cost * commitment_loss

        z_q = z + (z_q - z).detach()

        return z_q, codes, vq_loss


class ResidualQuantizer(nn.Module):
    """
    RQ-VAE残差量化器：D层逐层残差量化。
    每层对上一层的残差进行量化，捕获从粗到细的语义信息。
    """

    def __init__(self, d_model=256, n_codes=1024, d_code=32, n_layers=8, commitment_cost=0.25, decay=0.99, min_decay=0.9):
        super().__init__()
        self.d_model = d_model
        self.n_codes = n_codes
        self.d_code = d_code
        self.n_layers = n_layers

        self.proj_in = nn.Linear(d_model, d_code)
        self.proj_out = nn.Linear(d_code, d_model)

        self.quantizers = nn.ModuleList([
            VectorQuantizer(n_codes, d_code, commitment_cost, decay, min_decay)
            for _ in range(n_layers)
        ])

        self.codebook_size = n_codes
        self.vocab_size = n_codes

    def forward(self, z):
        """
        z: (B, d_model) 连续特征向量
        返回:
            z_hat: (B, d_model) 量化重建向量
            codes: (B, D) 各层离散token索引
            total_vq_loss: 标量 总VQ损失
            embeddings: (B, D, d_code) 各层码本嵌入
        """
        B = z.shape[0]
        device = z.device

        z_proj = self.proj_in(z)

        residual = z_proj
        all_codes = []
        all_embeddings = []
        total_vq_loss = 0.0

        for quantizer in self.quantizers:
            z_q, codes, vq_loss = quantizer(residual)

            residual = residual - z_q
            all_codes.append(codes)
            all_embeddings.append(z_q)
            total_vq_loss = total_vq_loss + vq_loss

        codes = torch.stack(all_codes, dim=1)
        embeddings = torch.stack(all_embeddings, dim=1)

        z_hat = z_proj - residual
        z_hat = self.proj_out(z_hat)

        return z_hat, codes, total_vq_loss, embeddings

    def encode(self, z):
        """
        仅编码，返回离散token序列。
        z: (B, d_model)
        返回: (B, D) 离散token索引
        """
        z_proj = self.proj_in(z)
        residual = z_proj
        codes = []

        for quantizer in self.quantizers:
            _, code, _ = quantizer(residual)
            z_q = F.embedding(code, quantizer.embedding)
            residual = residual - z_q
            codes.append(code)

        return torch.stack(codes, dim=1)

    def decode(self, codes):
        """
        从离散token序列解码回连续特征。
        codes: (B, D) 离散token索引
        返回: (B, d_model) 重建特征
        """
        B, D = codes.shape
        device = codes.device

        z_proj = torch.zeros(B, self.d_code, device=device)
        for d, quantizer in enumerate(self.quantizers):
            z_q = F.embedding(codes[:, d], quantizer.embedding)
            z_proj = z_proj + z_q

        return self.proj_out(z_proj)

    def get_codebook_usage(self):
        """统计每个码本的使用率，用于监控训练质量。"""
        usage = []
        for quantizer in self.quantizers:
            active = (quantizer.ema_count > 0.5).sum().item()
            usage.append(active / self.n_codes)
        return usage

    @torch.no_grad()
    def init_codebook_kmeans(self, z, n_iter=10):
        """对每层量化器用真实残差做K-means初始化。
        z: (N, d_model) 一批来自pool_head的特征
        """
        z_proj = self.proj_in(z)
        residual = z_proj
        for quantizer in self.quantizers:
            quantizer.init_codebook_kmeans(residual, n_iter=n_iter)
            # 用初始化后的码本算出量化结果，传递残差给下一层
            dist = (
                residual.pow(2).sum(dim=-1, keepdim=True)
                - 2 * residual @ quantizer.embedding.t()
                + quantizer.embedding.pow(2).sum(dim=-1).unsqueeze(0)
            )
            codes = dist.argmin(dim=-1)
            z_q = F.embedding(codes, quantizer.embedding)
            residual = residual - z_q


class RQVAEEncoder(nn.Module):
    """
    RQ-VAE编码器：将TPM输出的token特征映射到离散空间。
    包含一个两层MLP投影头 + 残差量化器。
    """

    def __init__(self, d_model=256, n_codes=1024, d_code=32, n_layers=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.rq = ResidualQuantizer(d_model, n_codes, d_code, n_layers)

    def forward(self, z):
        """
        z: (B, d_model) 来自TPM Block的token特征
        返回: z_hat, codes, vq_loss, embeddings
        """
        z = self.mlp(z)
        return self.rq(z)

    def encode(self, z):
        z = self.mlp(z)
        return self.rq.encode(z)

    def decode(self, codes):
        return self.rq.decode(codes)

    @torch.no_grad()
    def init_codebook_kmeans(self, z, n_iter=10):
        """用一批pool_head特征对码本做K-means初始化。"""
        z = self.mlp(z)
        self.rq.init_codebook_kmeans(z, n_iter=n_iter)


if __name__ == "__main__":
    d_model = 256
    n_codes = 1024
    d_code = 32
    n_layers = 8
    B = 4

    rqvae = RQVAEEncoder(d_model, n_codes, d_code, n_layers)

    z = torch.randn(B, d_model)

    z_hat, codes, vq_loss, embeddings = rqvae(z)

    print(f"Input: {z.shape}")
    print(f"Codes: {codes.shape}")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Reconstructed: {z_hat.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Codebook usage: {rqvae.rq.get_codebook_usage()}")

    codes_enc = rqvae.encode(z)
    z_dec = rqvae.decode(codes_enc)
    print(f"\nEncode -> Decode test:")
    print(f"  Encoded codes: {codes_enc.shape}")
    print(f"  Decoded z: {z_dec.shape}")
    print(f"  Reconstruction MSE: {F.mse_loss(z, z_dec).item():.4f}")
