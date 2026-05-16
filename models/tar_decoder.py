import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class RotaryPositionEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE)：将位置信息编码到Q/K的相位中。
    相比绝对位置编码，RoPE天然支持相对位置建模。
    """

    def __init__(self, d_model, max_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def _build_cache(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin

    @staticmethod
    def _apply_rotary(x, cos, sin):
        """
        x: (B, H, L, d)
        cos, sin: (L, d//2)
        """
        d = x.shape[-1]
        x1, x2 = x[..., :d // 2], x[..., d // 2:]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.cat([out1, out2], dim=-1)

    def forward(self, q, k):
        """
        q, k: (B, H, L, d)
        返回: 旋转后的q, k
        """
        L = q.shape[2]
        cos, sin = self._build_cache(L, q.device)
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)


class CausalTransformerLayer(nn.Module):
    """
    因果Transformer层：多头因果注意力 + FFN。
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope):
        """
        x: (B, L, D) 输入序列
        rope: RotaryPositionEmbedding实例
        返回: (B, L, D)
        """
        B, L, D = x.shape

        residual = x
        x_norm = self.norm1(x)

        Q = rearrange(self.q_proj(x_norm), 'b l (h d) -> b h l d', h=self.n_heads)
        K = rearrange(self.k_proj(x_norm), 'b l (h d) -> b h l d', h=self.n_heads)
        V = rearrange(self.v_proj(x_norm), 'b l (h d) -> b h l d', h=self.n_heads)

        Q, K = rope(Q, K)

        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out_proj(out)
        x = residual + out

        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


class TARHead(nn.Module):
    """
    拓扑感知自回归头 (Topology-Aware Autoregressive Head)。
    将异常检测转化为next-token prediction：
    训练时：学习P(c_d | c_{1:d-1}, z)的交叉熵损失
    推理时：计算token级交叉熵作为异常分数
    """

    def __init__(
        self,
        vocab_size=1024,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_code=32,
        max_seq_len=32,
        dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_code = d_code
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.code_embed = nn.Embedding(vocab_size, d_code)
        self.query_embed = nn.Embedding(1, d_model)
        self.pos_proj = nn.Linear(d_code, d_model)
        self.cond_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionEmbedding(d_model // n_heads)

        self.layers = nn.ModuleList([
            CausalTransformerLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, codes, z_cond):
        """
        训练前向：输入完整codes序列，输出每个位置的预测logits。
        codes: (B, D) 离散token序列（来自RQ-VAE）
        z_cond: (B, d_model) 条件特征（来自TPM Block的全局表示）
        返回:
            logits: (B, D, vocab_size) 每个位置的预测分布
            ar_loss: 标量 自回归交叉熵损失
        """
        B, D = codes.shape
        device = codes.device

        code_emb = self.code_embed(codes)
        code_pos = self.pos_proj(code_emb)

        cond = self.cond_proj(z_cond).unsqueeze(1)

        x = torch.cat([cond, code_pos], dim=1)

        for layer in self.layers:
            x = layer(x, self.rope)

        x = self.norm(x)
        logits = self.head(x[:, 1:, :])

        targets = codes
        ar_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            reduction='mean',
        )

        return logits, ar_loss

    def compute_anomaly_score(self, codes, z_cond):
        """
        推理时：计算每个token的交叉熵作为异常分数。
        异常区域的token偏离正常分布，交叉熵更高。

        codes: (B, D)
        z_cond: (B, d_model)
        返回:
            token_scores: (B, D) 每个token的异常分数
            image_score: (B,) 图像级异常分数（所有token分数的均值）
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(codes, z_cond)

            log_probs = F.log_softmax(logits, dim=-1)
            targets = codes.unsqueeze(-1)
            token_scores = -log_probs.gather(dim=-1, index=targets).squeeze(-1)

            image_score = token_scores.mean(dim=-1)

        return token_scores, image_score

    def generate(self, z_cond, max_len=None, temperature=1.0):
        """
        自回归生成：逐token采样，用于异常检测时的分布外检测。
        z_cond: (B, d_model)
        返回: (B, D) 生成的token序列
        """
        B = z_cond.shape[0]
        device = z_cond.device
        max_len = max_len or self.max_seq_len

        cond = self.cond_proj(z_cond).unsqueeze(1)

        generated = []
        x = cond

        for step in range(max_len):
            for layer in self.layers:
                x = layer(x, self.rope)

            x_norm = self.norm(x[:, -1:, :])
            logits = self.head(x_norm)[:, 0, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)

            next_emb = self.pos_proj(self.code_embed(next_token))
            x = torch.cat([x, next_emb], dim=1)

        return torch.cat(generated, dim=1)


class TARHeadWithMemory(TARHead):
    """
    带记忆缓存的TAR Head：支持增量推理，避免重复计算历史token的KV。
    适用于推理加速场景。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kv_cache = None
        self._cache_len = 0

    def _build_kv_cache(self, x):
        """构建KV缓存"""
        B, L, D = x.shape
        kv = []
        for layer in self.layers:
            x_norm = layer.norm1(x)
            k = layer.k_proj(x_norm)
            v = layer.v_proj(x_norm)
            kv.append((k, v))
            x = layer(x, self.rope)
        self._kv_cache = kv
        self._cache_len = L

    def _update_cache(self, x_new):
        """增量更新KV缓存"""
        for i, layer in enumerate(self.layers):
            x_norm = layer.norm1(x_new)
            k = layer.k_proj(x_norm)
            v = layer.v_proj(x_norm)
            k_old, v_old = self._kv_cache[i]
            self._kv_cache[i] = (
                torch.cat([k_old, k], dim=1),
                torch.cat([v_old, v], dim=1),
            )

    def clear_cache(self):
        self._kv_cache = None
        self._cache_len = 0


if __name__ == "__main__":
    vocab_size = 1024
    d_model = 256
    d_code = 32
    n_layers = 6
    n_heads = 8
    B = 4
    D = 8

    tar = TARHead(vocab_size, d_model, n_layers, n_heads, d_code)

    codes = torch.randint(0, vocab_size, (B, D))
    z_cond = torch.randn(B, d_model)

    logits, ar_loss = tar(codes, z_cond)
    print(f"Codes: {codes.shape}")
    print(f"Logits: {logits.shape}")
    print(f"AR Loss: {ar_loss.item():.4f}")

    token_scores, image_score = tar.compute_anomaly_score(codes, z_cond)
    print(f"\nToken anomaly scores: {token_scores.shape}")
    print(f"Image anomaly scores: {image_score.shape}")

    generated = tar.generate(z_cond, max_len=D, temperature=0.8)
    print(f"\nGenerated tokens: {generated.shape}")

    params = sum(p.numel() for p in tar.parameters())
    print(f"\nParameters: {params:,}")
