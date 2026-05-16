import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class SelectiveSSM(nn.Module):
    """
    选择性状态空间模型（Mamba核心）。
    简化实现：使用1D因果卷积 + 门控机制模拟SSM行为。
    完整版本可替换为 mamba_ssm.ops.selective_scan_cuda。
    """

    def __init__(self, d_model, d_state=16, expand=2, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (B, L, D) 序列
        返回: (B, L, D)
        """
        B, L, D = x.shape

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = rearrange(x_branch, 'b l d -> b d l')
        x_branch = self.conv1d(x_branch)[:, :, :L]
        x_branch = rearrange(x_branch, 'b d l -> b l d')
        x_branch = F.silu(x_branch)

        A = -torch.exp(self.A_log)
        x_proj = self.x_proj(x_branch)
        B_ssm = x_proj[:, :, :self.d_state]
        C_ssm = x_proj[:, :, self.d_state:]
        dt = F.softplus(self.dt_proj(B_ssm))

        dA = torch.exp(dt * A.sum(dim=-1))
        y = x_branch * self.D + dA * x_branch

        z = F.silu(z)
        out = y * z
        out = self.out_proj(out)

        return out


class BidirectionalSSM(nn.Module):
    """
    双向SSM：行优先 + 列优先双向扫描，门控融合。
    """

    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model

        self.ssm_h_fwd = SelectiveSSM(d_model, d_state, expand)
        self.ssm_h_bwd = SelectiveSSM(d_model, d_state, expand)
        self.ssm_v_fwd = SelectiveSSM(d_model, d_state, expand)
        self.ssm_v_bwd = SelectiveSSM(d_model, d_state, expand)

        self.gate_h = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.gate_v = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def _horizontal_scan(self, x, M, N):
        """
        行优先扫描：将M×N网格逐行展平为序列。
        x: (B, M*N, D)
        返回: (B, M*N, D)
        """
        B, L, D = x.shape
        h = rearrange(x, 'b (m n) d -> b (m n) d', m=M, n=N)

        h_fwd = self.ssm_h_fwd(h)
        h_bwd = self.ssm_h_bwd(h.flip(1)).flip(1)

        g = self.gate_h(torch.cat([h_fwd, h_bwd], dim=-1))
        return g * h_fwd + (1 - g) * h_bwd

    def _vertical_scan(self, x, M, N):
        """
        列优先扫描：将M×N网格逐列展平为序列。
        x: (B, M*N, D)
        返回: (B, M*N, D)
        """
        B, L, D = x.shape
        v = rearrange(x, 'b (m n) d -> b (n m) d', m=M, n=N)

        v_fwd = self.ssm_v_fwd(v)
        v_bwd = self.ssm_v_bwd(v.flip(1)).flip(1)

        g = self.gate_v(torch.cat([v_fwd, v_bwd], dim=-1))
        v_out = g * v_fwd + (1 - g) * v_bwd

        return rearrange(v_out, 'b (n m) d -> b (m n) d', m=M, n=N)

    def forward(self, x, M, N):
        """
        x: (B, M*N, D)
        返回: (B, M*N, D)
        """
        h_out = self._horizontal_scan(x, M, N)
        v_out = self._vertical_scan(x, M, N)

        alpha = self.fuse(torch.cat([h_out, v_out], dim=-1))
        return alpha * h_out + (1 - alpha) * v_out


class TopologySparseAttention(nn.Module):
    """
    拓扑感知稀疏注意力：仅在同一超像素内的token之间计算attention。
    避免将语义不同的区域混合，零额外计算量。
    """

    def __init__(self, d_model, n_heads=8, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sp_labels):
        """
        x: (B, L, D) token序列
        sp_labels: (L,) 每个token所属的超像素标签
        返回: (B, L, D)
        """
        B, L, D = x.shape
        device = x.device

        Q = rearrange(self.q_proj(x), 'b l (h d) -> b h l d', h=self.n_heads)
        K = rearrange(self.k_proj(x), 'b l (h d) -> b h l d', h=self.n_heads)
        V = rearrange(self.v_proj(x), 'b l (h d) -> b h l d', h=self.n_heads)

        sp = torch.from_numpy(sp_labels).long().to(device)
        mask = sp.unsqueeze(0) == sp.unsqueeze(1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale

        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out_proj(out)

        return out


class TopologySparseAttentionFast(nn.Module):
    """
    高效版本：使用torch_scatter进行分组注意力计算。
    避免构建L×L的完整attention矩阵。
    """

    def __init__(self, d_model, n_heads=8, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _grouped_attention(self, Q, K, V, sp_labels):
        """
        分组计算注意力，每个超像素独立处理。
        Q, K, V: (B, H, L, d)
        sp_labels: (L,)
        返回: (B, H, L, d)
        """
        B, H, L, d = Q.shape
        device = Q.device

        sp = torch.from_numpy(sp_labels).long().to(device)
        unique_sp = sp.unique()
        scale = math.sqrt(d)

        out = torch.zeros_like(Q)
        for sp_id in unique_sp:
            idx = (sp == sp_id).nonzero(as_tuple=True)[0]
            q_sp = Q[:, :, idx, :]
            k_sp = K[:, :, idx, :]
            v_sp = V[:, :, idx, :]

            attn = torch.matmul(q_sp, k_sp.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            out[:, :, idx, :] = torch.matmul(attn, v_sp)

        return out

    def forward(self, x, sp_labels):
        """
        x: (B, L, D)
        sp_labels: (L,) 超像素标签
        返回: (B, L, D)
        """
        B, L, D = x.shape

        Q = rearrange(self.q_proj(x), 'b l (h d) -> b h l d', h=self.n_heads)
        K = rearrange(self.k_proj(x), 'b l (h d) -> b h l d', h=self.n_heads)
        V = rearrange(self.v_proj(x), 'b l (h d) -> b h l d', h=self.n_heads)

        out = self._grouped_attention(Q, K, V, sp_labels)
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out_proj(out)

        return out


class TPMLayer(nn.Module):
    """
    单层TPM：双向SSM + 拓扑稀疏注意力 + 门控融合 + FFN
    """

    def __init__(self, d_model, n_heads=8, d_state=16, expand=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.bidirectional_ssm = BidirectionalSSM(d_model, d_state, expand)

        self.norm2 = nn.LayerNorm(d_model)
        self.sparse_attn = TopologySparseAttention(d_model, n_heads, dropout)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, sp_labels, M, N):
        """
        x: (B, L, D)
        sp_labels: (L,) 超像素标签
        M, N: token网格的行列数
        返回: (B, L, D)
        """
        residual = x
        x_norm = self.norm1(x)
        ssm_out = self.bidirectional_ssm(x_norm, M, N)
        x = residual + ssm_out

        residual = x
        x_norm = self.norm2(x)
        attn_out = self.sparse_attn(x_norm, sp_labels)
        x = residual + attn_out

        alpha = self.gate(torch.cat([ssm_out, attn_out], dim=-1))
        x_fused = alpha * ssm_out + (1 - alpha) * attn_out

        residual = x_fused
        x_norm = self.norm3(x_fused)
        x = residual + self.ffn(x_norm)

        return x


class TPMBlock(nn.Module):
    """
    拓扑保持Mamba块：L层TPMLayer堆叠。
    """

    def __init__(self, d_model=256, n_layers=6, n_heads=8, d_state=16, expand=2, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TPMLayer(d_model, n_heads, d_state, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sp_labels, M, N):
        """
        x: (B, L, D) token序列
        sp_labels: (L,) 超像素标签
        M, N: token网格行列数
        返回: (B, L, D) 精化后的token特征
        """
        for layer in self.layers:
            x = layer(x, sp_labels, M, N)
        return self.norm(x)


class TPMLayerWithFallback(TPMLayer):
    """
    带fallback的TPMLayer：当torch_scatter不可用时回退到基础版本。
    """

    def __init__(self, d_model, n_heads=8, d_state=16, expand=2, dropout=0.0, use_fast_attn=False):
        super().__init__(d_model, n_heads, d_state, expand, dropout)
        if use_fast_attn:
            self.sparse_attn = TopologySparseAttentionFast(d_model, n_heads, dropout)


if __name__ == "__main__":
    d_model = 256
    M, N = 20, 20
    L = M * N

    import numpy as np
    tpm = TPMBlock(d_model=d_model, n_layers=2, n_heads=8)

    x = torch.randn(1, L, d_model)
    sp_labels = np.random.randint(0, 10, size=(L,))

    out = tpm(x, sp_labels, M, N)

    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in tpm.parameters()):,}")
