import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic
from collections import defaultdict


class T2MTokenizer(nn.Module):
    """
    Topology-aware Token Modulation Tokenizer
    使用多尺度超像素池化替代固定网格分割，保持缺陷拓扑完整性。
    """

    def __init__(self, d_model=256, superpixel_scales=(50, 100, 200)):
        super().__init__()
        self.d_model = d_model
        self.superpixel_scales = superpixel_scales

        self.proj = nn.Linear(d_model, d_model)
        self.scale_embed = nn.Embedding(len(superpixel_scales), d_model)
        self.pos_proj = nn.Linear(d_model, d_model)

    def _slic_segment(self, image_np, n_segments, compactness=10.0):
        """
        调用skimage SLIC生成超像素标签图。
        image_np: (H, W, 3) float32 [0,1]
        返回: (H, W) int64 超像素标签
        """
        segments = slic(
            image_np,
            n_segments=n_segments,
            compactness=compactness,
            start_label=0,
            channel_axis=-1,
        )
        return segments

    def _superpixel_pool(self, feature_map, seg_mask):
        """
        对特征图按超像素mask做自适应平均池化。
        feature_map: (B, C, H, W)
        seg_mask: (H, W) int64 超像素标签
        返回: (B, N_sp, C) N_sp为该尺度的超像素数量
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device

        seg_tensor = torch.from_numpy(seg_mask).long().to(device)
        num_sp = int(seg_tensor.max().item()) + 1

        flat = feature_map.reshape(B, C, H * W)
        seg_flat = seg_tensor.reshape(H * W)

        sum_features = torch.zeros(B, C, num_sp, device=device)
        count = torch.zeros(num_sp, device=device)

        for b in range(B):
            sum_features[b].scatter_add_(
                1, seg_flat.unsqueeze(0).expand(C, -1), flat[b]
            )
        count.scatter_add_(0, seg_flat, torch.ones(H * W, device=device))
        count = count.clamp(min=1.0)
        pooled = sum_features / count.unsqueeze(0).unsqueeze(0)

        return pooled.permute(0, 2, 1)

    def _build_adjacency(self, seg_mask):
        """
        构建超像素邻接图。
        seg_mask: (H, W) int64
        返回: set of (i, j) 邻接超像素对
        """
        H, W = seg_mask.shape
        edges = set()
        for i in range(H):
            for j in range(W):
                sp = seg_mask[i, j]
                if j + 1 < W and seg_mask[i, j + 1] != sp:
                    a, b = min(sp, seg_mask[i, j + 1]), max(sp, seg_mask[i, j + 1])
                    edges.add((a, b))
                if i + 1 < H and seg_mask[i + 1, j] != sp:
                    a, b = min(sp, seg_mask[i + 1, j]), max(sp, seg_mask[i + 1, j])
                    edges.add((a, b))
        return edges

    def _graph_laplacian_pe(self, edges, num_sp, d_model):
        """
        计算Graph Laplacian Position Encoding。
        返回: (num_sp, d_model)
        """
        if num_sp <= 1:
            return torch.zeros(num_sp, d_model)

        adj = np.zeros((num_sp, num_sp), dtype=np.float32)
        for a, b in edges:
            adj[a, b] = 1.0
            adj[b, a] = 1.0

        deg = adj.sum(axis=1)
        deg[deg == 0] = 1.0
        deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        L = np.eye(num_sp, dtype=np.float32) - deg_inv_sqrt @ adj @ deg_inv_sqrt

        k = min(d_model, num_sp - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        pe = eigenvectors[:, :k]
        if k < d_model:
            pe = np.pad(pe, ((0, 0), (0, d_model - k)))
        pe = pe.astype(np.float32)

        return torch.from_numpy(pe)

    def forward(self, x):
        """
        x: (B, C, H, W) 输入特征图或图像
        返回:
            token_grid: (B, total_tokens, D) 多尺度token序列
            superpixel_masks: list of np.ndarray [(H,W), ...] 各尺度超像素标签
            token_counts: list of int 各尺度token数量
        """
        B, C, H, W = x.shape
        device = x.device

        if C == 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x_norm = (x - mean) / std
        else:
            x_norm = x

        all_tokens = []
        all_masks = []
        token_counts = []

        for scale_idx, n_seg in enumerate(self.superpixel_scales):
            img_np = x_norm[0].permute(1, 2, 0).detach().cpu().numpy()
            img_np = np.clip(img_np, 0, 1)

            seg_mask = self._slic_segment(img_np, n_segments=n_seg)

            pooled = self._superpixel_pool(x_norm, seg_mask)

            edges = self._build_adjacency(seg_mask)
            num_sp = pooled.shape[1]
            pe = self._graph_laplacian_pe(edges, num_sp, self.d_model).to(device)

            pooled_proj = self.proj(pooled)
            pe_proj = self.pos_proj(pe)
            scale_tok = self.scale_embed(
                torch.tensor(scale_idx, device=device)
            ).unsqueeze(0).unsqueeze(0)

            tokens = pooled_proj + pe_proj.unsqueeze(0) + scale_tok
            all_tokens.append(tokens)
            all_masks.append(seg_mask)
            token_counts.append(num_sp)

        token_grid = torch.cat(all_tokens, dim=1)

        return token_grid, all_masks, token_counts


class T2MTokenizerFast(nn.Module):
    """
    T2MTokenizer的高性能版本：
    - 使用向量化scatter操作替代for循环
    - 支持batch内多图并行
    - 缓存邻接图和Laplacian PE
    """

    def __init__(self, d_model=256, superpixel_scales=(50, 100, 200), compactness=10.0):
        super().__init__()
        self.d_model = d_model
        self.superpixel_scales = superpixel_scales
        self.compactness = compactness

        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.scale_embed = nn.Embedding(len(superpixel_scales), d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self._pe_cache = {}

    def _slic_batch(self, images_np, n_segments):
        """批量SLIC分割"""
        masks = []
        for img in images_np:
            seg = slic(
                img,
                n_segments=n_segments,
                compactness=self.compactness,
                start_label=0,
                channel_axis=-1,
            )
            masks.append(seg)
        return masks

    def _vectorized_pool(self, feature_map, seg_mask):
        """
        向量化超像素池化。
        feature_map: (B, C, H, W)
        seg_mask: (H, W) int64
        返回: (B, N_sp, C)
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device

        seg = torch.from_numpy(seg_mask).long().to(device).reshape(-1)
        num_sp = int(seg.max().item()) + 1

        flat = feature_map.reshape(B, C, -1)
        one_hot = F.one_hot(seg, num_sp).float().to(device)

        sum_feat = torch.bmm(flat, one_hot.unsqueeze(0).expand(B, -1, -1))
        count = one_hot.sum(dim=0).clamp(min=1.0)

        pooled = sum_feat / count.unsqueeze(0).unsqueeze(0)
        return pooled.permute(0, 2, 1)

    def _compute_pe_cached(self, seg_mask, n_segments, d_model):
        """带缓存的Laplacian PE计算"""
        key = (hash(seg_mask.tobytes()), n_segments, d_model)
        if key in self._pe_cache:
            return self._pe_cache[key]

        H, W = seg_mask.shape
        num_sp = int(seg_mask.max()) + 1

        sp_i, sp_j = [], []
        for i in range(H):
            for j in range(W):
                sp = seg_mask[i, j]
                if j + 1 < W and seg_mask[i, j + 1] != sp:
                    sp_i.extend([sp, seg_mask[i, j + 1]])
                    sp_j.extend([seg_mask[i, j + 1], sp])
                if i + 1 < H and seg_mask[i + 1, j] != sp:
                    sp_i.extend([sp, seg_mask[i + 1, j]])
                    sp_j.extend([seg_mask[i + 1, j], sp])

        adj = np.zeros((num_sp, num_sp), dtype=np.float32)
        if sp_i:
            adj[sp_i, sp_j] = 1.0

        deg = adj.sum(axis=1)
        deg[deg == 0] = 1.0
        deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        L = np.eye(num_sp, dtype=np.float32) - deg_inv_sqrt @ adj @ deg_inv_sqrt

        k = min(d_model, num_sp - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        pe = eigenvectors[:, :k]
        if k < d_model:
            pe = np.pad(pe, ((0, 0), (0, d_model - k)))

        result = torch.from_numpy(pe.astype(np.float32))
        self._pe_cache[key] = result
        return result

    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: token_grid (B, total_tokens, D), masks, counts
        """
        B, C, H, W = x.shape
        device = x.device

        if C == 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x_norm = (x - mean) / std
        else:
            x_norm = x

        img_np = x_norm[0].permute(1, 2, 0).detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        all_tokens = []
        all_masks = []
        token_counts = []

        for scale_idx, n_seg in enumerate(self.superpixel_scales):
            seg_mask = self._slic_batch([img_np], n_segments=n_seg)[0]

            pooled = self._vectorized_pool(x_norm, seg_mask)

            pe = self._compute_pe_cached(seg_mask, n_seg, self.d_model).to(device)

            pooled_proj = self.proj(pooled)
            pe_proj = self.pos_proj(pe)
            scale_tok = self.scale_embed(
                torch.tensor(scale_idx, device=device)
            ).unsqueeze(0).unsqueeze(0)

            tokens = pooled_proj + pe_proj.unsqueeze(0) + scale_tok
            all_tokens.append(tokens)
            all_masks.append(seg_mask)
            token_counts.append(pooled.shape[1])

        token_grid = torch.cat(all_tokens, dim=1)

        return token_grid, all_masks, token_counts


if __name__ == "__main__":
    tokenizer = T2MTokenizer(d_model=256, superpixel_scales=(50, 100, 200))

    dummy_input = torch.randn(1, 3, 512, 512)

    tokens, masks, counts = tokenizer(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Token grid shape: {tokens.shape}")
    print(f"Token counts per scale: {counts}")
    print(f"Total tokens: {sum(counts)}")
    print(f"Number of superpixel masks: {len(masks)}")
    for i, m in enumerate(masks):
        print(f"  Scale {i}: mask shape {m.shape}, unique labels {len(np.unique(m))}")
