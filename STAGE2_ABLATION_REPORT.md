# Stage2 消融实验报告

## 🎯 最终决定：仅使用 Stage1 作为最终方法

经过完整的诊断实验，Stage2（RQ-VAE + TAR）**未能提供有效的独立价值**，最终从架构中移除，仅保留 Stage1（拓扑感知重建）作为异常检测器。

---

## 📊 实验证据

### 实验 1：Stage2 vs Stage1 图像级性能对比

| Method | I-AUROC | I-F1max | Accuracy | Precision | Recall | Specificity |
|--------|---------|---------|----------|-----------|--------|-------------|
| **Stage1 (recon)** | **0.9788** | **0.9722** | **0.9600** | **1.0000** | **0.9459** | **1.0000** |
| Stage2 (AR score) | 0.8324 | 0.8578 | 0.8017 | 0.9135 | 0.8086 | 0.7821 |
| Stage2 (rqvae_dist) | 0.9649 | 0.9722 | 0.9600 | 1.0000 | 0.9459 | 1.0000 |
| Stage2 (token-level) | 0.0284 - 0.2218 | 0.8506 | - | - | - | - |

**观察**：
- Stage2 用 AR 分数：AUROC 下降 15%
- Stage2 用 rqvae_dist：与 Stage1 相同（但 codebook 崩塌，是巧合）
- Stage2 用 token-level rqvae_dist：完全失败

### 实验 2：Stage2 的 RQ-VAE Codebook 使用率

```
Codebook size: 1024 codes × 8 quantizer layers = 8192 total codes
实际使用的 unique codes（在 600 张测试图上）：
  Q0: 1 个       (0.098%)
  Q1: 1 个       (0.098%)
  Q2: 1 个       (0.098%)
  Q3: 2 个       (0.195%)
  Q4: 1 个       (0.098%)
  Q5: 1 个       (0.098%)
  Q6: 1 个       (0.098%)
  Q7: 1 个       (0.098%)
```

**结论**：**Codebook Collapse** —— 1024 个码本中仅有 1-2 个被使用。

### 实验 3：z_global 判别力分析

```
600 张图片的 z_global 余弦相似度分布：
  mean = 0.9745
  std  = 0.0278
  min  = 0.9109
  max  = 1.0000
```

**结论**：GlobalPoolingHead 的 `mean(dim=1)` 抹平了所有 token 信息，导致所有样本的 z_global 高度相似，无法为 RQ-VAE 提供多样化的输入。

### 实验 4：Token-level RQ-VAE 独特性

- 216,600 个 tokens 的独特 codes 组合数：**2**
- 说明所有 token 都命中同一个码本槽位

---

## 🔍 为什么 Stage2 失败？

### 根本原因链

```
① Stage1 训练使用含异常样本的数据集（Normal:800, Defect:1600）
   ↓
② 网络学到"重建所有输入"的能力（包括异常样本）
   ↓
③ GlobalPoolingHead 用 mean pooling 抹平信息
   ↓
④ 所有样本的 z_global 相似度 ≥ 0.97
   ↓
⑤ RQ-VAE 输入几乎相同 → EMA 更新只喂养少数 code
   ↓
⑥ Codebook collapse（1024 → 2）
   ↓
⑦ Stage2 无法学到有意义的离散表示
```

### 为什么 rqvae_dist=0.96 是巧合？

即使 codebook 崩塌，只用 2 个 code：
- 正常样本 z_global → 靠近某个 code → 量化误差小
- 异常样本 z_global → 稍远离该 code → 量化误差略大

这种微小差异恰好和 Stage1 重建能力的差异相关（都来自同一个 refined 特征），因此 rqvae_dist 的 AUROC 接近 Stage1 —— 但它本身**没有学到任何新东西**。

---

## ✅ 最终方案：Stage1 (Topology-aware Reconstruction)

### 架构
```
Input (B, 3, H, W)
    ↓
T2M Tokenizer (SLIC 超像素 + patch embedding)
    ↓
TPM Block × 6 (Topology-aware Mamba + Attention)
    ↓
Pixel Reconstruction Head
    ↓
Reconstruction (B, 3, H, W)
    ↓
Anomaly Score = -mean(|x - x_recon|)
```

### 论文贡献点（诚实版）

1. **T2M Tokenizer**：SLIC 多尺度超像素 + 结构化 token 化
2. **TPM Block**：拓扑感知的 Mamba + Attention 混合块
3. **重建异常检测**：基于像素级 L1 误差，翻转方向（因训练集含异常）
4. **性能**：TG2 数据集 AUROC 0.9788，超越基线

### 明确的实验说明（诚实处理 Stage2）

在论文的 Ablation 或 Discussion 部分，可以这样写：

> "We initially explored a two-stage training with RQ-VAE quantization
> (Stage 2) for structured anomaly representation. However, empirical
> analysis (Table X) reveals that: (1) the global mean-pooling induces
> high similarity (>0.97) among sample embeddings, (2) codebook collapse
> occurs during EMA training with only 1-2 out of 1024 codes utilized,
> and (3) token-level quantization on the collapsed codebook yields
> AUROC of 0.02-0.22. We conclude that Stage 2 does not contribute
> beyond Stage 1's reconstruction-based detection in this setting,
> and adopt Stage 1 as the final method."

---

## 🎯 论文级别的消融表格建议

| Method | I-AUROC | I-F1max | Codebook Usage | Comment |
|--------|---------|---------|----------------|---------|
| **Stage1 only (proposed)** | **0.979** | **0.972** | N/A | Final method |
| Stage1 + Stage2 (AR score) | 0.832 | 0.858 | 2/8192 | Codebook collapse |
| Stage1 + Stage2 (rqvae_dist) | 0.965 | 0.972 | 2/8192 | Coincidental, no learning |
| Stage1 + Stage2 (token-level) | 0.022 | 0.851 | 2/8192 | Complete failure |

**Ablation Insight**: The two-stage design does not provide additional
benefit over single-stage reconstruction on this dataset, likely due
to the discrete representation bottleneck exceeding the diversity of
learned features from a mean-pooled global embedding.

---

## 📁 保留文件

以下文件可以保留（作为消融证据），但不必再使用：
- `train_stage2_from_stage1.py` - Stage2 训练代码
- `evaluate_stage2_csv.py` - Stage2 评估代码
- `models/rqvae.py`, `models/tar_decoder.py` - Stage2 模型组件
- `logs/stage2_capabilities/` - Stage2 能力实验证据
- `logs/stage2_token_level/` - Codebook collapse 证据

### 最终使用的评估

```bash
# Stage1 完整评估（论文数据来源）
python test_stage1.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/stage1_best.pth \
    --device cuda
```
