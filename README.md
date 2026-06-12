# TopoVarAD: 拓扑感知变分自回归异常检测

面向弱监督高分辨率工业图像的拓扑感知自回归异常检测模型。

## 目录

- [核心创新](#核心创新)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [训练](#训练)
- [评估](#评估)
- [基准对比实验](#基准对比实验)
- [消融实验](#消融实验)
- [实验结果](#实验结果)
- [License](#license)

## 核心创新

### 1. T2M-Tokenizer（拓扑感知Token化器）

解决固定网格Patch分割**割裂缺陷完整性**的问题。

- 使用 SLIC 多尺度超像素分割（s=50, 100, 200）替代固定网格
- 超像素自适应池化保持缺陷几何完整性
- Graph Laplacian 位置编码注入拓扑空间结构信息

### 2. TPM Block（拓扑保持Mamba块）

解决 Mamba 全局展平扫描**破坏空间拓扑**的问题。

- **Branch A**: 双向 SSM（行列正反向扫描，O(n) 复杂度）
- **Branch B**: 拓扑稀疏注意力（仅同超像素内计算，O(k) 复杂度）
- **门控融合**: α·SSM + (1-α)·Attn，可学习参数动态平衡

### 3. TAR Head（拓扑感知自回归头）

将异常检测从连续高斯分布建模转为**离散token自回归预测**。

- RQ-VAE：8层残差量化，码本 K=1024，EMA 动态衰减 (0.9→0.99)
- GPT 风格因果 Transformer 解码器（6层，8头，RoPE）
- 异常分数：`-log P(codes | z_global)` 交叉熵

## 模型架构

```
输入图像 (512×512×3)
    ↓
Conv2d 降采样 → 特征图 (128×128×256)
    ↓
┌─ T2M-Tokenizer ───────────────────────────┐
│ SLIC多尺度分割 → 超像素池化 → Graph Lap PE  │
│ 输出: Tokens (L≈850, 256)                   │
└────────────────────────────────────────────┘
    ↓
┌─ TPM Block ×6 ────────────────────────────┐
│ Branch A: 双向SSM + Branch B: 拓扑稀疏注意力 │
│ 门控融合 → 精化Tokens                        │
└────────────────────────────────────────────┘
    ↓                              ↓
PixelReconHead              GlobalPooling → RQ-VAE → TAR Head
(Stage1 重建)                (Stage2 自回归)
    ↓                              ↓
|x - x̂| L1误差               -log P(codes|z) 交叉熵
→ 异常分数                    → 异常分数
```

## 项目结构

```
TopoVarAD/
├── configs/
│   └── default.yaml              # 全局配置
├── models/
│   ├── topovarad.py              # 主模型 + Config
│   ├── t2m_tokenizer.py          # 创新1: T2M-Tokenizer
│   ├── tpm_block.py              # 创新2: TPM Block
│   ├── rqvae.py                  # RQ-VAE 残差量化器
│   └── tar_decoder.py            # GPT 自回归解码器
├── data/
│   └── dataset.py                # CSV/MVTec 数据集
├── utils/
│   ├── losses.py                 # 损失函数 (SSIM/LPIPS/VQ/AR)
│   ├── metrics.py                # 评估指标 (AUROC/F1max/AU-PR/PRO)
│   ├── logger.py                 # 训练日志
│   └── visualize.py              # 可视化
├── benchmark/
│   ├── run_all.py                # 基准对比实验入口
│   ├── methods.py                # PaDiM/PatchCore/AE/RD4AD/EfficientAD
│   ├── ablation_real.py          # 消融实验（真实模型）
│   ├── ablation_merge.py         # 消融结果汇总
│   └── visualize_compare.py      # 热力图对比可视化
├── train.py                      # 两阶段训练
├── train_stage1_normal.py        # 仅正常样本 Stage1 训练
├── train_stage2_from_stage1.py   # 从 Stage1 继续 Stage2
├── test_stage1.py                # Stage1 评估（重建误差法）
├── infer.py                      # 单图推理
├── test_run.py                   # 冒烟测试
└── logs/                         # 训练日志与指标
```

## 环境配置

```bash
conda create -n topovarad python=3.10
conda activate topovarad

# PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install numpy scipy scikit-learn scikit-image matplotlib tqdm pyyaml pillow einops
```

## 数据准备

### CSV 格式（默认）

```
data/
├── train.csv          # image,label  ([无缺陷] / [有缺陷])
├── test.csv           # 同上
└── images/            # 所有图片
```

训练集: 2400 张（800 正常 + 1600 缺陷）
测试集: 600 张（156 正常 + 444 缺陷）
图像规格: 4024×3036 灰度，自动转 3 通道 RGB

### MVTec AD 格式

支持标准 MVTec AD 目录结构，通过 `configs/default.yaml` 中 `dataset_type: mvtec` 切换。

## 训练

### 两阶段训练流程

**Stage 1: 正常样本重建预训练**（推荐）

```bash
# 仅用正常样本训练（推荐用于异常检测）
python train_stage1_normal.py --config configs/default.yaml --device cuda

# 或用全部样本训练（用于基准测试）
python train.py --config configs/default.yaml --stage 1 --device cuda
```

训练完成后 checkpoint 保存在 `checkpoints/stage1_normal_best.pth`。

**Stage 2: 联合自回归训练**（从 Stage1 继续）

```bash
python train_stage2_from_stage1.py \
    --config configs/default.yaml \
    --stage1_checkpoint checkpoints/stage1_normal_best.pth \
    --device cuda
```

### 训练配置

| 参数 | Stage 1 | Stage 2 |
|------|:---:|:---:|
| Epochs | 100 | 150 |
| 学习率 | 1e-4 | 1e-5 |
| 批大小 | 16 | 16 |
| 优化器 | AdamW | AdamW |
| 调度器 | Cosine + 10ep Warmup | Cosine + 10ep Warmup |
| 梯度裁剪 | 1.0 | 0.5 |
| 早停 | 20 epochs | 20 epochs |

## 评估

### Stage 1 评估（重建误差法）

Stage 1 仅训练了 T2M-Tokenizer + TPM + PixelHead，评估使用**重建 L1 误差**作为异常分数：

```bash
python test_stage1.py --checkpoint checkpoints/stage1_normal_best.pth --device cuda
```

输出: AUROC, AU-PR, F1max, Accuracy, Precision, Recall, Specificity, 混淆矩阵

### 推理

```bash
python infer.py --config configs/default.yaml --checkpoint checkpoints/stage1_best.pth --image path/to/image.jpg
```

## 基准对比实验

与 5 种 SOTA 异常检测方法对比：

| 方法 | 范式 | 骨干 |
|------|------|------|
| AE | 重建误差 | 5层CNN |
| PaDiM | 高斯分布 | ResNet-18 |
| PatchCore | Coreset k-NN | WideResNet-50 |
| RD4AD | 逆向蒸馏 | ResNet-18 |
| EfficientAD | 轻量蒸馏 | WideResNet-50 |

```bash
# 运行所有基准对比
python benchmark/run_all.py --config configs/default.yaml --methods all --device cuda

# 仅运行特定方法
python benchmark/run_all.py --config configs/default.yaml --methods ae padim --device cuda
```

结果保存在 `logs/benchmark_results.json`。

## 消融实验

验证三个核心组件的独立贡献：

| 变体 | 改动 | 验证目标 |
|------|------|------|
| full | 完整模型 | 基线 |
| no_slic | SLIC → 固定Patch | 超像素贡献 |
| no_topo | TPM → 纯SSM | 拓扑注意力贡献 |
| no_glpe | Graph Lap PE → 可学习PE | 拓扑PE贡献 |

### 并行训练（推荐，4个同时跑）

```bash
mkdir -p logs/ablation

nohup python benchmark/ablation_real.py --config configs/default.yaml --variant full --device cuda > logs/ablation/full.log 2>&1 &
nohup python benchmark/ablation_real.py --config configs/default.yaml --variant no_slic --device cuda > logs/ablation/no_slic.log 2>&1 &
nohup python benchmark/ablation_real.py --config configs/default.yaml --variant no_topo --device cuda > logs/ablation/no_topo.log 2>&1 &
nohup python benchmark/ablation_real.py --config configs/default.yaml --variant no_glpe --device cuda > logs/ablation/no_glpe.log 2>&1 &
```

全部跑完后汇总：

```bash
python benchmark/ablation_merge.py
```

### 热力图可视化

```bash
python benchmark/visualize_compare.py --device cuda
# → logs/visualization/comparison_heatmaps.png
```

## 实验结果

### TopoVarAD Stage 1

| 指标 | 数值 |
|------|:---:|
| I-AUROC | **0.9788** |
| AU-PR | 0.9935 |
| F1-max | 0.9722 |
| Accuracy | 0.9600 |
| Precision | 1.0000 |
| Recall | 0.9459 |
| Specificity | 1.0000 |

混淆矩阵: TN=156, FP=0, FN=24, TP=420

### 基准对比

| 方法 | I-AUROC | F1max | AU-PR |
|------|:---:|:---:|:---:|
| **TopoVarAD S1** | **0.9788** | 0.9722 | 0.9935 |
| AE | 0.9758 | 0.9722 | 0.9926 |
| EfficientAD | 0.9524 | 0.9566 | 0.9816 |
| PatchCore | 0.9128 | 0.9256 | 0.9575 |
| PaDiM | 0.8508 | 0.8632 | 0.9327 |
| RD4AD | 0.8292 | 0.8916 | 0.9172 |

详细对比报告: `../benchmark_report.html` 和 `../对比实验.html`

## License

MIT License
