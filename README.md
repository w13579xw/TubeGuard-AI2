# TopoVarAD

面向弱监督高分辨率工业图像的拓扑感知自回归异常检测

## 项目结构

```
TopoVarAD/
├── configs/
│   └── default.yaml                # 模型与训练配置
├── data/
│   └── dataset.py                  # 数据集加载（MVTec / CSV / 自定义）
├── models/
│   ├── t2m_tokenizer.py            # 拓扑感知Token化器（SLIC超像素池化）
│   ├── tpm_block.py                # 拓扑保持Mamba块（双向SSM+稀疏注意力）
│   ├── rqvae.py                    # 残差量化VAE（RQ-VAE）
│   ├── tar_decoder.py              # 拓扑感知自回归头（GPT风格解码器）
│   └── topovarad.py                # 完整模型组装
├── utils/
│   ├── losses.py                   # 损失函数（SSIM、LPIPS、Focal、联合损失）
│   ├── metrics.py                  # 评估指标（AUROC、F1、PRO）
│   └── visualize.py                # 异常热力图可视化
├── train.py                        # 两阶段训练脚本
├── infer.py                        # 推理与评估
├── test_run.py                     # 快速冒烟测试
└── requirements.txt                # 依赖清单
```

## 核心创新

1. **T2M-Tokenizer**：多尺度SLIC超像素池化替代固定网格分块，保持缺陷拓扑完整性。引入Graph Laplacian位置编码注入空间结构信息。

2. **TPM Block**：双向SSM（水平+垂直扫描）结合拓扑感知稀疏注意力——仅在同一超像素区域内计算attention，避免语义混合。

3. **TAR Head**：RQ-VAE残差量化器将连续特征压缩为离散token序列，GPT风格自回归解码器学习 `P(c_d | c_{1:d-1}, z)`，通过交叉熵作为异常分数。

## 环境安装

```bash
conda create -n TG2 python=3.10 -y
conda activate TG2
pip install torch torchvision einops scikit-image scikit-learn opencv-python matplotlib pyyaml tqdm
```

## 数据格式

支持三种数据集格式：

**CSV格式**（默认，自定义工业数据）：
```
data/
├── train.csv          # image,label
├── test.csv
└── images/            # 所有图片（支持灰度/RGB，自动转3通道）
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

CSV格式示例：
```csv
image,label
1.jpg,[有缺陷]
2.jpg,[无缺陷]
```

数据特点（以当前数据集为例）：
- 4024×3036 灰度工业图像
- 训练集：1600有缺陷 + 800无缺陷（2:1不平衡）
- 测试集：444有缺陷 + 156无缺陷
- 自动通过WeightedRandomSampler处理类别不平衡

**MVTec AD**（标准基准）：
```
data/mvtec/
└── bottle/
    ├── train/good/
    ├── test/good/
    ├── test/broken/
    └── ground_truth/broken/
```

**自定义目录**：
```
data/custom/
├── train/normal/
├── train/abnormal/
├── test/normal/
└── test/abnormal/
```

## 训练

```bash
# 阶段1：重建预训练
python train.py --config configs/default.yaml --stage 1 --device cuda

# 阶段2：联合训练（RQ-VAE + 自回归）
python train.py --config configs/default.yaml --stage 2 --resume checkpoints/stage1_best.pth --device cuda
```

## 推理

```bash
# 批量评估
python infer.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_best.pth \
    --mode eval --output results --device cuda

# 单图推理
python infer.py --config configs/default.yaml \
    --checkpoint checkpoints/stage2_best.pth \
    --mode demo --image test.png --output results --device cuda
```

## 评估指标

| 指标 | 含义 |
|------|------|
| I-AUROC | 图像级异常分类AUROC |
| P-AUROC | 像素级异常分割AUROC |
| PRO | 多FPR阈值下的区域重叠度 |
| I-F1max | 最大F1分数 |
| AU-PR | 精确率-召回率曲线下面积 |

## 模型架构

```
输入图像 (H×W)
    ↓
T2M-Tokenizer：SLIC超像素池化 → 拓扑保持token序列
    ↓
TPM Blocks × L层：双向SSM + 拓扑稀疏注意力
    ↓
全局池化 → z_global
    ↓
RQ-VAE：D层残差量化 → 离散codes
    ↓
TAR Head：自回归next-token预测 → 异常分数
```

## 开源协议

MIT