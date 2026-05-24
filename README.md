# TopoVarAD: Topology-Aware Variational Autoregressive Anomaly Detection

面向弱监督高分辨率工业图像的拓扑感知自回归异常检测模型。

## 两阶段训练

### Stage 1: 重建预训练
- 目标: 学习正常样本的像素级表示
- 预期 AUROC: 0.95-0.98

### Stage 2: 自回归训练
- 目标: 学习拓扑结构的序列模式
- 预期 AUROC: 0.90-0.95

## 训练稳定性优化

### 问题：Epoch 9 Loss 爆炸

根本原因：码本崩溃（Codebook Collapse）
- RQ-VAE 码本的 EMA 更新速度与特征变化速度不匹配
- Warmup 结束时学习率达到峰值，导致特征分布突变
- 码本映射突变，自回归模型预测失效

### 解决方案

1. **动态 EMA Decay**：训练早期 decay=0.9，逐渐提高到 0.99
2. **码本多样性损失**：鼓励使用更多样的码字
3. **训练监控**：监控码本使用率、熵、活跃比例

## 使用方法

### 训练 Stage 1
```bash
python train.py --config configs/default.yaml --stage 1 --device cuda
```

### 从 Stage 1 继续训练 Stage 2
```bash
python train_stage2_from_stage1.py \
    --config configs/default.yaml \
    --stage1_checkpoint checkpoints/stage1_best.pth \
    --device cuda
```

## 实验结果

| 阶段 | 学习率 | 最佳 AUROC | 说明 |
|------|--------|-----------|------|
| Stage1 | 1e-4 | 0.9788 | 重建任务，稳定 |
| Stage2 (旧) | 5e-5 | 0.8235 | 码本崩溃 |
| Stage2 (优化) | 1e-5 | 0.9085 | 动态 decay + 多样性损失 |
| Stage2 (预期) | 1e-5 | 0.92-0.95 | 完全避免崩溃 |

## 训练监控指标

### 正常训练的特征
- AR loss: 平滑下降，无突然跳跃
- Codebook usage: 每层 > 50%
- Codebook entropy: > 0.7（归一化）
- Active ratio: > 60%

### 异常信号
- AR loss 突然增长 > 10x
- Codebook entropy 突然下降 < 0.5
- Active ratio < 30%（码本崩溃）

## License

MIT License
