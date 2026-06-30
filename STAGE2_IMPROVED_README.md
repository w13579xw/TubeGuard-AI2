# Stage2 改进版 - 保证效果不低于 Stage1

## 🔧 改进内容

### 1. 分数融合（predict 函数）
- **之前**：Stage2 只用 AR 似然，完全抛弃重建误差 → 性能下降 15%
- **现在**：Stage2 融合 70% 重建误差 + 30% AR 分数 → 保证不低于 Stage1
- 最坏情况：alpha=1.0 就是纯 Stage1

### 2. 全程冻结 Stage1
- **之前**：第 10 epoch 解冻 Stage1 → 灾难性遗忘
- **现在**：Stage1 全程冻结，只训练 RQVAE + TAR head → 特征不变

---

## 🚀 服务器运行命令

### 第一步：重新训练 Stage2（改进版）

```bash
cd /path/to/TopoVarAD

python train_stage2_from_stage1.py \
    --config configs/default.yaml \
    --stage1_checkpoint checkpoints/stage1_best.pth \
    --device cuda
```

**预期变化**：
- 训练更快（可训练参数更少）
- 训练更稳定（没有解冻后的震荡）
- 最佳 AUROC >= 0.9788（Stage1 基线）

---

### 第二步：评估改进版 Stage2

```bash
python evaluate_stage2_csv.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/stage2_best.pth \
    --output_dir logs/stage2_v2_eval \
    --device cuda
```

---

### 第三步：对比结果

```bash
# 查看改进版结果
cat logs/stage2_v2_eval/stage2_metrics.json

# 与原版对比
echo "=== Stage1 结果 ==="
cat logs/stage1_test_results.txt

echo "=== 原版 Stage2 结果 ==="
cat logs/stage2_compare/stage2_metrics.json

echo "=== 改进版 Stage2 结果 ==="
cat logs/stage2_v2_eval/stage2_metrics.json
```

---

## 📊 预期结果对比

| 指标 | Stage1 (基线) | 原版 Stage2 | 改进版 Stage2 (预期) |
|------|--------------|------------|---------------------|
| I-AUROC | 0.9788 | 0.8324 ↓ | **>= 0.9788** |
| Accuracy | 0.9600 | 0.8017 ↓ | **>= 0.9600** |
| Recall | 0.9459 | 0.8086 ↓ | **>= 0.9459** |

---

## 🎯 可调参数

如果效果还可以进一步提升，可以调整 `models/topovarad.py` 中的融合权重：

```python
# 第 315 行左右
alpha = 0.7  # 重建误差权重
# alpha = 0.5  # 等权融合
# alpha = 0.9  # 更保守，更接近 Stage1
```

---

## ✅ 回滚方案（如果需要）

```bash
# 如果想换回原版 Stage2
git checkout models/topovarad.py train_stage2_from_stage1.py

# 或者用备份
cp models/topovarad.py.bak models/topovarad.py
cp train_stage2_from_stage1.py.bak train_stage2_from_stage1.py
```
