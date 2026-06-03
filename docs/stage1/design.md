# Stage 1 变化检测模型训练——设计思路

> 本文档说明 Stage 1（变化检测模型训练与验证）的设计目标、模型架构选择、数据构造逻辑和训练策略。

---

## 1. 任务定义

Stage 1 的目标是训练一个**二分类语义分割模型**，输入 t₁ 时刻的驱动因子 + 土地利用（LU）图，输出逐像素的**变化概率图**（pred_diff），标记哪些像素在 t₁→t₂ 期间发生了土地利用类型变化。

该模型是 Stage 2 随机优化的**先验信息来源**——优化器利用 pred_diff 缩小搜索空间，引导变化区域聚焦于模型预测的高概率区域。

### 1.1 输入输出

| 项目 | 形状 | 说明 |
|------|------|------|
| 输入 | `(B, factor_num+6, 128, 128)` | 驱动因子（DEM/DL/GDP/RK）+ LU 的 6 类 one-hot 编码 |
| 输出 | `(B, 1, 128, 128)` | 每个像素的变化概率（logit，经 sigmoid 后为概率） |
| 标签 | `(B, 1, 128, 128)` | 差分图（srcLU ≠ tgtLU 处为 1，否则为 0） |

### 1.2 通道拼接逻辑

```python
inputs = torch.cat([factors_src, lu_src], dim=1)
# factors_src: (B, 4, 128, 128) — DEM, DL, GDP, RK
# lu_src:      (B, 6, 128, 128) — LU 的 6 类 one-hot
# inputs:      (B, 10, 128, 128)
```

LU 的 one-hot 编码将原始 19 类标签映射为 6 个简化类别（通过 `one_hot_lu()` 函数），减少类别稀疏性。

---

## 2. 模型架构

系统支持 5 种语义分割网络：

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| **AttU-Net** | 注意力门控的 U-Net，抑制无关特征 | 默认选择，平衡精度与速度 |
| **U-Net** | 经典编码-解码结构，跳跃连接 | 基线对比 |
| **FCN** | 全卷积网络，无跳跃连接 | 轻量基线 |
| **SegFormer** | 基于 Transformer 的分割网络 | 大感受野场景 |
| **TransUNet** | Transformer + U-Net 混合架构 | 兼顾全局与局部信息 |

所有模型的输入通道数 = `len(factor_list) + 6`，输出通道数 = 1。

---

## 3. 数据构造

### 3.1 数据来源

- **路径**：`_data/training_data/{train|val}/valid_{10|15|20}/{FACTOR}/*.tif`
- **三个年份**：2010、2015、2020（对应内部标识 `10`、`15`、`20`）
- **每个样本**：128×128 像素切片，相邻 50% 重叠

### 3.2 时间组合（pack）

每个样本可按不同时间跨度组合：

| pack | 含义 | 增强倍数 |
|------|------|---------|
| `[10, 15]` | 2010→2015 的 5 年变化 | 6× |
| `[15, 20]` | 2015→2020 的 5 年变化 | 2× |
| `[10, 20]` | 2010→2020 的 10 年变化 | 2× |
| `[-1, -1]` | 以上全部 | 1× |

增强倍数反映不同时间跨度的数据稀缺程度——5 年变化（10→15）样本最少，增强最多。

### 3.3 变化样本筛选

- **变化阈值**：`PIXEL_NUM_THRESHOLD_ISCHANGE`（默认 30 像素）
- **`only_changed=True`**：仅使用有变化的样本训练（过滤无变化切片）
- **变化标签**：`changed_flag = 1 if count_nonzero(srcLU ≠ tgtLU) ≥ 30 else 0`

### 3.4 数据增强

对训练集中变化样本（`changed_flag=1`）应用随机变换（翻转、旋转等），增强次数由 pack 类型决定。

---

## 4. 损失函数

采用 **Focal Loss + DICE Loss** 的复合损失：

```
Loss = 0.5 × Focal_loss + 0.5 × DICE_loss
```

- **Focal Loss**：解决正负样本不平衡（变化像素远少于未变化像素），通过降低易分类样本的权重聚焦于难分类样本
- **DICE Loss**：直接优化分割 overlap 指标，对类别不平衡更鲁棒

---

## 5. 训练策略

### 5.1 优化器

| 模型 | 优化器 | 学习率 | 权重衰减 |
|------|--------|--------|---------|
| 默认（AttU-Net/U-Net/FCN/TransUNet） | Adam | 1e-4 | 1e-5 |
| SegFormer | AdamW | 5e-5 | 1e-4 |

### 5.2 学习率调度

- **策略**：验证集损失连续 8 个 epoch 未改善时，学习率乘以 0.8
- **梯度裁剪**：max_norm = 1.0

### 5.3 模型保存

- **最佳模型**：每个 epoch 结束后，若验证集损失下降，保存 `best_model.pth`
- **定期备份**：每 20 个 epoch 保存一次 `epoch_{n}.pth`
- **断点续训**：支持从 checkpoint 恢复（`latest_checkpoint.pth`）

---

## 6. 评估指标

验证阶段计算以下指标：

| 指标 | 含义 |
|------|------|
| OA (Overall Accuracy) | 总体准确率 |
| Precision (UA) | 精确率 |
| Recall (PA) | 召回率 |
| F1 | F1 分数 |
| IoU | 交并比 |
| Kappa | Kappa 系数 |
| FOM | Figure of Merit |

---

## 7. 与 Stage 2 的衔接

Stage 1 的输出（训练好的模型权重）直接用于 Stage 2：

1. Stage 2 加载模型权重（`config.model.model_path`）
2. 对每个优化样本，调用 `get_pred_diff(ANN, ori_plan, factors)` 生成 pred_diff
3. pred_diff 经阈值 θ=0.5 二值化后，作为目标函数中 change_score 项的先验
4. 同时用于构造 hard_restrict（grid_restrict），约束优化搜索空间

```
Stage 1 输出: best_model.pth
                ↓
Stage 2 输入: ANN model → pred_diff → change_score + hard_restrict
```
