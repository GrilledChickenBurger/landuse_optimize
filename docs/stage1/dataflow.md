# Stage 1 变化检测模型训练——数据流追踪

> 本文档逐条追踪 Stage 1 训练和验证流程中每个阶段的数据流动。

---

## 1. 数据加载

### 1.1 原始数据读取

**文件路径**：`_data/training_data/{train|val}/valid_{year}/{FACTOR}/*.tif`

```
每个年份读取的因子:
  condition_list = ["DEM", "DL", "GDP", "RK", "LU"]
  → 逐因子读取 .tif 文件
  → 堆叠为 (num_samples, factor_num, 128, 128)
```

**三个年份的数据分别加载**：
```
allxf[0] = get_year_data("valid_10")  # shape: (N, 5, 128, 128)
allxf[1] = get_year_data("valid_15")  # shape: (N, 5, 128, 128)
allxf[2] = get_year_data("valid_20")  # shape: (N, 5, 128, 128)
```

### 1.2 数据标准化

对 GDP、RK、DL 三个因子进行对数变换 + Z-score 标准化：

```
x_log = log(x + 1)          # 对数变换（仅 GDP, RK, DL）
x_norm = (x_log - μ) / σ    # Z-score 标准化
```

参数 μ、σ 来自 `libs/const.py` 中的 `FACTOR_PARAMS_DICT[year]`。

---

## 2. 样本构造（create_training_data_pack）

### 2.1 单个 pack 的构造过程

输入：`allxf[0][idx]`（src 年份）和 `allxf[1][idx]`（tgt 年份）

```
步骤 1: 计算 diff 标签
  diff = (srcLU ≠ tgtLU)         # bool, shape (128, 128)
  diff = remove_lines(diff)       # 去除线状伪变化
  diff = np.expand_dims(diff, 0)  # shape (1, 128, 128)

步骤 2: 计算 changed_flag
  num_unequal = count_nonzero(diff)
  changed_flag = 1 if num_unequal ≥ 30 else 0

步骤 3: LU one-hot 编码
  srcLU_raw = xf_src[4]           # shape (128, 128), 值为 LU 标签
  srcLU_onehot = one_hot_lu(srcLU_raw)  # shape (6, 128, 128)

步骤 4: 组装 pack
  pack = [
    xf_src[:4],       # 驱动因子 (4, 128, 128)
    srcLU_onehot,      # LU one-hot (6, 128, 128)
    diff,              # 差分标签 (1, 128, 128)
    idx_label,         # "10_760_5" (字符串)
    changed_flag,      # 0 或 1
  ]
```

### 2.2 数据增强

对训练集中的变化样本（`changed_flag=1`）应用随机变换：

```
apply_transform(pack, transform, repeat_num)
  → 对 [factors, srcLU, diff] 拼接后统一变换
  → 拆分回原始结构
  → repeat_num 次（[10,15]: 6次, [15,20]: 2次, [10,20]: 2次）
```

### 2.3 最终数据集

```
xf_pack_list = [
  [factors, srcLU_onehot, diff, idx_label, changed_flag],  # 原始样本
  [factors', srcLU_onehot', diff', idx_label, changed_flag],  # 增强样本 1
  ...
]
```

---

## 3. 训练循环（training_pipeline）

### 3.1 Batch 构造

```
DataLoader 从 xf_pack_list 中采样 batch_size=8 个 pack
→ data_pack = [factors_batch, srcLU_batch, diff_batch, idx_labels, changed_flags]
  - factors_batch: (8, 4, 128, 128)
  - srcLU_batch:   (8, 6, 128, 128)
  - diff_batch:    (8, 1, 128, 128)
```

### 3.2 前向传播（process_batch）

```
inputs = torch.cat([factors_src, lu_src], dim=1)
  # (8, 10, 128, 128)

diff = remove_lines_forbatch(diff)
  # 去除 batch 中每张图的线状伪变化

outputs = model(inputs)
  # (8, 1, 128, 128) — 原始 logit 输出

loss = cal_loss(outputs, diff)
  # Focal_loss × 0.5 + DICE_loss × 0.5
```

### 3.3 反向传播

```
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 3.4 学习率调整

```python
if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
    adjust_learning_rate(optimizer, 0.8)  # lr *= 0.8
```

### 3.5 模型保存

```
每个 epoch 结束:
  if val_loss < best_val_loss:
      save(best_model.pth)              # 最佳模型
      epochs_since_improvement = 0
  else:
      epochs_since_improvement += 1

  save(latest_checkpoint.pth)            # 断点续训用

  if (epoch+1) % 20 == 0:
      save(epoch_{n}.pth)                # 定期备份
```

---

## 4. 验证循环（valid_pipeline）

### 4.1 验证流程

```
加载 checkpoint → model.load_state_dict()
model.eval()

for data_pack in val_dataloader:
    loss, outputs, diff = process_batch(data_pack, model, device)
    outputs = F.sigmoid(outputs)        # logit → 概率
    dataset_metric.update(outputs, diff) # 累积指标
```

### 4.2 指标计算

```
dataset_metric 累积所有 batch 的 TP/FP/FN/TN
→ 计算: OA, Precision, Recall, F1, IoU, Kappa, FOM
```

### 4.3 可视化输出

每个 batch 保存一张可视化图：

```python
visualize_tensor_list({
    'DEM':     data_pack[0][0, 0],    # 驱动因子 1
    'DL':      data_pack[0][0, 1],    # 驱动因子 2
    'GDP':     data_pack[0][0, 2],    # 驱动因子 3
    'RK':      data_pack[0][0, 3],    # 驱动因子 4
    'tgt':     data_pack[2],          # ground truth diff
    'outputs': outputs,               # 模型预测概率
    'diff':    diff,                  # 实际差分
})
```

---

## 5. 输出文件结构

```
checkpoints/{model_name}/{run_timestamp}/
  ├── best_model.pth          # 最佳验证损失对应的权重
  ├── latest_checkpoint.pth   # 最新 checkpoint（含优化器状态）
  ├── epoch_20.pth            # 定期备份
  ├── epoch_40.pth
  ├── val_metrics_{pack}.json # 验证指标 JSON
  └── val_pic_{pack}/         # 验证可视化图片
      ├── val_0.png
      ├── val_1.png
      └── ...

logs/{model_name}/{run_timestamp}/
  └── events.out.tfevents.*   # TensorBoard 日志
```

---

## 6. 数据流总览

```
_tif 文件 (DEM/DL/GDP/RK/LU)
    ↓ get_year_data()
(num_samples, 5, 128, 128) × 3 年
    ↓ create_training_data_pack()
[factors(4,128,128), srcLU_onehot(6,128,128), diff(1,128,128), idx_label, changed_flag]
    ↓ apply_transform() (仅变化样本)
xf_pack_list (含增强样本)
    ↓ DataLoader (batch_size=8)
data_pack = [factors_batch(8,4,128,128), srcLU_batch(8,6,128,128), diff_batch(8,1,128,128), ...]
    ↓ process_batch()
inputs = cat([factors, lu], dim=1)  → (8, 10, 128, 128)
    ↓ model(inputs)
outputs (8, 1, 128, 128) — logit
    ↓ cal_loss()
loss = 0.5 × Focal + 0.5 × DICE
    ↓ backward + step
更新模型权重
    ↓ 验证
metrics = {OA, Precision, Recall, F1, IoU, Kappa, FOM}
    ↓ 保存
best_model.pth → Stage 2 使用
```
