# Stage 1 变化检测模型训练——操作手册

> 本文档覆盖 Stage 1 的完整操作流程：训练、验证、断点续训、结果查看。

---

## 1. 前置检查清单

运行前请确认以下事项：

1. **训练数据就绪**：`_data/training_data/` 目录下包含 `train/` 和 `val/` 子目录，每个子目录含 `valid_10/`、`valid_15/`、`valid_20/` 三个年份的数据。
2. **驱动因子完整**：每个年份目录下需有 `DEM/`、`DL/`、`GDP/`、`RK/`、`LU/` 五个因子文件夹，内含 `.tif` 文件。
3. **GPU 可用**：`configs/trainConfig.yaml` 中 `device.use_gpu: true`，且系统有 CUDA 支持的 GPU。
4. **依赖库**：PyTorch（含 CUDA）、torchvision、numpy、rasterio（读取 TIF）。

---

## 2. 训练模型

### 2.1 配置文件

编辑 `configs/trainConfig.yaml`，关键参数：

```yaml
training:
  model_name: "attunet"    # 可选: attunet, unet, fcn, segformer, transunet
  num_epochs: 100

data:
  pack: [[10, 15]]         # 训练数据时间范围
  batch_size: 8
  only_changed: false      # true: 仅用变化样本; false: 全部样本
  normalize: true          # 是否对驱动因子标准化

optimizer:
  default:
    learning_rate: 0.0001
```

### 2.2 启动训练

```python
# main.py 中：
def run_train_pipeline():
    config_path = os.path.join("configs", "trainConfig.yaml")
    training_pipeline(config_path)

# 取消 main.py 底部 run_train_pipeline() 的注释，注释掉其他 pipeline
```

### 2.3 训练输出

```
Epoch [1/100], Batch [0/xxx], Loss: 0.xxx(0.xxx), Batch Time: 0.xxx(0.xxx)
Epoch [1/100], Train Loss: 0.xxxx
Epoch [1/100], Val Loss: 0.xxxx
Best model saved: checkpoints/attunet/20260601_120000/best_model.pth (Val Loss: 0.xxxx)
```

### 2.4 训练时间参考

| 模型 | 约 100 epoch 耗时（GPU） |
|------|------------------------|
| U-Net | ~30 分钟 |
| AttU-Net | ~45 分钟 |
| FCN | ~20 分钟 |
| SegFormer | ~1 小时 |
| TransUNet | ~1.5 小时 |

---

## 3. 断点续训

若训练中断，可从 checkpoint 恢复：

```python
def run_train_pipeline():
    config_path = os.path.join("configs", "trainConfig.yaml")
    checkpoint_path = os.path.join("checkpoints", "attunet", "20260601_120000", "latest_checkpoint.pth")
    training_pipeline(config_path, checkpoint_path)
```

程序会自动：
- 加载模型权重和优化器状态
- 从中断的 epoch 继续训练
- 恢复 `epochs_since_improvement` 计数

---

## 4. 验证模型

### 4.1 启动验证

```python
# main.py 中：
def run_valid_pipeline():
    config_path = os.path.join("configs", "trainConfig.yaml")
    model_checkpoint_path = os.path.join("checkpoints", "attunet", "20260601_120000", "best_model.pth")
    valid_pipeline(config_path, model_checkpoint_path, only_changed=False, datasets=["val"])
```

### 4.2 验证参数

| 参数 | 说明 |
|------|------|
| `model_checkpoint_path` | 模型权重文件路径（`best_model.pth` 或 `epoch_*.pth`） |
| `only_changed` | `True`: 仅验证有变化的样本; `False`: 全部样本 |
| `datasets` | `["val"]` 验证集, `["train"]` 训练集, `["train", "val"]` 两者都验证 |

### 4.3 验证输出

```
================= 验证数据集: val =================
开始验证 xxx 个batch...
Batch [0/xxx], Loss: 0.xxxx(0.xxxx)
Batch [1/xxx], Loss: 0.xxxx(0.xxxx)
...
val Loss: 0.xxxx
OA: 0.9xxx, Precision: 0.8xxx, Recall: 0.7xxx, F1: 0.8xxx, IoU: 0.6xxx, Kappa: 0.7xxx
val 指标已保存: checkpoints/attunet/20260601_120000/val_metrics_10_15.json
```

---

## 5. 查看变化样本索引

若需知道哪些样本发生了变化（用于 Stage 2 优化）：

```python
# main.py 中：
def run_get_changed_indices_pipeline():
    config_path = os.path.join("configs", "trainConfig.yaml")
    datasets = ["train", "val"]
    changed_indices = get_changed_indices(config_path, datasets)
    print(f"所有数据集共有 {len(changed_indices)} 个变化样本, 索引为: {changed_indices}")
```

---

## 6. TensorBoard 可视化

训练过程中 loss 曲线实时记录到 TensorBoard：

```bash
tensorboard --logdir=logs/attunet
```

浏览器打开 `http://localhost:6006`，查看：
- `Loss/train_batch`: 每个 batch 的训练损失
- `Loss/train_epoch`: 每个 epoch 的平均训练损失
- `Loss/val_epoch`: 每个 epoch 的验证损失

---

## 7. 输出文件说明

```
checkpoints/{model_name}/{run_timestamp}/
  ├── best_model.pth          # 最佳验证损失对应的模型权重
  │                            # → Stage 2 使用此文件
  ├── latest_checkpoint.pth   # 最新 checkpoint（含 epoch、优化器状态）
  ├── epoch_20.pth            # 第 20 epoch 备份
  ├── epoch_40.pth            # 第 40 epoch 备份
  ├── val_metrics_10_15.json  # 验证指标
  └── val_pic_10_15/          # 验证可视化图片目录
      ├── val_0.png
      └── ...

logs/{model_name}/{run_timestamp}/
  └── events.out.tfevents.*   # TensorBoard 日志
```

---

## 8. 常见问题

### Q1: 训练 loss 不下降怎么办？

- 检查数据路径是否正确，`.tif` 文件是否完整
- 尝试降低学习率（如 `0.00005`）
- 检查 `only_changed` 设置——若为 `true` 但变化样本太少，可改为 `false`

### Q2: 如何切换模型架构？

修改 `trainConfig.yaml` 中 `training.model_name`：

```yaml
training:
  model_name: "unet"  # attunet, unet, fcn, segformer, transunet
```

SegFormer 使用独立优化器配置（AdamW），无需额外修改。

### Q3: 如何使用不同时间跨度的数据？

修改 `data.pack`：

```yaml
data:
  pack: [[15, 20]]      # 仅 2015→2020
  # pack: [[10, 20]]    # 仅 2010→2020
  # pack: [[10, 15], [15, 20]]  # 两个 5 年跨度
  # pack: [[-1, -1]]    # 全部时间跨度
```

### Q4: best_model.pth 和 latest_checkpoint.pth 有什么区别？

- `best_model.pth`：仅包含 `model.state_dict()`，是验证集损失最低时的权重
- `latest_checkpoint.pth`：包含 `model_state_dict`、`optimizer_state_dict`、`epoch`、`loss` 等，用于断点续训

Stage 2 使用 `best_model.pth`。

### Q5: 验证时出现 CUDA out of memory？

将 `validation.batch_size` 设为 1（默认已是 1），或改用 CPU 验证（`device.use_gpu: false`）。
