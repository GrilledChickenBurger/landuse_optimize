# Datasets

## 这个模块做什么

datasets 是项目的数据入口——所有 pipeline 都从这里加载数据。磁盘上存储着大量 128×128 的栅格切片（驱动因子、LU 类型、ES 数据等），datasets 负责把它们组织成统一格式的数据样本，供 DataLoader 送入模型训练或优化器求解。

项目两阶段需要**完全不同格式**的数据：

- Stage-1（训练变化检测模型）：需要多个年份的时间对 + 变化标签 + 数据增强
- Stage-2（SA/GA 优化）：需要单年份的驱动因子 + ES 数据 + 约束掩膜 + 目标年 LU

datasets 通过一个统一的工厂函数 `get_dataset(type, config)` 区分这两个场景，返回对应格式的样本。

## 模块在项目中的位置

```
磁盘 .tif 栅格
      │
      ▼
┌────────── datasets ──────────┐
│  get_dataset("training")     │ →  [factors, srcLU_oh, diff, label, changed_flag]
│  get_dataset("optimize")     │ →  [factors, restrict, es_data, oriLU, tgtLU, label]
│  get_dataset("statistic")    │ →  {"factors": ..., "src_lu": ...}
└──────────────┬───────────────┘
               │
               ▼
    torch.utils.data.DataLoader
               │
               ▼
  ┌───────── training ────────┐    ┌────────── optimize ─────────┐
  │  pipelines/train.py       │    │  pipelines/run_runner.py    │
  │  → ANN 模型               │    │  → SA/GA 优化器             │
  └───────────────────────────┘    └─────────────────────────────┘
```

datasets 只负责加载和组装数据，不参与模型推理或优化计算。

## 核心设计：一口进、三样出

所有数据加载都通过 `base_dataloader.py` 中的 `get_dataset(type, data_config)` 工厂函数完成：

```python
from datasets.base_dataloader import get_dataset

dataset = get_dataset("training", config.data)     # 训练数据
dataset = get_dataset("optimize", config.data)     # 优化数据
dataset = get_dataset("statistic", config)          # 统计数据
```

内部有三个 Dataset 类，都继承自 `BaseDataset(torch.utils.data.Dataset)`：

| 类                 | type 参数     | 返回格式   | 用于                       |
| ------------------ | ------------- | ---------- | -------------------------- |
| `TrainingDataset`  | `"training"`  | 5 元素列表 | 训练/验证 ANN 变化检测模型 |
| `OptimizeDataset`  | `"optimize"`  | 6 元素列表 | SA/GA 优化 + 消融实验      |
| `StatisticDataset` | `"statistic"` | 2 键字典   | 计算驱动因子归一化参数     |

`BaseDataset` 的核心逻辑非常简单：子类负责调用加载函数填充 `self.xf_pack_list`（数据包列表），基类负责 `__len__` 和 `__getitem__`。三种 Dataset 的加载逻辑分别在 `training_dataloader.py` 和 `optimize_dataloader.py` 中实现。

此外，`create_dataset.py` 提供数据集构建辅助函数：`create_maxent_dataset`（构建 MaxEnt 文化服务训练数据）、`create_SA_dataset`（构建 SA 优化评估数据）和 `split_SA_dataset`（数据拆分）。

## TrainingDataset — 训练数据

对应 Stage-1：训练变化检测模型。

### 数据来源

从三个年份（10、15、20）加载驱动因子和 LU 类型，构造**时间对**进行变化检测：

- 源年(10) → 目标年(15)：pack `[10,15]`
- 源年(15) → 目标年(20)：pack `[15,20]`
- 源年(10) → 目标年(20)：pack `[10,20]`

具体加载哪些时间对由 `data.pack` 配置决定（如 `[[10,15]]` 只加载 10→15）。`pack: [[-1,-1]]` 表示加载全部三个时间对。

### 样本格式

每个样本是一个包含 5 个元素的列表：

```
[
  factors:      torch.Tensor  (F, 128, 128)       # 源年驱动因子（len(factor_list) 个通道）
  srcLU_oh:     torch.Tensor  (6, 128, 128)       # 源年 LU 的 one-hot 编码
  diff:         torch.Tensor  (1, 128, 128)       # 变化标签（0/1，tgt_LU ≠ src_LU）
  idx_label:    str                                # 样本标识，如 "10_42_5"
  changed_flag: int                                # 0 或 1，该样本是否发生了变化
]
```

- **factors**：通道数 = `len(factor_list)`。只包含源年的因子，不含 LU
- **srcLU_oh**：6 通道 one-hot，顺序为：耕地 / 林地 / 草地 / 水域 / 建设用地 / 未利用地
- **diff**：`(src_LU != tgt_LU)` 的二值图，是模型训练的目标
- **changed_flag**：经过过滤后的标签——先 `remove_lines()` 去除面积 < 20 像素的小斑块，再判断变化像素是否 ≥ `PIXEL_NUM_THRESHOLD_ISCHANGE`（默认 30）

### 数据增强

训练时（`split="train"`）对**发生变化**的样本应用增强：

1. `Resize(128)` + `CenterCrop(128, 128)` — 几何一致性
2. `RandomHorizontalFlip()` + `RandomVerticalFlip()` — 随机翻转

由 `apply_transform()` 实现：将 factors、srcLU_oh、diff 拼接为一个大张量 → 统一变换 → 再拆回原始结构，确保各通道组的空间变换一致。不同时间对有不同重复次数（`[10,15]` 重复 6 次，`[15,20]` 和 `[10,20]` 重复 2 次），以平衡训练数据中不同变化尺度的样本数量。

### 归一化

如果 `normalize=True`，对 `ALL_LOG_FACTORS` 中的因子（如 GDP）先取 `log1p`，然后所有因子用预计算的 `FACTOR_PARAMS_DICT` 做 Z-score 归一化（减均值除标准差）。

### only_changed 过滤

`only_changed=True` 时，仅保留 `changed_flag=1` 的样本。默认 `false`。

## OptimizeDataset — 优化数据

对应 Stage-2：SA/GA 搜索最优 LU 方案。

### 数据来源

从单一年份（如 10）加载驱动因子、RA（耕地红线）、LU 和 ES 数据，同时加载 +5 年后的目标年 LU（如 15）。特殊处理：`year="20"` 时目标年与源年相同（没有更晚的数据）。

### 样本格式

每个样本是一个包含 6 个元素的列表：

```
[
  factors:    np.ndarray  (F, 128, 128)   # 驱动因子（len(factor_list) 个通道，不含 RA 和 LU）
  restrict:   np.ndarray  (1, 128, 128)   # 耕地红线 / 不可修改掩膜
  es_data:    np.ndarray  (E, 128, 128)   # 生态系统服务数据（E = len(ES_list)）
  ori_LU:     np.ndarray  (1, 128, 128)   # 源年土地利用
  tgt_LU:     np.ndarray  (1, 128, 128)   # 目标年土地利用（+5 年后的真值）
  idx_label:  str                          # 样本标识，如 "10_42_5"
]
```

- **factors**：加载时自动附加了 RA 和 LU（`cond_list = factor_list + ["RA", "LU"]`），但在样本中 RA 和 LU 被拆出为独立的 `restrict` 和 `ori_LU`，所以 factors 不含它们
- **restrict**：耕地红线掩膜，SA/GA 硬约束的来源——这些像素不可修改
- **es_data**：所有 ES 计算所需的环境数据（RAIN/NJW/ZSL/PODU 等），进入 info_dict 供 calculator 使用
- **ori_LU / tgt_LU**：整数编码的 LU 类型（非 one-hot），tgt_LU 用于事后性能评估
- 数据类型是 `np.ndarray`（不是 `torch.Tensor`），没有数据增强，没有归一化

### 并行加载

支持通过 `ThreadPoolExecutor` 并行读取多个样本（`use_parallel=True`，`max_workers` 控制线程数）。

## StatisticDataset — 统计数据

对应 Step 0：计算驱动因子的归一化参数。

它是 OptimizeDataset 的轻量版：只加载驱动因子和源年 LU，跳过 ES 数据、RA 和目标年 LU，节省约 30% 的 I/O。

样本格式是字典而非列表：

```
{
  "factors":  np.ndarray  (F, 128, 128)   # 驱动因子
  "src_lu":   np.ndarray  (1, 128, 128)   # 源年 LU
}
```

## 训练 vs 优化：数据格式差异

两个阶段的数据需求完全不同，这决定了样本格式的差异：

| 维度             | TrainingDataset                    | OptimizeDataset                 |
| ---------------- | ---------------------------------- | ------------------------------- |
| 项目阶段         | Stage-1：训练变化检测模型          | Stage-2：SA/GA 优化             |
| 每个索引的样本数 | 最多 3 个（三个时间对）            | 1 个（单一年份）                |
| 年份数           | 3 个（10/15/20）                   | 1 个（如 10） + 目标年（如 15） |
| 源 LU 编码       | one-hot（6 通道）                  | 整数编码（1 通道）              |
| 变化信息         | `diff`（二值标签）+ `changed_flag` | `tgt_LU`（完整目标方案）        |
| 约束             | 无                                 | `restrict`（耕地红线）          |
| ES 数据          | 无                                 | `es_data`（多通道环境数据）     |
| 数据类型         | `torch.Tensor`（直接进模型）       | `np.ndarray`（进优化器运行时）  |
| 数据增强         | 有（翻转 + 裁剪 + resize）         | 无                              |
| 归一化           | 可选（Z-score）                    | 无                              |
| 样本筛选         | `only_changed` 可选                | 无筛选                          |
| 并行加载         | 无                                 | 可选                            |

为什么不同：

- 训练需要用 one-hot LU + diff 构造 `(factors, one_hot_LU) → diff` 的监督学习任务，需要数据增强提升泛化
- 优化需要原始的整数 LU（用于 ES 计算和约束判断），需要 RA 和 ES 数据（用于目标函数），不需要增强或归一化
- 优化阶段的模型推理归一化在推理前单独处理

## 磁盘上的数据组织

两个数据根目录遵循相同的结构：

```
_data/
  training_data/                  # trainConfig.data.root
    train/                        # split="train"
      valid_10/
        DEM/   *.tif              # 每个因子/类别一个子目录，下含若干个 128×128 切片
        DL/    *.tif
        GDP/   *.tif
        RK/    *.tif
        LU/    *.tif
      valid_15/  (同上)
      valid_20/  (同上)
    val/                          # split="val"
      valid_10/  (同上)
      valid_15/  (同上)
      valid_20/  (同上)

  optimize_data/                  # optimizerConfig.data.root
    valid_10/
      DEM/    *.tif
      DL/     *.tif
      GDP/    *.tif
      RK/     *.tif
      RA/     *.tif               # 耕地红线（仅优化数据有）
      LU/     *.tif
      RAIN/   *.tif               # ES 数据
      NJW/    *.tif
      ZSL/    *.tif
      PODU/   *.tif
      WATER/  *.tif
      CITY/   *.tif
      GD/     *.tif
    valid_15/  (同上)
    valid_20/  (同上)
```

- 每个 `.tif` 是一个 128×128 的单通道栅格切片
- 文件名包含样本索引（如 `LULC_001_1.TIF`），按索引排序后同索引的切片对应同一地理位置
- `optimize_data` 比 `training_data` 多出：`RA/`（耕地红线）和 7 个 ES 因子目录

## 与其他模块的协作

### 谁调用 datasets

| Pipeline / 脚本                            | type          | 模块                      |
| ------------------------------------------ | ------------- | ------------------------- |
| `main.py → run_train_pipeline()`           | `"training"`  | `pipelines/train.py`      |
| `main.py → run_valid_pipeline()`           | `"training"`  | `pipelines/train.py`      |
| `main.py → run_runner_pipeline()`          | `"optimize"`  | `pipelines/run_runner.py` |
| `main.py → run_experiment_pipeline()`      | `"optimize"`  | `pipelines/experiment.py` |
| `main.py → run_statistic_pipeline()`       | `"statistic"` | `pipelines/statistic.py`  |
| `main.py → run_ablation_pipeline()`        | `"optimize"`  | `pipelines/ablation.py`   |
| `scripts/temp_pred_visualize.py`           | `"optimize"`  | 直接调用                  |
| `scripts/select_representative_samples.py` | `"optimize"`  | 直接调用                  |

### 数据流向

```
TrainingDataset → DataLoader → train.py
  └→ [factors, srcLU_oh, diff, label, changed_flag]
       └→ torch.cat([factors, srcLU_oh], dim=1) → ANN 模型 → 预测 diff
                                                              ↓
                                                         对比真实 diff → loss

OptimizeDataset → DataLoader → run_runner.py
  └→ [factors, restrict, es_data, oriLU, tgtLU, label]
       └→ factors → ANN 模型 → pred_diff
       └→ restrict → SA/GA 约束掩膜
       └→ es_data → info_dict → calculator.cal_es_score
       └→ oriLU → 优化起点
       └→ tgtLU → 事后性能评估

StatisticDataset → get_params() → _data/statistic_20.json
  └→ 计算每个因子的均值/方差 → 供训练阶段归一化使用
```

### 依赖的工具函数

datasets 的加载逻辑依赖 `utils/dataset_utils.py` 中的底层函数：

- `get_year_data(root, year, cond_list, shape)` — 加载一个年份目录下指定因子列表的全部切片
- `one_hot_lu(lu_tensor)` — 将整数 LU 编码转为 6 通道 one-hot
- `apply_norm(factor_tensor, factor_name, shape)` — Z-score 归一化
- `remove_lines(binary_map)` — 去除微小斑块（面积 < 20 像素）
