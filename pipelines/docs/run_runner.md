# Run Runner Pipeline 数据流说明

> 本文档覆盖 `pipelines/run_runner.py` 的完整操作流程，对应 `main.py` 入口 `run_runner_pipeline()`。该 pipeline 是 Stage 2 的核心入口，负责执行单样本或批量 SA/GA 优化。

---

## 一、关键参数

### 1.1 `run_runner_pipeline()`（main.py 入口）

| 参数 | 来源 | 含义 |
|------|------|------|
| `config_path` | 硬编码 `"configs/optimizerConfig.yaml"` | 优化器+模型+数据配置 |
| `idxs` | 硬编码列表，如 `[760, 1903, 2207, 479, 596, 634]` | 待优化样本索引列表；设为 `None` 则跑全部样本 |
| `is_processed` | 硬编码 `False` | 是否断点续传 |
| `processed_timestamp` | 硬编码 `"20251230_212323"` | 断点续传时间戳 |

### 1.2 `pipeline()`（pipelines/run_runner.py 核心）

| 参数 | 类型 | 作用 |
|------|------|------|
| `config_path` | `str` | 配置文件路径 |
| `idxs` | `List[int]` | 待优化样本索引列表；`None` 则跑全部 |
| `is_processed` | `bool` | 是否已处理过（用于断点续传跳过已跑样本）|
| `processed_timestamp` | `str` | 已处理数据的时间戳 |

---

## 二、数据流总览

```
main.py
  ↓ 调用 pipeline(config_path, idxs, is_processed, processed_timestamp)

pipelines/run_runner.py
  ├─ 1. 加载配置
  │     └─ optimizerConfig.yaml → config (Namespace)
  │         ├─ config.runner.type: "SA" | "GA"
  │         ├─ config.runner.scenario: "default" | "enterprise" | "farmer" | "tourist" | "all"
  │         ├─ config.SA / config.GA: 各优化器参数
  │         ├─ config.model: Stage 1 模型配置
  │         └─ config.data: 数据路径、batch_size、shuffle 等
  │
  ├─ 2. 设置设备
  │     └─ cuda (if available & use_gpu) else cpu
  │
  ├─ 3. 加载数据
  │     └─ get_dataset(type="optimize", data_config=config.data)
  │         → DataLoader(batch_size, shuffle)
  │
  ├─ 4. 训练 MaxEnt 模型
  │     └─ train_maxent(root, year, maxent_path)
  │         → 用于 cal_es_score 中的 culture 项
  │         → 若模型文件已存在，内部会跳过训练
  │
  ├─ 5. 解析 runner 配置
  │     ├─ runner_type = config.runner.type ("SA" | "GA")
  │     ├─ use_landscape = config.runner.use_landscape
  │     ├─ optimize_num = config.runner.optimize_num
  │     ├─ type_config = getattr(config, runner_type)  # SA 或 GA 专属配置
  │     ├─ scenario = config.runner.scenario
  │     └─ scenario_list: "all" → [default, enterprise, farmer, tourist]
  │                      其他 → [scenario]
  │
  ├─ 6. 生成时间戳
  │     └─ run_timestamp = now() (新跑) 或 processed_timestamp (续传)
  │
  ├─ 7. 遍历 scenario
  │     ├─ weight_dict = vars(config.runner.weight.{scenario})
  │     ├─ scenario_save_folder = results/{SA|GA}/{timestamp}/{scenario}/
  │     │
  │     └─ 8. 遍历 dataloader
  │           ├─ 断点续传检查: index 已在 processed_samples → skip
  │           ├─ idxs 过滤: index.split('_')[1] 不在 idxs → skip
  │           ├─ process_batch(data_pack, config) → data_dict
  │           │
  │           └─ 9. 创建并运行优化器
  │                 create_runner(
  │                     runner_type, config, common_config,
  │                     model_config, device, data_dict,
  │                     index, save_path, weight_dict, use_landscape
  │                 ) → type_runner
  │                 type_runner.run() → best_plan
  │
  └─ 10. 保存结果
        ├─ best_plan.npy
        ├─ performance.json
        ├─ convergence.json
        ├─ total_score_list.npy
        └── visualize/*.png
```

---

## 三、`process_batch()` 详解

### 3.1 输入：`data_pack`

`data_pack` 来自 `DataLoader`，是一个包含 6 个元素的列表：

| 索引 | 内容 | 维度 | 说明 |
|------|------|------|------|
| `data_pack[0]` | srcFactors（不含 RA/LU） | `(N, factor_num, 128, 128)` | 驱动因子（DEM, DL, GDP, RK） |
| `data_pack[1]` | restrict (RA) | `(N, 1, 128, 128)` | 区域约束（耕地红线等） |
| `data_pack[2]` | es_data | `(N, ES_num, 128, 128)` | ES 数据（RAIN, NJW, ZSL, PODU, WATER, CITY, GD） |
| `data_pack[3]` | src LU | `(N, 1, 128, 128)` | 原始土地利用方案 |
| `data_pack[4]` | tgt LU | `(N, 1, 128, 128)` | 目标年份土地利用方案（仅用于参考） |
| `data_pack[5]` | idx_label | `List[str]` | 样本标识，如 `"10_760_5"` |

### 3.2 输出：`data_dict`

```python
data_dict = {
    'factors':    tensor,   # (1, factor_num, 128, 128) — 全部驱动因子拼接
    'restrict':   tensor,   # (1, 1, 128, 128) — 区域约束
    'es_data':    tensor,   # (1, ES_num, 128, 128) — 全部 ES 数据拼接
    'ori_plan':   tensor,   # (1, 1, 128, 128) — 原始 LU
    'final_plan': tensor,   # (1, 1, 128, 128) — 目标 LU
    # 以下由 get_data_dict 展开:
    'DEM': tensor, 'DL': tensor, 'GDP': tensor, 'RK': tensor,
    'RAIN': tensor, 'NJW': tensor, 'ZSL': tensor,
    'PODU': tensor, 'WATER': tensor, 'CITY': tensor, 'GD': tensor,
}
```

### 3.3 `get_data_dict()` 的作用

从 concat 的张量中按 `factor_list` / `es_list` 的顺序拆分为独立键值对：

```python
data[:, i] → d.unsqueeze(1)  # (1, H, W) → (1, 1, H, W)
```

这样 `cal_es_score()` 等函数可以直接通过键名（如 `"RAIN"`、`"GDP"`）访问对应的单通道张量。

---

## 四、Scenario 遍历逻辑

```python
scenario = config.runner.scenario
if scenario == "all":
    scenario_list = [name for name in vars(config.runner.weight).keys()]
    # 动态从配置读取: ["default", "enterprise", "farmer", "tourist"]
else:
    scenario_list = [scenario]
```

对每个 scenario 提取对应权重：

```python
weight_dict = vars(getattr(config.runner.weight, current_scenario))
# 例: default → {'food': 0.25, 'carbon': 0.25, 'water': 0.25, 'culture': 0.25}
```

**遍历顺序**：外层遍历 scenario → 内层遍历 dataloader 样本。即先跑完一个 scenario 的全部样本，再切到下一个 scenario。

---

## 五、样本数量控制与断点续传

### 5.1 `optimize_num` 控制

```python
if optimize_num != -1 and actual_optimize_num >= optimize_num:
    break
```

| `optimize_num` | 行为 |
|----------------|------|
| `-1` | 优化数据集中的**全部**样本（约 270 个，需 5-7 天） |
| `N > 0` | 每个 scenario 最多优化 N 个样本 |

### 5.2 `idxs` 过滤

```python
if idxs and len(idxs) > 0 and index.split('_')[1] not in [str(idx) for idx in idxs]:
    continue
```

注意：`index` 格式为 `"{year}_{idx}_{delta}"`（如 `"10_760_5"`），取 `split('_')[1]` 得到 `760` 来匹配 `idxs` 列表。

### 5.3 断点续传

```python
# main.py 中的断点续传示例
def run_runner_pipeline():
    config_path = os.path.join("configs", "optimizerConfig.yaml")
    is_processed = True
    processed_timestamp = "20260601_120000"
    idxs = [760, 1903, 2207]
    pipeline(config_path, idxs, is_processed, processed_timestamp)
```

程序会检查 `results/{SA|GA}/{timestamp}/{scenario}/{index}/` 目录是否存在，存在则跳过该样本。

---

## 六、结果目录结构

```
results/
└── SA/ 或 GA/
    └── 20260601_120000/          ← 时间戳
        └── default/              ← scenario
            ├── 10_760_5/         ← 样本标识
            │   ├── best_plan.npy               # 最优 LU 方案 (128, 128)
            │   ├── performance.json            # 性能指标（ES、IoU、变化区域）
            │   ├── convergence.json            # 收敛信息（迭代数、最优值、耗时）
            │   ├── total_score_list.npy        # 每步目标函数值序列
            │   └── visualize/                  # 过程可视化
            │       ├── step_500.png
            │       ├── step_1000.png
            │       └── final.png
            ├── 10_1903_5/
            └── ...
        └── enterprise/
            └── ...
```

### 6.1 `performance.json` 示例

```json
{
  "best_plan_es_score": 0.7234,
  "ori_plan_es_score": 0.6512,
  "food_score": 0.8123,
  "carbon_score": 0.6789,
  "water_score": 0.7012,
  "culture_score": 0.6543,
  "iou_with_pred": 0.4567,
  "change_area_pixels": 234
}
```

### 6.2 `convergence.json` 示例

```json
{
  "final_best": -0.1234,
  "wall_time_sec": 345.67,
  "total_iter": 5000,
  "guided": true,
  "disable_early_stop": false
}
```

---

## 七、与 Ablation Pipeline 的区别

| 特性 | Run Runner Pipeline | Ablation Pipeline |
|------|---------------------|-------------------|
| 入口 | `run_runner_pipeline()` | `run_ablation_pipeline()` |
| 核心函数 | `pipeline()` | `ablation_pipeline()` |
| 单次/重复 | 每个样本只跑 **1 次** | 每个样本跑 **n_repeats 次** |
| Scenario | 支持多 scenario 遍历 | 固定 single scenario |
| 实验组 | 无（标准流程） | A/B/C 三组消融对比 |
| 结果路径 | `results/{SA\|GA}/{ts}/{scenario}/` | `results/ablation/{ts}/{exp_name}/` |
| score_cfg | 直接加载，不覆写 | 深拷贝 + overrides 覆写 |
| 断点续传 | 按样本目录检查 | 按 `run_r/convergence.json` 检查 |

---

## 八、典型调用示例

### 8.1 跑指定样本（单 scenario）

```python
def run_runner_pipeline():
    config_path = os.path.join("configs", "optimizerConfig.yaml")
    idxs = [760]  # 只跑样本 760
    pipeline(config_path, idxs, is_processed=False, processed_timestamp=None)
```

### 8.2 跑全部样本（所有 scenario）

```python
# optimizerConfig.yaml 中:
# runner.scenario: "all"
# runner.optimize_num: -1

def run_runner_pipeline():
    config_path = os.path.join("configs", "optimizerConfig.yaml")
    pipeline(config_path, idxs=None, is_processed=False, processed_timestamp=None)
```

### 8.3 断点续传

```python
def run_runner_pipeline():
    config_path = os.path.join("configs", "optimizerConfig.yaml")
    is_processed = True
    processed_timestamp = "20260601_120000"
    idxs = [760, 1903, 2207]
    pipeline(config_path, idxs, is_processed, processed_timestamp)
```
