# Stage 2 随机优化——操作手册

> 本文档覆盖 Stage 2 的完整操作流程：单样本优化、批量优化、策略实验、结果查看。

---

## 1. 前置检查清单

运行前请确认以下事项：

1. **模型权重就绪**：`configs/optimizerConfig.yaml` 中 `model.model_path` 指向正确的 Stage 1 预训练权重文件（如 `checkpoints\\attunet_best_260212.pth`）。
2. **优化数据就绪**：`_data/optimize_data/` 目录下包含 `valid_10/` 和 `valid_15/` 子目录，每个子目录含 `DEM/`、`DL/`、`GDP/`、`RK/`、`RA/`、`LU/` 及 ES 因子文件夹。
3. **MaxEnt 模型**：首次运行会自动训练 MaxEnt 模型并保存到 `checkpoints/maxent-model_sklearn_10.ela`。
4. **GPU 可用**：建议使用 GPU，单样本 SA 约需 5-15 分钟，GA 约需 10-30 分钟。
5. **依赖库**：PyTorch、scipy、numpy、rasterio、elapid（MaxEnt 用）。

---

## 2. 配置文件

### 2.1 核心配置（`configs/optimizerConfig.yaml`）

```yaml
runner:
  type: "SA"              # "SA" 或 "GA"
  use_landscape: true     # 是否启用景观破碎化约束
  optimize_num: 3         # 优化样本数量（-1 表示全部）
  scenario: "default"     # 情景: default|enterprise|farmer|tourist|all
  weight:
    default: {food: 0.25, carbon: 0.25, water: 0.25, culture: 0.25}
    enterprise: {food: 0.0, carbon: 0.5, water: 0.5, culture: 0.0}
    farmer: {food: 0.8, carbon: 0.1, water: 0.1, culture: 0.0}
    tourist: {food: 0.0, carbon: 0.0, water: 0.0, culture: 1.0}

model:
  type: "attunet"
  model_path: "checkpoints\\attunet_best_260212.pth"
```

### 2.2 SA 参数

```yaml
SA:
  T_start: 100            # 初始温度
  T_end: 0.1              # 终止温度
  cooling_rate: 0.9995    # 降温率
  max_noimprove_iter: 1000  # 最大无改进迭代数
  max_beststuck_iter: 3000  # 最优解停滞阈值
  print_frequency: 100
  visualize_frequency: 500
  save_path: "results\\SA"
```

### 2.3 GA 参数

```yaml
GA:
  ori_pop_size: 30        # 初始种群大小
  final_pop_size: 8       # 最终种群大小
  max_gen: 3000           # 最大迭代代数
  crossover_rate: 0.8
  mutation_rate: 1.0
  strategy:
    parent_pool: "elite"       # "all" 或 "elite"
    elite_ratio_for_pool: 0.2  # 父本来源的前 20%
    parent_selection: "roulette"  # "roulette" 或 "tournament"
    elite_ratio: 0.1
    non_crossover_handling: "copy"  # "copy" 或 "copy_mutate"
    elite_mutation: false
  save_path: "results\\GA"
```

---

## 3. 运行单样本优化

### 3.1 配置入口函数

```python
# main.py 中：
def run_runner_pipeline():
    config_path = os.path.join("configs", "optimizerConfig.yaml")
    is_processed = False
    processed_timestamp = None
    idxs = [760]  # 指定样本索引
    pipeline(config_path, idxs, is_processed, processed_timestamp)
```

### 3.2 运行全部样本

```python
def run_runner_pipeline():
    config_path = os.path.join("configs", "optimizerConfig.yaml")
    pipeline(config_path, idxs=None, is_processed=False)  # idxs=None 跑全部
```

### 3.3 断点续传

若中途退出，可用 `is_processed=True` 跳过已完成的样本：

```python
def run_runner_pipeline():
    config_path = os.path.join("configs", "optimizerConfig.yaml")
    is_processed = True
    processed_timestamp = "20260601_120000"  # 上次运行的时间戳
    idxs = [479, 596, 760, 634, 1903, 2207]
    pipeline(config_path, idxs, is_processed, processed_timestamp)
```

程序会检查 `results/{SA|GA}/{timestamp}/{scenario}/{idx}/` 目录是否存在，存在则跳过。

---

## 4. 运行批量策略实验

遍历所有 GA 策略变体 × 所有情景：

```python
# main.py 中：
def run_experiment_pipeline():
    config_path = os.path.join("configs", "optimizerConfig.yaml")
    idxs = [479, 596, 760, 634, 1903, 2207]
    experiment_pipeline(config_path, idxs, is_processed=False)
```

该函数会：
1. 遍历 `GA.strategy` 中定义的所有策略变体
2. 遍历所有 scenario
3. 对每个组合运行 GA 优化
4. 结果保存在 `results/GA/{timestamp}/{scenario}_{strategy}/` 下

---

## 5. 运行输出

### 5.1 控制台输出

```
================= 开始随机优化... =================
Configuration loaded from configs/optimizerConfig.yaml
Using device: cuda

================= 随机优化类型: 【SA】 =================
  优化样本数量: 3
  使用景观破碎化约束: True

================= 当前 scenario: 【default】 =================
  权重配置: {'food': 0.25, 'carbon': 0.25, 'water': 0.25, 'culture': 0.25}

Processing sample 1/270
================= SA Initing... =================
Step 100: T=95.12, cur_score=-0.xxxx, best_score=-0.xxxx, no_improve=0
Step 200: T=90.48, cur_score=-0.xxxx, best_score=-0.xxxx, no_improve=12
...
SA 完成! 总迭代: 5000, 最优值: -0.xxxx, 耗时: 345.67s
```

### 5.2 结果目录

```
results/{SA|GA}/{timestamp}/{scenario}/
  └── {year}_{idx}_{delta}/           # 如 10_760_5
      ├── best_plan.npy               # 最优 LU 方案 (128, 128)
      ├── performance.json            # 性能指标
      ├── convergence.json            # 收敛信息
      ├── total_score_list.npy        # 每步目标函数值
      └── visualize/                  # 过程可视化
          ├── step_500.png
          ├── step_1000.png
          └── final.png
```

### 5.3 performance.json 示例

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

---

## 6. 查看优化结果

### 6.1 可视化 LU 方案对比

使用 `pipelines/results_analysis.py` 中的 `sample_visualization_pipeline`，对比原始方案、最优方案和真实方案的空间分布。该 pipeline 只负责调度，具体样本图生成逻辑在 `utils.analysis_plot_utils.py` 中实现。

### 6.2 收敛曲线分析

读取 `total_score_list.npy` 绘制目标函数随迭代的变化曲线：

```python
import numpy as np
import matplotlib.pyplot as plt

scores = np.load("results/SA/20260601_120000/default/10_760_5/total_score_list.npy")
plt.plot(scores)
plt.xlabel("Iteration")
plt.ylabel("Objective Function F(S)")
plt.title("Convergence Curve")
plt.savefig("convergence.png")
```

### 6.3 批量结果汇总

使用 `pipelines/results_analysis.py` 中的 `single_experiment_stat_pipeline` 或 `comparison_pipeline` 汇总多样本结果。pipeline 负责组织入口，业务级出图在 `utils.analysis_plot_utils.py`，指标读取、缓存和聚合在 `utils.analysis_utils.py`。

---

## 7. 常见问题

### Q1: 如何切换 SA 和 GA？

修改 `optimizerConfig.yaml` 中 `runner.type`：

```yaml
runner:
  type: "SA"   # 改为 "GA" 使用遗传算法
```

### Q2: 如何同时跑多种情景？

```yaml
runner:
  scenario: "all"  # 遍历 default, enterprise, farmer, tourist
```

### Q3: 单样本运行时间太长怎么办？

- SA：降低 `max_beststuck_iter`（如 3000→1000），或增大 `cooling_rate`（如 0.9995→0.999）
- GA：降低 `max_gen`（如 3000→1000），或减小 `ori_pop_size`（如 30→15）

### Q4: 优化结果与 pred_diff 的 IoU 很低怎么办？

- 检查 Stage 1 模型是否正确训练（验证指标是否合理）
- 确认 `model_path` 指向正确的权重文件
- IoU 低也可能是正常现象——优化器在 ES 和景观约束下可能偏离预测区域

### Q5: GA 策略实验如何自定义？

编辑 `optimizerConfig.yaml` 中的 `GA.strategy` 部分，修改选择方式、精英比例等参数。可参考 `runner/GA.py` 中 `GeneticAlgorithmStrategy` 的文档字符串。

### Q6: optimize_num 设为 -1 有什么效果？

`optimize_num: -1` 表示优化数据集中的全部样本（约 270 个变化样本）。注意全量运行需 5-7 天。

### Q7: 如何禁用景观破碎化约束？

```yaml
runner:
  use_landscape: false
```

禁用后目标函数变为：`F(S) = change_score - λ_ES × es_score`。
