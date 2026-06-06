# Pipelines Module

## 这个模块做什么

pipelines 是项目的**流程编排层**。项目本身分为两阶段：

1. **Stage-1：训练变化检测模型** —— 用驱动因子 + 当前 LU 训练 ANN，预测每个像素是否会变化，输出 `pred_diff`
2. **Stage-2：随机优化搜索** —— 以 `pred_diff` 为先验，用 SA/GA 搜索更优的土地利用方案

pipelines 把这两阶段串成可执行的工作流：从统计归一化参数、训练模型、验证模型，到运行优化器、批量 GA 策略实验、消融实验和结果分析。它不直接实现模型结构、目标函数或优化算法，而是负责**按配置加载数据、调用模块、组织循环、保存结果**。

```
Step 0: statistic.py      → 统计驱动因子均值/方差
          ↓
Step 1: train.py          → 训练/验证 ANN 变化检测模型
          ↓
Step 2: run_runner.py     → 调用 SA/GA 优化器搜索 best_plan
          ├── experiment.py       → GA 策略组合实验
          └── ablation.py         → 引导/参数/随机约束消融
          ↓
Step 3: results_analysis.py → 汇总结果并出论文图
```

所有正式 pipeline 都从 `main.py` 中的入口函数启动。项目没有 CLI 参数解析，路径和样本索引通常在 `main.py` 或 YAML 配置中写死。

## 模块在项目中的位置

```
main.py
  │
  ├── run_statistic_pipeline()  ──→  pipelines/statistic.py
  │                                  └→ datasets(statistic) → _data/statistic_*.json
  │
  ├── run_train_pipeline()      ──→  pipelines/train.py
  │                                  └→ datasets(training) + models → checkpoints/ logs/
  │
  ├── run_valid_pipeline()      ──→  pipelines/train.py::valid_pipeline
  │                                  └→ checkpoint + validation set → metrics + pic
  │
  ├── run_runner_pipeline()     ──→  pipelines/run_runner.py
  │                                  └→ datasets(optimize) + runner(SA/GA) → results/
  │
  ├── run_experiment_pipeline() ──→  pipelines/experiment.py
  │                                  └→ 多 GA.strategy → results/GA/<ts>/<experiment>/
  │
  ├── run_ablation_pipeline()   ──→  pipelines/ablation.py
  │                                  └→ score_cfg 覆写 + 消融矩阵 → results/ablation/
  │
  ├── run_tornado_pipeline()    ──→  pipelines/tornado.py
  │                                  └→ 参数粗筛 → results/ablation/
  │
  ├── run_sensitivity_pipeline()──→  pipelines/tornado.py
  │                                  └→ 一维线搜索 → results/ablation/
  │
  ├── run_ablation_plots_pipeline() ──→  pipelines/ablation_plots.py
  │
  └── run_tornado_plots_pipeline()  ──→  pipelines/tornado_plots.py
```

pipelines 依赖几乎所有底层模块：

- `datasets/`：加载 training / optimize / statistic 三类数据
- `models/`：创建 ANN 变化检测模型
- `runner/`：创建 SA/GA 优化器
- `calculator/`：通过 runner 间接参与目标函数计算
- `utils/`：配置加载、训练工具、性能评估、可视化、出图
- `libs/`：读取常量、因子列表、LU/ES 类型定义

## 文件一览

| 文件 | 类型 | 对应阶段 | 用途 |
|---|---|---|---|
| `__init__.py` | helper | — | 统一导出常用 pipeline 入口，使用 `try/except ImportError` 容错 |
| `statistic.py` | pipeline | Step 0 | 统计驱动因子的均值/标准差，为归一化提供参数 |
| `train.py` | pipeline | Step 1 | 训练 ANN 变化检测模型、验证 checkpoint、导出变化样本索引 |
| `run_runner.py` | pipeline | Step 2 | 对指定样本运行 SA 或 GA 优化器 |
| `experiment.py` | pipeline | Step 2 实验 | 批量跑 GA 策略组合，并汇总各策略平均表现 |
| `experiment_base.py` | 基类 | 消融实验/参数扫描 | `BaseSweepPipeline`：参数扫描类实验的公共骨架（配置加载、数据准备、断点续传、循环执行） |
| `ablation.py` | pipeline | 消融实验 | `AblationPipeline`：跑稳定性实验与 A/B/C 多次重复消融实验 |
| `tornado.py` | pipeline | 参数扫描 | `TornadoPipeline`：参数粗筛和一维敏感度线搜索 |
| `ablation_plots.py` | 顶层编排 | 消融出图 | 调度消融实验结果出图（数据通过 `ExperimentResultStore` 获取） |
| `tornado_plots.py` | 顶层编排 | 参数扫描出图 | 调度 tornado / sensitivity 出图（数据通过 `ExperimentResultStore` 获取） |
| `results_analysis.py` | pipeline | Step 3 | 结果分析顶层入口：单实验统计图、样本可视化、SA-GA 对比、模型指标对比 |

## main.py 入口

| main.py 函数 | 调用对象 | 配置文件 | 说明 |
|---|---|---|---|
| `run_statistic_pipeline()` | `statistic.get_params()` | `configs/statisticConfig.yaml` | 计算归一化参数 |
| `run_train_pipeline()` | `train.training_pipeline()` | `configs/trainConfig.yaml` | 训练 ANN 模型 |
| `run_valid_pipeline()` | `train.valid_pipeline()` | `configs/trainConfig.yaml` + 硬编码 checkpoint | 验证已训练模型 |
| `run_get_changed_indices_pipeline()` | `train.get_changed_indices()` | `configs/trainConfig.yaml` | 扫描 train/val 中变化样本 |
| `run_runner_pipeline()` | `run_runner.pipeline()` | `configs/optimizerConfig.yaml` | 运行 SA/GA 优化 |
| `run_experiment_pipeline()` | `experiment.experiment_pipeline()` | `configs/optimizerConfig.yaml` | 跑 GA 策略组合实验 |
| `run_ablation_pipeline()` | `ablation.ablation_pipeline()` | `optimizerConfig.yaml` + `score_cfg.yaml` | 跑稳定性实验或 A/B/C 多次重复消融 |
| `run_tornado_pipeline()` | `tornado.tornado_pipeline(mode='tornado')` | 同上 | 跑参数粗筛 |
| `run_sensitivity_pipeline()` | `tornado.tornado_pipeline(mode='sensitivity')` | 同上 | 跑一维线搜索 |
| `run_ablation_plots_pipeline(ablation_root, sibling_ablation_roots)` | `pipelines.ablation_plots.run_all_ablation_plots()` | 结果目录 | 消融出图；传入 `sibling_ablation_roots` 可合并 SA+GA 的 `stability_runs.csv` 产出 `stability_per_sample.csv` / `stability_summary.csv` / `stability_cv_line.png` |
| `run_tornado_plots_pipeline()` | `pipelines.tornado_plots.run_tornado_plots() / plot_sensitivity_lines()` | 结果目录 | 对 tornado / sensitivity 结果出图 |
| `run_results_analysis_pipeline()` | `results_analysis.*_pipeline()` | 结果目录 / JSON | 对 SA/GA 结果与模型指标出图 |

`main.py` 底部通过手动取消注释选择要跑的 pipeline。当前没有命令行参数，`idxs`、checkpoint 路径、ablation 结果时间戳等也在 `main.py` 内硬编码。

## 总体数据流

### Step 0 → Step 1

```
StatisticDataset
  └→ statistic.py::get_params
      └→ _data/statistic_{year}.json
             ↓
        datasets / runner_utils 中的因子归一化
```

统计阶段只读取驱动因子和源年 LU，用 Welford 算法得到每个 factor 的 `miu / epsilon`。这些参数后续供训练和优化阶段的 ANN 推理使用。

### Step 1 → Step 2

```
TrainingDataset
  └→ train.py::training_pipeline
      └→ checkpoints/<model>/<timestamp>/best_model.pth
             ↓
run_runner.py / experiment.py / ablation.py
  └→ runner_utils.get_pred_diff()
      └→ pred_diff
             ↓
        SA/GA 优化目标函数的变化先验
```

训练阶段输出的是 ANN 权重；优化阶段不会重新训练 ANN，只加载配置中指定的 checkpoint，通过 `pred_diff` 指导 SA/GA 搜索。

### Step 2 → Step 3

```
OptimizeDataset
  └→ run_runner / experiment / ablation
      └→ runner(SA/GA).run()
          └→ results/<SA|GA|ablation>/<timestamp>/...
                 ├── plans_*/best_plan.TIF
                 ├── performance.json / performance_score.json
                 ├── charts/*.png
                 └── convergence.json
                         ↓
                  results_analysis.py / pipelines.ablation_plots.py / pipelines.tornado_plots.py
                         ↓
                  figures/ 与论文图表
```

## Step 0：statistic.py — 统计归一化参数

### 入口

```python
get_params(config_path: str) -> Dict[str, Any]
```

### 这个 pipeline 做什么

训练和推理阶段需要对驱动因子做 Z-score 归一化。`statistic.py` 从全部样本中统计每个因子的均值和标准差，并写成 JSON，供 `datasets` 与 `runner_utils.get_pred_diff()` 使用。

### 核心流程

```
1. get_yaml(config_path) 读取 statisticConfig.yaml
2. get_dataset("statistic", config) 加载 StatisticDataset
3. DataLoader 批量遍历 {factors, src_lu}
4. 对每个 factor：
   - 若在 ALL_LOG_FACTORS 中，先 log1p
   - 根据 src_lu 构造 nodata mask，跳过 LU=0 像素
   - RunningStats.update() 做 Welford 在线统计
5. RunningStats.finalize() 得到 miu / epsilon
6. 写入 config.result_path.format(year=config.year)
```

### RunningStats

`RunningStats` 是 Welford 累加器，支持 batch 级合并，避免一次性把所有像素读入内存。输出字段：

```json
{
  "DEM": {"n": 123456, "miu": 0.0, "epsilon": 1.0},
  "DL":  {"n": 123456, "miu": 0.0, "epsilon": 1.0}
}
```

`epsilon = std + 1e-6`，避免除零。

### 输入输出

| 项 | 内容 |
|---|---|
| 输入配置 | `configs/statisticConfig.yaml` |
| 输入数据 | `_data/optimize_data/valid_{year}/` 中的驱动因子和 LU |
| 输出文件 | `_data/statistic_{year}.json`（由 `result_path` 决定） |
| 下游消费者 | `utils/dataset_utils.apply_norm()`、`utils/runner_utils.get_pred_diff()` |

## Step 1：train.py — 训练与验证变化检测模型

`train.py` 对应项目第一阶段：学习 `(驱动因子 + 当前 LU) → diff` 的二分类语义分割模型。支持 AttU-Net、U-Net、FCN、SegFormer、TransUNet 等模型。

### 入口函数

| 函数 | 用途 |
|---|---|
| `training_pipeline(config_path, checkpoint_path=None)` | 训练模型，支持断点续训 |
| `valid_pipeline(config_path, model_checkpoint_path, idxs=None, datasets=['val'], only_changed=False)` | 验证指定 checkpoint，保存指标和可视化 |
| `get_changed_indices(config_path, datasets=['train','val'])` | 扫描发生变化的样本索引 |
| `process_batch(data_pack, model, device)` | 单 batch 前向、loss 计算 |
| `visualize_outputs(outputs, diff, epoch, step, save_path=None, isvalid=False)` | 输出预测 vs 标签图 |

### training_pipeline 流程

```
1. 读取 trainConfig.yaml
2. get_dataset("training", config.data) 构造训练集
3. 覆写 split/pack/batch_size 构造验证集
4. create_model(model_name, in_ch=len(factor_list)+6, out_ch=1)
5. 根据 model_name 选择 optimizer 配置（optimizer.<model_name> 或 optimizer.default）
6. 创建 checkpoint_dir / log_dir（带 model_name 和 run_timestamp）
7. 若 checkpoint_path 不为空，加载 model / optimizer / epoch / timestamp
8. 每个 epoch：
   - 可按 epochs_since_improvement 调低学习率
   - 每个 batch：cat(factors, srcLU_oh) → model → BCEWithLogitsLoss → backward → gradient clip → step
   - 按频率打印 loss、保存可视化
   - epoch 末跑验证集，更新 best_model
9. 保存 latest_checkpoint / best_model / epoch_N
```

模型输入通道数固定为：

```
in_ch = len(config.data.factor_list) + 6
```

其中 6 是简化 LU 的 one-hot 通道。标签是 `diff = (tgt_LU != src_LU)` 的二值图。

### valid_pipeline 流程

```
1. 读取 trainConfig.yaml
2. create_model 并加载 model_checkpoint_path
3. 对 datasets 列表中的 train / val 逐个验证
4. 可选 only_changed，只验证变化样本
5. 每个 batch：process_batch → sigmoid → metrics.update
6. 保存可视化图和 metrics JSON
```

验证可视化会把 DEM/DL/GDP/RK、目标 diff、模型输出等放在同一张图中，方便检查模型是否真的学到了变化区域。

### 输入输出

| 阶段 | 输入 | 输出 |
|---|---|---|
| 训练 | `configs/trainConfig.yaml` + `TrainingDataset` | `checkpoints/<model>/<timestamp>/best_model.pth`、`latest_checkpoint.pth`、`epoch_*.pth`、TensorBoard 日志、训练可视化 |
| 验证 | checkpoint + `TrainingDataset(split=train/val)` | `<checkpoint_dir>/<split>_metrics_<pack>.json`、`<split>_pic_<pack>/*.png` |
| 索引扫描 | train/val 数据集 | changed indices 列表（打印到终端） |

## Step 2：run_runner.py — 运行 SA/GA 优化器

`run_runner.py` 对应项目第二阶段的标准入口：对选定 optimize 样本运行一个优化器（SA 或 GA），输出最优 LU 方案和性能指标。

### 入口函数

```python
pipeline(config_path, idxs=None, is_processed=False, processed_timestamp=None)
```

辅助函数：

| 函数 | 用途 |
|---|---|
| `process_batch(data_pack, config)` | 将 OptimizeDataset 的 6 元素样本解包为 runner 需要的 `data_dict` |
| `get_data_dict(data, factor_list, prev_data_dict=None)` | 将多通道张量按名称拆成 `{name: [1,1,H,W]}` |

### process_batch 的数据转换

`OptimizeDataset` 返回：

```
[factors, restrict, es_data, ori_LU, tgt_LU, idx_label]
```

`process_batch` 转为：

```python
data_dict = {
    "factors": factors,      # [1,F,128,128]
    "restrict": restrict,    # [1,1,128,128]
    "ori_plan": srcLU,       # [1,1,128,128]
    "final_plan": tgtLU,     # [1,1,128,128]
    "DEM": ..., "DL": ...,  # factor_list 每个因子单独拆出
    "RAIN": ..., "NJW": ... # es_list 每个 ES 因子单独拆出
}
```

runner 后续用这些字段生成 `pred_diff`、构建约束和计算 ES。

### 主流程

```
1. 校验断点续跑参数：is_processed=True 时必须提供 processed_timestamp
2. 读取 optimizerConfig.yaml
3. get_dataset("optimize", config.data) 加载优化数据
4. train_maxent(root, year, maxent_model_path) 训练文化服务 MaxEnt 模型
5. 读取 runner 配置：type(SA/GA)、use_landscape、scenario、weight、common_params
6. scenario == "all" 时遍历所有权重场景，否则只跑一个场景
7. 对每个 scenario：
   - 创建 save_folder = <SA/GA.save_path>/<timestamp>/<scenario>
   - 若断点续跑，读取已完成 idx 目录并跳过
   - 遍历 dataloader：按 idxs 过滤样本
   - create_runner(...) 创建 SA/GA
   - runner.run() 运行优化并保存结果
   - 若 best_plan != ori_plan，actual_optimize_num += 1
   - 达到 optimize_num 后退出
```

### 场景机制

`runner.weight.<scenario>` 定义四项 ES 权重：

| scenario | 含义 |
|---|---|
| `default` | food / carbon / water / culture 均衡 |
| `enterprise` | 企业视角，偏重 carbon / water |
| `farmer` | 农民视角，偏重 food |
| `tourist` | 游客视角，偏重 culture |
| `all` | 依次跑所有权重场景 |

### 输入输出

| 项 | 内容 |
|---|---|
| 输入配置 | `configs/optimizerConfig.yaml` |
| 输入数据 | `OptimizeDataset`：驱动因子、RA、ES 数据、ori_LU、tgt_LU |
| 输入模型 | `config.model.model_path` 指向的 ANN checkpoint |
| 额外模型 | `train_maxent()` 训练/覆盖 `maxent_model_path` |
| 输出目录 | `<SA|GA.save_path>/<timestamp>/<scenario>/<idx_label>/` |
| 输出内容 | runner 负责写入 plans、performance、charts、history、convergence 等 |

## Step 2 实验：experiment.py — GA 策略组合实验

`experiment.py` 与 `run_runner.py` 的结构几乎相同，但多了一层**实验变体循环**。它用于比较不同 GA 策略组合的效果，如父本来源、选择方式、精英比例、未交叉个体处理和精英是否变异。

### 入口

```python
experiment_pipeline(config_path, idxs=None, is_processed=False, processed_timestamp=None)
```

### 实验列表

文件顶部有硬编码的 `experiments` 列表，每项形如：

```python
{
    "name": "GA_A_T_0.1_C_F",
    "parent_pool": "all",
    "parent_selection": "tournament",
    "tournament_k": 2,
    "elite_ratio": 0.1,
    "non_crossover_handling": "copy",
    "elite_mutation": False,
}
```

运行时会把该 dict 转为 Namespace，覆盖 `type_config.strategy`，再传给 `GeneticAlgorithmStrategy`。

### 主流程

```
1. 与 run_runner.py 一样读取 optimizerConfig、加载 OptimizeDataset、训练 MaxEnt
2. 读取 runner.type 对应的 type_config（通常是 GA）
3. 对 experiments 列表逐个循环：
   - type_config.strategy = Namespace(**experiment)
   - 创建 experiment_save_folder = <GA.save_path>/<timestamp>/<experiment_name>
   - 支持按 experiment 目录断点续跑
   - 遍历 dataloader，按 idxs 和已完成目录过滤
   - create_runner(...).run()
   - 收集每个样本的 performance_score.json
   - get_avg_performance() 计算该实验平均表现
   - 写 experiment_save_folder/performance_score.json
4. 全部实验结束后，写 <timestamp>/performance_dict.json
```

### 聚合指标

`get_avg_performance(json_paths, include_change_area=True)` 会从多个样本的 `performance_score.json` 中提取并平均：

- `ES` improvement（四项生态系统服务提升）
- `Change_area` metrics（如果 `include_change_area=True`）

### 输入输出

| 项 | 内容 |
|---|---|
| 输入配置 | `configs/optimizerConfig.yaml` |
| 实验配置 | `experiment.py` 顶部的 `experiments` 列表 |
| 输出目录 | `<GA.save_path>/<timestamp>/<experiment_name>/<idx_label>/` |
| 实验级输出 | `<experiment_name>/performance_score.json` |
| 总表输出 | `<timestamp>/performance_dict.json` |

## 消融实验：ablation.py

`ablation.py` 是对 Stage-2 优化机制的实验平台。它继承自 `experiment_base.BaseSweepPipeline`，复用参数扫描类实验的公共骨架（配置加载、数据准备、断点续传、循环执行、manifest 与 summary 写入），自身只保留消融实验特有差异。

### 入口函数

| 函数 | 用途 |
|---|---|
| `build_experiment_matrix(mode='ablation', n_repeats=1)` | 构造实验矩阵（stability / ablation 两种模式） |
| `ablation_pipeline(opt_cfg_path, score_cfg_path, idxs=None, mode='ablation', n_repeats=1, ...)` | 薄包装：实例化并运行 `AblationPipeline` |
| `class AblationPipeline(BaseSweepPipeline)` | 消融实验主类，覆写 `build_matrix` / `apply_type_config_hook` / `get_random_mask` |

### mode 选项

| mode | 内容 |
|---|---|
| `stability` | 只跑 `A_guided` 多次重复，用于随机优化稳定性统计 |
| `ablation` | 跑 `A_guided` / `B_unguided` / `C_random` 三组多次重复消融 |

### A/B/C 组设置

| 组 | 名称 | 核心变化 | 目的 |
|---|---|---|---|
| A | `A_guided` | 默认 ANN 引导：`guided=True`，change_score 启用 | 完整方法基线 |
| B | `B_unguided` | 去掉 ANN 位置引导，`disable_change_score=True` 启用数量惩罚 | 测试无位置引导但有面积约束时优化器表现 |
| C | `C_random` | 用随机连通块 mask 替代 pred_diff/grid restrict | 测试随机先验 vs ANN 先验 |

`AblationPipeline` 的实现要点：

- `apply_type_config_hook`：ablation 模式下 SA `T_end=1e-6`、GA `max_gen=8000`，强制收敛
- `get_random_mask`：C 组生成随机连通 mask 时，ANN 模型在样本级别缓存，多次 run 复用同一份 `pred_diff`
- 实验循环结束后自动调用 `libs.const.restore_const_defaults()`，避免跨实验状态泄漏
- **stability 模式收尾自动落盘 `stability_runs.csv`**：`ablation_pipeline()` 在 `pipeline.run()` 返回后，会调用 `utils.exp_sweep_utils.write_stability_runs_csv`，扫描所有 `convergence.json` 反查 `_data/representative_samples.json`（取 `n_pred`、`Q1–Q5` 分位标签），写到 `<root>/plots/stability_runs.csv`，字段：`sample_id, n_pred, level, algorithm, run_idx, F_final`。每次 pipeline 跑出一个算法的一半数据；SA + GA 合并由 `run_ablation_plots_pipeline(sibling_ablation_roots=...)` 在出图阶段完成

### 主流程（由 BaseSweepPipeline 实现）

```
1. 读取 optimizerConfig.yaml 与 score_cfg.yaml
2. 加载 OptimizeDataset，训练 MaxEnt
3. AblationPipeline.build_matrix() 得到实验列表
4. 对每个 experiment：
   - 深拷贝 base_score_cfg，按 overrides 覆写 score_cfg
   - apply_const_overrides(score_cfg) 覆写全局常量
   - 深拷贝 type_config，apply_type_config_hook 注入极端收敛设置
   - 按 opt_overrides 覆写 type_config（属性不存在时跳过，避免 GA 模式下 SA 专属参数崩溃）
   - 生成 exp_save_folder = results/ablation/<timestamp>/<exp_name>
   - 遍历样本：process_batch → data_dict
   - get_random_mask 为 C 组生成随机连通 mask（样本级别缓存 ANN）
   - 按 n_repeats 循环运行 runner
   - 写 manifest.json
   - restore_const_defaults() 恢复全局常量
5. 写根目录 ablation_summary.json
```

### 输出目录

```
results/ablation/<timestamp>/
├── ablation_summary.json
├── A_guided/
│   ├── manifest.json
│   ├── run_0/
│   │   ├── 10_479_5/           ← 样本 index
│   │   │   └── ... runner 产物 ...
│   │   └── 10_596_5/
│   ├── run_1/
│   └── ...
├── B_unguided/
│   └── run_0/
│       └── 10_479_5/
│           └── ... runner 产物 ...
└── C_random/
    └── run_0/
        ├── 10_479_5/
        │   ├── random_mask.npy
        │   └── ... runner 产物 ...
        └── 10_596_5/
```

### 与 score_cfg 的关系

普通 pipeline 使用默认评分参数；消融 pipeline 会显式加载 `configs/score_cfg.yaml`，并在实验中覆写其中的字段。这样可以不改 calculator/runner 源码，就完成目标函数参数、约束阈值和常量的敏感性实验。

## 参数扫描：tornado.py

`tornado.py` 与 `ablation.py` 共用 `BaseSweepPipeline`，专门用于参数粗筛（tornado）和一维敏感度线搜索（sensitivity）。

### 入口函数

| 函数 | 用途 |
|---|---|
| `tornado_pipeline(opt_cfg_path, score_cfg_path, idxs=None, mode='tornado', n_repeats=2, top_k_params=None, ...)` | 薄包装：实例化并运行 `TornadoPipeline` |
| `class TornadoPipeline(BaseSweepPipeline)` | 覆写 `build_matrix` / `get_run_seed`，提供确定性随机种子和 tornado 参数矩阵 |
| `run_tornado_pipeline()` | 内置入口：8 个核心参数 × 3 个扰动值的粗筛 |
| `run_sensitivity_pipeline(top_k_params=None)` | 内置入口：对 tornado 筛选出的 Top-K 参数做细粒度扫描 |

### mode 选项

| mode | 内容 |
|---|---|
| `tornado` | 参数粗筛：从 `tornado_utils._TORNADO_PARAMS` 构造 baseline + 各参数扰动实验 |
| `sensitivity` | 一维线搜索：根据 `top_k_params` 列表对每个参数遍历多个取值 |

### 与 ablation 的差异

| 维度 | ablation | tornado |
|---|---|---|
| 实验矩阵 | A/B/C 三组 | baseline + 多参数扰动 |
| 随机种子 | C 组用 hash(index)+r | 全部用 hash(exp_name)+hash(index)+r |
| 收敛设置 | 应用极端设置（T_end=1e-6 / max_gen=8000） | 保持原始配置 |
| C 组 mask | 生成 random_mask | 不使用 |

## 消融/参数扫描出图：ablation_plots.py / tornado_plots.py

这两个文件是**顶层编排层**，与 `results_analysis.py` 处于同一抽象层级，只负责"决定画哪些图"。所有目录扫描、文件读取、指标聚合都通过 `utils.exp_sweep_utils.ExperimentResultStore` 完成，绘图层不直接接触文件系统。

### ablation_plots.py 入口

| 函数 | 用途 |
|---|---|
| `run_all_ablation_plots(ablation_root, groups_for_stability=None, sibling_ablation_roots=None)` | 一站式消融后处理：稳定性叠图、A/B/C 收敛对比、箱线图和统计表；传入 `sibling_ablation_roots` 时会把多个 timestamp（典型为 SA + GA）的 `stability_runs.csv` 合并后产出 per_sample / summary / CV 折线图 |
| `plot_stability_convergence(store, exp_name, index, save_path, smooth_window=50)` | 单样本多次重复收敛曲线叠图 |
| `plot_stability_convergence_all_samples(store, exp_name, save_path, ...)` | 跨样本聚合稳定性曲线 |
| `plot_ablation_convergence(store, idx, save_path, ...)` | A/B/C 三组单样本对比收敛曲线 |
| `plot_boxplot_final(store, save_path, metric='final_best')` | A/B/C 最终目标函数箱线图 |
| `plot_stability_cv_line(per_sample_csv_or_df, save_path, ...)` | 10 样本按 `n_pred` 升序、SA vs GA 的 CV 双折线图（论文 `fig:stability-cv-line`） |

### tornado_plots.py 入口

| 函数 | 用途 |
|---|---|
| `run_tornado_plots(ablation_root, baseline_name='A_baseline')` | 对多个 metric 批量生成 tornado 图 |
| `plot_tornado(store, baseline_name='A_baseline', ...)` | 单 metric 的 tornado 图 |
| `plot_sensitivity_lines(store, metric='final_best', ...)` | 一维敏感度线搜索图 |

### 数据流

```
main.py::run_ablation_plots_pipeline / run_tornado_plots_pipeline
    │
    ▼
pipelines/ablation_plots.py / tornado_plots.py  ← 顶层编排
    │
    ▼  store = ExperimentResultStore(ablation_root)
       store.get_convergence_series(...)
       store.get_ablation_comparison_data(...)
       store.get_knob_deviations(...)
       store.get_sensitivity_data(...)
    │
    ▼
utils/exp_sweep_utils.py        ← 数据准备层
    │
    ▼
utils/visualize_utils.py        ← 底层绘图原语（grouped_box_plot_figure 等）
```

## Step 3：results_analysis.py — 结果分析与出图

`results_analysis.py` 是结果分析的顶层编排入口。它读取 `results/<SA|GA>/<timestamp>/...` 中的性能 JSON 和方案结果，生成 LU/ES 统计图、样本可视化、SA-GA 对比图和模型指标对比图。具体业务级出图逻辑在 `utils.analysis_plot_utils.py`，数据读取/缓存/聚合在 `utils.analysis_utils.py`，底层图表原语在 `utils.visualize_utils.py`。

### 主要 pipeline

| 函数 | 用途 |
|---|---|
| `single_experiment_stat_pipeline(expr_path, plot_types=None, comparison_type='target')` | 对一个 SA/GA 结果目录生成 LU 与 ES 统计图 |
| `sample_visualization_pipeline(expr_path, mode='picked', compare_expr_path=None, **kwargs)` | 筛选/遍历样本并生成空间方案对比图，或并排展示 SA 与 GA 样本图 |
| `comparison_pipeline(sa_expr_path, ga_expr_path, plot_types=None, save_folder=None, comparison_type='target')` | 生成 SA vs GA 的 LU/ES 对比图和合并面积 JSON |
| `model_comparison_pipeline(json_path='model_comparison_data.json', save_dir='figures', save_name='model_comparison_bar.png')` | 根据模型指标 JSON 绘制 ANN 模型指标对比图 |

兼容入口 `complete_pipeline`、`picker`、`all_samples`、`show_SA_GA` 仍保留为薄包装，实际调用上述 pipeline。

### 输入输出

| 项 | 内容 |
|---|---|
| 输入 | `results/<SA|GA>/<timestamp>/<scenario>/<idx>/performance.json` 或 `performance_score.json` |
| 中间 JSON | `total_area.json`、`total_es.json`、`sample_area_*.json`、`sample_es_*.json`，不存在时会从 performance 现算并写回 |
| 输出图 | `figures/<SA|GA>/<timestamp>/...`、`figures/comparison_results/...`、`figures/model_comparison_bar.png` |

该文件适合在实验完成后单独运行；SA/GA 对比只由 `comparison_pipeline` 负责，模型指标对比由独立的 `model_comparison_pipeline` 负责。

## 配置对应关系

| Pipeline | 配置文件 | 加载方式 | 主要字段 |
|---|---|---|---|
| `statistic.py` | `configs/statisticConfig.yaml` | `get_yaml(config_path)` | `root`、`shape`、`factor_list`、`year`、`batch_size`、`result_path` |
| `train.py` | `configs/trainConfig.yaml` | `get_yaml(config_path)` | `device`、`data`、`training`、`validation`、`model`、`optimizer` |
| `run_runner.py` | `configs/optimizerConfig.yaml` | `get_yaml(config_path)` | `device`、`data`、`model`、`runner`、`SA`/`GA` |
| `experiment.py` | `configs/optimizerConfig.yaml` | `get_yaml(config_path)` | 同 run_runner，但覆写 `GA.strategy` |
| `ablation.py` / `tornado.py` | `optimizerConfig.yaml` + `score_cfg.yaml` | `get_yaml` + `load_score_cfg` | runner 配置 + 目标函数/约束 magic number |
| `results_analysis.py` | 无统一 YAML | 函数参数 | `expr_path`、`plot_types`、`save_folder`、`comparison_type` 等 |

所有配置都被转为 `argparse.Namespace`，使用点号访问，如：

```python
config.data.factor_list
config.training.model_name
config.runner.weight.default
config.GA.strategy.parent_selection
```

## 与其他模块的协作

### datasets

| Pipeline | Dataset type | 返回格式 |
|---|---|---|
| `statistic.py` | `"statistic"` | `{"factors", "src_lu"}` |
| `train.py` | `"training"` | `[factors, srcLU_oh, diff, idx_label, changed_flag]` |
| `run_runner.py` / `experiment.py` / `ablation.py` | `"optimize"` | `[factors, restrict, es_data, oriLU, tgtLU, idx_label]` |

### models

`train.py` 通过 `models.create_model()` 创建并训练模型；优化相关 pipeline 只通过 runner 间接加载模型，用于生成 `pred_diff`。

### runner

`run_runner.py`、`experiment.py`、`ablation.py`、`tornado.py` 都通过：

```python
from runner import create_runner
runner = create_runner(runner_type, config=type_config, ...)
best_plan = runner.run()
```

区别在于：

- `run_runner.py`：只跑当前配置中的一个 runner
- `experiment.py`：多次覆写 `GA.strategy`
- `ablation.py` / `tornado.py`：通过 `BaseSweepPipeline` 多次覆写 `score_cfg`、`type_config`、`guided`、`disable_early_stop`、`random_restrict_mask`

### calculator

pipelines 不直接调用 `cal_tgt_function`，而是由 runner 在每次 SA/GA 迭代中调用。pipeline 只负责把 ES 数据、MaxEnt 模型路径、carbon pool 路径、score_cfg 等材料传给 runner。

### utils

| utils 模块 | 在 pipelines 中的作用 |
|---|---|
| `utils.tools` | `get_yaml`、`ensure_folder`、可视化、Namespace 处理、`_set_by_path` 配置路径写入 |
| `utils.training_utils` | 优化器创建、loss、学习率、metrics、滑动平均 |
| `utils.calculator_utils` | `train_maxent`、辅助 calculator 构建 info_dict |
| `utils.ablation_utils` | 消融实验特有：C 组随机连通 mask 生成 |
| `utils.tornado_utils` | tornado/sensitivity 实验矩阵构造、参数目录 |
| `utils.exp_sweep_utils` | 参数扫描类实验的结果存储访问：`ExperimentResultStore`、目录扫描、指标聚合、统计表；stability 模式专用工具 `write_stability_runs_csv` / `merge_stability_runs` / `build_stability_per_sample_csv` / `build_stability_summary_csv`（产出 runs/per_sample/summary 三层 CSV）|
| `utils.analysis_plot_utils` | results_analysis 的业务级出图：单实验统计、样本图、SA/GA 对比、模型指标图 |
| `utils.analysis_utils` | results_analysis 的指标读取、JSON 缓存、统计聚合和样本筛选 |
| `utils.visualize_utils` | grouped bar / box plot 等底层论文图原语 |

## 已知问题 / 注意事项

- **`run_runner.py` 和 `experiment.py` 的 `process_batch/get_data_dict` 重复**：两边实现几乎相同，`ablation.py` / `tornado.py` 已通过 `BaseSweepPipeline` 复用 `experiment.process_batch`，后续可抽到 `utils/runner_utils.py` 或单独的数据适配器中
- **MaxEnt 每次都会重训**：`run_runner.py` / `experiment.py` / `BaseSweepPipeline` 中 `train_maxent(...)` 前的 `if not os.path.exists(maxent_path)` 守卫被注释，当前每次运行都会覆盖训练 MaxEnt 模型
- **`experiment.py` 的策略列表硬编码**：改 GA 变体需要直接改源码；大量候选实验处于注释状态，默认只跑少数几个
- **`results_analysis.py` 不使用 YAML**：结果分析入口通过函数参数选择实验目录和图表类型，仍需在 `main.py` 或交互环境中指定路径
- **Windows 字体路径硬编码**：`analysis_plot_utils.py` 中使用 `C:/Windows/Fonts/msyh.ttc`，非 Windows 环境出图可能失败
- **结果分析保留兼容包装**：`complete_pipeline`、`picker`、`all_samples`、`show_SA_GA` 仍可用，但推荐直接调用 4 个 pipeline 入口
- **`pipelines/__init__.py` 静默吞 ImportError**：虽然能避免缺依赖时整体导入失败，但也会掩盖真实导入错误，调试时需直接运行对应文件查看 traceback
- **`main.py` 当前默认运行**：底部 `if __name__ == "__main__"` 中具体跑哪个 pipeline 由注释控制；绘图入口的 `ablation_root` 默认会自动查找 `results/ablation/` 下最新时间戳，无需手动改源码
- **`BaseSweepPipeline` 的 `opt_overrides` 不存在属性时跳过**：tornado/sensitivity 在 GA 模式下若覆写 SA 专属参数（如 `cooling_rate`/`T_start`），会打印 warning 但不会崩溃
- **`ablation_plots.py` 顶部直接 `import matplotlib.pyplot` 不设后端**：在无图形界面的环境（CI、远程 SSH、WSL 无 X server）跑出图会因 Qt platform 报错。临时用 `MPLBACKEND=Agg python -c "..."` 回避，持久化可在 import 前设 `matplotlib.use('Agg')`（注意要早于 `import pyplot`）
- **stability 模式的 SA+GA 合并需要两个 timestamp**：单次 pipeline 只跑一种算法，`stability_per_sample.csv` / `stability_summary.csv` / `stability_cv_line.png` 需要调用 `run_ablation_plots_pipeline(sibling_ablation_roots=[<GA_ts>])` 才能生成

