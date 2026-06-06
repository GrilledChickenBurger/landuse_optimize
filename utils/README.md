# Utils Module

## 这个模块做什么

utils 是项目的**公共工具层**和**胶水层**。项目核心逻辑分两阶段：

1. **Stage-1：ANN 变化检测** —— 用驱动因子 + 当前 LU 训练语义分割模型，预测 `pred_diff`
2. **Stage-2：SA/GA 随机优化** —— 以 `pred_diff` 为先验，搜索生态系统服务更优、景观更合理的 LU 方案

utils 不直接定义 ANN 模型、目标函数或优化器，但它支撑这些模块运行：

- 为 Stage-1 提供 TIFF 读取、LU one-hot、归一化、loss、metrics、训练可视化
- 为 Stage-2 提供 ANN 推理、约束区构建、MaxEnt 训练/加载、calculator 的 info_dict、方案保存与性能评估
- 为消融实验和论文图提供 score_cfg 管理、结果聚合、统计表、收敛曲线、LU/ES 可视化

可以把 utils 理解为：**数据怎么进来、配置怎么读、分数需要哪些预计算信息、结果怎么保存和分析**，这些跨模块共用逻辑都放在这里。

## 在项目中的位置

```
Stage-1: 训练变化检测模型

configs/trainConfig.yaml
        │
        ▼
utils.config_utils.get_yaml
        │
        ▼
datasets/ ── utils.dataset_utils ──→ [factors, srcLU_oh, diff]
        │                                  │
        ▼                                  ▼
models/ANN  ◄──────── utils.training_utils(loss/metrics/optimizer)
        │
        ▼
checkpoints/<model>/<timestamp>/best_model.pth


Stage-2: SA/GA 优化搜索

configs/optimizerConfig.yaml + configs/score_cfg.yaml
        │
        ▼
pipelines/run_runner.py / experiment.py / ablation.py
        │
        ├── utils.calculator_utils.train_maxent()  → culture MaxEnt 模型
        │
        └── runner/SA.py 或 runner/GA.py
              │
              ├── utils.runner_utils.load_ANN_model()
              ├── utils.runner_utils.get_pred_diff()
              ├── utils.runner_utils.get_grid_restrict_area()
              ├── utils.calculator_utils.get_info_dict()
              └── utils.performance.write_performance_score()
                    │
                    ▼
              results/<SA|GA|ablation>/<timestamp>/...


Post-analysis: 结果汇总和论文图

results/...
   │
   ├── pipelines.results_analysis
   │       └── utils.analysis_plot_utils
   │              ├── utils.analysis_utils
   │              └── utils.visualize_utils
   └── pipelines.ablation_plots / pipelines.tornado_plots  ← 顶层编排
          │   ├── utils.exp_sweep_utils (ExperimentResultStore)  ← 数据准备
          │   ├── utils.ablation_utils / utils.tornado_utils      ← 实验特有工具
          │   └── utils.visualize_utils                          ← 底层绘图
          ▼
      figures/ plots/ tables
```

utils 被几乎所有上层模块使用：

| 调用方 | 使用 utils 的目的 |
|---|---|
| `datasets/` | 读取 TIFF、归一化因子、LU one-hot、去除小斑块 |
| `pipelines/` | 加载配置、训练工具、MaxEnt 训练、消融覆写、结果出图 |
| `runner/` | 加载 ANN、生成 pred_diff、构建约束、保存方案、评估性能 |
| `calculator/` | 读取 info_dict、距离图、碳库、MaxEnt、food 参数等辅助信息 |
| `scripts/` | 临时预测可视化、代表性样本筛选 |
| `libs/` | utils 从 `libs.const` 读取 LU/ES 标签、颜色、像素面积、归一化参数和阈值 |

## 文件一览

| 文件 | 主要阶段 | 作用 |
|---|---|---|
| `__init__.py` | all | 轻量导出常用工具；对 `calculator_utils` 的重依赖做 ImportError 容错 |
| `tools.py` | all | 通用工具和兼容 re-export：可视化、插值器、数组转换、`_set_by_path` 配置路径写入 |
| `config_utils.py` | all / ablation | YAML → Namespace；默认 `score_cfg`；加载评分参数配置 |
| `dataset_utils.py` | Stage-1 / Stage-2 | TIFF 读取、样本排序、去小斑块、因子归一化、LU one-hot |
| `training_utils.py` | Stage-1 | ANN 训练相关：optimizer、learning rate、loss、metrics |
| `runner_utils.py` | Stage-2 | ANN 加载与推理、pred_diff、约束区、尺度分档、方案保存、分数曲线 |
| `calculator_utils.py` | Stage-2 | calculator 预计算：碳库、MaxEnt、距离图、food 参数、info_dict |
| `performance.py` | Stage-2 / post | best_plan 的最终评估：Change area、ES、Fragstats、FOM、Kappa |
| `analysis_utils.py` | post | 读取 results、聚合/缓存 LU/ES 指标、筛样本、组织分析数据 |
| `analysis_plot_utils.py` | post | 结果分析业务出图：单实验 LU/ES 图、样本可视化、SA-GA 对比、模型指标图 |
| `visualize_utils.py` | post | 论文图基础组件：堆叠图、柱状图、箱线图、图例等 |
| `ablation_utils.py` | ablation | 消融实验特有：C 组随机连通 mask 生成（`make_random_connected_mask`） |
| `tornado_utils.py` | tornado / sensitivity | tornado / sensitivity 实验矩阵构造、参数目录 |
| `exp_sweep_utils.py` | ablation / tornado / post | 参数扫描类实验的结果存储访问：`ExperimentResultStore`、目录扫描、指标聚合、统计表 |
| `io_utils.py` | all / post | I/O 工具：文件夹创建、候选文件查找、GeoTIFF 保存、图片序列转视频 |
| `train_maxent.py` | Stage-2 prep | MaxEnt 文化服务模型训练与预测工具 |

## Stage-1 工具：ANN 变化检测训练

Stage-1 的数据流是：

```
磁盘 TIFF → dataset_utils → TrainingDataset → train.py → training_utils → checkpoints
```

### dataset_utils：把栅格数据变成模型输入

| 函数 | 作用 | 典型调用方 |
|---|---|---|
| `load_tiff(file_path)` | 读取单个 `.TIF` 栅格，优先 GDAL，失败后 rasterio | `get_year_data` |
| `get_year_data(root, key_list, shape, ...)` | 读取某一年/某 split 下多个因子的全部样本，组织成 `(N,C,H,W)` | `TrainingDataset`、`OptimizeDataset` |
| `apply_norm(x, name, params, log=False, mask=None)` | 按统计参数做 Z-score；可对 GDP/RK/DL 等先 `log1p` | 训练数据加载、ANN 推理 |
| `one_hot_lu(lu_tensor)` | 将二级 LU 编码聚合为 6 个一级类 one-hot 通道 | `TrainingDataset`、`runner_utils.get_pred_diff` |
| `remove_lines(area, threshold=None)` | 去除小连通斑块，默认阈值来自 `PIXEL_NUM_THRESHOLD_REMOVE` | diff 后处理、性能评估 |
| `remove_lines_forbatch(area, threshold=None)` | 批量版去小斑块 | `train.py::process_batch` |

模型输入中的 LU 并不是原始二级编码，而是 6 类简化 one-hot：耕地 / 林地 / 草地 / 水域 / 建设用地 / 未利用地。因此 Stage-1 的输入通道数是：

```
in_ch = len(factor_list) + 6
```

### training_utils：训练损失、优化器和指标

| 工具 | 作用 |
|---|---|
| `get_optimizer(...)` | 根据配置创建 Adam / AdamW / RMSProp / SGD |
| `adjust_learning_rate(optimizer, shrink_factor)` | 将所有参数组学习率乘以 shrink factor |
| `ExpoAverageMeter` | 指数滑动平均，用于平滑 loss 日志 |
| `cal_loss(output, target, focal_weight=0.5)` | Focal loss + Dice loss，用于变化检测二分类 |
| `focal_loss_with_logits(...)` | logits 版本 focal loss，缓解正负样本不平衡 |
| `dice_loss(prediction, target)` | Dice loss，关注变化区域重叠 |
| `MetricsMeter` | 累积 OA / Precision / Recall / F1 / IoU / TP/FP/TN/FN |

`MetricsMeter` 期望输入是概率图（已经 sigmoid），不是 logits。`train.py::valid_pipeline` 中会先对模型输出做 sigmoid 再更新指标。

### tools：训练过程可视化

`visualize_tensor_list(...)` 是训练和验证最常用的图像工具。它把多个 `(H,W)` 张量拼成一张图，可使用 `COLOR_MAP` 显示 LU 类型，也可以显示模型输出概率图、真实 diff、预测 diff 等。

训练阶段常见输出：

```
checkpoints/<model>/<timestamp>/best_model.pth
checkpoints/<model>/<timestamp>/latest_checkpoint.pth
checkpoints/<model>/<timestamp>/epoch_<n>.pth
logs/<model>/<timestamp>/events.*
checkpoints/<model>/<timestamp>/<split>_metrics_<pack>.json
checkpoints/<model>/<timestamp>/<split>_pic_<pack>/*.png
```

## Stage-2 工具：SA/GA 优化搜索

Stage-2 的数据流是：

```
OptimizeDataset → data_dict
      │
      ├── runner_utils.get_pred_diff()      → pred_diff
      ├── runner_utils.get_grid_restrict_area() / get_cls_restrict_area()
      ├── calculator_utils.get_info_dict()  → calculator 所需辅助信息
      └── runner(SA/GA).run()
              │
              ├── generator 生成方案
              ├── calculator 计算目标函数
              └── runner_utils.save_plan_and_get_performance()
```

### runner_utils：连接 ANN 先验和随机优化器

| 函数 | 作用 |
|---|---|
| `load_ANN_model(model_type, model_path, in_ch, out_ch, device, model_config)` | 创建 ANN 并加载 checkpoint |
| `get_pred_diff(model, ori_plan, factors, device, year, condition_list, normalize, sigmoid_threshold)` | 对驱动因子 + LU one-hot 做模型推理，输出二值 `pred_diff` |
| `get_grid_restrict_area(pred_diff, grid_num=4)` | 将 pred_diff 扩展到所在网格，作为 guided 搜索的空间约束 |
| `get_cls_restrict_area(ori_plan)` | 标记 51/53 城镇用地像素，限制只能改为建设用地类 |
| `get_lu_labels(plan)` | 获取当前方案中出现的 LU 类型 |
| `get_scale_minmax_dict(num)` | 根据预测变化像素数生成 small/mid/large 三档矩形扰动尺度 |
| `get_initial_score_list()` | 初始化目标函数五项分数 `[total, change, es, es_dict, landscape]` |
| `draw_score_list(...)` / `draw_score_list_general(...)` | 保存优化过程分数曲线 |
| `save_all_plans(...)` | 保存 best/ori/tgt 三个方案为 TIF |
| `save_plan_and_get_performance(...)` | 保存方案并调用 `performance.write_performance_score` 计算性能 |

`get_pred_diff` 是 Stage-1 与 Stage-2 的关键接口：它加载训练好的 ANN，在优化阶段把当前样本转成变化概率图，再阈值化成 `pred_diff`。runner 后续用它构建 `hard_restrict` 和变化得分先验。

### calculator_utils：目标函数的预计算信息

calculator 每次打分需要很多静态信息。如果每个候选方案都重复计算会很慢，因此 runner 初始化时通过 `get_info_dict(...)` 一次性准备好：

| info_dict 内容 | 用途 |
|---|---|
| `carbon_pool` | carbon ES 查表 |
| `maxent_model` / `maxent_list` | culture ES 适宜性预测 |
| `pred_diff_distance_map` | change_score 中的距离惩罚 |
| `ori_lu_distancemap_dict` | landscape_score 中同类原始斑块距离 |
| `ori_num_feature_list` | landscape_score 中原始连通斑块数基准 |
| `food_supply_param` | food ES 的五因子复合修正 |
| `max_es_score` | ES 归一化理论上限 |
| `min_change_score` | change_score 归一化参考（当前目标函数中部分用途已弱化） |

相关工具：

| 函数 | 作用 |
|---|---|
| `load_carbon_pool(file_path)` | 读取 `libs/carbon_pool.csv` |
| `train_maxent(root, year, save_path)` | 训练文化服务 MaxEnt 模型 |
| `load_maxent_model(model_path)` | 加载 `.ela` MaxEnt 模型 |
| `get_pred_diff_distance_map(pred_diff)` | 对预测变化区生成距离图 |
| `get_lu_distancemap_dict(plan)` | 对每种 LU 生成 EDT 距离图 |
| `get_numfeature_dict(plan, lu_labels=None)` | 统计每种 LU 的连通斑块数 |
| `calculate_food_supply_param(info_dict, weight_dict=...)` | 计算粮食生产复合修正系数 |
| `get_perimeter(area)` | 计算连通区域周长 |

### performance：优化结果评估

runner 得到 `best_plan` 后，`performance.py` 负责写最终性能 JSON：

| 评估项 | 说明 |
|---|---|
| Change area | 优化变化区域与真实变化/预测变化的 OA、Accuracy、IoU 等 |
| ES | food / carbon / water / culture 的变化与提升 |
| Landscape | 通过 Fragstats 输出 land/class 指标 |
| FINAL | FOM、Kappa、best vs ori / target 的 LU 面积统计 |

主要入口：

```python
write_performance_score(...)
```

常见输出：

```
results/<SA|GA>/<timestamp>/<scenario>/<idx>/performance.json
results/<SA|GA>/<timestamp>/<scenario>/<idx>/plans_<idx>/best_plan.TIF
results/<SA|GA>/<timestamp>/<scenario>/<idx>/plans_<idx>/ori_plan.TIF
results/<SA|GA>/<timestamp>/<scenario>/<idx>/plans_<idx>/tgt_plan.TIF
```

## 消融与敏感性分析工具

消融实验的核心是：不改 calculator/runner 源码，通过 `score_cfg.yaml` 和 runner flags 改变目标函数、约束和早停方式。

### config_utils：score_cfg 管理

| 函数 / 常量 | 作用 |
|---|---|
| `get_yaml(path)` | 加载普通 YAML 并递归转 Namespace |
| `get_namespace(config_dict)` | dict → argparse.Namespace |
| `_DEFAULT_SCORE_CFG_DICT` | 评分参数默认值，保证未传 `score_cfg` 时仍能使用历史默认配置 |
| `get_default_score_cfg()` | 返回默认 `score_cfg` Namespace |
| `load_score_cfg(path)` | 加载 `configs/score_cfg.yaml` |

`score_cfg` 的主要分组：

| 分组 | 作用 |
|---|---|
| `tgt_function` | λ_ES / λ_LS 的上下界与像素数分段阈值 |
| `change_score` | pred_diff 命中奖励系数 |
| `runner_utils` | pred_diff sigmoid 阈值、grid restrict 网格数 |
| `landscape` | 景观纯度项权重 |
| `const` | 可运行时覆写的常量，如变化像素阈值、水源径流系数 |
| `ablation` | guided / disable_change_score / disable_early_stop 等开关 |

注意：`_DEFAULT_SCORE_CFG_DICT` 必须与 `configs/score_cfg.yaml`、`libs.const.py`、calculator 和 runner 中的默认值保持一致，否则消融覆写语义会漂移。

### ablation_utils / tornado_utils / exp_sweep_utils：参数扫描类实验

参数扫描类实验（消融、tornado、sensitivity）的工具分三层：

| 模块 | 层级 | 职责 |
|---|---|---|
| `tornado_utils.py` | 实验矩阵 | `build_experiment_matrix(mode='tornado' / 'sensitivity', ...)` 构造 tornado / sensitivity 实验列表，定义参数目录 `_TORNADO_PARAMS` |
| `ablation_utils.py` | 实验特有 | `make_random_connected_mask(pred_diff, hard_restrict_base, seed)` 为 C 组消融生成与 pred_diff 连通块数和面积匹配的随机 mask |
| `exp_sweep_utils.py` | 结果存储访问 | `ExperimentResultStore` 封装目录扫描、指标读取、聚合；以及 `build_stability_table` / `build_ablation_compare_table` 表格生成 |

### exp_sweep_utils：ExperimentResultStore

`ExperimentResultStore(ablation_root)` 是参数扫描类实验的**结果存储访问对象**，封装所有目录扫描和数据读取逻辑。绘图 pipeline 通过它获取已结构化的数据，不再直接 `os.listdir` 或 `np.load`。

| 方法 | 返回 | 用途 |
|---|---|---|
| `list_experiments(prefix='')` | `List[str]` | 列出 ablation_root 下指定前缀的实验目录 |
| `list_samples(exp_name)` | `List[str]` | 列出某实验下的所有样本 index |
| `list_run_dirs(exp_name, index)` | `List[str]` | 列出某样本的所有 run 目录（兼容新旧目录结构） |
| `get_convergence_series(exp_name, index, smooth_window=1)` | `np.ndarray (n_runs, min_len)` 或 `None` | 多 run 收敛曲线（已对齐、可选平滑） |
| `get_stability_series(exp_name, smooth_window=1)` | `Dict[str, np.ndarray]` | 整个实验所有样本的收敛序列 |
| `get_ablation_comparison_data(groups=None)` | `Dict[group, Dict[index, agg]]` | A/B/C 三组对比数据 |
| `get_final_best_distribution(groups=None)` | `Dict[sample, Dict[label, [values]]]` | 箱线图数据 |
| `get_baseline_metric(baseline_name, metric)` | `float` 或 `None` | tornado baseline 聚合指标 |
| `get_knob_deviations(baseline_name, prefix='B_', metric)` | `Dict[knob, Dict[val_str, deviation]]` | tornado 参数扰动数据 |
| `get_sensitivity_data(prefix='S_', metric)` | `Dict[param, Dict[val, [sample_means]]]` | sensitivity 一维线搜索数据 |
| `recompute_and_aggregate(exp_name, index)` | `Dict[metric, agg]` | 重算 iter_to_95pct 并返回聚合指标 |

模块内部还提供供 store 复用的工具函数：

| 函数 | 作用 |
|---|---|
| `_list_samples / _list_run_dirs_for_sample` | 目录扫描（兼容新老结构） |
| `_read_perf_metric / _read_conv_metric / _read_nested` | 单文件指标读取 |
| `_aggregate_exp / _make_extractor` | 跨样本指标聚合与提取器工厂 |
| `aggregate_runs(exp_dir, index)` | 单样本多 run 聚合：mean / std / 95% CI（t 临界值按样本量动态计算） |
| `recompute_iter_to_95pct(exp_dir, index)` | 用多次 final_best 均值作为参考最优，重算每次 run 的 `iter_to_95pct_repeat` |
| `build_stability_table(ablation_root, exp_name, save_path=None)` | 单实验稳定性 CSV |
| `build_ablation_compare_table(ablation_root, save_path=None)` | A/B/C 对比 CSV |

典型输出：

```
results/ablation/<timestamp>/plots/stability_convergence_<idx>.png
results/ablation/<timestamp>/plots/ablation_convergence_<idx>.png
results/ablation/<timestamp>/plots/boxplot_final_f.png
results/ablation/<timestamp>/plots/stability_table_<exp>.csv
results/ablation/<timestamp>/plots/ablation_compare_table.csv
results/ablation/<timestamp>/plots/tornado_<metric>.png
results/ablation/<timestamp>/plots/sensitivity_<metric>.png
```

### representative_samples：代表性样本筛选

`scan_n_pred(...)` 用训练好的 ANN 扫描 optimize 数据集，按预测变化像素数 `n_pred` 分层抽样，输出消融实验用的代表性样本。

流程：

```
optimizerConfig + score_cfg
      ↓
OptimizeDataset
      ↓
load_ANN_model + get_pred_diff
      ↓
hard_restrict ∧ pred_diff → n_pred
      ↓
按 n_pred 分位数分层抽样
      ↓
_data/representative_samples.json
```

输出 JSON 包含：`selected`、`idxs_for_pipeline`、`quantile_ranges`、`all_records`。

## Post-analysis 与论文图工具

Stage-2 完成后，项目会产生大量 `results/<SA|GA>/<timestamp>/...` 目录。`pipelines/results_analysis.py` 负责顶层调度，`analysis_plot_utils.py` 负责业务级出图，`analysis_utils.py` 负责读取、缓存和聚合结果，`visualize_utils.py` 提供底层图表组件。

### analysis_utils：读取结果和聚合指标

| 函数 | 作用 |
|---|---|
| `get_performance_metrics(folder_path)` | 扫描 scenario/sample 目录，读取 `performance.json` 或 `performance_score.json` |
| `load_or_compute_metric(data_folder, cache_name, calc_fn, ...)` | JSON 存在则读取，不存在则从 performance 现算并写回缓存 |
| `cal_total_area(metrics, aggregate_by_first_digit=False)` | 汇总各 LU 类型面积 |
| `cal_sample_area_change_ratio(metrics, comparison_type='origin')` | 每个样本的 LU 变化比例 |
| `cal_total_es(metrics)` | 汇总 ES 总值 |
| `cal_sample_es_change_ratio(metrics, comparison_type='origin')` | 每个样本的 ES 变化比例 |
| `cal_scenario_diff_ratio(sa_total_area, ga_total_area, ...)` | 计算 SA/GA 或 scenario 间差异 |
| `list_sample_indices(data_folder)` | 从结果目录扫描样本索引 |
| `build_sa_ga_total_area_data(sa_folder, ga_folder)` | 分别准备 SA/GA 的 LU 总面积数据 |
| `build_sa_ga_combined_area_data(sa_folder, ga_folder)` | 生成 SA/GA 合并面积数据 |
| `get_best_samples(...)` | 根据多场景 ES 改善筛选表现好的样本 |

### analysis_plot_utils：业务级结果出图

| 函数 | 作用 |
|---|---|
| `plot_single_experiment_lu_bars(...)` | 单个 SA/GA 实验的 LU 总面积堆叠图和分类柱状图 |
| `plot_single_experiment_lu_distribution(...)` | 单个实验的 LU 变化比例直方图/箱线图 |
| `plot_single_experiment_es_bars(...)` | 单个实验的 ES 总价值柱状图 |
| `plot_single_experiment_es_distribution(...)` | 单个实验的 ES 变化比例直方图/箱线图 |
| `visualize_picked_samples(...)` / `visualize_all_samples(...)` | 筛选样本或遍历样本并生成空间方案对比图 |
| `plot_sa_ga_diff_bars(...)` | SA vs GA 的 LU 差异分组柱状图 |
| `plot_sa_ga_boxplots(...)` | SA vs GA 的 LU/ES 变化比例箱线图 |
| `plot_model_comparison_bar(...)` | 变化检测模型指标对比图 |

### visualize_utils：基础图表组件

| 函数 | 图形类型 |
|---|---|
| `color_bar_figure(...)` | 单组柱状图 |
| `stacked_bar_figure(...)` | 堆叠柱状图 |
| `hist_figure(...)` | 直方图 |
| `box_plot_figure(...)` | 箱线图 |
| `grouped_bar_figure(...)` | 分组柱状图 |
| `grouped_box_plot_figure(...)` | 分组箱线图 |
| `legend(...)` | 单独图例 |

这些函数通常不直接从 pipeline 调用，而是由 `analysis_plot_utils.py` 包装成更具体的论文图。

## I/O 工具

`io_utils.py`（合并自原 `file_utils.py` 和 `video_utils.py`）提供文件系统和数据导出功能：

| 函数 | 作用 |
|---|---|
| `ensure_folder(folder)` | 文件夹不存在则创建 |
| `find_valid_file(file_paths, file_description='文件')` | 从候选路径中返回第一个存在的文件 |
| `save_tif(plan, output_folder, output_filename)` | 将 `[1,1,H,W]` 方案保存为 GeoTIFF |
| `get_video_from_imgs(src_folder, video_folder)` | 将 history 图片序列合成 mp4 |

`save_tif` 写入的是简化地理信息（dummy CRS/transform）。优化结果的 `.tfw` 坐标文件复制由 `runner_utils.save_all_plans` 另外处理。

`get_video_from_imgs(src_folder, video_folder)` 会将某个目录下按编号排序的图片合成为 `output_video.mp4`，适合把 SA/GA 的 `history/iter_*.png` 合成优化过程视频。

## MaxEnt 工具

`train_maxent.py` 是文化服务适宜性模型的训练/预测工具，主要被 `calculator_utils.train_maxent` 包装使用。

| 函数 | 作用 |
|---|---|
| `train_pipeline(root, save_path)` | 在 root 下查找 CSV 和 raster，训练并保存 `.ela` 模型 |
| `train_with_alldata(csv, rasters, labels, save_path, ...)` | 用全部样本训练 MaxEnt |
| `predict_and_save_raster(model_path, rasters, result_folder, result_name)` | 预测并保存 suitability raster |
| `predict_and_return_arrays(model_path, raster_arrays, labels)` | 对内存数组预测 suitability |

在主优化流程中，MaxEnt 模型用于 `calculator/cal_es_score.py` 的 culture ES 计算。`optimizerConfig.yaml` 中的 `maxent_list` 指定输入变量，通常包含 `LU`，表示当前候选方案的 LU 类型也会影响文化服务适宜性。

## 关键调用链

### 训练变化检测模型

```
pipelines/train.py
  ├── config_utils.get_yaml(trainConfig.yaml)
  ├── datasets.get_dataset("training")
  │     ├── dataset_utils.get_year_data()
  │     ├── dataset_utils.apply_norm()
  │     └── dataset_utils.one_hot_lu()
  ├── models.create_model()
  ├── training_utils.get_optimizer()
  ├── training_utils.cal_loss()
  ├── training_utils.MetricsMeter()
  └── tools.visualize_tensor_list()
```

### 运行 SA/GA 优化

```
pipelines/run_runner.py / experiment.py / ablation.py
  ├── config_utils.get_yaml(optimizerConfig.yaml)
  ├── config_utils.load_score_cfg(score_cfg.yaml)        # ablation only
  ├── calculator_utils.train_maxent()
  ├── datasets.get_dataset("optimize")
  └── runner.create_runner()
        ├── runner_utils.load_ANN_model()
        ├── runner_utils.get_pred_diff()
        ├── runner_utils.get_grid_restrict_area()
        ├── runner_utils.get_cls_restrict_area()
        ├── runner_utils.get_scale_minmax_dict()
        ├── calculator_utils.get_info_dict()
        ├── generator / calculator                       # runner 内部调用
        └── runner_utils.save_plan_and_get_performance()
              ├── file_utils.save_tif()
              └── performance.write_performance_score()
```

### 做结果分析和论文图

```
pipelines/results_analysis.py
  └── analysis_plot_utils
        ├── analysis_utils.get_performance_metrics()
        ├── analysis_utils.load_or_compute_metric()
        ├── analysis_utils.cal_total_area() / cal_total_es()
        ├── analysis_utils.get_best_samples()
        └── visualize_utils.grouped_bar_figure() / grouped_box_plot_figure()

main.py::run_ablation_plots_pipeline()
  └── pipelines.ablation_plots.run_all_ablation_plots()
        └── exp_sweep_utils.ExperimentResultStore(ablation_root)
              ├── ExperimentResultStore.get_convergence_series()
              ├── ExperimentResultStore.get_ablation_comparison_data()
              ├── exp_sweep_utils.build_stability_table()
              └── exp_sweep_utils.build_ablation_compare_table()

main.py::run_tornado_plots_pipeline()
  └── pipelines.tornado_plots.run_tornado_plots() / plot_sensitivity_lines()
        └── exp_sweep_utils.ExperimentResultStore(ablation_root)
              ├── ExperimentResultStore.get_knob_deviations()
              └── ExperimentResultStore.get_sensitivity_data()
```

## 配置与常量

### 普通 YAML 配置

项目所有 YAML 都通过 `config_utils.get_yaml()` 加载为 `argparse.Namespace`：

```python
config.data.factor_list
config.training.model_name
config.runner.scenario
config.GA.strategy.parent_selection
```

这让 pipeline 和 runner 可以用点号访问配置，但也意味着缺少字段时不会在加载阶段报错，而是在实际访问时触发 AttributeError。

### score_cfg

`score_cfg` 是目标函数和实验魔数的集中配置。普通优化流程若不显式传入，会通过 `get_default_score_cfg()` 使用默认值；消融实验会从 `configs/score_cfg.yaml` 加载并覆写。

重要字段：

| 字段 | 消费者 | 作用 |
|---|---|---|
| `tgt_function.lambda_ES_high/low` | `calculator/tgt_function_cal.py` | ES 权重随 pred_diff 像素数插值 |
| `tgt_function.lambda_LS_high/low` | `calculator/tgt_function_cal.py` | 景观权重随 pred_diff 像素数插值 |
| `tgt_function.pixel_high/low` | `calculator/tgt_function_cal.py` | λ 插值分段阈值 |
| `change_score.inside_reward_coef` | `calculator/cal_change_score.py` | 变化区域落入 pred_diff 的奖励倍率 |
| `runner_utils.sigmoid_threshold` | `runner_utils.get_pred_diff()` | ANN 输出二值化阈值 |
| `runner_utils.grid_num` | `runner_utils.get_grid_restrict_area()` | guided 约束的网格数量 |
| `landscape.pure_base/pure_change` | `calculator/cal_landscape_score.py` | 景观纯度项权重 |
| `const.pixel_num_threshold_ischange` | `libs.const.apply_const_overrides` | 小变化样本跳过阈值 |
| `const.water_supply_param` | `libs.const.apply_const_overrides` | water ES 径流系数表 |
| `ablation.*` | `pipelines/ablation.py` | 消融实验开关 |

### libs.const

utils 大量读取 `libs/const.py` 中的项目级基础知识：

| 常量 | 用途 |
|---|---|
| `ALL_LU_LABELS` / `ALL_LU_LABELS_DICT` | LU 类型遍历、面积统计、图例标签 |
| `ALL_LU_LABELS_SIMPLIFIED` | 6 类 one-hot 和简化可视化 |
| `ALL_ES_TYPES` / `ALL_ES_TYPES_DICT` | ES 指标统计与可视化 |
| `ALL_SCENARIOS` / `ALL_SCENARIOS_DICT` | 优化场景顺序和显示名 |
| `COLOR_MAP` / `SIMPLIFIED_COLOR_MAP` / `ES_COLOR_MAP` | LU/ES 图表颜色 |
| `SCENARIO_COLOR_MAP` / `SA_GA_COLOR_MAP` | 结果分析和 SA/GA 对比图颜色 |
| `PIXEL_AREA_IN_HECTARES` | 像素面积换算 |
| `PIXEL_NUM_THRESHOLD_ISCHANGE` | 小变化样本跳过阈值 |
| `PIXEL_NUM_THRESHOLD_REMOVE` | 去除小斑块阈值 |
| `FACTOR_PARAMS_DICT` | 因子 Z-score 归一化参数 |
| `ALL_LOG_FACTORS` | 需要 log1p 的因子 |
| `FOOD_SUPPLY_PARAM` | food ES 的因子权重 |
| `WATER_SUPPLY_PARAM` | water ES 的 LU 径流系数 |

## 输出文件约定

| 工具 / 阶段 | 输出 |
|---|---|
| `training_utils` + `train.py` | `checkpoints/<model>/<timestamp>/*.pth`、`logs/<model>/<timestamp>/`、验证图和 metrics JSON |
| `runner_utils.save_all_plans` | `plans_<idx>/best_plan.TIF`、`ori_plan.TIF`、`tgt_plan.TIF`、`.tfw` |
| `performance.write_performance_score` | `performance.json`（部分旧分析代码也兼容 `performance_score.json`） |
| runner 收敛记录 | `total_score_list.npy`、`convergence.json`、`charts/*.png`、`history/*.png` |
| `representative_samples.scan_n_pred` | `_data/representative_samples.json` |
| `exp_sweep_utils` + `pipelines.ablation_plots / tornado_plots` | `results/ablation/<ts>/plots/*.png`、`*.csv`、summary JSON |
| `analysis_plot_utils` / `results_analysis.py` | `figures/<SA|GA>/<ts>/...`、`figures/comparison_results/...` |
| `video_utils` | `output_video.mp4` |

## 与其他模块的协作

### 与 datasets

`datasets/training_dataloader.py` 和 `datasets/optimize_dataloader.py` 主要依赖 `dataset_utils`：

- `get_year_data` 从磁盘组织多因子栅格
- `apply_norm` 归一化驱动因子
- `one_hot_lu` 构造 ANN 输入通道
- `remove_lines` / `remove_lines_forbatch` 清洗变化标签

### 与 pipelines

| pipeline | 主要使用的 utils |
|---|---|
| `statistic.py` | `get_yaml`、`ALL_LOG_FACTORS`、Welford 统计逻辑 |
| `train.py` | `get_yaml`、`training_utils`、`visualize_tensor_list`、`remove_lines_forbatch` |
| `run_runner.py` | `get_yaml`、`ensure_folder`、`train_maxent` |
| `experiment.py` | `get_yaml`、`train_maxent`、性能聚合辅助 |
| `ablation.py` / `tornado.py` | `load_score_cfg`、`_set_by_path`、`ablation_utils.make_random_connected_mask`、`tornado_utils.build_experiment_matrix` |
| `ablation_plots.py` / `tornado_plots.py` | `exp_sweep_utils.ExperimentResultStore`、`visualize_utils` |
| `results_analysis.py` | `analysis_plot_utils` |

### 与 runner

SA/GA 初始化和收尾大量依赖 `runner_utils` 与 `calculator_utils`：

- 初始化 ANN、计算 `pred_diff`
- 构建 `hard_restrict`、`cls_restrict`、`scale_minmax_dict`
- 构建 calculator 所需 `info_dict`
- 保存最优方案并计算性能
- 绘制分数曲线和最终对比图

### 与 calculator

calculator 不负责加载外部资源，它依赖 `calculator_utils.get_info_dict()` 提前准备：

- `pred_diff_distance_map`
- `ori_lu_distancemap_dict`
- `carbon_pool`
- `maxent_model`
- `food_supply_param`
- `max_es_score`

这样 `cal_tgt_function()` 在优化迭代中只做当前方案的评分，不反复加载模型或数据。

## 已知问题 / 注意事项

- **`tools.py` 是兼容门面**：配置、文件和视频工具已经拆到 `config_utils.py`、`io_utils.py`，但旧代码仍从 `utils.tools` 导入；`tools.py` 仅保留通用工具（可视化、插值器、数组转换、`_set_by_path` 配置路径写入）
- **`exp_sweep_utils.py` 包含跨模块的结果存储约定**：实验目录布局（`<exp>/run_*/<index>/{convergence.json, performance.json}`）和指标 key 都集中在此模块；若结果格式变更只需修改一处
- **GDAL 环境路径硬编码**：`dataset_utils.py` 中设置了 `D:/anaconda3/envs/pytorch/...` 的 `PROJ_LIB` 和 `GDAL_DATA`，换机器可能需要调整
- **TIFF 地理信息较简化**：`io_utils.save_tif` 写 dummy CRS/transform，真实 `.tfw` 坐标由 `runner_utils.save_all_plans` 尝试复制
- **MaxEnt 依赖较重**：`calculator_utils.py` / `train_maxent.py` 依赖 elapid、geopandas、rasterio 等，缺依赖时部分功能不可用
- **Fragstats 是外部程序**：`performance.py` 通过环境变量 `frg_cmd` 调用 Fragstats，偏 Windows 环境，路径配置不对会导致 landscape 指标失败
- **性能文件名有历史差异**：当前 runner 写 `performance.json`，部分分析工具同时兼容旧的 `performance_score.json`
- **消融出图会写回结果**：`exp_sweep_utils.recompute_iter_to_95pct` 会向已有 `convergence.json` 写入 `iter_to_95pct_repeat`、`ref_best_repeat`
- **SA/GA 收敛曲线方向不同**：SA 的 `total_score_list` 是 tgt_func（越小越好），GA 的是 fitness = -tgt_func（越大越好），`plot_stability_convergence_all_samples` 自动按 runner_type 做符号翻转，但单独直接读 `total_score_list.npy` 时需自行注意
- **`one_hot_lu` 不覆盖所有 LU 字典键**：部分在 `ALL_LU_LABELS_DICT` 中存在但当前数据未使用的类型，若出现可能被编码为全零通道
- **部分绘图函数会 `plt.show()`**：在无图形界面或批处理环境中可能需要关闭交互后端
- **`representative_samples.py` 直接运行可能不稳定**：文件使用相对导入，更推荐通过项目脚本或 package 方式调用
- **`video_utils.get_video_from_imgs` 对文件名和帧数有假设**：少于 10 帧时 fps 可能为 0，Windows 反斜杠路径也可能影响排序解析
- **`make_random_connected_mask` 使用局部 `np.random.RandomState`**：不再污染全局 `random` 状态，但要求调用方传入显式 `seed` 以保证可复现
- **`aggregate_runs` 的置信区间动态计算**：t 临界值按 `df=len(vals)-1` 用 `scipy.stats.t.ppf` 动态查表，n_repeats 任意值都正确（旧版硬编码 df=4 在 n≠5 时偏差较大）

