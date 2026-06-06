# Runner Module

## 这个模块做什么

runner 是项目 Stage-2 的**运行循环层**。它把三件事串成一个完整的 LU 优化器：
- ANN 模型预测的变化掩膜 `pred_diff`（来自 Stage-1）作为搜索的先验
- generator 生成候选方案
- calculator 给每个候选方案打分

runner 自己不生成方案、不打分，只负责"决定是否接受候选 / 怎么演化种群 / 何时停止"。当前实现了两种元启发式：**模拟退火 SA** 和**遗传算法 GA**，两者共享同一套构造参数、约束体系和输出格式，差别只在搜索策略。

## 模块在项目中的位置

```
Stage-1: 因子 + srcLU ──→ [ANN 模型] ──→ pred_diff
                                              │
                                              ▼
Stage-2: ┌────────── runner ──────────────────────────┐
         │                                            │
         │   每次迭代：                                │
         │     generator ──► 候选方案 cur_plan         │
         │                       │                    │
         │   calculator ◄────────┘                    │
         │     │                                      │
         │     ▼ tgt_func                             │
         │   Metropolis 准则 / 精英选择 ──► best_plan │
         └────────────────────────────────────────────┘
                              │
                              ▼
                       最优 LU 方案
```

runner 的唯一入口是 `pipelines/run_runner.py → create_runner()`。所有共用前置（加载 ANN、构建约束、生成 pred_diff、保存结果与性能）都在 `utils/runner_utils.py` 中。

## 文件一览

| 文件 | 用途 |
|---|---|
| `__init__.py` | 工厂函数 `create_runner(type, config, **kwargs)`，按 `type` 分发到 SA 或 GA |
| `SA.py` | `SimulatedAnnealing` 类——基于温度控制的单点搜索 |
| `GA.py` | `GeneticAlgorithm` 类 + `GeneticAlgorithmStrategy` 策略组装器 |

```python
from runner import create_runner

runner = create_runner('SA', config=sa_config, common_config=..., model_config=...,
                        device=..., data_dict=..., index='10_764_5',
                        save_path='results/SA/default', weight_dict=..., use_landscape=True)
best_plan = runner.run()    # → torch.Tensor [1,1,128,128]
```

## 共通初始化

SA 和 GA 构造签名**完全一致**，所做的前置工作也相同：

### 构造参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `config` | Namespace | `optimizerConfig.SA` 或 `optimizerConfig.GA` 块 |
| `common_config` | Namespace | `runner.common_params`（carbon_pool / maxent_model / frg_model / maxent_list） |
| `model_config` | Namespace | `config.model`（含 `type` 和 `model_path` 及对应架构超参） |
| `device` | torch.device | CPU/CUDA |
| `data_dict` | dict | 来自 `OptimizeDataset`，含 `ori_plan / final_plan / factors / restrict` 等 |
| `index` | str | 样本标识，如 `'10_764_5'` |
| `save_path` | str | 场景级输出目录，runner 会在其下创建 `index/` 子目录 |
| `weight_dict` | dict | ES 权重，如 `{'food':0.25,'carbon':0.25,'water':0.25,'culture':0.25}` |
| `use_landscape` | bool | 是否启用景观破碎化项 |

关键字参数（消融实验用）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `guided` | `True` | 是否将 `pred_grid_restrict` 并入 `hard_restrict` |
| `disable_early_stop` | `False` | 关掉所有早停阈值（实验 A 用） |
| `score_cfg` | `None` → 默认值 | 评分配置，参见 `configs/score_cfg.yaml` |
| `random_restrict_mask` | `None` | 实验 C：以随机掩膜替换 grid restrict，同时替换 `change_target` |

### 构造时按顺序做的事

1. **加载 ANN 模型**：`load_ANN_model(model_type, model_path, in_ch=factor_num+6, out_ch=1, ...)`，`+6` 是 LU one-hot 的 6 个简化通道
2. **生成 pred_diff**：`get_pred_diff(model, ori_plan, factors, year, ...)`——对 `ALL_LOG_FACTORS` 取 log1p、按 `FACTOR_PARAMS_DICT[year]` Z-score 归一化、拼接 LU one-hot、过 sigmoid 阈值二值化
3. **构建三层约束**（详见下一节）
4. **确定 change_target**：默认为 `pred_diff`；若提供 `random_restrict_mask` 则使用 mask（实验 C）
5. **预计算 info_dict**：`cUtils.get_info_dict(...)` 一次性算好 EDT 距离图、carbon_pool、MaxEnt 模型、food_supply_param 等所有 ES/景观评分所需材料
6. **早停跳过守卫**：若 `pred_diff_pixelnum < PIXEL_NUM_THRESHOLD_ISCHANGE`（默认 30），`run()` 会**直接返回 ori_plan 不做优化**——下游 pipeline 通过 `best_plan == ori_plan` 检测此情况

## 三层约束

| 约束 | 谁构建 | 谁使用 | 语义 |
|---|---|---|---|
| `hard_restrict` | runner 用 `functools.reduce(logical_and, ...)` 合并三个 mask | generator 的 `area_wise_change` / `crossover` 做矩形校验 | 1 = 可改，0 = 不可改 |
| `cls_restrict` | `get_cls_restrict_area(ori_plan)` | generator：城镇用地像素只允许改成 5 类（51/52/53） | 1 = 该像素是 51/53 |
| `scale_minmax_dict` | `get_scale_minmax_dict(pred_diff_pixelnum)` | generator 决定矩形面积上下界 | 三档：small / mid / large |

`hard_restrict` 由三部分按位与组成：

1. `redline_restrict = data_dict['restrict']`——耕地红线（RA 层）
2. `lu_zero_restrict = (ori_plan != 0)`——nodata 像素不可改
3. **从下列三者中恰好选一个**：
   - `random_restrict_mask`（提供时；实验组 C）
   - `pred_grid_restrict = get_grid_restrict_area(pred_diff, grid_num=4)`（默认；`guided=True`）
   - 无（实验组 B；`guided=False` 且无 random mask）

`pred_diff_pixelnum = count_nonzero(hard_restrict ∧ pred_diff)` 既是早停守卫阈值，也是 generator 矩形尺寸分档和 calculator λ 调度的依据。

## SA 算法流程

### Metropolis 准则

```
delta = new_total − cur_total
p     = max(exp(−delta / T), 1e-6)
接受  ⇔ delta < 0  或  rand() < p
```

高温时几乎全盘接受（含变差解，鼓励探索）；低温时只接受改进解（精细收敛）。

### 主循环

```
初始化：cur_temp = T_start, cur_plan = ori_plan, iter = 0, noimprove_iter = 0
while cur_temp > T_end:
    1. new_plan, d_diff = generate_new_plan(cur_plan, ori_plan,
                              T_ratio=cur_temp/T_start, noimprove_step=noimprove_iter, ...)
    2. new_score = cal_tgt_function(new_plan, ..., cur_temp=cur_temp,
                                     change_scale_n=pred_diff_pixelnum)
    3. cur_plan, cur_score = evaluate_new_plan(...)        # Metropolis
    4. noimprove_iter += 1 / 重置
    5. 更新全局 best；记录历史；按频率打印 / 可视化
    6. 早停判断
    7. cur_temp *= cooling_rate                            # 几何降温
    8. iter += 1
```

### 关键设计

- **`change_scale_n` 透传给 calculator**：calculator 据此（pred_diff 非零像素数）在 `pixel_low`/`pixel_high` 阈值之间线性插值 λ_ES / λ_LS（参见 `calculator/README.md` 的 λ 自适应调度）。SA 通过 `change_scale_n=self.pred_diff_pixelnum` 显式传入像素数
- **降温节奏**：`cooling_rate` 默认 0.9995，从 `T_start=100` 到 `T_end=0.1` 约需上万步
- **停滞救援由 generator 接管**：当 `noimprove_iter ≥ 30` 或 `T_ratio ≤ 0.1` 时，generator 内部的 `generate_new_plan` 会引入 return 操作（部分还原至 ori_plan）跳出局部最优——runner 不直接干预

### 早停

| 条件 | 默认阈值 |
|---|---|
| `noimprove_iter ≥ max_noimprove_iter` | 1000 |
| `iter − best_score_iter ≥ max_beststuck_iter` | 3000 |

`disable_early_stop=True` 时两个阈值都被设为 `inf`（用于实验 A）。

## GA 算法流程

### 适应度定义

```
fitness = − tgt_func        （SA 最小化 tgt_func；GA 最大化 fitness）
```

GA 调用 `cal_tgt_function` 时**不传 `cur_temp`** ——GA 没有温度概念，calculator 的 λ 调度对 GA 不生效（设计上的非对称，详见"已知问题"）。

### 主循环

```
初始化：population = initial_new_population(ori_pop_size, scale_ratio, ori_plan,
                       scale_minmax_dict, lu_labels, cls_restrict, hard_restrict)
                     （首个个体始终为 ori_plan，作为精英种子）
for gen in range(max_gen):
    1. fitness_list = fitness(population)                  # 每个体调一次 cal_tgt_function
    2. pop_stats = cal_population_stats(...)               # best/worst/avg/std
    3. 当代最佳 = population[argmax]；更新全局 best
    4. 早停判断；按频率打印 / quick_shot 可视化
    5. adaptive_pop_size(gen/max_gen, noimprove_step)      # 动态种群规模
       adaptive_noimprove_threshold(gen/max_gen)           # 早停阈值随代数收紧
    6. parent_pool   = strategy['get_parent_pool'](pop, fit)        # all / elite top-N%
    7. parent_idx    = strategy['select_parents'](pool, fit, ...)   # roulette / tournament
    8. offspring     = crossover_for_offspring(...)        # 交叉 + 变异
    9. 精英保留：offspring[:elite_num] ← strategy['apply_elite_mutation'](top elites)
   10. population    = offspring
```

### 关键设计

- **变异 = SA 扰动**：`mutate_func` 内部以 `T_ratio = max(0.0001, 1 − gen_ratio)` 复用 generator 的 `mutate`（连续两次 `generate_new_plan`），代数越大变异幅度越小
- **`mutation_rate < 1` 时直接克隆**：包装器在概率不命中时返回 `individual.clone()`，不做变异
- **自适应种群与阈值**：`PiecewiseLinear` 插值器使种群规模随 `gen_ratio` 与 `noimprove_step` 调整；早停阈值随代数线性收紧（`max_noimprove_gen` 后 1/2、`max_beststuck_gen` 后 1/4）

### 早停

| 条件 | 默认阈值 |
|---|---|
| `noimprove_step ≥ max_noimprove_gen` | 100（随代数 → 50） |
| `gen − best_score_generation ≥ max_beststuck_gen` | 400（随代数 → 100） |

`disable_early_stop=True` 时所有插值器返回 `inf`。

## GeneticAlgorithmStrategy — GA 策略组装器

设计意图：GA 的"选择 / 精英 / 变异"算子按配置字典动态组装，让 `pipelines/experiment.py` 能批量跑 7 个策略变体。

`build()` 返回 4 个可调用：`get_parent_pool` / `select_parents` / `handle_non_crossover` / `apply_elite_mutation`，由如下策略键组合而成：

| 配置键 | 取值 | 作用 |
|---|---|---|
| `parent_pool` | `'all'` / `'elite'` | 父本来源：全部种群 or 顶部 `elite_ratio_for_pool` |
| `elite_ratio_for_pool` | float（默认 0.2） | `parent_pool='elite'` 时取前 N% |
| `parent_selection` | `'roulette'` / `'tournament'` | 选择方式 |
| `tournament_k` | int（默认 2） | 锦标赛规模 |
| `elite_ratio` | float（默认 0.1） | 下一代中精英保留比例 |
| `non_crossover_handling` | `'copy'` / `'copy_mutate'` | 未交叉个体的处理 |
| `elite_mutation` | bool（默认 False） | 精英是否参与变异 |

### GA1–GA7 策略变体对照

| 变体 | 父本选择 | 父本来源 | 精英比例 | 未交叉处理 | 变异对象 | 精英是否变异 | 预期速度 | 预期质量 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| GA1（基准） | 轮盘赌 | 全部种群 | 10% | 直接复制 | 仅交叉子代 | 否 | 中等 | 中等 | 经典 GA，平衡探索与开发 |
| GA2 | 锦标赛(k=2) | 全部种群 | 10% | 直接复制 | 仅交叉子代 | 否 | 稍快 | 较好 | 选择压力更稳定 |
| GA3 | 轮盘赌 | 前 20% 精英 | 10% | 直接复制 | 仅交叉子代 | 否 | 快 | 较低 | 父本局限精英，易早熟 |
| GA4 | 轮盘赌 | 全部种群 | 0% | 直接复制 | 仅交叉子代 | 否 | 慢 | 可能高 | 无精英保留，强探索 |
| GA5 | 轮盘赌 | 全部种群 | 50% | 直接复制 | 仅交叉子代 | 否 | 很快 | 较低 | 高精英比例，开发过度 |
| GA6 | 轮盘赌 | 全部种群 | 10% | 复制后变异 | 所有新个体 | 否 | 稍慢 | 较高 | 增加变异强度，保持多样性 |
| GA7 | 轮盘赌 | 全部种群 | 10% | 直接复制 | 仅交叉子代 | 是 | 中等 | 不稳定 | 精英变异可能破坏最优 |

变体的具体策略字典在 `configs/optimizerConfig.yaml` 文件末尾以注释形式保留，由 `pipelines/experiment.py` 加载使用。

## SA vs GA 对比

| 维度 | SA | GA |
|---|---|---|
| 搜索方式 | 单点逐步降温 | 种群逐代进化 |
| 新方案来源 | `generate_new_plan` 1 次/迭代 | `initial` + `crossover` + `mutate` |
| 接受准则 | Metropolis | 适应度排序 + 精英保留 |
| 扰动强度控制 | `T_ratio = cur_temp / T_start`（温度自然下降） | `T_ratio = max(0.0001, 1 − gen_ratio)`（按代数反向计算） |
| 还原机制 | generator 内 stagnation 时触发 | 仅在 mutate 链中可能触发 |
| 早停阈值 | 固定 | 随代数 `PiecewiseLinear` 收紧 |
| 是否传 change_scale_n 给 calculator | 是（触发 λ 调度） | 否 |
| 策略可组装 | 无 | `GeneticAlgorithmStrategy` 七变体 |

## 输出与产物

`.run()` 返回 `torch.Tensor` 形状 `[1, 1, 128, 128]`（`self.best_plan`）。所有副产物写入 `save_path/index/`：

```
save_path/index/
├── plans_{idx}/                  # best_plan.TIF、ori_plan.TIF、tgt_plan.TIF（+ .tfw）
├── performance_{idx}/            # FRAGSTATS / 性能评估中间产物
├── performance.json              # 最终性能指标
├── charts/
│   ├── total_score.png
│   ├── change_score.png
│   ├── ES_score.png
│   ├── ES详情.png                # 四项 ES 分项曲线
│   ├── landscape_score.png       # 仅当 use_landscape=True
│   └── Multi_Scores.png          # 仅 GA：种群 best/worst/avg/std 曲线
├── history/iter_{i}.png          # 按 visualize_frequency 出图
├── best_score.png                # 最终对比面板（ori vs best vs final vs pred_diff vs opt_diff vs real_diff）
├── total_score_list.npy          # 完整收敛轨迹
└── convergence.json              # 消融实验 A 用：{iter_to_95pct, final_best, wall_time_sec, ...}
```

**注意符号约定**：SA 的 `total_score_list` 存 tgt_func（越小越好），GA 存 fitness（越大越好）；`convergence.json` 的 `final_best` 已统一回 tgt_func 语义，跨 runner 比较请用 `convergence.json` 而非直接读 `.npy`。

## 配置

runner 实际消费的配置字段全部位于 `configs/optimizerConfig.yaml`（字段含义见 `configs/_README.md`，此处不重复）：

| 配置块 | 消费者 | 关键字段 |
|---|---|---|
| `runner` | runner 实例化前的 pipeline | `type`、`use_landscape`、`optimize_num`、`scenario` |
| `runner.weight.<scenario>` | runner | 四项 ES 权重，传给 `weight_dict` |
| `runner.common_params` | runner | carbon_pool / maxent / frg_model 路径与 maxent_list |
| `SA` | `SimulatedAnnealing` | `T_start`、`T_end`、`cooling_rate`、`max_noimprove_iter`、`max_beststuck_iter`、`print_frequency`、`visualize_frequency` |
| `GA` | `GeneticAlgorithm` | `ori_pop_size`、`final_pop_size`、`max_gen`、`max_noimprove_gen`、`max_beststuck_gen`、`crossover_rate`、`mutation_rate`、`scale_ratio` |
| `GA.strategy` | `GeneticAlgorithmStrategy` | 见上节策略键表 |
| `model` | `load_ANN_model` | `type`、`model_path` 及对应架构超参 |

消融评分参数在 `configs/score_cfg.yaml`，由 `score_cfg` 关键字参数传入（参见 `calculator/_README.md` 与 `configs/_README.md`）。

## 与其他模块的协作

按调用方向整理：

| 方向 | 模块 | 调用 |
|---|---|---|
| runner → calculator | `calculator/tgt_function_cal.py` | `cal_tgt_function`（每步 / 每个体一次） |
| runner → calculator_utils | `utils/calculator_utils.py` | `get_info_dict` / `get_pred_diff_distance_map` / `get_lu_distancemap_dict` / `get_numfeature_dict` |
| runner → generator | `generator/sa_generator.py` | SA：`generate_new_plan` |
| runner → generator | `generator/ga_generator.py` | GA：`initial_new_population` / `crossover` / `mutate` |
| runner → runner_utils | `utils/runner_utils.py` | `load_ANN_model` / `get_pred_diff` / `get_lu_labels` / `get_grid_restrict_area` / `get_cls_restrict_area` / `get_scale_minmax_dict` / `get_initial_score_list` / `save_plan_and_get_performance` / `draw_score_list` |
| runner → libs | `libs/const.py` | `COLOR_MAP` / `RAW_DATA_ROOT` / `PIXEL_NUM_THRESHOLD_ISCHANGE` / `FACTOR_PARAMS_DICT` / `ALL_LOG_FACTORS` |
| pipeline → runner | `pipelines/run_runner.py` | `create_runner(...).run()`；遍历 `scenario_list × dataloader`；用 `best_plan != ori_plan` 累计 `actual_optimize_num` |

## 已知问题 / 注意事项

- **`elite_ratio` 必须放在 `GA.strategy` 内**（不在 `GA` 顶层）。`GeneticAlgorithmStrategy` 通过 `vars(strategy).get('elite_ratio', 0.1)` 读取，写错位置时静默使用默认值 0.1
- **GA fitness 评估不传 `cur_temp`**，calculator 的 λ_ES / λ_LS 自适应调度对 GA 不生效；这是设计上的非对称——GA 内部用代数代替温度，但 calculator 端没有等价机制
- **跳过守卫的副作用**：`pred_diff_pixelnum < PIXEL_NUM_THRESHOLD_ISCHANGE` 时 `run()` 原样返回 `ori_plan`。下游 pipeline 依赖 `best_plan == ori_plan` 来判定"未实际优化"，修改该返回行为会破坏 `actual_optimize_num` 统计与消融实验计数
- **`total_score_list.npy` 符号不统一**：SA 存 tgt_func（小=好），GA 存 fitness（大=好）。跨 runner 比较请走 `convergence.json` 的 `final_best`（已统一回 tgt_func）
