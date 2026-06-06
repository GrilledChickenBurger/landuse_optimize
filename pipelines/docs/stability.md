# Ablation Pipeline — Stability 模式数据流说明

> 本文档聚焦 `pipelines/ablation.py` 在 `mode='stability'` 下的内部方法逻辑与数据流。
>
> 在项目整体架构中，本 pipeline 属于 **Stage-2 优化搜索** 的实验平台（详见 `pipelines/README.md`）。
> 对应 `main.py` 入口：`run_ablation_pipeline(..., mode='stability')`。

---

## 一、关键参数

### 1.1 `run_ablation_pipeline()`（main.py 入口）

| 参数 | 来源 | 含义 |
|------|------|------|
| `opt_cfg_path` | 硬编码 `"configs/optimizerConfig.yaml"` | 优化器+模型+数据配置 |
| `score_cfg_path` | 硬编码 `"configs/score_cfg.yaml"` | 目标函数各项权重和阈值 |
| `idxs` | 从 `_data/representative_samples.json` 读取 `idxs_for_pipeline` | 要跑的样本索引列表（默认 fallback: `[479, 596, 760, 634, 1903]`）|
| `mode` | 硬编码 `'stability'` | 实验模式：`'stability'` 或 `'ablation'` |
| `n_repeats` | 硬编码 `5` | 每个样本重复跑的次数 |

### 1.2 `ablation_pipeline()`（pipelines/ablation.py 核心）

| 参数 | 类型 | 作用 |
|------|------|------|
| `is_processed` | `bool` | 是否断点续传（默认 `False`，每次都重新跑）|
| `processed_timestamp` | `Optional[str]` | 断点续传用的时间戳（`is_processed=True` 时必须提供）|

---

## 二、数据流总览

```
main.py
  ↓ 读取 _data/representative_samples.json → idxs_for_pipeline (10 个代表性样本)
  ↓ 调用 ablation_pipeline(..., mode='stability', n_repeats=5)

pipelines/ablation.py
  ├─ 1. 加载配置
  │     ├─ optimizerConfig.yaml → opt_cfg (Namespace, runner.type 决定 SA/GA)
  │     └─ score_cfg.yaml → base_score_cfg (Namespace)
  │
  ├─ 2. 构造实验矩阵 (build_experiment_matrix)
  │     mode='stability' → 只有 1 组实验: A_guided × n_repeats 次
  │     ├─ overrides: {} (不改任何配置)
  │     ├─ flags: {guided=True, disable_early_stop=True}
  │     └─ group: 'A'
  │
  ├─ 3. 遍历每个实验 (仅 A_guided)
  │     ├─ 深拷贝 score_cfg → apply_const_overrides() 同步到 libs.const
  │     ├─ 禁用早停 (disable_early_stop=True), SA/GA 均跑到底
  │     │
  │     └─ 4. 遍历 dataloader 中所有样本
  │           ├─ 过滤: 只保留 idxs 列表中的样本
  │           ├─ process_batch() → data_dict
  │           │
  │           └─ 5. 重复 n_repeats 次 (run_0 … run_{n-1})
  │                 ├─ 生成 save_path:
  │                 │   results/ablation/{timestamp}/A_guided/run_0~4/{index}/
  │                 ├─ create_runner(SA/GA,
  │                 │               guided=True,
  │                 │               disable_early_stop=True)
  │                 └─ runner.run()
  │
  ├─ 6. 保存 manifest.json + experiment_summary.json
  │
  └─ 7. mode == 'stability' → write_stability_runs_csv()
        扫描所有 run/sample 的 convergence.json,
        反查 _data/representative_samples.json 得到 n_pred / level (Q1-Q5),
        落盘 results/ablation/{timestamp}/plots/stability_runs.csv
```

---

## 三、稳定性实验的具体行为（mode='stability'）

### 3.1 实验矩阵

```
实验矩阵只有一项:
┌─────────────────────────────────────────┐
│ name: "A_guided"                        │
│ overrides: {}                           │
│ flags: {guided=True, disable_early_stop=True} │
│ n_repeats: 5                            │
│ group: "A"                              │
└─────────────────────────────────────────┘
```

**关键行为**：
- 每个样本跑 **5 次独立重复**
- 每次使用相同的 `guided=True`（有 pred_diff 引导）
- `disable_early_stop=True`（不提前终止，跑到 T_end 或 max_gen）
- 没有任何 score_cfg 的覆写（和主流程完全一致）
- **不固定随机种子**：SA/GA 本身就是随机优化算法，每次运行结果天然有波动，这正是稳定性验证的目标对象

### 3.2 SA + GA 双算法的组织方式

`runner.type` 在 `configs/optimizerConfig.yaml` 中是单选（`SA` 或 `GA`），单次 pipeline 只能跑一种算法。要拿到完整 SA + GA 稳定性对比的工作流：

1. 设 `runner.type=SA` + `n_repeats=5`，跑 `run_ablation_pipeline()` → 得 `<SA_timestamp>`
2. 改 `runner.type=GA` + `n_repeats=5`，再跑 `run_ablation_pipeline()` → 得 `<GA_timestamp>`
3. 用 `run_ablation_plots_pipeline(experiment_root=<SA_ts>, sibling_experiment_roots=[<GA_ts>])` 合并出图

两次跑独立产出各自的 `stability_runs.csv`，后处理阶段合并到一个 per_sample/summary 体系中。

### 3.3 样本来源与 Q1–Q5 分层

10 个代表性样本来自 `scripts/select_representative_samples.py` 的产出：

- 对所有满足 `n_pred ≥ pixel_num_threshold_ischange`（取自 `configs/score_cfg.yaml::const.pixel_num_threshold_ischange`）的样本，按 `n_pred` 升序
- 等频划分 5 个分位层级 Q1–Q5（脚本内部 key 为 `"0"`–`"4"`，写入 csv 时映射为 `Q1`–`Q5`）
- 固定 `seed=42` 在每个分位内分层抽样 2 个，共 10 个
- 结果写入 `_data/representative_samples.json`，含三个关键字段：
  - `idxs_for_pipeline`：传给 `run_ablation_pipeline` 的 idx 列表
  - `quantile_ranges`：每个分位的 `min/max/count`
  - `all_records`：所有候选样本的 `index → n_pred` 映射

---

## 四、结果目录结构

```
results/
└── ablation/
    └── 20260602_HHMMSS/                ← 自动生成的 timestamp
        ├── experiment_summary.json       ← 全局清单（mode/idxs/experiments/runner_type）
        ├── A_guided/                   ← 实验名
        │   ├── manifest.json           ← 实验配置（overrides/flags/scenario/weight_dict）
        │   ├── run_0/                  ← 第 1 次重复
        │   │   ├── 10_1417_5/          ← 样本1
        │   │   │   ├── convergence.json        ← final_best / iter_to_95pct / wall_time / ...
        │   │   │   ├── performance.json        ← ES.{food,carbon,water,culture}.best_ori_diff / ...
        │   │   │   ├── total_score_list.npy    ← 逐迭代 total_score (供收敛叠图)
        │   │   │   ├── best_score.png
        │   │   │   ├── charts/                 ← change/ES/landscape/total 分项曲线
        │   │   │   ├── history/                ← 每 500 iter 一帧的过程可视化
        │   │   │   ├── plans_{idx}/            ← best/ori/tgt 的 TIF + tfw
        │   │   │   └── performance_{idx}/      ← fragstats 中间产物
        │   │   ├── 10_2502_5/                  ← 样本2
        │   │   └── ...
        │   ├── run_1/ ... run_4/
        └── plots/                      ← 由 run_ablation_plots_pipeline 产出 (见第七节)
            ├── stability_runs.csv              ← 单算法 100/2=50 行扁平 (跑完 pipeline 立即落盘)
            ├── stability_per_sample.csv        ← (SA+GA 合并后) 20 行 μ/σ/CV/95%CI
            ├── stability_summary.csv           ← (SA+GA 合并后) 2 行汇总 (论文表直接读)
            ├── stability_cv_line.png           ← (SA+GA 合并后) 论文 fig:stability-cv-line
            ├── stability_convergence_{idx}.png ← 单样本多 run 收敛叠图
            ├── stability_convergence_all_A_guided.png
            ├── stability_table_A_guided.csv    ← 单组 per_sample 聚合 (与 stability_per_sample 互补)
            ├── boxplot_final_f.png
            └── ablation_compare_table.csv      ← (ablation 模式) A/B/C 并排
```

### 4.1 单 run 落盘的核心字段

`convergence.json`（SA: `runner/SA.py::_dump_convergence`；GA: `runner/GA.py::_dump_convergence`）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `final_best` | float | **最终最佳目标函数值 F_final**。SA 是 `best_score`；GA 已翻号成 `-best_score`，与 SA 同口径（越小越好）|
| `iter_to_95pct` | int | 首次触及 95% 最优阈值的迭代步数；触不到为 -1 |
| `wall_time_sec` | float | `run()` 主流程耗时 |
| `total_iter` | int | `len(total_score_list)` |
| `guided` / `disable_early_stop` | bool | ablation 开关回显 |
| `iter_to_95pct_repeat` / `ref_best_repeat` | int / float | 聚合阶段由 `recompute_iter_to_95pct` 回写 |

注：**不记录随机种子**，与 stability 实验"算法本身具有不可控随机性"的论文论点一致。

`performance.json`（`utils/performance.py::write_performance_score`）：嵌套结构含 `Change_area / ES / Landscape / FINAL` 四块；stability 后处理目前不直接消费该文件的字段。

---

## 五、与 ablation 模式的区别

| 模式 | 实验组 | 说明 |
|------|--------|------|
| `stability` | 仅 **A_guided** | 验证同一配置重复跑的稳定性 |
| `ablation` | A_guided + B_unguided + C_random | A=正常; B=去掉 change_score; C=随机 mask 约束 |

### 5.1 三组实验的具体差异

| 实验 | overrides | flags | 核心差异 |
|------|-----------|-------|----------|
| **A_guided** | `{}` | `guided=True` | 正常流程，pred_diff 引导 |
| **B_unguided** | `{'ablation.disable_change_score': True}` | `guided=False` | 目标函数去掉 change_score 项 |
| **C_random** | `{}` | `guided=False` | 用随机连通块 mask 替代 pred_diff 的 grid_restrict |

### 5.2 C 组特殊处理

C 组在 runner 初始化前会额外生成 `random_mask`（`AblationPipeline.get_random_mask`）：

```python
# 1. 构造 base_hard_restrict (耕地红线 + LU≠0)
redline = data_dict['restrict']
lu_nonzero = (data_dict['ori_plan'] != 0)
base_hard = torch.logical_and(redline, lu_nonzero)

# 2. 临时跑一次 ANN 拿 pred_diff (样本级别缓存, 多次 run 复用)
pred_diff_c = get_pred_diff(ann_model, ori_plan, factors, device, ...)

# 3. 生成与 pred_diff 面积相近的随机连通块 (mask 种子按 hash(index)+r 派生)
random_mask = make_random_connected_mask(pred_diff_c, base_hard, seed=base_seed + r)
```

该 mask 通过 `random_restrict_mask` 参数传入 runner，替代正常的 grid_restrict。

---

## 六、断点续传机制

若某次运行中途退出，可设置 `is_processed=True` 并传入 `processed_timestamp`：

```python
ablation_pipeline(
    opt_cfg_path="configs/optimizerConfig.yaml",
    score_cfg_path="configs/score_cfg.yaml",
    idxs=idxs,
    mode='stability',
    n_repeats=5,
    is_processed=True,
    processed_timestamp="20260602_120000",  # 上次运行的时间戳
)
```

程序会检查 `results/ablation/{timestamp}/A_guided/run_r/{index}/convergence.json` 是否存在，存在则跳过该 run。续传完成后会重新写出 `stability_runs.csv`（增量样本/run 自动并入）。

---

## 七、结果分析与出图

### 7.1 入口

`main.py::run_ablation_plots_pipeline()` 支持两种调用方式：

```python
# 仅单算法 (SA 或 GA) 出图
run_ablation_plots_pipeline(experiment_root="results/ablation/<SA_ts>")

# SA + GA 合并出图 (生成 per_sample / summary / cv_line)
run_ablation_plots_pipeline(
    experiment_root="results/ablation/<SA_ts>",
    sibling_experiment_roots=["results/ablation/<GA_ts>"],
)
```

`experiment_root` 为 None 时自动找 `results/ablation/` 下最新 timestamp。

### 7.2 数据流

```
main.py::run_ablation_plots_pipeline(experiment_root, sibling_experiment_roots)
  ↓ 调用 pipelines.ablation_plots.run_all_ablation_plots()

pipelines/ablation_plots.py
  ├─ 1. 扫描 experiment_root 下的实验组
  │     stability 模式 → 只有 A_guided/
  │
  ├─ 2. 稳定性叠图
  │     遍历 A_guided/ 下所有样本 index
  │     对每个样本：
  │       ├─ _iter_run_dirs(A_guided, idx) → [run_0, run_1, ..., run_4]
  │       ├─ 读取每个 run 的 total_score_list.npy
  │       ├─ recompute_iter_to_95pct() → 重算 iter_to_95pct_repeat
  │       └─ plot_stability_convergence() → 5 条半透明曲线 + 均值粗线 + ±1 std 阴影
  │
  ├─ 3. 箱线图
  │     plot_boxplot_final() → 各样本 × A_guided 最终 F(S) 箱线图
  │
  ├─ 4. 单组统计表
  │     build_stability_table(experiment_root, 'A_guided')
  │       └─ aggregate_runs() → 每样本 mean / std / 95% CI
  │       └─ 输出 plots/stability_table_A_guided.csv
  │
  └─ 5. SA + GA 合并稳定性分析 (新增, 见 7.4)
        ├─ merge_stability_runs([<SA>/plots/stability_runs.csv,
        │                        <GA>/plots/stability_runs.csv])
        ├─ build_stability_per_sample_csv() → plots/stability_per_sample.csv
        ├─ build_stability_summary_csv()    → plots/stability_summary.csv
        └─ plot_stability_cv_line()         → plots/stability_cv_line.png
```

### 7.3 stability 模式输出文件（单算法）

| 文件 | 说明 |
|------|------|
| `plots/stability_runs.csv` | **(pipeline 跑完立即落盘)** 10 列: `sample_id, n_pred, level, algorithm, config, run_idx, F_final, ESV, Runtime, T95` |
| `plots/stability_convergence_<idx>.png` | A_guided 多次重复收敛曲线叠图（1 条均值粗线 + ±1 std 阴影） |
| `plots/stability_convergence_all_A_guided.png` | 跨样本聚合稳定性曲线 |
| `plots/stability_table_A_guided.csv` | A 组单算法 per_sample 聚合表（`final_best / iter_to_95pct / wall_time / ES.*` 的 mean±std / 95%CI） |
| `plots/boxplot_final_f.png` | 各样本最终 F(S) 箱线图 |

### 7.4 stability 模式输出文件（SA + GA 合并，论文表/图直接读取）

| 文件 | 说明 | 论文位置 |
|---|---|---|
| `plots/stability_per_sample.csv` | 20 行（10 样本 × 2 算法）。字段：`sample_id, n_pred, level, algorithm, n_runs, mu, sigma, cv_pct, ci95_half, ci95_width_ratio_pct` | 正文中位样本 95% CI 论述的数据源 |
| `plots/stability_summary.csv` | 2 行（SA / GA）。字段：`algorithm, n_samples, mean_cv_pct, max_cv_pct, cv_le_5pct_ratio_pct` | `tab:stability-summary` |
| `plots/stability_cv_line.png` | 10 样本按 `n_pred` 升序、SA 与 GA 双折线 CV 对比图（CV=5% 阈值参考线） | `fig:stability-cv-line` |

#### 派生统计计算口径

对每个 (sample_id, algorithm) 在 5 次运行的 `F_final` 上计算：

```
μ      = mean(F_final)
σ      = std(F_final, ddof=1)                    # 样本标准差
cv_pct = σ / |μ| * 100                            # 变异系数 (%), 量纲无关
t_crit = stats.t.ppf(0.975, df=n-1)               # n=5 时 ≈ 2.776
ci95_half            = t_crit * σ / √n
ci95_width_ratio_pct = (2 * ci95_half) / |μ| * 100
```

跨样本汇总（每算法独立）：

```
mean_cv_pct           = mean(cv_pct, axis=samples)
max_cv_pct            = max(cv_pct, axis=samples)
cv_le_5pct_ratio_pct  = count(cv_pct ≤ 5) / n_samples * 100
```

### 7.5 stability 模式不会产出的图

以下图表需要 A/B/C 三组数据，stability 模式因缺少 B/C 组而不会生成（即使本子目录有 stub csv 也只有 A 列有值）：

| 文件 | 原因 |
|------|------|
| `plots/ablation_convergence_<idx>.png` | 需要 A/B/C 三组平均收敛曲线对比 |
| `plots/ablation_compare_table.csv` | 需要三组并排对比（含相对 A 的变化百分比） |

如需生成上述对比图，需用 `mode='ablation'` 跑完 A/B/C 三组后再调用 `run_all_ablation_plots()`。

---

## 八、字段对照速查

论文 `temp.md` 数据拆解 ↔ 代码实现：

| 论文字段 | CSV 列 | 来源 |
|---|---|---|
| sample_id | `sample_id` | 目录名 `<index>`（如 `10_479_5`） |
| n_pred | `n_pred` | `_data/representative_samples.json::all_records` |
| level | `level`（Q1–Q5） | `_data/representative_samples.json::quantile_ranges` + n_pred 映射 |
| algorithm | `algorithm` | `experiment_summary.json::runner_type` |
| run_idx | `run_idx`（1-5） | 目录名 `run_{r}` 解析后 +1 |
| ~~seed~~ | （已弃用） | 不固定，SA/GA 固有随机性即稳定性验证对象 |
| F_final | `F_final` | `convergence.json::final_best` |

---

## 九、关键代码索引

| 关注点 | 位置 |
|---|---|
| stability 模式实验矩阵 | `pipelines/ablation.py::build_experiment_matrix` |
| 主循环（样本 × run） | `pipelines/experiment_base.py::BaseSweepPipeline.run` |
| pipeline 收尾写 runs.csv | `pipelines/ablation.py::ablation_pipeline` 末尾 |
| 扫描 convergence 写 runs.csv | `utils/exp_sweep_utils.py::write_stability_runs_csv` |
| 合并 SA+GA runs | `utils/exp_sweep_utils.py::merge_stability_runs` |
| per_sample 派生统计 | `utils/exp_sweep_utils.py::build_stability_per_sample_csv` |
| summary 跨样本汇总 | `utils/exp_sweep_utils.py::build_stability_summary_csv` |
| CV 双折线绘图 | `pipelines/ablation_plots.py::plot_stability_cv_line` |
| 出图入口编排 | `pipelines/ablation_plots.py::run_all_ablation_plots` |
| 样本选择脚本 | `scripts/select_representative_samples.py` |
| 代表性样本元数据 | `_data/representative_samples.json` |
