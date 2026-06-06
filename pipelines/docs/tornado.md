# Ablation Pipeline — Tornado 参数粗筛数据流说明

> 本文档聚焦 `pipelines/tornado.py` 在 `mode='tornado'` 下的内部方法逻辑与数据流。
>
> 在项目整体架构中，本 pipeline 属于 **Stage-2 优化搜索** 的实验平台（详见 `pipelines/README.md`）。
> 对应 `main.py` 入口：`run_tornado_pipeline(..., mode='tornado')`。
>
> Tornado 模式用于**快速筛选对结果影响最大的核心参数**，通过粗粒度扰动（每个参数 2~3 个取值）观察指标偏离基线的幅度。

---

## 一、关键参数

### 1.1 `run_tornado_pipeline()`（main.py 入口）

| 参数 | 来源 | 含义 |
|------|------|------|
| `opt_cfg_path` | 硬编码 `"configs/optimizerConfig.yaml"` | 优化器+模型+数据配置 |
| `score_cfg_path` | 硬编码 `"configs/score_cfg.yaml"` | 目标函数各项权重和阈值 |
| `idxs` | 从 `_data/representative_samples.json` 读取 | 要跑的样本索引列表（默认 fallback: `[479, 596, 760, 634, 1903]`）|
| `mode` | 硬编码 `'tornado'` | 实验模式 |
| `n_repeats` | 硬编码 `2` | 每个配置重复次数（tornado 为快速筛选，重复次数较少）|

### 1.2 `tornado_pipeline()`（pipelines/tornado.py 核心）

| 参数 | 类型 | 作用 |
|------|------|------|
| `is_processed` | `bool` | 是否断点续传 |
| `processed_timestamp` | `Optional[str]` | 断点续传时间戳 |
| `top_k_params` | `Optional[List[Dict]]` | **sensitivity 模式专用**，tornado 模式下无需传入 |

---

## 二、数据流总览

```
main.py
  ↓ 读取 _data/representative_samples.json → idxs
  ↓ 调用 tornado_pipeline(..., mode='tornado', n_repeats=2)

pipelines/tornado.py
  ├─ 1. 加载配置
  │     ├─ optimizerConfig.yaml → opt_cfg
  │     └─ score_cfg.yaml → base_score_cfg
  │
  ├─ 2. 构造实验矩阵 (utils.tornado_utils.build_experiment_matrix)
  │     mode='tornado' → 1 组基线 + 20 组参数扰动实验
  │     ├─ A_baseline: 无覆写，正常配置
  │     └─ B_<param>=<value>: 单参数扰动
  │
  ├─ 3. 遍历每个实验
  │     ├─ 深拷贝 score_cfg → 按 overrides 覆写 → apply_const_overrides()
  │     ├─ 深拷贝 type_config → 按 opt_overrides 覆写 SA/GA 参数
  │     ├─ 设置确定性随机种子（torch/np/random）
  │     │
  │     └─ 4. 遍历样本 + 重复 run
  │           ├─ 过滤 idxs
  │           ├─ process_batch() → data_dict
  │           └─ create_runner() → runner.run()
  │
  └─ 6. 保存 manifest.json + experiment_summary.json
```

---

## 三、实验矩阵

```python
build_experiment_matrix(mode='tornado', n_repeats=2)
```

返回 **21 项**实验配置（1 基线 + 20 扰动）：

### 3.1 基线实验

| name | overrides | opt_overrides | flags | group |
|------|-----------|---------------|-------|-------|
| `A_baseline` | `{}` | `{}` | `guided=True`, `disable_early_stop=False`, `skip_threshold_check=False` | A |

### 3.2 参数扰动实验

`_TORNADO_PARAMS` 中定义 8 个参数，每个参数 2~3 个扰动值：

| # | 参数路径 | 基线 | 扰动值 | 类别 |
|---|---------|------|--------|------|
| 1 | `tgt_function.lambda_ES_high` | 1000 | [500, 1500, 2000] | score_cfg |
| 2 | `tgt_function.lambda_ES_low` | 5000 | [2500, 7500, 10000] | score_cfg |
| 3 | `tgt_function.lambda_LS_high` | 5 | [2, 8, 10] | score_cfg |
| 4 | `change_score.inside_reward_coef` | 10 | [5, 15, 20] | score_cfg |
| 5 | `runner_utils.sigmoid_threshold` | 0.5 | [0.3, 0.7] | score_cfg |
| 6 | `landscape.pure_base` | 0.3 | [0.1, 0.5] | score_cfg |
| 7 | `T_start` | 100 | [50, 200] | opt_cfg |
| 8 | `cooling_rate` | 0.9995 | [0.999, 0.9998] | opt_cfg |

每个取值生成一个实验，命名格式：`B_<dotted_path>=<value>`，例如：
- `B_tgt_function.lambda_ES_high=500`
- `B_T_start=50`

### 3.3 参数覆写机制

**score_cfg 参数**（如 `lambda_ES_high`）：
```python
overrides = {'tgt_function.lambda_ES_high': 500}
# → _set_by_path(score_cfg, 'tgt_function.lambda_ES_high', 500)
```

**opt_cfg 参数**（如 `T_start`）：
```python
opt_overrides = {'T_start': 50}
# → _set_by_path(type_config, 'T_start', 50)
# type_config 即 opt_cfg.SA 或 opt_cfg.GA
```

---

## 四、与 ablation / stability 模式的核心区别

| 维度 | `mode='tornado'` | `mode='ablation'` | `mode='stability'` |
|------|------------------|-------------------|-------------------|
| 实验组数 | 21 组（1+20） | 3 组（A/B/C） | 1 组（A_guided） |
| 目的 | **筛选敏感参数** | 验证模块有效性 | 验证随机稳定性 |
| 参数覆写 | 单参数扰动 | B 组覆写 `disable_change_score` | 无 |
| `disable_early_stop` | `False`（正常早停） | `True`（跑到底） | `True`（跑到底） |
| SA.T_end | **0.1**（正常） | **1e-6**（极端收敛） | 0.1（正常） |
| GA.max_gen | **3000**（正常） | **8000**（极端收敛） | 3000（正常） |
| 随机种子 | ✅ 确定性种子 | ❌ 无 | ❌ 无 |
| 时间成本 | 5 样本 × 21 组 × 2 次 ≈ **~54 h** | 5×3×5 ≈ 3~6 h | 5×1×5 ≈ 1~2 h |

> **为什么 tornado 用正常配置？** tornado 目的是筛选日常使用的敏感参数，而非测试极端收敛行为。若用 `T_end=1e-6`，耗时翻倍且偏离实际场景。

---

## 五、结果目录结构

```
results/
└── tornado/
    └── 20260602_HHMMSS/               ← timestamp
        ├── experiment_summary.json      ← 全局清单（含 parameters_tested）
        │
        ├── A_baseline/                ← 基线
        │   ├── manifest.json
        │   ├── run_0/
        │   │   ├── 10_479_5/
        │   │   │   ├── convergence.json
        │   │   │   ├── performance.json
        │   │   │   └── total_score_list.npy
        │   │   └── 10_596_5/
        │   └── run_1/
        │
        ├── B_tgt_function.lambda_ES_high=500/
        ├── B_tgt_function.lambda_ES_high=1500/
        ├── B_tgt_function.lambda_ES_high=2000/
        ├── B_T_start=50/
        ├── B_T_start=200/
        └── ...（共 20 个 B_* 目录）
```

---

## 六、断点续传

与 stability / ablation 模式相同，支持 `is_processed=True`：

```python
tornado_pipeline(
    ..., mode='tornado', n_repeats=2,
    is_processed=True,
    processed_timestamp="20260602_120000",
)
```

---

## 七、后处理出图

### 7.1 入口

```python
from pipelines.tornado_plots import run_tornado_plots
run_tornado_plots("results/tornado/<timestamp>", baseline_name='A_baseline')
```

### 7.2 输出文件

| 文件 | 说明 |
|------|------|
| `plots/tornado_final_best.png` | 各参数对 `final_best` 的归一化扰动幅度 |
| `plots/tornado_iter_to_95pct.png` | 对收敛速度的影响 |
| `plots/tornado_wall_time_sec.png` | 对运行时间的影响 |
| `plots/tornado_ES.food.best_ori_diff.png` | 对粮食服务的影响 |
| `plots/tornado_ES.carbon.best_ori_diff.png` | 对碳服务的影响 |
| `plots/tornado_ES.water.best_ori_diff.png` | 对水源服务的影响 |
| `plots/tornado_ES.culture.best_ori_diff.png` | 对文化服务的影响 |
| `plots/tornado_<metric>_summary.json` | 各 metric 的数值摘要 |

### 7.3 解读 tornado 图

- **横轴**：归一化偏离幅度 `(perturbed - baseline) / |baseline|`
- **纵轴**：按区间宽度排序的参数，**越靠上越敏感**
- **条形越长**：该参数扰动导致结果变化越大，越值得深入线搜索
- **左右分布**：左=负向偏离，右=正向偏离

---

## 八、下一步

tornado 筛选出敏感参数后，将其传入 `mode='sensitivity'` 做细粒度一维线搜索。详见 `sensitivity.md`。
