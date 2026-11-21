# Tornado 参数粗筛数据整合流程

> 本文档说明 tornado 参数粗筛如何从每次 run 的原始输出，逐步生成 CSV 表格、tornado 图和跨指标综合排名。
> 当前口径聚焦**解的质量**：ES 使用 best 相对于 target 的比例差，变化区域只使用与 target 对齐的 F1 和 IOU。
>
> 注：文中数值示例沿用 λ=1× 历史数据；当前 `configs/score_cfg.yaml` 已改为 λ×200（λ_ES 200000~1000000，λ_LS 1000~200）。

---

## 0. 当前代码入口与 mode

### 0.1 实验执行入口

```python
from pipelines.tornado import tornado_pipeline

tornado_pipeline(
    opt_cfg_path='configs/optimizerConfig.yaml',
    score_cfg_path='configs/score_cfg.yaml',
    idxs=idxs,
    mode='tornado',
    n_repeats=3,
)
```

主入口封装在 `main.py`：

```python
run_tornado_pipeline(max_samples=None)
```

### 0.2 后处理入口

```python
from pipelines.tornado_plots import run_tornado_plots

run_tornado_plots(
    experiment_root='results/tornado/<timestamp>',
    baseline_name='A_baseline',
    mode='tornado',      # 可选: 'tornado' | 'sensitivity' | 'auto'
)
```

主入口封装在 `main.py`：

```python
run_tornado_plots_pipeline(
    experiment_root='results/tornado/<timestamp>',
    mode='tornado',
    baseline_name='A_baseline',
)
```

`mode='auto'` 时，后处理会优先读取 `experiment_summary.json` 中的 `mode` 字段；若没有，则根据目录中是否存在 `A_baseline` / `B_*` / `S_*` 自动推断。

---

## 1. 指标口径：只用 best-vs-target 解质量做 tornado 排名

### 1.1 为什么不用 `final_best`、T95、Runtime

Tornado 的目标是回答：**某个参数变化是否会改变最终解的质量？**

因此默认 tornado 排名不使用会被参数机械影响的指标：

| 不用于默认 tornado 排名的指标 | 原因 |
| ----------------------------- | ---- |
| `final_best` | 目标函数值会被乘数/权重直接缩放。例如 `inside_reward_coef` 作为乘数进入距离奖励项，可能等比例改变目标函数数值，但不一定改变最终方案。 |
| `iter_to_95pct` | 受温度、冷却率、最大代数、早停等运行控制参数直接影响。 |
| `wall_time_sec` | 受迭代次数、早停、硬件波动和实现细节影响，不是解质量指标。 |
| `total_iter` | 运行过程指标，不是解质量。 |
| `es_score` | 目标函数内部 ES 标量，仍可能受权重/缩放口径影响。 |

这些字段仍保留在 `tornado_runs.csv` 和 `tornado_mean_table.csv` 中，作为原始运行记录，但**不进入默认 tornado 偏差、排名、综合得分和图**。

> 关于各指标 run-to-run 波动 (CV%) 的完整评估，见 [§8 统计可靠性评估](#8-统计可靠性评估)。已验证：`iter_to_95pct` 94% 不可靠，`FINAL.FOM` 90% 不可靠，`es_score` 和 `FINAL.kappa` 反而 100% 可靠。

### 1.2 默认参与 tornado 排名的指标

当前默认指标常量为 `pipelines/tornado_plots.py::TORNADO_PERFORMANCE_METRICS`：

```python
TORNADO_PERFORMANCE_METRICS = [
    'ES.food.best_tgt_diff',
    'ES.carbon.best_tgt_diff',
    'ES.water.best_tgt_diff',
    'ES.culture.best_tgt_diff',
    'Change_area.real_metrics.iou',
    'Change_area.real_metrics.f1_score',
]
```

| 指标 | 来源 | 含义 |
| ---- | ---- | ---- |
| `ES.*.best_tgt_diff` | `performance.json::ES` | `(best_ES - target_ES) / target_ES`，best 相对于 target 的比例差 |
| `Change_area.real_metrics.iou` | `performance.json::Change_area.real_metrics` | 最优方案变化区域与真实 target 变化区域的 IoU |
| `Change_area.real_metrics.f1_score` | 同上 | 最优方案变化区域与真实 target 变化区域的 F1 |

> `Change_area.pred_metrics.*` 只反映与 ANN 预测变化区的一致性，不再进入默认 tornado 粗筛。`FINAL.FOM` / `FINAL.kappa` 也不再作为默认指标，只保留在原始表中用于追溯。

---

## 2. 表1：原始运行级数据 `tornado_runs.csv`

### 2.1 生成函数

```python
pipelines.tornado_plots.write_tornado_runs_csv(store)
```

### 2.2 颗粒度

每行对应一次独立 run：

```text
sample_id × experiment × run_idx
```

### 2.3 数据来源

| 文件 | 字段 |
| ---- | ---- |
| `convergence.json` | `final_best`, `iter_to_95pct`, `wall_time_sec`, `total_iter`, `es_score` |
| `performance.json` | `ES.*.best_ori_diff`, `ES.*.best_tgt_diff`, `Change_area.*`, `FINAL.FOM`, `FINAL.kappa` |

### 2.4 示例

| sample_id | experiment | run_idx | algorithm | final_best | ES.carbon.best_tgt_diff | Change_area.real_metrics.iou | Change_area.real_metrics.f1_score |
| --------- | ---------- | ------- | --------- | ---------- | ----------------------- | ---------------------------- | --------------------------------- |
| 10_479_5 | A_baseline | 1 | SA | 0.876 | -0.018 | 0.31 | 0.47 |
| 10_479_5 | B_lambda_ES_high=1500 | 1 | SA | 0.881 | -0.022 | 0.33 | 0.50 |

> 表1保留 convergence 指标只是为了追溯运行过程，不代表它们参与 tornado 排名。

---

## 3. 表2：样本-配置均值 `tornado_mean_table.csv`

### 3.1 生成函数

```python
pipelines.tornado_plots.build_tornado_mean_table(store)
```

内部复用：

```python
utils.exp_sweep_utils.aggregate_runs(exp_dir, sample_id)
```

### 3.2 颗粒度

每行对应一个样本在一个实验配置下的多 run 均值：

```text
sample_id × experiment
```

### 3.3 示例

| sample_id | experiment | ES.carbon.best_tgt_diff_mean | Change_area.real_metrics.iou_mean | Change_area.real_metrics.f1_score_mean |
| --------- | ---------- | ---------------------------- | --------------------------------- | -------------------------------------- |
| 10_479_5 | A_baseline | -0.0195 | 0.315 | 0.479 |
| 10_479_5 | B_lambda_ES_high=1500 | -0.0225 | 0.332 | 0.499 |

---

## 4. 表3：参数-取值归一化偏差 `tornado_deviation_table.csv`

### 4.1 生成函数

```python
pipelines.tornado_plots.build_tornado_deviation_table(
    store,
    baseline_name='A_baseline',
)
```

### 4.2 计算公式

先计算基线跨样本均值：

\[
baseline_{metric} = \frac{1}{N_{samples}} \sum_s mean_{metric}(s, A\_baseline)
\]

再计算扰动实验跨样本均值：

\[
perturbed_{metric}(knob, value) = \frac{1}{N_{samples}} \sum_s mean_{metric}(s, B\_knob=value)
\]

归一化偏差为：

\[
Deviation(knob, value, metric) = \frac{perturbed_{metric} - baseline_{metric}}{|baseline_{metric}|}
\]

### 4.3 输出列

| 列名 | 含义 |
| ---- | ---- |
| `metric` | 当前指标，如 `ES.carbon.best_tgt_diff` |
| `parameter` | 参数路径，如 `tgt_function.lambda_ES_high` |
| `value` | 扰动取值 |
| `baseline` | 该指标在 `A_baseline` 下的跨样本均值 |
| `normalized_deviation` | 相对 baseline 的归一化偏差 |

---

## 5. 表4：参数敏感度排名 `tornado_ranking_table.csv`

### 5.1 生成函数

```python
pipelines.tornado_plots.build_tornado_ranking_table(
    store,
    baseline_name='A_baseline',
)
```

### 5.2 排名逻辑

对每个指标、每个参数，取该参数所有扰动取值下的归一化偏差最小值与最大值，计算区间宽度：

\[
RangeWidth(parameter, metric) = max(Deviation) - min(Deviation)
\]

区间越宽，说明该参数越会改变该解质量指标。

### 5.3 输出列

| 列名 | 含义 |
| ---- | ---- |
| `metric` | 指标 |
| `rank` | 该 metric 下的敏感度排名 |
| `parameter` | 参数路径 |
| `min_deviation` | 最小归一化偏差 |
| `max_deviation` | 最大归一化偏差 |
| `range_width` | 偏差区间宽度 |
| `sensitivity` | high / medium / low |
| `perturbed_values` | 参与排名的扰动值 |

---

## 6. 表5：跨指标综合敏感度 `tornado_composite_ranking.csv`

### 6.1 生成函数

```python
pipelines.tornado_plots.plot_tornado_composite(
    store,
    baseline_name='A_baseline',
)
```

### 6.2 计算公式

默认对 6 个解质量指标等权汇总：

\[
CompositeScore(parameter) = \sum_{m \in M} \frac{1}{|M|} RangeWidth(parameter, m)
\]

其中：

\[
M = \{ES.food.best\_tgt\_diff, ES.carbon.best\_tgt\_diff, ES.water.best\_tgt\_diff, ES.culture.best\_tgt\_diff, IoU, F1\}
\]

### 6.3 输出

- `tornado_composite_ranking.csv`
- `tornado_composite_summary.json`
- `tornado_composite.png`

---

## 7. 图示输出

### 7.1 单指标 tornado 图

每个默认指标输出一张图：

- `plots/tornado_ES.food.best_tgt_diff.png`
- `plots/tornado_ES.carbon.best_tgt_diff.png`
- `plots/tornado_ES.water.best_tgt_diff.png`
- `plots/tornado_ES.culture.best_tgt_diff.png`
- `plots/tornado_Change_area.real_metrics.iou.png`
- `plots/tornado_Change_area.real_metrics.f1_score.png`

### 7.2 综合图

- `plots/tornado_composite.png`

---

## 8. 统计可靠性评估

> tornado 每个样本每个参数设置仅 3 次重复运行。使用 `scripts/check_run_reliability.py` 评估 run-to-run 波动 (CV%) 判断各指标在 tornado 中的可信度。

### 8.1 评估方法

```bash
python scripts/check_run_reliability.py results/tornado/<timestamp> --run-range 0-2
```

对每个 `(样本, 实验组, 指标)` 组合计算 CV% = std / mean × 100，并按阈值分类：

| CV% 范围 | 分类 | 含义 |
| -------- | ---- | ---- |
| ≤ 5% | highly_reliable | 3 次运行高度一致，可用于 tornado 排名 |
| 5%~10% | reliable | 可接受波动 |
| 10%~20% | moderate | 需谨慎解读 |
| > 20% | unreliable | 波动过大，可能误导排名 |
| mean=0 → CV=inf | unreliable (FOM专属) | 所有 run 均无法计算该指标 |

### 8.2 已验证的两批数据

基于 `20260606_033319`（3 样本 × 21 组）和 `20260611_023835`（10 样本 × 11 组）的评估结果：

#### 8.2.1 默认 tornado 排名指标的可靠性

| 指标 | highly_reliable | reliable | moderate+unreliable | 评估 |
| ---- | --------------- | -------- | ------------------- | ---- |
| `ES.carbon.best_tgt_diff` | 6% | 12% | **81%** | ⚠️ 高 CV%，但根因是 mean→0 |
| `ES.food.best_tgt_diff` | 10% | 18% | **72%** | ⚠️ 高 CV%，但根因是 mean→0 |
| `ES.water.best_tgt_diff` | 14% | 9% | **77%** | ⚠️ 最高 CV%（最大 1946%），噪声大 |
| `ES.culture.best_tgt_diff` | 12% | 21% | **68%** | ⚠️ 高 CV%，但根因是 mean→0 |
| `Change_area.real_metrics.iou` | **87%** | 6% | 7% | ✅ 可靠 |
| `Change_area.real_metrics.f1_score` | **92%** | 6% | 1% | ✅ 最可靠 |

> **关键判断**：4 个 ES tgt_diff 指标 CV% 高的根因是均值接近 0（优化成功逼近 target），而非标准差大。avg_std / avg\|mean\| 仅 0.16~0.26，绝对噪声小于绝对信号。**对 tornado 排名而言，参数引起的组间差异是否超过组内噪声才是关键，单组 CV% 高不代表排名无效。**

#### 8.2.2 已排除指标的实际可靠性

| 指标 | 不可靠比例 | 排除原因（已有） |
| ---- | --------- | ---------------- |
| `iter_to_95pct` | 94% / 88% | SA 收敛迭代数天然高方差 |
| `FINAL.FOM` | 90% / 90% | 10_1417_5 等样本无变化→FOM≡0 |
| `ES.water.best_ori_diff` | 52% / 82% | 第二批数据恶化明显 |
| `wall_time_sec` | 33% / 20% | 受硬件波动影响 |

#### 8.2.3 始终可靠的指标（但不用于 tornado 排名）

| 指标 | highly_reliable | 不用于排名的原因 |
| ---- | --------------- | ---------------- |
| `FINAL.kappa` | **100%** | 衡量 ANN 预测准确性，非解质量 |
| `es_score` | **100%** | 目标函数内部标量，受权重缩放影响 |
| `Landscape.land.best_plan.CONTAG` | **100%** | 景观指标，非 tornado 默认排名维度 |
| `Landscape.land.best_plan.SHEI` | **99%** | 同上 |

### 8.3 已知可疑数据

#### 8.3.1 `FINAL.FOM ≡ 0`（18 条记录）

| 样本 | 影响范围 | 原因 |
| ---- | -------- | ---- |
| `10_1417_5` | 全部 11 组 FOM=0 | 该样本无变化区域，变化检测和实际变化均为空 → FOM 分子分母为 0 |
| `10_1788_5` | 7/32 组 FOM=0 | 变化区域极小，部分 run 未检测到变化 |

> 这些记录已在 tornado 排名中排除（§1.1），不影响结果。若未来 tornado 覆盖更多样本，建议在 `experiment_summary.json` 中标记 `fom_zero_samples` 便于追溯。

#### 8.3.2 极端高 CV% 记录（CV% > 200，共 82 条）

集中在 `ES.water.best_tgt_diff`（27条）和 `ES.food.best_tgt_diff`（18条），典型如：

| 样本/组 | 指标 | mean | std | CV% |
| ------- | ---- | ---- | --- | --- |
| `10_1788_5/A_baseline` | ES.water.best_ori_diff | 0.00004 | 0.00184 | 4338% |
| `10_1723_5/A_baseline` | ES.water.best_tgt_diff | 0.00795 | 0.02523 | 317% |
| `10_1788_5/B_T_start=200` | ES.carbon.best_tgt_diff | 0.00019 | 0.00055 | 295% |

这些极端记录几乎全部来自 sample `10_1788_5`，该样本变化区域极小，ES 变化量接近测量噪声水平。

### 8.4 对 tornado 排名的实际影响

综合以上分析，对 tornado 排名的建议：

1. **IoU 和 F1 是最可信的排名依据**：CV% 低，物理意义明确（变化区域匹配质量）。
2. **ES tgt_diff 指标的排名需谨慎解读**：高 CV% 不代表排名错误，但应关注以下信号：
   - 若某参数在 **IoU/F1** 上无影响但在 **ES tgt_diff** 上有大幅偏差，可能是噪声。
   - 若某参数在**所有 6 个指标**上方向一致，则可信度更高。
3. **建议在 tornado 输出中增加 CV% 列**：在 `tornado_mean_table.csv` 中附加各指标 CV%，便于在排名时过滤高噪声条目。
4. **`es_score` 和 `FINAL.kappa` 的 100% 可靠性值得重新考虑**：如果未来需要更稳定的排名，可将 kappa 纳入辅助指标（它虽然不是解质量指标，但对参数变化仍敏感且高度可重复）。
5. **样本筛选**：对于变化区域极小（如 `10_1788_5`、`10_1417_5`）的样本，其 ES diff 和 FOM 指标天然噪声大，可考虑在 tornado 粗筛阶段排除或单独标注。

---

## 附录：数据流总结

| 处理阶段 | 输入 | 输出 | 颗粒度 |
| -------- | ---- | ---- | ------ |
| 1. 原始记录 | 每次运行输出 | `tornado_runs.csv` | 样本 × 实验 × Run |
| 2. 配置均值 | 表1 | `tornado_mean_table.csv` | 样本 × 实验 |
| 3. 归一化偏差 | 表2 + 基线均值 | `tornado_deviation_table.csv` | metric × 参数 × 取值 |
| 4. 敏感度排名 | 表3 | `tornado_ranking_table.csv` | metric × 参数 |
| 5. 综合排名 | 表4（多 metric） | `tornado_composite_ranking.csv` | 参数 |
| 6. 图表 | 表4/表5 | `tornado_<metric>.png`, `tornado_composite.png` | 可视化 |

> 核心函数链：`aggregate_runs()` → `ExperimentResultStore.get_baseline_metric()` + `get_knob_deviations()` → `build_tornado_*_table()` / `plot_tornado()`。
