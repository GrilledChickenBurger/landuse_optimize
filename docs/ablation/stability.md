# 稳定性实验数据整合流程

> 本文档展示如何从最原始的“每次独立运行”数据出发，
> 逐步汇总到论文中最终呈现的稳定性实验汇总表格。
> 目标：同时量化 **综合目标函数 (F(S))**、**生态系统服务 (ES)** 和 **运行时间 Runtime** 的稳定性。
> 示例同时展示 SA 和 GA 算法。

***

## 1. 原始运行级数据（颗粒度：每次独立运行）

对 **每个代表性样本** × **每种算法**，分别独立运行 **5 次**（若资源紧张可改为 3\~4 次，但需统一声明）。
每次运行记录最终的目标函数值、ES 得分和总运行时间。

**表1：原始记录示例 —— 样本 S01，算法 SA**

| Sample\_ID | Algorithm | Run | F(S)  | ES    | Runtime (s) |
| ---------- | --------- | --- | ----- | ----- | ----------- |
| S01        | SA        | 1   | 0.876 | 0.913 | 512         |
| S01        | SA        | 2   | 0.869 | 0.908 | 520         |
| S01        | SA        | 3   | 0.873 | 0.911 | 505         |
| S01        | SA        | 4   | 0.871 | 0.910 | 518         |
| S01        | SA        | 5   | 0.880 | 0.915 | 509         |

| Sample\_ID | Algorithm | Run | F(S)  | ES    | Runtime (s) |
| ---------- | --------- | --- | ----- | ----- | ----------- |
| S01        | GA        | 1   | 0.865 | 0.904 | 1340        |
| S01        | GA        | 2   | 0.872 | 0.909 | 1310        |
| S01        | GA        | 3   | 0.858 | 0.901 | 1365        |
| S01        | GA        | 4   | 0.869 | 0.906 | 1328        |
| S01        | GA        | 5   | 0.861 | 0.902 | 1352        |

> 以上为单个样本的记录。完整实验需要对 **全部 10 个代表性样本** 完成相同的数据采集。
> 样本应按变化区域大小等级（如小、中、大）分层抽取，建议在原始表中附加一列 `Change_Level` 便于分组分析。

***

## 2. 第一步整合：按样本、算法计算均值和变异系数

对每个样本 × 算法组合，基于 5 次重复运行计算：

- **均值** (Mean)
- **标准差** (SD)
- **变异系数** (CV) = (SD / Mean) × 100%

分别对 (F(S))、`ES` 和 `Runtime` 执行上述计算。

**表2：样本 S01 的统计量（SA 和 GA）**

| Sample\_ID | Algorithm | Mean\_F | CV\_F (%) | Mean\_ES | CV\_ES (%) | Mean\_RT (s) | CV\_RT (%) |
| ---------- | --------- | ------- | --------- | -------- | ---------- | ------------ | ---------- |
| S01        | SA        | 0.8738  | 0.48      | 0.9114   | 0.30       | 512.8        | 1.20       |
| S01        | GA        | 0.8650  | 0.64      | 0.9044   | 0.36       | 1339.0       | 1.64       |

> 重复此操作，得到所有 10 个样本的均值和 CV。

***

## 3. 第二步整合：跨样本汇总稳定性指标

对每种算法（SA 和 GA），汇总 10 个样本的 CV 值，计算以下统计量：

- **平均 CV**：所有样本 CV 的算术平均
- **最大 CV**：所有样本 CV 中的最大值
- **CV ≤ 5% 样本占比**：CV 值不超过 5% 的样本数 ÷ 总样本数 × 100%

分别对 (CV_{F(S)})、(CV_{ES}) 和 (CV_{Runtime}) 汇总。

**表3：SA 与 GA 稳定性汇总（10 个样本，5 次独立运行）**

| Algorithm | Metric  | Mean CV (%) | Max CV (%) | CV ≤ 5% 样本占比 (%) |
| --------- | ------- | ----------- | ---------- | ---------------- |
| SA        | F(S)    | 0.52        | 0.75       | 100.0            |
| SA        | ES      | 0.31        | 0.48       | 100.0            |
| SA        | Runtime | 1.35        | 2.10       | 100.0            |
| GA        | F(S)    | 0.68        | 1.12       | 100.0            |
| GA        | ES      | 0.42        | 0.61       | 100.0            |
| GA        | Runtime | 1.82        | 3.45       | 90.0             |

> 上表中的数值为**示例占位**，实际应根据你的实验数据填入。
> Runtime 行可由 `show_runtime=False` 隐藏；ES 行同样可在不需要时手工删除。

***

## 4. 填入最终论文表格

最终论文表合并展示 **目标函数**、**解质量 (ES)** 与 **运行时间** 三类稳定性指标。
若篇幅紧张，可将 Runtime 行省略（对应 `show_runtime=False`）。

```latex
\begin{table}[htbp]
    \centering
    \caption{SA与GA稳定性汇总（10个代表性样本，基准情景，5次独立运行）}
    \label{tab:stability-summary}
    \begin{tabularx}{0.95\linewidth}{c c *{3}{>{\centering\arraybackslash}X}}
        \toprule
        算法 & 稳定性指标 & 平均 CV (\%) & 最大 CV (\%) & CV $\leq$ 5\% 样本占比 (\%) \\
        \midrule
        \multirow{3}{*}{SA}
        & $F(S)$        & 0.52 & 0.75 & 100.0 \\
        & $ES$          & 0.31 & 0.48 & 100.0 \\
        & Runtime       & 1.35 & 2.10 & 100.0 \\
        \cmidrule{2-5}
        \multirow{3}{*}{GA}
        & $F(S)$        & 0.68 & 1.12 & 100.0 \\
        & $ES$          & 0.42 & 0.61 & 100.0 \\
        & Runtime       & 1.82 & 3.45 & 90.0  \\
        \bottomrule
    \end{tabularx}
    \par\smallskip\small
    注：$CV = (\text{SD} / \text{Mean}) \times 100\%$；
    每个样本的CV基于该样本5次独立运行的最终值计算；
    平均CV和最大CV为10个样本的汇总值。
\end{table}
```

***

## 5. 图示所需数据

### 5.1 箱线图（推荐正文使用）

- **数据来源**：表1 原始运行数据。
- **绘制方式**：横轴为展示的指标（F(S) / ES / Runtime，按需开关 Runtime），纵轴为各指标的 N 个样本的 CV 分布。每个指标位置画两组箱线图（SA vs GA，不同颜色），并在箱旁叠加散点（5 次运行）。
- 该图直观展示解质量与运行成本的分布稳定性。
- 控制参数：`show_runtime=True` (默认) 含三组指标；`show_runtime=False` 则只画 F(S) 和 ES。

### 5.2 CV 点图/折线图（可作为附图）

- **数据来源**：表3 中的 `CV_F(S)`、`CV_ES` 与 `CV_Runtime` 列。
- **绘制方式**：横轴为样本（按等级排序，同一个层级背景颜色不同），纵轴为 CV (%)，用不同颜色的点区分 SA 和 GA（无连线），并添加 5% 的参考虚线。每个指标输出一张独立的图（`stability_cv_line_f.png` / `stability_cv_line_es.png` / `stability_cv_line_rt.png`）。
- 该图聚焦相对波动大小，便于快速判断所有样本的变异系数是否落在可接受范围内。

***

## 6. 可选扩展：分变化等级的子表

若想进一步说明稳定性与变化区域大小的关系，可将表3按 `Change_Level` 分组，计算各等级的统计量：

| Change\_Level | 样本数 | 平均 CV (SA) | 最大 CV (SA) | 平均 CV (GA) | 最大 CV (GA) |
| ------------- | --- | ---------- | ---------- | ---------- | ---------- |
| 低             | 4   | 0.42       | 0.55       | ...        | ...        |
| 中             | 3   | 0.61       | 0.78       | ...        | ...        |
| 高             | 3   | 0.48       | 0.62       | ...        | ...        |

该表可作为附录材料，强化“跨变化级别均稳定”的结论。

***

## 7. SA + GA 合并目录约定

当 `run_ablation_plots_pipeline` 同时传入 `experiment_root` (SA) 和 `sibling_experiment_roots` (GA) 时：

1. 先分别处理 SA 和 GA 各自的实验目录，在各自 `<root>/plots/` 下产出单算法的 `stability_runs.csv` / `stability_per_sample.csv` / `stability_summary.csv` / `dar_detail.csv` / `dar_matrix.csv` / `dar_summary.csv` 及对应图表。
2. 在 SA / GA 的 **同级目录** 新建合并文件夹 `<sa_timestamp>+<ga_timestamp>_combined/`，内容包含：
   - `plots/stability_runs.csv`：SA + GA 合并的原始记录
   - `plots/stability_per_sample.csv`：每 (sample × algorithm) 的 CV 统计 (含 F/ES/RT)
   - `plots/stability_summary.csv`：每 algorithm × metric 的跨样本汇总 (6 行)
   - `plots/stability_cv_line_f.png`、`stability_cv_line_es.png`、`stability_cv_line_rt.png`
   - `plots/stability_boxplot.png`
   - `combined_meta.json`：记录 `sa_root` / `sibling_roots` / `algorithms_present` / `generated_at`

`combined_meta.json` 用于明确指明哪一个时间戳目录对应 SA、哪一个对应 GA，避免后续追溯时混淆。

***

## 8. 维度达标率矩阵 (DAR)

> DAR 回答 CV 分析无法回答的问题："ES score 稳定的表象下，各维度的优化方向是否一致？"

### 8.1 问题背景

§1-§6 的稳定性分析回答了"优化器是否收敛到相近的综合目标值"——`F(S)` 和 `es_score`（来自 `convergence.json`）的 CV% 很低，说明同一样本 5 次独立运行产出相近的加权综合得分。

**但这不能排除维度间补偿效应**：优化器可能在不同 run 中以 water 超额收益补偿 food 的损失，维持加权和恒定。DAR 通过统计各维度 `best_plan > tgt_plan` 的频率，直接暴露这种结构性不一致。

### 8.2 数据来源

与 CV 分析不同，DAR 读取 `performance.json` 中的 **ES 绝对原始值**（而非 convergence.json 中的加权标量）：

| 字段 | 含义 | 示例 (food, 10_1417_5) |
|------|------|----------------------|
| `ES.{d}.best_plan` | 该 run 产出方案在维度 d 的 ES 原始值 | 17,867,210 |
| `ES.{d}.tgt_plan` | 目标方案在维度 d 的 ES 值（跨 run 恒定）| 17,825,510 |

### 8.3 计算流程

对每个样本 $s$ 和维度 $d \in \{\text{food}, \text{carbon}, \text{water}, \text{culture}\}$：

1. **逐 Run 判断**：$\mathbb{1}_{r,d} = \mathbf{1}[\text{best\_plan}_{r,d} > \text{tgt\_plan}_d]$
2. **单样本达标率**：$\text{DAR}_d^{(s)} = \frac{1}{R} \sum_{r=1}^{R} \mathbb{1}_{r,d}$，$R=5$
3. **跨样本汇总**：对每个维度计算均值、标准差、最小/最大值

DAR 取值范围 $[0, 1]$，步长 $1/R = 0.2$：
- **DAR = 1.0**：该维度在所有 5 次 run 中均优于 target（方向完全确定）
- **DAR = 0.0**：该维度在所有 5 次 run 中均未达标（方向完全确定）
- **DAR ≈ 0.5**：方向不确定，存在反转

### 8.4 输出文件

| 文件 | 颗粒度 | 关键列 |
|------|--------|--------|
| `dar_detail.csv` | 样本 × Run | `sample_id, run, food_beat, carbon_beat, water_beat, culture_beat` |
| `dar_matrix.csv` | 样本 | `sample_id, n_pred, level, n_runs, food_dar, carbon_dar, water_dar, culture_dar` |
| `dar_summary.csv` | 维度 | `dimension, mean_dar, std_dar, min_dar, max_dar, qualitative_pattern` |

**定性模式分类**（基于跨样本 DAR 均值）：
- DAR ≥ 0.8 → **稳定达标型**
- DAR ≤ 0.2 → **系统未达标型**
- 其余 → **方向不确定型**

### 8.5 实现入口

```python
from utils.exp_sweep_utils import build_dar_matrix_csv, build_dar_summary_csv

# 对 baseline 实验目录计算 DAR
matrix = build_dar_matrix_csv(
    'results/ablation/<ts>/A_guided',
    rep_json_path='results/ablation/<ts>/representative_samples.json',  # 可选
    run_start=0, run_end=4,
)
build_dar_summary_csv(matrix)

# 或使用便捷函数一键生成三张表
from utils.exp_sweep_utils import build_dar_all
build_dar_all('results/ablation/<ts>/A_guided', run_start=0, run_end=4)
```

后处理 pipeline `run_all_ablation_plots()` 在 stability 模式下自动调用 DAR，输出到 `<experiment_root>/plots/dar_*.csv`。

### 8.6 DAR 与 CV 的分工

| 问题 | 适用指标 | 数据来源 |
|------|---------|---------|
| 综合目标值是否稳定收敛？ | $F(S)$ 的 CV, ES score 的 CV | `convergence.json` |
| 各维度方向是否一致？ | DAR（达标率） | `performance.json` ES 绝对原始值 |
| 同维度波动幅度多大？ | ES 绝对值的 CV（同维度内） | `performance.json` ES 绝对原始值 |

> **注意**：DAR 仅对 baseline（A_guided，完整方法）计算。B_unguided / C_random 作为 ablation 对照，其优化方向不确定属于预期行为。

***

## 附录：数据流总结

| 处理阶段     | 输入     | 输出                   | 颗粒度       |
| -------- | ------ | -------------------- | --------- |
| 1. 原始记录  | 每次运行输出 | `stability_runs.csv` | 样本 × Run  |
| 2. 样本统计量 | 表1     | `stability_per_sample.csv` (F/ES/RT 三组 Mean,SD,CV) | 样本 × 算法 |
| 3. 跨样本汇总 | 表2     | `stability_summary.csv` (平均 CV, 最大 CV, CV≤5% 占比) | 算法 × 指标 |
| 4. SA+GA 合并 | SA、GA 的 `stability_runs.csv` | 合并目录下的 CSV + 图表 + `combined_meta.json` | 算法对比 |
| 5. 图表    | 表1, 表2 | 箱线图 / CV 折线图          | （可视化用）    |
| 6. DAR 逐 Run 判断 | `performance.json` (ES 绝对原始值) | `dar_detail.csv` (best_plan > tgt_plan 的 0/1 判断) | 样本 × Run |
| 7. DAR 矩阵 | `dar_detail.csv` | `dar_matrix.csv` (每样本每维度达标率, 取值 0~1) | 样本 |
| 8. DAR 汇总 | `dar_matrix.csv` | `dar_summary.csv` (跨样本 DAR 均值/标准差/定性模式) | 维度 |

> 实现入口：`pipelines/ablation_plots.py::run_all_ablation_plots`，
> 通过 `mode='stability' / 'ablation' / 'auto'` 选择产出范围，
> 通过 `show_runtime` 控制 Runtime 维度是否参与画图。
