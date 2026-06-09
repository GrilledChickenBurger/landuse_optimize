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

1. 先分别处理 SA 和 GA 各自的实验目录，在各自 `<root>/plots/` 下产出单算法的 `stability_runs.csv` / `stability_per_sample.csv` / `stability_summary.csv` 及对应图表。
2. 在 SA / GA 的 **同级目录** 新建合并文件夹 `<sa_timestamp>+<ga_timestamp>_combined/`，内容包含：
   - `plots/stability_runs.csv`：SA + GA 合并的原始记录
   - `plots/stability_per_sample.csv`：每 (sample × algorithm) 的 CV 统计 (含 F/ES/RT)
   - `plots/stability_summary.csv`：每 algorithm × metric 的跨样本汇总 (6 行)
   - `plots/stability_cv_line_f.png`、`stability_cv_line_es.png`、`stability_cv_line_rt.png`
   - `plots/stability_boxplot.png`
   - `combined_meta.json`：记录 `sa_root` / `sibling_roots` / `algorithms_present` / `generated_at`

`combined_meta.json` 用于明确指明哪一个时间戳目录对应 SA、哪一个对应 GA，避免后续追溯时混淆。

***

## 附录：数据流总结

| 处理阶段     | 输入     | 输出                   | 颗粒度       |
| -------- | ------ | -------------------- | --------- |
| 1. 原始记录  | 每次运行输出 | `stability_runs.csv` | 样本 × Run  |
| 2. 样本统计量 | 表1     | `stability_per_sample.csv` (F/ES/RT 三组 Mean,SD,CV) | 样本 × 算法 |
| 3. 跨样本汇总 | 表2     | `stability_summary.csv` (平均 CV, 最大 CV, CV≤5% 占比) | 算法 × 指标 |
| 4. SA+GA 合并 | SA、GA 的 `stability_runs.csv` | 合并目录下的 CSV + 图表 + `combined_meta.json` | 算法对比 |
| 5. 图表    | 表1, 表2 | 箱线图 / CV 折线图          | （可视化用）    |

> 实现入口：`pipelines/ablation_plots.py::run_all_ablation_plots`，
> 通过 `mode='stability' / 'ablation' / 'auto'` 选择产出范围，
> 通过 `show_runtime` 控制 Runtime 维度是否参与画图。
