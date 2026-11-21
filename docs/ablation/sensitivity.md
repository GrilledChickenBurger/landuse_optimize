# Sensitivity 一维线搜索数据整合流程

> 本文档展示如何从 tornado 粗筛出的 Top-K 敏感参数出发，
> 对每个参数做细粒度取值扫描（7~9 个取值），逐步汇总到一维线图（参数-指标趋势）和最优取值区间汇总表。
> 实验目的：确定每个敏感参数的最优取值区间，为最终参数组合提供依据。
> 示例仅展示 **SA 算法**，GA 流程完全对称，仅需替换算法标签。

---

## 1. 原始运行级数据（颗粒度：每次独立运行）

对 **每个代表性样本** × **每个参数取值**（Top-K 参数 × 7~9 取值），分别独立运行 **3 次**（比 tornado 的 2 次多，因需要可靠的 ±1 std 误差条）。

每次运行记录收敛统计（`convergence.json`）和 ES 分项指标（`performance.json`）。

**表1：原始记录示例 —— 样本 10_479_5，参数 lambda_ES_high 各取值，SA 算法**

| Sample_ID | Experiment | Run | final_best | iter_to_95pct | wall_time_sec | ES.food.best_ori_diff | ES.carbon.best_ori_diff | ES.water.best_ori_diff | ES.culture.best_ori_diff |
| --------- | ---------- | --- | ---------- | ------------- | ------------- | --------------------- | ----------------------- | ---------------------- | ------------------------ |
| 10_479_5  | S_lambda_ES_high=200  | 1   | 0.852      | 160           | 580           | 0.031                 | -0.012                  | 0.022                  | -0.003                   |
| 10_479_5  | S_lambda_ES_high=200  | 2   | 0.848      | 168           | 575           | 0.029                 | -0.014                  | 0.020                  | -0.004                   |
| 10_479_5  | S_lambda_ES_high=200  | 3   | 0.855      | 155           | 582           | 0.032                 | -0.011                  | 0.023                  | -0.002                   |
| 10_479_5  | S_lambda_ES_high=400  | 1   | 0.862      | 142           | 545           | 0.036                 | -0.016                  | 0.026                  | -0.005                   |
| 10_479_5  | S_lambda_ES_high=400  | 2   | 0.858      | 148           | 550           | 0.034                 | -0.015                  | 0.024                  | -0.004                   |
| 10_479_5  | S_lambda_ES_high=400  | 3   | 0.863      | 139           | 540           | 0.037                 | -0.017                  | 0.027                  | -0.006                   |
| ...       | ...        | ... | ...        | ...           | ...           | ...                   | ...                     | ...                    | ...                      |

> sensitivity 模式下每个参数 7~9 个取值。以 2 个 Top-K 参数、各 9 个取值、3 个样本、3 次重复为例，共约 162 条原始运行记录。
> 实验命名规则：`S_<参数路径>=<取值>`（如 `S_cooling_rate=0.9993`），所有实验 group='S'。
> Top-K 参数由 tornado 粗筛结果确定，参数规格见 `main.py` 中 `run_sensitivity_pipeline()` 的 `top_k_params` 定义。

---

## 2. 第一步整合：按样本、参数取值取均值

对每个样本 × 每个 (参数路径, 参数取值) 组合，计算 3 次独立运行的算术均值，
作为该参数取值在该样本上的代表值，以消除运行随机性。

**表2：样本 10_479_5，参数 lambda_ES_high 各取值的指标均值（SA）**

| Sample_ID | Param_Path                  | Param_Value | Mean_final_best | Mean_T95 | Mean_Runtime | Mean_ES_carbon |
| --------- | --------------------------- | ----------- | --------------- | -------- | ------------ | -------------- |
| 10_479_5  | tgt_function.lambda_ES_high | 200         | 0.8517          | 161.0    | 579.0        | −0.0123        |
| 10_479_5  | tgt_function.lambda_ES_high | 400         | 0.8610          | 143.0    | 545.0        | −0.0160        |
| 10_479_5  | tgt_function.lambda_ES_high | 600         | 0.8685          | 132.0    | 528.0        | −0.0182        |
| 10_479_5  | tgt_function.lambda_ES_high | 800         | 0.8730          | 126.0    | 519.0        | −0.0198        |
| 10_479_5  | tgt_function.lambda_ES_high | 1000        | 0.8760          | 121.0    | 513.0        | −0.0205        |
| 10_479_5  | tgt_function.lambda_ES_high | 1200        | 0.8775          | 118.0    | 507.0        | −0.0212        |
| 10_479_5  | tgt_function.lambda_ES_high | 1500        | 0.8780          | 116.0    | 503.0        | −0.0218        |
| 10_479_5  | tgt_function.lambda_ES_high | 1800        | 0.8765          | 115.0    | 501.0        | −0.0220        |
| 10_479_5  | tgt_function.lambda_ES_high | 2000        | 0.8730          | 114.0    | 498.0        | −0.0225        |

> 对全部 3 个样本重复此操作，每个参数得到 3 × 9 = 27 行类似数据。此表对应 `ExperimentResultStore.get_sensitivity_data()` 内部 `sample_means` 列表的中间结构。

---

## 3. 第二步整合：跨样本均值与标准差（一维线数据）

对上一步每个参数的每个取值，跨样本（N 个样本的均值得出）计算描述性统计量：

- **跨样本均值**：

\[
\mu_{param, value} = \frac{1}{N_{samples}} \sum_{i=1}^{N} mean_{metric}(sample_i, param, value)
\]

- **跨样本标准差**：

\[
\sigma_{param, value} = \sqrt{\frac{1}{N_{samples}-1} \sum_{i=1}^{N} \left(mean_{metric}(sample_i, param, value) - \mu_{param, value}\right)^2}
\]

- **变异系数**（可选，用于判断样本间一致性）：

\[
CV_{param, value} = \frac{\sigma_{param, value}}{|\mu_{param, value}|} \times 100\%
\]

**表3：参数 lambda_ES_high 的跨样本统计量（metric=final_best, SA，3 个样本）**

| Param_Path                  | Param_Value | Cross-Sample Mean (μ) | Cross-Sample Std (σ) | μ − σ   | μ + σ   | CV (%) |
| --------------------------- | ----------- | --------------------- | -------------------- | ------- | ------- | ------ |
| tgt_function.lambda_ES_high | 200         | 0.8520                | 0.0052               | 0.8468  | 0.8572  | 0.61   |
| tgt_function.lambda_ES_high | 400         | 0.8615                | 0.0048               | 0.8567  | 0.8663  | 0.56   |
| tgt_function.lambda_ES_high | 600         | 0.8690                | 0.0041               | 0.8649  | 0.8731  | 0.47   |
| tgt_function.lambda_ES_high | 800         | 0.8735                | 0.0035               | 0.8700  | 0.8770  | 0.40   |
| tgt_function.lambda_ES_high | 1000        | 0.8760                | 0.0028               | 0.8732  | 0.8788  | 0.32   |
| tgt_function.lambda_ES_high | 1200        | 0.8775                | 0.0031               | 0.8744  | 0.8806  | 0.35   |
| tgt_function.lambda_ES_high | 1500        | 0.8780                | 0.0038               | 0.8742  | 0.8818  | 0.43   |
| tgt_function.lambda_ES_high | 1800        | 0.8765                | 0.0045               | 0.8720  | 0.8810  | 0.51   |
| tgt_function.lambda_ES_high | 2000        | 0.8730                | 0.0058               | 0.8672  | 0.8788  | 0.66   |

> 此表由 `ExperimentResultStore.get_sensitivity_data()` 返回，value 为 `{value: [sample_means]}` 列表（每个元素是一个样本的 run 内均值）。绘图时 `np.mean()` 和 `np.std(ddof=1)` 即从该列表计算。
> 对每个 Top-K 参数和每个指标均生成一份类似的趋势表。误差条使用 μ ± σ 区间。

---

## 4. 最终整合：最优取值区间判定

对每个参数的 (metric, value) 趋势数据，按曲线形态判定最优取值区间：

- **峰值法（final_best / ES.* 类指标——越大越好）**：取 μ 最大值附近的连续取值区间（如 μ 在最高峰 1% 以内的所有取值）。若最大值 1% 范围内仅一个取值点，则将相邻取值一并纳入
- **谷值法（iter_to_95pct / wall_time_sec 类指标——越小越好）**：取 μ 最小值附近的连续取值区间
- **平坦判定（不敏感参数）**：若所有取值的 (μ_max − μ_min) / |μ_median| < 阈值（建议 0.01），则该参数在该 metric 上不敏感，可固定为默认值（baseline）

**表4：各参数最优取值区间汇总（SA，metric=final_best）**

| 参数                      | 扫描范围          | 最优取值区间  | 区间内 μ 范围     | 最优取值（峰值处） | 趋势类型     | 建议                                   |
| ------------------------- | ----------------- | ------------- | ----------------- | ------------------ | ------------ | -------------------------------------- |
| tgt_function.lambda_ES_high | 200 – 2000      | 1000 – 1500   | 0.8760 – 0.8780   | 1500               | 先升后平     | 取值 1000–1500 已近饱和，>1500 收益递减 |
| cooling_rate              | 0.9985 – 0.9999  | 0.9993 – 0.9997 | 0.8745 – 0.8770   | 0.9996             | 倒 U 型       | 峰值在 0.9996 附近，避免 >0.9998       |

> 此表为论文中"参数调优建议"章节的数据源。各 metric 独立判定——若 final_best 和 T95 推荐的最优区间不一致，需综合权衡（见第 7 节）。

---

## 5. 填入最终论文表格

Sensitivity 实验的最终产出是参数最优取值建议表：

```latex
\begin{table}[htbp]
    \centering
    \caption{敏感参数一维线搜索最优取值区间（3个代表性样本，SA算法）}
    \label{tab:sensitivity-optimal}
    \begin{tabular}{c l c c c}
        \toprule
        排名 & 参数 & 扫描范围 & 最优区间 & 趋势描述 \\
        \midrule
        1 & $\lambda_{\text{ES\_high}}$  & [200, 2000]   & [1000, 1500] & 先升后平，$>1000$ 收益递减 \\
        2 & cooling\_rate                 & [0.9985, 0.9999] & [0.9993, 0.9997] & 倒U型，峰值 0.9996 \\
        \bottomrule
    \end{tabular}
    \par\smallskip\small
    注：最优区间定义为峰值 $\pm$ 1\% 内的连续取值；
    若曲线平坦（全范围变化 $<$ 1\%）则标注"不敏感"，建议固定为默认值。
\end{table}
```

> 若扫描参数超过 2 个，每参数一行；若指标间推荐区间冲突，另附"跨指标权衡表"（见第 7 节表5）。

---

## 6. 图示所需数据

### 6.1 一维线图（核心图表）

- **数据来源**：**表3**（跨样本 μ ± σ）
- **横轴**：参数取值（连续数值，从小到大排列）
- **纵轴**：指标跨样本均值 μ（最终目标值 / 收敛速度 / ES 提升等）
- **误差条**：跨样本标准差 σ（`yerr = stds`），使用 `ddof=1`
- **数据点**：`marker='o'` 标注每个扫描取值
- **连线**：实线连接各取值点，展示整体趋势
- **子图布局**：Top-K 个参数，1 行 N 列（N ≤ 3）或 2 行 M 列（N > 3）
- **解读要素**：
  - 曲线**平坦** → 参数不敏感，可放宽或固定为默认值
  - 曲线有**明显峰值/谷值** → 最优取值在极值点附近
  - 曲线**单调** → 参数有方向性影响，需结合实际意义选择
- **输出文件**：`plots/sensitivity_final_best.png`、`plots/sensitivity_iter_to_95pct.png`、`plots/sensitivity_wall_time_sec.png`、`plots/sensitivity_ES.{food,carbon,water,culture}.best_ori_diff.png`

### 6.2 对比参考线（建议添加）

- 在每张线图上叠加 **tornado baseline 值**作为水平参考虚线，便于对比细扫与粗筛的一致性
- 标注 tornado 扰动值位置（竖虚线），展示粗筛取值与细扫取值的覆盖关系

### 6.3 多参数联合响应面图（可选，附录）

若 Top-K=2 且需考察参数交互效应：
- 横轴 = 参数 1 取值，纵轴 = 参数 2 取值，颜色 = metric 值
- **注意**：两个独立的 S_* 一维扫描实验无法直接获得联合数据，需额外做 2D grid search（不在当前 pipeline 范围内）
- 若不追加联合实验，应在论文中注明"当前仅支持独立的一维扫描，参数间交互效应未量化"

---

## 7. 可选扩展：跨指标最优区间冲突分析

当同一参数在 final_best 和 T95 / Runtime 上的最优区间不一致时，需要记录各指标的推荐值并做综合权衡。

**表5：lambda_ES_high 跨指标最优取值对比（SA，示例）**

| Metric        | 最优取值 | 该取值下 μ  | 扫描范围 μ 范围    | 趋势          | 业务含义                         |
| ------------- | -------- | ----------- | ------------------ | ------------- | -------------------------------- |
| final_best    | 1500     | 0.8780      | 0.8520 – 0.8780    | 先升后平      | 目标函数在 1500 饱和             |
| iter_to_95pct | 2000     | 114.0       | 161.0 – 114.0      | 单调递减      | 取值越大收敛越快                 |
| wall_time_sec | 2000     | 498.0       | 582.0 – 498.0      | 单调递减      | 取值越大运行越快（因收敛快早停）  |
| ES.carbon     | 2000     | −0.0225     | −0.0123 – −0.0225  | 单调递减      | 取值越大碳汇提升越多             |

> 此表揭示：final_best 在 1500 附近饱和，但收敛速度和 ES 指标在更高取值仍持续改善。
> 最终论文推荐值需结合多指标业务权重综合权衡，或在论文中分指标给出推荐区间，供不同应用场景选择。

---

## 附录：数据流总结

| 处理阶段        | 输入         | 输出                                                   | 颗粒度              |
| --------------- | ------------ | ------------------------------------------------------ | ------------------- |
| 1. 原始记录     | 每次运行输出 | `sensitivity_runs.csv`（运行级长表）                   | 样本 × 实验 × Run   |
| 2. 样本-取值均值 | 表1         | `sensitivity_sample_mean_table.csv`（`aggregate_runs()` 产生） | 样本 × 参数 × 取值  |
| 3. 跨样本统计量 | 表2          | `sensitivity_trend_table.csv`（`get_sensitivity_data()` + `_calc_metric_stats()`） | metric × 参数 × 取值 |
| 4. 最优区间判定 | 表3          | `sensitivity_optimal_interval_table.csv`                | metric × 参数       |
| 5. 跨指标权衡   | 表4          | `sensitivity_metric_conflict_table.csv`                 | 参数 × metric       |
| 6. 图表         | 表3          | `sensitivity_<metric>.png`                              | （可视化用）        |

> 核心函数链：`aggregate_runs()` → `ExperimentResultStore.get_sensitivity_data()` → `_calc_metric_stats()` → `build_sensitivity_*_table()` → `plot_sensitivity_lines()`。
> `get_sensitivity_data()` 返回结构 `{param_path: {value: [sample_means]}}`，其中 `[sample_means]` 是各样本内 run 均值的列表，长度 = 样本数。
> 当前 pipeline 会落盘表1~表5，并为 `final_best`、T95、Runtime、四项 ES 指标生成一维线图。
