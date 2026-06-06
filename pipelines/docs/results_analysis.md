# Results Analysis Pipeline 数据流说明

> 本文档覆盖 `pipelines/results_analysis.py` 的完整操作流程。该文件不是优化计算入口，而是 Stage 2 结果后处理入口：读取 `results/` 中 SA/GA/实验结果，计算统计指标，输出图表到 `figures/`。

---

## 一、文件定位

`results_analysis.py` 主要封装 `utils/analysis_plot_utils.py` 中的绘图和统计函数，提供更高层的 pipeline 调用接口。

```python
from utils.analysis_plot_utils import (
    plot_single_experiment_lu_bars,
    plot_single_experiment_lu_distribution,
    plot_single_experiment_es_bars,
    plot_single_experiment_es_distribution,
    plot_origin_target_lu_stackbar,
    visualize_picked_samples,
    visualize_all_samples,
    compare_sample_images,
    plot_sa_ga_diff_bars,
    plot_sa_ga_boxplots,
    export_sa_ga_combined_area_data,
    plot_model_comparison_bar,
)
```

核心用途分四类：

| 类别 | 入口函数 | 用途 |
|------|----------|------|
| 单实验统计图 | `single_experiment_stat_pipeline()` | 对一个实验结果目录绘制 LU/ES 分布图 |
| 样本可视化 | `sample_visualization_pipeline()` | 可视化优秀样本、全部样本、SA-GA 对比样本图 |
| SA-GA 对比 | `comparison_pipeline()` | 对比两个实验目录（通常一个 SA，一个 GA） |
| 模型指标对比 | `model_comparison_pipeline()` | 根据 JSON 绘制变化检测模型指标对比柱状图 |

---

## 二、关键路径解析

该文件所有入口都接受 `expr_path`，通过两个内部函数解析为数据目录和图像输出目录。

### 2.1 `_resolve_result_folder(expr_path)`

```python
def _resolve_result_folder(expr_path: str):
    norm_path = os.path.normpath(expr_path)
    if os.path.isabs(norm_path) or norm_path.split(os.sep)[0] == "results":
        return expr_path
    return os.path.join("results", expr_path)
```

| 输入示例 | 输出 |
|----------|------|
| `"SA/20260601_120000/default"` | `"results/SA/20260601_120000/default"` |
| `"results/SA/20260601_120000/default"` | 原样返回 |
| 绝对路径 `D:\...\results\SA\...` | 原样返回 |

### 2.2 `_resolve_figure_folder(expr_path)`

```python
def _resolve_figure_folder(expr_path: str):
    norm_path = os.path.normpath(expr_path)
    if os.path.isabs(norm_path):
        return os.path.join("figures", os.path.basename(os.path.dirname(norm_path)), os.path.basename(norm_path))
    parts = norm_path.split(os.sep)
    if parts[0] == "results":
        parts = parts[1:]
    return os.path.join("figures", *parts)
```

| 输入示例 | 输出 |
|----------|------|
| `"SA/20260601_120000/default"` | `"figures/SA/20260601_120000/default"` |
| `"results/SA/20260601_120000/default"` | `"figures/SA/20260601_120000/default"` |
| 绝对路径 `D:\...\results\SA\20260601_120000\default` | `"figures/20260601_120000/default"` |

---

## 三、单实验统计：`single_experiment_stat_pipeline()`

### 3.1 函数签名

```python
def single_experiment_stat_pipeline(expr_path: str,
                                    plot_types: Iterable[str] = None,
                                    comparison_type: str = "target"):
```

### 3.2 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `expr_path` | 必填 | 一个实验结果目录，如 `"SA/20260601_120000/default"` |
| `plot_types` | `None` → `{"lu_box", "es_box", "origin_target_lu"}` | 要绘制的图类型集合 |
| `comparison_type` | `"target"` | 变化比例的参照对象，常用 `"target"` 或 `"origin"` |

### 3.3 数据流

```
single_experiment_stat_pipeline(expr_path)
  ├─ data_folder = _resolve_result_folder(expr_path)
  │     → results/{SA|GA}/{timestamp}/{scenario}/
  │
  ├─ figure_folder = _resolve_figure_folder(expr_path)
  │     → figures/{SA|GA}/{timestamp}/{scenario}/
  │
  ├─ area_save_folder = figure_folder/area
  ├─ es_save_folder   = figure_folder/es
  │
  └─ 根据 plot_types 调用不同绘图函数
        ├─ lu_bars          → plot_single_experiment_lu_bars()
        ├─ lu_box           → plot_single_experiment_lu_distribution(plot_type="box")
        ├─ lu_hist          → plot_single_experiment_lu_distribution(plot_type="hist")
        ├─ es_bars          → plot_single_experiment_es_bars()
        ├─ es_box           → plot_single_experiment_es_distribution(plot_type="box")
        ├─ es_hist          → plot_single_experiment_es_distribution(plot_type="hist")
        └─ origin_target_lu → plot_origin_target_lu_stackbar()
```

### 3.4 `plot_types` 可选值

| plot_type | 调用函数 | 输出内容 |
|-----------|----------|----------|
| `lu_bars` | `plot_single_experiment_lu_bars()` | 各情景土地利用总面积堆叠图 + 分类型柱状图 |
| `lu_box` | `plot_single_experiment_lu_distribution(..., plot_type="box")` | 样本级 LU 面积变化比例箱线图 |
| `lu_hist` | `plot_single_experiment_lu_distribution(..., plot_type="hist")` | 样本级 LU 面积变化比例直方图 |
| `es_bars` | `plot_single_experiment_es_bars()` | 各情景 ES 总价值柱状图 |
| `es_box` | `plot_single_experiment_es_distribution(..., plot_type="box")` | 样本级 ES 变化比例箱线图 |
| `es_hist` | `plot_single_experiment_es_distribution(..., plot_type="hist")` | 样本级 ES 变化比例直方图 |
| `origin_target_lu` | `plot_origin_target_lu_stackbar()` | 原始 LU 与目标 LU 总面积占比对比图 |

### 3.5 默认行为

如果不传 `plot_types`：

```python
plot_types = {"lu_box", "es_box", "origin_target_lu"}
```

即默认只画：
- LU 面积变化比例箱线图
- ES 变化比例箱线图
- 原始与目标 LU 堆叠柱状图

---

## 四、样本可视化：`sample_visualization_pipeline()`

### 4.1 函数签名

```python
def sample_visualization_pipeline(expr_path: str,
                                  mode: str = "picked",
                                  compare_expr_path: str = None,
                                  **kwargs):
```

### 4.2 mode 分支

| mode | 行为 | 返回值 |
|------|------|--------|
| `"picked"` | 根据性能指标筛选优秀样本并可视化 | `best_samples` |
| `"all"` | 可视化该实验目录下所有样本 | `indices` |
| `"sa_ga_pair"` | 将两个实验的样本图并排对比 | `None` |

### 4.3 mode='picked'

```python
visualize_picked_samples(
    data_folder=data_folder,
    save_folder=figure_folder/samples_new,
    min_improvement_threshold=0.0,
    min_positive_scenarios=2,
    min_other_threshold=-0.05,
    layout=(1, 6),
)
```

筛选逻辑由 `utils.analysis_utils.get_best_samples()` 完成，核心含义：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `min_improvement_threshold` | `0.0` | 至少达到该提升阈值才算正向改善 |
| `min_positive_scenarios` | `2` | 至少多少个 scenario 中表现为正向改善 |
| `min_other_threshold` | `-0.05` | 其他 scenario 允许的最低表现阈值 |
| `layout` | `(1, 6)` | 输出样本图布局 |

数据流：

```
visualize_picked_samples()
  ├─ get_performance_metrics(data_folder)
  │     → 读取各样本 performance.json
  ├─ get_best_samples(metrics, thresholds)
  │     → 筛选代表性优秀样本
  └─ visualize_best_samples(best_samples, data_folder, save_folder)
        → 输出样本空间图
```

### 4.4 mode='all'

```python
visualize_all_samples(
    data_folder=data_folder,
    save_folder=figure_folder/all_samples,
    layout=(1, 6),
)
```

数据流：

```
visualize_all_samples()
  ├─ list_sample_indices(data_folder)
  └─ visualize_best_samples(indices, data_folder, save_folder)
```

虽然函数名中是 `visualize_best_samples`，但这里传入的是全部样本 index。

### 4.5 mode='sa_ga_pair'

该模式用于对比两个实验目录中已经生成的 `all_samples` 图片。

```python
sample_visualization_pipeline(
    expr_path="SA/20260601_120000/default",
    mode="sa_ga_pair",
    compare_expr_path="GA/20260601_130000/default",
    desc1="SA优化结果",
    desc2="GA优化结果",
    save_folder="figures/comparison_results",
)
```

内部路径：

```python
img_folder1 = _resolve_figure_folder(expr_path) / "all_samples"
img_folder2 = _resolve_figure_folder(compare_expr_path) / "all_samples"
compare_sample_images(img_folder1, img_folder2, desc1, desc2, save_folder)
```

注意：使用该模式前，通常需要先分别对 SA 和 GA 跑过：

```python
sample_visualization_pipeline(sa_expr_path, mode="all")
sample_visualization_pipeline(ga_expr_path, mode="all")
```

---

## 五、SA-GA 对比：`comparison_pipeline()`

### 5.1 函数签名

```python
def comparison_pipeline(sa_expr_path: str,
                        ga_expr_path: str,
                        plot_types: Iterable[str] = None,
                        save_folder: str = None,
                        comparison_type: str = "target"):
```

### 5.2 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `sa_expr_path` | 必填 | SA 实验结果目录 |
| `ga_expr_path` | 必填 | GA 实验结果目录 |
| `plot_types` | `None` → `{"lu_diff_bars", "lu_box", "es_box"}` | 要绘制/导出的对比类型 |
| `save_folder` | `None` → `"figures/comparison_results"` | 对比图输出目录 |
| `comparison_type` | `"target"` | 箱线图参照对象，常用 `"target"` 或 `"origin"` |

### 5.3 数据流

```
comparison_pipeline(sa_expr_path, ga_expr_path)
  ├─ sa_folder = _resolve_result_folder(sa_expr_path)
  ├─ ga_folder = _resolve_result_folder(ga_expr_path)
  ├─ save_folder = figures/comparison_results (默认)
  │
  └─ 根据 plot_types 执行：
        ├─ lu_diff_bars
        │   └─ plot_sa_ga_diff_bars(sa_folder, ga_folder)
        │       ├─ build_sa_ga_total_area_data()
        │       ├─ cal_scenario_diff_ratio()
        │       └─ 输出 SA/GA 土地利用面积差异柱状图
        │
        ├─ lu_box
        │   └─ plot_sa_ga_boxplots(data_type="LU")
        │       ├─ cal_sample_area_change_ratio()
        │       └─ 输出 LU 样本级变化比例箱线图
        │
        ├─ es_box
        │   └─ plot_sa_ga_boxplots(data_type="ES")
        │       ├─ cal_sample_es_change_ratio()
        │       └─ 输出 ES 样本级变化比例箱线图
        │
        ├─ combined_area_json
        │   └─ export_sa_ga_combined_area_data()
        │       └─ 输出 stacked_bar_data.json
```

### 5.4 `plot_types` 可选值

| plot_type | 调用函数 | 输出 |
|-----------|----------|------|
| `lu_diff_bars` | `plot_sa_ga_diff_bars()` | SA vs GA 土地利用面积差异柱状图 |
| `lu_box` | `plot_sa_ga_boxplots(data_type="LU")` | SA vs GA 的 LU 变化比例箱线图 |
| `es_box` | `plot_sa_ga_boxplots(data_type="ES")` | SA vs GA 的 ES 变化比例箱线图 |
| `combined_area_json` | `export_sa_ga_combined_area_data()` | `stacked_bar_data.json` |

### 5.5 默认行为

如果不传 `plot_types`：

```python
plot_types = {"lu_diff_bars", "lu_box", "es_box"}
```

即默认输出：
- SA-GA LU 面积差异柱状图
- SA-GA LU 样本级箱线图
- SA-GA ES 样本级箱线图

---

## 六、模型指标对比：`model_comparison_pipeline()`

### 6.1 函数签名

```python
def model_comparison_pipeline(json_path: str = "model_comparison_data.json",
                              save_dir: str = "figures",
                              save_name: str = "model_comparison_bar.png"):
```

### 6.2 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `json_path` | `"model_comparison_data.json"` | 模型对比数据 JSON 路径 |
| `save_dir` | `"figures"` | 图像保存目录 |
| `save_name` | `"model_comparison_bar.png"` | 输出图片文件名 |

### 6.3 数据格式

JSON 需包含 `models` 和 `metrics` 两个字段：

```json
{
  "models": ["AttU-Net", "U-Net", "FCN", "SegFormer", "TransUNet"],
  "metrics": {
    "OA": [0.95, 0.93, 0.91, 0.94, 0.92],
    "Precision": [0.87, 0.85, 0.83, 0.86, 0.84],
    "Recall": [0.82, 0.80, 0.78, 0.81, 0.79],
    "F1": [0.84, 0.82, 0.80, 0.83, 0.81],
    "IoU": [0.73, 0.71, 0.69, 0.72, 0.70]
  }
}
```

### 6.4 输出

- `{save_dir}/{save_name}`：模型指标对比分组柱状图

---

## 七、快捷封装函数

`results_analysis.py` 还提供了几个轻量封装，便于在 `main.py` 或交互式脚本中调用。

| 函数 | 等价调用 | 用途 |
|------|----------|------|
| `complete_pipeline(expr_path)` | `single_experiment_stat_pipeline(expr_path)` | 跑默认单实验统计图 |
| `picker(expr_path, ...)` | `sample_visualization_pipeline(expr_path, mode="picked", ...)` | 筛选并可视化优秀样本 |
| `all_samples(expr_path)` | `sample_visualization_pipeline(expr_path, mode="all")` | 可视化全部样本 |
| `show_SA_GA(expr_path1, expr_path2)` | `sample_visualization_pipeline(expr_path1, mode="sa_ga_pair", compare_expr_path=expr_path2)` | 并排展示 SA/GA 样本图 |

---

## 八、典型调用示例

### 8.1 对单个实验结果画默认统计图

```python
from pipelines.results_analysis import single_experiment_stat_pipeline

single_experiment_stat_pipeline(
    expr_path="SA/20260601_120000/default",
)
```

输入目录：

```text
results/SA/20260601_120000/default/
```

输出目录：

```text
figures/SA/20260601_120000/default/
```

---

### 8.2 对单个实验画全部统计图

```python
single_experiment_stat_pipeline(
    expr_path="GA/20260601_130000/default",
    plot_types={
        "lu_bars",
        "lu_box",
        "lu_hist",
        "es_bars",
        "es_box",
        "es_hist",
        "origin_target_lu",
    },
    comparison_type="target",
)
```

---

### 8.3 可视化优秀样本

```python
from pipelines.results_analysis import picker

best_samples = picker(
    expr_path="SA/20260601_120000/default",
    min_improvement_threshold=0.0,
    min_positive_scenarios=2,
    min_other_threshold=-0.05,
)
```

输出目录：

```text
figures/SA/20260601_120000/default/samples_new/
```

---

### 8.4 可视化所有样本

```python
from pipelines.results_analysis import all_samples

indices = all_samples("GA/20260601_130000/default")
```

输出目录：

```text
figures/GA/20260601_130000/default/all_samples/
```

---

### 8.5 并排对比 SA 和 GA 样本图

先分别生成 `all_samples`：

```python
all_samples("SA/20260601_120000/default")
all_samples("GA/20260601_130000/default")
```

再并排对比：

```python
from pipelines.results_analysis import show_SA_GA

show_SA_GA(
    "SA/20260601_120000/default",
    "GA/20260601_130000/default",
)
```

默认输出目录：

```text
figures/comparison_results/
```

---

### 8.6 SA-GA 综合对比

```python
from pipelines.results_analysis import comparison_pipeline

comparison_pipeline(
    sa_expr_path="SA/20260601_120000/default",
    ga_expr_path="GA/20260601_130000/default",
    plot_types={"lu_diff_bars", "lu_box", "es_box", "combined_area_json"},
    save_folder="figures/comparison_results",
    comparison_type="target",
)
```

输出内容包括：
- `sa_ga_diff_pct_signed.png`
- `SA_GA_lu_boxplot/SA_vs_GA_target.png`
- `SA_GA_es_boxplot/SA_vs_GA_target.png`
- `stacked_bar_data.json`

---

## 九、输出文件结构

### 9.1 单实验统计输出

```text
figures/{SA|GA}/{timestamp}/{scenario}/
├── area/
│   ├── bars/
│   │   ├── total_area_stackbar.png
│   │   └── origin_target_stackbar.png
│   ├── box/
│   │   └── target/
│   └── hist/
│       └── target_density_True/
├── es/
│   ├── bars/
│   ├── box/
│   │   ├── target/
│   │   └── target_new/
│   └── hist/
│       └── target_density_True/
├── samples_new/
└── all_samples/
```

### 9.2 SA-GA 对比输出

```text
figures/comparison_results/
├── sa_ga_diff_pct_signed.png
├── stacked_bar_data.json
├── model_comparison_data.json          # 输入文件（若启用 model_comparison）
├── model_comparison_bar.png
├── SA_GA_lu_boxplot/
│   ├── SA_vs_GA_target.png
│   └── {scenario}_target.png
└── SA_GA_es_boxplot/
    ├── SA_vs_GA_target.png
    └── {scenario}_target.png
```

---

## 十、依赖的数据文件

这些 pipeline 不重新运行优化器，只读取已经存在的优化结果。

典型依赖：

```text
results/{SA|GA}/{timestamp}/{scenario}/{sample_index}/
├── best_plan.npy
├── performance.json
├── convergence.json
└── visualize/
```

统计函数会按需读取或生成缓存 JSON：

| 缓存文件 | 生成函数 | 含义 |
|----------|----------|------|
| `total_area.json` | `cal_total_area()` | 各情景总 LU 面积 |
| `sample_area_target.json` | `cal_sample_area_change_ratio()` | 样本级 LU 面积变化比例 |
| `total_es.json` | `cal_total_es()` | 各情景总 ES 值 |
| `sample_es_target.json` | `cal_sample_es_change_ratio()` | 样本级 ES 变化比例 |

这些缓存通常写入结果目录或由 `load_or_compute_metric()` 自动复用。

---

## 十一、常见注意事项

1. **`expr_path` 可以简写**  
   传 `"SA/20260601_120000/default"` 即可，函数会自动补成 `results/SA/20260601_120000/default`。

2. **`sa_ga_pair` 依赖已有图片**  
   `show_SA_GA()` 对比的是 `figures/.../all_samples` 中已有的图片，不是直接读取 `results/` 重新生成。

3. **`comparison_type` 会影响缓存文件名和图含义**  
   - `target`：相对目标 LU / 目标 ES 的变化比例
   - `origin`：相对原始 LU / 原始 ES 的变化比例

4. **`model_comparison_pipeline` 需要先准备 JSON**  
   调用 `model_comparison_pipeline()` 前需确保 `model_comparison_data.json` 存在（包含 `models` 列表和 `metrics` 字典），否则会报错。

5. **该 pipeline 不修改优化结果**  
   它只读取 `results/`，并在 `figures/` 与部分缓存 JSON 中写入统计/绘图结果。
