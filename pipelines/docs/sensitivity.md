# Ablation Pipeline — Sensitivity 一维线搜索数据流说明

> 本文档聚焦 `pipelines/tornado.py` 在 `mode='sensitivity'` 下的内部方法逻辑与数据流。
>
> 在项目整体架构中，本 pipeline 属于 **Stage-2 优化搜索** 的实验平台（详见 `pipelines/README.md`）。
> 对应 `main.py` 入口：`run_sensitivity_pipeline(..., mode='sensitivity')`。
>
> Sensitivity 模式用于**对 tornado 筛选出的敏感参数做细粒度取值扫描**，绘制参数-指标趋势曲线，确定最优取值区间。

---

## 一、关键参数

### 1.1 `run_sensitivity_pipeline()`（main.py 入口）

| 参数 | 来源 | 含义 |
|------|------|------|
| `opt_cfg_path` | 硬编码 `"configs/optimizerConfig.yaml"` | 优化器+模型+数据配置 |
| `score_cfg_path` | 硬编码 `"configs/score_cfg.yaml"` | 目标函数各项权重和阈值 |
| `idxs` | 从 `_data/representative_samples.json` 读取 | 样本索引（默认 fallback，取前 3 个） |
| `mode` | 硬编码 `'sensitivity'` | 实验模式 |
| `n_repeats` | 函数默认值 `2`，`main.py` 中传入 `3` | 每个配置重复次数（比 tornado 多，因需误差条） |
| `top_k_params` | 函数内默认定义 | 待扫描的参数规格列表 |

### 1.2 `top_k_params` 格式

```python
[
    {
        'path': 'tgt_function.lambda_ES_high',   # 参数在 score_cfg/opt_cfg 中的点号路径
        'baseline': 1000,                         # 基线值（仅作参考）
        'values': [200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000],  # 扫描取值
        'category': 'score_cfg',                  # 'score_cfg' 或 'opt_cfg'
    },
    {
        'path': 'cooling_rate',
        'baseline': 0.9995,
        'values': [0.9985, 0.9990, 0.9992, 0.9993, 0.9995, 0.9996, 0.9997, 0.9998, 0.9999],
        'category': 'opt_cfg',
    },
]
```

---

## 二、数据流总览

```
main.py
  ↓ 构造 top_k_params（敏感参数 + 扫描取值）
  ↓ 调用 tornado_pipeline(..., mode='sensitivity', n_repeats=3, top_k_params=...)

pipelines/tornado.py
  ├─ 1. 加载配置
  │     ├─ optimizerConfig.yaml → opt_cfg
  │     └─ score_cfg.yaml → base_score_cfg
  │
  ├─ 2. 构造实验矩阵 (build_experiment_matrix)
  │     mode='sensitivity' + top_k_params
  │     → 为每个参数的每个取值生成 S_<param>=<value> 实验
  │
  ├─ 3. 遍历每个实验
  │     ├─ 深拷贝 score_cfg → overrides 覆写
  │     ├─ 深拷贝 type_config → opt_overrides 覆写
  │     ├─ 确定性随机种子
  │     │
  │     └─ 4. 遍历样本 + 重复 run
  │           └─ create_runner() → runner.run()
  │
  └─ 6. 保存 manifest.json + experiment_summary.json
```

---

## 三、实验矩阵

```python
build_experiment_matrix(mode='sensitivity', n_repeats=3, top_k_params=top_k_params)
```

### 3.1 命名规则

实验名格式：`S_<dotted_path>=<value>`

示例：
- `S_tgt_function.lambda_ES_high=800`
- `S_cooling_rate=0.9993`

### 3.2 group 标识

所有 sensitivity 实验的 `group='S'`，与 tornado 的 `group='B'`、ablation 的 `group='A/B/C'` 区分。

---

## 四、与 tornado / ablation 模式的核心区别

| 维度 | `mode='sensitivity'` | `mode='tornado'` | `mode='ablation'` |
|------|----------------------|------------------|-------------------|
| 实验组数 | 取决于 `top_k_params`（如 2 参数 × 9 取值 = 18 组） | 21 组 | 3 组 |
| 目的 | **细粒度趋势分析** | **粗粒度筛选** | **模块有效性验证** |
| 取值密度 | **7~9 个取值/参数**（连续扫描） | 2~3 个取值/参数（离散扰动） | 无 |
| `n_repeats` | **3**（误差条需要） | 2（快速筛选） | 5 |
| `disable_early_stop` | `False`（正常早停） | `False` | `True`（跑到底） |
| SA.T_end | **0.1**（正常） | 0.1 | 1e-6（极端） |
| 随机种子 | ✅ 确定性种子 | ✅ | ❌ |
| 时间成本 | 3 样本 × 18 组 × 3 次 ≈ **43 h** | 5×21×2 ≈ ~54 h | 5×3×5 ≈ 3~6 h |

---

## 五、结果目录结构

```
results/
└── tornado/
    └── 20260602_HHMMSS/
        ├── experiment_summary.json
        │
        ├── S_tgt_function.lambda_ES_high=200/
        ├── S_tgt_function.lambda_ES_high=400/
        ├── ...（共 9 个）
        │
        ├── S_cooling_rate=0.9985/
        ├── S_cooling_rate=0.9990/
        ├── ...（共 9 个）
        │
        └── plots/
            └── sensitivity_final_best.png   ← 一维线图输出
```

---

## 六、断点续传

与所有 ablation 模式相同：

```python
tornado_pipeline(
    ..., mode='sensitivity', n_repeats=3,
    top_k_params=top_k_params,
    is_processed=True,
    processed_timestamp="20260602_120000",
)
```

---

## 七、后处理出图

### 7.1 入口

```python
from pipelines.tornado_plots import plot_sensitivity_lines
plot_sensitivity_lines("results/tornado/<timestamp>", metric='final_best')
```

支持更换 metric：
```python
plot_sensitivity_lines("results/tornado/<timestamp>", metric='iter_to_95pct')
plot_sensitivity_lines("results/tornado/<timestamp>", metric='ES.carbon.best_ori_diff')
```

### 7.2 输出文件

| 文件 | 说明 |
|------|------|
| `plots/sensitivity_final_best.png` | x=参数取值, y=`final_best` 均值 ± std |
| `plots/sensitivity_iter_to_95pct.png` | 对收敛速度的敏感度 |
| `plots/sensitivity_ES.*.png` | 对各项 ES 指标的敏感度 |

### 7.3 解读线图

- **横轴**：参数取值（连续排列）
- **纵轴**：指标均值（跨样本平均）
- **误差条**：跨样本标准差（`n_repeats=3` 提供）
- **趋势判断**：
  - 曲线平坦 → 参数不敏感，可放宽或固定
  - 曲线有明显峰值/谷值 → 最优取值区间在极值点附近
  - 曲线单调 → 参数有方向性影响，需结合实际意义选择

### 7.4 多参数子图布局

若 `top_k_params` 包含 2~3 个参数，`plot_sensitivity_lines` 会自动排列子图：
- 1 个参数：单图
- 2 个参数：1 行 2 列
- 3 个参数：1 行 3 列
- 4~6 个参数：2 行 3 列
- ...

---

## 八、完整工作流示例

### Step 1: Tornado 粗筛

```python
# main.py
run_tornado_pipeline()
```

跑完后查看 `plots/tornado_final_best.png`，假设发现 `lambda_ES_high` 和 `cooling_rate` 最敏感。

### Step 2: 构造 top_k_params

```python
top_k_params = [
    {
        'path': 'tgt_function.lambda_ES_high',
        'baseline': 1000,
        'values': [200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000],
        'category': 'score_cfg',
    },
    {
        'path': 'cooling_rate',
        'baseline': 0.9995,
        'values': [0.9985, 0.9990, 0.9992, 0.9993, 0.9995, 0.9996, 0.9997, 0.9998, 0.9999],
        'category': 'opt_cfg',
    },
]
```

### Step 3: 跑线搜索

```python
run_sensitivity_pipeline(top_k_params=top_k_params)
```

### Step 4: 出图分析

```python
from pipelines.tornado_plots import plot_sensitivity_lines
plot_sensitivity_lines("results/tornado/<timestamp>", metric='final_best')
plot_sensitivity_lines("results/tornado/<timestamp>", metric='ES.carbon.best_ori_diff')
```

根据曲线趋势确定最优参数组合，后续可用 `mode='ablation'` 或正常 `run_runner_pipeline()` 验证。
