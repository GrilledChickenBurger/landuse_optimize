# Ablation Pipeline — A/B/C 消融对比实验数据流说明

> 本文档聚焦 `pipelines/ablation.py` 在 `mode='ablation'` 下的内部方法逻辑与数据流。
>
> 在项目整体架构中，本 pipeline 属于 **Stage-2 优化搜索** 的实验平台（详见 `pipelines/README.md`）。
> 对应 `main.py` 入口：`run_ablation_pipeline(..., mode='ablation')`。
>
> 与 `mode='stability'`（仅 A_guided 单组重复）的区别见文末对照表。

---

## 一、关键参数

### 1.1 `run_ablation_pipeline()`（main.py 入口）

| 参数 | 来源 | 含义 |
|------|------|------|
| `opt_cfg_path` | 硬编码 `"configs/optimizerConfig.yaml"` | 优化器+模型+数据配置 |
| `score_cfg_path` | 硬编码 `"configs/score_cfg.yaml"` | 目标函数各项权重和阈值 |
| `idxs` | 从 `_data/representative_samples.json` 读取 `idxs_for_pipeline` | 要跑的样本索引列表 |
| `mode` | 硬编码 `'ablation'` | 实验模式：跑 A/B/C 三组对比 |
| `n_repeats` | 函数默认值 `1`，`main.py` 中传入 `2` | 每组每个样本重复跑的次数 |

### 1.2 `ablation_pipeline()`（pipelines/ablation.py 核心）

| 参数 | 类型 | 作用 |
|------|------|------|
| `is_processed` | `bool` | 是否断点续传（默认 `False`）|
| `processed_timestamp` | `Optional[str]` | 断点续传用的时间戳 |

---

## 二、数据流总览

```
main.py
  ↓ 读取 _data/representative_samples.json → idxs = [1417, 2502, ...]
  ↓ 调用 ablation_pipeline(..., mode='ablation', n_repeats=5)

pipelines/ablation.py
  ├─ 1. 加载配置
  │     ├─ optimizerConfig.yaml → opt_cfg (Namespace)
  │     └─ score_cfg.yaml → base_score_cfg (Namespace)
  │
  ├─ 2. 构造实验矩阵 (build_experiment_matrix)
  │     mode='ablation' → 3 组实验: A_guided / B_unguided / C_random
  │     每组各 × n_repeats 次
  │
  ├─ 3. 遍历每个实验（外层循环: A → B → C）
  │     ├─ 深拷贝 score_cfg → 按 overrides 覆写 → apply_const_overrides()
  │     ├─ 禁用早停 (disable_early_stop=True)
  │     │
  │     └─ 4. 遍历 dataloader 中所有样本
  │           ├─ 过滤: 只保留 idxs 列表中的样本
  │           ├─ process_batch() → data_dict
  │           │
  │           └─ 5. 重复 n_repeats 次
  │                 ├─ 生成 save_path: results/ablation/{ts}/{exp}/run_r/{index}/
  │                 ├─ C 组: 生成 random_mask → random_restrict_mask
  │                 ├─ create_runner(SA/GA, guided=..., random_restrict_mask=...)
  │                 └─ runner.run()
  │
  └─ 6. 每组保存 manifest.json + 全局 experiment_summary.json
```

---

## 三、实验矩阵（A/B/C 三组）

```python
build_experiment_matrix(mode='ablation', n_repeats=5)
```

返回 3 项实验配置：

| 组 | name | overrides | flags | n_repeats | group |
|---|------|-----------|-------|-----------|-------|
| A | `A_guided` | `{}` | `guided=True`, `disable_early_stop=True`, `skip_threshold_check=True` | 5 | A |
| B | `B_unguided` | `{'ablation.disable_change_score': True}` | `guided=False`, `disable_early_stop=True`, `skip_threshold_check=True` | 5 | B |
| C | `C_random` | `{}` | `guided=False`, `disable_early_stop=True`, `skip_threshold_check=True` | 5 | C |

### 3.1 A_guided — 完整方法（基线）

- `guided=True`：runner 内部用 `pred_diff` 生成 `grid_restrict`，限制搜索区域
- `disable_change_score=False`（默认）：目标函数包含 `change_score` 项
- `random_restrict_mask=None`
- **等价于 stability 模式的单组行为**

### 3.2 B_unguided — 移除位置引导，保留面积约束

- `guided=False`：不生成 `grid_restrict`，搜索范围 = `redline ∧ lu_zero`
- `disable_change_score=True`：`cal_change_score` 不再计算距离变换，而是返回**纯变化面积惩罚**
  - 惩罚公式：`(area_diff²) × inside_reward_coef`，`area_diff = |cur_area - target_area|`
  - 软死区：±10% 内线性奖励（`quantity_penalty_delta = 0.1`）
- **验证**：没有 `pred_diff` 位置引导、但变化面积被约束时，优化器表现如何？

### 3.3 C_random — 随机空间约束

- `guided=False`：不生成 `grid_restrict`
- `disable_change_score=False`（默认）：目标函数仍包含 `change_score`
- 但 `change_target` 被替换为 `random_mask`（与 `pred_diff` 面积相近的随机连通块）
- **验证**：任意空间约束能否替代深度学习预测先验？

#### C 组 random_mask 生成流程

```python
# 1. 构造 base_hard_restrict（耕地红线 + LU≠0）
redline = data_dict['restrict']
lu_zero = (data_dict['ori_plan'] != 0)
base_hard = torch.logical_and(redline, lu_zero)

# 2. 临时跑一次 ANN 拿 pred_diff（仅用于统计连通块面积）
pred_diff_c = get_pred_diff(ann_model, ori_plan, factors, device, ...)

# 3. 生成与 pred_diff 面积/连通块数相近的随机连通块
random_mask = make_random_connected_mask(
    pred_diff_c, base_hard, seed=base_seed + r)
```

`random_mask` 通过 `random_restrict_mask` 参数传入 runner：
- 替代正常的 `grid_restrict`
- `change_target` 在 runner 内部被设为 `random_mask`
- `change_score` 计算的是当前方案变化区域与 `random_mask` 的空间重合度

---

## 四、三组核心差异对照

| 维度 | A (guided) | B (unguided) | C (random) |
|------|-----------|--------------|-----------|
| `guided` | True | False | False |
| `grid_restrict` | ✅ 有 | ❌ 无 | ❌ 无 |
| `change_term` | `change_score` (距离+面积) | `quantity_penalty` (纯面积) | `change_score_random` |
| `change_target` | `pred_diff` (ANN) | `pred_diff` (仅用于计算 target_area) | `random_mask` |
| `hard_restrict` | `redline ∧ lu_zero ∧ grid` | `redline ∧ lu_zero` | `redline ∧ lu_zero ∧ random_mask` |
| 搜索空间 | 被 grid_restrict 压缩 | 无 grid_restrict，但有面积惩罚约束 | 被 random_mask 压缩 |
| 预期收敛 | 最快 | 慢（无位置引导，但面积被约束） | 中等 |
| 预期 IoU(vs ANN) | 最高 | 低 | 低 |

---

## 五、结果目录结构

```
results/
└── ablation/
    └── 20260602_HHMMSS/            ← 自动生成的 timestamp
        ├── experiment_summary.json   ← 全局清单（mode='ablation', experiments=[A,B,C]）
        │
        ├── A_guided/               ← 实验 A
        │   ├── manifest.json
        │   ├── run_0/
        │   │   ├── 10_1417_5/
        │   │   ├── 10_2502_5/
        │   │   └── ...
        │   ├── run_1/
        │   └── ...
        │
        ├── B_unguided/             ← 实验 B
        │   ├── manifest.json
        │   └── run_0/
        │       └── ...
        │
        └── C_random/               ← 实验 C
            ├── manifest.json
            ├── run_0/
            │   ├── 10_1417_5/
            │   │   ├── random_mask.npy
            │   │   ├── convergence.json
            │   │   └── ...
            │   └── ...
            └── ...
```

每组目录下的 `manifest.json` 记录该组的 `overrides` 和 `flags`。

---

## 六、断点续传机制

若某次运行中途退出，可设置 `is_processed=True`：

```python
ablation_pipeline(
    opt_cfg_path="configs/optimizerConfig.yaml",
    score_cfg_path="configs/score_cfg.yaml",
    idxs=idxs,
    mode='ablation',
    n_repeats=5,
    is_processed=True,
    processed_timestamp="20260602_120000",
)
```

程序会按 **实验名 → run_r → 样本 index** 的层级检查 `convergence.json` 是否存在，存在则跳过该 run。

---

## 七、与 stability 模式的区别

| 对比项 | `mode='stability'` | `mode='ablation'` |
|--------|-------------------|-------------------|
| 实验组数 | 1 组（A_guided） | 3 组（A/B/C） |
| 目的 | 验证随机优化稳定性 | 验证预测引导的有效性 |
| 时间成本 | 5 样本 × 1 算法 × 5 次 ≈ 1~2 h | 5 样本 × 3 组 × 5 次 ≈ 3~6 h |
| score_cfg 覆写 | 无 | B 组覆写 `disable_change_score=True` |
| random_mask | 无 | C 组生成 |
| 核心假设检验 | 5 次结果 95% CI 是否窄 | A 的 T95 < B 且 A 的 IoU >> C |

---

## 八、后处理出图

对 `mode='ablation'` 结果调用 `pipelines.ablation_plots.run_all_ablation_plots(experiment_root)`：

| 输出文件 | 内容 |
|---------|------|
| `plots/ablation_convergence_<idx>.png` | A/B/C 三组平均收敛曲线对比（含 T95 竖线） |
| `plots/stability_convergence_<idx>.png` | A 组多次重复收敛曲线叠图 |
| `plots/boxplot_final_f.png` | 各样本 × A/B/C 最终目标函数箱线图 |
| `plots/ablation_compare_table.csv` | A/B/C 并排对比表（含相对 A 的变化百分比） |
| `plots/stability_table_*.csv` | 每组稳定性统计表 |
