# Stage 2 随机优化——数据流追踪

> 本文档逐条追踪 Stage 2 优化流程中每个阶段的数据流动。

---

## 1. 数据加载

### 1.1 优化数据读取

**文件路径**：`_data/optimize_data/valid_{year}/{FACTOR}/*.tif`

```
加载内容:
  condition_list = ["DEM", "DL", "GDP", "RK", "RA", "LU"]   # 驱动因子 + 约束 + LU
  ES_list = ["RAIN", "NJW", "ZSL", "PODU", "WATER", "CITY", "GD"]  # ES 数据
  year = "10" (基准年份)

  xf_ori = get_year_data("valid_10")  # 基准年份: (N, 6, 128, 128)
  xf_tgt = get_year_data("valid_15")  # 目标年份: (N, 1, 128, 128) — 仅 LU
  es_data = get_year_data("valid_10") # ES 数据: (N, 7, 128, 128)
```

### 1.2 单样本打包

```
pack = [
  xf_ori[idx][:4],              # factors (DEM, DL, GDP, RK) — (4, 128, 128)
  xf_ori[idx][4:5],             # restrict (RA) — (1, 128, 128)
  es_data[idx],                 # ES 数据 — (7, 128, 128)
  xf_ori[idx][5:6],             # ori LU — (1, 128, 128)
  xf_tgt[idx][0:1],             # tgt LU — (1, 128, 128)
  f"{year}_{idx}_5",            # idx_label
]
```

---

## 2. Pipeline 主循环（run_runner.py）

### 2.1 配置加载

```
config = get_yaml("configs/optimizerConfig.yaml")
→ config.runner.type = "SA" | "GA"
→ config.runner.scenario = "default" | "enterprise" | "farmer" | "tourist" | "all"
→ config.SA / config.GA — 各优化器的参数
→ config.model — Stage 1 模型配置
```

### 2.2 MaxEnt 模型训练

```python
train_maxent(config.data.root, config.data.year, maxent_path)
# 训练文化服务适宜性的 MaxEnt 模型
# 输入: POI、距道路距离、LU 等
# 输出: maxent_model → 用于 cal_es_score 中的 culture 项
```

### 2.3 Scenario 遍历

```
for current_scenario in scenario_list:
    weight_dict = {'food': 0.25, 'carbon': 0.25, 'water': 0.25, 'culture': 0.25}
    # → 传入优化器，用于 cal_es_score 的加权
```

---

## 3. 样本处理（process_batch）

### 3.1 数据转换

```python
data_dict = {
    'factors':    tensor(factors),      # (1, 4, 128, 128)
    'restrict':   tensor(restrict),     # (1, 1, 128, 128)
    'DEM':        tensor(DEM),          # (1, 1, 128, 128)
    'DL':         tensor(DL),           # (1, 1, 128, 128)
    'GDP':        tensor(GDP),          # (1, 1, 128, 128)
    'RK':         tensor(RK),           # (1, 1, 128, 128)
    'RAIN':       tensor(RAIN),         # (1, 1, 128, 128)
    'NJW':        tensor(NJW),          # (1, 1, 128, 128)
    ...                                 # 其他 ES 因子
    'ori_plan':   tensor(ori_LU),       # (1, 1, 128, 128)
    'final_plan': tensor(tgt_LU),       # (1, 1, 128, 128)
}
```

---

## 4. Runner 初始化（SA/GA.__init__）

### 4.1 ANN 推理

```python
self.pred_diff = get_pred_diff(ANN, ori_plan, factors, sigmoid_threshold=0.5)
# 输入: ori_plan(1,1,128,128) + factors(1,4,128,128)
# → concat → (1, 10, 128, 128)
# → model(inputs) → sigmoid → 阈值 0.5 二值化
# 输出: pred_diff (1,1,128,128), 值为 0.0 或 1.0
```

### 4.2 hard_restrict 拼装

**SA:**
```python
restrict_parts = [redline_restrict, lu_zero_restrict]
hard_restrict = redline AND lu_zero   # 逐像素 AND
```

**GA:**
```python
restrict_parts = [redline_restrict]   # GA 无 lu_zero_restrict
hard_restrict = redline
```

### 4.3 pred_diff_pixelnum

```python
self.pred_diff_pixelnum = count_nonzero(hard_restrict AND pred_diff)
# → 用于 λ 调度和 scale_minmax 计算
```

### 4.4 scale_minmax_dict

```python
self.scale_minmax_dict = get_scale_minmax_dict(pred_diff_pixelnum)
# small: [1, 0.25 × n]
# mid:   [0.25n, 0.5n]
# large: [0.5n, n]
```

### 4.5 info_dict 构造

```python
self.info_dict = get_info_dict(ori_plan, pred_diff, data_dict, ...)
# 包含:
#   - pred_diff_distance_map: pred_diff 的距离变换图
#   - carbon_pool: 碳库数据
#   - maxent_model: MaxEnt 模型
#   - ori_distance_map_dict: 原始方案各类别的距离图
#   - ori_num_feature_dict: 原始方案各类别的斑块数
#   - ... 其他 ES/景观计算所需数据
```

---

## 5. 优化主循环

### 5.1 SA 单步

```
for step in range(max_iter):
    T = T_start × cooling_rate^step

    # 1. 生成新方案
    new_plan = generate_new_plan(cur_plan, hard_restrict, cls_restrict, scale_minmax, T_ratio)
    #   → 随机选矩形区域 → 改为随机 LU 类型（受 cls_restrict 约束）
    #   → 或 return 操作（恢复为原始方案）

    # 2. 计算 new_diff
    new_diff = (new_plan ≠ ori_plan).float()

    # 3. 计算目标函数
    tgt_func, change_score, es_score, es_dict, landscape = cal_tgt_function(
        cur_plan=new_plan,
        cur_diff=new_diff,
        pred_diff=pred_diff,
        ori_plan=ori_plan,
        info_dict=info_dict,
        es_weight_dict=weight_dict,
        use_landscape=True,
        T_ratio=T/T_start,
    )

    # 4. Metropolis 准则
    ΔE = tgt_func - cur_tgt_func
    if ΔE < 0 or random() < exp(-ΔE / T):
        cur_plan = new_plan
        cur_tgt_func = tgt_func
```

### 5.2 GA 单代

```
for gen in range(max_gen):
    # 1. 评估适应度
    fitness = [-cal_tgt_function(plan) for plan in population]
    # GA 最大化适应度 = 最小化 tgt_func 的相反数

    # 2. 选择（轮盘赌/锦标赛）
    parents = select_parents(population, fitness)

    # 3. 交叉
    children = crossover(parents, hard_restrict, cls_restrict, scale_minmax)
    #   → 交换矩形区域

    # 4. 变异
    children = mutate(children, hard_restrict, cls_restrict, scale_minmax, T_ratio)
    #   → 复用 SA 的扰动策略

    # 5. 精英保留 + 种群更新
    population = next_generation(elite, children, fitness)
```

---

## 6. 目标函数计算（cal_tgt_function）

```python
def cal_tgt_function(cur_plan, cur_diff, pred_diff, ori_plan, info_dict,
                     es_weight_dict, use_landscape, T_ratio, score_cfg, change_scale_n):

    # 第 1 项: change_score
    change_score = cal_change_score(cur_diff, pred_diff, info_dict)
    #   → 连通区域到 pred_diff 的距离变换 + inside_reward

    # 第 2 项: es_score
    es_ori_dict, es_rel_dict, es_score = cal_es_score(cur_plan, info_dict, es_weight_dict)
    #   → food + carbon + water + culture 的加权归一化

    # 第 3 项: landscape_score
    landscape_score = cal_landscape_score(cur_plan, cur_diff, ori_plan, info_dict)
    #   → 连通分量纯度 + 类别空间距离

    # 第 4 项: λ 自适应
    n = change_scale_n or count_nonzero(pred_diff)
    lambda_ES, lambda_LS = interpolate(n, pixel_low=300, pixel_high=6000)

    # 第 5 项: 目标函数
    tgt_func = change_score - lambda_ES * es_score + lambda_LS * landscape_score
    return tgt_func, change_score, es_score, es_rel_dict, landscape_score
```

---

## 7. 保存与评估

### 7.1 save_plan_and_get_performance

```python
save_plan_and_get_performance(
    best_plan=..., ori_plan=..., final_plan=...,
    pred_diff=self.pred_diff,   # ← 始终是真实 ANN 预测
    info_dict=self.info_dict,
    ...
)
```

内部计算：
- `new_diff = best_plan ≠ ori_plan` → 优化后变化区域
- `IoU = calculate_change_area_metrics(new_diff, pred_diff)` → 与 ANN 预测的交并比
- `ES = calculate_es_score(ori_plan, best_plan, tgt_plan, info_dict)` → 四项 ES 得分

### 7.2 convergence.json

```python
{
    "final_best": -0.xxxx,         # SA: best_score; GA: -best_score
    "wall_time_sec": 123.45,
    "total_iter": 5000,
    "guided": true,
    "disable_early_stop": false
}
```

### 7.3 结果目录结构

```
results/{SA|GA}/{timestamp}/{scenario}/
  └── {idx_label}/
      ├── best_plan.npy           # 最优 LU 方案 (128, 128)
      ├── performance.json        # 性能指标
      ├── convergence.json        # 收敛信息
      ├── total_score_list.npy    # 每步目标函数值序列
      └── visualize/              # 过程可视化图片
          ├── step_500.png
          └── ...
```

---

## 8. 数据流总览

```
_data/optimize_data/valid_10/{FACTOR}/*.tif
    ↓ get_year_data()
[factors(4), restrict(1), ES(7), ori_LU(1), tgt_LU(1)]
    ↓ process_batch()
data_dict = {factors, restrict, DEM, DL, GDP, RK, RAIN, ..., ori_plan, final_plan}
    ↓ Runner 初始化
    ├── get_pred_diff(ANN, ori_plan, factors) → pred_diff [1,1,128,128]
    ├── hard_restrict = redline ∧ lu_zero
    ├── pred_diff_pixelnum = count_nonzero(hard_restrict ∧ pred_diff)
    ├── scale_minmax = get_scale_minmax_dict(pred_diff_pixelnum)
    └── info_dict = get_info_dict(ori_plan, pred_diff, data_dict, ...)
    ↓ 优化主循环 (SA: 单步迭代 / GA: 种群进化)
    ├── generate_new_plan / crossover + mutate
    ├── new_diff = (new_plan ≠ ori_plan)
    └── cal_tgt_function(new_plan, new_diff, pred_diff, ...)
        ├── change_score = cal_change_score(new_diff, pred_diff, info_dict)
        ├── es_score = cal_es_score(new_plan, info_dict, weight_dict)
        ├── landscape_score = cal_landscape_score(new_plan, new_diff, ori_plan, info_dict)
        └── tgt_func = change_score - λ_ES × es_score + λ_LS × landscape
    ↓ 保存
best_plan.npy + performance.json + convergence.json + total_score_list.npy
```
