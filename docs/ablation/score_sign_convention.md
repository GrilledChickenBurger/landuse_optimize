# SA / GA 分数符号约定全流程梳理

> 目的：澄清三项分量（change_score, es_score, landscape_score）在每个环节（**原始计算 → 目标函数 → 内存列表 → 收敛图 → performance.json → convergence.json**）中**符号方向**与**优劣方向**的变化，避免读图、写报告时方向混淆。

---

## TL;DR 速查表

| 出现位置 | 字段名 | 数值符号 | 越什么越好？ |
|---------|--------|---------|-------------|
| **目标函数 `cal_tgt_function` 内部** | `change_relative_score` (原始) | 一般为负 | **越小越好**（更负） |
| | `es_relative_score` (原始) | 正，[0, 1] | **越大越好** |
| | `landscape_score` (原始) | 正或负 | **越小越好**（更负） |
| | `tgt_func` = change − λ_ES·es + λ_LS·ls | 通常为负（大样本） | **越小越好** |
| **SA 内存列表（绘图/打印用）** | `change_score_list` | **取反** → 一般为正 | **越大越好** |
| | `es_score_list` | 不变，正 | **越大越好** |
| | `landscape_score_list` | **取反** → 一般为正或负 | **越大越好** |
| | `total_score_list` | 不变（= tgt_func） | **越小越好** |
| **GA 内存列表** | 同 SA（change/landscape 取反） | 同上 | 同上 |
| | `total_score_list` | **不变（= tgt_func，未取反）** | **越小越好** ⚠️注意与 fitness 不同 |
| | （内部 `fitness` 变量 = −tgt_func） | （不入列表） | 越大越好（GA 内部用） |
| **`charts/*.png` 图表** | `change_score.png` | y = −change_relative_score | **越大越好**（曲线上升=改善） |
| | `ES_score.png` | y = es_relative_score | **越大越好** |
| | `landscape_score.png` | y = −landscape_score | **越大越好**（曲线上升=改善） |
| | `total_score.png` | y = tgt_func | **越小越好**（曲线下降=改善） |
| | `ES详情.png` | 4 条曲线 = es_weighted_score per type | **越大越好** |
| **`convergence.json`** | `final_best` | = self.best_score（SA）/ −best_fitness（GA） | **越小越好**（与 tgt_func 同向） |
| | `es_score` | es_score_list[best_iter] | **越大越好**（[0, 1]） |
| | `iter_to_95pct`, `wall_time_sec`, `total_iter` | 正整数/正实数 | **越小越好** |
| **`performance.json`** | `Change_area.*.{accuracy, precision, recall, f1, iou}` | [0, 1] | **越大越好** |
| | `ES.{food,carbon,water,culture}.best_plan/tgt_plan/ori_plan` | 正绝对值 | （绝对量，无方向）|
| | `ES.{food,carbon,water,culture}.best_ori_diff` | 比例（可正可负） | **越大越好** |
| | `ES.{food,carbon,water,culture}.best_tgt_diff` | 比例（可正可负） | （趋近 0 越好）|
| | `Landscape.land.*.{LPI,LSI,AI,ED,SHEI,SHDI,CONTAG}` | 字符串数字 | 视具体指标 |
| | `FINAL.FOM`, `FINAL.kappa` | [0, 1] | **越大越好** |
| | `FINAL.best_ori.<lu>.{change_hectares, change_ratio}` | 可正可负 | （绝对量，无方向）|

---

## 1. 三个分量在 `cal_tgt_function` 中的原始定义

```python
# calculator/tgt_function_cal.py:42-67
# 1. 变化得分 — 越小越好
change_relative_score = cal_change_score(...)
# 2. 环境得分 — 越大越好，范围 [0, 1]
es_value_dict, es_relative_score_dict, es_relative_score = cal_es_score(...)
# 3. 景观格局得分 — 越小越好
landscape_score = cal_landscape_score(...)

# 4. 目标函数: 越小越好
tgt_func = change_relative_score - lambda_ES * es_relative_score + lambda_LS * landscape_score
```

### 1.1 `change_relative_score` —— 越小越好（一般为负）

来源：`calculator/cal_change_score.py::cal_change_value()`

```python
# 关键累加逻辑（line 96-104）
if distance > 0 and intersection_area == 0:
    total_distance += (area ** 2) * distance           # 正惩罚
elif distance > 0 and intersection > 0 and area − intersection > 0:
    total_distance += ((diff_area ** 2) * distance
                       − (intersection ** 2) * inside_reward_coef)  # 负奖励
elif distance == 0 or area − intersection == 0:
    total_distance −= (area ** 2) * inside_reward_coef  # 全负奖励
```

- **完全位于预测变化区域内的连通区域 → 负贡献（奖励）**
- **完全位于预测变化区域外的连通区域 → 正贡献（惩罚）**
- **部分重叠 → 负奖励 + 正惩罚的代数和**

实际数据中，由于预测区域内部分通常占多数，`change_relative_score` 是一个较大的**负数**。"越小（越负）→ 越好"。

### 1.2 `es_relative_score` —— 越大越好（[0, 1]）

来源：`calculator/cal_es_score.py::cal_es_score()`

```python
# line 30-35
for es_name, es_value in es_value_dict.items():
    max_score = info_dict['max_es_score'][es_name]
    es_score_relative[es_name] = es_value / max_score                  # 归一化到 [0, 1]
    es_weighted_score_relative[es_name] = es_value / max_score * weight_dict[es_name]
es_weighted_score_relative_sum = sum(es_weighted_score_relative.values())
```

- 每个 ES 子项（food/carbon/water/culture）的当前绝对值除以其理论最大值
- 各子项加权求和（权重通常 0.25 × 4）
- 范围 [0, 1]，**越大越好**

### 1.3 `landscape_score` —— 越小越好

来源：`calculator/cal_landscape_score.py::cal_landscape_score()`

```python
# total_pure_score 项：纯度惩罚，>= 0
# total_cls_distance 项：距离 + 增加斑块数惩罚
#   若同类原始斑块就在边上 (min_distance == 0) → cls_distance -= 10/ratio  (负奖励)
#   否则 → cls_distance += (min+max)*ratio² (正惩罚)
total_score = total_cls_distance + total_pure_score * (...)
```

- 通常为正（除非新斑块刚好长在原始同类斑块旁边）
- **越小越好**（破碎化越低）

### 1.4 `tgt_func` 总和 —— 越小越好

```
tgt_func = change_relative_score    # 越小越好（一般为负）
         − λ_ES · es_relative_score # 减号：es 越大 → tgt 越小（即 es 改善 → tgt 改善）
         + λ_LS · landscape_score   # 加号：ls 越小 → tgt 越小（即 ls 改善 → tgt 改善）
```

✅ 三项方向都汇集为"**越小 → 越好**"。

实测数值范围：典型样本下 `tgt_func` 在 `[-78,000,000, -20,000]` 之间，**绝对值大的样本变化像素多，绝对值小的样本变化像素少**。

---

## 2. SA 中的符号变换（`runner/SA.py`）

### 2.1 内存列表写入（`update_score_lists`, line 177-201）

```python
def update_score_lists(self, score_list:List):
    total_score, change_score, es_score, es_dict, landscape_score = score_list
    self.total_score_list.append(total_score)              # 不变
    self.change_score_list.append(-change_score)           # ⚠️ 取反
    self.es_score_list.append(es_score)                    # 不变
    if self.use_landscape:
        self.landscape_score_list.append(-landscape_score) # ⚠️ 取反
    for key, value in es_dict.items():
        self.es_dict[key].append(value)                    # 不变（各 ES 子项）
```

**变换：**
- `change_score` → **取反**保存，使得列表中**越大越好**
- `landscape_score` → **取反**保存，使得列表中**越大越好**
- `es_score`, `total_score`, `es_dict` 各项 → 原样保存

### 2.2 控制台打印（`update_score_lists`, line 193）

```python
self.text = f'change_score={-change_score:.4f}, ES_score={-es_score:.4f}'
```

⚠️ **注意：打印 `ES_score` 时也加了负号**，所以控制台看到的 `ES_score` 是**负值**（与列表中保存的正值方向相反）。这是一个**展示口径不一致**的地方——打印里 ES 越小越好，列表/图里 ES 越大越好。

> 📌 同样的逻辑在 `SA.py:323` 的 `visualize_frequency` 分支也出现：
> `print(f"change_score={cur_score_list[1]:.4f}, ES_score={cur_score_list[2]:.4f}, landscape_score={cur_score_list[4]:.4f}.")`
> 这里**直接打印 cur_score_list 的原始值**（未取反），所以 change_score 是负数、ES_score 是正数、landscape_score 是正数。**与上面 `update_score_lists` 里的 text 又不一致**——这是个潜在的混淆点。

### 2.3 best_score 跟踪（line 304-308）

```python
if cur_score < self.best_score:    # cur_score = tgt_func，越小越好
    self.best_plan = cur_plan.clone()
    self.best_score = cur_score
```

`self.best_score` 与 `tgt_func` 同向 —— **越小越好**，一般为负。

### 2.4 收敛指标计算（`_dump_convergence`, line 381-408）

```python
score_arr = np.array([float(s) for s in self.total_score_list])  # 原始 tgt_func，越小越好
np.save('total_score_list.npy', score_arr)

best = float(self.best_score)        # 负值
if best < 0:
    target = best * 0.95             # |target| < |best|，例：best=-100 → target=-95
else:
    target = best * 1.05
iter_to_95pct = int(np.argmax(score_arr <= target)) if (score_arr <= target).any() else -1

es_score = self.es_score_list[self.best_score_iter]  # 取最优解出现时的 ES（正值）

conv = {
    'iter_to_95pct': iter_to_95pct,    # 正整数，越小越好
    'final_best': best,                # 负值，越小越好
    'es_score': es_score,              # 正值，[0,1]，越大越好
    'wall_time_sec': wall_time,        # 正实数，越小越好
    'total_iter': len(score_arr),      # 正整数
    'guided': self.guided,
    'disable_early_stop': self.disable_early_stop,
}
```

**`total_score_list.npy` 中保存的是原始 `tgt_func`，未取反，越小越好。**

---

## 3. GA 中的符号变换（`runner/GA.py`）

### 3.1 内部 fitness 转换（`fitness()`, line 606-637）

```python
score_list[0] = -score_list[0]  # 适应度为负总得分 → 越大越好
```

⚠️ **关键差异**：GA 内部把 `tgt_func` **取反**作为 `fitness`，使其成为**最大化问题**（"适应度越大越好"）。

但是！

### 3.2 内存列表写入（`update_score_lists`, line 469-500）

```python
total_score, change_score, es_score, es_dict, landscape_score = score_list
self.total_score_list.append(total_score)              # ⚠️ total_score 此时是 fitness (已取反)
self.change_score_list.append(-change_score)           # 取反
self.es_score_list.append(es_score)                    # 不变
self.landscape_score_list.append(-landscape_score)     # 取反
```

注意：此时传入的 `score_list` 是经过 `fitness()` 处理的（第一项已被取反），所以 **GA 的 `total_score_list` 中存的是 fitness，越大越好**，与 SA 的 `total_score_list`（tgt_func，越小越好）**方向相反**。

### 3.3 best_score 跟踪（line 386, 760-763）

```python
self.best_score = float('-inf')  # 初始化为负无穷（最大化问题）

# line 760-763
if cur_score > self.best_score:   # cur_score = fitness，越大越好
    self.best_plan = cur_plan.clone()
    self.best_score = cur_score   # 保存的是 fitness，正值
```

`self.best_score` 是 **fitness（已取反），越大越好，一般为正**。

### 3.4 收敛指标计算（`_dump_convergence`, line 837-866）

```python
score_arr = np.array([float(s) for s in self.total_score_list])  # fitness 序列，越大越好
np.save('total_score_list.npy', score_arr)

best_fitness = float(self.best_score)  # 正值（fitness）
if best_fitness < 0:
    target = best_fitness * 1.05
else:
    target = best_fitness * 0.95       # 例：best=100 → target=95
iter_to_95pct = int(np.argmax(score_arr >= target)) if (score_arr >= target).any() else -1

es_score = self.es_score_list[self.best_score_generation]  # 正值

conv = {
    'iter_to_95pct': iter_to_95pct,    # 正整数
    'final_best': -best_fitness,       # ⚠️ 再取反！与 SA 统一口径，输出原始 tgt_func（负值，越小越好）
    'es_score': es_score,              # 正值，[0,1]
    'wall_time_sec': wall_time,
    'total_iter': len(score_arr),
}
```

⚠️ **关键设计**：`final_best` 在写入 `convergence.json` 时**又取反一次**（`-best_fitness`），所以最终落盘的 `final_best` **与 SA 完全同向（都是负值，越小越好）**。

🔴 **但是！** `total_score_list.npy` 文件本身**没有再取反**：
- **SA 的 `total_score_list.npy`**: 原始 tgt_func，**越小越好**
- **GA 的 `total_score_list.npy`**: fitness = −tgt_func，**越大越好**

`utils/exp_sweep_utils.py::recompute_iter_to_95pct()` 中专门处理了这个差异（line 233-251）：根据 `runner_type` 判断方向。

---

## 4. 图表（`charts/*.png`）的方向

入口：`SA.py:360-362` / `GA.py:799-807` 调用 `draw_score_list()`（`utils/runner_utils.py:219`）

```python
score_list_data = [self.total_score_list,
                   self.change_score_list,      # SA: −change（正）, GA: 同
                   self.es_score_list,          # 不变（正）
                   self.landscape_score_list,   # SA: −landscape（一般正）, GA: 同
                   self.es_dict]                # 不变（各 ES 子项，正值）
rUtils.draw_score_list(score_list_data, ['total_score', 'change_score', 'ES_score', 'landscape_score'], ...)
```

| 图表文件 | y 轴数据 | 方向 |
|---------|---------|------|
| `total_score.png` | SA: tgt_func（负）<br>GA: fitness（正） | **SA: 越低越好（曲线下降=改善）**<br>**GA: 越高越好（曲线上升=改善）** ⚠️ |
| `change_score.png` | −change_relative_score（一般正） | **越高越好** |
| `ES_score.png` | es_relative_score（[0, 1]） | **越高越好** |
| `landscape_score.png` | −landscape_score | **越高越好** |
| `ES详情.png` | 4 条曲线：es_weighted_score per type | **越高越好** |

**⚠️ 这是一个隐藏的方向不对称**：`total_score.png` 在 SA 和 GA 中的方向**相反**。读图时必须先知道是 SA 还是 GA。

---

## 5. `performance.json` 的字段方向

入口：`SA.py:346-358` / `GA.py:785-797` → `runner_utils.py::save_plan_and_get_performance()` → `performance.py::write_performance_score()`

`performance.json` 与目标函数完全独立——它是优化结束后**单独评估最优方案的质量**，不参与训练/优化。

### 5.1 `Change_area`（变化区域质量）

```json
"Change_area": {
    "real_metrics": { "accuracy(OA)": 0.978, "precision(UA)": 0.854, "recall(PA)": 0.991, "f1_score": 0.917, "iou": 0.848, ... },
    "pred_metrics": { ... }
}
```

- `real_metrics`：与**真实未来 LU 变化区域**的对比指标
- `pred_metrics`：与 **ANN 预测变化区域**的对比指标
- 所有指标 ∈ [0, 1]，**越大越好**

### 5.2 `ES`（生态系统服务）

```json
"ES": {
    "food": {
        "best_plan": 8638719.0,        // 当前方案的绝对得分 (kg/a)
        "tgt_plan": 8827327.0,         // 真实未来方案的绝对得分
        "ori_plan": 11543444.0,        // 原始方案的绝对得分
        "best_ori_diff": -0.252,       // (best - ori) / ori，相对原始方案的变化率
        "best_tgt_diff": -0.021        // (best - tgt) / tgt，相对真实方案的变化率
    },
    "carbon": { ... },  "water": { ... },  "culture": { ... }
}
```

公式（`performance.py:278-279`）：

```python
best_ori_diff = (best_value - ori_value) / ori_value
best_tgt_diff = (best_value - tgt_value) / tgt_value
```

| 字段 | 方向 | 解读 |
|------|------|------|
| `best_plan`/`tgt_plan`/`ori_plan` | 绝对值 | 大小因 ES 类型量纲而异 |
| `best_ori_diff` | **越大越好**（正=ES 提升，负=ES 下降） | 优化后 vs 原始的变化率 |
| `best_tgt_diff` | **趋近 0 越好**（接近真实方案） | 优化后 vs 真实未来的偏离度 |

**注意：tornado 分析里使用的 `ES.<type>.best_ori_diff` 越大越好。例如：**
- `ES.food.best_ori_diff = -0.25` 意味着粮食服务下降了 25%（**负面**）
- `ES.culture.best_ori_diff = +0.03` 意味着文化服务提升了 3%（**正面**）

### 5.3 `Landscape.land`（景观破碎化指标）

```json
"Landscape": {
    "land": {
        "best_plan": {
            "LPI": "38.4338",      // Largest Patch Index
            "LSI": "5.2559",       // Landscape Shape Index
            "AI": "94.2003",       // Aggregation Index
            "ED": "1329.9561",     // Edge Density
            "SHEI": "0.7186",      // Shannon's Evenness Index
            "SHDI": "1.3983",      // Shannon's Diversity Index
            "CONTAG": "55.8808"    // Contagion Index
        },
        "ori_plan": { ... },
        "tgt_plan": { ... }
    }
}
```

- ⚠️ **存储为字符串**，使用时需 `float()` 转换
- 各指标方向不同（部分越大越好，部分越小越好，没有统一）
- 在 ablation 分析中**未直接使用**，作为辅助参考
- 由 fragstats 外部软件计算（若 fragstats 不可用，此字段为空 `{}`）

### 5.4 `FINAL`（最终评估）

```json
"FINAL": {
    "FOM": 0.4307,            // Figure of Merit
    "kappa": 0.8728,          // Cohen's Kappa
    "best_ori": { "<lu>": { "ori_plan": ..., "best_plan": ..., "change_hectares": ..., "change_ratio": ... } },
    "best_tgt": { "<lu>": { "tgt_plan": ..., "best_plan": ..., "change_hectares": ..., "change_ratio": ... } }
}
```

| 字段 | 方向 |
|------|------|
| `FOM` | [0, 1]，**越大越好** |
| `kappa` | [-1, 1]，**越大越好** |
| `best_ori.<lu>.change_hectares` | 可正可负（绝对量，无方向） |
| `best_ori.<lu>.change_ratio` | 可正可负（比例，无方向） |

---

## 6. 完整数据流图

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  原始计算 (cal_*_score 函数返回值)                                  │
 │  ─────────────────────────────────                                  │
 │  change_relative_score  → 负值, 越小越好                            │
 │  es_relative_score      → 正值[0,1], 越大越好                       │
 │  landscape_score        → 一般正值, 越小越好                        │
 └─────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  cal_tgt_function 返回的 score_list                                 │
 │  ─────────────────────────────────                                  │
 │  [tgt_func, change, es, es_dict, landscape]                         │
 │   ↓负     ↓负   ↓正  ↓字典     ↓正                                  │
 └─────────────────┬───────────────────────────────────────────────────┘
                   │
                   ├──── SA 分支 ──────────────────────────────┐
                   │                                            │
                   ▼                                            │
 ┌──────────────────────────────────────────────────┐          │
 │  SA.run() 循环, evaluate_new_plan 用 tgt_func    │          │
 │  最小化判定                                       │          │
 │  best_score = min(tgt_func), 越小越好            │          │
 └─────────────────┬────────────────────────────────┘          │
                   │                                            │
                   ▼                                            │
 ┌──────────────────────────────────────────────────┐          │
 │  update_score_lists 写入内存列表                  │          │
 │  ────────────────────────────                    │          │
 │  total_score_list  ← tgt_func   (原值, 越小越好) │          │
 │  change_score_list ← -change    (取反, 越大越好) │          │
 │  es_score_list     ← es         (原值, 越大越好) │          │
 │  landscape_*_list  ← -landscape (取反, 越大越好) │          │
 │  es_dict[type]     ← es_weighted (原值, 越大越好)│          │
 └─────────────────┬────────────────────────────────┘          │
                   │                                            │
                   │     ┌──── GA 分支 ─────────────────────────┘
                   │     ▼
                   │   ┌──────────────────────────────────────────────────┐
                   │   │  GA.fitness() 内部把 tgt_func 取反 → fitness     │
                   │   │  selection / crossover 用 fitness 最大化         │
                   │   │  best_score = max(fitness), 越大越好            │
                   │   └─────────────────┬────────────────────────────────┘
                   │                     │
                   │                     ▼
                   │   ┌──────────────────────────────────────────────────┐
                   │   │  update_score_lists 写入内存列表                  │
                   │   │  ────────────────────────────                    │
                   │   │  total_score_list  ← fitness    ⚠️ 取反过的 tgt │
                   │   │                                  (越大越好)      │
                   │   │  change_score_list ← -change    (取反, 越大越好)│
                   │   │  es_score_list     ← es         (原值, 越大越好)│
                   │   │  landscape_*_list  ← -landscape (取反, 越大越好)│
                   │   └─────────────────┬────────────────────────────────┘
                   │                     │
                   └─────────────────────┴────────────┬─────────────────────┐
                                                      │                     │
                                                      ▼                     ▼
 ┌─────────────────────────────────────┐  ┌────────────────────────────────────┐
 │  draw_score_list                    │  │  _dump_convergence                  │
 │  charts/*.png                       │  │  ──────────────────                 │
 │  ────────────                       │  │  total_score_list.npy:             │
 │  total_score.png:                   │  │     SA: tgt_func 序列 (越小越好)   │
 │     SA: y=tgt_func, 下降=改善       │  │     GA: fitness 序列  (越大越好)   │
 │     GA: y=fitness,  上升=改善 ⚠️    │  │                                     │
 │  change_score.png: y=−change, ↑=好  │  │  convergence.json:                 │
 │  ES_score.png:     y=es,      ↑=好  │  │     final_best:                    │
 │  landscape_score:  y=−ls,     ↑=好  │  │       SA: best_score (负, 越小越好)│
 │  ES详情.png:       4 条 ES 加权曲线 │  │       GA: -best_fitness            │
 │                                     │  │            (再取反, 与 SA 同向)    │
 │                                     │  │     es_score: es_score_list[best]  │
 │                                     │  │            (正, [0,1], 越大越好)   │
 │                                     │  │     iter_to_95pct, wall_time, ...  │
 │                                     │  │            (正, 越小越好)          │
 └─────────────────────────────────────┘  └────────────────────────────────────┘
                                                      │
                                                      ▼
                            ┌──────────────────────────────────────────────────┐
                            │  save_plan_and_get_performance                    │
                            │  → write_performance_score                        │
                            │  performance.json (独立评估, 不来自优化历史)     │
                            │  ──────────────────────────────────              │
                            │  Change_area.*.iou/f1/precision/recall:           │
                            │     [0,1], 越大越好                              │
                            │  ES.<type>.best_plan/tgt_plan/ori_plan:           │
                            │     绝对值, 无方向                                │
                            │  ES.<type>.best_ori_diff:                         │
                            │     正=ES 提升, 负=ES 下降, **越大越好**          │
                            │  ES.<type>.best_tgt_diff:                         │
                            │     **趋近 0 越好**                              │
                            │  Landscape.land.*.{LPI,LSI,AI,...}:               │
                            │     字符串数字, 方向各异                          │
                            │  FINAL.FOM:    [0,1], 越大越好                   │
                            │  FINAL.kappa:  [-1,1], 越大越好                  │
                            │  FINAL.best_ori.<lu>.change_*:                    │
                            │     绝对量/比例, 无方向                           │
                            └──────────────────────────────────────────────────┘
```

---

## 7. 实测样例对照（来自 tornado A_baseline / 样本 10_1723_5 / run_0）

### `convergence.json`

```json
{
  "iter_to_95pct": 2070,        // 正整数，越小越好
  "final_best": -20666445.23,   // 负值，越小越好（= SA 的 best_score = 最优 tgt_func）
  "es_score": 0.77,             // [0,1]，越大越好（取自 es_score_list 在最优 iter 处）
  "wall_time_sec": 1188.5,      // 正实数，越小越好
  "total_iter": 13813,
  "guided": true,
  "disable_early_stop": false
}
```

### `performance.json` 关键字段

```json
{
  "Change_area": {
    "real_metrics": {
      "iou": 0.848,             // [0,1] 越大越好
      "f1_score": 0.917
    }
  },
  "ES": {
    "food":   {"best_ori_diff": -0.252, "best_tgt_diff": -0.021},  // 食物 ES 下降 25.2% （恶化）
    "carbon": {"best_ori_diff": -0.051, "best_tgt_diff": -0.003},  // 碳汇 ES 下降 5.1%
    "water":  {"best_ori_diff": -0.213, "best_tgt_diff": -0.021},  // 水源 ES 下降 21.3%
    "culture":{"best_ori_diff": +0.030, "best_tgt_diff": +0.002}   // 文化 ES 提升 3.0% （改善）
  },
  "FINAL": {
    "FOM": 0.43,                // 越大越好
    "kappa": 0.87               // 越大越好
  }
}
```

⚠️ 这意味着对此样本（10_1723_5），baseline 优化结果**牺牲了大部分 ES 子项**（food/carbon/water 都下降）以换取目标函数最小化（change_score 改善 + 景观改善的总效应主导）。**这是因为 `es_relative_score` 在目标函数里被 λ_ES（baseline 默认 1000~5000）放大，但 change_score 的奖励 (`inside_reward_coef × area²`) 可能更大。**

---

## 8. 常见混淆点 & 排查清单

| 混淆点 | 怎么避免 |
|--------|---------|
| 看 `charts/total_score.png` 是 SA 还是 GA | SA: 应下降；GA: 应上升。看 manifest.json 的 `runner_type` |
| 看 `convergence.json` 的 `final_best` | 永远是 tgt_func 同向，**负值越小越好**（GA 已在写盘时取反过） |
| 看 `convergence.json` 的 `es_score` | 永远是 [0, 1]，**越大越好** |
| 看 `total_score_list.npy` 计算 iter_to_95pct | **必须根据 runner_type 判断方向**（参考 `recompute_iter_to_95pct()`） |
| 看 SA 控制台 `text` 中的 `ES_score` | text 里加了负号（line 193），**显示为负值，但 list 里存的是正值**——这是 bug 性的不一致 |
| 看 SA 控制台 `visualize_frequency` 分支的打印 | 直接打印 `cur_score_list[2]` 原值（正），**与 text 又不一致** |
| `performance.json` 的 `ES.*.best_ori_diff` | 越大越好（正=ES 改善），与 tornado 分析中"越大越好"一致 |
| `performance.json` 的 `ES.*.best_tgt_diff` | 趋近 0 越好（与真实未来方案的接近度） |

---

## 9. 建议的代码改进点（可选）

1. **统一 `total_score_list.npy` 的方向**：建议 GA 在写盘时也将列表取反，让 SA/GA 落盘的 `total_score_list.npy` 都是"越小越好"的 tgt_func，避免下游分析需要看 `runner_type` 才能正确处理。

2. **修复 SA 控制台 `ES_score` 显示**：`SA.py:193` 的 `f'ES_score={-es_score:.4f}'` 应改为 `f'ES_score={es_score:.4f}'`，与 list 中存的方向一致。

3. **添加方向注释**：在每个 `update_score_lists` 函数开头用 docstring 明确每个 list 的方向，避免后续修改踩坑。

4. **可考虑 `charts/total_score.png` 统一方向**：例如 GA 也画 `-fitness` 即 `tgt_func`，让两个算法图表方向一致。
