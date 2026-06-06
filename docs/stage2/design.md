# Stage 2 随机优化——设计思路

> 本文档说明 Stage 2（随机优化求解最优土地利用方案）的设计目标、目标函数构建、约束体系和优化器选择。

---

## 1. 任务定义

Stage 2 的目标是在给定驱动因子和初始土地利用方案的基础上，通过随机优化算法（SA/GA）搜索**最优土地利用配置方案**，使其在以下三个维度达到平衡：

1. **变化位置**：变化区域尽量靠近 Stage 1 模型预测的高概率变化区域
2. **生态系统服务（ES）**：最大化四项 ES 得分（粮食供给、碳储量、水源涵养、文化服务）
3. **景观破碎化**：变化区域内 LU 类型尽量单一、斑块紧凑

### 1.1 整体流程

```
驱动因子 + 历史LU → [Stage 1 ANN 模型] → pred_diff (变化概率图)
                                              ↓
pred_diff + 约束 + 原始LU → [Stage 2 SA/GA 优化器] → 最优LU方案
                                ↓
                        目标函数 = change_score - λ_ES × es_score + λ_LS × landscape_score
```

---

## 2. 目标函数

### 2.1 数学定义

```
F(S) = change_score - λ_ES × es_score + λ_LS × landscape_score
```

这是一个**最小化**问题：SA 直接最小化 F(S)；GA 适应度 = -F(S) 求最大化。

| 项 | 函数 | 范围 | 含义 |
|----|------|------|------|
| change_score | `cal_change_score()` | [-∞, 1] | 变化区域越靠近 pred_diff，得分越高（越大越好） |
| es_score | `cal_es_score()` | [0, 1] | 四项 ES 的加权归一化得分（越大越好） |
| landscape_score | `cal_landscape_score()` | [0, +∞] | 变化区域越破碎，得分越高（越小越好） |

**注意符号**：change_score 和 es_score 前面是负号（取反后最小化），landscape_score 前面是正号（直接最小化）。

### 2.2 λ 自适应调度

λ_ES 和 λ_LS 根据 pred_diff 的像素数量 n 动态线性插值：

```
n < 300 (pixel_low):   λ_ES = 5000, λ_LS = 1
n > 6000 (pixel_high): λ_ES = 1000, λ_LS = 5
中间: 线性插值
```

**设计意图**：变化区域较小时，ES 权重大（λ_ES=5000），优先保证生态服务；变化区域较大时，景观权重增大（λ_LS=5），控制破碎化。

### 2.3 子项计算

#### change_score（变化位置得分）

```python
cal_change_score(cur_diff, pred_diff, info_dict)
```

- `cur_diff`：当前方案与原始方案的差分图（当前变化区域）
- `pred_diff`：Stage 1 模型预测的变化区域
- 计算方法：对 pred_diff 做距离变换 → 连通区域距离惩罚 + 交集奖励（`inside_reward_coef`）

#### es_score（生态系统服务得分）

```python
cal_es_score(cur_plan, info_dict, weight_dict)  # 形参名 weight_dict，调用时传入 es_weight_dict
```

四项 ES 的计算：

| ES | 模型 | 输入 |
|----|------|------|
| food | Miami + Thornthwaite Memorial | 气温、降水 → 粮食生产潜力（仅耕地上累加） |
| carbon | 查表法 | LU 类型 × 碳密度 × 面积（`libs/carbon_pool.csv`） |
| water | 水量平衡 | `Q = P - P×R - ET`，不同 LU 有不同径流系数 |
| culture | MaxEnt 模型 | POI、距道路距离、LU 等 → 文化服务适宜性 |

#### landscape_score（景观破碎化得分）

```python
cal_landscape_score(cur_plan, cur_diff, ori_plan, info_dict)
```

- 变化区域内 LU 类型越单一 → 得分越低
- 斑块数越少 → 得分越低
- 离同类原始斑块越近 → 得分越低

---

## 3. 三层约束体系

### 3.1 hard_restrict（硬约束）

不可修改的区域，`0 = 受限，1 = 可修改`：

| 约束来源 | 含义 |
|----------|------|
| `redline_restrict` | 耕地红线区域（RA=0） |
| `lu_zero_restrict` | LU=0 的区域（无数据区域） |
| `grid_restrict` | 预测无变化区域的网格化约束（仅 A 组消融实验） |

最终：`hard_restrict = redline ∧ lu_zero`（主流程）或 `hard_restrict = redline ∧ lu_zero ∧ grid_restrict`（消融 A 组）

### 3.2 cls_restrict（类别约束）

特定 LU 类型的转换规则：

- **类别 51、53（城镇用地）**：只能改成建设用地（类别 5），不能改为耕地/林地等
- 通过 `generate_new_plan()` 中的类别过滤实现

### 3.3 scale_minmax（扰动尺度约束）

根据 pred_diff 的像素数动态调整每次扰动的矩形区域面积范围：

```python
scale_minmax = get_scale_minmax_dict(pred_diff_pixelnum)
# small: [1, 0.25 × n]
# mid:   [0.25n, 0.5n]
# large: [0.5n, n]
```

---

## 4. 优化器

### 4.1 SA（模拟退火）

**配置参数**（`configs/optimizerConfig.yaml` → `SA`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| T_start | 100 | 初始温度 |
| T_end | 0.1 | 终止温度 |
| cooling_rate | 0.9995 | 指数降温系数 |
| max_noimprove_iter | 1000 | 连续无改进最大迭代数 |
| max_beststuck_iter | 3000 | 最优解停滞最大迭代数 |

**新方案生成策略**（`generator/sa_generator.py`）：

- **小范围变化**：随机选小矩形，改为随机 LU 类型
- **中范围变化**：随机选中等矩形
- **大范围变化**：随机选大矩形
- **return 操作**：将区域恢复为原始方案

概率随温度自适应：高温时倾向大范围变化，低温时倾向小范围和 return。

### 4.2 GA（遗传算法）

**配置参数**（`configs/optimizerConfig.yaml` → `GA`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ori_pop_size | 30 | 初始种群大小 |
| final_pop_size | 8 | 最终种群大小 |
| max_gen | 3000 | 最大迭代代数 |
| crossover_rate | 0.8 | 交叉率 |
| mutation_rate | 1.0 | 变异率 |

**策略模式**（`GeneticAlgorithmStrategy`）：

系统支持 7 种策略变体（当前代码中仅 GA2/GA3 变体处于激活状态，其余在 `pipelines/experiment.py` 中注释）：

| 变体 | 父本选择 | 父本来源 | 精英比例 | 未交叉处理 | 精英变异 | 激活状态 |
|------|---------|---------|---------|-----------|---------|---------|
| GA1 (基准) | 轮盘赌 | 全部种群 | 10% | 直接复制 | 否 | 已注释 |
| GA2 | 锦标赛(k=2) | 全部种群 | 10% | 直接复制 | 否 | ✅ 激活 |
| GA3 | 轮盘赌 | 前20%精英 | 10% | 直接复制 | 否 | ✅ 激活 |
| GA4 | 轮盘赌 | 全部种群 | 0% | 直接复制 | 否 | 已注释 |
| GA5 | 轮盘赌 | 全部种群 | 50% | 直接复制 | 否 | 已注释 |
| GA6 | 轮盘赌 | 全部种群 | 10% | 复制后变异 | 否 | 已注释 |
| GA7 | 轮盘赌 | 全部种群 | 10% | 直接复制 | 是 | 已注释 |

---

## 5. 利益相关者情景

系统支持 4 种情景，通过调整 ES 权重实现：

| 情景 | food | carbon | water | culture | 适用场景 |
|------|------|--------|-------|---------|---------|
| default | 0.25 | 0.25 | 0.25 | 0.25 | 综合多功能景观 |
| farmer | 0.80 | 0.10 | 0.10 | 0.00 | 粮食安全导向 |
| enterprise | 0.00 | 0.50 | 0.50 | 0.00 | 低碳生态工业区 |
| tourist | 0.00 | 0.00 | 0.00 | 1.00 | 生态旅游区 |

---

## 6. 与 Stage 1 的衔接

Stage 1 训练好的模型在 Stage 2 中的两个作用：

1. **生成 pred_diff**：`get_pred_diff(ANN, ori_plan, factors)` → 二值化（θ=0.5）→ 作为 change_score 的目标
2. **构造 grid_restrict**：pred_diff 的连通区域膨胀后作为搜索空间约束（消融实验 A 组）

```
Stage 1: best_model.pth
    ↓ Stage 2 加载
ANN model
    ↓ get_pred_diff()
pred_diff [1,1,128,128] (0.0 或 1.0)
    ↓
├── change_score 的目标区域
├── hard_restrict 的 grid_restrict（消融 A 组）
└── λ 调度的 pixel_num 参考
```

---

## 7. 输出与评估

### 7.1 优化输出

每个样本运行结束后保存：
- `best_plan.TIF`：最优 LU 方案
- `performance.json`：性能指标（ES 得分、IoU、变化区域统计）
- `convergence.json`：收敛信息（迭代次数、最优值、墙钟时间）
- `total_score_list.npy`：每步/每代的目标函数值序列

### 7.2 评估指标

| 指标 | 含义 |
|------|------|
| F(S) 最优值 | 最终解的目标函数值 |
| T95 | 达到 95% 最优值所需的迭代次数 |
| runtime | 墙钟运行时间（秒） |
| ES 得分 | 四项生态系统服务归一化得分 |
| IoU | 变化区域与 pred_diff 的交并比 |
