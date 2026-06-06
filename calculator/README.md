# Calculator Module

## 这个模块做什么

项目分两阶段工作：Stage-1 用 ANN（AttU-Net/U-Net 等）预测变化区域，输出 `pred_diff`；Stage-2 用 SA/GA 在此先验下搜索最优 LU 方案。calculator 就是 Stage-2 的**评分器**——每当 SA/GA 生成一个候选方案，calculator 就计算一个标量得分 `tgt_func`，告诉优化器这个方案有多好。优化器通过反复调用 calculator 来指导搜索方向。

calculator 不生成方案，只评判方案。方案的生成在 `generator/`，搜索策略在 `runner/`。

## 目标函数

优化一个 LU 方案需要同时满足三个目标：

- **变化区域应靠近 ANN 预测的位置** → change_score（越小越好）
- **生态系统服务应尽可能高** → es_score（越大越好）
- **景观格局应尽量连贯** → landscape_score（越小越好）

三者方向不同，统一为最小化问题：

```
tgt_func = change_score − λ_ES · es_score + λ_LS · landscape_score
```

- change_score 正号：越小越好 → 减小 tgt_func
- es_score 取负：越大越好 → 减小 tgt_func
- landscape_score 正号：越小越好 → 减小 tgt_func
- SA 直接最小化 tgt_func；GA 适应度 = -tgt_func（最大化）

λ_ES 和 λ_LS 不是定值，而是根据 pred_diff 的像素数自适应调整（见第 7 节）。

## 数据流：从 pred_diff 到最终得分

```
每次 SA/GA 迭代：
  1. generator 生成候选方案 cur_plan（基于扰动/交叉/变异）
  2. 计算 cur_diff = (cur_plan ≠ ori_plan) 的变化区域
  3. calculator 评估：
     - cur_diff 与 pred_diff 的匹配程度 → change_score
     - cur_plan 的生态系统服务总量 → es_score
     - cur_diff 区域的景观破碎度 → landscape_score
     - 三项加权 → tgt_func
  4. 优化器根据 tgt_func 决定接受/拒绝/保留该方案
```

`pred_diff` 是 Stage-1 ANN 模型的输出，代表"模型认为哪里会变"。calculator 将它作为变化得分的先验——如果候选方案的变化区域恰好落在预测位置，就获得高分；如果偏移很远，就被惩罚。

## 变化得分 change_score

**核心思想**：候选方案的变化区域越靠近 ANN 预测的变化区域，得分越高。

算法步骤：

1. 用 `scipy.ndimage.label` 标记 `cur_diff` 的连通区域
2. 对每个连通区域，计算它与 `pred_diff` 的位置关系：

| 位置关系         | 累加项                                                   | 说明                |
| ---------------- | -------------------------------------------------------- | ------------------- |
| 完全在预测区域内 | `− area² · inside_reward_coef`                           | 越大越奖励          |
| 部分重叠         | `+ diff_area² · distance − ∩_area² · inside_reward_coef` | 差集惩罚 + 交集奖励 |
| 完全不重叠       | `+ area² · distance`                                     | 距离越远罚越重      |

3. 所有连通区域的累加结果即为变化得分

面积取平方意味着大区域被更强地奖惩。`inside_reward_coef`（默认 10）控制重叠奖励的强度。

关键参数：

- `inside_reward_coef`：重叠奖励倍率，来自 `configs/score_cfg.yaml`
- `pred_diff_distance_map`：pred_diff 的欧几里得距离变换图（EDT），由 `utils/calculator_utils` 预计算存入 info_dict

注：代码中有归一化行 `change_value / min_change_score`，但已被注释掉，当前返回绝对值。

## 生态系统服务得分 es_score

**核心思想**：方案的生态系统服务相对于理论最大值的比例。

### 总公式

```
es_score = Σ weight[k] · (value[k] / max_score[k])
```

每项 ES 先除以自己的理论最大值归一化到 [0, 1]，再加权求和。权重由 `es_weight_dict` 指定（如 `{'food': 0.25, 'carbon': 0.25, 'water': 0.25, 'culture': 0.25}`）。

### 四项 ES

| 名称            | 模型/公式                                                                                                 | 仅计算特定 LU | 单位       |
| --------------- | --------------------------------------------------------------------------------------------------------- | ------------- | ---------- |
| food（粮食）    | Miami + Thornthwaite Memorial，Liebig 最小定律 `W = min(WT, WR, WV)`，再乘 food_supply_param 复合修正系数 | 耕地(11, 12)  | kg/年      |
| carbon（碳汇）  | carbon_pool.csv 查表，4 库(above/below/soil/dead) × 面积求和                                              | 全部 LU       | 吨         |
| water（水源）   | `Q = P − P·R − ET`，R 由 LU 类型查径流系数表                                                              | 按 LU 二级类  | m³/年      |
| culture（文化） | 预训练 MaxEnt 模型(elapid)预测文化服务适宜性                                                              | —             | 适宜性总和 |

- **food** 的复合修正系数 `food_supply_param` 由五因子（坡度PODU、距农地GD、距水系WATER、距道路DL、距城镇CITY）分桶评分后加权得到，仅耕地像素累加
- **carbon** 的碳密度来自 `libs/carbon_pool.csv`，每种 LU 类型对应 above/below/soil/dead 四个碳库
- **water** 的径流系数 R 来自 `libs/const.WATER_SUPPLY_PARAM`，按 LU 二级类精确查表（林地/草地内部按覆盖度区分）
- **culture** 的 MaxEnt 模型使用 `maxent_list` 指定的特征列表，其中 `'LU'` 特征使用当前方案的 LU 类型

### 归一化基准（理论最大值）

`get_es_max_score` 的策略：将所有变化像素替换为对该 ES 最有利的 LU 类型，计算出的值即为最大值。

| ES      | 替换为         | 理由                            |
| ------- | -------------- | ------------------------------- |
| carbon  | 林地(21)       | 森林碳密度最高                  |
| food    | 水田(11)       | 水田粮食产量最高                |
| water   | 河渠(41)       | 水体水源涵养最强                |
| culture | 原始方案 × 1.5 | 无明确最优 LU，用放大系数设上限 |

## 景观格局得分 landscape_score

**核心思想**：变化区域内的 LU 类型越单一、斑块越少、离同类原始斑块越近，得分越低（越好）。

```
landscape_score = total_cls_distance + total_pure_score · (pure_base + change_ratio · pure_change)
```

两个组件：

### total_pure_score（纯度惩罚）

遍历变化区域的每个连通区域，计算：

```
pure_score = LU类型数 × 斑块总数 × (Shannon熵 + 1)
```

- 类型越单一、斑块越少 → 分越低 → 越好
- Shannon 熵加 1 作为权重：类型越均匀混合，惩罚越重

### total_cls_distance（空间距离惩罚）

对每个 LU 类，找到变化区域内该类的连通区域，查 `ori_lu_distancemap_dict`（该类在原始方案中的 EDT 距离图）：

- **邻接同类原始斑块**（min_distance=0）→ 奖励 `−= 10/ratio`，其中 ratio = 周长/面积
- **孤立**（min_distance>0）→ 惩罚 `+= (min_distance + max_distance) · ratio²`
- 连通区域数比原始方案增加 → 额外惩罚 `+= delta_num²`

默认 `use_landscape=False`，景观项是可选的（计算开销较大，且部分场景下变化区域形态不关键）。

## λ 自适应调度

pred_diff 的像素数 n 变化很大（几十到上万）。n 越大，单像素 ES 贡献越分散，ES 项容易被淹没；n 越小，每个像素的 ES 影响很大，需要更大的 λ_ES 来平衡。

| pred_diff 像素数 n | λ_ES     | λ_LS     | 场景                                      |
| ------------------ | -------- | -------- | ----------------------------------------- |
| n > 6000           | 1000     | 5        | 大面积变化，ES 项权重较低，景观项权重较高 |
| n < 300            | 5000     | 1        | 小面积变化，ES 项权重高，景观项不太重要   |
| 300 ~ 6000         | 线性插值 | 线性插值 | 中间状态平滑过渡                          |

插值公式：

```
ratio = (n - pixel_low) / (pixel_high - pixel_low)
λ_ES  = lambda_ES_low  + (lambda_ES_high  - lambda_ES_low)  · ratio
λ_LS  = lambda_LS_low  + (lambda_LS_high  - lambda_LS_low)  · ratio
```

`change_scale_n` 参数可覆盖像素数（用于 C 组消融实验，此时优化目标不是 ANN 预测区域而是随机约束，但 λ 仍需基于模型预测的像素数取值）。

## 配置

所有可调参数在 `configs/score_cfg.yaml` 中：

| 配置路径                          | 默认值 | 作用                                   |
| --------------------------------- | ------ | -------------------------------------- |
| `tgt_function.lambda_ES_high`     | 1000   | n > pixel_high 时 ES 权重              |
| `tgt_function.lambda_ES_low`      | 5000   | n < pixel_low 时 ES 权重               |
| `tgt_function.lambda_LS_high`     | 5      | n > pixel_high 时景观权重              |
| `tgt_function.lambda_LS_low`      | 1      | n < pixel_low 时景观权重               |
| `tgt_function.pixel_high`         | 6000   | λ 分段上界                             |
| `tgt_function.pixel_low`          | 300    | λ 分段下界                             |
| `change_score.inside_reward_coef` | 10     | 变化区域落入 pred_diff 内的奖励倍率    |
| `landscape.pure_base`             | 0.3    | pure_score 基准权重                    |
| `landscape.pure_change`           | 0.2    | pure_score 随 change_ratio 缩放系数    |
| `ablation.disable_change_score`   | false  | 屏蔽 change_score 项（无引导消融实验） |

## 与其他模块的协作

calculator 的运行依赖大量预计算数据，这些数据在优化开始前由 `utils/calculator_utils.get_info_dict()` 一次性构建：

| info_dict 键                   | 用途                         | 来源                             |
| ------------------------------ | ---------------------------- | -------------------------------- |
| `pred_diff_distance_map`       | change_score 计算距离惩罚    | 对 pred_diff 做 EDT              |
| `ori_lu_distancemap_dict`      | landscape_score 中同类距离   | 对 ori_plan 各 LU 类做 EDT       |
| `ori_num_feature_list`         | landscape_score 中连通数基准 | 统计 ori_plan 各类连通数         |
| `food_supply_param`            | food 的复合修正系数          | PODU/GD/WATER/DL/CITY 五因子加权 |
| `carbon_pool`                  | carbon 查表                  | `libs/carbon_pool.csv`           |
| `maxent_model` + `maxent_list` | culture 预测                 | 预训练 MaxEnt 模型               |
| `NJW`, `RAIN`, `ZSL`           | food/water 的气温降水        | data_dict（驱动因子）            |
| `max_es_score`                 | ES 归一化基准                | `get_es_max_score`               |
| `min_change_score`             | change 归一化基准（未启用）  | `get_change_min_score`           |

调用关系：

- **`runner/SA.py`、`runner/GA.py`**：每次迭代调用 `cal_tgt_function`
- **`utils/calculator_utils.py`**：优化前调用 `get_change_min_score`、`get_es_max_score` 构建 info_dict
- **`utils/performance.py`**：事后评估调用 `cal_es_value`、`get_es_cal_func_dict`

## 已知问题

- `cal_change_score` 中归一化行已注释，旧注释 "范围:[-inf, 1]" 与当前实际返回值不符
