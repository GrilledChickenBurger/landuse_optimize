# Generator Module

## 这个模块做什么

generator 是 SA/GA 优化器的**方案生成层**。如果说 calculator 是"评分裁判"（评判方案好坏），generator 就是"出题者"（创造候选方案）。每次 SA/GA 迭代中，optimizer 调用 generator 生成一个新的候选 LU 方案，然后交给 calculator 打分，优化器再根据分数决定接受还是拒绝。

SA 和 GA 共享同一套矩形区域修改策略——两者的底层操作（在方案上随机选一块矩形区域，统一改为某个 LU 类型）完全相同，区别在于**上层控制逻辑**：SA 每次做一步扰动，GA 则操作整个种群（初始化、交叉、变异）。

## 模块在项目中的位置

```
SA/GA runner 的每次迭代：

  ┌─────── runner ───────────────────────────────────────────┐
  │                                                          │
  │     generator                    calculator              │
  │   ┌─────────────┐             ┌──────────────┐           │
  │   │ 生成候选方案  │  new_plan  │ 计算目标函数  │  tgt_func │
  │   │ cur_plan    │ ────────→  │ change_score │ ────────→ │
  │   │ d_diff      │            │ es_score     │           │
  │   └─────────────┘            │ landscape    │           │
  │                              └──────────────┘           │
  │                                                          │
  │   SA: 按 tgt_func 决定接受/拒绝，更新温度                   │
  │   GA: 按 fitness=-tgt_func 选择、保留精英                  │
  └──────────────────────────────────────────────────────────┘
```

generator 只关心"怎么生成新方案"，不关心"新方案好不好"——后者是 calculator 的事。

## 核心操作：矩形区域修改

所有方案修改都基于同一个原子操作：**在 128×128 的土地利用图上随机选一个矩形区域，将区域内所有像素统一改为某个 LU 类型**。

### 为什么是矩形

- 简单可控，面积可以通过 scale 参数精确约束
- 实际 LU 规划中，变更通常以斑块（patch）为单位，矩形是斑块的合理近似
- 便于施加约束（只需检查矩形内像素是否可修改）

### 两种基本操作

| 操作 | 函数 | 效果 |
|---|---|---|
| change | `area_wise_change` | 将矩形区域统一改为随机 LU 类型 |
| return | `area_wise_return` | 将矩形区域**还原**为原始方案的 LU 类型 |

change 创造新方案，return 部分撤销之前的修改，帮助逃离局部最优。

### 矩形大小分档

矩形面积分 small / mid / large 三档，由 `scale_minmax_dict` 控制（默认值来自 `configs/optimizerConfig.yaml`）：

| 档位 | 面积范围（像素） | 边长约 | 占全图比例 |
|---|---|---|---|
| small | 100 ~ 200 | 10 ~ 14 | 0.6% ~ 1.2% |
| mid | 200 ~ 400 | 14 ~ 20 | 1.2% ~ 2.4% |
| large | 400 ~ 800 | 20 ~ 28 | 2.4% ~ 4.9% |

档位是名义上的——矩形实际面积由随机采样决定，只落在范围内即可。

## SA 的方案生成

SA 每次迭代调用 `generate_new_plan` 做一步扰动。文件：`sa_generator.py`。

### generate_new_plan

```python
def generate_new_plan(cur_plan, ori_plan, T_ratio, noimprove_step,
                      scale_minmax_dict, lu_labels, cls_restrict: torch.Tensor, hard_restrict: torch.Tensor)
    → (new_plan, d_diff)
```

**算法步骤：**

1. **计算四种操作的概率**，基于当前温度比 `T_ratio`（当前温度/初始温度）和连续无改进步数 `noimprove_step`：

   - `large_prob = 0.4 × T_ratio + 0.2` — 高温偏大区域，低温时降至 0.2
   - 正常情况：`return_prob = 0`，`mid_prob = 0.3`，`small_prob = 1 − 0.3 − large_prob`
   - **停滞时**（`noimprove_step ≥ 30` 或 `T_ratio ≤ 0.1`）：`return_prob = 0.2`，`mid_prob = 0.2`，引入 return 操作帮助跳出局部最优

2. **按概率采样操作类型**，从 `['small', 'mid', 'large', 'return']` 中选择

3. **执行操作**：如果是 change → 调用 `area_wise_change`；如果是 return → 调用 `area_wise_return`

4. **返回** `(new_plan, d_diff)`，其中 `d_diff` 是本次修改像素的二进制掩码

设计意图：高温期大范围探索，低温期精细调整，陷入停滞时通过 return 退回原始方案重新出发。

### area_wise_change

```python
def area_wise_change(cur_plan, scale, scale_minmax_dict, lu_labels,
                     cls_restrict: torch.Tensor, hard_restrict: torch.Tensor)
    → (new_plan, d_change_area)
```

**执行一次矩形修改，步骤：**

1. **确定面积边界**：从 `scale_minmax_dict[scale]` 读取 `min_area` / `max_area`

2. **寻找合法矩形**（最多 10000 次尝试）：
   - 随机选取起始点 `(start_x, start_y)`
   - 随机采样边长（在面积边界对应的边长范围内）
   - **校验 `hard_restrict`**：矩形内每个像素的 `hard_restrict` 必须为 1（可修改），否则拒绝该矩形
   - **校验非零**：矩形内所有像素的 LU 类型不能为 0（背景/无数据）

3. **失败降档**：10000 次都没找到合法矩形时，如果当前不是 small 档，降一档重试（large→mid→small）；如果已经是 small 且还找不到，返回原始方案（不做修改）并打印警告

4. **选择新 LU 类型**：
   - 如果矩形内**任一像素**的 `cls_restrict == 1`（城镇用地区域），则新类型只能从 `[51, 52, 53]`（建设用地）中选择
   - 否则从 `lu_labels` 中除去 0 后均匀随机选取

5. **应用修改**：将矩形区域全部赋值为选中的 LU 类型，记录修改掩码 `d_change_area`

### area_wise_return

```python
def area_wise_return(cur_plan, ori_plan, T_ratio)
    → (new_plan, d_diff)
```

将矩形区域**还原**为原始方案：

- 还原面积 = `max(100 × 0.05 / T_ratio, 100)`，温度越低还原面积越大——低温时更需要通过退回原方案来跳出局部最优
- 随机选取矩形，用 `ori_plan` 的对应区域覆盖 `cur_plan`
- **不检查** `hard_restrict` 和 `cls_restrict`（因为还原到原始方案本身就是约束合规的）

## GA 的方案生成

GA 操作一个种群而非单点。文件：`ga_generator.py`，依赖 `sa_generator` 的底层操作。

### initial_new_population

```python
def initial_new_population(pop_size, scale_ratio, ori_plan,
                           scale_minmax_dict, lu_labels,
                           cls_restrict, hard_restrict)
    → List[Tensor]
```

**生成初始种群：**

1. 根据 `scale_ratio`（如 `[0.5, 0.3, 0.2]`）将种群名额分配到 small / mid / large 三个档位
2. **原始方案始终作为第一个个体**保留（精英种子）
3. 其余个体：从原始方案出发，对其**连续调用 2 次** `area_wise_change`（通过 `utils.tools.repeat`），每次选择一个随机矩形修改
4. 返回 `pop_size` 个方案张量，每个 `[1, 1, 128, 128]`

每个初始个体已经在原始方案基础上做了两个矩形修改。

### crossover

```python
def crossover(parent1, parent2, gen_ratio, hard_restrict,
              cls_restrict=None, ori_plan=None, max_ratio=0.8, min_ratio=0.2)
    → [child1, child2]
```

**单点矩形交换：**

1. 计算当前交叉矩形比例：`current_ratio = max_ratio × (1 − gen_ratio) + min_ratio × gen_ratio`，随着代数推进（`gen_ratio` 从 0 → 1），`current_ratio` 从 `max_ratio` 线性缩小到 `min_ratio`，即**早期交换大块，后期交换小块**

2. **单次**选取矩形位置和尺寸（边长在 `[5, H × current_ratio]` 和 `[5, W × current_ratio]` 范围内），不做拒绝采样重试

3. **校验 `cls_restrict`**（如果提供）：对交换矩形内 cls_restrict==1 的像素，检查 incoming 值是否在 [51,52,53]（建设用地）中。若 incoming 值不在允许范围内，该像素保留原值不交换，其他像素照常交换

4. 交换两个父本在矩形区域内的像素 → 产生两个子代

5. **可行性检查**（提供 `ori_plan` 时）：子代若违反硬约束（改了 `hard_restrict==0` 的像素）或类别约束（51/53 改为非 5 类），则该子代**直接丢弃**、回退为对应父本（单次生成、不重试）

### mutate

```python
def mutate(individual, **kwargs)
    → mutated_individual
```

**变异实现为 SA 扰动链**：

```python
mutated = repeat(generate_new_plan, repeat_num=2,
                 replace_element=individual, **kwargs)
```

对个体连续调用 2 次 `generate_new_plan`（每次可能做 change 或 return），本质上是复用 SA 的扰动逻辑。

GA 传给 `generate_new_plan` 的 `T_ratio = max(0.0001, 1 − gen_ratio)`，即：
- **早期**（`gen_ratio ≈ 0`）→ `T_ratio ≈ 1` → 偏向大范围变异 → 探索
- **晚期**（`gen_ratio ≈ 1`）→ `T_ratio ≈ 0.0001` → 偏向小范围变异 → 精细调整

这与 SA 相反：SA 的 T_ratio 是温度自然下降，GA 的 T_ratio 是按代数计算。

`noimprove_step` 在 GA 中默认传入 0（不触发 return），GA 有自己独立的停滞处理逻辑（在 runner 层）。

## 约束集成

三层约束在 generator 各函数中的覆盖情况：

| 约束 | `area_wise_change` | `area_wise_return` | `crossover` | `mutate` | `initial_new_population` |
|---|---|---|---|---|---|
| hard_restrict | ✓ 矩形逐像素校验 | ✗ | ✓ 矩形逐像素校验 | ✓ (继承自 area_wise_change) | ✓ (继承自 area_wise_change) |
| cls_restrict | ✓ 标签筛选 | ✗ | ✓ 像素级选择性交换 | ✓ (继承自 area_wise_change) | ✓ (继承自 area_wise_change) |
| scale_minmax | ✓ 面积边界 | ✗ | ✗ (自有机制) | ✓ (继承) | ✓ (继承) |

**约束覆盖说明：**

- `area_wise_return` 不检查任何约束——但它只还原到原始方案，原始方案本身就是约束合规的
- `crossover` 检查 `hard_restrict`（矩形校验）+ `cls_restrict`（像素级：cls_restrict==1 的像素只允许交换入 [51,52,53] 类型的值，不合规则保留原值不交换）
- `mutate` 和 `initial_new_population` 间接依赖 `area_wise_change`，约束覆盖完整

`hard_restrict` 的语义：1 = 可修改，0 = 不可修改。`cls_restrict` 的语义：1 = 该像素是 51/53 类型，只能改为 5 类（建设用地），不能改为耕地/林地/水体等。

## SA vs GA 生成方式对比

| 维度 | SA | GA |
|---|---|---|
| 搜索方式 | 单点迭代，一次扰动一个方案 | 种群并行，同时维护多个方案 |
| 新方案来源 | `generate_new_plan`（1 次调用） | 初始化 + 交叉 + 变异 |
| 初始方案生成 | 直接用 `ori_plan` | `initial_new_population`（ori_plan + 多次 change） |
| 扰动强度控制 | 温度 `T_ratio`（概率自适应） | 代数 `gen_ratio`（T_ratio 反向计算） |
| 还原机制 | `area_wise_return`（停滞时触发） | 无显式还原（仅在 mutate 中可能触发 return） |
| 交叉 | 无 | `crossover`（单点矩形交换） |
| 底层修改操作 | `area_wise_change` / `area_wise_return` | 同左（`area_wise_change` + `generate_new_plan`） |

## 已知问题

（此前记录的三个问题已修复：crossover 已添加 cls_restrict 像素级检查、repeat 注释已与实现一致、crossover docstring 默认值已修正。）
