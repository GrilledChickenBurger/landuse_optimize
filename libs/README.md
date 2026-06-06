# Libs Module

## 这个模块做什么

libs 是项目的**配置与常量中枢**。土地利用分类体系、生态系统服务参数、颜色映射、驱动因子归一化参数、碳密度数据、数据增强策略、优化算法阈值——所有这些跨模块共享的"基础知识"都在这里统一定义。

libs 自身不包含任何算法逻辑，只提供数据和常量。它被项目中几乎所有模块导入：calculator（目标函数计算）、runner（SA/GA 优化器）、datasets（数据加载）、utils（工具函数）、pipelines（管道入口）。

## 在项目中的位置

```
libs/  ──→  calculator/  （从 libs 读取 PIXEL_AREA_IN_HECTARES、WATER_SUPPLY_PARAM 等计算 ES）
  │         runner/       （从 libs 读取 LU 标签、阈值控制优化流程）
  │         datasets/     （从 libs 读取 ALL_FACTORS、ALL_PACK 控制数据加载）
  │         utils/        （从 libs 读取 COLOR_MAP、FOOD_SUPPLY_PARAM 等辅助计算）
  │         pipelines/    （从 libs 读取 ALL_LOG_FACTORS 控制统计流程）
  └──────── scripts/      （从 libs 读取 COLOR_MAP 等用于可视化）
```

libs 不依赖任何项目内模块，只依赖 Python 标准库的 `os`。它是整个项目的**零依赖根节点**。

## 文件一览

| 文件 | 类型 | 用途 |
|---|---|---|
| `__init__.py` | Python | 对外门面，从 const 中选出 13 个最常用的符号统一暴露 |
| `const.py` | Python | 所有常量、参数、颜色映射、覆写函数的定义 |
| `carbon_pool.csv` | CSV | 27 种 LU 类型的碳密度查表（above/below/soil/dead 四库） |
| `frg_model.fca` | SQLite | FRAGSTATS 景观格局分析预配置模型 |
| `lu_legend.png` | PNG | LU 分类图例图片（1443×400），可视化时用作引用图例 |

## 土地利用分类体系

### 两级层次结构

项目使用两级 LU 分类：**6 个一级大类**（1-6）和 **27 个二级细类**（11-67）。一级类用于简化的聚合分析与可视化，二级类用于精确的优化计算。

| 一级类 | 名称 | 二级类 |
|---|---|---|
| 1 | 耕地 | 11=水田, 12=旱地 |
| 2 | 林地 | 21=有林地, 22=灌木林, 23=疏林地, 24=其他林地 |
| 3 | 草地 | 31=高覆盖度草地, 32=中覆盖度草地, 33=低覆盖度草地 |
| 4 | 水域 | 41=河渠, 42=湖泊, 43=水库坑塘, 44=永久性冰川雪地, 45=滩涂, 46=滩地 |
| 5 | 建设用地 | 51=城镇用地, 52=农村居民点, 53=其他建设用地 |
| 6 | 未利用地 | 61=沙地, 62=戈壁, 63=盐碱地, 64=沼泽地, 65=裸土地, 66=裸岩石质地, 67=其他 |

### 三个列表

`const.py` 中定义了三个与 LU 标签相关的列表/字典：

- **`ALL_LU_LABELS_SIMPLIFIED`** = `[1, 2, 3, 4, 5, 6]`：6 个一级大类，用于简化图可视化
- **`ALL_LU_LABELS`** = `[11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 46, 51, 52, 53, 65, 66]`：实际使用的 19 个二级类
- **`ALL_LU_LABELS_DICT`**：全部 27 个二级类的中文名称映射

注：`ALL_LU_LABELS` 与 `ALL_LU_LABELS_DICT` 的键不完全一致——44、61、62、63、64、67 在字典中存在但不在列表中（见"已知问题"）。

## 颜色映射

项目有三套颜色体系，分别用于不同场景的可视化：

### COLOR_MAP：完整 LU 颜色

将每个二级 LU 代码和 diff/non-diff 状态映射到 RGB 元组（值域 0-1）。22 个条目，同时支持 `int`（LU 代码）和 `bool`（变化状态）作为键：

- `0` / `False` → `(0, 0, 0)` 黑色（无变化区域）
- `1` / `True` → `(1, 1, 1)` 白色（变化区域）
- `11`~`67` → 各 LU 类型的专属颜色（如 11=浅黄、21=深绿、51=红色等）

### SIMPLIFIED_COLOR_MAP：简化 LU 颜色

将 6 个一级大类映射到 RGB 元组，用于聚合可视化和简化图。

### ES_COLOR_MAP：生态系统服务颜色

将 4 种 ES 类型映射到 RGB 元组：

| ES 类型 | 颜色 |
|---|---|
| food | 黄色 |
| carbon | 绿色 |
| water | 蓝色 |
| culture | 粉色 |

## 生态系统服务参数

### 水源涵养径流系数

`WATER_SUPPLY_PARAM` 定义了各 LU 二级类的地表径流系数 R，用于 water ES 计算中的 `Q = P − P·R − ET` 公式。**按二级类精确赋值**，林地和草地内部按覆盖度区分：

| LU 二级类 | 径流系数 | 说明 |
|---|---:|---|
| 11 水田 | 0.3470 | 中等径流 |
| 12 旱地 | 0.4969 | 旱地裸露面较多，径流较高 |
| 21 有林地 | 0.02038 | 几乎不产生径流（涵养能力最强） |
| 22 灌木林 | 0.0420 | 低径流 |
| 23 疏林地 | 0.1920 | 中等径流（覆盖度低） |
| 24 其他林地 | 0.1222 | 中等偏低径流 |
| 31 高覆盖度草地 | 0.0478 | 低径流 |
| 32 中覆盖度草地 | 0.08785 | 中等径流 |
| 33 低覆盖度草地 | 0.1827 | 较高径流（保水能力差） |
| 41-46 水域 | 0.0 | 不计径流（本身就是水体） |
| 51-53 建设用地 | 1.0 | 完全不渗透 |
| 61-67 未利用地 | 1.0 | 完全产流 |

注：原表格为百分比，本字典存储已除以 100 的小数形式。未在原表列出的二级类（水域 41-46、建设用地 52/53、未利用地 61-67）按其一级类兜底。

### 粮食生产修正系数

`FOOD_SUPPLY_PARAM` 定义了驱动力因子对粮食生产潜力的复合修正权重：

| 因子 | 权重 | 含义 |
|---|---|---|
| PODU (坡度) | 0.3317 | 坡度对粮食生产影响最大 |
| CITY (距城镇) | 0.1907 | 距城镇距离影响次之 |
| GD (距农地) | 0.1887 | 与耕地聚集度相关 |
| WATER (距水系) | 0.1592 | 灌溉条件 |
| DL (距道路) | 0.1297 | 交通可达性 |

这些权重与五因子分桶评分体系配合使用（参见 `utils/calculator_utils.calculate_food_supply_param`）。

## 驱动因子归一化参数

`FACTOR_PARAMS_DICT` 存储了三个年份（2010/2015/2020）的驱动因子归一化参数 `(miu, epsilon)`，用于对 DEM、DL、GDP、RK 四个因子做 `(x - miu) / epsilon` 标准化。

这些参数由 `pipelines/statistic.py` 通过 Welford 算法从全部样本中统计得出，写入 `_data/param_*.json`，并被 `utils/dataset_utils.apply_norm()` 消费。

`DEFAULT_CONDITION_LIST` = `["DEM", "DL", "GDP", "RK"]` 是 pred_diff 的默认条件因子集合（不包括 NLMY）。

### 对数变换因子

`ALL_LOG_FACTORS` = `["GDP", "RK", "DL"]` 标识了在归一化前需要做 `log1p` 变换的因子。统计阶段（`pipelines/statistic.py`）和数据加载阶段（`utils/dataset_utils`）都会检查此列表。

## 数据增强

训练数据集通过年份对组合实现数据增强。`ALL_PACK` 定义了四种配对方案：

| 年份对 | 含义 | 重复次数 |
|---|---|---|
| (10, 15) | 2010→2015 | 6 |
| (15, 20) | 2015→2020 | 2 |
| (10, 20) | 2010→2020 | 2 |
| (-1, -1) | 同一年内的自我配对 | 1 |

`ALL_PACK_TRANSFORM_REPEAT_DICT` 控制每种配对的重复采样次数，使时间间隔短的配对（更常见的变化模式）在训练中被更多采样。

## 算法阈值

| 常量 | 值 | 含义 |
|---|---|---|
| `PIXEL_NUM_THRESHOLD_ISCHANGE` | 30 | pred_diff 像素数低于此值的样本不会进入 SA/GA 优化（认为变化太小，不值得优化） |
| `PIXEL_NUM_THRESHOLD_REMOVE` | 20 | 在 diff 后处理中，连通区域小于此值的被移除（去噪） |

## 像素几何

`PIXEL_AREA_IN_HECTARES` = `(30 × 30) / 10000` = 0.09 公顷。

所有栅格数据使用 30m 分辨率，因此单个像素代表 900 m² = 0.09 ha。ES 计算中的面积相关单位（kg/hm² → kg、t/hm² → t）都通过这个常数转换。

## 碳密度数据 (carbon_pool.csv)

为 carbon ES 计算提供每种 LU 类型的四库碳密度（单位：t/ha）：

| LU 类型 | above | below | soil | dead | 特点 |
|---|---|---|---|---|---|
| 21 (有林地) | 36.34 | 7.27 | 102.40 | 3.35 | 地上碳最高 |
| 11 (水田) | 18.87 | 12.46 | 103.73 | 2.41 | 土壤碳最高 |
| 31 (高覆盖草地) | 17.37 | 20.85 | 105.85 | 2.94 | 地下碳比例高 |
| 41 (河渠) | 0 | 0 | 81.10 | 0 | 无生物量碳 |
| 51 (城镇用地) | 16.15 | 3.23 | 73.48 | 0 | 无死有机物 |

总计 27 行，覆盖所有 LU 二级类。`calculator/cal_es_score.calculate_carbon_storage()` 读取此 CSV 文件进行查表计算。

## FRAGSTATS 模型 (frg_model.fca)

一个预配置的 FRAGSTATS 4.2 景观格局分析模型（SQLite 格式），针对特定 128×128 栅格区域。内含景观指标配置（斑块 ID、斑块类型、面积等），用于计算景观格局指数的标准化流程。

## 配置覆写机制

`apply_const_overrides(score_cfg)` 允许在运行时修改 `const.py` 中的模块级常量，而无需改动源码。

目前它可以覆写：
- `PIXEL_NUM_THRESHOLD_ISCHANGE`：来自 `score_cfg.const.pixel_num_threshold_ischange`
- `WATER_SUPPLY_PARAM`：来自 `score_cfg.const.water_supply_param`

这个机制**仅**被 `pipelines/ablation.py`（消融实验管道）调用，普通训练和优化流程不会触发，常量保持默认值不变。

SA/GA 优化器中同时使用了 `from libs.const import *` 和 `import libs.const as _libs_const`：前者导入所有符号，后者保留模块对象的引用，以便 `apply_const_overrides` 修改常量后，后续通过模块属性访问能读到新值。

## 各模块依赖 libs 的方式

libs 被 18 个文件导入，覆盖了项目的所有子系统：

| 消费模块 | 导入方式 | 主要使用的符号 |
|---|---|---|
| `calculator/` | `from libs.const import *` | WATER_SUPPLY_PARAM、PIXEL_AREA_IN_HECTARES、FOOD_SUPPLY_PARAM、ALL_ES_TYPES |
| `runner/` | `from libs.const import *` | LU 标签、PIXEL_NUM_THRESHOLD_ISCHANGE、ALL_LU_LABELS_DICT |
| `datasets/` | `from libs.const import ALL_FACTORS` 等 | ALL_FACTORS、ALL_YEARS、ALL_PACK、归一化参数 |
| `utils/` | 按需导入 | FOOD_SUPPLY_PARAM、COLOR_MAP、PIXEL_AREA_IN_HECTARES、DEFAULT_CONDITION_LIST |
| `pipelines/` | 按需导入 | ALL_LOG_FACTORS、apply_const_overrides |

## 已知问题

- **`ALL_LU_LABELS` 与 `ALL_LU_LABELS_DICT` 不一致**：字典中定义了 27 个 LU 类型的名称（44、61-67 都在），但 `ALL_LU_LABELS` 列表只有 19 个。这可能是刻意为之（这些类型在当前研究区不存在），也可能需要统一
- **碳密度存在硬编码回退**：`calculator/cal_es_score.py` 中的 `calculate_carbon_storage` 函数在查表失败时，使用硬编码的默认碳密度值作为回退，而不是直接报错。回退值可能与 CSV 中的值不完全一致
