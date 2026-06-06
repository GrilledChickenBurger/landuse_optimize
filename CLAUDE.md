# CLAUDE.md

每次回复称呼我zsu，用中文回复。
在写计划之前，若有任何疑问请直接问我，完全没有疑问后，再开始写计划。

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **代码生成规范**：若需要生成或修改代码，请先查阅 `docs/coding-standards.md`，确保新增代码位于正确层次、优先复用已有工具、数据记录字段与 `docs/ablation/字段标准.md` 保持一致。

## 项目概述

土地利用（LU）优化系统，分两阶段工作：

1. **深度学习变化检测** — 训练语义分割模型（AttU-Net/U-Net/FCN/SegFormer/TransUNet），输入驱动因子 + 当前 LU，预测每个像素是否变化
2. **随机优化搜索** — 以模型预测的变化概率图为先验，用 SA 或 GA 搜索在接近预测变化区域的前提下最大化生态系统服务的 LU 方案

```
驱动因子 + 历史LU → [ANN模型] → pred_diff (变化概率图)
                                        ↓
pred_diff + 约束 + 原始LU → [SA/GA优化器] → 最优LU方案
                                ↓
                        目标函数 = 变化位置 + ES + 景观破碎化
```

## 核心逻辑

### 目标函数 (`calculator/tgt_function_cal.py`)

```
tgt_func = change_score - λ_ES × es_score + λ_LS × landscape_score
```

这是一个**最小化**问题。SA 直接最小化 tgt_func；GA 适应度 = -tgt_func 求最大化。λ_ES 和 λ_LS 根据 pred_diff 的像素数量动态线性插值。

- **change_score** (`cal_change_score.py`): 改变区域的连通区域越靠近预测变化区域，得分越高。用 `scipy.ndimage.label` 标记连通区域，基于距离变换图计算距离惩罚
- **es_score** (`cal_es_score.py`): 四项生态系统服务相对其理论最大值的加权和（权重由配置决定）：
  - food: Miami + Thornthwaite Memorial 模型，基于气温/降水计算粮食生产潜力，仅在耕地上累加
  - carbon: 查表法，每种 LU 类型 × 碳密度 × 面积
  - water: `Q = P - P×R - ET`，不同 LU 类型有不同地表径流系数
  - culture: 预训练 MaxEnt 模型预测文化服务适宜性
- **landscape_score** (`cal_landscape_score.py`): 改变区域内 LU 类型越单一、斑块数越少、离同类原始斑块越近，得分越低（越好）

### 方案生成器 (`generator/`)

- SA 和 GA 共用矩形区域修改策略：随机选一个矩形 → 全部改为随机 LU 类型，分 small/mid/large 三档，受温度/代数控制。另有 `return` 操作将区域恢复为原始方案
- SA: `generate_new_plan()` 单步扰动；GA: `crossover()` 交换矩形区域 + `mutate()` 复用 SA 扰动

### 三层约束

1. **hard_restrict**: 耕地红线 + 预测无变化区域 + LU=0 → 完全不可修改
2. **cls_restrict**: 城镇用地(51/53)只能改成建设用地(5 类)，不能改为耕地/林地
3. **scale_minmax**: 根据预测变化像素数动态调整每次扰动的大小范围

### 优化器对比

| | SA | GA |
|---|---|---|
| 搜索方式 | 单点搜索，逐步降温 | 种群搜索，逐代进化 |
| 接受新解 | Metropolis 准则 | 精英保留 + 选择压力 |
| 终止条件 | 连续无改进 / 最优解停滞 | 同左，阈值随代数自适应递减 |
| 策略可变 | 无 | `GeneticAlgorithmStrategy` 组装 7 种策略变体（父本来源/选择方式/精英比例/变异控制） |

## 入口

所有 pipeline 从 `main.py` 启动（取消底部对应函数的注释）：

| 函数 | Pipeline | 配置文件 |
|---|---|---|
| `run_train_pipeline()` | 训练变化检测模型 | `configs/trainConfig.yaml` |
| `run_valid_pipeline()` | 验证已训练模型 | `configs/trainConfig.yaml` |
| `run_runner_pipeline()` | 对选定样本运行 SA/GA 优化 | `configs/optimizerConfig.yaml` |
| `run_experiment_pipeline()` | 批量 GA 策略组合实验 | `configs/optimizerConfig.yaml` |
| `run_statistic_pipeline()` | 计算驱动因子归一化参数 | `configs/statisticConfig.yaml` |

没有 CLI 参数解析 — 参数和路径硬编码在各函数中。

## 架构

```
main.py
├── pipelines/train.py        # 阶段1: 训练 + 验证 ANN 变化检测模型
├── pipelines/run_runner.py   # 阶段2a: 单个样本 SA/GA 优化
├── pipelines/experiment.py   # 阶段2b: 批量 GA 策略组合实验
├── pipelines/statistic.py    # 前置: 计算驱动因子归一化参数
│
models/                       # ANN 模型 (二分类语义分割)
│   __init__.py               # create_model(): attunet, unet, fcn, segformer, transunet
│
runner/                       # 随机优化器
│   __init__.py               # create_runner(): SA, GA
│   SA.py                     # SimulatedAnnealing
│   GA.py                     # GeneticAlgorithm + GeneticAlgorithmStrategy
│
generator/                    # 方案扰动算子 (SA/GA 共用)
│   sa_generator.py           # generate_new_plan(), area_wise_change(), area_wise_return()
│   ga_generator.py           # initial_new_population(), crossover(), mutate()
│
calculator/                   # 目标函数各子项
│   tgt_function_cal.py       # cal_tgt_function(): 三项加权求和
│   cal_change_score.py       # 变化位置得分 (连通区域 + 距离变换)
│   cal_es_score.py           # ES 得分 (food/carbon/water/culture)
│   cal_landscape_score.py    # 景观破碎化得分
│
datasets/                     # 数据加载
│   training_dataloader.py    # → [factors, srcLU, diff, label, changed_flag]
│   optimize_dataloader.py    # → [factors, restrict, es, srcLU, tgtLU, label]
│
libs/
│   const.py                  # LU标签, ES类型, 颜色映射, 驱动因子归一化参数
│   carbon_pool.csv           # 碳库查找表
│
utils/                        # 工具函数
│   tools.py                  # get_yaml()→Namespace, visualize_tensor_list()
│   runner_utils.py           # load_ANN_model(), get_pred_diff(), save_plan_and_get_performance()
│   calculator_utils.py       # MaxEnt, 距离图, info_dict 构建
│   performance.py            # ES/变化/景观指标的性能评分
│
configs/                      # YAML 配置 (加载为 Namespace)
results/ checkpoints/         # git-ignored
```

## 关键设计模式

- **配置是 Namespace。** `get_yaml(path)` 递归将 dict 转为 `argparse.Namespace`，点号访问: `config.data.factor_list`
- **data_pack 格式因阶段而异。** 训练: `[factors, srcLU, diff, idx_label, changed_flag]`；优化: `[factors, restrict, es_data, srcLU, tgtLU, idx_label]`。均为 numpy 列表，由 pipeline 转为 tensor
- **模型输入是因子 + LU 通道拼接。** `inputs = torch.cat([factors_src, lu_src], dim=1)` → `in_ch = len(factor_list) + 6`（6 类简化 LU 的 one-hot）
- **ANN 模型在优化阶段复用。** `runner_utils.get_pred_diff()` 生成 pred_diff，作为目标函数变化得分项的先验
- **GA 策略模式。** `GeneticAlgorithmStrategy` 按配置字典组装选择/交叉/变异函数，实验 pipeline 遍历策略组合

## 路径约定

配置和数据路径使用 Windows 风格反斜杠（如 `"checkpoints\\attunet\\20260212_045750\\best_model.pth"`）。
