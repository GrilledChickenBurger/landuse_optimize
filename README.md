# LU Optim - 土地利用优化系统

## 项目概述

Landuse Optimize 是一个基于深度学习和优化算法的土地利用变化检测与优化系统。项目结合了变化检测模型和随机优化算法，用于土地利用规划优化。

## 项目结构

### 核心文件夹

- **calculator/**: 计算模块
  - `cal_change_score.py` - 变化得分计算
  - `cal_es_score.py` - 生态系统服务得分计算
  - `cal_landscape_score.py` - 景观得分计算
  - `tgt_function_cal.py` - 目标函数计算，包含变化得分、生态系统服务得分和景观得分（可选）的加权组合

- **checkpoints/**: 模型检查点
  - 存储训练好的模型权重文件
  - 包含多种模型架构：attunet、unet、fcn、segformer、transunet

- **configs/**: 配置文件
  - `trainConfig.yaml` - 训练配置，包含模型架构、训练参数（如学习率、批量大小）等
  - `optimizerConfig.yaml` - 优化器配置，包含模型选择、优化算法参数（如模拟退火SA、遗传算法GA）等
  - `apiConfig.yaml` - API配置
  - `statisticConfig.yaml` - 统计配置

- **datasets/**: 数据加载模块
  - `base_dataloader.py` - 基础数据加载器
  - `create_dataset.py` - 数据集创建
  - `training_dataloader.py` - 训练数据加载器
  - `optimize_dataloader.py` - 优化数据加载器

- **generator/**: 方案生成器模块
  - `sa.py` - 模拟退火 方案生成器

- **libs/**: 库文件
  - `const.py` - 常量定义
  - `carbon_pool.csv` - 碳池数据
  - `frg_model.fca` - 森林资源增长模型

- **models/**: 模型定义
  - `unet.py`、`fcn.py`、`segformer.py`、`transunet.py` - 不同架构的变化预测模型
  - `modules/*` - 模型需要的其他模块

- **pipelines/**: 流水线模块
  - `train.py` - 训练流水线
  - `run_runner.py` - 优化器运行流水线
  - `statistic.py` - 统计流水线
  - `call_llm.py` - LLM调用流水线

- **runner/**: 随机优化算法运行模块
  - `GA.py` - 遗传算法
  - `SA.py` - 模拟退火

- **utils/**: 工具函数
  - 各种辅助工具和计算函数

## 主要Pipeline功能

### 1. 训练Pipeline (`pipelines/train.py`)
- 训练模型预测土地利用变化
- 支持多种模型架构：Att U-Net、U-Net、FCN、SegFormer、TransUNet
- 模型训练和验证流程

### 2. 优化器Pipeline (`pipelines/run_runner.py`)
- 运行随机优化算法（模拟退火SA、遗传算法GA）
- 土地利用规划优化
- 目标函数计算和性能评估

### 3. 统计Pipeline (`pipelines/statistic.py`)
- 参数统计，为训练数据提供归一化参数

### 4. LLM Pipeline (`pipelines/call_llm.py`)
- 大语言模型调用接口，作为变化预测的补充

## 核心功能

1. **变化检测**: 使用深度学习模型检测土地利用变化
2. **优化规划**: 基于随机优化算法生成最优土地利用方案
3. **生态系统服务评估**: 计算水源涵养、粮食生产等生态指标
4. **景观分析**: 评估土地利用方案的景观效果


## 使用说明

通过 `main.py` 中的不同函数调用各个pipeline：

```python
# 训练模型
run_train_pipeline()

# 验证模型
run_valid_pipeline()

# 运行优化器
run_runner_pipeline()

# 调用LLM
run_llm_pipeline()
```
通过 `configs/` 目录下的配置文件调整参数，如模型架构、训练参数、优化算法参数等。
```
# trainConfig.yaml
device:
  use_gpu: true


# 数据配置, 通用
data:
  root: "_data\\training_data"
  split: "train"  # train|val, 训练集或验证集
  shape: [1, 128, 128]  # [通道数, 高度, 宽度]
  ......
```

## 注意事项

- 配置文件位于 `configs/` 目录，可根据需求调整参数
- 数据应放置在 `_data/` 文件夹中（已被git忽略）
- 模型检查点存储在 `checkpoints/` 目录，按模型类型和时间戳组织

## 依赖环境

详见 `requirements.txt` 文件。

## 更新日志

- 2025-02-23:
  - 预测模型选择：Att U-Net、U-Net、FCN、SegFormer、TransUNet
  - 优化器选择：模拟退火SA
  - 目标函数计算：变化得分、生态系统服务得分、景观得分（可选）的加权组合
  - 训练过程和随机优化过程支持断点续训