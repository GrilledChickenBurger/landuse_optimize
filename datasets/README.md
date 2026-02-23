# Datasets Module

土地利用优化项目的数据集管理模块，负责训练模型和优化算法所需的数据加载、预处理和创建。

## 文件说明

### `base_dataloader.py`
- **功能**: 基础数据加载器，提供统一的数据集接口
- **核心组件**:
  - `get_dataset()`: 统一的数据集获取函数，根据类型返回相应数据集
  - `BaseDataset`: 基础数据集类，提供数据集基本功能
  - `TrainingDataset`: 训练数据集类
  - `OptimizeDataset`: 随机优化数据集类
- **使用场景**: 被各pipeline模块调用，作为数据访问的统一入口

### `create_dataset.py`(to be deprecated)
- **功能**: 数据集创建工具，用于从原始栅格数据生成训练和优化所需的数据集
- **核心函数**:
  - `create_maxent_dataset()`: 创建MaxEnt模型所需数据集
  - `create_SA_dataset()`: 创建模拟退火优化所需数据集
  - `split_SA_dataset()`: 将数据集按8:2比例划分为训练集和验证集
- **使用场景**: 数据预处理阶段，将原始栅格数据转换为标准格式

### `optimize_dataloader.py`
- **功能**: 优化算法数据加载器，专门为随机优化算法准备数据
- **核心函数**:
  - `load_optimize_data()`: 加载SA优化所需的单样本打包数据
  - 数据格式: `[factors(不含RA/LU), restrict(RA), es_data(ES_list), ori_LU(year), tgt_LU(year+5), idx_label]`
- **使用场景**: 模拟退火(SA)和遗传算法(GA)优化过程中加载单样本数据

### `training_dataloader.py`
- **功能**: 训练数据加载器，为深度学习模型准备训练数据
- **核心函数**:
  - `load_training_data()`: 加载多年份的LU+驱动因子数据
  - `create_training_data_pack()`: 创建训练数据pack
  - `apply_transform()`: 数据增强变换
- **数据格式**: `[factors(src年), srcLU(one-hot), diff, idx_label, changed_flag]`
- **使用场景**: 深度学习模型训练过程中的数据加载

## 使用方式

### 主要调用路径
1. **训练管道** (`pipelines/train.py`): 导入`get_dataset`加载训练数据
2. **优化管道** (`pipelines/run_runner.py`): 导入`get_dataset`加载优化数据
3. **统计管道** (`pipelines/statistic.py`): 导入`get_dataset`进行数据分析


### 数据集类型
- **training**: 模型训练数据集，支持数据增强和归一化
- **optimize**: 优化算法数据集，包含单样本打包数据

### 数据格式规范
- **输入**: 128×128栅格数据，支持多种因子（DEM、DL、NLMY、GDP等）
- **输出**: 标准化的PyTorch Tensor格式
- **年份支持**: 10年、15年、20年三个时间点
