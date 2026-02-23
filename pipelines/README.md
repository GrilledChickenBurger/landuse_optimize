# Pipelines Module

土地利用优化项目的主要工作流程模块，包含从数据统计到模型训练、优化算法运行和LLM预测的完整流程。

## 文件说明

### `statistic.py`
- **功能**: 数据统计管道，获取驱动因子的数据分布参数
- **核心函数**: `get_params()` - 计算各驱动因子的均值和标准差
- **使用场景**: 训练开始前，为数据标准化提供参数
- **输出**: 保存参数到JSON文件，用于训练时的标准化处理

### `train.py`
- **功能**: 模型训练管道，训练土地利用变化检测模型
- **核心函数**: 
  - `training_pipeline()` - 主训练流程
  - `valid_pipeline()` - 模型验证流程
- **支持模型**: Att U-Net, U-Net, FCN, SegFormer, TransUNet等
- **功能特性**: 断点续训、TensorBoard日志、学习率调整、梯度裁剪

### `run_runner.py`
- **功能**: 随机优化算法运行管道
- **核心函数**: `pipeline()` - 运行SA或GA优化器
- **支持算法**: 模拟退火(SA)、遗传算法(GA)
- **功能特性**: 多场景优化、结果保存、已处理样本跳过


## 使用方式

### 主要调用路径
1. **主程序** (`main.py`): 导入所有管道函数
2. **统计参数**: `get_params()` 用于数据预处理
3. **模型训练**: `training_pipeline()` 和 `valid_pipeline()` 用于模型训练和验证
4. **优化运行**: `pipeline()` 用于执行随机优化算法

### 管道执行流程

#### 标准工作流
1. **Step 0** (statistic.py): 获取数据分布参数
2. **Step 1** (train.py): 训练变化检测模型
3. **Step 2** (run_runner.py): 运行随机优化算法

#### 配置驱动
所有管道都通过YAML配置文件进行参数配置：
- `configs/trainConfig.yaml`: 训练配置
- `configs/optimizerConfig.yaml`: 优化器配置
- `configs/statisticConfig.yaml`: 统计配置
