# Utils Module

土地利用优化项目的工具函数模块，包含数据处理、模型训练、优化算法运行等各个方面的辅助工具。

## 文件说明

### `tools.py`
- **功能**: 通用工具函数集合
- **核心函数**:
  - `ensure_folder()`: 确保文件夹存在
  - `get_yaml()`: 加载YAML配置文件
  - `convert_to_nparray()`: 数据类型转换
  - `visualize_tensor_list()`: 张量可视化
- **使用场景**: 被各模块广泛调用，提供基础功能支持

### `dataset_utils.py`
- **功能**: 数据集处理工具
- **核心函数**:
  - `load_tiff()`: 加载TIFF栅格数据
  - `get_year_data()`: 获取指定年份的因子数据
  - `remove_lines()`: 移除小斑块区域
  - `apply_norm()`: 数据标准化
- **使用场景**: 数据预处理、数据加载、数据清洗

### `calculator_utils.py`
- **功能**: 计算器相关工具
- **核心函数**:
  - `load_carbon_pool()`: 加载碳密度表
  - `train_maxent()`: 训练MaxEnt模型
  - `get_pred_diff_distance_map()`: 计算变化区域距离地图
  - `get_perimeter()`: 计算连通区域周长
- **使用场景**: 生态系统服务计算、目标函数计算辅助

### `runner_utils.py`
- **功能**: 运行器相关工具
- **核心函数**:
  - `load_ANN_model()`: 加载深度学习模型
  - `get_pred_diff()`: 获取预测变化区域
  - `get_cls_restrict_area()`: 获取类别约束区域
  - `get_grid_restrict_area()`: 获取网格限制区域
- **使用场景**: 优化算法运行过程中的辅助功能

### `performance.py`
- **功能**: 性能评估工具
- **核心函数**:
  - `calculate_FOM()`: 计算FOM系数
  - `calculate_kappa()`: 计算kappa系数
  - `calculate_OA()`: 计算总体准确率
  - `calculate_iou_01()`: 计算IoU指标
- **使用场景**: 模型性能评估、结果质量分析

### `training_utils.py`
- **功能**: 训练相关工具
- **核心函数**:
  - `get_optimizer()`: 获取优化器实例
  - `adjust_learning_rate()`: 调整学习率
  - `ExpoAverageMeter`: 指数加权平均计器
- **使用场景**: 模型训练过程中的辅助功能

### `param_utils.py`
- **功能**: 参数统计工具
- **核心类**: `RunningStats` - 运行统计类
- **使用场景**: 数据分布参数计算，用于标准化处理

### `train_maxent.py`
- **功能**: MaxEnt模型训练工具
- **核心函数**: `train_pipeline()` - MaxEnt训练流程
- **使用场景**: 生态系统服务模型训练

## 使用方式

### 主要调用路径
1. **数据集模块**: 导入`dataset_utils`进行数据加载和处理
2. **运行器模块**: 导入`calculator_utils`、`runner_utils`、`performance`等
3. **管道模块**: 导入`tools`、`training_utils`等通用工具
4. **计算器模块**: 导入`calculator_utils`进行辅助计算

