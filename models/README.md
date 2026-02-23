# Models Module

土地利用变化预测的深度学习模型模块，包含多种语义分割网络架构。

## 文件说明

### `__init__.py`
- **功能**: 模型工厂函数，提供统一的模型创建接口
- **核心函数**: `create_model()` - 根据模型类型创建对应模型实例
- **支持模型**: attunet, fcn, unet, segformer, transunet

### `unet.py`
- **功能**: U-Net系列模型实现
- **包含模型**:
  - `U_Net`: 基础U-Net模型
  - `AttU_Net`: 注意力U-Net模型
- **架构特点**: 编码器-解码器结构，跳跃连接

### `fcn.py`
- **功能**: 全卷积网络实现
- **包含模型**: `FCN` - 全卷积网络
- **架构特点**: 全卷积结构，无全连接层，支持任意尺寸输入

### `segformer.py`
- **功能**: SegFormer模型实现
- **包含模型**: `Segformer` - 基于Transformer的语义分割网络
- **架构特点**: 分层Transformer编码器，轻量级MLP解码器


### `transunet.py`
- **功能**: TransUNet模型实现
- **包含模型**: `TransUNet` - Transformer与U-Net的混合架构
- **架构特点**: Transformer编码器，U-Net解码器，全局上下文建模


## 使用方式

### 主要调用路径
1. **训练管道** (`pipelines/train.py`): 导入`create_model`创建训练模型
2. **运行器工具** (`utils/runner_utils.py`): 导入`create_model`创建预测模型

### 模型创建示例
```python
from models import create_model

# 创建注意力U-Net模型
model = create_model('attunet', in_ch=3, out_ch=1)

# 创建SegFormer模型
model = create_model('segformer', in_ch=3, out_ch=1, model_config={...})
```

### 支持的模型类型
- `attunet`: 注意力U-Net
- `fcn`: 全卷积网络  
- `unet`: 基础U-Net
- `segformer`: SegFormer
- `transunet`: TransUNet

## 输入输出格式

### 输入参数
- `in_ch`: 输入通道数 (通常为3或因子数量)
- `out_ch`: 输出通道数 (通常为1或类别数量)
- `model_config`: 模型配置字典 (可选)

### 输出结果
- 所有模型输出形状为 `[batch_size, out_ch, height, width]`
- 适用于128×128栅格数据的语义分割任务