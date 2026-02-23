# Calculator Module

土地利用优化算法中的目标函数计算模块，用于随机优化算法运行过程中计算各目标函数项。

## 文件说明


### `cal_change_score.py`
- **功能**: 计算变化得分，衡量当前更改区域到预测变化区域的距离
- **核心函数**: 
  - `cal_change_score()`: 计算相对变化得分（范围：[-∞, 1]，越大越好）
  - `cal_change_value()`: 计算绝对变化值
  - `get_change_min_score()`: 计算最小变化得分，用于计算目标函数项时归一化
- **使用场景**: 在随机优化算法中评估【方案变化区域】与【预测变化区域】的匹配程度

### `cal_es_score.py`
- **功能**: 计算生态系统服务（ES）得分，包括粮食供给、碳汇、水源涵养、生态服务
- **核心函数**:
  - `cal_es_score()`: 计算相对ES得分（范围：[0, 1]，越大越好）
  - `cal_es_value()`: 计算各项ES得分的绝对值
  - `get_es_max_score()`: 计算各项ES指标最大值，用于计算目标函数项时归一化
- **使用场景**: 评估土地利用方案的环境效益，在优化算法中作为重要目标函数项

### `cal_landscape_score.py`
- **功能**: 计算景观格局得分，衡量变化区域的连通性和空间布局
- **核心函数**:
  - `cal_landscape_score()`: 计算景观格局得分，考虑连通区域数、类型纯度、空间距离等
- **使用场景**: 可选约束项，用于控制土地利用变化的景观格局合理性

### `tgt_function_cal.py`
- **功能**: 目标函数综合计算器，整合所有得分项
- **核心函数**:
  - `cal_tgt_function()`: 计算最终目标函数值，越小表示方案越优秀
  - 公式：`目标函数 = 变化得分 - λ_ES × ES得分 + λ_LS × 景观得分`
- **使用场景**: GA和SA优化算法的核心评估函数，被`runner/GA.py`和`runner/SA.py`直接调用

## 使用方式

### 主要调用路径
1. **优化算法** (`runner/GA.py`, `runner/SA.py`): 直接导入`cal_tgt_function`
2. **工具函数** (`utils/calculator_utils.py`): 导入`get_change_min_score`和`get_es_max_score`
3. **性能评估** (`utils/performance.py`): 导入`cal_es_value`和`get_es_cal_func_dict`

### 输入输出
- **输入**: 土地利用方案张量、变化区域、预测区域、原始方案、参数字典等
- **输出**: 各项得分及综合目标函数值
- **数据格式**: PyTorch Tensor

