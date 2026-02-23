"""
随机优化算法的运行器模块
"""

from .SA import SA
from .GA import GA  
# 创建优化器工厂函数
def create_runner(runner_type, config, **kwargs):
    if runner_type == 'SA':
        return SA(config=config, **kwargs)
    elif runner_type == 'GA':  # 未来扩展
        return GA(config=config, **kwargs)
    else:
        raise ValueError(f"不支持的运行器类型: {runner_type}")
