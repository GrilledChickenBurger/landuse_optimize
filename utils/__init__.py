# 常用工具函数 (向后兼容：细粒度导入路径仍可用)
from .tools import ensure_folder, get_yaml, get_namespace, convert_to_nparray, PiecewiseLinear

# 可视化函数 (不依赖 elapid，安全导入)
from .visualize_utils import (color_bar_figure, stacked_bar_figure, box_plot_figure,
                               grouped_bar_figure, grouped_box_plot_figure)

# calculator_utils 依赖 elapid/Numba，导入失败时跳过
try:
    from .calculator_utils import get_numfeature_dict, get_perimeter, train_maxent
except ImportError:
    pass
