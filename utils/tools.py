# 通用工具函数
# 注意: 配置加载函数已移至 config_utils.py, 文件/视频 I/O 已移至 io_utils.py。
#       此处保留 re-export 确保向后兼容。

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 向后兼容 re-exports (旧代码仍可用 from utils.tools import get_yaml, ensure_folder 等)
from .config_utils import (get_yaml, get_namespace, get_default_score_cfg, load_score_cfg,
                           _DEFAULT_SCORE_CFG_DICT)
from .io_utils import ensure_folder, find_valid_file, save_tif, get_video_from_imgs

def repeat(func: callable, repeat_num: int, replace_element: Any, **kwargs) -> Any:
    """
    重复调用函数repeat_num次，并将每次的结果作为下一次调用的参数。\n
    Parameters:
    ----------
        func: callable, 要重复调用的函数
        repeat_num: int, 重复调用次数
        replace_element: Any, 初始参数
        **kwargs: 其他固定参数

    Returns:
    ----------
        Any: 经过repeat_num次调用后的最终结果
    """
    e = replace_element
    for _ in range(repeat_num):
        res = func(e, **kwargs)
        # 如果函数返回多个值，只取第一个值作为下一次调用的参数
        if isinstance(res, tuple):
            e = res[0]  # 取第一个返回值
        else:
            e = res
    return e

class PiecewiseLinear:
    """
    分段线性插值类
    
    Example:
    --------
        # 创建实例
        pl = PiecewiseLinear([0, 10, 30, 50], [1.0, 1.3, 1.9, 2.0])
        # 使用实例
        result = pl(20)  # 在10和30之间线性插值
        # result = 1.3 + (1.9-1.3)*(20-10)/(30-10) = 1.6
    """
    
    def __init__(self, breakpoints: List[float], values: List[float]):
        """
        初始化分段线性插值器
        
        Parameters:
        ----------
            breakpoints: List[float], 分段点列表（必须按升序排列）
            values: List[float], 对应分段点的值列表
        """
        if len(breakpoints) != len(values):
            raise ValueError("分段点和值列表长度必须相同")
        
        # 确保分段点按升序排列
        sorted_pairs = sorted(zip(breakpoints, values))
        self.breakpoints = [pair[0] for pair in sorted_pairs]
        self.values = [pair[1] for pair in sorted_pairs]
        
    def __call__(self, x: float) -> float:
        """
        计算分段线性插值结果
        
        Parameters:
        ----------
            x: float, 输入值
            
        Returns:
        ----------
            float: 分段线性插值结果
        """
        if x <= self.breakpoints[0]:
            return self.values[0]
        if x >= self.breakpoints[-1]:
            return self.values[-1]
        
        # 找到x所在的分段区间
        for i in range(len(self.breakpoints) - 1):
            if self.breakpoints[i] <= x <= self.breakpoints[i + 1]:
                # 线性插值
                ratio = (x - self.breakpoints[i]) / (self.breakpoints[i + 1] - self.breakpoints[i])
                return self.values[i] + ratio * (self.values[i + 1] - self.values[i])
        
        return self.values[-1]  # 理论上不会执行到这里


def convert_to_nparray(array: Any) -> Optional[np.ndarray]:  
    """
    将输入转换为numpy数组.\n
    Parameters:
    ----------
        array: 输入张量或numpy数组

    Returns:
    ----------
        numpy数组
    """  
    if isinstance(array, np.ndarray):  
        return array  
    elif isinstance(array, torch.Tensor):  
        return array.detach().cpu().numpy() 
    elif hasattr(array, 'numpy') and callable(getattr(array, 'numpy')):   
        return array.numpy()  
    else:  
        print("错误：输入不是张量或numpy数组")  
        return None  


def draw_single_line(x_data: List[float], y_data: List[float],
                    label: str, x_label: str, y_label: str, 
                    title: Optional[str] = None, 
                    save_path: Optional[str] = None):
    """
    绘制只有一组数据的折线图。
    Parameters:
        x_data: List[float], 数据
        y_data: List[float], 数据
        label: str, 标签
        x_label: str, x轴标签
        y_label: str, y轴标签
        title: str, 标题
        save_path: str, 保存路径
    """
    plt.figure(figsize=(40,20))
    plt.plot(x_data, y_data, label=label, linewidth=4)
    plt.xlabel(x_label, fontsize = 24)
    plt.ylabel(y_label, fontsize = 24)
    plt.title(title if title is not None else "Result", fontsize = 30)
    plt.tick_params(labelsize=20)     #调整坐标轴数字大小
    plt.legend(prop = {'size':20})
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def draw_multi_line(x_data: List[float], y_datas: List[List[float]], 
                    labels: List[str], x_label: str, y_label: str, 
                    title: Optional[str] = None, 
                    save_path: Optional[str] = None):
    """
    绘制多组数据的折线图。
    Parameters:
        x_data: List[float], 数据
        y_datas: List[List[float]], 数据
        labels: List[str], 标签
        x_label: str, x轴标签
        y_label: str, y轴标签
        title: str, 标题
        save_path: str, 保存路径
    """
    plt.figure(figsize=(40,20))
    for y_data, label in zip(y_datas, labels):
        plt.plot(x_data, y_data, label=label, linewidth=4)
    plt.xlabel(x_label, fontsize = 24)
    plt.ylabel(y_label, fontsize = 24)
    plt.title(title if title is not None else "Result", fontsize = 30)
    plt.tick_params(labelsize=20)     #调整坐标轴数字大小
    plt.legend(prop = {'size':20})
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def show_array_result(array: Any, title: Optional[str] = None, label: Optional[str] = None) -> None:
    """
    可视化二维数组类型的结果。\n
    Parameters:
    ----------
        array: ndarray
        title: str, 标题
        label: str, 标签
    """
    array_np= convert_to_nparray(array)
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=100)
    plot = ax.imshow(array_np.squeeze(), cmap='GnBu',vmin=array_np.min(), vmax=array_np.max())
    ax.set_title(title if title is not None else "Result")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(plot, ax=ax, label=label if label is not None else "Value", pad=0.04)
    plt.tight_layout()
    plt.show()

def visualize_tensor_list(tensor_dict: Dict[str, torch.Tensor], 
                        main_title: str = 'Tensor Visualization',
                        save_path: str = None, 
                        color_map: Union[Dict[Any, Tuple[float, float, float]], str] = None, 
                        figsize: Tuple[int, int] = None, 
                        layout: Tuple[int, int] = None,
                        show: bool = False,
                        show_colorbar: bool = True) -> List[str]:
    """
    通用张量可视化函数，支持自定义颜色映射和图像布局配置。
    
    Parameters:
    ----------
        tensor_dict: Dict[str, torch.Tensor], 张量字典, 格式为
            {'subtitle1': tensor1, 'subtitle2': tensor2, ...}
            每个张量可以是 [1, 1, height, width]、[1, height, width] 或 [height, width] 格式
        main_title: str, 可视化窗口的主标题，默认值为 'Tensor Visualization'
        save_path: str, 保存图像的路径，默认不保存
        color_map: dict, 自定义颜色映射字典，格式为 {value: (r, g, b)}，其中r, g, b的范围为0-1
        figsize: tuple, 图像大小，格式为 (width, height)，默认根据张量数量自动计算
        layout: tuple, 子图布局，格式为 (rows, cols)，默认自动计算
        show: bool, 是否显示图像，默认值为 False
        show_colorbar: bool, 是否显示颜色带，默认值为 True
    
    Returns:
    -------
        List[str]: 保存的文件路径列表（如果指定了save_path）
    """   
    if not tensor_dict:
        raise ValueError("张量字典不能为空")
    
    # 处理张量形状，确保为2D
    processed_tensors = {}
    for name, tensor in tensor_dict.items():
        processed_tensor = tensor.detach().cpu().squeeze()  # 移到CPU，压缩所有尺寸为1的维度
        if len(processed_tensor.shape) != 2:
            raise ValueError(f"张量 '{name}' 处理后形状应为 [height, width]，当前形状：{processed_tensor.shape}")
        processed_tensors[name] = convert_to_nparray(processed_tensor)  # 转换为numpy数组
    
    saved_paths = []  
    # 计算子图布局
    num_tensors = len(processed_tensors)
    if layout is None:
        # 根据张量数量自动确定布局
        if num_tensors <= 4:
            rows, cols = 1, num_tensors
        elif num_tensors <= 6:
            rows, cols = 2, 3
        elif num_tensors <= 9:
            rows, cols = 3, 3
        else:
            # 对于更多张量，使用尽量方形的布局
            cols = int(np.ceil(np.sqrt(num_tensors)))
            rows = int(np.ceil(num_tensors / cols))
    else:
        rows, cols = layout
    
    # 计算图像大小
    if figsize is None:
        # 根据布局自动计算合适的图像大小
        base_size = 5
        figsize = (cols * base_size, rows * base_size)
    
    # 限制图像大小以避免超过matplotlib的限制
    max_figsize = 50  # 限制最大图像尺寸
    if figsize[0] > max_figsize or figsize[1] > max_figsize:
        scale_factor = max_figsize / max(figsize)
        figsize = (figsize[0] * scale_factor, figsize[1] * scale_factor)
        print(f"警告: 图像尺寸过大，已调整为 {figsize}")
    
    # 创建图像和子图
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]  # 确保axes是列表
    elif rows == 1 or cols == 1:
        axes = axes.flatten()  # 一维数组
    else:
        axes = axes.flatten()  # 二维数组展平
    
    # 设置主标题
    if main_title:
        fig.text(0.1, 0.9, main_title, ha='left', va='baseline', fontsize=16, color='black', fontweight='bold')  
        # fig.suptitle(main_title, fontsize=16, fontweight='bold')
    # 绘制每个张量
    for i, (title, nparr) in enumerate(processed_tensors.items()):
            ax = axes[i]
            # 使用自定义颜色映射或默认颜色映射
            if isinstance(color_map, dict):                
                # 获取张量中的唯一值
                unique_values = np.unique(nparr)
                
                # 创建颜色映射数组
                cmap_values = []
                for val in unique_values:
                    if val in color_map:
                        cmap_values.append(color_map[val])
                    else:
                        # 对于没有在color_map中定义的值，使用灰色
                        cmap_values.append((0.5, 0.5, 0.5))
                
                # 创建自定义颜色映射
                custom_cmap = ListedColormap(cmap_values)
                
                # 将原始张量数据映射到新的索引
                mapped_data = np.zeros_like(nparr, dtype=int)
                for idx, val in enumerate(unique_values):
                    mapped_data[nparr == val] = idx
                
                # 绘制热图
                im = ax.imshow(mapped_data, cmap=custom_cmap)
                
                # 添加颜色条并自定义标签（如果启用）
                if show_colorbar:
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_ticks(range(len(unique_values)))
                    cbar.set_ticklabels([str(int(v)) if isinstance(v, (int, np.integer)) else str(v) for v in unique_values])
            elif isinstance(color_map, str):
                # 使用默认颜色映射
                im = ax.imshow(nparr, cmap=color_map)
                if show_colorbar:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                # 使用默认颜色映射
                im = ax.imshow(nparr, cmap='viridis')
                if show_colorbar:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 设置标题
            ax.set_title(title)
            ax.axis('off')  # 关闭坐标轴
    
    # 隐藏多余的子图
    for i in range(num_tensors, len(axes)):
        axes[i].axis('off')
    
    if main_title:
        plt.subplots_adjust(top=0.9)
    
    # 保存图像
    if save_path:
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 限制DPI以避免超过matplotlib的尺寸限制
        plt.savefig(save_path, dpi=min(150, 300), bbox_inches='tight')  # 降低DPI避免尺寸过大
        saved_paths.append(save_path)
    
    # 显示图像
    if show:
        plt.show()
    else:
        plt.close()
    return saved_paths

# ===== 以下函数已移至专用子模块，tools.py 通过 re-export 保持向后兼容 =====
# get_video_from_imgs -> video_utils.py
# save_tif, find_valid_file -> file_utils.py


