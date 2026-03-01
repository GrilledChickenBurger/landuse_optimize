# 随机优化算法运行相关工具


from argparse import Namespace
import os
import shutil
import copy
from typing import Dict, List, Union
import torch
import torch.nn.functional as F
import numpy as np

from models import create_model
from libs.const import *
from .dataset_utils import apply_norm, one_hot_lu
from .tools import ensure_folder, convert_to_nparray, draw_single_line, draw_multi_line, save_tif


def load_ANN_model(model_type: str, model_path: str, in_ch: int, out_ch: int, device: torch.device=None, model_config: Namespace=None) -> torch.nn.Module:
    """加载ANN模型检查点\n
    Parameters:
    ----------
        model_type: str, ANN模型类型, 可选值为 'attunet,unet'
        model_path: str, ANN模型检查点文件路径, 可以处理后缀名为 '.pt' 或 '.pth'
        in_ch: int, 输入通道数
        out_ch: int, 输出通道数
        device: torch.device, 运行设备（CPU/GPU）
        model_config: Namespace, 模型配置参数

    Returns:
    --------
        torch.nn.Module: 加载的ANN模型
    """
    model = create_model(
        model_type=model_type,
        in_ch=in_ch,
        out_ch=out_ch,
        model_config=model_config
    )
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 尝试加载模型检查点
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)  # 确保模型被移动到指定设备上
        return model
    except RuntimeError as e:
        # 处理模型参数不匹配的错误
        if "size mismatch" in str(e) or "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
            print(f"错误: 模型检查点与模型结构不匹配!")
            print(f"模型类型: {model_type}")
            print(f"输入通道数: {in_ch}")
            print(f"输出通道数: {out_ch}")
            print(f"检查点路径: {model_path}")
            raise RuntimeError(f"模型加载失败: {e}") from e
        else:
            # 其他运行时错误
            print(f"错误: 加载模型检查点时发生异常")
            print(f"错误信息: {e}")
            raise
    except FileNotFoundError:
        print(f"错误: 找不到模型检查点文件: {model_path}")
        raise
    except Exception as e:
        # 处理其他未知错误
        print(f"错误: 加载模型时发生未知异常: {e}")
        raise

def get_pred_diff(model: torch.nn.Module, 
                    ori_plan: torch.Tensor, 
                    factors: torch.Tensor, 
                    device: torch.device, 
                    year: str,
                    condition_list: List[str],
                    normalize: bool = False) -> torch.Tensor:
    """
    通过预训练的ANN模型, 得到5年后将会发生变化的区域
    
    Parameters:
    ----------
        model: torch.nn.Module, 预训练的ANN模型
        ori_plan: tensor, [1,1,128,128]
        factors: tensor, [1,factor_num,128,128], 驱动因子
        year: str, 年份, 可选值为 '10', '15', '20'
        condition_list: List[str], 驱动因子列表, 必须和factors中的channel顺序对应
        normalize: bool, 是否对因子进行标准化, default=False
    
    Returns:
    --------
        pred_diff: tensor, [1,1,128,128], 0-1 二值张量，0表示预测不变，1表示预测将发生变化
    """
    # 确保模型在正确的设备上
    model = model.to(device)
    if normalize:
        assert condition_list is not None, "condition_list must be provided when normalize is True."
        params = FACTOR_PARAMS_DICT[year]  # 获取该年份的miu, epsilon参数
        # 对每个因子分别进行标准化
        for j, factor_name in enumerate(condition_list):
            if factor_name in params:  # 只对在参数字典中定义的因子进行标准化
                log = factor_name in ALL_LOG_FACTORS  # 是否对该因子取对数
                factors[:, j] = apply_norm(factors[:, j], factor_name, params, log=log)
    
    ori_plan_onehot = one_hot_lu(ori_plan[0]).unsqueeze(0)

    # 注意concat的顺序！！！！！
    cond = torch.cat((factors, ori_plan_onehot),dim=1).to(device)
    pred_diff_raw = model(cond)
    pred_diff=F.sigmoid(pred_diff_raw)
    pred_diff=torch.where(pred_diff>=0.5, 1, 0)

    return pred_diff.detach().cpu()

def get_cls_restrict_area(ori_plan: torch.Tensor) -> torch.Tensor:
    """
    得到ori_plan里LU类别= 51 or 53的区域, 这些区域不允许改变\n
    Parameters:
    ----------
        ori_plan: tensor, [1,1,128,128], 土地利用方案
    
    Returns:
    -------
        cls_restrict: tensor, [1,1,128,128], 0-1 二值张量，值为1的区域表示该像素的原类型是51或53, 这些区域不允许改变
    """
    area_51 = (ori_plan == 51)
    area_53 = (ori_plan == 53)
    cls_restrict = torch.logical_or(area_51, area_53)
    return cls_restrict

def get_grid_restrict_area(pred_diff: torch.Tensor, grid_num: int = 4) -> torch.Tensor:
    """
    根据pred_diff获取变化区域的格子。变化区域的格子标记为1，否则标记为0。\n
    Parameters:
    ----------
        grid_num: int, 网格大小, default=4
        pred_diff: tensor, [1,1,128,128], 0-1二值张量，0表示没有变化，1表示变化了
    
    Returns:
    --------
        marked_area: tensor, [1,1,128,128], 标记后的张量，包含变化区域的格子标记为1，否则标记为0
    """
    _, _, height, width = pred_diff.shape
    grid_size_h = height // grid_num
    grid_size_w = width // grid_num
    
    marked_area = torch.zeros_like(pred_diff)

    for i in range(0, height, grid_size_h):
        for j in range(0, width, grid_size_w):
            grid = pred_diff[0, 0, i:i+grid_size_h, j:j+grid_size_w]
            if torch.sum(grid) > 0:
                marked_area[0, 0, i:i+grid_size_h, j:j+grid_size_w] = 1

    return marked_area

def get_lu_labels(plan: torch.Tensor) -> List[int]:
    """
    得到ori_plan里所有土地利用类别\n
    Parameters:
    ----------
        plan: tensor, [1,1,128,128], 土地利用方案
    
    Returns:
    -------
        unique_values: List[int], 土地利用类别列表
    """
    # 确保ori_plan是整数类型
    plan = plan.to(torch.int32)
    unique_values = torch.unique(plan).tolist()
    return unique_values 

def get_scale_minmax_dict(num: int) -> Dict[str, Dict[str, float]]:
    """
    得到ori_plan里每个土地利用类别的最小最大缩放因子\n
    Parameters:
    ----------
        num: int, 像素数限制, 用于计算变化区域的最小最大面积
    
    Returns:
    --------
        scale_minmax_dict: Dict[str, Dict[str, float]], 变化区域的最小最大面积, 格式为
            {'small': {'min': 100, 'max': 200},
             'mid': {'min': 200, 'max': 400},
             'large': {'min': 400, 'max': 800}}
    """
    small_area = num * 0.25
    mid_area = num * 0.5
    large_area = num 
    scale_minmax_dict = {
        'small': {'min': 1, 'max': small_area},
        'mid': {'min': small_area, 'max': mid_area},
        'large': {'min': mid_area, 'max': large_area}
    }
    return scale_minmax_dict
    
def get_initial_score_list() -> List[float]:
    """
    获取初始解的分数列表\n
    Returns:
    ----------
        initial_score_list: List, 初始解的分数列表, 包括:
        - tgt_func: 目标函数值
        - change_relative_score: 变化得分
        - es_relative_score: 环境得分
        - es_relative_score_dict: 环境得分字典
        - landscape_score: 景观格局得分
    """
    inf = float('inf')
    return [inf, inf, 0.0, {}, 0.0]
    

def draw_score_list(score_list_data: List[Union[List[float], Dict[str, List[float]]]], title_list: List[str] = None, save_folder: str = None) -> None:
    """
    绘制目标函数项的变化曲线\n
    Parameters:
    ----------
        score_list_data: List[Union[List, Dict[str, List]]], 目标函数项的变化列表, 包括:
            total_score_list: 目标函数值list
            change_relative_score_list: 变化得分list
            es_relative_score_list: 环境得分list
            landscape_score_list: 景观格局得分list
            es_relative_score_list_dict: 环境得分字典dict, 格式为
                {'food': 食物得分list,
                 'carbon': 碳足迹得分list,
                 ...}
        title_list: List[str], 图表标题列表
        save_folder: str, 图表保存文件夹
    """
    score_list = score_list_data[:-1]
    es_score_list_dict = score_list_data[-1]
    for i, score_list in enumerate(score_list):
        length = len(score_list)
        steps = list(range(1,length+1))
        ensure_folder(save_folder)
        save_path = save_folder + f'/{title_list[i]}.png'
        draw_single_line(steps, score_list, title_list[i], 
                        x_label='Step', y_label='score',
                        title=title_list[i],
                        save_path=save_path)
    
    es_length = len(es_score_list_dict['food'])
    es_steps = list(range(1,es_length+1))
    datas = es_score_list_dict.values()
    labels = es_score_list_dict.keys()
    save_path = f'{save_folder}/ES详情.png'
    draw_multi_line(es_steps, datas, labels, 
                            x_label='Step', y_label='score',
                            title='Ecosystem Services', 
                            save_path=save_path)

def draw_score_list_general(score_list_data: List[List[float]], 
                            title_list: List[str] = None, 
                            save_folder: str = None,
                            multi: bool = False) -> None:
    """
    更通用的绘制目标函数项的变化曲线\n
    Parameters:
    ----------
        score_list_data: List[List[float]], 数据列表, 每个元素为一个目标函数项的变化列表
        title_list: List[str], 图表标题列表
        save_folder: str, 图表保存文件夹
        multi: bool, 是否将多条曲线绘制在一张图上
    """
    ensure_folder(save_folder)
    if title_list is None:
        title_list = [f'Score_{i}' for i in range(len(score_list_data))]
    
    if len(score_list_data) != len(title_list):
        raise ValueError("score_list_data and title_list must have the same length")

    if multi:
        # 将多条曲线绘制在一张图上
        # 检查所有数据列表长度是否一致
        data_lengths = [len(data) for data in score_list_data]
        if len(set(data_lengths)) > 1:
            raise ValueError("All data lists must have the same length when multi=True")
        
        length = data_lengths[0]
        steps = list(range(1, length + 1))
        
        save_path = os.path.join(save_folder, 'Multi_Scores.png')
        draw_multi_line(steps, score_list_data, title_list, 
                        x_label='Step', y_label='score',
                        title='Multi Scores', 
                        save_path=save_path)
    else:
        # 将每条曲线分别绘制在单独的图上
        for i, score_data in enumerate(score_list_data):
            length = len(score_data)
            steps = list(range(1, length + 1))
            save_path = os.path.join(save_folder, f'{title_list[i]}.png')
            draw_single_line(steps, score_data, title_list[i], 
                            x_label='Step', y_label='score',
                            title=title_list[i],
                            save_path=save_path)


def save_all_plans(ori_plan: torch.Tensor, best_plan: torch.Tensor, final_plan: torch.Tensor, index: str, 
                save_folder: str, raw_data_root: str = None) -> None:
    """
    保存最佳解、初始解和目标解的土地利用方案为TIFF文件。\n
    Parameters:
    ----------
        ori_plan: tensor, [1,1,128,128], 初始解土地利用方案
        best_plan: tensor, [1,1,128,128], 最佳解土地利用方案
        final_plan: tensor, [1,1,128,128], 目标解土地利用方案
        index: str, 数据索引, 格式为 'year_idx_delta'
        result_path: str, 结果保存路径
        raw_data_root: str, 原始数据根目录, 用于复制TFW文件; 若为None, 则不复制TFW文件
    """
    save_tif(best_plan, output_folder=save_folder,output_filename='best_plan.TIF')
    save_tif(ori_plan, output_folder=save_folder,output_filename='ori_plan.TIF')
    save_tif(final_plan, output_folder=save_folder,output_filename='tgt_plan.TIF')
    # 一个栅格完整的TIFF文件有6种, 使用fragstats分析TIF需要.tfw
    # 需要TFW文件的名称和TIF的名称完全对应，需要复制3份并重命名。
    year = index.split("_")[0]
    idx = index.split("_")[1]
    if raw_data_root is not None:
        other_tiff_folder = os.path.join(raw_data_root, f"LU{year}")
        other_tiff_path = [os.path.join(other_tiff_folder, f) for f in os.listdir(other_tiff_folder) if f.startswith(f"LU{year}_{idx}.") and f.endswith(".tfw")]
        assert len(other_tiff_path)==1
        for path in other_tiff_path:
            shutil.copy(path, os.path.join(save_folder, 'best_plan.tfw'))
            shutil.copy(path, os.path.join(save_folder, 'ori_plan.tfw'))
            shutil.copy(path, os.path.join(save_folder, 'tgt_plan.tfw'))
