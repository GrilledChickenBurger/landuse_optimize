import torch
import numpy as np
import scipy.ndimage
from typing import Dict, Any

"""
核心函数1:
    scipy.ndimage.distance_transform_edt: 计算图像中非零点到最近背景点（即0）的距离。
核心函数2:
    scipy.ndimage.label: 标记图像中的连通区域
"""

def cal_change_score(cur_diff: torch.Tensor, pred_diff: torch.Tensor, info_dict: Dict[str, Any]) -> torch.Tensor:
    """
    计算变化得分, 表示当前更改区域到预测变化区域的距离【相对值】, 越大表示越好, 范围:[-inf, 1]\n
    Parameters:
    ----------
        cur_diff: tensor, [1,1,128,128], 0-1 binary, 1 表示新方案对比原方案的变化区域,
        pred_diff: tensor, [1,1,128,128], 0-1 binary, 1 表示预测方案对比原方案的变化区域
        info_dict: Dict[str, Any], 包含预测变化区域距离地图和最小变化得分等信息
    
    Returns:
    -------
        change_score_relative: torch.Tensor, 变化得分, 表示当前更改区域到预测变化区域的距离
    """
    change_value = cal_change_value(cur_diff, pred_diff, info_dict['pred_diff_distance_map'])
    # min_change_score = info_dict['min_change_score']
    # change_score_relative = change_value / min_change_score
    return change_value


def cal_change_value(cur_diff: torch.Tensor, pred_diff: torch.Tensor, pred_diff_distance_map: torch.Tensor) -> torch.Tensor:
    """
    第1项: 衡量当前更改区域到real_change_area区域之间的距离【绝对值】.\n
    Parameters:
    -------
        cur_diff: tensor, [1,1,128,128], 0-1 binary, 1 表示新方案对比原方案的变化区域,
        pred_diff: tensor, [1,1,128,128], 0-1 binary, 1 表示预测方案对比原方案的变化区域

    Returns:
    -------
        change_value: tensor, 变化值, 表示当前更改区域到预测变化区域的距离【绝对值】
    """
        # 将输入的tensor转换为numpy array
    cur_diff_np = cur_diff.squeeze().numpy()
    pred_diff_np = pred_diff.squeeze().numpy()
    pred_diff_distance_map_np = pred_diff_distance_map.squeeze().numpy()

    # 标记连通区域
    labeled_array, num_features = scipy.ndimage.label(cur_diff_np)

    total_distance = 0
    # 遍历所有连通区域
    for i in range(1, num_features + 1):
        # 提取当前连通区域的像素位置
        cur_area = np.where(labeled_array == i, 1, 0)
        # 计算当前连通区域的面积
        area = np.sum(cur_area)
        # 计算当前连通区域到pred_diff的距离
        distance, intersection_area = cal_distance(cur_area, pred_diff_np, pred_diff_distance_map_np)
        # cur_area 和 pred_diff 完全不重叠
        if distance > 0 and intersection_area == 0:
            total_distance += (area ** 2) * distance 
        elif distance > 0 and intersection_area > 0:
            diff_area = area - intersection_area
            assert diff_area > 0
            total_distance += ((diff_area ** 2) * distance - (intersection_area ** 2) * 10)
        # cur_area 完全在 pred_diff 内
        elif distance == 0:
            total_distance -= (area ** 2) * 10 

    return torch.tensor(total_distance)


def cal_distance(cur_diff: np.ndarray, pred_diff: np.ndarray, pred_diff_distance_map: np.ndarray) -> tuple:
    """
    计算单个变化区域到预测变化区域的距离
    
    Parameters:
        cur_diff: nparray, [128,128], 0-1 binary, 1 表示新方案对比原方案的变化区域 A
        pred_diff: nparray, [128,128], 0-1 binary, 1 表示预测方案对比原方案的变化区域 B
        pred_diff_distance_map: nparray, [128,128], 预测变化区域到最近背景点的距离地图  
    """
    # 计算A和B的并集和交集
    union = np.logical_or(cur_diff, pred_diff)
    intersection = np.logical_and(cur_diff, pred_diff)

    # 计算A和B的并集与交集的差集
    diff = np.logical_and(union, np.logical_not(intersection))

    # 如果A完全在B内（即A和B的交集就是A），则距离为0
    if np.array_equal(intersection, cur_diff):
        return 0, np.count_nonzero(intersection)

    # 如果A和B完全不重叠，则计算A到B的最长距离
    if not intersection.any():
        max_distance = np.max(cur_diff * pred_diff_distance_map)
        min_distance = np.min(cur_diff * pred_diff_distance_map)
        return (max_distance + min_distance), np.count_nonzero(intersection)
    
    # # 如果B完全在A内（即A和B的交集就是B），则距离为一个较大的常数
    # if np.array_equal(intersection, pred_diff):
    #     return 5, np.count_nonzero(intersection)  # 可以根据实际情况调整这个值

    # 否则，计算差集部分到B的最长距离
    max_distance = np.max(diff * pred_diff_distance_map)
    min_distance = np.min(diff * pred_diff_distance_map)

    return (max_distance + min_distance), np.count_nonzero(intersection)

def get_change_min_score(pred_diff: torch.Tensor, pred_diff_distance_map: torch.Tensor) -> torch.Tensor:
    """
    计算变化得分的最小值，便于后续归一化\n
    Parameters:
    ----------
        pred_diff: torch.Tensor, 预测变化区域，形状为 [1, 1, 128, 128]
        pred_diff_distance_map: torch.Tensor, 预测变化区域到最近背景点的距离地图，形状为 [1, 1, 128, 128]
    
    Returns:
    -------
        min_change_score: float, 最小变化得分, 表示最佳变化区域
    """
    return cal_change_value(pred_diff, pred_diff, pred_diff_distance_map)