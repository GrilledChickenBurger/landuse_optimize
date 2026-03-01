# 生成随机解——模拟退火算法
from typing import Dict, Any, List
import numpy as np
import torch

def generate_new_plan(cur_plan:torch.Tensor, ori_plan:torch.Tensor,
                      T_ratio:float, increase_step:int,
                      scale_minmax_dict:Dict, lu_labels:List,
                      cls_restrict:bool, hard_restrict:bool):
    """
    在当前方案的基础上生成新的方案，并返回变化区域的二值张量.\n
    Parameters:
    ----------
        cur_plan: tensor, [1,1,128,128], 当前土地利用方案
        ori_plan: tensor, [1,1,128,128], 原始土地利用情况
        T_ratio: float, 当前温度 / 初始温度 t_0
        increase_step: int, 目标函数已有多少步未被优化
        scale_minmax_dict: Dict, 变化区域的最小最大面积, 格式为
            {'small': {'min': 100, 'max': 200},
             'mid': {'min': 200, 'max': 400},
             'large': {'min': 400, 'max': 800}}
        lu_labels: List, 土地利用类别列表
        cls_restrict: tensor, [1,128,128], 0-1,
            类别约束: 51 53类别无法修改为非5类别
        hard_restrict: tensor, [1,128,128], 0-1
            硬约束: 不可修改的区域        
    
    Returns:
    --------
        new_plan: tensor, [1,1,128,128], 经过调整的土地利用图;
        d_diff: tensor, [1,1,128,128], 本次发生调整的位置
    """
    # 根据温度设置change_type的概率
    large_prob = 0.4 * T_ratio + 0.2  
    if increase_step>=30 or T_ratio<=0.1:
        return_prob = 0.2
        mid_prob = 0.2
    else:
        return_prob = 0.0
        mid_prob = 0.3
    small_prob = 1.0 - return_prob - mid_prob - large_prob 

    # 随机选择一种变化方式
    change_type = np.random.choice(['small', 'mid', 'large', 'return'], 
                                p=[small_prob, mid_prob, large_prob, return_prob])
    # print(f"Increase_step={increase_step}. Choose a {change_type}-wise move.")

    # 根据选定的变化方式生成新的方案
    if change_type == 'return':
        return area_wise_return(cur_plan, ori_plan, T_ratio)
    else:
        return area_wise_change(cur_plan, scale=change_type, scale_minmax_dict=scale_minmax_dict, lu_labels=lu_labels,
                                cls_restrict=cls_restrict, hard_restrict=hard_restrict)


def area_wise_change(cur_plan: torch.Tensor, scale: str, scale_minmax_dict: Dict, lu_labels: List,
                        cls_restrict: torch.Tensor, hard_restrict: torch.Tensor):
    """
    对当前土地利用图进行【区域】级别的变化，返回新的方案和本次变化区域的mask。\n
    Parameters:
    ----------
        cur_plan: tensor, [1,1,128,128], 当前土地利用情况
        scale: 'small', 'mid', 'large', 变化区域的大小
        scale_minmax_dict: Dict, 变化区域的最小最大面积, 格式为
            {'small': {'min': 100, 'max': 200},
             'mid': {'min': 200, 'max': 400},
             'large': {'min': 400, 'max': 800}}
        lu_labels: List, 土地利用类别列表
        cls_restrict: tensor, [1,128,128], 0-1, 1表示不可修改为非5类别的区域
        hard_restrict: tensor, [1,128,128], 0-1, 1表示可以修改的区域

    Returns:
    ----------
        new_plan: tensor, [1,1,128,128], 经过调整的土地利用图
        d_diff: tensor, [1,1,128,128], 本次发生调整的位置
    """
    # 获取土地利用图的尺寸
    _, _, height, width = cur_plan.shape

    # 根据scale参数设置最小和最大面积
    assert scale in ['small', 'mid', 'large']
    scale_idx = ['small', 'mid', 'large'].index(scale)
    min_area = scale_minmax_dict[scale]['min']
    max_area = scale_minmax_dict[scale]['max']
    # hard restrict: 随机选择一个区域，确保区域内像素都可以被修改
    valid_area_selected = False
    count = 0  # 在循环超过1000次还未找到新方案时，将min max均修改成次一级
    while not valid_area_selected:
        start_x = torch.randint(0, width, (1,))
        start_y = torch.randint(0, height, (1,))
        # 根据最小面积和最大面积选择矩形的长度和宽度
        max_length = int(torch.sqrt(torch.tensor(max_area)).item())+1
        min_length = int(torch.sqrt(torch.tensor(min_area)).item())

        end_x = min(start_x + torch.randint(min_length, max_length, (1,)),  width)
        end_y = min(start_y + torch.randint(min_length, max_length, (1,)), height)

        # 检查选中的区域是否包含不能被修改的像素
        selected_area_hard_restrict = hard_restrict[0, 0, start_y:end_y, start_x:end_x]
        # 新增检查：确保选中的区域不包含类别0的像素
        selected_area_original = cur_plan[0, 0, start_y:end_y, start_x:end_x]
        if torch.all(selected_area_hard_restrict) and not (selected_area_original == 0).any():
            valid_area_selected = True
        count+=1

        if count==10000:
            if scale_idx > 0:
                scale = ['small', 'mid', 'large'][scale_idx-1]
                print(f"Unable to generate, change to the previous level: {scale}-wise move.")
                min_area = scale_minmax_dict[scale]['min']
                max_area = scale_minmax_dict[scale]['max']
                scale_idx -= 1
                count = 0
            else:
                # 如果已经是smallest尺度，无法降级，直接返回原方案
                print("Warning: Unable to find valid area after 10000 attempts at smallest scale. Returning original plan.")
                d_change_area = torch.zeros_like(cur_plan)
                return cur_plan, d_change_area


    # 从lu_labels列表中随机选择新的像素值，但排除0值
    # cls_restrict: 原类型是51或53
    # 确保新标签不包含0值
    if lu_labels is None or len(lu_labels) == 0:
        lu_labels = torch.unique(cur_plan).tolist()
    valid_labels = [l for l in lu_labels if l != 0]

    if (cls_restrict[0, 0, start_y:end_y, start_x:end_x] == 1).any():
        restricted_labels = [l for l in [51, 52, 53] if l in valid_labels]
        new_labels = torch.tensor(restricted_labels)[torch.randint(0, len(restricted_labels), (1,))]
    else:
        new_labels = torch.tensor(valid_labels)[torch.randint(0, len(valid_labels), (1,))]

    # 改变这个区域内的所有像素的值
    new_plan = cur_plan.clone()
    new_plan[0, 0, start_y:end_y, start_x:end_x] = new_labels.float().item()
    # 创建一个全0的张量，然后将发生改变的区域设置为1
    d_change_area = torch.zeros_like(cur_plan)
    d_change_area[0, 0, start_y:end_y, start_x:end_x] = 1
    return new_plan, d_change_area



def area_wise_return(cur_plan: torch.Tensor, ori_plan: torch.Tensor, T_ratio: float):   
    """
    对当前土地利用图进行【区域】级别的回退(恢复到原始土地利用情况), 返回新的方案和本次返回区域的mask。\n
    Parameters:
    ----------
        cur_plan: tensor, [1,1,128,128], 当前土地利用方案
        ori_plan: tensor, [1,1,128,128], 原始土地利用情况
        T_ratio: float, 当前温度与初始温度的比例

    Returns:
    ----------
        new_plan: tensor, [1,1,128,128], 经过调整的土地利用图
        d_diff: tensor, [1,1,128,128], 本次发生调整的位置
    """
    # 获取土地利用图的尺寸
    _, _, height, width = cur_plan.shape
    max_area = max(int(100*0.05/T_ratio), 100)
    # 随机选择一个区域
    start_x = torch.randint(0, width, (1,))
    start_y = torch.randint(0, height, (1,))
    max_length = int(torch.sqrt(torch.tensor(max_area)).item())
    end_x = min(min(start_x+max_length, width),torch.randint(start_x.item(), width, (1,)))
    end_y = min(min(start_y+max_length, height),torch.randint(start_y.item(), height, (1,)))

    # 改变这个区域内的所有像素的值
    new_plan = cur_plan.clone()
    new_plan[0, 0, start_y:end_y, start_x:end_x] = ori_plan[0, 0, start_y:end_y, start_x:end_x]
    # 创建一个全0的张量，然后将发生改变的区域设置为1
    d_change_area = torch.zeros_like(cur_plan)
    d_change_area[0, 0, start_y:end_y, start_x:end_x] = 1
    return new_plan, d_change_area
