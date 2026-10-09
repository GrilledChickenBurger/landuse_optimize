# 生成随机解——模拟退火算法
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import torch

def _build_integral_image(valid_mask: torch.Tensor) -> torch.Tensor:
    """
    构建有效区域掩膜的积分图(summed-area table), 用于 O(1) 校验任意矩形是否完全合法.\n
    Parameters:
    ----------
        valid_mask: tensor, [H, W], 1 表示可以修改的像素
    Returns:
    --------
        ii: tensor, [H+1, W+1], 首行首列为 0 的积分图.
            矩形 (sy,sx,ey,ex) 内合法像素数 = ii[ey,ex]-ii[sy,ex]-ii[ey,sx]+ii[sy,sx],
            当且仅当该值 == (ey-sy)*(ex-sx) 时矩形完全合法.
    """
    ii = valid_mask.float().cumsum(0).cumsum(1)
    ii = torch.nn.functional.pad(ii, (1, 0, 1, 0))
    return ii


def _rect_fully_valid(ii: torch.Tensor, sy: int, sx: int, ey: int, ex: int) -> bool:
    """用积分图 O(1) 校验矩形 (sy,sx,ey,ex) 是否完全合法."""
    s = ii[ey, ex] - ii[sy, ex] - ii[ey, sx] + ii[sy, sx]
    return float(s) == float((ey - sy) * (ex - sx))


def build_valid_anchor_sets(ori_plan: torch.Tensor, hard_restrict: torch.Tensor,
                            scale_minmax_dict: Dict) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    预计算每个尺度下的合法锚点(矩形左上角)集合与积分图, 用于受控采样.\n
    锚点 (y,x) 合法 当且仅当 以它为左上角、边长为该尺度最大边长 s_max 的正方形完全落在
    hard_restrict 内且不含 LU=0 像素; 这样的锚点必能容纳任意 h,w <= s_max 的矩形.\n
    由于 hard_restrict 固定且 LU=0 像素永不改写, 本集合在整个优化过程中保持合法, 只需构建一次.\n
    Parameters:
    ----------
        ori_plan: tensor, [1,1,H,W], 原始土地利用情况
        hard_restrict: tensor, [1,1,H,W], 0-1, 1表示可以修改的区域
        scale_minmax_dict: Dict, 变化区域的最小最大面积
    Returns:
    --------
        anchor_sets: Dict[str, LongTensor], {'small': [N1,2], 'mid': [N2,2], 'large': [N3,2]},
            每个元素是该尺度合法锚点的 (y,x) 坐标集, 可能为空张量
        ii: tensor, [H+1, W+1], 有效区域积分图
    """
    valid_mask = torch.logical_and(hard_restrict[0, 0], ori_plan[0, 0] != 0)
    ii = _build_integral_image(valid_mask)
    _, _, height, width = ori_plan.shape

    anchor_sets = {}
    for scale in ['small', 'mid', 'large']:
        s_max = int(math.sqrt(scale_minmax_dict[scale]['max'])) + 1
        s_max = min(s_max, height, width)
        # 向量化腐蚀: 批量计算所有锚点的 s_max×s_max 窗口合法像素数
        window_sum = (ii[s_max:height+1, s_max:width+1]
                      - ii[0:height-s_max+1, s_max:width+1]
                      - ii[s_max:height+1, 0:width-s_max+1]
                      + ii[0:height-s_max+1, 0:width-s_max+1])
        anchors = torch.nonzero(window_sum == s_max * s_max)  # [N, 2], (y, x)
        anchor_sets[scale] = anchors
    return anchor_sets, ii


def generate_new_plan(cur_plan:torch.Tensor, ori_plan:torch.Tensor,
                      T_ratio:float, noimprove_step:int,
                      scale_minmax_dict:Dict, lu_labels:List,
                      cls_restrict:bool, hard_restrict:bool,
                      valid_anchors:Optional[Dict[str, torch.Tensor]]=None,
                      valid_integral:Optional[torch.Tensor]=None,
                      valid_prob:float=0.0):
    """
    在当前方案的基础上生成新的方案，并返回变化区域的二值张量.\n
    Parameters:
    ----------
        cur_plan: tensor, [1,1,128,128], 当前土地利用方案
        ori_plan: tensor, [1,1,128,128], 原始土地利用情况
        T_ratio: float, 当前温度 / 初始温度 t_0
        noimprove_step: int, 目标函数已有多少步未被优化
        scale_minmax_dict: Dict, 变化区域的最小最大面积, 格式为
            {'small': {'min': 100, 'max': 200},
             'mid': {'min': 200, 'max': 400},
             'large': {'min': 400, 'max': 800}}
        lu_labels: List, 土地利用类别列表
        cls_restrict: tensor, [1,128,128], 0-1,
            类别约束: 51 53类别无法修改为非5类别
        hard_restrict: tensor, [1,128,128], 0-1
            硬约束: 不可修改的区域
        valid_anchors: Dict[str, tensor], optional, 各尺度合法锚点集,
            由 build_valid_anchor_sets() 生成; None 时走旧的拒绝采样路径
        valid_integral: tensor, optional, [H+1, W+1] 有效区域积分图
        valid_prob: float, 以该概率从合法锚点集生成矩形(必中),
            以 1-valid_prob 全图单次采样(不合法则返回空 move)
    
    Returns:
    --------
        new_plan: tensor, [1,1,128,128], 经过调整的土地利用图;
        d_diff: tensor, [1,1,128,128], 本次发生调整的位置
    """
    # 根据温度设置change_type的概率
    large_prob = 0.4 * T_ratio + 0.2  
    if noimprove_step>=30 or T_ratio<=0.1:
        return_prob = 0.2
        mid_prob = 0.2
    else:
        return_prob = 0.0
        mid_prob = 0.3
    small_prob = 1.0 - return_prob - mid_prob - large_prob 

    # 随机选择一种变化方式
    change_type = np.random.choice(['small', 'mid', 'large', 'return'], 
                                p=[small_prob, mid_prob, large_prob, return_prob])
    # print(f"noimprove_step={noimprove_step}. Choose a {change_type}-wise move.")

    # 根据选定的变化方式生成新的方案
    if change_type == 'return':
        return area_wise_return(cur_plan, ori_plan, T_ratio)
    else:
        return area_wise_change(cur_plan, scale=change_type, scale_minmax_dict=scale_minmax_dict, lu_labels=lu_labels,
                                cls_restrict=cls_restrict, hard_restrict=hard_restrict,
                                valid_anchors=valid_anchors, valid_integral=valid_integral,
                                valid_prob=valid_prob)


def area_wise_change(cur_plan: torch.Tensor, scale: str, scale_minmax_dict: Dict, lu_labels: List,
                        cls_restrict: torch.Tensor, hard_restrict: torch.Tensor,
                        valid_anchors: Optional[Dict[str, torch.Tensor]] = None,
                        valid_integral: Optional[torch.Tensor] = None,
                        valid_prob: float = 0.0):
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
        valid_anchors: Dict[str, tensor], optional, 各尺度合法锚点集,
            提供时启用受控采样(零重试): 以 valid_prob 从锚点集生成(必中),
            否则全图单次采样, 不合法则返回空 move
        valid_integral: tensor, optional, [H+1, W+1] 有效区域积分图
        valid_prob: float, 从合法锚点集采样的概率

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

    if valid_anchors is not None:
        # 受控采样路径: 全程单次, 零重试
        min_length = int(math.sqrt(min_area))
        max_length = int(math.sqrt(max_area)) + 1
        if max_length <= min_length:
            max_length = min_length + 1
        anchors = valid_anchors[scale]
        if len(anchors) > 0 and np.random.rand() < valid_prob:
            # 以 valid_prob 概率: 从合法锚点集取角点, 当场随机边长(≤ s_max 必合法)
            anchor = anchors[np.random.randint(len(anchors))]
            start_y, start_x = int(anchor[0]), int(anchor[1])
            s_max = min(max_length, height - start_y, width - start_x)
            if s_max <= min_length:
                s_max = min_length + 1
            end_y = start_y + int(torch.randint(min_length, s_max, (1,)))
            end_x = start_x + int(torch.randint(min_length, s_max, (1,)))
        else:
            # 以 1-valid_prob 概率: 全图均匀单次采样, 不合法则返回空 move
            start_x = int(torch.randint(0, width, (1,)))
            start_y = int(torch.randint(0, height, (1,)))
            end_x = min(start_x + int(torch.randint(min_length, max_length, (1,))), width)
            end_y = min(start_y + int(torch.randint(min_length, max_length, (1,))), height)
            if end_y <= start_y or end_x <= start_x or \
                    not _rect_fully_valid(valid_integral, start_y, start_x, end_y, end_x):
                d_change_area = torch.zeros_like(cur_plan)
                return cur_plan, d_change_area
    else:
        # hard restrict: 随机选择一个区域，确保区域内像素都可以被修改
        valid_area_selected = False
        count = 0  # 在循环超过1000次还未找到新方案时，将min max均修改成次一级
        while not valid_area_selected:
            start_x = torch.randint(0, width, (1,))
            start_y = torch.randint(0, height, (1,))
            # 根据最小面积和最大面积选择矩形的长度和宽度
            max_length = int(torch.sqrt(torch.tensor(max_area)).item())+1
            min_length = int(torch.sqrt(torch.tensor(min_area)).item())

            # 确保max_length至少比min_length大1（解决torch.randint参数问题）
            if max_length <= min_length:
                max_length = min_length + 1

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

        start_x = int(start_x)
        start_y = int(start_y)
        end_x = int(end_x)
        end_y = int(end_y)


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
