# 生成随机解——遗传算法
import random
from typing import Dict, Any, List
import numpy as np
import torch
from .sa_generator import area_wise_change, generate_new_plan
from utils.tools import repeat

def initial_new_population(
        pop_size:int,
        scale_ratio:List[float],
        ori_plan:torch.Tensor,
        scale_minmax_dict:Dict, 
        lu_labels:List,
        cls_restrict:bool, 
        hard_restrict:bool,
        valid_anchors:Dict = None,
        valid_integral:torch.Tensor = None,
        valid_prob:float = 0.0) -> List[torch.Tensor]:
    """
    在当前方案的基础上生成一组新的方案，作为遗传算法的初始种群.
    新方案的生成方式同SA的`generate_new_plan`函数，.

    Parameters:
    ----------
        pop_size: int, 种群大小
        scale_ratio: List[float], 变化区域的缩放比例, 格式为
            [small_ratio, mid_ratio, large_ratio]
        ori_plan: tensor, [1,1,128,128], 原始土地利用情况
        scale_minmax_dict: Dict, 变化区域的最小最大面积, 格式为
            {'small': {'min': 100, 'max': 200},
             'mid': {'min': 200, 'max': 400},
             'large': {'min': 400, 'max': 800}}
        lu_labels: List, 土地利用类别列表
        cls_restrict: tensor, [1,128,128], 0-1,
            类别约束: 51 53类别无法修改为非5类别
        hard_restrict: tensor, [1,128,128], 0-1
            硬约束: 1表示可以修改的区域
        valid_anchors: Dict[str, tensor], optional, 各尺度合法锚点集,
            None 时走旧的拒绝采样路径
        valid_integral: tensor, optional, [H+1, W+1] 有效区域积分图
        valid_prob: float, 从合法锚点集采样的概率
    
    Returns:
    --------
        new_plan_list: List[tensor], 经过调整的土地利用图list, 每个元素为[1,1,128,128]
    """
    # 根据scale_ratio, 分配每个scale对应的数量
    ratio_sum = sum(scale_ratio)
    small_ratio, mid_ratio, large_ratio = [r / ratio_sum for r in scale_ratio]
    small_num = int(pop_size * small_ratio)
    mid_num = int(pop_size * mid_ratio)
    large_num = pop_size - small_num - mid_num - 1
    
    new_plan_list = []
    new_plan_list.append(ori_plan)  # 原始方案

    scale_configs = [
        ('small', small_num),
        ('mid', mid_num),
        ('large', large_num)
    ]
    for scale_type, num in scale_configs:
        for _ in range(num):
            # 使用repeat函数进行多次area_wise_change
            new_plan = repeat(
                area_wise_change,
                repeat_num=2,  # 对ori_plan连续应用2次area_wise_change，生成初始个体
                replace_element=ori_plan,
                scale=scale_type, 
                scale_minmax_dict=scale_minmax_dict, 
                lu_labels=lu_labels,
                cls_restrict=cls_restrict, 
                hard_restrict=hard_restrict,
                valid_anchors=valid_anchors,
                valid_integral=valid_integral,
                valid_prob=valid_prob
            )
            new_plan_list.append(new_plan)
        print(f"Init: Generated {num} {scale_type}-wise plans.")
    print(f"Init: Total {pop_size} plans generated.")
    
    return new_plan_list

def crossover(parent1:torch.Tensor, parent2:torch.Tensor,
              gen_ratio:float,
              hard_restrict:torch.Tensor,
              cls_restrict:torch.Tensor = None,
              max_ratio=0.8,
              min_ratio=0.2):
    """
    crossover: 对两个父代方案进行交叉操作，生成两个子代方案.
    交叉操作的方式为：随机选择一个矩形区域，将父代方案在该区域内的像素交换，得到两个子代方案.

    Parameters:
    ----------
        parent1: tensor, [1,1,128,128], 第一个父代方案
        parent2: tensor, [1,1,128,128], 第二个父代方案
        gen_ratio: float, 遗传代数, 范围[0,1], 越大表示crossover区域越小
        hard_restrict: tensor, [1,1,128,128], 0-1张量, 1表示可以修改的区域
            用于限制crossover区域，只能在可修改区域内进行交叉
        cls_restrict: tensor, [1,1,128,128], 0-1张量, optional, 类别约束
            1表示该像素是51/53类型，只能改为5类（建设用地）；为None时不检查
        max_ratio: float, optional, 最大交叉比例, default=0.8
        min_ratio: float, optional, 最小交叉比例, default=0.2

    Returns:
    --------
        children: List[tensor], 两个子代方案, 每个元素为[1,1,128,128]
    """

    # current_ratio决定了交叉区域的上限，随代数减小
    current_ratio = max_ratio * (1 - gen_ratio) + \
                    min_ratio * gen_ratio

    child1 = parent1.clone()
    child2 = parent2.clone()

    H, W = parent1.squeeze().shape
    
    # 使用hard_restrict限制交换矩形的位置
    valid_area_selected = False
    count = 0
    max_attempts = 1000  # 最大尝试次数
    
    while not valid_area_selected and count < max_attempts:
        # crossover矩形区域的高度和宽度、位置
        rect_h = random.randint(5, int(H * current_ratio))
        rect_w = random.randint(5, int(W * current_ratio))
        top = random.randint(0, H - rect_h)
        left = random.randint(0, W - rect_w)
        
        # 检查选中的区域是否包含不能被修改的像素
        selected_area_hard_restrict = hard_restrict[0, 0, top:top+rect_h, left:left+rect_w]
        if torch.all(selected_area_hard_restrict):
            valid_area_selected = True
        count += 1
    
    if not valid_area_selected:
        # 如果无法找到有效区域，使用原始方法（不限制）
        print("Warning: Unable to find valid crossover area after 1000 attempts. Using unrestricted crossover.")
        rect_h = random.randint(5, int(H * current_ratio))
        rect_w = random.randint(5, int(W * current_ratio))
        top = random.randint(0, H - rect_h)
        left = random.randint(0, W - rect_w)

    # 保存父本1在矩形区域的值，用于后续交换
    p1_rect = child1[:, :, top:top+rect_h, left:left+rect_w].clone()
    p2_rect = child2[:, :, top:top+rect_h, left:left+rect_w]

    if cls_restrict is not None:
        # cls_restrict: 1表示该像素为51/53类型，只能改为5类（建设用地）
        cls_rect = cls_restrict[:, :, top:top+rect_h, left:left+rect_w]  # [1,1,h,w]
        valid_5 = torch.tensor([51, 52, 53], device=cls_rect.device)

        # child1 接收 parent2 的值，但保护 cls_restrict==1 且 incoming 值不在 [51,52,53] 的像素
        protect1 = (cls_rect == 1) & ~torch.isin(p2_rect, valid_5)
        child1[:, :, top:top+rect_h, left:left+rect_w] = torch.where(protect1, p1_rect, p2_rect)

        # child2 接收 parent1 的值，同理
        protect2 = (cls_rect == 1) & ~torch.isin(p1_rect, valid_5)
        child2[:, :, top:top+rect_h, left:left+rect_w] = torch.where(protect2, p2_rect, p1_rect)
    else:
        # 无 cls_restrict 时，直接交换整个矩形区域
        child1[:, :, top:top+rect_h, left:left+rect_w] = p2_rect
        child2[:, :, top:top+rect_h, left:left+rect_w] = p1_rect

    return child1, child2

def mutate(individual:torch.Tensor, **kwargs):
    """
    mutate: 对一个个体进行变异操作，返回变异后的个体.
    变异操作的方式为：随机选择一个矩形区域，将该区域内的像素随机改变为其他类别.

    Parameters:
    ----------
        individual: tensor, [1,1,128,128], 一个个体方案
        gen_ratio: float, 遗传代数, 范围[0,1]
        **kwargs: Dict, 其他generate_new_plan所需的参数, 格式为
            ori_plan: tensor, [1,1,128,128], 原始土地利用情况
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
    
    Returns:
    --------
        mutated_individual: tensor, [1,1,128,128], 变异后的个体方案
    """

    mutated_individual = repeat(
        generate_new_plan,
        repeat_num=2,  # 对个体连续应用2次generate_new_plan扰动
        replace_element=individual,
        **kwargs
    )
    
    return mutated_individual

