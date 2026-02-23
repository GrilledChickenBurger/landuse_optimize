import numpy as np
import torch
import cv2
import time
from typing import Dict, Any, Tuple

from .cal_change_score import cal_change_score
from .cal_es_score import cal_es_score
from .cal_landscape_score import cal_landscape_score

def cal_tgt_function(cur_plan: torch.Tensor, 
                    cur_diff: torch.Tensor,
                    pred_diff: torch.Tensor, 
                    ori_plan: torch.Tensor,
                    info_dict: Dict[str, Any],
                    es_weight_dict: Dict[str, float],
                    cur_temp: float, 
                    use_landscape: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], float]:
    """
    计算退火算法目标函数
    
    Parameters:
        cur_plan: torch.Tensor, 当前土地利用方案, 形状为 [1, 1, 128, 128]
        cur_diff: torch.Tensor, 当前变化区域, 形状为 [1, 1, 128, 128]
        pred_diff: torch.Tensor, 预测变化区域, 形状为 [1, 1, 128, 128]
        info_dict: Dict[str, Any], 包含所有必要的参数
        es_weight_dict: Dict[str, float], 环境得分权重字典
        cur_temp: float, 当前温度
        use_landscape: bool, optional, 是否使用景观格局约束
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], float]: 
        tgt_func, change_score, es_score, es_scores(dict), landscape_score
    """
    eval_start_time = time.time()
    
    # 1. 计算变化得分 越大表示越好, 范围:[-inf, 1]
    change_relative_score = cal_change_score(cur_diff, pred_diff, info_dict)
    # 2. 计算环境得分  越大表示越好, 范围:[0, 1]
    es_ori_score_dict, es_relative_score_dict, es_relative_score = cal_es_score(cur_plan, info_dict, es_weight_dict)
    # 3. 计算景观格局得分
    landscape_score = 0.0
    if use_landscape:
        landscape_score = cal_landscape_score(cur_plan,cur_diff, ori_plan, info_dict = info_dict)
    
    # 4. 计算目标函数: 越小表示方案越优秀
    if torch.count_nonzero(pred_diff) > 6000:
        lambda_ES = 1000
        lambda_LS = 5
    elif torch.count_nonzero(pred_diff) < 300:
        lambda_ES = 5000
        lambda_LS = 1
    else:
        lambda_ES = 5000 + (1000 - 5000) * (torch.count_nonzero(pred_diff) - 300) / (6000 - 300)
        lambda_LS = 1 + (5 - 1) * (torch.count_nonzero(pred_diff) - 300) / (6000 - 300)

    eval_end_time = time.time()
    print(f"eval time: {eval_end_time - eval_start_time:.4f}")
    
    tgt_func = change_relative_score - lambda_ES * es_relative_score + lambda_LS * landscape_score
    return tgt_func, change_relative_score, es_relative_score, es_relative_score_dict, landscape_score





