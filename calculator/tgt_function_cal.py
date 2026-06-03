import numpy as np
import torch
import cv2
import time
from typing import Dict, Any, Tuple, Optional
from argparse import Namespace

from .cal_change_score import cal_change_score
from .cal_es_score import cal_es_score
from .cal_landscape_score import cal_landscape_score
from utils.tools import get_default_score_cfg

def cal_tgt_function(cur_plan: torch.Tensor,
                    cur_diff: torch.Tensor,
                    pred_diff: torch.Tensor,
                    ori_plan: torch.Tensor,
                    info_dict: Dict[str, Any],
                    es_weight_dict: Dict[str, float],
                    use_landscape: bool = False,
                    T_ratio: float = 0.0,
                    score_cfg: Optional[Namespace] = None,
                    change_scale_n: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], float]:
    """
    计算退火算法目标函数

    Parameters:
        cur_plan: torch.Tensor, 当前土地利用方案, 形状为 [1, 1, 128, 128]
        cur_diff: torch.Tensor, 当前变化区域, 形状为 [1, 1, 128, 128]
        pred_diff: torch.Tensor, 预测变化区域 (或 C 组的随机目标), 形状为 [1, 1, 128, 128]
        info_dict: Dict[str, Any], 包含所有必要的参数
        es_weight_dict: Dict[str, float], 环境得分权重字典
        T_ratio: float, 当前温度_ratio, 范围[0,1]
        use_landscape: bool, optional, 是否使用景观格局约束
        score_cfg: Namespace, optional, 评分常数配置. None 时使用默认值(=旧硬编码常数)
        change_scale_n: int, optional, 用于 λ 自适应调度的像素数. 为 None 时用 pred_diff 计算.
                       C 组需要传入真实 pred_diff 的像素数, 使 λ 取值基于模型预测而非随机约束.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], float]:
        tgt_func, change_score, es_score, es_scores(dict), landscape_score
    """
    cfg = score_cfg if score_cfg is not None else get_default_score_cfg()

    # 1. 计算变化得分 越大表示越好, 范围:[-inf, 1]
    change_relative_score = cal_change_score(cur_diff, pred_diff, info_dict, score_cfg=cfg)
    # 2. 计算环境得分  越大表示越好, 范围:[0, 1]
    es_ori_score_dict, es_relative_score_dict, es_relative_score = cal_es_score(cur_plan, info_dict, es_weight_dict)
    # 3. 计算景观格局得分
    landscape_score = 0.0
    if use_landscape:
        landscape_score = cal_landscape_score(cur_plan, cur_diff, ori_plan, info_dict=info_dict, score_cfg=cfg)

    # 4. 计算目标函数: 越小表示方案越优秀
    t = cfg.tgt_function
    n = torch.count_nonzero(pred_diff) if change_scale_n is None else change_scale_n
    if n > t.pixel_high:
        lambda_ES = t.lambda_ES_high
        lambda_LS = t.lambda_LS_high
    elif n < t.pixel_low:
        lambda_ES = t.lambda_ES_low
        lambda_LS = t.lambda_LS_low
    else:
        ratio = (n - t.pixel_low) / (t.pixel_high - t.pixel_low)
        lambda_ES = t.lambda_ES_low + (t.lambda_ES_high - t.lambda_ES_low) * ratio
        lambda_LS = t.lambda_LS_low + (t.lambda_LS_high - t.lambda_LS_low) * ratio

    change_term = change_relative_score
    tgt_func = change_term - lambda_ES * es_relative_score + lambda_LS * landscape_score
    return tgt_func, change_relative_score, es_relative_score, es_relative_score_dict, landscape_score





