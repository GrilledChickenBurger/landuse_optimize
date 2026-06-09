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

def compute_lambdas(score_cfg: Namespace, n: int) -> Tuple[float, float]:
    """预计算 λ_ES 和 λ_LS，基于变化像素数 n 在 [pixel_low, pixel_high] 区间线性插值。
    这些值在优化过程中不变，可提前计算避免每步重复插值。"""
    t = score_cfg.tgt_function
    if n > t.pixel_high:
        return t.lambda_ES_high, t.lambda_LS_high
    elif n < t.pixel_low:
        return t.lambda_ES_low, t.lambda_LS_low
    else:
        ratio = (n - t.pixel_low) / (t.pixel_high - t.pixel_low)
        lambda_ES = t.lambda_ES_low + (t.lambda_ES_high - t.lambda_ES_low) * ratio
        lambda_LS = t.lambda_LS_low + (t.lambda_LS_high - t.lambda_LS_low) * ratio
        return lambda_ES, lambda_LS


def cal_tgt_function(cur_plan: torch.Tensor,
                    cur_diff: torch.Tensor,
                    pred_diff: torch.Tensor,
                    ori_plan: torch.Tensor,
                    info_dict: Dict[str, Any],
                    es_weight_dict: Dict[str, float],
                    use_landscape: bool = False,
                    score_cfg: Optional[Namespace] = None,
                    change_scale_n: Optional[int] = None,
                    lambda_ES: Optional[float] = None,
                    lambda_LS: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], float]:
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
        lambda_ES: float, optional, 预计算的 ES 权重系数. 与 lambda_LS 同时提供时跳过插值.
        lambda_LS: float, optional, 预计算的 LS 权重系数. 与 lambda_ES 同时提供时跳过插值.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, float], float]:
        tgt_func, change_score, es_score, es_scores_relative(dict), es_scores_absolute(dict), landscape_score
    """
    cfg = score_cfg if score_cfg is not None else get_default_score_cfg()

    # 1. 计算变化得分 越小表示越好
    change_relative_score = cal_change_score(cur_diff, pred_diff, info_dict, score_cfg=cfg)
    # 2. 计算环境得分  越大表示越好, 范围:[0, 1]
    es_ori_score_dict, es_relative_score_dict, es_relative_score = cal_es_score(cur_plan, info_dict, es_weight_dict)
    # 3. 计算景观格局得分 越小表示越好
    landscape_score = 0.0
    if use_landscape:
        landscape_score = cal_landscape_score(cur_plan, cur_diff, ori_plan, info_dict=info_dict, score_cfg=cfg)

    # 4. 计算目标函数: 越小表示方案越优秀
    if lambda_ES is not None and lambda_LS is not None:
        pass  # 使用预计算值，跳过插值
    else:
        n = torch.count_nonzero(pred_diff) if change_scale_n is None else change_scale_n
        lambda_ES, lambda_LS = compute_lambdas(cfg, n)

    change_term = change_relative_score
    tgt_func = change_term - lambda_ES * es_relative_score + lambda_LS * landscape_score
    return (tgt_func, change_relative_score, es_relative_score,
            es_relative_score_dict, es_ori_score_dict, landscape_score)





