# 配置文件加载与解析工具

import os
import yaml
import argparse
from typing import Dict


# 评分相关 magic number 的默认配置, 数值必须与 calculator/runner_utils/const 中硬编码值一致
# 旧 pipeline 不传 score_cfg 时由 get_default_score_cfg() 兜底, 保证行为不变
_DEFAULT_SCORE_CFG_DICT = {
    "tgt_function": {
        "lambda_ES_high": 1000,
        "lambda_ES_low": 5000,
        "lambda_LS_high": 5,
        "lambda_LS_low": 1,
        "pixel_high": 2000,
        "pixel_low": 300,
    },
    "change_score": {
        "inside_reward_coef": 10,
    },
    "runner_utils": {
        "sigmoid_threshold": 0.5,
        "grid_num": 4,
    },
    "landscape": {
        "pure_base": 0.3,
        "pure_change": 0.2,
        # "neighbor_weight": 1.0,  # [预留] v2 minimal 用，当前主流程使用原版 landscape score，未启用
    },
    "const": {
        "pixel_num_threshold_ischange": 30,
        # LU 二级类粒度; 与 libs/const.WATER_SUPPLY_PARAM 一一对应
        "water_supply_param": {
            11: 0.3470,  12: 0.4969,
            21: 0.02038, 22: 0.0420,  23: 0.1920, 24: 0.1222,
            31: 0.0478,  32: 0.08785, 33: 0.1827,
            41: 0.0, 42: 0.0, 43: 0.0, 44: 0.0, 45: 0.0, 46: 0.0,
            51: 1.0, 52: 1.0, 53: 1.0,
            61: 1.0, 62: 1.0, 63: 1.0, 64: 1.0, 65: 1.0, 66: 1.0, 67: 1.0,
        },
    },
    "ablation": {
        "guided": True,
        "disable_change_score": False,
        "disable_early_stop": False,
        "quantity_penalty_C": 10000,
        "quantity_penalty_delta": 0.1,
        "quantity_penalty_power": 2,
    },
}


def get_namespace(config_dict: Dict) -> argparse.Namespace:
    """递归将 dict 转为 argparse.Namespace，支持点号访问"""
    config = argparse.Namespace()
    for key, value in config_dict.items():
        key_str = str(key)  # 确保属性名为字符串（YAML 可能产生 int 键）
        if isinstance(value, dict):
            value_new = get_namespace(value)
        else:
            value_new = value
        setattr(config, key_str, value_new)
    return config


def get_yaml(path: str) -> argparse.Namespace:
    """读取 YAML 配置文件并返回 Namespace 对象"""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = get_namespace(config)
    return config


def get_default_score_cfg() -> argparse.Namespace:
    """返回默认的评分配置 (兜底旧 pipeline 不传 score_cfg 的场景)"""
    return get_namespace(_DEFAULT_SCORE_CFG_DICT)


def load_score_cfg(path: str) -> argparse.Namespace:
    """从 YAML 文件加载评分配置"""
    return get_yaml(path)
