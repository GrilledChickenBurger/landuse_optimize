# 随机优化算法运行过程中，计算优化指标的相关工具

import os
import csv
import numpy as np
import cv2
import elapid as ela
import scipy.ndimage
import torch
from typing import Dict, List, Any
from libs.const import FOOD_SUPPLY_PARAM
from calculator.cal_change_score import get_change_min_score
from calculator.cal_es_score import get_es_max_score
from utils.train_maxent import train_pipeline
from utils.tools import convert_to_nparray

def load_carbon_pool(file_path: str) -> dict:
    """加载碳密度表，返回一个字典

    Paramters:
    ----------
        file_path: str,碳密度表文件路径

    Returns:
    --------
        dict: 碳密度表，键为land use code，值为对应4个碳库的碳密度字典，
            字典格式为：{'c_above': float, 'c_below': float, 'c_soil': float, 'c_dead': float}
    """
    carbon_pool = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # 使用 land use code（lucode）作为字典的键
            lucode = row['lucode']
            carbon_pool[lucode] = {
                'c_above': float(row['C_above']),
                'c_below': float(row['C_below']),
                'c_soil': float(row['C_soil']),
                'c_dead': float(row['C_dead'])
            }
    return carbon_pool


def train_maxent(root: str, year: str, save_path: str) -> None:
    """
    给定根目录和年份，获取相应的数据路径，训练MaxEnt模型并保存
    Parameters:
        root: str, 拼接后将从目录 "{root}/valid_{year}/_maxent" 下获取数据
        year: str, one of ['10', '15', '20']
        save_path: str, path to save the maxent dataset
    """
    print("================= Prior: MaxEnt Training... =================")
    data_path = f"{root}\\valid_{year}\\_maxent"
    train_pipeline(data_path, save_path)
    print("================= Prior: MaxEnt Training Done. =================")


def load_maxent_model(model_path: str) -> ela.MaxentModel:
    """加载Maxent模型\n
    Parameters:
    ----------
        model_path: str, Maxent模型文件路径

    Returns:
    --------
        ela.MaxentModel: 加载的Maxent模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MaxEnt模型文件不存在: {model_path}")
    
    maxent_model = ela.load_object(model_path)
    if maxent_model is None:
        raise ValueError(f"MaxEnt模型加载失败，文件可能已损坏: {model_path}")
    print(maxent_model.get_params())
    return maxent_model


def get_perimeter(area: np.ndarray) -> float:
    """计算某连通区域的周长，用于计算周长面积比\n
    Parameters:
    ----------
        area: numpy.ndarray, [128,128], 0-1 二值数组，=1的区域就是该连通区域
    
    Returns:
    --------
        perimeter: float, 该连通区域的周长
    """
    cv2.setNumThreads(0)  #设置opencv不使用多进程运行，但这条命令只在本作用域有效。
    cv2.ocl.setUseOpenCL(False)   #设置opencv不使用多进程运行，但这条命令只在本作用域有效。

    pixelcount = np.sum(area)   
    if pixelcount==1: # 单个像素周长视为1
        return 1.0
    
    contours, _ = cv2.findContours(area.copy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = cv2.arcLength(contours[0], True)
    else:
        # 如果某种原因没有找到轮廓（可能是个bug），可以添加错误处理
        perimeter = 1.0

    return perimeter


def get_pred_diff_distance_map(pred_diff: torch.Tensor) -> torch.Tensor:
    """
    计算变化区域的距离地图。变化区域的距离地图是一个矩阵，每个元素表示该像素到最近的0的距离。用于change_score的计算\n
    Parameters:
    ----------
        pred_diff: torch.Tensor, [128,128], 0-1二值数组，0表示没有变化，1表示变化了
    
    Returns:
    --------
        distance_map: torch.Tensor, [128,128], 变化区域的距离地图
    """
    pred_diff_np = convert_to_nparray(pred_diff)
    distance_map = scipy.ndimage.distance_transform_edt(1 - pred_diff_np)
    return torch.from_numpy(distance_map)
    
def calculate_food_supply_param(info_dict: dict, weight_dict: dict = FOOD_SUPPLY_PARAM) -> torch.Tensor:
    """计算某区域的生产潜力综合修订系数，用于计算ES-粮食供给\n
    Parameters:
    ----------
        info_dict: Dict[str, Any], 包含LU、NJW、RAIN、PODU等信息
        weight_dict: Dict[str, float], 包含各项的权重, 例如 {'PODU': 0.3, 'NJW': 0.7}
    
    Returns:
    -------
        total_score_tensor: tensor, [1,1,128,128]
    """
    def get_score(data, bound, score_standard, right_close=True):
        # 定义桶的边界
        boundaries = torch.tensor(bound, dtype=torch.float32)
        # 定义每个桶的评分
        scores = torch.tensor(score_standard, dtype=torch.float32)
        # 使用 torch.bucketize 进行条件判断
        # torch.bucketize 返回的是每个元素在 boundaries 中的桶索引
        bucket_indices = torch.bucketize(data, boundaries, right=right_close)
        # 根据桶索引获取评分
        final = scores[bucket_indices]
        return final

    param_dict = {}
    for key, weight in weight_dict.items():
        if key == "PODU":
            param_score = get_score(info_dict[key], [2,6,15,25], [100, 80, 60, 40, 20])
        elif key == "GD":
            param_score = get_score(info_dict[key]*0.001, [0.5,1,1.5], [100, 75, 50, 20])
        elif key == "WATER":
            param_score = get_score(info_dict[key]*0.001, [0.2,0.5,1], [100, 75, 50, 20])
        elif key == "DL":
            param_score = get_score(info_dict[key]*0.1, [0.2,0.5,1,1.5], [100, 80, 60,35, 10])
        elif key == "CITY":
            param_score = get_score(info_dict[key]*0.001, [1,3,5,10], [10,30,60,80,100], right_close=False)
        
        param_dict[key] = param_score * weight

    total_score_tensor = sum(param_dict.values())
    # utils.show_array_result(total_score, 'total_score')
    return total_score_tensor

def get_info_dict(ori_plan: torch.Tensor,
                    pred_diff: torch.Tensor,
                    data_dict: Dict[str, torch.Tensor], 
                    carbon_pool_path: str,
                    maxent_model_path: str,
                    maxent_list: List[str]
                    ) -> Dict[str, Any]:
    """
    从es_data中提取每个土地利用类别的信息, 用于es_score的计算\n
    Parameters:
    ----------
        ori_plan: torch.Tensor, 原始土地利用方案，形状为 [1, 1, 128, 128]
        pred_diff: torch.Tensor, 预测变化区域，形状为 [1, 1, 128, 128]
        data_dict: Dict[str, torch.Tensor], 包含所有驱动因子和生态系统服务数据的字典
        carbon_pool_path: str, 碳密度表文件路径
        maxent_model_path: str, maxent模型文件路径
        maxent_list: List[str], maxent模型需要的特征列表

    Returns:
    --------
        info_dict: Dict[str, Any], 包含计算ES得分需要的所有信息, 合法key包括:
            ['RAIN', 'NJW', 'ZSL', 'PODU', 'DL', 'WATER', 'CITY', 'GD', 'carbon_pool', 'maxent_list', 'maxent_model', 'food_supply_param']
    """
    info_dict = data_dict.copy()
    info_dict['carbon_pool'] = load_carbon_pool(carbon_pool_path) # 碳密度表
    info_dict['maxent_list'] = maxent_list # maxent模型需要的特征列表
    info_dict['maxent_model'] = load_maxent_model(maxent_model_path) # maxent模型
    
    info_dict['ori_num_feature_list'] = get_numfeature_dict(ori_plan) # 原始方案里每个LU类的连通区域数
    info_dict['ori_lu_distancemap_dict'] = get_lu_distancemap_dict(ori_plan) # 原始方案里每个LU类的distance map
    info_dict['pred_diff_distance_map'] = get_pred_diff_distance_map(pred_diff) # 变化区域的距离地图
    
    food_supply_score = calculate_food_supply_param(info_dict)
    info_dict['food_supply_param'] = food_supply_score

    # 计算各项生态系统服务最大得分, 用于归一化
    max_food_score, max_carbon_score, max_water_score, max_culture_score = get_es_max_score(ori_plan, pred_diff, info_dict)
    info_dict['max_es_score'] = {
        'food': max_food_score,
        'carbon': max_carbon_score,
        'water': max_water_score,
        'culture': max_culture_score
    }
    
    # 计算变化得分的最小值, 用于归一化
    min_change_score = get_change_min_score(pred_diff, info_dict['pred_diff_distance_map'])
    info_dict['min_change_score'] = min_change_score
    
    return info_dict


def get_numfeature_dict(plan, lu_labels: List[float] = None):
    """
    得到plan里每个土地利用类别的连通区域数, 用于landscape_score的计算\n
    Parameters:
    ----------
        plan: tensor或numpy array, 土地利用方案
        lu_labels: List[float], 土地利用类别列表, 如果为None, 则自动从plan中提取
    
    Returns:
    --------
        numfeatures_dict: Dict, 包含每个土地利用类别的连通区域数, 格式为{lu_type: num_features}
    """
    numfeatures_dict = {}
    
    # 确保plan是numpy数组
    plan = convert_to_nparray(plan)
    lu_types = np.unique(plan).tolist() if lu_labels is None else lu_labels
    
    for lu in lu_types:
        labels, num_features = scipy.ndimage.label(plan == lu)
        numfeatures_dict[lu] = num_features
    return numfeatures_dict

def get_lu_distancemap_dict(plan):
    """得到plan里所有LU类别区域的distance map,以字典的形式返回, 用于landscape_score的计算\n
    Parameters:
    ----------
        plan: tensor或numpy array, 土地利用方案
    
    Returns:
    --------
        all_distance_maps: dict, 包含每个土地利用类别的distance map, 格式为{lu_type: distance_map}
    """
    # 确保plan是numpy数组
    plan = convert_to_nparray(plan)
    lu_types = np.unique(plan).tolist()

    all_distance_maps={}
    for lu in lu_types:
        # 找出plan里lu_i的区域
        plan_lui_area = (plan == lu)
        distance_map = scipy.ndimage.distance_transform_edt(plan_lui_area)
        all_distance_maps[lu]=distance_map

    return all_distance_maps
