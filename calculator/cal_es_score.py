import torch
import numpy as np
import elapid as ela
from typing import Dict, List, Optional, Tuple, Any

from utils.tools import convert_to_nparray
from libs.const import *


def cal_es_score(cur_plan: torch.Tensor, info_dict: Dict[str, Any], weight_dict: Dict[str, float]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    """
    计算当前计划的ES得分【相对值】, 越大表示越好, 范围:[0, 1]\n
    Parameters:
    ----------
        cur_plan: torch.Tensor, 当前计划，形状为 [1, 128, 128]
        info_dict: Dict[str, Any], 包含LU、NJW、RAIN、PODU等信息
        weight_dict: Dict[str, float], 包含ES得分各项的权重, 例如 {'food': 0.3, 'carbon': 0.7}
    
    Returns:
    -------
        es_value_dict: Dict[str, torch.Tensor], 包含各项ES得分的原始值，例如 {'food': 10000, 'carbon': 2000}
        es_score_relative: Dict[str, torch.Tensor], 包含各项ES得分的相对值，例如 {'food': 0.5, 'carbon': 0.8}
        es_score_relative_sum: torch.Tensor, 所有ES得分的相对值加权总和，例如 0.71 = 0.5 * 0.3 + 0.8 * 0.7
    """
    es_value_dict = cal_es_value(cur_plan, info_dict, get_es_cal_func_dict())

    es_score_relative = {}
    es_weighted_score_relative= {}

    for es_name, es_value in es_value_dict.items():
        max_score = info_dict['max_es_score'][es_name]
        es_score_relative[es_name] = es_value / max_score
        es_weighted_score_relative[es_name] = es_value / max_score * weight_dict[es_name]
    
    es_weighted_score_relative_sum = torch.tensor(sum(es_weighted_score_relative.values()), dtype=torch.float32)

    return es_value_dict, es_score_relative, es_weighted_score_relative_sum


def cal_es_value(plan: torch.Tensor, info_dict: Dict[str, Any], cal_func_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    计算当前计划的各项ES得分【绝对值】，包括粮食供给、碳汇、生态服务等\n
    Parameters:
    ----------
        plan: torch.Tensor, 当前计划，形状为 [1, 128, 128]
        info_dict: Dict[str, Any], 包含LU、NJW、RAIN、PODU等信息
        cal_func_dict: Dict[str, Any], 包含计算各项ES得分的函数，例如 {'food': calculate_food_supply}
    
    Returns:
    -------
        es_value_dict: Dict[str, torch.Tensor], 包含各项ES得分，例如 {'food': 10000, 'carbon': 2000}
    """
    es_value_dict = {}
    for es_name, cal_func in cal_func_dict.items():
        # 根据函数名称传递相应的参数
        if es_name == 'food':
            # calculate_food_supply(plan, NJW, RAIN, food_supply_param)
            _, es_value_dict[es_name] = cal_func(
                plan,
                info_dict['NJW'],
                info_dict['RAIN'],
                info_dict['food_supply_param']
            )
        elif es_name == 'carbon':
            # calculate_carbon_storage(plan, carbon_pool)
            es_value_dict[es_name] = cal_func(
                plan,
                info_dict['carbon_pool']
            )
        elif es_name == 'water':
            # calculate_water_supply(plan, RAIN, ZSL, water_supply_param)
            _, es_value_dict[es_name] = cal_func(
                plan,
                info_dict['RAIN'],
                info_dict['ZSL'],
            )
        elif es_name == 'culture':
            # calculate_culture_service(plan, maxent_list, maxent_model, info)
            _, es_value_dict[es_name] = cal_func(
                plan,
                info_dict['maxent_list'],
                info_dict['maxent_model'],
                info_dict
            )
        else:
            # 默认处理方式，可以根据需要扩展
            raise ValueError(f"未知的ES类型: {es_name}")
    
    return es_value_dict
    
def get_es_max_score(ori_plan: torch.Tensor, pred_diff: torch.Tensor, info_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算ES的各项指标最大值，便于后续归一化\n
    Parameters:
    ----------
        ori_plan: torch.Tensor, 原始土地利用方案，形状为 [1, 1, 128, 128]
        pred_diff: torch.Tensor, 预测变化区域，形状为 [1, 1, 128, 128]
        info_dict: Dict[str, Any], 包含LU、NJW、RAIN、PODU等信息
    
    Returns:
    -------
        max_food_score: float, 最大粮食供给得分
        max_carbon_score: float, 最大碳汇得分
        max_water_score: float, 最大水源涵养得分
        max_culture_score: float, 最大生态服务得分
    """
    # 1. 最大碳汇得分: 假设所有变化区域都被替换为林地（21）
    tmp1 = ori_plan.clone()
    tmp1[pred_diff==1] = 21.0
    max_carbon_score = calculate_carbon_storage(tmp1, info_dict['carbon_pool'])

    # 2. 最大粮食供给得分: 假设所有变化区域都被替换为水田（11）
    tmp2 = ori_plan.clone()
    tmp2[pred_diff==1] = 11.0
    _, max_food_score = calculate_food_supply(tmp2, info_dict['NJW'], info_dict['RAIN'], info_dict['food_supply_param'])

    # 3. 最大水源涵养得分: 假设所有变化区域都被替换为水体（41）
    tmp3 = ori_plan.clone()
    tmp3[pred_diff==1] = 41.0
    _, max_water_score = calculate_water_supply(tmp3, info_dict['RAIN'], info_dict['ZSL'])

    # 4. 最大生态服务得分: 原始情况
    _, max_culture_score = calculate_culture_service(ori_plan, info_dict['maxent_list'], info_dict['maxent_model'], info_dict)
    max_culture_score = max_culture_score * 1.5
    
    print(f" Max food score = {max_food_score:.4f}")
    print(f" Max carbon score = {max_carbon_score:.4f}")
    print(f" Max water score = {max_water_score:.4f}")
    print(f" Max culture score = {max_culture_score:.4f}")

    return max_food_score, max_carbon_score, max_water_score, max_culture_score



def calculate_food_supply(plan: torch.Tensor, NJW: torch.Tensor, RAIN: torch.Tensor, food_supply_param: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    第2-1项: 计算ES-粮食供给
    ------
    公式：
        WT = 30000 /( 1 + e ^(1.315 - 0.119T))
        WR = 30000 /( 1 - e ^(-0.000664R))
        Wv = 30000 /( 1 - e ^(-0.0009695( v - 20)))
            = 30000 /( 1 - e ^(-0.0009695( 1.05R/[1+(1.05 R/L)2]1/2 - 20)))

        T: 年均气温，栅格
        R: 年均降雨量，栅格
        W: 粮食生产潜力，单位为kg/(ha*a)
    
    Parameters:
    ----------
        plan: tensor, [1,1,128,128]
        NJW: 年均气温, tensor, [1,1,128,128]
        RAIN: 年均降雨量, tensor, [1,1,128,128]
        food_supply_param: tensor, 粮食供给修正系数矩阵
    
    Returns:
    ----------
        food: tensor, [1,1,128,128]
        food_sum: tensor, 单位为kg/a
    """
    # Miami 模型公式
    WT = 30000 / (1 + torch.exp(1.315 - 0.119 * NJW))
    WR = 30000 / (1 - torch.exp(-0.000664 * RAIN))

    # Thornthwaite Memorial 模型公式
    L_ET = 300 + 25 * NJW + 0.05 * NJW**3  # T 计算年均最大蒸散量
    V_ET = 1.05 * RAIN / (1 + (1.05 * RAIN / L_ET) ** 2) ** 0.5  # L R 计算年均蒸散量
    WV = 30000 / (1 - torch.exp(-0.0009695 * (V_ET - 20)))

    W = torch.min(torch.min(WT, WR), WV)

    # 修正系数=food_supply_param
    W_final = W * food_supply_param*0.01

    available_area = torch.logical_or(plan==11, plan==12)  # 筛选出耕地区域
    food = W_final * available_area
    food_sum = torch.sum(food)*PIXEL_AREA_IN_HECTARES

    return food, food_sum


def calculate_carbon_storage(plan: torch.Tensor, carbon_pool: Dict) -> torch.Tensor:
    """第2-2项: 计算ES-碳汇
    ------

    Parameters:
    ----------
        plan: tensor, [1,1,128,128]
        carbon_pool: Dict, 每个土地利用类型的碳密度，格式为{lu: {'c_above': float, 'c_below': float, 'c_soil': float, 'c_dead': float}}
    
    Returns:
    ----------
        carbon_sum: tensor, 单位为t/a
    """
    plan_int = torch.tensor(plan, dtype=torch.int32)
    lu_label = torch.unique(plan_int)

    # 初始化碳储存总量字典
    total_carbon_storage = {'c_above': 0.0, 'c_below': 0.0, 'c_soil': 0.0, 'c_dead': 0.0}
    
    # 遍历方案中每个土地利用类型
    for lu in lu_label:
        pixel_count = torch.sum(plan_int == lu)
        # 计算当前土地利用类型的总面积（公顷）
        total_area_hectares = pixel_count.item() * PIXEL_AREA_IN_HECTARES
        lu_densities = carbon_pool[str(lu.item())]
        # 计算并累加每个碳储存类型的总量
        for storage_type in total_carbon_storage.keys():
            density_per_hectare = lu_densities[storage_type]
            total_carbon_storage[storage_type] += total_area_hectares * density_per_hectare
    carbon_sum = sum(total_carbon_storage.values())
    carbon_sum = torch.tensor(carbon_sum,dtype=torch.float32)
    return carbon_sum


def calculate_water_supply(plan: torch.Tensor, RAIN: torch.Tensor, ZSL: torch.Tensor, R_dict: Dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    第2-3项: 计算ES-水资源供给
    ------
        公式：Q = P - P * R - ET
        P: 年均降雨量，栅格
        ET: 年均蒸发量，栅格
        R: 不同生态系统的地表径流系数

    Parameters:
    ---------
        plan: tensor, [1,1,128,128]
        RAIN: 年均降雨量, tensor, [1,1,128,128]
        ZSL: 年均蒸发量, tensor, [1,1,128,128]
        R_dict: Dict, 水资源供给修正系数, 格式为{lu: float}; None 时延迟读取 libs.const.WATER_SUPPLY_PARAM

    Returns:
    ---------
        Q: tensor, [1,1,128,128]
        Q_sum: tensor, 单位为m3/a
    """
    if R_dict is None:
        # 延迟引用, 让 apply_const_overrides 修改的 const 模块级常量能被本次调用看到
        import libs.const as _const
        R_dict = _const.WATER_SUPPLY_PARAM
    # 不同生态系统的地表径流系数 (按 LU 二级类精确查表)
    plan_int = torch.tensor(plan, dtype=torch.int32)

    # 注意 dtype 必须为 float, 否则 R_dict 中的小数系数 (如 0.347) 会被 int32 截断为 0
    R = torch.zeros_like(plan_int, dtype=torch.float32)
    for lu, r in R_dict.items():
        R[plan_int == int(lu)] = r

    Q = RAIN - RAIN * R - ZSL

    # 每个像素代表的面积转换为平方米 
    pixel_area_in_meters = (30 * 30) 
    Q_sum = torch.sum(Q) * pixel_area_in_meters * 0.001  # 降水量 蒸散量单位均为mm, 需转换为m

    return Q, Q_sum


def calculate_culture_service(plan: torch.Tensor, maxent_list: List[str], maxent_model: ela.MaxentModel, info_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    第2-4项: 计算ES-文化服务
    ----------
    基于Maxent模型预测文化服务 suitability

    Parameters:
    ----------
        plan: tensor.float32, [1,1,128,128]
        maxent_list: List[str], 包含Maxent模型输入的变量名称，例如 ['LU', 'NJW', 'RAIN']
        maxent_model: ela.MaxentModel, 训练好的Maxent模型
        info_dict: Dict[str, Any], 包含LU、NJW、RAIN等信息

    Returns:
    ----------
        culture_service: tensor.float32, [1,1,128,128]
        culture_service_sum: tensor.float32
    """
    raster_array_list = []
    for f in maxent_list:
        if f != 'LU':
            raster_array = info_dict[f]
        else:
            raster_array = plan
        raster_array_list.append(raster_array)

    raster_arrays = torch.cat(raster_array_list, dim=1) # 1,n,128,128
    raster_arrays_np = convert_to_nparray(raster_arrays).squeeze(axis=0)  # n,128,128
    assert raster_arrays_np.shape[0]==len(maxent_list)

    nodata = 0.0
    plan_np = convert_to_nparray(plan)
    nodata_idx = (plan_np == nodata).squeeze(axis=0)    # 1,128,128
    nodata_idx_expand = np.repeat(nodata_idx, len(maxent_list), axis=0) # n,128,128
    pred = ela.apply_model_to_array(maxent_model, raster_arrays_np, nodata, nodata_idx_expand)    # 1,128,128

    # utils.show_array_result(pred.squeeze(axis=0), 'FJMS suitability', 'relative occurrence probability')

    pred_sum = np.sum(pred)
    culture_service = torch.tensor(pred,dtype=torch.float32).reshape(1,1,128,128)
    culture_service_sum = torch.tensor(pred_sum,dtype=torch.float32)

    return culture_service, culture_service_sum


def get_es_cal_func_dict() -> Dict[str, Any]:
    """
    返回计算ES得分的函数字典，key为ES类型，value为计算函数\n
    Returns:
    --------
        es_cal_func_dict: Dict[str, Any], 包含计算ES得分的函数，例如 {'food': calculate_food_supply}
    """
    es_cal_func_dict = {
        'food': calculate_food_supply,
        'carbon': calculate_carbon_storage,
        'water': calculate_water_supply,
        'culture': calculate_culture_service
    }
    return es_cal_func_dict