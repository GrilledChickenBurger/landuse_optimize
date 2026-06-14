# 得到SA|GA结果后的分析工具函数

import os
import json
from typing import List, Dict, Union
from matplotlib import pyplot as plt
from libs.const import *
from .dataset_utils import load_tiff
from .visualize_utils import *
from .tools import ensure_folder, visualize_tensor_list, find_valid_file

setup_plot_fonts()


def calculate_subplot_layout(num_items: int, max_cols: int = 3, max_rows: int = None, 
                            allow_overflow: bool = False) -> tuple:
    """
    根据子图数量计算合适的子图布局
    
    Parameters:
    -----------
    - num_items: int, 子图数量
    - max_cols: int, 默认3, 最大列数
    - max_rows: int, 默认None, 最大行数（如果为None则根据数量自动计算）
    - allow_overflow: bool, 默认False, 是否允许超出最大行列限制
    
    Returns:
    -----------
    - tuple: (rows, cols) 行数和列数
    """
    if num_items <= 0:
        return 1, 1
    
    # 如果有最大行数限制，优先考虑行数限制
    if max_rows is not None:
        # 计算每行应该有多少列
        cols = (num_items + max_rows - 1) // max_rows
        
        # 如果计算出的列数超过最大列数限制
        if cols > max_cols:
            if allow_overflow:
                # 允许溢出：保持计算出的列数
                rows = max_rows
            else:
                # 不允许溢出：使用最大列数，重新计算行数
                rows = (num_items + max_cols - 1) // max_cols
                cols = max_cols
                # 如果重新计算的行数超过最大行数，则限制为最大行数
                if rows > max_rows:
                    rows = max_rows
                    cols = (num_items + max_rows - 1) // max_rows
        else:
            # 列数在限制范围内，使用最大行数
            rows = max_rows
    else:
        # 没有最大行数限制，使用基础布局计算
        if num_items <= max_cols:
            # 如果数量小于等于最大列数，一行显示
            rows = 1
            cols = num_items
        else:
            # 计算行数：向上取整
            rows = (num_items + max_cols - 1) // max_cols
            cols = max_cols
    
    return rows, cols


def get_performance_metrics(folder_path: str):
    """
    从总实验结果文件夹中读取所有performance_score.json文件，并返回指标字典
    
    Parameters:
    -----------
    - folder_path: 实验结果文件夹路径, 格式为 "results\\SA|GA\\{timestamp}"
    
    Returns:
    -----------
    - metrics: 包含指标的字典，格式为 {场景: {样本索引: JSON数据}}\n
        {
            "default": {
                "0": JSON数据,
                "1": JSON数据,
                ...
            },
            "farmer": {
                "0": JSON数据,
                "1": JSON数据,
                ...
            },
            ...
        }
    """
    metrics = {}
    scenarios = [d for d in os.listdir(folder_path) 
                 if os.path.isdir(os.path.join(folder_path, d))]
    
    for s in scenarios:
        metrics[s] = {}
        scenario_folder = os.path.join(folder_path, s)
        # 检查场景文件夹是否存在
        if not os.path.exists(scenario_folder):
            print(f"警告: 场景文件夹 {scenario_folder} 不存在")
            continue
        
        # 获取场景文件夹下的所有样本索引文件夹
        try:
            sample_folders = [f for f in os.listdir(scenario_folder) 
                            if os.path.isdir(os.path.join(scenario_folder, f))]
        except Exception as e:
            print(f"错误: 无法读取场景文件夹 {scenario_folder}: {e}")
            continue
            
        for sample_folder in sample_folders:
            # 使用工具函数查找JSON文件，支持两种文件名：performance_score.json和performance.json
            json_files = [
                os.path.join(scenario_folder, sample_folder, 'performance_score.json'),
                os.path.join(scenario_folder, sample_folder, 'performance.json')
            ]
            json_file = find_valid_file(json_files, f"{sample_folder}的性能")
            
            if json_file is None:
                continue
            
            # 读取JSON文件
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                idx = os.path.basename(sample_folder).split("_")[1]
                metrics[s][idx] = json_data
            except Exception as e:
                print(f"错误: 无法读取JSON文件 {json_file}: {e}")
                continue
    
    return metrics


def cal_total_area(metrics: Dict[str, Dict[str, Dict]], 
                    aggregate_by_first_digit: bool = False):
    """
    计算每个场景中，每个LU的总面积大小
    
    Parameters:
    -----------
    - metrics: 包含指标的字典，格式为 {场景: {样本索引: JSON数据}}\n
        {
            "default": {"0": JSON数据, "1": JSON数据,...},
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...},
        }
    - aggregate_by_first_digit: 是否按LU类型的第一个数字进行聚合计算
        - True: 按简化LU类型聚合（1, 2, 3, 4, 5, 6）
        - False: 按详细LU类型聚合（11, 12, 21, 22, ...）
    
    Returns:
    -----------
    - total_area_change: 总面积变化字典，格式为\n 
        {
            "origin": {LU类型1: 总面积1, LU类型2: 总面积2, ...}, 
            "target": {...},
            "default": {...},
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...}
        }
        当aggregate_by_first_digit=True时，LU键为简化类型（1-6）
        当aggregate_by_first_digit=False时，LU键为详细类型（11, 12, 21, ...）
    """
    total_area_change = {}
    
    # 确定要使用的LU标签集合
    lu_labels = ALL_LU_LABELS_SIMPLIFIED if aggregate_by_first_digit else ALL_LU_LABELS
    
    # 初始化origin和target字典
    total_area_change["origin"] = {}
    total_area_change["target"] = {}
    # 初始化所有LU类型的origin和target为0
    for lu in lu_labels:
        total_area_change["origin"][lu] = 0.0
        total_area_change["target"][lu] = 0.0
    
    # 初始化场景字典
    scenarios = metrics.keys()
    for s in scenarios:
        total_area_change[s] = {}
        for lu in lu_labels:
            total_area_change[s][lu] = 0.0
    
    # 遍历每个场景和样本索引
    for s in scenarios:
        sample_idx = metrics[s].keys()
        
        for idx in sample_idx:
            # 提取面积变化指标
            best_ori_metrics = metrics[s][idx]['FINAL']['best_ori']
            best_tgt_metrics = metrics[s][idx]['FINAL']['best_tgt']
            
            # 获取所有出现的LU类型（合并ori和tgt中的LU）
            all_lu_in_sample = set(best_ori_metrics.keys()) | set(best_tgt_metrics.keys())
            
            # 遍历所有出现的LU类型
            for lu_label in all_lu_in_sample:
                lu_int = int(lu_label)
                if lu_int in ALL_LU_LABELS:
                    # 安全地获取ori_plan，如果不存在则设为0
                    ori_plan = best_ori_metrics.get(lu_label, {}).get("ori_plan", 0.0)
                    # 安全地获取best_plan（从ori或tgt中获取）
                    best_plan_ori = best_ori_metrics.get(lu_label, {}).get("best_plan", 0.0)
                    best_plan_tgt = best_tgt_metrics.get(lu_label, {}).get("best_plan", 0.0)
                    # 使用非零的best_plan值（通常两者相同，但以防万一）
                    best_plan = best_plan_ori if best_plan_ori != 0.0 else best_plan_tgt
                    # 安全地获取tgt_plan
                    tgt_plan = best_tgt_metrics.get(lu_label, {}).get("tgt_plan", 0.0)
                    
                    # 确定要使用的LU键（原始LU类型或简化LU类型）
                    lu_key = lu_int // 10 if aggregate_by_first_digit else lu_int
                    
                    # 累加origin和target面积信息（只需要累加一次，因为所有场景的ori和tgt都相同）
                    if s == list(scenarios)[0]:  # 只在第一个场景中累加origin和target
                        total_area_change["origin"][lu_key] += ori_plan
                        total_area_change["target"][lu_key] += tgt_plan
                    
                    # 累加每个场景的best面积信息
                    total_area_change[s][lu_key] += best_plan
                    print(f"已读取场景 {s} 样本索引 {idx} 中 LU {lu_int} 的面积变化指标")
    return total_area_change

def cal_sample_area_change_ratio(metrics: dict, aggregate_by_first_digit: bool = False, 
                    comparison_type: str = "origin"):
    """
    计算每个场景中，每个LU的面积变化比例列表
    
    Parameters:
    -----------
    - metrics: 包含指标的字典，格式为 {场景: {样本索引: JSON数据}}\n
        {
            "default": {"0": JSON数据, "1": JSON数据,...},
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...},
        }
    - aggregate_by_first_digit: 是否按LU类型的第一个数字进行聚合计算
        - True: 按简化LU类型聚合（1, 2, 3, 4, 5, 6）
        - False: 按详细LU类型聚合（11, 12, 21, 22, ...）
    - comparison_type: 比较类型，决定是与origin还是target比较
        - "origin": 计算best相比origin的变化比例
        - "target": 计算best相比target的变化比例
    
    Returns:
    -----------
    - sample_area_change: 每个场景中，每个LU的面积变化比例列表，格式为\n
        {
            "default": {
                LU类型1: [变化比例1，变化比例2，...],
                LU类型2: [变化比例1，变化比例2，...],
                ...
            },
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...}
        }
        当aggregate_by_first_digit=True时，LU键为简化类型（1-6）
        当aggregate_by_first_digit=False时，LU键为详细类型（11, 12, 21, ...）
    """
    sample_area_change = {}
    
    # 确定要使用的LU标签集合
    lu_labels = ALL_LU_LABELS_SIMPLIFIED if aggregate_by_first_digit else ALL_LU_LABELS
    
    # 初始化场景字典
    scenarios = metrics.keys()
    for s in scenarios:
        sample_area_change[s] = {}
        for lu in lu_labels:
            sample_area_change[s][lu] = []

    # 遍历每个场景和样本索引
    for s in scenarios:
        sample_idx = metrics[s].keys()
        
        for idx in sample_idx:
            # 提取面积变化指标
            best_ori_metrics = metrics[s][idx]['FINAL']['best_ori']
            best_tgt_metrics = metrics[s][idx]['FINAL']['best_tgt']
            
            # 根据比较类型确定要使用的指标
            if comparison_type == "origin":
                base_metrics = best_ori_metrics
                base_key = "ori_plan"
            else:  # comparison_type == "target"
                base_metrics = best_tgt_metrics
                base_key = "tgt_plan"
            
            # 收集所有出现的LU类型（合并ori和tgt中的LU）
            all_lu_in_sample = set(best_ori_metrics.keys()) | set(best_tgt_metrics.keys())
            
            # 按LU类型聚合面积数据
            aggregated_areas = {}
            
            for lu_label in all_lu_in_sample:
                lu_int = int(lu_label)
                if lu_int in ALL_LU_LABELS:
                    # 确定要使用的LU键（原始LU类型或简化LU类型）
                    lu_key = lu_int // 10 if aggregate_by_first_digit else lu_int
                    
                    # 获取基础面积（origin或target）
                    base_area = base_metrics.get(lu_label, {}).get(base_key, 0.0)
                    # 获取best面积（从ori或tgt中获取）
                    best_plan_ori = best_ori_metrics.get(lu_label, {}).get("best_plan", 0.0)
                    best_plan_tgt = best_tgt_metrics.get(lu_label, {}).get("best_plan", 0.0)
                    # 使用非零的best_plan值（通常两者相同，但以防万一）
                    best_plan = best_plan_ori if best_plan_ori != 0.0 else best_plan_tgt
                    
                    # 初始化聚合字典
                    if lu_key not in aggregated_areas:
                        aggregated_areas[lu_key] = {"base_total": 0.0, "best_total": 0.0}
                    
                    # 累加面积数据
                    aggregated_areas[lu_key]["base_total"] += base_area
                    aggregated_areas[lu_key]["best_total"] += best_plan
            
            # 计算每个LU类型的变化比例
            for lu in lu_labels:
                if lu in aggregated_areas:
                    base_total = aggregated_areas[lu]["base_total"]
                    best_total = aggregated_areas[lu]["best_total"]
                    change_ratio = (best_total - base_total) / base_total if base_total > 0 else 0.0
                    sample_area_change[s][lu].append(change_ratio)
                else:
                    sample_area_change[s][lu].append(0.0)
                
                print(f"已读取场景 {s} 样本索引 {idx} 中 LU {lu} 的面积变化指标")
    
    return sample_area_change



def cal_total_es(metrics: Dict[str, Dict[str, Dict]]):
    """
    计算每个场景中，每个ES的总价值大小
    
    Parameters:
    -----------
    - metrics: 包含指标的字典，格式为 {场景: {样本索引: JSON数据}}\n
        {
            "default": {"0": JSON数据, "1": JSON数据,...},
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...},
        }
    
    Returns:
    -----------
    - total_es_change: 总价值变化字典，格式为\n 
        {
            "origin": {"food": 总价值1, "water": 总价值2, ...}, 
            "target": {...},
            "default": {...},
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...}
        }
    """
    total_es_change = {}
    
    # ES类型列表
    es_types = ALL_ES_TYPES
    
    # 初始化origin和target字典
    total_es_change["origin"] = {}
    total_es_change["target"] = {}
    # 初始化所有ES类型的origin和target为0
    for es in es_types:
        total_es_change["origin"][es] = 0.0
        total_es_change["target"][es] = 0.0
    
    # 初始化场景字典
    scenarios = metrics.keys()
    for s in scenarios:
        total_es_change[s] = {}
        for es in es_types:
            total_es_change[s][es] = 0.0
    
    # 遍历每个场景和样本索引
    for s in scenarios:
        sample_idx = metrics[s].keys()
        
        for idx in sample_idx:
            # 提取ES指标
            es_metrics = metrics[s][idx]['ES']
            
            # 遍历所有ES类型
            for es in es_types:
                # 安全地获取ori_plan，如果不存在则设为0
                ori_plan = es_metrics.get(es, {}).get("ori_plan", 0.0)
                # 安全地获取best_plan
                best_plan = es_metrics.get(es, {}).get("best_plan", 0.0)
                # 安全地获取tgt_plan
                tgt_plan = es_metrics.get(es, {}).get("tgt_plan", 0.0)
                
                # 累加origin和target价值信息（只需要累加一次，因为所有场景的ori和tgt都相同）
                if s == list(scenarios)[0]:  # 只在第一个场景中累加origin和target
                    total_es_change["origin"][es] += ori_plan
                    total_es_change["target"][es] += tgt_plan
                
                # 累加每个场景的best价值信息
                total_es_change[s][es] += best_plan
                print(f"已读取场景 {s} 样本索引 {idx} 中 ES {es} 的价值指标")
    
    return total_es_change


def cal_sample_es_change_ratio(metrics: dict, 
                                comparison_type: str = "origin"):
    """
    计算每个场景中，每个ES的变化比例列表

    Parameters:
    -----------
    - metrics: 包含指标的字典，格式为 {场景: {样本索引: JSON数据}}\n
        {
            "default": {"0": JSON数据, "1": JSON数据,...},
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...},
        }
    - comparison_type: 比较类型，决定是与origin还是target比较
        - "origin": 计算best相比origin的变化比例（读取best_ori_diff）
        - "target": 计算best相比target的变化比例（读取best_tgt_diff）
    
    Returns:
    -----------
    - sample_es_change: 每个场景中，每个ES的变化比例列表，格式为\n
        {
            "default": {
                "food": [变化比例1，变化比例2，...],
                "carbon": [变化比例1，变化比例2，...],
                "water": [变化比例1，变化比例2，...],
                "culture": [变化比例1，变化比例2，...],
            },
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...}
        }
    """
    sample_es_change = {}
    
    # ES类型列表
    es_types = ALL_ES_TYPES
    
    # 确定要使用的比较字段
    if comparison_type == "origin":
        diff_field = "best_ori_diff"
    else:  # comparison_type == "target"
        diff_field = "best_tgt_diff"
    
    # 初始化场景字典
    scenarios = metrics.keys()
    for s in scenarios:
        sample_es_change[s] = {}
        for es in es_types:
            sample_es_change[s][es] = []

    # 遍历每个场景和样本索引
    for s in scenarios:
        sample_idx = metrics[s].keys()
        
        for idx in sample_idx:
            # 提取ES指标
            es_metrics = metrics[s][idx]['ES']
            
            # 收集每个ES类型的变化比例
            for es in es_types:
                if es in es_metrics:
                    change_ratio = es_metrics[es].get(diff_field, 0.0)
                    sample_es_change[s][es].append(change_ratio)
                else:
                    sample_es_change[s][es].append(0.0)
                
                print(f"已读取场景 {s} 样本索引 {idx} 中 ES {es} 的变化比例")
    
    return sample_es_change








def cal_scenario_diff_ratio(sa_total_area: dict, ga_total_area: dict,
                            lu_order: list = None,
                            scenario_order: list = None,
                            scenario_display_names: dict = None,
                            y_scale: str = "pct",
                            signed_mode: bool = True,
                            gap_threshold: float = 50):
    """
    计算 SA vs GA 各情景下 LU 类型面积变化差异比例。

    计算公式:
      - ratio 模式: Y = (S - O) / |O - T|  (归一化比值)
      - pct 模式:   Y = (S - O) / total_area * 100  (百分点)

    Returns:
        data: {地类名: {显示名: Y值}}
        lu_directions: {地类名: "increase"|"decrease"}
        lu_ref_lines: {地类名: ref_val}
    """
    from libs.const import ALL_LU_LABELS_DICT

    if lu_order is None:
        lu_order = ["1", "2", "3", "4", "5", "6"]
    if scenario_order is None:
        scenario_order = [
            "SA_default", "SA_enterprise", "SA_farmer", "SA_tourist",
            "GA_default", "GA_enterprise", "GA_farmer", "GA_tourist",
        ]
    if scenario_display_names is None:
        scenario_display_names = {
            "SA_default": "SA_基准", "SA_enterprise": "SA_企业",
            "SA_farmer": "SA_农民", "SA_tourist": "SA_游客",
            "GA_default": "GA_基准", "GA_enterprise": "GA_企业",
            "GA_farmer": "GA_农民", "GA_tourist": "GA_游客",
        }

    origin = sa_total_area["origin"]
    target = sa_total_area["target"]
    total_area = sum(origin.values())

    valid_lu = []
    for lu_code in lu_order:
        gap = abs(origin[lu_code] - target[lu_code])
        if gap >= gap_threshold:
            valid_lu.append((lu_code, origin[lu_code], target[lu_code], gap))

    data = {}
    lu_directions = {}
    lu_ref_lines = {}
    for lu_code, ori_val, tgt_val, gap in valid_lu:
        lu_name = ALL_LU_LABELS_DICT.get(int(lu_code), lu_code)
        lu_directions[lu_name] = "increase" if tgt_val > ori_val else "decrease"
        data[lu_name] = {}

        if y_scale == "ratio":
            ref_val = 1.0
        else:
            ref_val = round(abs(tgt_val - ori_val) / total_area * 100, 2)
        lu_ref_lines[lu_name] = ref_val

        for s_key in scenario_order:
            if s_key.startswith("SA_"):
                scenario_val = sa_total_area[s_key[3:]][lu_code]
            else:
                scenario_val = ga_total_area[s_key[3:]][lu_code]

            if y_scale == "ratio":
                if signed_mode:
                    y_val = round((scenario_val - ori_val) / gap, 4)
                else:
                    y_val = round(abs(scenario_val - ori_val) / gap, 4)
            else:
                if signed_mode:
                    y_val = round((scenario_val - ori_val) / total_area * 100, 2)
                else:
                    y_val = round(abs(scenario_val - ori_val) / total_area * 100, 2)

            display_name = scenario_display_names.get(s_key, s_key)
            data[lu_name][display_name] = y_val

        base_val = -ref_val if (signed_mode and lu_directions[lu_name] == "decrease") else ref_val
        data[lu_name]["真实"] = base_val

    return data, lu_directions, lu_ref_lines
def visualize_by_type_bars(total_data_dict: Dict[str, Dict[Union[int, str], float]],
                         total_title: str = "总面积变化",
                         save_folder: str = None,
                         subplot_labels: List[str] = None,
                         **kwargs):
    """
    可视化每个类型（LU或ES）在每个场景的总值
    
    Parameters:
    -----------
    - total_data_dict: 包含场景和总值变化的字典，格式为 {场景: {类型: 总值}}\n
        {
            "origin": {类型1: 总值1, 类型2: 总值2, ...},           
            "target": {...},
            "default": {...},
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...},
        }
        类型可以是LU类型（整数）或ES类型（字符串）
    - total_title: str, 默认"总面积变化", 图表标题
    - save_folder: str, 默认None, 保存路径，若为None则不保存图表
    - subplot_labels: List[str], 默认None, 子图标注列表，与类型数量相同

    - **kwargs: 其他参数，传递给color_bar_figure函数
        - width: float, 默认0.35, 条形宽度
        - y_axis_ratio: float, 默认1.0, 用于调整y轴范围, 如0.9表示最大值只占y轴的90%
        - show_legend: bool, 默认True, 是否显示图例
        - show_labels: bool, 默认True, 是否显示标签
        - label_type: str, 默认"edge", 标签类型，可选"edge"（边框）或"center"（居中）

    Returns:
    -----------
    - type_data_dict: 每个类型的总值变化字典，格式为 {类型: {场景: 总值}}
    """
    ensure_folder(save_folder)
    type_data_dict = {}
    
    # 确定要可视化的类型集合（从数据中动态获取）
    all_types = set()
    for scenario_data in total_data_dict.values():
        all_types.update(scenario_data.keys())
    
    # 检测数据类型：LU类型（整数）还是ES类型（字符串）
    is_es_data = any(isinstance(t, str) for t in all_types)
    
    # 检查是否包含场景类型（default, enterprise, farmer, tourist）
    is_scenario_data = any(isinstance(t, str) and t in ["default", "enterprise", "farmer", "tourist"] for t in all_types)
    
    if is_es_data:
        # ES类型数据：直接使用所有ES类型
        valid_types = sorted([t for t in all_types if t in ALL_ES_TYPES])
        data_type = "ES"
        y_label = "价值"
    elif is_scenario_data:
        # 场景类型数据：使用所有场景类型
        valid_types = sorted([t for t in all_types if t in ["default", "enterprise", "farmer", "tourist"]])
        data_type = "场景"
        y_label = "值"
    else:
        # LU类型数据：过滤掉不在ALL_LU_LABELS或ALL_LU_LABELS_SIMPLIFIED中的LU类型
        valid_types = sorted([t for t in all_types if t in ALL_LU_LABELS or t in ALL_LU_LABELS_SIMPLIFIED])
        data_type = "LU"
        y_label = "面积（公顷）"
    
    # 根据类型数量确定子图布局
    num_types = len(valid_types)
    rows, cols = calculate_subplot_layout(num_types, max_rows=2)
    
    # 创建动态布局的大图
    combined_fig, combined_axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5), layout="constrained")
    if num_types == 1:
        combined_axes = [combined_axes]
    else:
        combined_axes = combined_axes.flatten()
    
    for i, t in enumerate(valid_types):
        type_data_dict[t] = {}
        for s in total_data_dict.keys():
            # 直接使用传入的数据，不再进行简化处理
            type_data_dict[t][s] = total_data_dict[s].get(t, 0.0)
        
        print(f"{data_type}类型 {t} 总值变化:")
        print(type_data_dict[t])
        
        scenarios = list(type_data_dict[t].keys())
        values = list(type_data_dict[t].values())
        
        # 确定颜色和标题
        if is_es_data:
            # ES类型标签和颜色映射 
            t_name = ALL_ES_TYPES_DICT.get(t, f"ES类型{t}")
            t_color = ES_COLOR_MAP.get(t, (0.5, 0.5, 0.5))
        elif is_scenario_data:
            # 场景类型标签和颜色映射
            t_name = t  # 直接使用场景名称作为标签
            t_color = SCENARIO_COLOR_MAP.get(t, (0.5, 0.5, 0.5))
        else:
            # LU类型标签和颜色映射
            if t in SIMPLIFIED_COLOR_MAP:
                t_color = SIMPLIFIED_COLOR_MAP[t]
                t_name = ALL_LU_LABELS_DICT.get(t, f"LU类型{t}")
            else:
                # 对于详细LU类型，使用对应简化类型的颜色
                # 安全地处理整数转换
                try:
                    simplified_t = int(t) // 10
                    t_color = SIMPLIFIED_COLOR_MAP.get(simplified_t, (0.5, 0.5, 0.5))
                    t_name = ALL_LU_LABELS_DICT.get(t, f"LU类型{t}")
                except (ValueError, TypeError):
                    # 如果无法转换为整数，使用默认颜色和标签
                    t_color = (0.5, 0.5, 0.5)
                    t_name = f"类型{t}"
        
        # 创建单个子图
        fig = color_bar_figure(x_labels=scenarios, 
                                y_values=values, 
                                y_label=y_label, 
                                title=f"{t_name} {data_type}值变化", 
                                color_setting=[t_color],
                                **kwargs)
        if save_folder is not None:
            fig.savefig(os.path.join(save_folder, f"{t_name}_{data_type.lower()}.png"))
        
        # 直接在大图的对应子图上绘制
        color_bar_figure(x_labels=scenarios, 
                        y_values=values, 
                        y_label=y_label, 
                        title=f"{t_name} {data_type}值变化", 
                        color_setting=[t_color],
                        ax=combined_axes[i],
                        **kwargs)
        
        # 设置浅灰色spline（边框）和网格线
        ax = combined_axes[i]
        ax = handle_axes(ax)
        
        # 添加子图标注
        if subplot_labels and i < len(subplot_labels):
            ax.annotate(
                subplot_labels[i],
                xy=(0, 1), xycoords='axes fraction',
                xytext=(+0.5, -0.5), textcoords='offset fontsize',
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    
    # 隐藏多余的子图
    for j in range(num_types, len(combined_axes)):
        combined_axes[j].set_visible(False)
    
    # 设置整体标题
    combined_fig.suptitle(f"{total_title} - {data_type}类型对比", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.92,        # 为整体标题留出空间
        hspace=0.4,      # 增加行间距，避免x_labels和标题重叠
        wspace=0.3       # 适当增加列间距
    )
    # 显示大图
    plt.show()
    
    # 保存大图
    if save_folder is not None:
        combined_fig.savefig(os.path.join(save_folder, f"combined_{data_type.lower()}_comparison.png"))
    
    return type_data_dict

def visualize_by_stackbar(total_area_data_dict: Dict[str, Dict[int, float]], 
                          total_title: str = "总面积变化",
                          save_path: str = None,
                          label_format: str = "value",
                          **kwargs):
    """
    可视化每个场景中的各个LU的总面积占比
    
    Parameters:
    -----------
    - total_area_data_dict: 包含场景和总面积变化的字典，格式为 {场景: {LU类型: 总面积}}\n
        {
            "origin": {LU类型1: 总面积1, LU类型2: 总面积2, ...},           
            "target": {...},
            "default": {...},
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...},
        }
        LU类型可以是详细类型（11, 12, ...）或简化类型（1, 2, ...）
    - total_title: str, 默认"总面积变化", 图表标题
    - save_path: str, 默认None, 保存路径，若为None则不保存图表
    - label_format: str, 默认 "value", 标签格式，可选"value"（数值）或"percentage"（百分比）
    
    - **kwargs: 其他参数，传递给color_bar_figure函数
        - width: float, 默认0.35, 条形宽度
        - y_axis_ratio: float, 默认1.0, 用于调整y轴范围, 如0.9表示最大值只占y轴的90%
        - show_legend: bool, 默认True, 是否显示图例
        - show_labels: bool, 默认True, 是否显示标签
        - label_type: str, 默认"edge", 标签类型，可选"edge"（边框）或"center"（居中）
        - figsize: Tuple[int, int], 默认(10, 6), 图像大小

    """
    ensure_folder(os.path.dirname(save_path))
    scenarios = list(total_area_data_dict.keys())
    
    # 确定要可视化的LU类型集合（从数据中动态获取）
    all_lu_types = set()
    for scenario_data in total_area_data_dict.values():
        all_lu_types.update(scenario_data.keys())
    
    # 过滤掉不在ALL_LU_LABELS或ALL_LU_LABELS_SIMPLIFIED中的LU类型
    valid_lu_types = sorted([lu for lu in all_lu_types if lu in ALL_LU_LABELS or lu in ALL_LU_LABELS_SIMPLIFIED])
    
    # 获取LU类型的中文标签和颜色
    cn_lu_labels = []
    color_map = {}
    
    for lu in valid_lu_types:
        # 确定中文标签
        if lu in ALL_LU_LABELS_DICT:
            cn_label = ALL_LU_LABELS_DICT[lu]
        else:
            cn_label = f"LU类型{lu}"
        cn_lu_labels.append(cn_label)
        
        # 确定颜色
        if lu in SIMPLIFIED_COLOR_MAP:
            color_map[cn_label] = SIMPLIFIED_COLOR_MAP[lu]
        else:
            # 对于详细LU类型，使用对应简化类型的颜色
            simplified_lu = lu // 10
            color_map[cn_label] = SIMPLIFIED_COLOR_MAP.get(simplified_lu, (0.5, 0.5, 0.5))
    
    # 重新组织数据格式：{LU类型标签: [各场景的数值]}
    stacked_data = {}
    for cn_label in cn_lu_labels:
        stacked_data[cn_label] = []
    
    # 为每个场景填充对应LU类型的数值
    for scenario in scenarios:
        scenario_data = total_area_data_dict[scenario]
        
        # 计算当前场景的总面积（用于百分比计算）
        total_area = sum(scenario_data.values())
        
        # 填充每个LU类型的数值
        for i, lu in enumerate(valid_lu_types):
            lu_value = scenario_data.get(lu, 0.0)
            
            if label_format == "percentage" and total_area > 0:
                # 计算百分比
                percentage_value = lu_value / total_area
                stacked_data[cn_lu_labels[i]].append(percentage_value)
            else:
                # 使用实际值
                stacked_data[cn_lu_labels[i]].append(lu_value)
    
    # 调用堆叠条形图函数，使用自定义颜色
    fig = stacked_bar_figure(categories=scenarios, 
                        data_dict=stacked_data, 
                        y_label="面积（公顷）" if label_format == "value" else "占比",
                        title=total_title, 
                        width=0.6, 
                        label_format=label_format,
                        colors=color_map, 
                        **kwargs)
    if save_path is not None:
        fig.savefig(save_path)

def visualize_by_hist(sample_area_ratio_data_dict: Dict[str, Dict[int, List[float]]],
                      save_folder: str = None,
                      density: bool = False,
                      comparison_type: str = "origin",
                      subplot_labels: List[str] = None,
                      **kwargs):
    """
    可视化每个场景中的各个LU的面积变化比例（使用直方图）
    
    Parameters:
    -----------
    - sample_area_ratio_data_dict: 包含场景和面积变化比例的字典，格式为 {场景: {LU类型: [变化比例1，变化比例2，...]}}\n
        {
            "default": {
                LU类型1: [变化比例1，变化比例2，...],
                LU类型2: [变化比例1，变化比例2，...], ...
            },
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...}
        }
        LU类型可以是详细类型（11, 12, ...）或简化类型（1, 2, ...）
    - save_folder: 保存图表的文件夹路径
    - density: 是否显示密度而不是频数，默认False
    - comparison_type: "origin"或"target"，用于标题和文件名
    - subplot_labels: List[str], 默认None, 子图标注列表，与类型数量相同
    - **kwargs: 其他参数，传递给hist_figure函数
        - bins: int, 默认10, 直方图的箱子数量
        - edgecolor: str, 默认"black", 边框颜色
        - alpha: float, 默认0.7, 透明度，默认值为0.7
        - cumulative: bool, 默认False, 是否显示累计直方图，默认值为False
        - histtype: str, 默认"bar", 直方图类型，可选"bar", "step", "stepfilled"
        - orientation: str, 默认"vertical", 直方图方向，默认可选"vertical"（垂直）或"horizontal"（水平）
        - rwidth: float, 默认0.8, 直方图宽度，默认值为0.8
        - log: bool, 默认False, 是否使用对数缩放，默认值为False
        - remove_extremes: bool, 默认False, 是否移除异常值，默认值为False
        - method: str, 默认"percentile", 移除异常值的方法，默认可选"percentile"（百分位数）或"std"（标准差）
        - lower_bound: float, 默认0.05, 百分位方法下限，默认值为0.05
        - upper_bound: float, 默认0.95, 百分位方法上限，默认值为0.95
        - std_threshold: float, 默认3, 标准差阈值，默认值为3
    """
    ensure_folder(save_folder)
    
    # 获取所有场景（default、enterprise、farmer、tourist）
    scenarios = list(sample_area_ratio_data_dict.keys())
    
    # 为每个场景创建直方图可视化
    for s in scenarios:
        print(f"正在处理场景: {s}")
        
        # 确定要可视化的类型集合（从数据中动态获取）
        all_types = set(sample_area_ratio_data_dict[s].keys())
        
        # 检测数据类型：LU类型（整数）还是ES类型（字符串）
        is_es_data = any(isinstance(t, str) and t in ALL_ES_TYPES for t in all_types)
        
        # 检查是否包含场景类型（default, enterprise, farmer, tourist）
        is_scenario_data = any(isinstance(t, str) and t in ["default", "enterprise", "farmer", "tourist"] for t in all_types)
        
        if is_es_data:
            # ES类型数据：直接使用所有ES类型
            valid_types = sorted([t for t in all_types if t in ALL_ES_TYPES])
        elif is_scenario_data:
            # 场景类型数据：使用所有场景类型
            valid_types = sorted([t for t in all_types if t in ["default", "enterprise", "farmer", "tourist"]])
        else:
            # LU类型数据：过滤掉不在ALL_LU_LABELS或ALL_LU_LABELS_SIMPLIFIED中的LU类型
            # 安全地处理LU类型转换，跳过无法转换为整数的值
            valid_types = []
            for t in all_types:
                try:
                    t_int = int(t)
                    if t_int in ALL_LU_LABELS or t_int in ALL_LU_LABELS_SIMPLIFIED:
                        valid_types.append(t)
                except (ValueError, TypeError):
                    # 跳过无法转换为整数的值（可能是场景名称或其他字符串）
                    continue
            valid_types = sorted(valid_types)
        
        # 根据类型数量确定子图布局
        num_types = len(valid_types)
        rows, cols = calculate_subplot_layout(num_types, max_rows=2)
        
        # 创建动态布局的大图
        combined_fig, combined_axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5), layout="constrained")
        if num_types == 1:
            combined_axes = [combined_axes]
        else:
            combined_axes = combined_axes.flatten()
        
        # 创建单个小图
        for i, t in enumerate(valid_types):
            type_data = sample_area_ratio_data_dict[s].get(t, [])
            
            # 确定中文标签和颜色
            if is_es_data:
                # ES类型标签和颜色映射 
                t_name = ALL_ES_TYPES_DICT.get(t, f"ES类型{t}")
                t_color = ES_COLOR_MAP.get(t, (0.5, 0.5, 0.5))
            elif is_scenario_data:
                # 场景类型标签和颜色映射
                t_name = t  # 直接使用场景名称作为标签
                t_color = SCENARIO_COLOR_MAP.get(t, (0.5, 0.5, 0.5))
            else:
                # LU类型标签和颜色映射
                if t in ALL_LU_LABELS_DICT:
                    t_name = ALL_LU_LABELS_DICT[t]
                else:
                    t_name = f"LU类型{t}"
                
                if t in SIMPLIFIED_COLOR_MAP:
                    t_color = SIMPLIFIED_COLOR_MAP[t]
                else:
                    # 对于详细LU类型，使用对应简化类型的颜色
                    # 安全地处理整数转换
                    try:
                        simplified_t = int(t) // 10
                        t_color = SIMPLIFIED_COLOR_MAP.get(simplified_t, (0.5, 0.5, 0.5))
                    except (ValueError, TypeError):
                        # 如果无法转换为整数，使用默认颜色
                        t_color = (0.5, 0.5, 0.5)
            
            if not type_data:
                print(f"  警告: 子场景 {s} 中类型 {t} 没有数据")
                # 创建空的子图
                ax = combined_axes[i]
                ax.text(0.5, 0.5, f"{t_name}\n无数据", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(t_name)
                ax = handle_axes(ax)
                continue
            
            # 创建单个子图
            fig = hist_figure(np.array(type_data),
                             title=f"{t_name} {'ES' if is_es_data else '面积'}变化分布",
                             x_label="变化比例",
                             y_label="频数" if not density else "密度",
                             color=t_color,
                             density=density,
                             **kwargs,
                             figsize=(6, 4))
            
            # 保存单个小图
            if save_folder is not None:
                filename_prefix = f"{comparison_type}_{s}" if comparison_type else s
                fig.savefig(os.path.join(save_folder, f"{filename_prefix}_{t_name}_hist.png"))
            
            # 在大图的对应子图上绘制直方图
            hist_figure(np.array(type_data),
                       title=t_name,
                       x_label="变化比例",
                       y_label="频数" if not density else "密度",
                       color=t_color,
                       density=density,
                       **kwargs,
                       ax=combined_axes[i])
            
            # 设置子图样式
            ax = combined_axes[i]
            ax = handle_axes(ax)
            
            # 添加子图标注
            if subplot_labels and i < len(subplot_labels):
                ax.annotate(
                    subplot_labels[i],
                    xy=(0, 1), xycoords='axes fraction',
                    xytext=(+0.5, -0.5), textcoords='offset fontsize',
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
        
        # 隐藏多余的子图
        for j in range(num_types, len(combined_axes)):
            combined_axes[j].set_visible(False)
        
        # 设置整体标题
        title_prefix = f"{comparison_type} - " if comparison_type else ""
        data_type = "ES" if is_es_data else "LU"
        combined_fig.suptitle(f"{title_prefix}{s} - {data_type}类型{'ES' if is_es_data else '面积'}变化分布对比图", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.92,        # 为整体标题留出空间
            hspace=0.4,      # 增加行间距
            wspace=0.3       # 适当增加列间距
        )
        
        # 显示大图
        plt.show()
        
        # 保存大图
        if save_folder is not None:
            filename_prefix = f"{comparison_type}_{s}" if comparison_type else s
            combined_fig.savefig(os.path.join(save_folder, f"{filename_prefix}_combined_hist_comparison.png"))
        
        # 关闭当前图形以释放内存
        plt.close('all')
    
    print(f"直方图可视化完成！共生成{num_types+1}张图：{num_types}张单个类型图 + 1张整合大图")

def visualize_by_box(sample_area_ratio_data_dict: Dict[str, Dict[int, List[float]]],
                      save_folder: str = None,
                      comparison_type: str = "origin",
                      subplot_labels: List[str] = None,
                      **kwargs):
    """
    可视化每个场景中的各个LU的面积变化比例（使用箱线图）
    
    Parameters:
    -----------
    - sample_area_ratio_data_dict: 包含场景和面积变化比例的字典，格式为 {场景: {LU类型: [变化比例1，变化比例2，...]}}
        {
            "default": {
                LU类型1: [变化比例1，变化比例2，...],
                LU类型2: [变化比例1，变化比例2，...], ...
            },
            "farmer": {...},
            "enterprise": {...},
            "tourist": {...}
        }
        LU类型可以是详细类型（11, 12, ...）或简化类型（1, 2, ...）
    - save_folder: 保存图表的文件夹路径
    - comparison_type: "origin"或"target"，用于标题和文件名
    - subplot_labels: List[str], 默认None, 子图标注列表，与类型数量相同
    - **kwargs: 其他参数，传递给box_plot_figure函数
        - show_labels: bool, 默认True, 是否显示标签
        - show_means: bool, 默认True, 是否显示均值线
        - show_fliers: bool, 默认True, 是否显示异常值
        - notch: bool, 默认False, 是否显示缺口
        - vert: bool, 默认True, 是否垂直显示
        - patch_artist: bool, 默认True, 是否填充箱体颜色
        - widths: float, 默认0.5, 箱体宽度
        - whis: float, 默认1.5, 须线范围
    """
    ensure_folder(save_folder)
    
    # 获取所有场景（default、enterprise、farmer、tourist）
    scenarios = list(sample_area_ratio_data_dict.keys())
    
    # 确定要可视化的类型集合（从数据中动态获取，但限制为6个主要类型）
    all_types = set()
    for s in scenarios:
        all_types.update(sample_area_ratio_data_dict[s].keys())
    
    # 检测数据类型：LU类型（整数）还是ES类型（字符串）
    # 首先检查是否包含标准的ES类型（food, carbon, water, culture）
    is_es_data = any(isinstance(lu, str) and lu in ALL_ES_TYPES for lu in all_types)
    
    # 检查是否包含场景类型（default, enterprise, farmer, tourist）
    is_scenario_data = any(isinstance(lu, str) and lu in ["default", "enterprise", "farmer", "tourist"] for lu in all_types)
    
    if is_es_data and "food" in all_types:
        # ES类型数据：直接使用所有ES类型
        valid_types = sorted([t for t in all_types if t in ALL_ES_TYPES])
    elif is_scenario_data:
        # 场景类型数据：使用所有场景类型
        valid_types = sorted([t for t in all_types if t in ["default", "enterprise", "farmer", "tourist"]])
    else:
        # LU类型数据：过滤掉不在ALL_LU_LABELS或ALL_LU_LABELS_SIMPLIFIED中的LU类型
        # 安全地处理LU类型转换，跳过无法转换为整数的值
        valid_types = []
        for lu in all_types:
            try:
                lu_int = int(lu)
                if lu_int in ALL_LU_LABELS or lu_int in ALL_LU_LABELS_SIMPLIFIED:
                    valid_types.append(lu)
            except (ValueError, TypeError):
                # 跳过无法转换为整数的值（可能是场景名称或其他字符串）
                continue
        valid_types = sorted(valid_types)
    
    # 限制为6个主要类型（如果超过6个，取前6个）
    if len(valid_types) > 6:
        valid_types = valid_types[:6]
        print(f"注意: 只显示前6个类型: {valid_types}")
    
    # 根据类型数量确定子图布局
    num_types = len(valid_types)
    rows, cols = calculate_subplot_layout(num_types, max_rows=2)
    
    # 创建动态布局的大图
    combined_fig, combined_axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5), layout="constrained")
    if num_types == 1:
        combined_axes = [combined_axes]
    else:
        combined_axes = combined_axes.flatten()
    
    # 为每个类型创建箱线图可视化（6张小图 + 大图）
    for i, t in enumerate(valid_types):
        print(f"正在处理类型: {t}")
        
        # 确定中文标签和颜色
        if is_es_data:
            # ES类型标签和颜色映射 
            t_name = ALL_ES_TYPES_DICT.get(t, f"ES类型{t}")
            t_color = ES_COLOR_MAP.get(t, (0.5, 0.5, 0.5))
        elif is_scenario_data:
            # 场景类型标签和颜色映射
            t_name = t  # 直接使用场景名称作为标签
            t_color = SCENARIO_COLOR_MAP.get(t, (0.5, 0.5, 0.5))
        else:
            # LU类型标签和颜色映射
            if t in ALL_LU_LABELS_DICT:
                t_name = ALL_LU_LABELS_DICT[t]
            else:
                t_name = f"LU类型{t}"
            
            if t in SIMPLIFIED_COLOR_MAP:
                t_color = SIMPLIFIED_COLOR_MAP[t]
            else:
                # 对于详细LU类型，使用对应简化类型的颜色
                # 安全地处理整数转换
                try:
                    simplified_t = int(t) // 10
                    t_color = SIMPLIFIED_COLOR_MAP.get(simplified_t, (0.5, 0.5, 0.5))
                except (ValueError, TypeError):
                    # 如果无法转换为整数，使用默认颜色
                    t_color = (0.5, 0.5, 0.5)
        
        # 准备该类型在所有场景下的数据
        type_data_sets = []
        scenario_labels = []
        scenario_colors = []
        
        for s in scenarios:
            type_data = sample_area_ratio_data_dict[s].get(t, [])
            if type_data:  # 只添加有数据的场景
                type_data_sets.append(type_data)
                scenario_labels.append(s)
                # 为每个场景分配不同的颜色
                scenario_colors.append(t_color)
        
        if not type_data_sets:
            print(f"  警告: 类型 {t} 在所有场景中都没有数据")
            # 创建空的子图
            ax = combined_axes[i]
            ax.text(0.5, 0.5, f"{t_name}\n无数据", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(t_name)
            ax = handle_axes(ax)
            continue
        
        # 创建单个类型的箱线图（展示该类型在4种场景下的分布）- 小图
        fig = box_plot_figure(
            data_sets=type_data_sets,
            labels=scenario_labels,
            title=f"{t_name} - 不同场景下{'ES' if is_es_data else '面积'}变化分布",
            y_label="变化比例",
            colors=scenario_colors,
            figsize=(10, 6),
            **kwargs
        )
        
        # 保存单个类型的箱线图
        if save_folder is not None:
            filename_prefix = f"{comparison_type}" if comparison_type else ""
            fig.savefig(os.path.join(save_folder, f"{filename_prefix}_{t_name}_boxplot.png"))
        
        # 在大图的对应子图上绘制箱线图
        box_plot_figure(
            data_sets=type_data_sets,
            labels=scenario_labels,
            title=t_name,
            y_label="变化比例",
            colors=scenario_colors,
            figsize=(6, 4),
            ax=combined_axes[i],
            **kwargs
        )
        
        # 设置子图样式
        ax = combined_axes[i]
        ax = handle_axes(ax)
        
        # 添加子图标注
        if subplot_labels and i < len(subplot_labels):
            ax.annotate(
                subplot_labels[i],
                xy=(0, 1), xycoords='axes fraction',
                xytext=(+0.5, -0.5), textcoords='offset fontsize',
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    
    # 隐藏多余的子图
    for j in range(num_types, len(combined_axes)):
        combined_axes[j].set_visible(False)
    
    # 设置整体标题
    title_prefix = f"{comparison_type} - " if comparison_type else ""
    data_type = "ES" if is_es_data else "LU"
    combined_fig.suptitle(f"{title_prefix}所有{data_type}类型在不同场景下的{'ES' if is_es_data else '面积'}变化分布对比", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.92,        # 为整体标题留出空间
        hspace=0.4,      # 增加行间距，避免x_labels和标题重叠
        wspace=0.3       # 适当增加列间距
    )
    
    # 显示大图
    plt.show()
    
    # 保存大图
    if save_folder is not None:
        filename_prefix = f"{comparison_type}" if comparison_type else ""
        combined_fig.savefig(os.path.join(save_folder, f"{filename_prefix}_all_lu_types_boxplot_comparison.png"))
    
    # 关闭当前图形以释放内存
    plt.close('all')
    
    print(f"箱线图可视化完成！共生成{num_types+1}张图：{num_types}张单个类型图 + 1张整合大图")

def handle_axes(ax: plt.Axes):
    """
    处理子图轴样式，设置浅灰色边框和网格线
    """
    ax.spines['top'].set_color('lightgray')
    ax.spines['right'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    ax.spines['left'].set_color('lightgray')
    ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), fontsize=12)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    return ax


def convert_json_keys_to_int(data: Dict) -> Dict:
    """
    将JSON数据中的字符串键转换为整数键（递归实现）
    
    Parameters:
    -----------
    - data: 从JSON文件加载的数据字典，可能包含字符串键
    
    Returns:
    -----------
    - converted_data: 转换后的数据字典，其中数字键被转换为整数
    
    Example:
    -----------
    >>> json_data = {"origin": {"1": 100.0, "2": 200.0}, "target": {"3": 300.0}}
    >>> convert_json_keys_to_int(json_data)
    {"origin": {1: 100.0, 2: 200.0}, "target": {3: 300.0}}
    
    >>> json_data = {"level1": {"level2": {"1": 100, "2": 200}}}
    >>> convert_json_keys_to_int(json_data)
    {"level1": {"level2": {1: 100, 2: 200}}}
    """
    if not isinstance(data, dict):
        return data
    
    converted_data = {}
    
    for key, value in data.items():
        # 递归处理嵌套字典
        if isinstance(value, dict):
            converted_data[key] = convert_json_keys_to_int(value)
        else:
            converted_data[key] = value
    
    # 在当前层级转换字符串键为整数键
    converted_keys_data = {}
    for key, value in converted_data.items():
        try:
            # 尝试将当前层级的键转换为整数
            converted_keys_data[int(key)] = value
        except (ValueError, TypeError):
            # 如果转换失败，保持原键
            converted_keys_data[key] = value
    
    return converted_keys_data


def load_json_with_int_keys(json_path: str) -> Dict:
    """
    从JSON文件加载数据并自动将字符串键转换为整数键

    Parameters:
    -----------
    - json_path: JSON文件路径

    Returns:
    -----------
    - data: 转换后的数据字典，其中数字键被转换为整数

    Raises:
    -----------
    - FileNotFoundError: 如果JSON文件不存在
    - JSONDecodeError: 如果JSON文件格式错误
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON文件 {json_path} 不存在")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return convert_json_keys_to_int(data)


def load_or_compute_metric(data_folder: str,
                           cache_name: str,
                           calc_fn,
                           json_path: str = None,
                           **calc_kwargs):
    cache_path = json_path or os.path.join(data_folder, cache_name)
    if os.path.exists(cache_path):
        return load_json_with_int_keys(cache_path)

    if data_folder is None:
        raise ValueError("data_folder must be provided if cache json does not exist")

    print(f"JSON文件路径 {cache_path} 不存在, 通过get_performance_metrics计算")
    metrics = get_performance_metrics(data_folder)
    data = calc_fn(metrics, **calc_kwargs)
    ensure_folder(os.path.dirname(cache_path))
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data


def list_sample_indices(data_folder: str) -> List[str]:
    if not os.path.exists(data_folder):
        raise ValueError(f"数据文件夹不存在: {data_folder}")

    indices = set()
    for strategy_dir in os.listdir(data_folder):
        strategy_path = os.path.join(data_folder, strategy_dir)
        if not os.path.isdir(strategy_path):
            continue
        for item in os.listdir(strategy_path):
            item_path = os.path.join(strategy_path, item)
            if os.path.isdir(item_path) and '_' in item and item.replace('_', '').isdigit():
                indices.add(item.split("_")[1])
    return sorted(indices, key=lambda x: int(x) if str(x).isdigit() else str(x))


def build_sa_ga_total_area_data(sa_folder: str, ga_folder: str):
    sa_data = load_or_compute_metric(
        sa_folder, "total_area.json", cal_total_area,
        aggregate_by_first_digit=True)
    ga_data = load_or_compute_metric(
        ga_folder, "total_area.json", cal_total_area,
        aggregate_by_first_digit=True)
    return sa_data, ga_data


def build_sa_ga_combined_area_data(sa_folder: str, ga_folder: str) -> Dict:
    sa_data, ga_data = build_sa_ga_total_area_data(sa_folder, ga_folder)
    combined = {"origin": sa_data["origin"], "target": sa_data["target"]}
    for scenario in ALL_SCENARIOS:
        combined[f"SA_{scenario}"] = sa_data[scenario]
        combined[f"GA_{scenario}"] = ga_data[scenario]
    return combined


def get_best_samples(metrics, min_improvement_threshold=0.0, min_positive_scenarios=4, min_other_threshold=None):
    """
    找到一批索引，它们在各个场景中对ES都有一定提升，并且满足特定场景的排名要求
    
    Parameters:
    -----------
    - metrics: 格式为 {场景: {索引: {"ES": {"best_tgt_diff": 提升比例}}}
        其中 best_tgt_diff 是该索引在该场景下的提升比例
    - min_improvement_threshold: 最小提升阈值，默认0.0表示所有ES类型都需要有提升
    - min_positive_scenarios: 需要有多少个场景显示提升，默认4表示所有场景都需要有提升
    - min_other_threshold: 其他场景的最小阈值，如果为None，则所有场景都使用min_improvement_threshold
        如果设置此参数，则min_positive_scenarios个场景需要大于0，其余场景只需要大于min_other_threshold
    
    Returns:
    --------
    - selected_indices: 满足条件的索引列表
    - performance_data: 每个索引的性能数据字典
    """
    
    # 获取所有样本索引（假设所有场景都有相同的索引）
    sample_indices = set()
    for scenario in metrics.values():
        sample_indices.update(scenario.keys())
    sample_indices = sorted(sample_indices)
    
    # ES类型列表
    es_types = ALL_ES_TYPES
    scenarios = ["default", "farmer", "enterprise", "tourist"]
    
    selected_indices = []
    performance_data = {}
    
    for idx in sample_indices:
        # 检查该索引是否在所有场景中都存在
        if not all(scenario in metrics and idx in metrics[scenario] for scenario in scenarios):
            continue
            
        # 收集该索引在所有场景中的ES提升数据
        idx_improvements = {}
        scenario_scores = {}
        
        # 先计算每个场景的得分
        for scenario in scenarios:
            es_data = metrics[scenario][idx].get("ES", {})
            scenario_improvements = []
            
            # 检查每个ES类型是否有提升
            for es_type in es_types:
                if es_type in es_data:
                    best_tgt_diff = es_data[es_type].get("best_tgt_diff", 0.0)
                    scenario_improvements.append(best_tgt_diff)
                else:
                    scenario_improvements.append(0.0)
            
            scenario_scores[scenario] = scenario_improvements
            
            # 检查该场景是否满足提升条件
            if min_other_threshold is not None:
                # 新模式：检查场景是否满足严格条件（大于0）
                if all(imp > 0.0 for imp in scenario_improvements):
                    idx_improvements[scenario] = scenario_improvements
            else:
                # 原模式：所有场景都使用min_improvement_threshold
                if all(imp > min_improvement_threshold for imp in scenario_improvements):
                    idx_improvements[scenario] = scenario_improvements
        
        # 计算有效场景数
        valid_scenario_count = len(idx_improvements)
        
        # 检查是否满足最小有效场景数
        if valid_scenario_count < min_positive_scenarios:
            continue
        
        # 如果使用新模式，还需要检查剩余场景是否满足min_other_threshold
        if min_other_threshold is not None:
            remaining_scenarios = [s for s in scenarios if s not in idx_improvements]
            all_remaining_valid = True
            for scenario in remaining_scenarios:
                scenario_improvements = scenario_scores[scenario]
                # 剩余场景只需要大于min_other_threshold
                if not all(imp > min_other_threshold for imp in scenario_improvements):
                    all_remaining_valid = False
                    break
            
            # 如果剩余场景中有不满足条件的，则整个索引不满足条件
            if not all_remaining_valid:
                continue
        
        # 计算该索引在不同场景中特定ES类型的排名
        idx_rankings = {}
        
        # 对每个ES类型，计算该索引在不同场景中的排名
        for es_idx, es_type in enumerate(es_types):
            # 获取该ES类型在所有场景中的提升值
            es_values = []
            for scenario in scenarios:
                if scenario in scenario_scores:
                    es_value = scenario_scores[scenario][es_idx]
                    es_values.append((scenario, es_value))
            
            # 按提升值排序并计算排名
            es_values.sort(key=lambda x: x[1], reverse=True)
            for rank, (scenario, value) in enumerate(es_values):
                if es_type not in idx_rankings:
                    idx_rankings[es_type] = {}
                idx_rankings[es_type][scenario] = rank + 1  # 排名从1开始
        
        # 检查排名条件
        ranking_conditions_met = True
        
        # farmer场景的food需要排在前2
        if "food" in idx_rankings and "farmer" in idx_rankings["food"]:
            food_rank = idx_rankings["food"]["farmer"]
            if food_rank > 2:
                ranking_conditions_met = False
        
        # enterprise场景的carbon和water需要排在前2
        if ranking_conditions_met:
            carbon_rank = idx_rankings["carbon"].get("enterprise", float('inf'))
            water_rank = idx_rankings["water"].get("enterprise", float('inf'))
            if carbon_rank > 2 and water_rank > 2:
                ranking_conditions_met = False
        
        # tourist场景的culture需要排在前2
        if ranking_conditions_met:
            culture_rank = idx_rankings["culture"].get("tourist", float('inf'))
            if culture_rank > 2:
                ranking_conditions_met = False
        
        if not ranking_conditions_met:
            continue
        
        # 直接选择满足排名条件的索引
        selected_indices.append(idx)
        performance_data[idx] = {
            "improvements": idx_improvements,
            "scenario_count": valid_scenario_count,
            "rankings": idx_rankings
        }
    
    # 按有效场景数排序，场景数多的排在前面
    selected_indices.sort(key=lambda idx: performance_data[idx]["scenario_count"], reverse=True)
    
    return selected_indices, performance_data

def _load_tiff_file(file_path, tensor_name, tensor_dict, required=True):
    """
    辅助函数：加载TIF文件并添加到tensor_dict中
    
    Parameters:
    -----------
    file_path: str, TIF文件路径
    tensor_name: str, 在tensor_dict中的键名
    tensor_dict: dict, 存储张量的字典
    required: bool, 是否为必需文件，如果为True且文件不存在则抛出异常
    
    Returns:
    --------
    bool: 是否成功加载
    """
    import torch
    
    if os.path.exists(file_path):
        try:
            tif_data = load_tiff(file_path)
            tensor_dict[tensor_name] = torch.tensor(tif_data)
            return True
        except Exception as e:
            print(f"错误: 加载{tensor_name}文件失败 {file_path}: {e}")
            return False
    else:
        if required:
            print(f"警告: 找不到{tensor_name}文件 {file_path}")
        else:
            print(f"警告: 找不到{tensor_name}文件 {file_path}，跳过")
        return not required  # 如果是必需文件且不存在，返回False；如果是可选文件，返回True


def visualize_best_samples(sample_idx: str or list, 
                            data_folder: str, 
                            save_folder: str, 
                            comparison_type: str = "target", 
                            **kwargs):
    """
    可视化最佳样本的性能数据
    
    根据索引值，从data_folder里的四个场景中找到对应的文件夹，索引文件夹里的plans_{idx}里有ori_plan best_plan tgt_plan，
    分别是2010真实数据、当前优化场景下的优化结果、2015真实数据。将有6个栅格以nparray的形式被保存，
    顺序是origin default enterprise farmer tourist target。
    
    Parameters:
    -----------
    - sample_idx: str or list, 样本索引，可以是单个索引或索引列表
    - data_folder: str, 数据文件夹路径，默认为None
    - save_folder: str, 保存可视化结果的文件夹路径
    - comparison_type: str, 比较类型，"target"或"origin"，默认"target"
    - **kwargs: 其他参数
        - color_map: dict, 自定义颜色映射
        - figsize: tuple, 图像大小
        - show: bool, 是否显示图像
        - layout: tuple, 子图布局，默认None
    """

    # 获取参数
    color_map = kwargs.get('color_map', COLOR_MAP)
    figsize = kwargs.get('figsize', None)
    layout = kwargs.get('layout', None)
    show = kwargs.get('show', False)
    
    # 确保sample_idx是列表
    if isinstance(sample_idx, str):
        sample_idx = [sample_idx]
    
    # 场景列表
    scenarios = ["default", "enterprise", "farmer", "tourist"]
    
    for idx in sample_idx:
        print(f"\n正在处理样本索引 {idx}...")
        
        # 收集6个矩阵数据
        tensor_dict = {}
        performance_data = {}
        
        # 使用辅助函数加载origin和target矩阵
        base_scenario = scenarios[0]
        base_folder = os.path.join(data_folder, base_scenario, f"10_{idx}_5", f"plans_{idx}")
        
        # 1. 加载origin矩阵 (必需文件)
        origin_path = os.path.join(base_folder, "ori_plan.TIF")
        if not _load_tiff_file(origin_path, "origin", tensor_dict, required=True):
            continue
        
        # 2. 加载target矩阵 (必需文件)
        target_path = os.path.join(base_folder, "tgt_plan.TIF")
        if not _load_tiff_file(target_path, "target", tensor_dict, required=True):
            continue
        
        # 3. 加载四个场景的best_plan矩阵 (可选文件)
        for scenario in scenarios:
            best_plan_path = os.path.join(data_folder, scenario, f"10_{idx}_5", f"plans_{idx}", "best_plan.TIF")
            _load_tiff_file(best_plan_path, scenario, tensor_dict, required=False)
        
        # 4. 收集性能数据
        for scenario in scenarios:
            # 使用工具函数查找performance文件
            perf_paths = [
                os.path.join(data_folder, scenario, f"10_{idx}_5", "performance_score.json"),
                os.path.join(data_folder, scenario, f"10_{idx}_5", "performance.json")
            ]
            perf_path = find_valid_file(perf_paths, f"{scenario}场景的性能")
            
            if perf_path is None:
                continue
            
            try:
                with open(perf_path, 'r', encoding='utf-8') as f:
                    perf_data = json.load(f)
                
                # 提取ES性能指标
                es_data = perf_data.get("ES", {})
                scenario_performance = {}
                
                for es_type in ["food", "carbon", "water", "culture"]:
                    if es_type in es_data:
                        best_tgt_diff = es_data[es_type].get("best_tgt_diff", 0.0)
                        scenario_performance[es_type] = best_tgt_diff
                
                performance_data[scenario] = scenario_performance
                
            except Exception as e:
                print(f"错误: 读取{scenario}场景性能文件失败: {e}")
        
        # 5. 创建可视化标题，包含性能数据
        main_title = f"样本索引 {idx} - 土地利用方案对比\n"
        
        # 添加性能数据到标题
        if performance_data:
            main_title += "各场景ES提升比例 (best_tgt_diff):\n"
            for scenario in scenarios:
                if scenario in performance_data:
                    perf = performance_data[scenario]
                    main_title += f"{scenario}: food={perf.get('food', 0):.3f}, carbon={perf.get('carbon', 0):.3f}, water={perf.get('water', 0):.3f}, culture={perf.get('culture', 0):.3f}\n"
        print(main_title)

        # 6. 可视化结果
        ensure_folder(save_folder)
        save_path = os.path.join(save_folder, f"sample_{idx}_visualization.png")
        
        try:
            # 重新排列张量顺序为: origin, default, enterprise, farmer, tourist, target
            ordered_tensors = {}
            order = ["origin", "default", "enterprise", "farmer", "tourist", "target"]
            
            for key in order:
                if key in tensor_dict:
                    ordered_tensors[key] = tensor_dict[key]
            
            saved_paths = visualize_tensor_list(
                ordered_tensors,
                main_title=main_title,
                save_path=save_path,
                color_map=color_map,
                figsize=figsize,
                show=show,
                layout=layout,
                show_colorbar=False,
            )
            
            print(f"样本 {idx} 可视化完成，保存至: {save_path}")
            
        except Exception as e:
            print(f"错误: 可视化样本 {idx} 失败: {e}")
    
    print("\n所有样本可视化完成")

def get_subplot_default_labels(num_types: int, 
                            style: str = "(a)") -> List[str]:
    """
    生成默认的子图标注列表
    
    Parameters:
    -----------
    - num_types: int, 子图数量
    - style: str, 默认"(a)", 标注样式，可选:
        - "(a)": (a), (b), (c), ...
        - "(1)": (1), (2), (3), ...
        - "(A)": (A), (B), (C), ...
        - "a)": a), b), c), ...
        - "1)": 1), 2), 3), ...
        - "A)": A), B), C), ...
        - "a": a, b, c, ...
        - "1": 1, 2, 3, ...
        - "A": A, B, C, ...
    
    Returns:
    -----------
    - List[str]: 子图标注列表
    """
    if num_types <= 0:
        return []
    
    # 定义样式映射
    style_mappings = {
        "(a)": {"prefix": "(", "suffix": ")", "start": 97, "type": "lower_letter"},
        "(1)": {"prefix": "(", "suffix": ")", "start": 49, "type": "number"},
        "(A)": {"prefix": "(", "suffix": ")", "start": 65, "type": "upper_letter"},
        "a)": {"prefix": "", "suffix": ")", "start": 97, "type": "lower_letter"},
        "1)": {"prefix": "", "suffix": ")", "start": 49, "type": "number"},
        "A)": {"prefix": "", "suffix": ")", "start": 65, "type": "upper_letter"},
        "a": {"prefix": "", "suffix": "", "start": 97, "type": "lower_letter"},
        "1": {"prefix": "", "suffix": "", "start": 49, "type": "number"},
        "A": {"prefix": "", "suffix": "", "start": 65, "type": "upper_letter"}
    }
    
    # 获取样式配置
    if style not in style_mappings:
        # 默认使用"(a)"样式
        style = "(a)"
    
    config = style_mappings[style]
    prefix = config["prefix"]
    suffix = config["suffix"]
    start_char = config["start"]
    
    labels = []
    
    for i in range(num_types):
        if config["type"] == "lower_letter" or config["type"] == "upper_letter":
            # 字母样式：a-z, A-Z
            char_code = start_char + i
            if config["type"] == "lower_letter" and char_code > 122:
                # 小写字母超出z，转为aa, ab, ac等
                first_letter = chr(97 + (i // 26) - 1)
                second_letter = chr(97 + (i % 26))
                label = f"{prefix}{first_letter}{second_letter}{suffix}"
            elif config["type"] == "upper_letter" and char_code > 90:
                # 大写字母超出Z，转为AA, AB, AC等
                first_letter = chr(65 + (i // 26) - 1)
                second_letter = chr(65 + (i % 26))
                label = f"{prefix}{first_letter}{second_letter}{suffix}"
            else:
                # 单个字母
                label = f"{prefix}{chr(char_code)}{suffix}"
        else:
            # 数字样式：1, 2, 3, ...
            label = f"{prefix}{i + 1}{suffix}"
        
        labels.append(label)
    
    return labels


def parallel_show(img_folder1: str, img_folder2: str, desc1: str, desc2: str, save_folder: str = "comparison_results"):
    """
    同时显示SA和GA优化结果的对比图
    
    参数:
        img_folder1: 文件夹路径1
        img_folder2: 文件夹路径2
        desc1: 图表标题1
        desc2: 图表标题2
        save_folder: 保存对比图的文件夹路径，默认"comparison_results"
    """
    import os
    import cv2
    import matplotlib.pyplot as plt
    
    # 获取两个文件夹中的所有PNG文件
    sa_files = sorted([f for f in os.listdir(img_folder1) if f.endswith('.png')])
    ga_files = sorted([f for f in os.listdir(img_folder2) if f.endswith('.png')])
    
    # 找出两个文件夹中共同的文件名
    common_files = sorted(list(set(sa_files) & set(ga_files)))
    
    if not common_files:
        print("警告：两个文件夹中没有共同的PNG文件")
        return
    
    print(f"找到 {len(common_files)} 个共同的PNG文件")
    
    # 创建2x1的子图布局
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 设置子图标题
    axes[0].set_title(desc1, fontsize=14, fontweight='bold')
    axes[1].set_title(desc2, fontsize=14, fontweight='bold')
    
    # 为每个共同的文件创建对比图
    for filename in common_files:
        # 读取SA图片
        sa_path = os.path.join(img_folder1, filename)
        sa_img = cv2.imread(sa_path)
        sa_img = cv2.cvtColor(sa_img, cv2.COLOR_BGR2RGB)  # 转换颜色空间
        
        # 读取GA图片
        ga_path = os.path.join(img_folder2, filename)
        ga_img = cv2.imread(ga_path)
        ga_img = cv2.cvtColor(ga_img, cv2.COLOR_BGR2RGB)  # 转换颜色空间
        
        # 清除之前的图像
        axes[0].clear()
        axes[1].clear()
        
        # 显示SA图片
        axes[0].imshow(sa_img)
        axes[0].set_title(f'{desc1} - {filename}', fontsize=12)
        axes[0].axis('off')
        
        # 显示GA图片
        axes[1].imshow(ga_img)
        axes[1].set_title(f'{desc2} - {filename}', fontsize=12)
        axes[1].axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图像
        plt.pause(0.5)  # 暂停0.5秒
        
        # 等待用户输入继续或退出
        response = input("按Enter继续显示下一张图片，输入's'保存当前图片，输入'q'退出: ")
        if response.lower() == 'q':
            break
        elif response.lower() == 's':
            # 保存当前对比图
            ensure_folder(save_folder)
            
            save_path = os.path.join(save_folder, f"comparison_{filename}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
            
            # 继续显示下一张图片
            continue
    
    plt.close()



def get_legend(save_path: str = "lu_legend.png"):
    """
    单独生成土地利用方案的图例
    """
    color_dict={}
    for key, value in COLOR_MAP.items():
        if key in ALL_LU_LABELS_DICT and int(key) > 10:
            color_dict[ALL_LU_LABELS_DICT[key]]=value
    fig = legend(color_dict, ncol=6, figsize=(11, 4))
    fig.savefig(save_path)
