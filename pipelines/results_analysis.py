# step3: 根据performance.json分析实验结果
# 实验结果文件夹结构:
# results/SA/{timestamp}/
# ├── default/
# │   ├── 10_479_5/
# │   │       └── performance_score.json
# │   ├── 10_596_5/
# │   │       └── performance_score.json
# │   └── ...
# └── farmer/
#     └── ...

import os
import sys

from torch import layout

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from typing import Dict, List
from utils.analysis_utils import *

def generate_lu_bars(save_folder: str, 
                    data_folder: str = None, 
                    json_path: str = None):
    """
    从JSON文件中读取地类面积数据，可视化每个场景中的LU类型面积变化比例直方图
    
    Parameters:
    -----------
    - data_folder: 实验结果文件夹路径, 格式为 "results\\SA|GA\\{timestamp}"
    - save_folder: 保存图表的文件夹路径
    - json_path: 包含地类面积数据的JSON文件路径
    """
    if json_path is None or not os.path.exists(json_path):
        print(f"JSON文件路径 {json_path} 不存在, 通过get_performance_metrics计算")
        # 读取指标
        if data_folder is None:
            raise ValueError("data_folder must be provided if json_path is None")
        else:
            metrics = get_performance_metrics(data_folder)
            total_area_data = cal_total_area(metrics, True)
            with open(os.path.join(data_folder, "total_area.json"), 'w', encoding='utf-8') as f:
                json.dump(total_area_data, f, ensure_ascii=False, indent=4)
    else:
        # 使用工具函数读取JSON文件并自动转换字符串键为整数
        total_area_data = load_json_with_int_keys(json_path)

    # 展示【origin, target与4个场景】的总面积占比对比图
    visualize_by_stackbar(total_area_data, 
                          total_title="各情景土地利用总面积占比对比图",
                          label_format="percentage",
                          y_axis_ratio=0.9,
                          save_path=os.path.join(save_folder, "bars", "total_area_stackbar.png"))
    # 每个LU一张子图，每个子图展示【origin, target与4个场景】的地类面积柱状图
    visualize_by_type_bars(total_area_data, 
                          total_title="土地利用各情景总面积对比图",
                          show_legend=False,
                          y_axis_ratio=0.9,
                          save_folder=os.path.join(save_folder, "bars"))

def generate_lu_hist_or_box(save_folder: str, 
                        data_folder: str = None, 
                        json_path: str = None,
                        comparison_type: str = "origin",
                        type: str = "box",
                        density: bool = True):
    """
    从JSON文件中读取地类面积变化比例数据，可视化每个场景中的LU类型面积变化比例直方图
    
    Parameters:
    -----------
    - data_folder: 实验结果文件夹路径, 格式为 "results\\SA|GA\\{timestamp}"
    - save_folder: 保存图表的文件夹路径
    - json_path: 包含地类面积变化比例数据的JSON文件路径
    - comparison_type: 比较类型，"origin"或"target"
    - type: 可视化类型，"hist"或"box"
    - density: 直方图展示频数/密度，True表示展示密度，False表示展示频数
    """
    if json_path is None or not os.path.exists(json_path):
        print(f"JSON文件路径 {json_path} 不存在, 通过get_performance_metrics计算")
        # 读取指标
        if data_folder is None:
            raise ValueError("data_folder must be provided if json_path is None")
        else:
            metrics = get_performance_metrics(data_folder)
            sample_area_ratio_data = cal_sample_area_change_ratio(metrics, 
                                                    aggregate_by_first_digit=True, 
                                                    comparison_type=comparison_type)
            with open(os.path.join(data_folder, f"sample_area_{comparison_type}.json"), 'w', encoding='utf-8') as f:
                json.dump(sample_area_ratio_data, f, ensure_ascii=False, indent=4)
    else:
        # 使用工具函数读取JSON文件并自动转换字符串键为整数
        sample_area_ratio_data = load_json_with_int_keys(json_path)

    if type == "hist":
        # 每个LU一张子图，每个子图展示【4个场景】的地类面积变化比例分布直方图
        visualize_by_hist(sample_area_ratio_data, 
                          comparison_type=comparison_type,
                          density=density,
                          bins=20,
                          histtype="bar",
                          remove_extremes=True,
                          method="percentile",
                          lower_bound=0.05,
                          upper_bound=0.95,
                          save_folder=os.path.join(save_folder, "hist", f"{comparison_type}_density_{density}"))
    elif type == "box":
        # 每个LU一张子图，每个子图展示【4个场景】的地类面积变化比例分布箱线图
        visualize_by_box(sample_area_ratio_data, 
                          comparison_type=comparison_type,
                          show_fliers = False,   
                          show_means = False,
                          save_folder=os.path.join(save_folder, "box", f"{comparison_type}"))



def generate_es_bars(save_folder: str, data_folder: str = None, json_path: str = None):
    """
    从JSON文件中读取ES价值数据，可视化每个场景中的ES类型价值变化比例直方图
    
    Parameters:
    -----------
    - data_folder: 实验结果文件夹路径, 格式为 "results\\SA|GA\\{timestamp}"
    - save_folder: 保存图表的文件夹路径
    - json_path: 包含ES价值数据的JSON文件路径
    """
    if json_path is None or not os.path.exists(json_path):
        print(f"JSON文件路径 {json_path} 不存在, 通过get_performance_metrics计算")
        # 读取指标
        if data_folder is None:
            raise ValueError("data_folder must be provided if json_path is None")
        else:
            metrics = get_performance_metrics(data_folder)
            total_es_data = cal_total_es(metrics)
            with open(os.path.join(data_folder, "total_es.json"), 'w', encoding='utf-8') as f:
                json.dump(total_es_data, f, ensure_ascii=False, indent=4)
    else:
        # 使用工具函数读取JSON文件并自动转换字符串键为整数
        total_es_data = load_json_with_int_keys(json_path)


    # 每个ES一张子图，每个子图展示【origin, target与4个场景】的ES价值柱状图
    visualize_by_type_bars(total_es_data, 
                          total_title="各情景ES总价值对比图",
                          show_legend=False,
                          show_labels=False,
                          y_axis_ratio=0.95,
                          save_folder=os.path.join(save_folder, "bars"))



def genarate_es_hist_or_box(save_folder: str, 
                            data_folder: str = None, 
                            json_path: str = None,
                            comparison_type: str = "origin",
                            type: str = "box",
                            density: bool = True):
    """
    从JSON文件中读取ES变化比例数据，可视化每个场景中的ES类型价值变化比例直方图
    
    Parameters:
    -----------
    - data_folder: 实验结果文件夹路径, 格式为 "results\\SA|GA\\{timestamp}"
    - save_folder: 保存图表的文件夹路径
    - json_path: 包含ES变化比例数据的JSON文件路径
    - type: 可视化类型，"hist"或"box"
    - density: 直方图展示频数/密度，True表示展示密度，False表示展示频数
    """
    if json_path is None or not os.path.exists(json_path):
        print(f"JSON文件路径 {json_path} 不存在, 通过get_performance_metrics计算")
        # 读取指标
        if data_folder is None:
            raise ValueError("data_folder must be provided if json_path is None")
        else:
            metrics = get_performance_metrics(data_folder)
            sample_es_ratio_data = cal_sample_es_change_ratio(metrics, 
                                                    comparison_type=comparison_type)
            with open(os.path.join(data_folder, f"sample_es_{comparison_type}.json"), 'w', encoding='utf-8') as f:
                json.dump(sample_es_ratio_data, f, ensure_ascii=False, indent=4)
    else:
        # 使用工具函数读取JSON文件并自动转换字符串键为整数
        sample_es_ratio_data = load_json_with_int_keys(json_path)
    
    if type == "hist":
        # 每个ES一张子图，每个子图展示【4个场景】的价值变化比例分布直方图
        visualize_by_hist(sample_es_ratio_data, 
                          comparison_type=comparison_type,
                          density=density,
                          bins=20,
                          histtype="bar",
                          remove_extremes=True,
                          method="percentile",
                          lower_bound=0.05,
                          upper_bound=0.95,
                          save_folder=os.path.join(save_folder, "hist", f"{comparison_type}_density_{density}"))
    elif type == "box":
        # 每个ES一张子图，每个子图展示【4个场景】的价值变化比例分布箱线图
        visualize_by_box(sample_es_ratio_data, 
                          comparison_type=comparison_type,
                          show_fliers = False,   
                          show_means = False,
                          save_folder=os.path.join(save_folder, "box", f"{comparison_type}"))
        
        # 转换数据格式：从{场景: {ES类型: 列表}} 转换为 {ES类型: {场景: 列表}}
        new_dict = {}
        es_types = ["food", "carbon", "water", "culture"]
        scenarios = ["default", "farmer", "enterprise", "tourist"]
        
        for es_type in es_types:
            new_dict[es_type] = {}
            for scenario in scenarios:
                # 安全获取数据，如果不存在则设为空列表
                if scenario in sample_es_ratio_data and es_type in sample_es_ratio_data[scenario]:
                    new_dict[es_type][scenario] = sample_es_ratio_data[scenario][es_type]
                else:
                    new_dict[es_type][scenario] = []
        visualize_by_box(new_dict, 
                          comparison_type=comparison_type,
                          show_fliers = False,   
                          show_means = False,
                          save_folder=os.path.join(save_folder, "box", f"{comparison_type}_new"))


def complete_pipeline(expr_path):
    """
    expr_path: 格式为 "SA\\{timestamp}" 或 "GA\\{timestamp}"
    """
    data_folder = os.path.join("results", expr_path)
    area_save_folder = os.path.join("figures", expr_path, "area")
    es_save_folder = os.path.join("figures", expr_path, "es")
    # 生成地类面积柱状图(stackbar + bars)
    # generate_lu_bars(area_save_folder, data_folder)
    # 生成地类面积变化比例箱线图
    comparison_type = "target"
    generate_lu_hist_or_box(area_save_folder, data_folder, json_path=None,
                             type="box",
                             density=True,
                             comparison_type=comparison_type)
    # # 生成ES价值柱状图
    # generate_es_bars(es_save_folder, data_folder, json_path=None)
    # 生成ES价值变化比例箱线图
    genarate_es_hist_or_box(es_save_folder, data_folder, json_path=None, 
                             type="box",
                             density=True,
                             comparison_type=comparison_type)

def picker(expr_path, min_improvement_threshold=0.0, 
                    min_positive_scenarios=2,
                    min_other_threshold=-0.05):
    """
    找到一批索引，它们在各个场景中对ES都有一定提升，而且不同场景的提升比例有显著差距
    
    Parameters:
    -----------
    expr_path: 格式为 "SA\\{timestamp}" 或 "GA\\{timestamp}"
    min_improvement_threshold: 最小提升阈值，默认0.0表示所有ES类型都需要有提升
    min_variation_threshold: 最小变异阈值，用于衡量不同场景间的提升比例差异
    min_positive_scenarios: 需要有多少个场景显示提升，默认4表示所有场景都需要有提升
    
    Returns:
    --------
    selected_indices: 满足条件的索引列表
    performance_data: 每个索引的性能数据字典
    """
    data_folder = os.path.join("results", expr_path)
    sample_save_folder = os.path.join("figures", expr_path, "samples_new")
    
    metrics = get_performance_metrics(data_folder)
    best_samples, _ = get_best_samples(metrics=metrics, 
                                        min_improvement_threshold=min_improvement_threshold, 
                                        min_positive_scenarios=min_positive_scenarios, 
                                        min_other_threshold=min_other_threshold)
    print(best_samples)

    for idx in best_samples:
        visualize_best_samples(idx, data_folder, sample_save_folder, layout=(1, 6))

def all_samples(expr_path):
    """
    expr_path: 格式为 "SA\\{timestamp}" 或 "GA\\{timestamp}"
    返回所有索引的列表
    """
    data_folder = os.path.join("results", expr_path)
    sample_save_folder = os.path.join("figures", expr_path, "all_samples")
    
    if not os.path.exists(data_folder):
        raise ValueError(f"数据文件夹不存在: {data_folder}")
    
    indices = set()
    
    # 遍历数据文件夹下的所有子目录（策略文件夹）
    for strategy_dir in os.listdir(data_folder):
        strategy_path = os.path.join(data_folder, strategy_dir)
        if os.path.isdir(strategy_path):
            # 在策略文件夹中查找索引文件夹
            for item in os.listdir(strategy_path):
                item_path = os.path.join(strategy_path, item)
                if os.path.isdir(item_path):
                    # 检查是否是索引文件夹（格式如 10_355_5）
                    if '_' in item and item.replace('_', '').isdigit():
                        indices.add(item.split("_")[1])
    
    for idx in indices:
        visualize_best_samples(idx, data_folder, sample_save_folder, layout=(1, 6))


def show_SA_GA(expr_path1, expr_path2):
    """
    同时显示SA和GA优化结果的对比图
    
    参数:
    expr_path1: 格式为 "SA\\{timestamp}"
    expr_path2: 格式为 "GA\\{timestamp}"
    """
    img_folder1 = os.path.join("figures", expr_path1, "all_samples")
    img_folder2 = os.path.join("figures", expr_path2, "all_samples")
    save_folder = os.path.join("figures", "comparison_results")

    parallel_show(img_folder1, img_folder2, "SA优化结果", "GA优化结果", save_folder)


def generate_lu_bars_ori_tgt(save_folder: str, 
                    data_folder: str = None, 
                    json_path: str = None):
    """
    从JSON文件中读取地类面积数据，可视化origin和target两项的土地利用总面积占比堆叠柱状图
    
    Parameters:
    -----------
    - data_folder: 实验结果文件夹路径, 格式为 "results\\SA|GA\\{timestamp}"
    - save_folder: 保存图表的文件夹路径
    - json_path: 包含地类面积数据的JSON文件路径
    """
    if json_path is None or not os.path.exists(json_path):
        print(f"JSON文件路径 {json_path} 不存在, 通过get_performance_metrics计算")
        # 读取指标
        if data_folder is None:
            raise ValueError("data_folder must be provided if json_path is None")
        else:
            metrics = get_performance_metrics(data_folder)
            total_area_data = cal_total_area(metrics, True)
            with open(os.path.join(data_folder, "total_area.json"), 'w', encoding='utf-8') as f:
                json.dump(total_area_data, f, ensure_ascii=False, indent=4)
    else:
        # 使用工具函数读取JSON文件并自动转换字符串键为整数
        total_area_data = load_json_with_int_keys(json_path)

    # 只保留origin和target两项
    filtered_data = {}
    for key in ["origin", "target"]:
        if key in total_area_data:
            filtered_data[key] = total_area_data[key]

    # 展示【origin, target】的总面积占比对比图
    visualize_by_stackbar(filtered_data, 
                          total_title="原始与目标土地利用总面积占比对比图",
                          label_format="percentage",
                          y_axis_ratio=0.9,
                          save_path=os.path.join(save_folder, "bars", "origin_target_stackbar.png"))


if __name__ == "__main__":
    SA = "SA\\20260301_101241"
    GA1 = "GA\\20260305_120735"
    GA2 = "GA\\20260306_002357"
    complete_pipeline(GA1)

    data_folder = os.path.join("results", GA1)
    area_save_folder = os.path.join("figures", GA1, "area")
    generate_lu_bars_ori_tgt(area_save_folder, data_folder)
    
    # all_samples(GA1)
    show_SA_GA(SA, GA1)
    idxs1= ['1466', '1464', '885', '2502', '758', '2501', '1528', '2439', '1853', '2201', '701', '1454', '761', '1583', '635', '1359', '1852', '1526', '1297', '2377', '1937', '632', '823', '1527', '886', '1724', '825', '1455', '1725', '1071', '639', '1298', '1663']
    idx1 = ['1454', '1516', '1581', '1583', '1853', '2201', '577', '640', '701', '702', '758', '699', '700', '1403', '1514', '1582', '1917', '887', '948'] 
    # print("SA的idxs:", idxs1)
    
    idxs2 = ['1528', '886', '632', '1466', '1071', '1579', '1875', '2378', '823', '695', '2377', '825', '1663', '516', '700', '701', '571', '1580', '577', '1456']
    idx2 = ['1518', '2377', '571', '632', '1527', '1916', '758', '1390']
    # print("GA1的idxs:", idxs2)
    
    # both = set(idxs1) & set(idxs2)  # ['2377', '823', '825', '701', '886', '1663', '1071', '632', '1466', '1528']
    both = set(idxs1) & set(idx1)  # ['2377', '823', '825', '701', '886', '1663', '1071', '632', '1466', '1528']
    # both = set(idx1) & set(idx2)  # ['758']
    print("同时在SA和GA1中出现的idxs:", both)
    # picker(GA1)

