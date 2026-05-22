# before step2: 探索GA的最佳参数组合
from argparse import Namespace
import json
import os
import copy
import sys
from datetime import datetime
from typing import Dict, Any, List
import yaml
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入优化器模块
from runner import create_runner
# 导入数据加载模块
from datasets.base_dataloader import get_dataset
from utils.calculator_utils import train_maxent
from utils.tools import ensure_folder, get_yaml


# 实验配置列表
# 命名规则: GA_{pool}_{selection}_{elite}_{noncrossover}_{elitemut}
# pool: A=all, E=elite
# selection: R=roulette, T=tournament  
# elite: 精英比例 (0.1, 0.2, 0.5, 0.0)
# noncrossover: C=copy, CM=copy_mutate
# elitemut: F=False, T=True
experiments = [
    # {   # GA1 
    #     'name': 'GA_A_R_0.1_C_F',
    #     'parent_pool': 'all',
    #     'parent_selection': 'roulette',
    #     'elite_ratio': 0.1,
    #     'non_crossover_handling': 'copy',
    #     'elite_mutation': False
    # },
    {   # GA2
        'name': 'GA_E0.3_R_0.1_C_F',
        'parent_pool': 'elite',
        'elite_ratio_for_pool': 0.3,   # 父本来源的前20%
        'parent_selection': 'roulette',
        'elite_ratio': 0.2,
        'non_crossover_handling': 'copy',
        'elite_mutation': False
    },
    {   # GA3
        'name': 'GA_A_T_0.1_C_F',
        'parent_pool': 'all',
        'parent_selection': 'tournament',
        'tournament_k': 2,
        'elite_ratio': 0.1,
        'non_crossover_handling': 'copy',
        'elite_mutation': False
    },
    # {   # GA4
    #     'name': 'GA_A_R_0.0_C_F',
    #     'parent_pool': 'all',
    #     'parent_selection': 'roulette',
    #     'elite_ratio': 0.0,
    #     'non_crossover_handling': 'copy',
    #     'elite_mutation': False
    # },
    # {   # GA6
    #     'name': 'GA_A_R_0.1_CM_F',
    #     'parent_pool': 'all',
    #     'parent_selection': 'roulette',
    #     'elite_ratio': 0.1,
    #     'non_crossover_handling': 'copy_mutate',
    #     'elite_mutation': False
    # },
    # {   # GA7
    #     'name': 'GA_A_R_0.1_C_T',
    #     'parent_pool': 'all',
    #     'parent_selection': 'roulette',
    #     'elite_ratio': 0.1,
    #     'non_crossover_handling': 'copy',
    #     'elite_mutation': True
    # }
]

def experiment_pipeline(config_path: str, 
             idxs: List[int] = None,
             is_processed: bool = False, 
             processed_timestamp: str = None) -> None:
    """
    随机优化器运行流水线
    
    Parameters:
    ----------
    config_path : str
        配置文件路径
    idxs : List[int], optional
        待优化样本索引列表，默认None
    is_processed : bool, optional
        是否已处理过数据，默认False
    processed_timestamp : str, optional
        已处理数据的时间戳，默认None
    """
    # 检查参数
    if is_processed and not processed_timestamp:
        raise ValueError("若已处理数据, 必须提供对应处理时间戳, 用于跳过已处理样本和保存新结果")

    print("================= 开始随机优化... =================")
    
    # 1. 加载配置
    config = get_yaml(config_path)
    print(f"Configuration loaded from {config_path}")
    
    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config.device.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # 3. 加载数据
    dataset = get_dataset(
        type="optimize",
        data_config=config.data
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
    )
    
    # 4. 训练MaxEnt模型并保存, 若已存在则跳过
    # 使用模板字符串根据年份生成具体的模型路径
    maxent_path = config.data.maxent_model_path
    maxent_path = maxent_path.format(year=config.data.year)
    # if not os.path.exists(maxent_path):
    train_maxent(config.data.root, config.data.year, maxent_path)

    # 5. 选择并运行对应的优化器
    runner_config = config.runner
    use_landscape = runner_config.use_landscape  # 是否使用景观破碎化约束
    optimize_num = runner_config.optimize_num  # 优化样本数量
    type_config = getattr(config, runner_config.type)  # 优化器类型配置, SA|GA
    scenario = runner_config.scenario  # scenario类型

    # 若已处理过数据则使用已处理时间戳，否则使用当前时间戳
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if not is_processed else processed_timestamp

    # 6. 运行优化器
    print(f"\n================= 随机优化类型: 【{runner_config.type}】 =================")
    print(f"  优化样本数量: {optimize_num}")
    print(f"  使用景观破碎化约束: {use_landscape}")

    # 7. 实验级别检查点：检查哪些实验已经完成
    base_save_folder = os.path.join(type_config.save_path, run_timestamp)
    ensure_folder(base_save_folder)
    # 获取已完成的实验列表（已存在的文件夹）
    completed_experiments = set() if not is_processed else set(os.listdir(base_save_folder))
    
    # 8. 遍历每个实验配置
    performance_dict = {}
    for experiment in experiments:
        experiment_name = experiment['name']
        
        # 实验级别检查：如果实验已完成且处于断点续传模式，则跳过整个实验
        if is_processed and experiment_name in completed_experiments:
            print(f"\n================= 实验【{experiment_name}】已完成，跳过 =================")
            continue
            
        print(f"\n================= 当前实验: 【{experiment_name}】 =================")
        print(f"  实验配置: {experiment}")
        
        # 更新配置中的strategy字段
        experiment_config = copy.deepcopy(type_config)
        experiment_config.strategy = Namespace(**experiment)
        weight_dict = vars(getattr(runner_config.weight, scenario))
        
        print(f"\n================= 当前 scenario: 【{scenario}】 =================")
        print(f"  权重配置: {weight_dict}")

        actual_optimize_num = 0
        # 更新保存路径结构: GA/timestamp/实验名称
        experiment_save_folder = os.path.join(base_save_folder, experiment_name)
        ensure_folder(experiment_save_folder)
        print(f"  结果保存目录: {experiment_save_folder}")

        # 样本级别检查点：已处理过样本索引
        processed_samples = set() if not is_processed else set(os.listdir(experiment_save_folder))

        performance_json_list = []
        for i, data_pack in enumerate(dataloader):
            print(f"\nProcessing sample {i+1}/{len(dataloader)}")

            # 跳过已处理样本
            index = data_pack[5][0]
            if is_processed and index in processed_samples:
                print(f"实验【{experiment_name}】scenario【{scenario}】已处理样本 {index}, 跳过")
                continue

            # 若给定待优化索引，跳过非待优化样本，否则默认优化所有样本
            if idxs and len(idxs)>0 and index.split('_')[1] not in [str(idx) for idx in idxs]:
                continue
            
            # 处理批次数据
            data_dict = process_batch(data_pack, config)
                
            # 运行对应优化器
            type_runner = create_runner(
                runner_type=runner_config.type,
                config=experiment_config,
                common_config=runner_config.common_params,
                model_config=config.model,
                device=device,
                data_dict=data_dict,
                index=index,
                save_path=experiment_save_folder,
                weight_dict=weight_dict,
                use_landscape=use_landscape,
            )
                
            # 执行优化
            best_plan = type_runner.run()
            if not torch.equal(best_plan, data_dict["ori_plan"]):
                actual_optimize_num += 1
                # 记录性能指标
                performance_json_list.append(os.path.join(experiment_save_folder, index, "performance_score.json"))
            
            if optimize_num != -1 and actual_optimize_num >= optimize_num:
                break
        
        # 9. 计算实验级别的性能指标
        performance_dict[experiment_name] = get_avg_performance(performance_json_list)['avg']
        with open(os.path.join(experiment_save_folder, "performance_score.json"), "w") as f:
            json.dump(performance_dict[experiment_name], f, indent=4)

    # 10. 全部实验结束，保存performance_dict
    with open(os.path.join(base_save_folder, "performance_dict.json"), "w") as f:
        json.dump(performance_dict, f, indent=4)

    print("\n================= 随机优化器运行完成! =================")

def process_batch(data_pack: List[torch.Tensor], config: Namespace) -> Dict[str, torch.Tensor]:
    """
    处理一个批次的数据，返回runner所需数据的data_dict

    Parameters:
    ----------
    data_pack: List[torch.Tensor], 
        格式: [srcFactors(不含RA/LU), restrict(RA), es_data(ES_list), srcLU, tgtLU, idx_label]
    

    Returns:
    --------
    data_dict: Dict[str, torch.Tensor], 包含所有必要的张量数据, 包括:
        - factors: src 驱动因子 (1, factor_num, 128, 128)
        - restrict: src 区域约束 (1, 1, 128, 128)
        - es_data: src ES数据 (1, ES_num, 128, 128)
        - ori_plan: src LU (1, 1, 128, 128)
        - final_plan: tgt LU (1, 1, 128, 128)
        - 各项factors的data_dict: 键名对应factor_list中的元素，保持4维结构(1, 1, H, W)
        - 各项es_data的data_dict: 键名对应es_list中的元素，保持4维结构(1, 1, H, W)
    """
    # 1. src 驱动因子 (1, factor_num, 128, 128)
    factors = torch.tensor(data_pack[0], dtype=torch.float32)
    factor_dict = get_data_dict(factors, config.data.factor_list)
    factor_dict["factors"] = factors
    # 2. src 区域约束 (1, 1, 128, 128)
    restrict_area = torch.tensor(data_pack[1], dtype=torch.float32)
    factor_dict["restrict"] = restrict_area
    # 3. src ES数据 (1, ES_num, 128, 128)
    es_data = torch.tensor(data_pack[2], dtype=torch.float32)
    final_data_dict = get_data_dict(es_data, config.data.es_list, factor_dict)
    # 4. src LU (1, 1, 128, 128)
    ori_plan = torch.tensor(data_pack[3], dtype=torch.float32)
    final_data_dict["ori_plan"] = ori_plan
    # 5. tgt LU (1, 1, 128, 128)
    final_plan = torch.tensor(data_pack[4], dtype=torch.float32)
    final_data_dict["final_plan"] = final_plan

    return final_data_dict

def get_data_dict(data: torch.Tensor, factor_list: List[str], prev_data_dict: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
    """
    从concat的张量中提取并返回必要的张量字典
    
    Parameters:
    ----------
    data: torch.Tensor, shape=(1, factor_num, H, W)
    factor_list: List[str], 和data中的channel顺序对应
    prev_data_dict: Optional[Dict[str, torch.Tensor]] = None, 
        若不为None则把新的键值对添加到prev_data_dict中, 若为None则直接返回新的data_dict
    
    Returns:
    --------
    data_dict: Dict[str, torch.Tensor], 键名对应factor_list中的元素，保持4维结构(1, 1, H, W)
    """
    data_dict = {} if prev_data_dict is None else prev_data_dict.copy()
    for i, factor in enumerate(factor_list):
        # 提取对应因子的数据，并保持4维结构
        d = data[:, i]  # shape=(1, H, W)
        data_dict[factor] = d.unsqueeze(1)  # 添加因子维度，shape=(1, 1, H, W)
    return data_dict

def get_avg_performance(json_paths, include_change_area=True):
    """
    计算多个实验样本的性能指标平均值
    
    Parameters:
    -----------
    json_paths : list of str
        性能评分JSON文件路径列表
    include_change_area : bool, optional
        是否包含变化区域指标，默认False
        
    Returns:
    --------
    dict
        包含平均提升和各样本详细结果的字典
        {
            "avg": {
                "ES": {
                    "food": {
                        "best_ori_improvement": 平均提升百分比,
                        "best_tgt_improvement": 平均提升百分比
                    },
                    "carbon": {...},
                    "water": {...},
                    "culture": {...}
                },
                "Change_area": {
                    "real_metrics": {
                        "accuracy(OA)": 平均值,
                        "precision(UA)": 平均值,
                        "recall(PA)": 平均值,
                        "f1_score": 平均值,
                        "iou": 平均值
                    },
                    "pred_metrics": {
                        "accuracy(OA)": 平均值,
                        "precision(UA)": 平均值,
                        "recall(PA)": 平均值,
                        "f1_score": 平均值,
                        "iou": 平均值
                    }
                }
            },
            "samples": {
                "sample1_path": {
                    "ES": {
                        "food": {
                            "best_ori_improvement": 提升百分比,
                            "best_tgt_improvement": 提升百分比
                        },
                        ...
                    },
                    "Change_area": {
                        "real_metrics": {...},
                        "pred_metrics": {...}
                    }
                },
                ...
            }
        }
    """    
    # 初始化结果字典
    results = {
        "avg": {
            "ES": {
                "food": {"best_ori_improvement": 0.0, "best_tgt_improvement": 0.0},
                "carbon": {"best_ori_improvement": 0.0, "best_tgt_improvement": 0.0},
                "water": {"best_ori_improvement": 0.0, "best_tgt_improvement": 0.0},
                "culture": {"best_ori_improvement": 0.0, "best_tgt_improvement": 0.0}
            }
        },
        "samples": {}
    }
    
    # 如果需要包含变化区域指标
    if include_change_area:
        results["avg"]["Change_area"] = {
            "real_metrics": {
                "accuracy(OA)": 0.0, "precision(UA)": 0.0, "recall(PA)": 0.0,
                "f1_score": 0.0, "iou": 0.0
            },
            "pred_metrics": {
                "accuracy(OA)": 0.0, "precision(UA)": 0.0, "recall(PA)": 0.0,
                "f1_score": 0.0, "iou": 0.0
            }
        }
    
    # 统计每个指标的总和和样本数量
    total_samples = 0
    # 初始化统计字典
    improvement_sums = {
        "food": {"best_ori": 0.0, "best_tgt": 0.0},
        "carbon": {"best_ori": 0.0, "best_tgt": 0.0},
        "water": {"best_ori": 0.0, "best_tgt": 0.0},
        "culture": {"best_ori": 0.0, "best_tgt": 0.0}
    }
    
    # 如果需要包含变化区域指标，初始化相应的统计字典
    if include_change_area:
        change_area_sums = {
            "real_metrics": {
                "accuracy(OA)": 0.0, "precision(UA)": 0.0, "recall(PA)": 0.0,
                "f1_score": 0.0, "iou": 0.0
            },
            "pred_metrics": {
                "accuracy(OA)": 0.0, "precision(UA)": 0.0, "recall(PA)": 0.0,
                "f1_score": 0.0, "iou": 0.0
            }
        }
    
    # 处理每个JSON文件
    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"警告: 文件不存在: {json_path}")
            continue
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            sample_results = {"ES": {}}
            
            # 处理ES数据
            if "ES" in data:
                es_data = data["ES"]
                
                # 计算每个指标的提升百分比
                for item in ["food", "carbon", "water", "culture"]:
                    if item not in es_data:
                        print(f"警告: 文件 {json_path} 中缺少 {item} 数据")
                        continue
                        
                    item_data = es_data[item]
                    
                    # 使用已有的差异值（已转换为百分比）
                    best_ori_improvement = item_data.get("best_ori_diff", 0.0) * 100  # 转换为百分比
                    best_tgt_improvement = item_data.get("best_tgt_diff", 0.0) * 100  # 转换为百分比
                    
                    sample_results["ES"][item] = {
                        "best_ori_improvement": best_ori_improvement,
                        "best_tgt_improvement": best_tgt_improvement
                    }
                    
                    # 累加到总和
                    improvement_sums[item]["best_ori"] += best_ori_improvement
                    improvement_sums[item]["best_tgt"] += best_tgt_improvement
            else:
                print(f"警告: 文件 {json_path} 中没有ES数据")
            
            # 处理变化区域数据
            if include_change_area and "Change_area" in data:
                change_area_data = data["Change_area"]
                sample_results["Change_area"] = {}
                
                # 处理真实变化区域指标
                if "real_metrics" in change_area_data:
                    real_metrics = change_area_data["real_metrics"]
                    sample_results["Change_area"]["real_metrics"] = {}
                    
                    for metric in ["accuracy(OA)", "precision(UA)", "recall(PA)", "f1_score", "iou"]:
                        if metric in real_metrics:
                            value = real_metrics[metric]
                            sample_results["Change_area"]["real_metrics"][metric] = value
                            change_area_sums["real_metrics"][metric] += value
                
                # 处理预测变化区域指标
                if "pred_metrics" in change_area_data:
                    pred_metrics = change_area_data["pred_metrics"]
                    sample_results["Change_area"]["pred_metrics"] = {}
                    
                    for metric in ["accuracy(OA)", "precision(UA)", "recall(PA)", "f1_score", "iou"]:
                        if metric in pred_metrics:
                            value = pred_metrics[metric]
                            sample_results["Change_area"]["pred_metrics"][metric] = value
                            change_area_sums["pred_metrics"][metric] += value
            
            # 保存样本结果
            idx = os.path.dirname(json_path).split(os.sep)[-1]
            results["samples"][idx] = sample_results
            total_samples += 1
            
        except Exception as e:
            print(f"错误: 处理文件 {json_path} 时发生异常: {e}")
            continue
    
    # 计算平均值
    if total_samples > 0:
        # ES指标平均值
        for item in ["food", "carbon", "water", "culture"]:
            results["avg"]["ES"][item]["best_ori_improvement"] = improvement_sums[item]["best_ori"] / total_samples
            results["avg"]["ES"][item]["best_tgt_improvement"] = improvement_sums[item]["best_tgt"] / total_samples
        
        # 变化区域指标平均值
        if include_change_area:
            for metric_type in ["real_metrics", "pred_metrics"]:
                for metric in ["accuracy(OA)", "precision(UA)", "recall(PA)", "f1_score", "iou"]:
                    results["avg"]["Change_area"][metric_type][metric] = change_area_sums[metric_type][metric] / total_samples
    
    return results

if __name__ == "__main__":
    base_folder = "results\\GA\\20260302_052623"
    experiment_name = "GA_E0.3_R_0.1_C_F"
    json_paths = [
        os.path.join(base_folder, experiment_name, f"10_{idx}_5\\performance_score.json")
        for idx in ["479", "596", "634"]
    ]

    results = get_avg_performance(json_paths)
    save_folder = os.path.join(base_folder, experiment_name)
    with open(os.path.join(save_folder, "performance_score.json"), "w") as f:
        json.dump(results, f, indent=4)
