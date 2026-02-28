# step2: 运行随机优化算法(SA|GA)
from argparse import Namespace
import os
from datetime import datetime
from typing import Dict, Any, List
import yaml
import torch
from torch.utils.data import DataLoader
# 导入优化器模块
from runner import create_runner
from runner.SA import SimulatedAnnealing
# 导入数据加载模块
from datasets.base_dataloader import get_dataset
from utils.calculator_utils import train_maxent
from utils.tools import ensure_folder, get_yaml

def pipeline(config_path: str, 
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
    if scenario == "all":
        scenario_list = [name for name in vars(runner_config.weight).keys()]
    else:
        scenario_list = [scenario]
    # 若已处理过数据则使用已处理时间戳，否则使用当前时间戳
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if not is_processed else processed_timestamp

    # 6. 运行优化器
    print(f"\n================= 随机优化类型: 【{runner_config.type}】 =================")
    print(f"  优化样本数量: {optimize_num}")
    print(f"  使用景观破碎化约束: {use_landscape}")

    # 7. 遍历每个scenario
    for current_scenario in scenario_list: 
        weight_dict = vars(getattr(runner_config.weight, current_scenario))
        
        print(f"\n================= 当前 scenario: 【{current_scenario}】 =================")
        print(f"  权重配置: {weight_dict}")

        actual_optimize_num = 0
        scenario_save_folder = os.path.join(type_config.save_path, run_timestamp, current_scenario)
        ensure_folder(scenario_save_folder)
        print(f"  结果保存目录: {scenario_save_folder}")

        # 已处理过样本索引
        processed_samples = set() if not is_processed else set(os.listdir(scenario_save_folder))

        for i, data_pack in enumerate(dataloader):
            print(f"\nProcessing sample {i+1}/{len(dataloader)}")

            # 跳过已处理样本
            index = data_pack[5][0]
            if is_processed and index in processed_samples:
                print(f"scenario【{current_scenario}】已处理样本 {index}, 跳过")
                continue

            # 若给定待优化索引，跳过非待优化样本，否则默认优化所有样本
            if idxs and len(idxs)>0 and index.split('_')[1] not in [str(idx) for idx in idxs]:
                continue
            
            # 处理批次数据
            data_dict = process_batch(data_pack, config)
                
            # 运行对应优化器
            type_runner = create_runner(
                runner_type=runner_config.type,
                config=type_config,
                common_config=runner_config.common_params,
                model_config=config.model,
                device=device,
                data_dict=data_dict,
                index=index,
                save_path=scenario_save_folder,
                weight_dict=weight_dict,
                use_landscape=use_landscape,
            )
                
            # 执行优化
            best_plan = type_runner.run()
            if not torch.equal(best_plan, data_dict["ori_plan"]):
                actual_optimize_num += 1
            
            if optimize_num != -1 and actual_optimize_num >= optimize_num:
                break
        
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

