# step0: 开始训练之前，获取各项驱动因子的数据分布参数
# 这些参数将用于训练时的标准化/归一化处理
# 参数
from typing import Dict, Any, List
import json
import tqdm
import torch
from torch.utils.data import DataLoader

from libs.const import ALL_YEARS
from datasets.base_dataloader import get_dataset
from utils.param_utils import RunningStats
from utils.tools import get_yaml

def get_params(config_path: str) -> Dict[str, Any]:
    """
    获取数据分布参数
    """
    # 1. 加载数据
    config = get_yaml(config_path)
    dataset = get_dataset(
        type="optimize",
        data_config=config
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
    )

    # 初始化统计器
    stats = {
        name: RunningStats()
        for name in config.factor_list
    }

    # 使用tqdm显示进度条
    progress_bar = tqdm.tqdm(
        dataloader, 
        desc="Processing batches",
        total=len(dataloader),
        unit="batch",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for i_batch, data_pack in enumerate(progress_bar):
        # 驱动因子 (batch_size, factor_num, 128, 128)
        factors = torch.tensor(data_pack[0], dtype=torch.float32)
        factor_dict = get_data_dict(factors, config.factor_list)
        # src LU (batch_size, 1, 128, 128)
        src_lu = torch.tensor(data_pack[3], dtype=torch.int64)
        # 过滤nodata区域，元素=True表示有效像素，False表示nodata像素
        nodata_mask = (src_lu != 0).squeeze(1)  # shape=(batch_size, 128, 128)
        # 对每个驱动因子进行统计
        for name, x in factor_dict.items():
            if name in ["GDP", "RK", "DL"]:   # 示例
                x = torch.log1p(x)
            stats[name].update(x, mask=nodata_mask)
        
    # 汇总结果
    params = {}
    for name, rs in stats.items():
        miu, std = rs.finalize()
        params[name] = {
            "n": rs.n,
            "miu": float(miu),
            "epsilon": float(std + 1e-6)
        }
    print(params)
    
    # 保存参数
    result_path = config.result_path.format(year=config.year)
    with open(result_path, "w") as f:
        json.dump(params, f, indent=4)
    
    return params
    
def get_data_dict(data_pack: torch.Tensor, factor_list: List[str], prev_data_dict: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
    """
    从数据打包中提取并返回必要的张量字典
    
    Parameters:
    ----------
    data_pack: torch.Tensor, shape=(batch_size, factor_num, H, W)
    factor_list: List[str], 和data_pack中的驱动因子顺序对应
    prev_data_dict: Optional[Dict[str, torch.Tensor]] = None, 
        若不为None则把新的键值对添加到prev_data_dict中, 若为None则直接返回新的data_dict
    
    Returns:
    --------
    data_dict: Dict[str, torch.Tensor], 键名对应factor_list中的元素，保持4维结构(1, 1, H, W)
    """
    data_dict = {} if prev_data_dict is None else prev_data_dict.copy()
    for i, factor in enumerate(factor_list):
        # 提取对应因子的数据，并保持4维结构
        d = data_pack[:, i]  # shape=(batch_size, 1, H, W)
        data_dict[factor] = d
    return data_dict

