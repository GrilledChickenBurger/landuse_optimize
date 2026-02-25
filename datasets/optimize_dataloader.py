from typing import Iterator, List
import os
import numpy as np
from utils.dataset_utils import get_year_data

def load_optimize_data(
                root: str, 
                shape: List[int], 
                condition_list: List[str],
                ES_list: List[str], 
                year: str,
                use_parallel: bool = True, 
                max_workers: int = 2, 
                ) -> List[List]:
    """
    加载 SA 优化所需的单样本打包：
      [factors(不含RA/LU), restrict(RA), es_data(ES_list), ori_LU(year), tgt_LU(year+5), idx_label]
    
    Parameters:
    ----------
        root:  str 数据根目录
        shape: List[int] 数据形状
        condition_list: List[str] 因子列表
        ES_list: List[str] ES 数据列表
        year: str 年份
        use_parallel: bool, 是否使用并行处理
        max_workers: int, 并行工作线程数
        
    Returns:
    ----------
        xf_pack_list: 每个元素为 [factors(src年, 不含RA/LU), restrict(RA), es_data(ES_list), srcLU, tgtLU, idx_label]
    """
    print(f"\n================= 开始加载随机优化数据集【{year}】... =================")
    root_ori = os.path.join(root, f"valid_{year}")
    root_tgt = os.path.join(root, f"valid_{str(int(year)+5)}") 
    if year!= "20":
        root_tgt = os.path.join(root, "valid_20")
    cond_list = condition_list[:]
    cond_list.append("RA")
    cond_list.append("LU")
    factor_num = len(cond_list)

    files = [f for f in os.listdir(os.path.join(root_ori, cond_list[0])) if f.lower().endswith(".tif")]
    num_samples = len(files)
    print("Data: {} LUimage need to predict in total.".format(num_samples))
 
    # 原年份的 LU + 因子 + ES 数据
    xf_ori = get_year_data(root_ori, cond_list, shape, max_workers=max_workers, use_parallel=use_parallel)
    xf_tgt = get_year_data(root_tgt, ["LU"], shape, max_workers=max_workers, use_parallel=use_parallel)
    es_data = get_year_data(root_ori, ES_list, shape, max_workers=max_workers, use_parallel=use_parallel)
    print(f"es_data shape: {es_data.shape}")
    print(f"xf_ori shape: {xf_ori.shape}")
    print(f"xf_tgt shape: {xf_tgt.shape}")

    def tmp_sort_key(file_idx: str) -> int:
        return int(file_idx)
    true_idx = sorted(
        [f.replace(".TIF", "").split("_")[-1] for f in files],
        key=tmp_sort_key,
    )

    # 打包（与原实现一致）
    xf_pack_list: List[List] = []
    for idx in range(num_samples):
        idx_label = f"{year}_{true_idx[idx]}_5"
        pack1 = [
            xf_ori[idx][: factor_num - 2],                    # factors
            xf_ori[idx][factor_num - 2 : factor_num - 1],     # restrict (RA)
            es_data[idx],                                     # ES
            xf_ori[idx][factor_num - 1 :],                    # ori LU
            xf_tgt[idx][0:1],                                 # tgt LU
            idx_label,
        ]
        xf_pack_list.append([pack1][0])

    assert len(xf_pack_list) == num_samples
    print("Done. xf_pack_list includes {} LUimg packs.\n".format(num_samples))
    return xf_pack_list
   
