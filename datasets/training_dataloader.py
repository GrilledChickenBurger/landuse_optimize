from numbers import Number
from typing import Iterator, Any, List
import torch
import os
import numpy as np
from utils.dataset_utils import *
from libs.const import *


def get_changelabel(src: np.ndarray, tgt: np.ndarray) -> int:
    """
    计算某索引在两个年份之间是否发生显著变化（用于only_changed筛选）

    Parameters:
    ----------
        src: np.ndarray, 源年份LU数组，shape为(128, 128)
        tgt: np.ndarray, 目标年份LU数组，shape为(128, 128)
    Returns:
    ----------
        ischanged: int, 是否发生显著变化（1/0）
    """
    assert src.shape == tgt.shape and len(src.shape) == 2
    diff = (src != tgt).astype(np.uint8)
    diff_processed = remove_lines(diff)

    num_unequal = np.count_nonzero(diff_processed)
    
    ischanged = 1 if num_unequal >= PIXEL_NUM_THRESHOLD_ISCHANGE else 0
    return ischanged

def create_training_data_pack(xf_src: np.ndarray, xf_tgt: np.ndarray, 
                                src_year: str, tgt_year: str, 
                                true_idx: str, 
                                factor_num: int, 
                                only_changed: bool):
    """创建数据pack的独立函数。每个pack包含：
    [srcFactors, srcLU(one-hot), diff, idx_label, changed_flag]

    Parameters:
    ----------
        xf_src: np.ndarray, src驱动因子+LU数组，shape为(factor_num, 128, 128)
        xf_tgt: np.ndarray, tgt驱动因子+LU数组，shape为(factor_num, 128, 128)
        src_year: str, 源年份
        tgt_year: str, 目标年份
        true_idx: str, 真实索引
        factor_num: int, 驱动因子数量
        only_changed: bool, 是否只选择变化的样本
    
    Returns:
    ----------
        pack: list[torch.Tensor], 数据pack，
        包含驱动因子、srcLU(one-hot)、diff、idx_label、changed_flag
    """
    pack = []
    time_diff = int(tgt_year) - int(src_year)
    idx_label = f"{src_year}_{true_idx}_{time_diff}"
    change_label = get_changelabel(xf_src[-1], xf_tgt[-1])
    
    if (only_changed and change_label == 1) or (not only_changed):
        # 计算diff: tgtLU - srcLU
        diff = np.expand_dims(xf_src[-1] != xf_tgt[-1], axis=0)
        diff_processed = torch.from_numpy(diff)
        # 对srcLU进行one-hot编码
        srcLU_onehot = one_hot_lu(torch.from_numpy(xf_src[factor_num - 1 :]))
        
        pack.extend([
            torch.from_numpy(xf_src[: factor_num - 1]),  # 驱动因子
            srcLU_onehot,  # srcLU(one-hot)
            diff_processed,  # diff
            idx_label,  # 索引标签
            change_label,  # 变化标签
        ])
    return pack

def apply_transform(pack, transform, repeat_num, 
                    factor_idx=0, src_idx=1, diff_idx=2):
    """对数据pack应用变换增强

    Parameters:
    ----------
        pack: list[torch.Tensor], 原始数据pack，
            包含驱动因子、srcLU(one-hot)、diff、idx_label、changed_flag
        transform: 数据变换增强
        repeat_num: 增强次数
        factor_idx: 驱动因子索引
        src_idx: 源数据索引
        diff_idx: 目标数据索引
    """
    transform_pack = []
    factor = pack[factor_idx]
    src = pack[src_idx]
    diff = pack[diff_idx]
    for _ in range(repeat_num):
        transformed = transform(torch.cat([factor, src, diff], dim=0))
        # 创建一个pack的副本，保持原始结构
        t_pack = list(pack)
        # 根据传入的索引位置更新源数据和目标数据
        t_pack[factor_idx] = transformed[:factor.shape[0]]
        t_pack[src_idx] = transformed[factor.shape[0]:src.shape[0] + factor.shape[0]]
        t_pack[diff_idx] = transformed[src.shape[0] + factor.shape[0]:]
        transform_pack.append(t_pack)
    return transform_pack

def load_training_data(root: str, split: str, shape: List[int], condition_list: List[str],
                 only_changed: bool = False, pack: List = None, transform=None, normalize: bool = False) -> List[List]:
    """
    从 {root}/{train|val}/valid_{10,15,20}/{FACTOR}/ 中加载
    多年份的 LU + 驱动因子，并按 pack 组合成训练样本。

    Parameters:
    ----------
        root: str 数据根目录
        split: str 数据集划分（train/val）
        shape: List[int] 数据形状（[1, 128, 128]）
        condition_list: List[str] 驱动因子列表
        only_changed: Optional[bool] 是否只选择变化的样本, 默认False
        pack: Optional[List] 加载哪一年的数据对,包括 [10, 15] | [15, 20] | [10, 20] | [-1, -1], 默认None
        transform: Optional[Callable] 数据变换增强（可选）
        normalize: Optional[bool] 是否归一化数据, 默认False
        
    Returns:
    ----------
        xf_pack_list: 每个元素为 [factors(src年), srcLU(one-hot), diff, idx_label, changed_flag]
    """
    print(f"\n================= 开始加载训练模型数据集【{split}】 ... =================")
    cond_list = condition_list[:]
    assert set(cond_list).issubset(set(ALL_FACTORS)), "condition_list value error."
    cond_list.append("LU")
    factor_num = len(cond_list)
    years = ALL_YEARS

    # 对于非List[List]类型的pack，转换为List[List]，便于后续遍历列表添加对应数据pack
    packitems = pack if isinstance(pack[0], list) else [pack]
    for p in packitems:
        assert p in ALL_PACK, "pack value error."
    
    files = [f for f in os.listdir(os.path.join(root, split, f"valid_{years[0]}", cond_list[0])) if f.lower().endswith(".tif")]
    num_samples = len(files)
    print(f"Data: {num_samples} LUimage in total. Pack: {pack}")
    
    # 加载多年份的 LU + 驱动因子数据, 每个年份的数据shape为 (num_samples, factor_num, 128, 128)
    allxf = []
    for i in range(len(years)):
        xf_root = os.path.join(root, split, f"valid_{years[i]}")
        xf = get_year_data(xf_root, cond_list, shape)
        
        if normalize:
            params = FACTOR_PARAMS_DICT[years[i]]  # 获取该年份的miu, epsilon参数
            # 对每个因子分别进行标准化
            for j, factor_name in enumerate(condition_list):
                if factor_name in params:  # 只对在参数字典中定义的因子进行标准化
                    log = factor_name in ALL_LOG_FACTORS  # 是否对该因子取对数
                    xf[:, j] = apply_norm(xf[:, j], factor_name, params, log=log)
            
        allxf.append(xf)
    
    def tmp_sort_key(file_idx: str) -> int:
        return int(file_idx)
    true_idx = sorted(
        [f.replace(".TIF", "").replace(".tif", "").split("_")[-1] for f in files],
        key=tmp_sort_key,
    )

    # pack列表
    xf_pack_list:List[List] = []
    for idx in range(num_samples):
        # 原始的3个pack: [10,15], [15,20], [10,20]
        pack1 = create_training_data_pack(
            allxf[0][idx], allxf[1][idx], years[0], years[1], true_idx[idx], factor_num, only_changed)
        pack2 = create_training_data_pack(
            allxf[1][idx], allxf[2][idx], years[1], years[2], true_idx[idx], factor_num, only_changed)
        pack3 = create_training_data_pack(
            allxf[0][idx], allxf[2][idx], years[0], years[2], true_idx[idx], factor_num, only_changed)
        ALL_PACK_ORI_DICT = {
            (10, 15): pack1,
            (15, 20): pack2,
            (10, 20): pack3,
        }
        packlist = []
        # 遍历packitems，根据pack组合创建数据pack
        for p in packitems:
            if p == [-1, -1]:
                packlist = [pack1, pack2, pack3]
                break
            else:
                p_tuple = tuple(p)
                pack = ALL_PACK_ORI_DICT[p_tuple]
                # 检查pack是否为空（当only_changed=True且样本未变化时）
                if pack:  # 只有当pack不为空时才处理
                    packlist.append(pack)
                    is_change = pack[-1]
                    # 在训练数据中，只对变化的样本应用变换增强
                    if split != 'val' and is_change and (not only_changed) and transform is not None:
                        packlist.extend(apply_transform(pack, transform, ALL_PACK_TRANSFORM_REPEAT_DICT[p_tuple]))

        xf_pack_list.extend([p for p in packlist if p])

    print("only changed: {}, xf_pack_list includes {} LUimg packs, each pack has {} factors.".format(
        only_changed, len(xf_pack_list), len(xf_pack_list[0])
    ))
    
    # 输出每个pack中各个因子的意义和形状
    if len(xf_pack_list) > 0:
        pack_sample = xf_pack_list[0]
        factor_names = ["srcFactors", "srcLU(one-hot)", "diff", "idx_label", "changed_flag"]
        for i, (factor, name) in enumerate(zip(pack_sample, factor_names)):
            if isinstance(factor, torch.Tensor):
                print("  {}: {}, shape={}".format(i, name, list(factor.shape)))
            else:
                print("  {}: {}, value={}".format(i, name, factor))

    return xf_pack_list


