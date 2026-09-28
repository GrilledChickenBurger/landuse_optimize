# 混合训练集加载器
# 训练集由 10-15 与 15-20 两个来源构成, 每个来源各自贡献"变化样本"与"未变化样本",
# 未变化样本的数量与来源均与同来源增强后的变化样本一一对应.
#
# 控制参数 (configs/hybridTrainConfig.yaml 的 hybrid 段):
#   train_size (N)               : 目标训练集样本量
#   ratio_1015 (r1)              : 10-15 变化样本加入比例, 占 10-15 变化池 (train+val)
#   ratio_1520 (r2)              : 15-20 变化样本加入比例, 占 15-20 变化池 (train+val)
#   changed_unchanged_ratio (rcu): 变化:非变化 (rcu=1 表示 1:1)
#
# 推导链:
#   raw_1015 = round(r1 * a), raw_1520 = round(r2 * b)        # a / b 为变化池容量
#   raw_changed = raw_1015 + raw_1520
#   Q = round(N * rcu / (1 + rcu))                            # 目标(增强后)变化总数
#   Q_1015 = round(Q * raw_1015 / raw_changed), Q_1520 = Q - Q_1015
#   组内配额分配: extra_g = Q_g - raw_g, base_g, rem_g = divmod(extra_g, raw_g)
#                 每个样本给 1 份原始 + base_g 份副本, 再随机挑 rem_g 个样本各多 1 份副本
#                 => 该组最终样本数精确等于 Q_g, 不受"整数倍"限制
#   unchanged_g = round(Q_g / rcu)                            # 从同来源未变化池无放回抽取
#
# 增强倍数 k_g = Q_g / raw_g 允许多数情形下是小数, 各样本实际贡献介于 [1+base_g, 2+base_g) 份.
#
# 实测池子容量 (train + val 合计):
#   10-15: 变化 a=294,  未变化 u_a=1521
#   15-20: 变化 b=593,  未变化 u_b=1222
# r2=1, rcu=1 时 N 的可行区间交集为 [1774, 2444] (瓶颈是 15-20 未变化池 1222),
# 建议 N=2000. 单个 r1 下的可行区间由 _feasible_range 计算, 报错时会打印.
#
# 产出格式与 training_dataloader 完全一致.
import os
from typing import List, Tuple

import numpy as np
from torch.utils import data
from torchvision import transforms

from libs.const import ALL_FACTORS, FACTOR_PARAMS_DICT, ALL_LOG_FACTORS
from utils.dataset_utils import get_year_data, apply_norm
from .training_dataloader import create_training_data_pack, apply_transform


# 进程内缓存: 多个模型会各构造一次 HybridDataset, 而 6 份年份数据完全相同(加载+归一化成本高).
# 只缓存主副本, 每次调用返回副本, 保证与"每次重新加载"语义完全一致(调用方可自由原地修改).
_YEAR_DATA_CACHE = {}


def _load_year_data(root: str, split: str, year: str, condition_list: List[str],
                    shape: List[int], normalize: bool) -> np.ndarray:
    """加载指定 split 下指定年份的 [因子..., LU] 数据, 可选归一化.

    返回 shape 为 (num_samples, len(condition_list)+1, H, W) 的 numpy 数组,
    末尾一列为 LU 通道, 与 training_dataloader.load_training_data 内部保持一致.
    """
    key = (os.path.abspath(root), split, year,
           tuple(condition_list), tuple(shape), bool(normalize))
    cached = _YEAR_DATA_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    cond_list = condition_list[:] + ["LU"]
    xf = get_year_data(os.path.join(root, split, f"valid_{year}"), cond_list, shape)
    if normalize:
        params = FACTOR_PARAMS_DICT[year]
        for j, factor_name in enumerate(condition_list):
            if factor_name in params:
                log = factor_name in ALL_LOG_FACTORS
                xf[:, j] = apply_norm(xf[:, j], factor_name, params, log=log)
    _YEAR_DATA_CACHE[key] = xf
    return xf.copy()


def _get_sample_indices(root: str, split: str, year: str, condition_list: List[str]) -> List[str]:
    """提取某 split 下样本索引(文件名最后一段数字), 按数值升序."""
    folder = os.path.join(root, split, f"valid_{year}", condition_list[0])
    files = [f for f in os.listdir(folder) if f.lower().endswith(".tif")]
    return sorted(
        [f.replace(".TIF", "").replace(".tif", "").split("_")[-1] for f in files],
        key=lambda x: int(x),
    )


def _sample(seq: List, k: float, rng: np.random.Generator) -> List:
    """从 seq 中无放回随机抽取 round(k) 个样本 (自动 clip 到 [0, len(seq)])."""
    k = int(round(k))
    k = max(0, min(k, len(seq)))
    if k == len(seq):
        return list(seq)
    idxs = rng.choice(len(seq), size=k, replace=False)
    return [seq[i] for i in idxs]


def _augment_group(packs: List[List], base: int, rem: int,
                   transform, rng: np.random.Generator) -> List[List]:
    """对一组原始变化样本做配额增强.

    每个样本输出 1 份原始 + (base 或 base+1) 份增强副本, 其中 rem 个样本随机多分 1 份,
    保证该组最终样本数精确等于 len(packs) + base*len(packs) + rem.
    """
    extra_counts = [base] * len(packs)
    if rem > 0:
        for j in rng.choice(len(packs), size=rem, replace=False):
            extra_counts[j] += 1

    out: List[List] = []
    for pack, n_extra in zip(packs, extra_counts):
        out.append(pack)
        if n_extra > 0:
            out.extend(apply_transform(pack, transform, n_extra))
    return out


def _feasible_range(n_raw_1015: int, n_raw_1520: int,
                    u_1015: int, u_1520: int, r_cu: float) -> Tuple[int, int]:
    """给定变化样本抽样规模与未变化池容量, 返回该档位下 N 的可行区间 [n_min, n_max].

    n_min 由 Q >= raw_changed 得到, n_max 由 unchanged_g <= u_g 得到. 仅用于报错提示.
    """
    raw_changed = n_raw_1015 + n_raw_1520
    scale = (1.0 + r_cu) / r_cu
    caps = []
    if n_raw_1015 > 0:
        caps.append(u_1015 * r_cu * raw_changed / n_raw_1015)
    if n_raw_1520 > 0:
        caps.append(u_1520 * r_cu * raw_changed / n_raw_1520)
    n_min = raw_changed * scale
    n_max = min(caps) * scale if caps else float("inf")
    return int(np.ceil(n_min)), int(np.floor(n_max))


def load_hybrid_data(root: str, shape: List[int], condition_list: List[str],
                     train_size: int = 2000,
                     ratio_1015: float = 1.0,
                     ratio_1520: float = 1.0,
                     changed_unchanged_ratio: float = 1.0,
                     normalize: bool = True,
                     seed: int = 0) -> Tuple[List[List], dict]:
    """构建混合训练集 (含配额数据增强).

    Parameters
    ----------
    root: str
        数据根目录 (如 "_data\\training_data"), 下含 train/val 两个 split.
    shape: List[int]
        数据形状 [1, H, W].
    condition_list: List[str]
        驱动因子列表, 如 ["DEM", "DL", "GDP", "RK"].
    train_size: int
        目标训练集样本量 N (增强后变化样本 + 未变化样本).
    ratio_1015: float
        10-15 变化样本加入比例 r1, 占 10-15 变化池 (train + val), 取值范围 [0, 1].
    ratio_1520: float
        15-20 变化样本加入比例 r2, 占 15-20 变化池 (train + val), 取值范围 [0, 1].
    changed_unchanged_ratio: float
        变化:非变化 目标比例 r_cu (1.0 表示 1:1), 必须 > 0.
    normalize: bool
        是否对驱动因子做 Z-score 归一化.
    seed: int
        随机种子, 保证抽样与增强可复现.

    Returns
    -------
    xf_pack_list: List[List]
        与 training_dataloader 一致, 每个元素为
        [srcFactors(F,H,W), srcLU_onehot(6,H,W), diff(1,H,W), idx_label(str), changed_flag(int)]
    stats: dict
        数据构成统计, 供训练 pipeline 归档到文档.
    """
    assert set(condition_list).issubset(set(ALL_FACTORS)), "condition_list value error."
    if not 0.0 <= ratio_1015 <= 1.0:
        raise ValueError(f"ratio_1015 必须落在 [0, 1], 当前为 {ratio_1015}.")
    if not 0.0 <= ratio_1520 <= 1.0:
        raise ValueError(f"ratio_1520 必须落在 [0, 1], 当前为 {ratio_1520}.")
    if changed_unchanged_ratio <= 0:
        raise ValueError(f"changed_unchanged_ratio 必须 > 0, 当前为 {changed_unchanged_ratio}.")
    if train_size < 1:
        raise ValueError(f"train_size 必须 >= 1, 当前为 {train_size}.")

    factor_num = len(condition_list) + 1  # +1 为末尾 LU 通道
    r_cu = changed_unchanged_ratio

    print("\n================= 开始加载混合训练数据集 ... =================")

    # 1. 加载数据: 10-15 / 15-20 两个来源均取 train + val 全部
    t10 = _load_year_data(root, "train", "10", condition_list, shape, normalize)
    t15 = _load_year_data(root, "train", "15", condition_list, shape, normalize)
    t20 = _load_year_data(root, "train", "20", condition_list, shape, normalize)
    v10 = _load_year_data(root, "val", "10", condition_list, shape, normalize)
    v15 = _load_year_data(root, "val", "15", condition_list, shape, normalize)
    v20 = _load_year_data(root, "val", "20", condition_list, shape, normalize)

    train_idx = _get_sample_indices(root, "train", "10", condition_list)
    val_idx = _get_sample_indices(root, "val", "10", condition_list)

    changed_1015, unchanged_1015 = [], []
    changed_1520, unchanged_1520 = [], []

    # 2. 构造 pack 并按来源/变化与否四分类
    def _build(src, tgt, s_year, t_year, tidx):
        return create_training_data_pack(
            src, tgt, s_year, t_year, tidx, factor_num, only_changed=False)

    def _classify(pack, changed_bucket, unchanged_bucket):
        (changed_bucket if pack[-1] == 1 else unchanged_bucket).append(pack)

    for i, tidx in enumerate(train_idx):
        _classify(_build(t10[i], t15[i], "10", "15", tidx), changed_1015, unchanged_1015)
        _classify(_build(t15[i], t20[i], "15", "20", tidx), changed_1520, unchanged_1520)

    for i, tidx in enumerate(val_idx):
        _classify(_build(v10[i], v15[i], "10", "15", tidx), changed_1015, unchanged_1015)
        _classify(_build(v15[i], v20[i], "15", "20", tidx), changed_1520, unchanged_1520)

    pool = {
        'a': len(changed_1015), 'u_a': len(unchanged_1015),
        'b': len(changed_1520), 'u_b': len(unchanged_1520),
    }

    rng = np.random.default_rng(seed)

    # 3. 按比例从两个变化池抽样
    raw_1015 = int(round(ratio_1015 * pool['a']))
    raw_1520 = int(round(ratio_1520 * pool['b']))
    raw_changed = raw_1015 + raw_1520
    if raw_changed == 0:
        raise ValueError(
            "原始变化样本数为 0: ratio_1015 与 ratio_1520 同时为 0 或两个变化池为空, 无法构造训练集.")

    sel_1015 = _sample(changed_1015, raw_1015, rng)
    sel_1520 = _sample(changed_1520, raw_1520, rng)

    # 4. 目标变化总数 -> 按来源切分 -> 组内配额分配
    target_changed = int(round(train_size * r_cu / (1.0 + r_cu)))
    n_min, n_max = _feasible_range(raw_1015, raw_1520, pool['u_a'], pool['u_b'], r_cu)
    if target_changed < raw_changed:
        raise ValueError(
            f"train_size={train_size} 过小 (或 ratio 过大): 目标变化总数 {target_changed} "
            f"小于原始变化样本数 {raw_changed}, 无法完成增强. "
            f"当前 r1={ratio_1015}, r2={ratio_1520} 下 N 的可行区间为 [{n_min}, {n_max}].")

    if raw_1015 == 0:
        n_changed_aug_1015 = 0
    else:
        n_changed_aug_1015 = int(round(target_changed * raw_1015 / raw_changed))
        # 两个来源的目标数都必须 >= 各自原始数, 否则配额分配会出现负的额外副本
        n_changed_aug_1015 = max(raw_1015, min(n_changed_aug_1015, target_changed - raw_1520))
    n_changed_aug_1520 = target_changed - n_changed_aug_1015

    extra_1015 = n_changed_aug_1015 - raw_1015
    base_1015, rem_1015 = 0, 0
    if raw_1015 > 0:
        base_1015, rem_1015 = divmod(extra_1015, raw_1015)

    extra_1520 = n_changed_aug_1520 - raw_1520
    base_1520, rem_1520 = 0, 0
    if raw_1520 > 0:
        base_1520, rem_1520 = divmod(extra_1520, raw_1520)

    # 5. 未变化样本: 数量与同来源增强后变化样本数对应, 从同来源的未变化池无放回抽
    n_unchanged_1015 = int(round(n_changed_aug_1015 / r_cu))
    n_unchanged_1520 = int(round(n_changed_aug_1520 / r_cu))
    if n_unchanged_1015 > pool['u_a']:
        raise ValueError(
            f"10-15 未变化样本不足: 需要 {n_unchanged_1015}, 池子只有 {pool['u_a']}. "
            f"当前 r1={ratio_1015}, r2={ratio_1520} 下 N 的可行区间为 [{n_min}, {n_max}].")
    if n_unchanged_1520 > pool['u_b']:
        raise ValueError(
            f"15-20 未变化样本不足: 需要 {n_unchanged_1520}, 池子只有 {pool['u_b']}. "
            f"当前 r1={ratio_1015}, r2={ratio_1520} 下 N 的可行区间为 [{n_min}, {n_max}].")

    # 6. 数据增强: 每个来源按各自的配额分配生成副本
    transform = transforms.Compose([
        transforms.Resize(shape[-2]),
        transforms.CenterCrop(shape[-2:]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    changed_aug = _augment_group(sel_1015, base_1015, rem_1015, transform, rng)
    changed_aug += _augment_group(sel_1520, base_1520, rem_1520, transform, rng)

    # 7. 未变化样本抽样 (不增强)
    unchanged_pack = _sample(unchanged_1015, n_unchanged_1015, rng)
    unchanged_pack += _sample(unchanged_1520, n_unchanged_1520, rng)

    xf_pack_list = unchanged_pack + changed_aug
    n_actual = len(xf_pack_list)
    deviation = (n_actual - train_size) / train_size

    # 8. 打印推导过程与构成统计
    print(f"Data: 池子容量 10-15(变化 {pool['a']}, 未变化 {pool['u_a']}), "
          f"15-20(变化 {pool['b']}, 未变化 {pool['u_b']})")
    print(f"Data: 抽样 r1={ratio_1015} -> 10-15 变化 {raw_1015}, "
          f"r2={ratio_1520} -> 15-20 变化 {raw_1520}, 原始变化合计 {raw_changed}")
    print(f"Data: 目标变化总数 {target_changed} (train_size={train_size}, r_cu={r_cu}), "
          f"切分 -> 10-15 {n_changed_aug_1015} / 15-20 {n_changed_aug_1520}")
    print(f"Data: 配额增强 10-15 每样本 +{base_1015} 份(另 {rem_1015} 个 +1), "
          f"15-20 每样本 +{base_1520} 份(另 {rem_1520} 个 +1)")
    print(f"Data: 未变化 10-15 {n_unchanged_1015} + 15-20 {n_unchanged_1520} "
          f"= {n_unchanged_1015 + n_unchanged_1520}")
    print(f"Data: 最终训练集共 {n_actual} 个样本 (未变化 "
          f"{n_unchanged_1015 + n_unchanged_1520} + 变化 {len(changed_aug)}), "
          f"目标 {train_size}, 偏差 {deviation:+.2%}")

    # 输出每个 pack 中各因子的意义和形状 (与 training_dataloader 保持一致)
    if len(xf_pack_list) > 0:
        pack_sample = xf_pack_list[0]
        factor_names = ["srcFactors", "srcLU(one-hot)", "diff", "idx_label", "changed_flag"]
        for i, (factor, name) in enumerate(zip(pack_sample, factor_names)):
            if isinstance(factor, np.ndarray) or hasattr(factor, "shape"):
                print("  {}: {}, shape={}".format(i, name, list(factor.shape)))
            else:
                print("  {}: {}, value={}".format(i, name, factor))

    # 9. 汇总数据构成统计, 供训练 pipeline 归档到文档
    stats = {
        'train_size': train_size,
        'ratio_1015': ratio_1015,
        'ratio_1520': ratio_1520,
        'changed_unchanged_ratio': r_cu,
        'seed': seed,
        'n_changed_1015_pool': pool['a'],
        'n_unchanged_1015_pool': pool['u_a'],
        'n_changed_1520_pool': pool['b'],
        'n_unchanged_1520_pool': pool['u_b'],
        'n_raw_1015': raw_1015,
        'n_raw_1520': raw_1520,
        'n_raw_changed': raw_changed,
        'target_changed': target_changed,
        'n_changed_aug_1015': n_changed_aug_1015,
        'n_changed_aug_1520': n_changed_aug_1520,
        'augment_base_1015': base_1015,
        'augment_rem_1015': rem_1015,
        'augment_base_1520': base_1520,
        'augment_rem_1520': rem_1520,
        'n_changed_aug': len(changed_aug),
        'n_unchanged_1015': n_unchanged_1015,
        'n_unchanged_1520': n_unchanged_1520,
        'n_unchanged': n_unchanged_1015 + n_unchanged_1520,
        'n_actual': n_actual,
        'deviation': round(deviation, 6),
        'n_feasible_min': n_min,
        'n_feasible_max': n_max,
        'total': n_actual,
    }
    return xf_pack_list, stats


class HybridDataset(data.Dataset):
    """混合训练数据集, 直接喂给 DataLoader.

    用法示例:
        from datasets.hybrid_dataloader import HybridDataset
        ds = HybridDataset(root="_data\\\\training_data", shape=[1,128,128],
                           condition_list=["DEM","DL","GDP","RK"],
                           train_size=2000, ratio_1015=1.0, ratio_1520=1.0,
                           changed_unchanged_ratio=1.0)
        loader = DataLoader(ds, batch_size=8, shuffle=True)
    """

    def __init__(self, root: str, shape: List[int], condition_list: List[str],
                 train_size: int = 2000,
                 ratio_1015: float = 1.0,
                 ratio_1520: float = 1.0,
                 changed_unchanged_ratio: float = 1.0,
                 normalize: bool = True,
                 seed: int = 0):
        self.xf_pack_list, self.stats = load_hybrid_data(
            root=root,
            shape=shape,
            condition_list=condition_list,
            train_size=train_size,
            ratio_1015=ratio_1015,
            ratio_1520=ratio_1520,
            changed_unchanged_ratio=changed_unchanged_ratio,
            normalize=normalize,
            seed=seed,
        )

    def __getitem__(self, i: int):
        return self.xf_pack_list[i]

    def __len__(self) -> int:
        return len(self.xf_pack_list)
