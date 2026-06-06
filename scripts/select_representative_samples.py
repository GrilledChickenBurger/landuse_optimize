# 扫描 optimize 样本, 按 n_pred 等频分位数分层抽样输出代表性样本.
# 使用方法: 在项目根目录直接 python scripts/select_representative_samples.py
import os
import sys
import json
import random
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.base_dataloader import get_dataset
from utils.tools import get_yaml, ensure_folder
import utils.runner_utils as rUtils
from pipelines.experiment import process_batch
from libs.const import DEFAULT_CONDITION_LIST


def scan_n_pred(
    idxs: Optional[List[int]] = None,
    opt_cfg_path: str = "configs/optimizerConfig.yaml",
    save_json: str = "_data/representative_samples.json",
    n_samples: int = 5,
    n_quantiles: int = 5,
    seed: int = 42,
    score_cfg_path: str = "configs/score_cfg.yaml",
) -> None:
    """扫描 optimize 样本, 按 n_pred 等频分位数分层抽样.

    Parameters
    ----------
    idxs : List[int], optional
        待扫描样本索引列表. 为 None 时扫描全部样本.
    opt_cfg_path : str
        optimizerConfig.yaml 路径
    save_json : str
        输出 JSON 路径
    n_samples : int
        总样本数 (默认 5)
    n_quantiles : int
        分档数 (默认 5)
    seed : int
        随机种子
    score_cfg_path : str
        score_cfg.yaml 路径
    """
    cfg = get_yaml(opt_cfg_path)
    with open(score_cfg_path, encoding="utf-8") as f:
        score_cfg = yaml.safe_load(f)
    min_n_pred = score_cfg['const']['pixel_num_threshold_ischange']
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device.use_gpu else "cpu")

    # 数据
    dataset = get_dataset(type="optimize", data_config=cfg.data)
    dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False)

    # ANN
    model_type = cfg.model.type
    type_config = getattr(cfg.model, model_type, None)
    factors_ch = None  # 等遇到第一个 batch 再算
    ann_model = None

    grid_num = score_cfg['runner_utils']['grid_num']
    idxs_set = set(idxs) if idxs else None
    records = []  # [{index, n_pred, n_lu_types}]
    for i, data_pack in enumerate(dataloader):
        index = data_pack[5][0]
        idx_num = int(index.split('_')[1])
        # 如果指定了 idxs, 只扫描这些样本
        if idxs_set is not None and idx_num not in idxs_set:
            continue

        data_dict = process_batch(data_pack, cfg)

        if ann_model is None:
            factors_ch = data_dict['factors'].shape[1]
            ann_model = rUtils.load_ANN_model(
                model_type, cfg.model.model_path,
                factors_ch + 6, 1, device, type_config)

        year = index.split('_')[0]
        pred_diff = rUtils.get_pred_diff(
            ann_model, data_dict['ori_plan'], data_dict['factors'], device,
            year=year, condition_list=DEFAULT_CONDITION_LIST,
            normalize=True, sigmoid_threshold=0.5)

        # 与 runner 一致: hard_restrict = restrict & (ori!=0) & pred_grid_restrict
        ori = data_dict['ori_plan']
        hard_restrict = data_dict['restrict'].bool()
        hard_restrict &= (ori != 0)
        hard_restrict &= rUtils.get_grid_restrict_area(pred_diff, grid_num=grid_num).bool()
        n_pred = int(torch.count_nonzero(torch.logical_and(hard_restrict, pred_diff)).item())

        # 地类多样性: ori_plan 中 hard_restrict & pred_diff 区域的不同 LU 类型数
        affected = ori[hard_restrict & pred_diff.bool()]
        n_lu_types = int(torch.unique(affected).numel()) if affected.numel() > 0 else 0

        records.append({'index': index, 'n_pred': n_pred, 'n_lu_types': n_lu_types})
        print(f"  scanned {index}  n_pred={n_pred}  n_lu_types={n_lu_types}")

    # 仅保留 n_pred >= min_n_pred 的真正变化样本（避免过小区域）
    changed = [r for r in records if r['n_pred'] >= min_n_pred]
    print(f"\n扫描样本 {len(records)}, 其中预测有变化(n_pred>={min_n_pred}) {len(changed)}")

    if len(changed) == 0:
        print("[warn] 没有符合条件的样本, 无法选择")
        return

    # 等频分位数分档
    changed.sort(key=lambda r: r['n_pred'])
    n_per_quantile = max(1, len(changed) // n_quantiles)
    strata_buckets = defaultdict(list)
    for i, r in enumerate(changed):
        q_idx = min(i // n_per_quantile, n_quantiles - 1)  # 最后一档包含余数
        strata_buckets[q_idx].append(r)

    # 统计每档范围
    quantile_ranges = {}
    for q_idx in range(n_quantiles):
        bucket = strata_buckets.get(q_idx, [])
        if bucket:
            n_preds = [r['n_pred'] for r in bucket]
            quantile_ranges[str(q_idx)] = {
                'min': min(n_preds),
                'max': max(n_preds),
                'count': len(bucket),
            }
            print(f"  [档位{q_idx}] n_pred ∈ [{min(n_preds)}, {max(n_preds)}]  候选 {len(bucket)} 个")

    # 从每档中按地类多样性选样本
    rng = random.Random(seed)
    selected = []
    for q_idx in range(n_quantiles):
        bucket = strata_buckets.get(q_idx, [])
        if not bucket:
            continue
        # 优先选地类多样性高的
        bucket.sort(key=lambda r: (-r['n_lu_types'], r['n_pred']))
        n_pick = max(1, n_samples // n_quantiles)  # 每档至少选1个
        picks = rng.sample(bucket, min(n_pick, len(bucket)))
        selected.extend(picks)

    # 如果选中数量不足, 从最大档位补齐
    if len(selected) < n_samples:
        max_q = max(strata_buckets.keys())
        existing_idx = {s['index'] for s in selected}
        extras = [r for r in strata_buckets[max_q] if r['index'] not in existing_idx]
        rng.shuffle(extras)
        selected.extend(extras[:max(0, n_samples - len(selected))])

    out = {
        'selected': [{'index': s['index'],
                      'idx_num': s['index'].split('_')[1],
                      'n_pred': s['n_pred'],
                      'n_lu_types': s['n_lu_types']} for s in selected],
        'idxs_for_pipeline': [int(s['index'].split('_')[1]) for s in selected],
        'quantile_ranges': quantile_ranges,
        'all_records': records,
    }
    ensure_folder(os.path.dirname(save_json))
    with open(save_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n选中 {len(selected)} 个样本:")
    for s in selected:
        print(f"  - {s['index']}  n_pred={s['n_pred']}  n_lu_types={s['n_lu_types']}")
    print(f"\nidxs_for_pipeline = {out['idxs_for_pipeline']}")
    print(f"输出: {save_json}")


def resample_from_json(
    json_path: str = "_data/representative_samples.json",
    save_json: Optional[str] = None,
    n_samples: int = 5,
    n_quantiles: int = 5,
    seed: int = 42,
    score_cfg_path: str = "configs/score_cfg.yaml",
) -> None:
    """从已有的 JSON 文件读取 all_records, 重新分档和抽样 (跳过数据加载和预测).

    Parameters
    ----------
    json_path : str
        已有的 representative_samples.json 路径
    save_json : str, optional
        输出路径, 为 None 时覆盖原文件
    n_samples : int
        总样本数 (默认 5)
    n_quantiles : int
        分档数 (默认 5)
    seed : int
        随机种子
    score_cfg_path : str
        score_cfg.yaml 路径 (用于读取 min_n_pred)
    """
    with open(score_cfg_path, encoding="utf-8") as f:
        score_cfg = yaml.safe_load(f)
    min_n_pred = score_cfg['const']['pixel_num_threshold_ischange']

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = data['all_records']
    print(f"从 {json_path} 读取 {len(records)} 条记录")

    # 仅保留 n_pred >= min_n_pred 的真正变化样本
    changed = [r for r in records if r['n_pred'] >= min_n_pred]
    print(f"其中预测有变化(n_pred>={min_n_pred}) {len(changed)} 条")

    if len(changed) == 0:
        print("[warn] 没有符合条件的样本, 无法选择")
        return

    # 等频分位数分档
    changed.sort(key=lambda r: r['n_pred'])
    n_per_quantile = max(1, len(changed) // n_quantiles)
    strata_buckets = defaultdict(list)
    for i, r in enumerate(changed):
        q_idx = min(i // n_per_quantile, n_quantiles - 1)
        strata_buckets[q_idx].append(r)

    # 统计每档范围
    quantile_ranges = {}
    for q_idx in range(n_quantiles):
        bucket = strata_buckets.get(q_idx, [])
        if bucket:
            n_preds = [r['n_pred'] for r in bucket]
            quantile_ranges[str(q_idx)] = {
                'min': min(n_preds),
                'max': max(n_preds),
                'count': len(bucket),
            }
            print(f"  [档位{q_idx}] n_pred ∈ [{min(n_preds)}, {max(n_preds)}]  候选 {len(bucket)} 个")

    # 从每档中按地类多样性选样本
    rng = random.Random(seed)
    selected = []
    for q_idx in range(n_quantiles):
        bucket = strata_buckets.get(q_idx, [])
        if not bucket:
            continue
        bucket.sort(key=lambda r: (-r['n_lu_types'], r['n_pred']))
        n_pick = max(1, n_samples // n_quantiles)
        picks = rng.sample(bucket, min(n_pick, len(bucket)))
        selected.extend(picks)

    # 如果选中数量不足, 从最大档位补齐
    if len(selected) < n_samples:
        max_q = max(strata_buckets.keys())
        existing_idx = {s['index'] for s in selected}
        extras = [r for r in strata_buckets[max_q] if r['index'] not in existing_idx]
        rng.shuffle(extras)
        selected.extend(extras[:max(0, n_samples - len(selected))])

    out = {
        'selected': [{'index': s['index'],
                      'idx_num': s['index'].split('_')[1],
                      'n_pred': s['n_pred'],
                      'n_lu_types': s['n_lu_types']} for s in selected],
        'idxs_for_pipeline': [int(s['index'].split('_')[1]) for s in selected],
        'quantile_ranges': quantile_ranges,
        'all_records': records,
    }
    save_path = save_json or json_path
    ensure_folder(os.path.dirname(save_path))
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n选中 {len(selected)} 个样本:")
    for s in selected:
        print(f"  - {s['index']}  n_pred={s['n_pred']}  n_lu_types={s['n_lu_types']}")
    print(f"\nidxs_for_pipeline = {out['idxs_for_pipeline']}")
    print(f"输出: {save_path}")


def load_idxs_from_file(filepath: str) -> List[int]:
    """从 txt 文件读取 idx 列表 (每行一个数字)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [int(line.strip()) for line in f if line.strip()]


if __name__ == '__main__':
    save_json = "_data/representative_samples.json"
    n_samples = 10  # 可调整样本数量
    n_quantiles = 5
    seed = 42

    # 如果已有 JSON 文件, 直接读取重新抽样 (跳过数据加载和预测)
    if os.path.exists(save_json):
        print(f"[快速模式] 从已有 {save_json} 重新抽样")
        resample_from_json(
            json_path=save_json,
            n_samples=n_samples,
            n_quantiles=n_quantiles,
            seed=seed,
        )
    else:
        # 从 changed_regions_indices.txt 读取 idxs
        idxs_file = "changed_regions_indices.txt"
        idxs = load_idxs_from_file(idxs_file) if os.path.exists(idxs_file) else None
        print(f"从 {idxs_file} 读取 {len(idxs) if idxs else '全部'} 个样本")

        scan_n_pred(
            idxs=idxs,
            opt_cfg_path="configs/optimizerConfig.yaml",
            save_json=save_json,
            n_samples=n_samples,
            n_quantiles=n_quantiles,
            seed=seed,
        )
