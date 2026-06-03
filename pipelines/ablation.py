# ablation pipeline: 多次重复稳定性实验 + A/B/C 三组消融实验
import os
import sys
import json
import copy
import argparse
from argparse import Namespace
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner import create_runner
from datasets.base_dataloader import get_dataset
from utils.calculator_utils import train_maxent
from utils.tools import ensure_folder, get_yaml, get_namespace, load_score_cfg, get_default_score_cfg
from libs.const import apply_const_overrides, DEFAULT_CONDITION_LIST
from pipelines.experiment import process_batch


# ------------------------------------------------------------
# 1. 实验矩阵构造
# ------------------------------------------------------------
def build_experiment_matrix(mode: str = 'ablation', n_repeats: int = 1) -> List[Dict[str, Any]]:
    """构造当前消融实验矩阵.

    Parameters:
    ----------
    mode : str
        'stability' 只跑 A_guided × n_repeats 次; 'ablation' 跑 A/B/C 三组 × n_repeats 次
    n_repeats : int
        每个实验配置重复次数

    Returns:
    --------
    List[Dict[str, Any]]
        每项是 {name, overrides, flags, n_repeats, group} 的实验配置列表
    """
    if mode in ('A', 'B', 'all'):
        raise ValueError("旧单次消融/tornado mode 已移除；请使用 mode='stability' 或 mode='ablation'")

    matrix: List[Dict[str, Any]] = []

    if mode == 'stability':
        matrix.append({
            'name': 'A_guided',
            'overrides': {},
            'flags': {'guided': True, 'disable_early_stop': True},
            'n_repeats': n_repeats, 'group': 'A',
        })
    elif mode == 'ablation':
        matrix.append({
            'name': 'A_guided',
            'overrides': {},
            'flags': {'guided': True, 'disable_early_stop': True, 'skip_threshold_check': True},
            'n_repeats': n_repeats, 'group': 'A',
        })
        matrix.append({
            'name': 'B_unguided',
            'overrides': {'ablation.disable_change_score': True},
            'flags': {'guided': False, 'disable_early_stop': True, 'skip_threshold_check': True},
            'n_repeats': n_repeats, 'group': 'B',
        })
        matrix.append({
            'name': 'C_random',
            'overrides': {},
            'flags': {'guided': False, 'disable_early_stop': True, 'skip_threshold_check': True},
            'n_repeats': n_repeats, 'group': 'C',
        })
    else:
        raise ValueError(f"不支持的消融实验 mode: {mode}")

    return matrix


# ------------------------------------------------------------
# 2. 工具: 在 Namespace 上按点号路径写值
# ------------------------------------------------------------
def _set_by_path(ns: Namespace, dotted_path: str, value: Any) -> None:
    """在 Namespace 中按 'a.b.c' 路径写入 value (in-place).
    若中间某段是 dict 形态的 Namespace (如 water_supply_param), 自动适配 int 键.
    """
    parts = dotted_path.split('.')
    cur = ns
    for p in parts[:-1]:
        if hasattr(cur, p):
            cur = getattr(cur, p)
        else:
            # water_supply_param 的 int 键在 yaml 加载后会变为 str 属性名
            cur = getattr(cur, str(p))
    last = parts[-1]
    if hasattr(cur, last):
        setattr(cur, last, value)
    else:
        setattr(cur, str(last), value)


def _deep_copy_cfg(cfg: Namespace) -> Namespace:
    return copy.deepcopy(cfg)


# ------------------------------------------------------------
# 3. 主流水线
# ------------------------------------------------------------
def ablation_pipeline(opt_cfg_path: str,
                      score_cfg_path: str,
                      idxs: Optional[List[int]] = None,
                      mode: str = 'ablation',
                      n_repeats: int = 1,
                      is_processed: bool = False,
                      processed_timestamp: Optional[str] = None) -> None:
    """消融实验主线流水线.

    Parameters:
    ----------
    opt_cfg_path : str
        optimizerConfig.yaml 路径 (复用其 data/model/runner 段)
    score_cfg_path : str
        score_cfg.yaml 路径
    idxs : List[int], optional
        待优化样本索引列表
    mode : str
        'stability' 只跑 A_guided; 'ablation' 跑 A/B/C 三组
    n_repeats : int
        每个实验配置重复次数
    is_processed : bool
        是否断点续传
    processed_timestamp : str
        断点续传时间戳, results/ablation/<ts>/

    Returns:
    --------
    None
    """
    if is_processed and not processed_timestamp:
        raise ValueError("断点续传必须提供 processed_timestamp")

    print(f"================= 开始 Ablation 流水线 [mode={mode}] =================")

    # 1. 加载配置
    opt_cfg = get_yaml(opt_cfg_path)
    base_score_cfg = load_score_cfg(score_cfg_path)
    print(f"Optimizer config: {opt_cfg_path}")
    print(f"Score config:     {score_cfg_path}")

    # 2. 设备
    device = torch.device("cuda" if torch.cuda.is_available() and opt_cfg.device.use_gpu else "cpu")
    print(f"Using device: {device}")

    # 3. 数据
    dataset = get_dataset(type="optimize", data_config=opt_cfg.data)
    dataloader = DataLoader(dataset, batch_size=opt_cfg.data.batch_size, shuffle=opt_cfg.data.shuffle)

    # 4. MaxEnt 模型
    maxent_path = opt_cfg.data.maxent_model_path.format(year=opt_cfg.data.year)
    train_maxent(opt_cfg.data.root, opt_cfg.data.year, maxent_path)

    # 5. runner 通用配置
    runner_config = opt_cfg.runner
    use_landscape = runner_config.use_landscape
    type_config_base = getattr(opt_cfg, runner_config.type)
    scenario = runner_config.scenario if runner_config.scenario != 'all' else 'default'
    weight_dict = vars(getattr(runner_config.weight, scenario))

    # 6. 时间戳与根目录
    run_ts = processed_timestamp if is_processed else datetime.now().strftime("%Y%m%d_%H%M%S")
    root_save = os.path.join('results', 'ablation', run_ts)
    ensure_folder(root_save)
    print(f"  结果根目录: {root_save}")
    print(f"  scenario: {scenario}, weight: {weight_dict}")

    # 7. 构造矩阵
    matrix = build_experiment_matrix(mode, n_repeats=n_repeats)
    print(f"  实验数量: {len(matrix)} (mode={mode}, n_repeats={n_repeats})")

    completed_exps = set(os.listdir(root_save)) if is_processed else set()

    # 8. 遍历实验矩阵
    for exp in matrix:
        exp_name = exp['name']
        if is_processed and exp_name in completed_exps:
            print(f"\n实验【{exp_name}】已完成, 跳过")
            continue

        print(f"\n================= 实验【{exp_name}】=================")
        print(f"  overrides: {exp['overrides']}")
        print(f"  flags:     {exp['flags']}")

        # 8.1 构造本实验的 score_cfg (深拷贝 + 覆写)
        score_cfg = _deep_copy_cfg(base_score_cfg)
        for dotted_path, v in exp['overrides'].items():
            _set_by_path(score_cfg, dotted_path, v)
        # 同步覆写 libs.const 的模块级常量
        apply_const_overrides(score_cfg)

        n_rep = exp.get('n_repeats', 1)
        group = exp.get('group', 'A')

        type_config = copy.deepcopy(type_config_base)
        if mode == 'ablation' and runner_config.type == 'SA':
            type_config.T_end = 1e-6  # ablation 需要完整收敛，降低终止温度
        if mode == 'ablation' and runner_config.type != 'SA':
            type_config.max_gen = 8000  # ablation GA 需要更多代数观察完整收敛

        exp_save_folder = os.path.join(root_save, exp_name)
        ensure_folder(exp_save_folder)
        # 新结构: run_r/{index}/; 老结构: {index}/run_r/ — 统一兼容扫描
        processed_samples = set()
        if is_processed:
            for entry in os.listdir(exp_save_folder):
                entry_path = os.path.join(exp_save_folder, entry)
                if not os.path.isdir(entry_path):
                    continue
                if entry.startswith('run_'):
                    # 新结构: run_r/ 下直接是样本 index
                    processed_samples.update(d for d in os.listdir(entry_path)
                                             if os.path.isdir(os.path.join(entry_path, d)))
                else:
                    # 老结构: entry 本身就是样本 index
                    processed_samples.add(entry)

        # 8.3 遍历样本
        for i, data_pack in enumerate(dataloader):
            index = data_pack[5][0]
            if idxs and len(idxs) > 0 and index.split('_')[1] not in [str(x) for x in idxs]:
                continue

            print(f"  -> 样本 {index} ({i+1}/{len(dataloader)}), n_repeats={n_rep}, group={group}")
            data_dict = process_batch(data_pack, opt_cfg)

            # C 组每个样本一个 base_seed (跨 run 用 base_seed+r)
            base_seed = (abs(hash(index)) & 0xFFFFFFFF) if group == 'C' else 0

            for r in range(n_rep):
                # 目录结构: A_guided/run_r/{index}/
                # 传给 runner 的 save_path 不含 index, runner 内部会 join 一次 index 形成最终目录
                save_path = os.path.join(exp_save_folder, f'run_{r}')
                sample_save = os.path.join(save_path, index)
                sample_processed_key = f'run_{r}/{index}'

                if is_processed and os.path.exists(os.path.join(sample_save, 'convergence.json')):
                    print(f"     run {r} 已完成, 跳过")
                    continue

                ensure_folder(sample_save)

                # C 组: 生成随机连通块 mask
                random_mask = None
                if group == 'C':
                    from utils.ablation_plots import make_random_connected_mask
                    # 需要先构造一个 base_hard_restrict (耕地红线 + LU≠0) 供 BFS 选种子
                    redline = data_dict['restrict']
                    lu_zero = (data_dict['ori_plan'] != 0)
                    base_hard = torch.logical_and(redline, lu_zero)
                    # pred_diff 在 runner 内部生成, 这里临时跑一次 ANN 拿 pred_diff
                    # 为复用, 直接用 runner_utils 的接口
                    import utils.runner_utils as _rUtils
                    year = index.split('_')[0]
                    ann_model = _rUtils.load_ANN_model(
                        opt_cfg.model.type, opt_cfg.model.model_path,
                        data_dict['factors'].shape[1] + 6, 1, device,
                        getattr(opt_cfg.model, opt_cfg.model.type, None))
                    pred_diff_c = _rUtils.get_pred_diff(
                        ann_model, data_dict['ori_plan'], data_dict['factors'], device,
                        year=year, condition_list=DEFAULT_CONDITION_LIST,
                        normalize=True,
                        sigmoid_threshold=score_cfg.runner_utils.sigmoid_threshold)
                    random_mask = make_random_connected_mask(pred_diff_c, base_hard, seed=base_seed + r)
                    np.save(os.path.join(sample_save, 'random_mask.npy'),
                            random_mask.detach().cpu().numpy().astype(np.uint8))

                runner = create_runner(
                    runner_type=runner_config.type,
                    config=type_config,
                    common_config=runner_config.common_params,
                    model_config=opt_cfg.model,
                    device=device,
                    data_dict=data_dict,
                    index=index,
                    save_path=save_path,
                    weight_dict=weight_dict,
                    use_landscape=use_landscape,
                    guided=exp['flags'].get('guided', True),
                    disable_early_stop=exp['flags'].get('disable_early_stop', False),
                    skip_threshold_check=exp['flags'].get('skip_threshold_check', False),
                    score_cfg=score_cfg,
                    random_restrict_mask=random_mask,
                )
                runner.run()

        # 8.4 实验级 manifest, 方便 tornado 后处理读取
        manifest = {
            'name': exp_name,
            'overrides': exp['overrides'],
            'flags': exp['flags'],
            'runner_type': runner_config.type,
            'scenario': scenario,
            'weight_dict': weight_dict,
        }
        with open(os.path.join(exp_save_folder, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 9. 全局清单
    summary = {
        'timestamp': run_ts,
        'mode': mode,
        'runner_type': runner_config.type,
        'idxs': idxs,
        'experiments': [e['name'] for e in matrix],
    }
    with open(os.path.join(root_save, 'ablation_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n================= Ablation 流水线结束 =================")
    print(f"  结果根目录: {root_save}")
