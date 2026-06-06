# ablation pipeline: 多次重复稳定性实验 + A/B/C 三组消融实验
import os
import sys
from typing import Dict, Any, List, Optional

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.experiment_base import BaseSweepPipeline
from utils.ablation_utils import make_random_connected_mask
from utils.runner_utils import load_ANN_model, get_pred_diff
from utils.io_utils import ensure_folder
from utils.exp_sweep_utils import write_stability_runs_csv, write_ablation_runs_csv
from libs.const import DEFAULT_CONDITION_LIST

# 代表性样本元数据 JSON, 供 stability_runs.csv 反查 n_pred / level
REPRESENTATIVE_SAMPLES_JSON = "_data/representative_samples.json"


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
        raise ValueError(f"ablation.py 不支持的 mode: {mode}")

    return matrix


class AblationPipeline(BaseSweepPipeline):
    """消融实验流水线 (stability / ablation 模式)."""

    result_category = 'ablation'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 缓存每个样本的 ANN pred_diff，避免 C 组每 run 重复加载模型
        self._ann_cache: Dict[str, torch.Tensor] = {}

    def build_matrix(self) -> List[Dict[str, Any]]:
        return build_experiment_matrix(self.mode, n_repeats=self.n_repeats)

    def apply_type_config_hook(self, type_config, exp: Dict[str, Any]) -> None:
        """ablation 模式应用极端收敛设置."""
        if self.mode == 'ablation' and self.runner_config.type == 'SA':
            type_config.T_end = 1e-6
        if self.mode == 'ablation' and self.runner_config.type != 'SA':
            type_config.max_gen = 8000

    def get_random_mask(
        self,
        data_dict: Dict[str, torch.Tensor],
        index: str,
        exp: Dict[str, Any],
        r: int,
    ) -> Optional[torch.Tensor]:
        """C 组生成随机连通块 mask；ANN 模型在样本级别只加载一次."""
        group = exp.get('group', 'A')
        if group != 'C':
            return None

        redline = data_dict['restrict']
        lu_nonzero = (data_dict['ori_plan'] != 0)
        base_hard = torch.logical_and(redline, lu_nonzero)

        # 样本级别缓存 ANN pred_diff
        if index not in self._ann_cache:
            year = index.split('_')[0] if '_' in index else ''
            in_ch = data_dict['factors'].shape[1] + 6
            ann_model = load_ANN_model(
                self.opt_cfg.model.type, self.opt_cfg.model.model_path,
                in_ch, 1, self.device,
                getattr(self.opt_cfg.model, self.opt_cfg.model.type, None))
            pred_diff = get_pred_diff(
                ann_model, data_dict['ori_plan'], data_dict['factors'], self.device,
                year=year, condition_list=DEFAULT_CONDITION_LIST,
                normalize=True,
                sigmoid_threshold=self.base_score_cfg.runner_utils.sigmoid_threshold)
            self._ann_cache[index] = pred_diff
        else:
            pred_diff = self._ann_cache[index]

        base_seed = (abs(hash(index)) & 0xFFFFFFFF)
        random_mask = make_random_connected_mask(pred_diff, base_hard, seed=base_seed + r)

        # 保存 mask 供后续分析
        save_path = os.path.join(self.root_save, exp['name'], f'run_{r}', index)
        ensure_folder(save_path)
        np.save(os.path.join(save_path, 'random_mask.npy'),
                random_mask.detach().cpu().numpy().astype(np.uint8))
        return random_mask


def ablation_pipeline(opt_cfg_path: str,
                      score_cfg_path: str,
                      idxs: Optional[List[int]] = None,
                      mode: str = 'ablation',
                      n_repeats: int = 1,
                      is_processed: bool = False,
                      processed_timestamp: Optional[str] = None) -> str:
    """消融实验主线流水线 (stability / ablation 模式).

    两种模式跑完后都会自动产出 <root>/plots/<mode>_runs.csv，
    字段与论文原始记录表一致: sample_id, n_pred, level, algorithm, config,
    run_idx, F_final, ESV, Runtime, T95.

    Returns:
    --------
    str
        结果根目录路径
    """
    pipeline = AblationPipeline(
        opt_cfg_path=opt_cfg_path,
        score_cfg_path=score_cfg_path,
        idxs=idxs,
        mode=mode,
        n_repeats=n_repeats,
        is_processed=is_processed,
        processed_timestamp=processed_timestamp,
    )
    root_save = pipeline.run()

    # stability / ablation 模式: 落盘扁平 runs.csv (供后处理 + 论文表使用)
    try:
        if mode == 'stability':
            write_stability_runs_csv(root_save, REPRESENTATIVE_SAMPLES_JSON)
        elif mode == 'ablation':
            write_ablation_runs_csv(root_save, REPRESENTATIVE_SAMPLES_JSON)
    except Exception as e:
        print(f"  [{mode}_runs] 写出失败 (不影响主流程): {e}")

    return root_save
