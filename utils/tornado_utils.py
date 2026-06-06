# tornado / sensitivity 实验特有工具: 参数目录与实验矩阵构造
from typing import Dict, Any, List, Optional


# Tornado 参数目录: (dotted_path, baseline, [perturbed_values], category)
# category: 'score_cfg' -> 覆写 score_cfg.yaml; 'opt_cfg' -> 覆写 optimizerConfig.yaml 的 SA/GA 段
_TORNADO_PARAMS = [
    ('tgt_function.lambda_ES_high', 1000, [500, 1500, 2000], 'score_cfg'),
    ('tgt_function.lambda_ES_low', 5000, [2500, 7500, 10000], 'score_cfg'),
    ('tgt_function.lambda_LS_high', 5, [2, 8, 10], 'score_cfg'),
    ('change_score.inside_reward_coef', 10, [5, 15, 20], 'score_cfg'),
    ('runner_utils.sigmoid_threshold', 0.5, [0.3, 0.7], 'score_cfg'),
    ('landscape.pure_base', 0.3, [0.1, 0.5], 'score_cfg'),
    ('T_start', 100, [50, 200], 'opt_cfg'),
    ('cooling_rate', 0.9995, [0.999, 0.9998], 'opt_cfg'),
]


def build_experiment_matrix(mode: str = 'tornado', n_repeats: int = 1,
                            top_k_params: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """构造 tornado / sensitivity 实验矩阵.

    Parameters:
    ----------
    mode : str
        'tornado' 参数粗筛; 'sensitivity' 一维线搜索
    n_repeats : int
        每个实验配置重复次数
    top_k_params : List[Dict], optional
        sensitivity 模式下传入的参数规格

    Returns:
    --------
    List[Dict[str, Any]]
        每项是 {name, overrides, opt_overrides, flags, n_repeats, group} 的实验配置列表
    """
    matrix: List[Dict[str, Any]] = []

    if mode == 'tornado':
        matrix.append({
            'name': 'A_baseline',
            'overrides': {},
            'opt_overrides': {},
            'flags': {'guided': True, 'disable_early_stop': False, 'skip_threshold_check': False},
            'n_repeats': n_repeats, 'group': 'A',
        })
        for dotted_path, baseline, values, category in _TORNADO_PARAMS:
            for v in values:
                exp_name = f"B_{dotted_path}={v}"
                overrides = {}
                opt_overrides = {}
                if category == 'score_cfg':
                    overrides[dotted_path] = v
                else:
                    opt_overrides[dotted_path] = v
                matrix.append({
                    'name': exp_name,
                    'overrides': overrides,
                    'opt_overrides': opt_overrides,
                    'flags': {'guided': True, 'disable_early_stop': False, 'skip_threshold_check': False},
                    'n_repeats': n_repeats, 'group': 'B',
                })
    elif mode == 'sensitivity':
        if top_k_params is None or len(top_k_params) == 0:
            raise ValueError("mode='sensitivity' 必须提供 top_k_params")
        for param_spec in top_k_params:
            dotted_path = param_spec['path']
            values = param_spec['values']
            category = param_spec['category']
            for v in values:
                exp_name = f"S_{dotted_path}={v}"
                overrides = {}
                opt_overrides = {}
                if category == 'score_cfg':
                    overrides[dotted_path] = v
                else:
                    opt_overrides[dotted_path] = v
                matrix.append({
                    'name': exp_name,
                    'overrides': overrides,
                    'opt_overrides': opt_overrides,
                    'flags': {'guided': True, 'disable_early_stop': False, 'skip_threshold_check': False},
                    'n_repeats': n_repeats, 'group': 'S',
                })
    else:
        raise ValueError(f"tornado_utils 不支持的 mode: {mode}")

    return matrix
