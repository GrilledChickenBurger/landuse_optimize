# tornado / sensitivity 实验特有工具: 参数目录与实验矩阵构造
from typing import Dict, Any, List, Optional


# Tornado 参数目录: (dotted_path, baseline, [perturbed_values], category)
# category: 'score_cfg' -> 覆写 score_cfg.yaml; 'opt_cfg' -> 覆写 optimizerConfig.yaml 的 SA/GA 段
#
# 设计原则（本次聚焦算法超参数）:
#   1. 排除 lambda_ES / lambda_LS / inside_reward_coef 等目标函数权重参数，
#      避免用 ES 提升评判 ES 权重导致的循环论证。
#   2. 加入 max_noimprove_iter — 它是 SA 实际的主导终止条件，对解质量影响极大。
#   3. cooling_rate 范围要能在 1000 步内产生显著差异：
#        0.995 -> T≈0.67,  0.998 -> T≈13.5,  0.9995 -> T≈60.6
#      原范围 [0.999, 0.9998] 在 1000 步后分别约为 36.8 / 82.2，差异不够明显。
_TORNADO_PARAMS = [
    # --- Round 1: 核心算法控制参数 ---
    # 实际主导终止条件，预期效应最大
    ('max_noimprove_iter', 1000, [500, 2000, 5000], 'opt_cfg'),

    # 冗余安全机制，测试是否与 noimprove 有交互；只测两极节省时间
    ('max_beststuck_iter', 3000, [1000, 6000], 'opt_cfg'),

    # 初始温度；旧数据 [50,200] 效应低，加入 500 拉大跨度
    ('T_start', 100, [50, 500], 'opt_cfg'),

    # 降温率；新范围 [0.995, 0.998, 0.9995] 在 1000 步内差异显著
    ('cooling_rate', 0.9995, [0.995, 0.998, 0.9995], 'opt_cfg'),

    # [预留] v2 minimal 新增参数，当前主流程使用原版 landscape score，未启用
    # ('landscape.pure_change', 0.2, [0.1, 0.3], 'score_cfg'),
    # ('landscape.neighbor_weight', 1.0, [0.5, 1.5], 'score_cfg'),
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
            'flags': {'guided': True, 'disable_early_stop': False},
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
                    'flags': {'guided': True, 'disable_early_stop': False},
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
                    'flags': {'guided': True, 'disable_early_stop': False},
                    'n_repeats': n_repeats, 'group': 'S',
                })
    else:
        raise ValueError(f"tornado_utils 不支持的 mode: {mode}")

    return matrix
