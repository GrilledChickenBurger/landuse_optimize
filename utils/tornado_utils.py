# tornado / sensitivity 实验特有工具: 参数目录与实验矩阵构造
from typing import Dict, Any, List, Optional


# Tornado 参数目录: (dotted_path, baseline, [perturbed_values], category)
# category: 'score_cfg' -> 覆写 score_cfg.yaml; 'opt_cfg' -> 覆写 optimizerConfig.yaml 的 SA/GA 段

# ===================== Round 1: SA 核心算法控制参数 (已完成 10 样本) =====================
# _TORNADO_PARAMS_R1 = [
#     # 实际主导终止条件，预期效应最大
#     ('max_noimprove_iter', 1000, [500, 2000, 5000], 'opt_cfg'),
#     # 冗余安全机制，测试是否与 noimprove 有交互；只测两极节省时间
#     ('max_beststuck_iter', 3000, [1000, 6000], 'opt_cfg'),
#     # 初始温度；旧数据 [50,200] 效应低，加入 500 拉大跨度
#     ('T_start', 100, [50, 500], 'opt_cfg'),
#     # 降温率；新范围 [0.995, 0.998, 0.9995] 在 1000 步内差异显著
#     ('cooling_rate', 0.9995, [0.995, 0.998, 0.9995], 'opt_cfg'),
# ]

# ===================== Round 1: 通用参数初筛 (已完成 3 样本) =====================
# _TORNADO_PARAMS_R1_GENERAL = [
#     # lambda_ES: 1500 提升 ES (d=+0.99), 2000 反降; 2500 全面恶化, 7500 IoU/F1 大幅下降
#     ('tgt_function.lambda_ES_high', 1000, [500, 1500, 2000], 'score_cfg'),
#     ('tgt_function.lambda_ES_low', 5000, [2500, 7500, 10000], 'score_cfg'),
#     # lambda_LS: 2~10 范围效应全是 negligible/small，被 λ_ES=1000 完全压制
#     ('tgt_function.lambda_LS_high', 5, [2, 8, 10], 'score_cfg'),
#     # pure_base: 0.5 时 IoU/F1 大幅下降 (d≈-0.99)
#     ('landscape.pure_base', 0.3, [0.1, 0.5], 'score_cfg'),
#     # inside_reward_coef: IoU/F1 中等效应，方向单调
#     ('change_score.inside_reward_coef', 10, [5, 15, 20], 'score_cfg'),
# ]

# ===================== Round 2: 通用参数精筛 + 扩范围 (10 样本) =====================

_TORNADO_PARAMS = [
    ('cooling_rate', 0.9995, [0.9998], 'opt_cfg'),

    # --- 目标函数权重参数 (保留 Round 1 有趋势的) ---
    # lambda_ES_high: 1500 提升 ES (d=+0.99), 2000 反降 → 非单调, 值得验证
    ('tgt_function.lambda_ES_high', 1000, [500, 1500, 2000], 'score_cfg'),
    # lambda_ES_low: 2500/7500 导致 IoU/F1 大幅下降 (d≈-0.9~-1.0)
    ('tgt_function.lambda_ES_low', 5000, [2500, 7500, 10000], 'score_cfg'),
    # lambda_LS_high: Round 1 范围 [2,8,10] 效应太弱 (landscape 被 λ_ES 压制)
    #   扩大到 [20,50,100], 使 λ_LS×landscape 占目标函数 10%~20%
    ('tgt_function.lambda_LS_high', 5, [20, 50, 100], 'score_cfg'),
    # lambda_LS_low: 新参数, 对 n_pred 小的样本 (44, 98) 插值到 low 值
    ('tgt_function.lambda_LS_low', 1, [2, 5, 10], 'score_cfg'),

    # --- 变化位置得分参数 (保留 Round 1 有趋势的) ---
    # inside_reward_coef: IoU/F1 中等效应, 方向单调 (值越大 IoU 越低)
    ('change_score.inside_reward_coef', 10, [5, 15, 20], 'score_cfg'),

    # --- 景观破碎化参数 ---
    # pure_base: 0.5 时 IoU/F1 大幅下降 (d≈-0.99), 0.1 时中等下降
    ('landscape.pure_base', 0.3, [0.1, 0.5], 'score_cfg'),
    # pure_change: 新参数, 与 pure_base 配合: weight = pure_base + change_ratio × pure_change
    ('landscape.pure_change', 0.2, [0.05, 0.1, 0.4], 'score_cfg'),

    # --- 搜索空间约束参数 ---
    # grid_num: pred_diff 网格划分数, 2→4格(搜索范围大), 8→64格(搜索范围小)
    ('runner_utils.grid_num', 4, [2, 8], 'score_cfg'),
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
