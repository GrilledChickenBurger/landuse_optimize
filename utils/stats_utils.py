# 统计推断工具: Wilcoxon 符号秩检验 + Holm-Bonferroni 校正 + Cohen's d
from typing import Any, Dict, List, Optional, Tuple, Sequence
import numpy as np
from scipy import stats


def cohens_d(baseline: Sequence[float],
             perturbed: Sequence[float],
             correction: bool = True) -> Optional[float]:
    """计算 Cohen's d (配对版本).

    使用配对样本的标准差作为分母:
        d = (mu_perturbed - mu_baseline) / sigma_diff
    其中 sigma_diff 是差值 (perturbed - baseline) 的标准差.

    Parameters
    ----------
    baseline, perturbed : sequence of float
        配对样本值, 长度必须相同.
    correction : bool
        是否使用 Hedge's g 小样本校正: g = d * (1 - 3/(4n - 1)).
        当 n_repeats 较小时 (如 3-5) 推荐开启.

    Returns
    -------
    float | None
        效应量; 输入不合法时返回 None.
    """
    b = np.asarray(baseline, dtype=float)
    p = np.asarray(perturbed, dtype=float)
    if len(b) != len(p) or len(b) == 0:
        return None
    diffs = p - b
    n = len(diffs)
    if n < 2:
        return None
    mu_diff = float(np.mean(diffs))
    sigma_diff = float(np.std(diffs, ddof=1))
    if sigma_diff < 1e-12:
        return 0.0 if abs(mu_diff) < 1e-12 else float('inf')
    d = mu_diff / sigma_diff
    if correction:
        # Hedge's g 校正因子
        d *= (1.0 - 3.0 / (4.0 * n - 1.0))
    return float(d)


def wilcoxon_signed_rank(baseline: Sequence[float],
                         perturbed: Sequence[float],
                         alternative: str = 'two-sided') -> Tuple[Optional[float], Optional[float]]:
    """配对 Wilcoxon 符号秩检验.

    Parameters
    ----------
    baseline, perturbed : sequence of float
        配对样本值, 长度必须相同.
    alternative : str
        'two-sided' (默认) | 'greater' | 'less'.

    Returns
    -------
    (statistic, pvalue) : tuple[float | None, float | None]
        Wilcoxon 统计量和 p 值; 输入不合法时返回 (None, None).
    """
    b = np.asarray(baseline, dtype=float)
    p = np.asarray(perturbed, dtype=float)
    if len(b) != len(p) or len(b) == 0:
        return None, None
    # 过滤非有限值
    mask = np.isfinite(b) & np.isfinite(p)
    b, p = b[mask], p[mask]
    if len(b) < 2:
        return None, None
    try:
        res = stats.wilcoxon(p, b, alternative=alternative, zero_method='zsplit')
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return None, None


def holm_bonferroni(p_values: Sequence[Optional[float]],
                    alpha: float = 0.05) -> List[Optional[float]]:
    """Holm-Bonferroni 逐步多重比较校正.

    比标准 Bonferroni 功效更高, 同时控制族错误率 FWER.

    Parameters
    ----------
    p_values : sequence of float | None
        原始 p 值列表, None 表示缺失.
    alpha : float
        显著性水平, 默认 0.05.

    Returns
    -------
    List[Optional[float]]
        校正后的 p 值, 缺失项保持 None.
    """
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if p is not None and np.isfinite(float(p))]
    if not indexed:
        return [None for _ in p_values]

    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    corrected = {}
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        # Holm: p_corrected = max(prev, p * (m - rank + 1))
        adj = min(p * (m - rank + 1), 1.0)
        adj = max(adj, prev_adj)
        corrected[orig_idx] = adj
        prev_adj = adj

    return [corrected.get(i, None) for i in range(len(p_values))]


def format_stars(p_value: Optional[float]) -> str:
    """将校正后 p 值格式化为显著性星号.

    *  p < 0.05
    ** p < 0.01
    *** p < 0.001
    n.s. 不显著 / 缺失
    """
    if p_value is None or not np.isfinite(float(p_value)):
        return 'n.s.'
    p = float(p_value)
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def interpret_cohens_d(d: Optional[float]) -> str:
    """Cohen's d 效应量解释.

    |d| < 0.2 : negligible
    0.2 <= |d| < 0.5 : small
    0.5 <= |d| < 0.8 : medium
    |d| >= 0.8 : large
    """
    if d is None or not np.isfinite(float(d)):
        return 'unknown'
    ad = abs(float(d))
    if ad < 0.2:
        return 'negligible'
    if ad < 0.5:
        return 'small'
    if ad < 0.8:
        return 'medium'
    return 'large'


def summarize_knob_significance(
    baseline_by_index: Dict[str, float],
    perturbed_by_index: Dict[str, Dict[str, Dict[str, float]]],
    alpha: float = 0.05,
    alternative: str = 'two-sided',
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """对 tornado 的一组配对样本做统计推断汇总.

    Parameters
    ----------
    baseline_by_index : dict[str, float]
        {index: baseline_sample_mean}.
    perturbed_by_index : dict[str, dict[str, dict[str, float]]]
        {knob: {value_str: {index: perturbed_sample_mean}}}.
    alpha : float
        显著性水平.
    alternative : str
        Wilcoxon 备择假设方向.

    Returns
    -------
    dict[str, dict[str, dict]]
        {knob: {value_str: {cohens_d, wilcoxon_stat, raw_p, corrected_p, stars, effect_size}}}.
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # 1) 先收集所有 (knob, value_str) 组合的 raw p 值, 用于一次性 Holm 校正
    raw_results: List[Tuple[str, str, Optional[float], Optional[float], Optional[float]]] = []
    for knob, vmap in perturbed_by_index.items():
        for val_str, idx_map in vmap.items():
            common_idxs = sorted(set(baseline_by_index.keys()) & set(idx_map.keys()))
            if len(common_idxs) < 2:
                raw_results.append((knob, val_str, None, None, None))
                continue
            base_vals = [baseline_by_index[i] for i in common_idxs]
            pert_vals = [idx_map[i] for i in common_idxs]
            d = cohens_d(base_vals, pert_vals)
            stat, p = wilcoxon_signed_rank(base_vals, pert_vals, alternative=alternative)
            raw_results.append((knob, val_str, d, stat, p))

    # 2) Holm-Bonferroni 校正 (基于所有有效 p 值)
    p_values = [r[4] for r in raw_results]
    corrected = holm_bonferroni(p_values, alpha=alpha)

    # 3) 组装输出
    for (knob, val_str, d, stat, p), cp in zip(raw_results, corrected):
        out.setdefault(knob, {})[val_str] = {
            'cohens_d': d,
            'effect_size': interpret_cohens_d(d),
            'wilcoxon_statistic': stat,
            'p_value_raw': p,
            'p_value_corrected': cp,
            'stars': format_stars(cp),
        }
    return out
