# 实验结果扫描、指标聚合与表格生成 (专用于 ablation / tornado / sensitivity 等参数扫描实验)
import os
import json
import copy
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy import stats

from utils.io_utils import ensure_folder

# 元组: (目录名, 显示标签, 短标签, 颜色, 线型)
# 目录名与磁盘目录一致（未重命名），显示标签反映新 ABC 语义
ABLATION_GROUPS = [
    ('B_unguided', 'A: 无空间约束',   'A_none',   'tab:orange', '--'),
    ('C_random',   'B: 随机区域约束', 'B_random', 'tab:green',  ':'),
    ('A_guided',   'C: 完整方法',     'C_full',   'tab:blue',   '-'),
]

# 从 ABLATION_GROUPS 派生的映射，供表格函数统一使用
_DIR_TO_SHORT = {g[0]: g[2] for g in ABLATION_GROUPS}
_DIR_TO_LABEL = {g[0]: g[1] for g in ABLATION_GROUPS}
# baseline 目录名（完整方法，作为对比基准）
_BASELINE_DIR = 'A_guided'
# 对照目录名列表（与 baseline 对比的两组）
_COMPARISON_DIRS = ['B_unguided', 'C_random']

# ------------------------------------------------------------
# 目录扫描（兼容新老结构）
# ------------------------------------------------------------
def _list_samples(exp_dir: str) -> List[str]:
    """返回该实验目录下所有出现过的样本 index (去重 + 排序).

    扫描 <exp_dir>/run_*/<index>/. 兼容老结构 <exp_dir>/<index>/run_*/.
    """
    if not os.path.isdir(exp_dir):
        return []
    samples = set()
    for entry in os.listdir(exp_dir):
        sub = os.path.join(exp_dir, entry)
        if not os.path.isdir(sub):
            continue
        if entry.startswith('run_'):
            for idx in os.listdir(sub):
                if os.path.isdir(os.path.join(sub, idx)):
                    samples.add(idx)
        else:
            samples.add(entry)
    return sorted(samples)


def _list_run_dirs_for_sample(exp_dir: str, index: str) -> List[str]:
    """返回某样本所有 run 的目录绝对路径列表 (按 run_r 序号升序).

    新结构: [<exp_dir>/run_0/<index>, ...]
    兼容老结构: [<exp_dir>/<index>/run_0, ...] 或 [<exp_dir>/<index>] (n_rep=1)
    """
    if not os.path.isdir(exp_dir):
        return []
    runs_new = []
    for entry in sorted(os.listdir(exp_dir)):
        if not entry.startswith('run_'):
            continue
        sample_dir = os.path.join(exp_dir, entry, index)
        if os.path.isdir(sample_dir):
            runs_new.append(sample_dir)
    if runs_new:
        return runs_new
    old_root = os.path.join(exp_dir, index)
    if not os.path.isdir(old_root):
        return []
    runs_old = sorted([d for d in os.listdir(old_root)
                       if d.startswith('run_') and os.path.isdir(os.path.join(old_root, d))])
    if runs_old:
        return [os.path.join(old_root, d) for d in runs_old]
    if os.path.exists(os.path.join(old_root, 'convergence.json')):
        return [old_root]
    return []


# ------------------------------------------------------------
# 指标读取
# ------------------------------------------------------------
def _read_perf_metric(perf_path: str, metric_path: List[str]) -> Optional[float]:
    """从 performance.json 中按嵌套 key 路径读数值."""
    if not os.path.exists(perf_path):
        return None
    try:
        with open(perf_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k in metric_path:
            data = data[k]
        return float(data)
    except (KeyError, ValueError, TypeError):
        return None


def _read_conv_metric(conv_path: str, key: str) -> Optional[float]:
    if not os.path.exists(conv_path):
        return None
    with open(conv_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    v = data.get(key)
    return float(v) if v is not None else None


def _read_nested(d: dict, dotted: str) -> Optional[float]:
    cur = d
    for k in dotted.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------
# 收敛曲线后处理工具
# ------------------------------------------------------------
def compute_best_so_far(series: np.ndarray, minimize: bool = True) -> np.ndarray:
    """将当前解序列转换为 best-so-far 序列.

    SA 为最小化问题, GA 为最大化问题 (fitness = -tgt_func).
    """
    series = np.asarray(series, dtype=float)
    if minimize:
        return np.minimum.accumulate(series)
    return np.maximum.accumulate(series)


def compute_normalized_improvement(
    series: np.ndarray,
    f_init: Optional[float] = None,
    f_final: Optional[float] = None,
    minimize: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """计算归一化改善百分比序列.

    公式: I(t) = (f_init - f_best(t)) / (f_init - f_final) * 100%

    Parameters
    ----------
    series : np.ndarray
        原始 total_score 序列 (当前解).
    f_init : float, optional
        初始解目标函数值. 若未提供, 用 best-so-far 第一个值近似
        (要求后续实验在 convergence.json 中显式记录 initial_total_score).
    f_final : float, optional
        最终 best-so-far 值. 若未提供, 取 series 的 best-so-far 终点.
    minimize : bool
        True 为最小化问题 (SA), False 为最大化问题 (GA).

    Returns
    -------
    improvement : np.ndarray
        归一化改善百分比序列, 已 clip 到 [0, 100].
    f_init_used : float
        实际使用的初始值.
    f_final_used : float
        实际使用的最终值.
    """
    best = compute_best_so_far(series, minimize=minimize)
    if f_final is None:
        f_final = float(best[-1]) if len(best) else 0.0
    if f_init is None:
        f_init = float(best[0]) if len(best) else f_final

    if minimize:
        denom = f_init - f_final
    else:
        denom = f_final - f_init

    if abs(denom) < 1e-12:
        return np.zeros_like(best), f_init, f_final

    if minimize:
        imp = (f_init - best) / denom * 100.0
    else:
        imp = (best - f_init) / denom * 100.0
    return np.clip(imp, 0.0, 100.0), f_init, f_final


def compute_aocc(improvement: np.ndarray, max_pct: float = 100.0) -> float:
    """计算 Area Over Convergence Curve (越小代表 anytime 性能越好).

    对归一化改善百分比曲线, AOCC = mean(max_pct - improvement).
    """
    if len(improvement) == 0:
        return float('nan')
    return float(np.mean(max_pct - np.asarray(improvement, dtype=float)))


def compute_fixed_budget(
    improvement: np.ndarray,
    budgets: List[int],
) -> Dict[int, float]:
    """提取固定预算点的归一化改善百分比."""
    improvement = np.asarray(improvement, dtype=float)
    result: Dict[int, float] = {}
    for b in budgets:
        if b < len(improvement):
            result[b] = float(improvement[b])
        elif len(improvement) > 0:
            result[b] = float(improvement[-1])
        else:
            result[b] = 0.0
    return result


def compute_t95_from_improvement(
    improvement: np.ndarray,
    threshold: float = 95.0,
) -> int:
    """从归一化改善百分比序列计算首次达到 threshold 的迭代索引."""
    improvement = np.asarray(improvement, dtype=float)
    if len(improvement) == 0:
        return -1
    hits = improvement >= threshold
    if not hits.any():
        return -1
    return int(np.argmax(hits))


# ------------------------------------------------------------
# 指标聚合
# ------------------------------------------------------------
def _aggregate_exp(exp_dir: str, metric_extractor) -> Optional[float]:
    """对一个实验文件夹下所有样本的指标做平均.

    metric_extractor(run_dir) -> float|None.
    多 run 时先在样本内取均值, 再跨样本均值.
    """
    if not os.path.isdir(exp_dir):
        return None
    sample_means = []
    for idx in _list_samples(exp_dir):
        run_dirs = _list_run_dirs_for_sample(exp_dir, idx)
        vals = [v for v in (metric_extractor(rd) for rd in run_dirs) if v is not None]
        if vals:
            sample_means.append(float(np.mean(vals)))
    if not sample_means:
        return None
    return float(np.mean(sample_means))


def _make_extractor(metric: str):
    """根据 metric 名称返回提取器函数 extractor(run_dir) -> float|None."""
    # convergence.json 指标
    if metric in ('final_best', 'wall_time_sec', 'iter_to_95pct', 'es_score',
                  'total_iter', 'iter_to_95pct_best', 'aocc',
                  'fixed_budget_1000', 'fixed_budget_5000'):
        return lambda d, m=metric: _read_conv_metric(
            os.path.join(d, 'convergence.json'), m)
    else:
        path = metric.split('.')
        return lambda d: _read_perf_metric(
            os.path.join(d, 'performance.json'), path)


_METRIC_KEYS_CONV = ['final_best', 'iter_to_95pct', 'wall_time_sec', 'total_iter', 'es_score',
                     'iter_to_95pct_best', 'aocc', 'fixed_budget_1000', 'fixed_budget_5000']
_METRIC_KEYS_PERF = [
    'ES.food.best_ori_diff', 'ES.carbon.best_ori_diff',
    'ES.water.best_ori_diff', 'ES.culture.best_ori_diff',
    'ES.food.best_tgt_diff', 'ES.carbon.best_tgt_diff',
    'ES.water.best_tgt_diff', 'ES.culture.best_tgt_diff',
    'Change_area.real_metrics.iou', 'Change_area.real_metrics.f1_score',
    'Change_area.pred_metrics.iou', 'Change_area.pred_metrics.f1_score',
    'FINAL.FOM', 'FINAL.kappa',
    # 景观格局指标：用于 ablation A/B/C 景观差异分析
    'Landscape.land.best_plan.CONTAG', 'Landscape.land.best_plan.SHEI',
]


def aggregate_runs(exp_dir: str, index: str) -> Dict[str, Dict[str, Any]]:
    """从 <exp_dir>/run_*/<index>/{convergence.json, performance.json} 聚合所有 run 的指标.
    返回 {metric: {mean, std, ci_95, runs}}.
    """
    run_dirs = _list_run_dirs_for_sample(exp_dir, index)
    runs_records: List[Dict[str, float]] = []
    for rd in run_dirs:
        rec: Dict[str, float] = {}
        conv_p = os.path.join(rd, 'convergence.json')
        if os.path.exists(conv_p):
            with open(conv_p, 'r', encoding='utf-8') as f:
                conv = json.load(f)
            for k in _METRIC_KEYS_CONV:
                v = conv.get(k)
                if v is not None:
                    rec[k] = float(v)
            if 'iter_to_95pct_repeat' in conv:
                rec['iter_to_95pct_repeat'] = float(conv['iter_to_95pct_repeat'])
        perf_p = os.path.join(rd, 'performance.json')
        if os.path.exists(perf_p):
            with open(perf_p, 'r', encoding='utf-8') as f:
                perf = json.load(f)
            for mk in _METRIC_KEYS_PERF:
                v = _read_nested(perf, mk)
                if v is not None:
                    rec[mk] = v
        runs_records.append(rec)

    out: Dict[str, Dict[str, Any]] = {}
    if not runs_records:
        return out
    all_metrics = set()
    for r in runs_records:
        all_metrics.update(r.keys())
    for m in all_metrics:
        vals = [r[m] for r in runs_records if m in r]
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        if len(vals) >= 2:
            std = float(np.std(vals, ddof=1))
            # 修复: 使用与样本量匹配的 t 临界值，而非硬编码 df=4
            t_crit = float(stats.t.ppf(0.975, df=len(vals) - 1))
            ci_half = t_crit * std / np.sqrt(len(vals))
        else:
            std = 0.0
            ci_half = 0.0
        out[m] = {
            'mean': mean, 'std': std,
            'ci_95': (mean - ci_half, mean + ci_half),
            'runs': vals, 'n': len(vals),
        }
    return out


def recompute_iter_to_95pct(exp_dir: str, index: str) -> None:
    """用多次 final_best 均值作为参考最优, 重算每次 run 的 iter_to_95pct_repeat.

    同时基于 best-so-far 序列计算:
      - iter_to_95pct_best: 首次达到 95% 改善的迭代 (基于 best-so-far, 更稳定)
      - aocc: Area Over Convergence Curve
      - fixed_budget_{T}: 固定预算 T 处的归一化改善百分比
      - improvement_npy: 归一化改善百分比序列文件路径
    """
    run_dirs = _list_run_dirs_for_sample(exp_dir, index)
    if len(run_dirs) < 1:
        return
    finals = []
    for rd in run_dirs:
        conv_p = os.path.join(rd, 'convergence.json')
        if os.path.exists(conv_p):
            with open(conv_p, 'r', encoding='utf-8') as f:
                conv = json.load(f)
            v = conv.get('final_best')
            if v is not None:
                finals.append(float(v))
    if not finals:
        return
    ref_best = float(np.mean(finals))

    experiment_root = os.path.dirname(exp_dir)
    summary_path = os.path.join(experiment_root, 'experiment_summary.json')
    runner_type = 'SA'
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        runner_type = summary.get('runner_type', 'SA').upper()

    minimize = runner_type == 'SA'

    if minimize:
        target = ref_best * 1.05 if ref_best >= 0 else ref_best * 0.95
    else:
        ref_best_fitness = -ref_best
        target = ref_best_fitness * 0.95 if ref_best_fitness >= 0 else ref_best_fitness * 1.05

    fixed_budgets = [1000, 5000]

    for rd in run_dirs:
        npy_p = os.path.join(rd, 'total_score_list.npy')
        conv_p = os.path.join(rd, 'convergence.json')
        if not (os.path.exists(npy_p) and os.path.exists(conv_p)):
            continue
        arr = np.load(npy_p)

        with open(conv_p, 'r', encoding='utf-8') as f:
            conv = json.load(f)

        # 兼容: 若已有 initial_total_score 则用其作为 f_init, 否则用 best-so-far 首值近似
        f_init = conv.get('initial_total_score')
        f_final = conv.get('final_best')
        improvement, f_init_used, f_final_used = compute_normalized_improvement(
            arr, f_init=f_init, f_final=f_final, minimize=minimize)

        # 保存归一化改善百分比序列 (后处理绘图用)
        imp_path = os.path.join(rd, 'improvement.npy')
        np.save(imp_path, improvement)

        # 基于 best-so-far 的 T95
        iter_best = compute_t95_from_improvement(improvement, threshold=95.0)

        # AOCC
        aocc = compute_aocc(improvement)

        # 固定预算
        fb = compute_fixed_budget(improvement, fixed_budgets)

        # 旧逻辑: 基于当前解序列的 iter_to_95pct_repeat (保留以兼容旧表)
        if minimize:
            hit = (arr <= target).any()
            iter_repeat = int(np.argmax(arr <= target)) if hit else -1
        else:
            hit = (arr >= target).any()
            iter_repeat = int(np.argmax(arr >= target)) if hit else -1

        conv['iter_to_95pct_repeat'] = iter_repeat
        conv['ref_best_repeat'] = ref_best
        conv['iter_to_95pct_best'] = iter_best
        conv['aocc'] = aocc
        for b in fixed_budgets:
            conv[f'fixed_budget_{b}'] = fb[b]
        conv['f_init_used'] = f_init_used
        conv['f_final_used'] = f_final_used
        conv['improvement_npy'] = os.path.basename(imp_path)

        with open(conv_p, 'w', encoding='utf-8') as f:
            json.dump(conv, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------
# 表格生成
# ------------------------------------------------------------
def _resolve_group_dir(experiment_root: str, group_name: str) -> str:
    return os.path.join(experiment_root, group_name)


def build_stability_table(experiment_root: str,
                          exp_name: str,
                          save_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """跨样本聚合单个实验组的统计指标, 输出 csv."""
    exp_dir = _resolve_group_dir(experiment_root, exp_name)
    if not os.path.isdir(exp_dir):
        print(f"  [stability_table] 实验 {exp_name} 不存在, 跳过")
        return []
    rows = []
    for idx in _list_samples(exp_dir):
        recompute_iter_to_95pct(exp_dir, idx)
        agg = aggregate_runs(exp_dir, idx)
        if not agg:
            continue
        row = {'sample': idx, 'n_runs': agg.get('final_best', {}).get('n', 0)}
        for m, v in agg.items():
            row[f'{m}_mean'] = v['mean']
            row[f'{m}_std'] = v['std']
            row[f'{m}_ci95_low'] = v['ci_95'][0]
            row[f'{m}_ci95_high'] = v['ci_95'][1]
        rows.append(row)

    save_path = save_path or os.path.join(experiment_root, 'plots', f'stability_table_{exp_name}.csv')
    ensure_folder(os.path.dirname(save_path))
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"  stability_table saved: {save_path}")
    return rows


def _load_ablation_data(experiment_root: str):
    """统一扫描 A/B/C 三组数据，返回 (per_group, samples).

    供多个 build_ablation_*_table 函数复用，避免对同一目录重复扫描。
    """
    groups = [g[0] for g in ABLATION_GROUPS]
    per_group: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    samples = set()
    for g in groups:
        gd = _resolve_group_dir(experiment_root, g)
        if not os.path.isdir(gd):
            continue
        per_group[g] = {}
        for idx in _list_samples(gd):
            recompute_iter_to_95pct(gd, idx)
            per_group[g][idx] = aggregate_runs(gd, idx)
            samples.add(idx)
    return per_group, samples


def build_ablation_compare_table(experiment_root: str,
                                 save_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """三组并排对比, 每个样本一行 (各组 mean±std 与相对 baseline 的 % 变化)."""
    per_group, samples = _load_ablation_data(experiment_root)
    groups = [g[0] for g in ABLATION_GROUPS]

    rows = []
    metrics = ['final_best', 'iter_to_95pct_repeat', 'iter_to_95pct_best', 'aocc',
               'fixed_budget_1000', 'fixed_budget_5000', 'wall_time_sec', 'es_score',
               'Landscape.land.best_plan.CONTAG', 'Landscape.land.best_plan.SHEI']
    for idx in sorted(samples):
        row = {'sample': idx}
        for g in groups:
            agg = per_group.get(g, {}).get(idx, {})
            short = _DIR_TO_SHORT.get(g, g)
            for m in metrics:
                v = agg.get(m, {})
                row[f'{short}.{m}.mean'] = v.get('mean', '')
                row[f'{short}.{m}.std'] = v.get('std', '')
        a = per_group.get(_BASELINE_DIR, {}).get(idx, {})
        for g in _COMPARISON_DIRS:
            other = per_group.get(g, {}).get(idx, {})
            for m in metrics:
                am = a.get(m, {}).get('mean')
                om = other.get(m, {}).get('mean')
                if am and am != 0 and om is not None:
                    row[f'{_DIR_TO_SHORT[g]}.{m}.rel_to_baseline_pct'] = (om - am) / abs(am) * 100
        rows.append(row)

    save_path = save_path or os.path.join(experiment_root, 'plots', 'ablation_compare_table.csv')
    ensure_folder(os.path.dirname(save_path))
    if rows:
        keys = ['sample']
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"  ablation_compare_table saved: {save_path}")
    return rows


def build_ablation_mean_table(experiment_root: str,
                              save_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """每组按样本取均值（三组配置）, 对应 ablation.md 表2 (样本-配置均值)."""
    per_group, samples = _load_ablation_data(experiment_root)
    groups = [g[0] for g in ABLATION_GROUPS]

    rows = []
    metrics = ['final_best', 'iter_to_95pct', 'wall_time_sec', 'es_score',
               'Landscape.land.best_plan.CONTAG', 'Landscape.land.best_plan.SHEI']
    metric_map = {'final_best': 'Mean_F_S', 'iter_to_95pct': 'Mean_T95',
                  'wall_time_sec': 'Mean_Runtime', 'es_score': 'Mean_ESV',
                  'Landscape.land.best_plan.CONTAG': 'Mean_CONTAG',
                  'Landscape.land.best_plan.SHEI': 'Mean_SHEI'}
    for idx in sorted(samples):
        for g in groups:
            agg = per_group.get(g, {}).get(idx, {})
            row = {'Sample_ID': idx, 'Config': _DIR_TO_SHORT.get(g, g)}
            for m in metrics:
                v = agg.get(m, {})
                row[metric_map[m]] = v.get('mean', '')
            rows.append(row)

    save_path = save_path or os.path.join(experiment_root, 'plots', 'ablation_mean_table.csv')
    ensure_folder(os.path.dirname(save_path))
    if rows:
        keys = ['Sample_ID', 'Config', 'Mean_T95', 'Mean_Runtime', 'Mean_ESV', 'Mean_F_S',
                'Mean_CONTAG', 'Mean_SHEI']
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"  ablation_mean_table saved: {save_path}")
    return rows


def build_ablation_improvement_table(experiment_root: str,
                                     save_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """计算每个样本 baseline vs 对照组的提升率, 对应 ablation.md 表3 (样本提升率)."""
    per_group, samples = _load_ablation_data(experiment_root)

    rows = []
    metrics = ['iter_to_95pct', 'iter_to_95pct_best', 'aocc',
               'fixed_budget_1000', 'fixed_budget_5000', 'wall_time_sec', 'es_score',
               'Landscape.land.best_plan.CONTAG', 'Landscape.land.best_plan.SHEI']
    metric_label = {
        'iter_to_95pct': 'T95',
        'iter_to_95pct_best': 'T95_best',
        'aocc': 'AOCC',
        'fixed_budget_1000': 'FB1000',
        'fixed_budget_5000': 'FB5000',
        'wall_time_sec': 'Runtime',
        'es_score': 'ESV',
        'Landscape.land.best_plan.CONTAG': 'CONTAG',
        'Landscape.land.best_plan.SHEI': 'SHEI',
    }
    # 越大越好的指标集合（CONTAG 越大表示景观越聚合；SHEI 越小越好）
    larger_is_better = {'es_score', 'fixed_budget_1000', 'fixed_budget_5000',
                        'Landscape.land.best_plan.CONTAG'}
    for idx in sorted(samples):
        a = per_group.get(_BASELINE_DIR, {}).get(idx, {})
        row = {'Sample_ID': idx}
        for g in _COMPARISON_DIRS:
            other = per_group.get(g, {}).get(idx, {})
            comp_short = f'{_DIR_TO_SHORT[_BASELINE_DIR]}vs{_DIR_TO_SHORT[g]}'
            for m in metrics:
                am = a.get(m, {}).get('mean')
                om = other.get(m, {}).get('mean')
                label = metric_label[m]
                if am is not None and om is not None and om != 0:
                    if m in larger_is_better:
                        # 越大越好: (baseline - 对照) / |对照| * 100
                        row[f'{comp_short}_{label}_Imp_pct'] = (am - om) / abs(om) * 100
                    else:
                        # 越小越好: (对照 - baseline) / |对照| * 100
                        row[f'{comp_short}_{label}_Red_pct'] = (om - am) / abs(om) * 100
                else:
                    suffix = 'Imp_pct' if m in larger_is_better else 'Red_pct'
                    row[f'{comp_short}_{label}_{suffix}'] = ''
        rows.append(row)

    save_path = save_path or os.path.join(experiment_root, 'plots', 'ablation_improvement_table.csv')
    ensure_folder(os.path.dirname(save_path))
    if rows:
        # 动态构造列名：Sample_ID + 两组对比 × 各指标
        keys = ['Sample_ID']
        for g in _COMPARISON_DIRS:
            comp_short = f'{_DIR_TO_SHORT[_BASELINE_DIR]}vs{_DIR_TO_SHORT[g]}'
            for m in metrics:
                label = metric_label[m]
                suffix = 'Imp_pct' if m in larger_is_better else 'Red_pct'
                keys.append(f'{comp_short}_{label}_{suffix}')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"  ablation_improvement_table saved: {save_path}")
    return rows


def build_ablation_summary_table(experiment_root: str,
                                 save_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """跨样本汇总提升率的均值与范围, 对应 ablation.md 最终汇总表."""
    per_group, samples = _load_ablation_data(experiment_root)

    # 读算法类型
    runner_type = 'SA'
    summary_path = os.path.join(experiment_root, 'experiment_summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                runner_type = json.load(f).get('runner_type', 'SA').upper()
        except Exception:
            pass

    metrics = ['iter_to_95pct', 'iter_to_95pct_best', 'aocc',
               'fixed_budget_1000', 'fixed_budget_5000', 'wall_time_sec', 'es_score',
               'Landscape.land.best_plan.CONTAG', 'Landscape.land.best_plan.SHEI']
    metric_label = {
        'iter_to_95pct': 'T95 Reduction',
        'iter_to_95pct_best': 'T95(best) Reduction',
        'aocc': 'AOCC Reduction',
        'fixed_budget_1000': 'Fixed Budget T=1000 Improvement',
        'fixed_budget_5000': 'Fixed Budget T=5000 Improvement',
        'wall_time_sec': 'Runtime Reduction',
        'es_score': 'ESV Improvement',
        'Landscape.land.best_plan.CONTAG': 'CONTAG Improvement',
        'Landscape.land.best_plan.SHEI': 'SHEI Reduction',
    }
    # 越大越好的指标集合（CONTAG 越大表示景观越聚合；SHEI 越小越好）
    larger_is_better = {'es_score', 'fixed_budget_1000', 'fixed_budget_5000',
                        'Landscape.land.best_plan.CONTAG'}
    rows = []
    for g in _COMPARISON_DIRS:
        comp = f'{_DIR_TO_LABEL[_BASELINE_DIR]} vs {_DIR_TO_LABEL[g]}'
        for m in metrics:
            vals = []
            for idx in sorted(samples):
                a = per_group.get(_BASELINE_DIR, {}).get(idx, {})
                other = per_group.get(g, {}).get(idx, {})
                am = a.get(m, {}).get('mean')
                om = other.get(m, {}).get('mean')
                if am is not None and om is not None and om != 0:
                    if m in larger_is_better:
                        # 越大越好: (baseline - 对照) / |对照| * 100
                        vals.append((am - om) / abs(om) * 100)
                    else:
                        # 越小越好: (对照 - baseline) / |对照| * 100
                        vals.append((om - am) / abs(om) * 100)
            if vals:
                rows.append({
                    'Algorithm': runner_type,
                    'Comparison': comp,
                    'Metric': metric_label[m],
                    'Mean': round(float(np.mean(vals)), 2),
                    'Median': round(float(np.median(vals)), 2),
                    'Range_min': round(float(np.min(vals)), 2),
                    'Range_max': round(float(np.max(vals)), 2),
                })

    save_path = save_path or os.path.join(experiment_root, 'plots', 'ablation_summary_table.csv')
    ensure_folder(os.path.dirname(save_path))
    if rows:
        keys = ['Algorithm', 'Comparison', 'Metric', 'Mean', 'Median', 'Range_min', 'Range_max']
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"  ablation_summary_table saved: {save_path}")
    return rows


# ------------------------------------------------------------
# ExperimentResultStore: 封装目录扫描与数据读取，供绘图层调用
# ------------------------------------------------------------
class ExperimentResultStore:
    """实验结果存储访问对象，封装目录扫描和数据读取逻辑.

    绘图 pipeline 只通过本类获取已结构化的数据，不再直接操作文件系统.
    """

    def __init__(self, experiment_root: str):
        self._experiment_root = experiment_root

    @property
    def experiment_root(self) -> str:
        return self._experiment_root

    @property
    def ablation_root(self) -> str:
        """向后兼容别名, 等同于 experiment_root."""
        return self._experiment_root

    # --- 目录扫描 ---
    def list_experiments(self, prefix: str = '') -> List[str]:
        """返回指定前缀的实验目录名列表."""
        if not os.path.isdir(self._experiment_root):
            return []
        names = [n for n in os.listdir(self._experiment_root)
                 if os.path.isdir(os.path.join(self._experiment_root, n))]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return sorted(names)

    def list_samples(self, exp_name: str) -> List[str]:
        """返回某实验下的所有样本 index."""
        return _list_samples(os.path.join(self._experiment_root, exp_name))

    def list_run_dirs(self, exp_name: str, index: str) -> List[str]:
        """返回某实验某样本的所有 run 目录绝对路径."""
        return _list_run_dirs_for_sample(os.path.join(self._experiment_root, exp_name), index)

    # --- 数据读取 ---
    def get_convergence_series(
        self, exp_name: str, index: str, smooth_window: int = 1,
    ) -> Optional[np.ndarray]:
        """返回某样本所有 run 的 total_score 序列（已对齐、可选平滑）.

        Returns
        -------
        np.ndarray | None
            shape=(n_runs, min_len) 的数组；无数据时返回 None.
        """
        run_dirs = self.list_run_dirs(exp_name, index)
        series = []
        for rd in run_dirs:
            p = os.path.join(rd, 'total_score_list.npy')
            if os.path.exists(p):
                series.append(np.load(p))
        if not series:
            return None
        min_len = min(len(s) for s in series)
        aligned = np.stack([s[:min_len] for s in series], axis=0)
        if smooth_window > 1 and min_len > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.stack([
                np.convolve(s, kernel, mode='valid') for s in aligned
            ], axis=0)
            return smoothed
        return aligned

    def get_improvement_series(
        self, exp_name: str, index: str, smooth_window: int = 1,
    ) -> Optional[np.ndarray]:
        """返回某样本所有 run 的归一化改善百分比序列.

        优先读取已保存的 improvement.npy; 不存在时现场从 total_score_list.npy 计算.
        Returns shape=(n_runs, min_len), 值域 [0, 100].
        """
        run_dirs = self.list_run_dirs(exp_name, index)
        series = []
        for rd in run_dirs:
            imp_p = os.path.join(rd, 'improvement.npy')
            if os.path.exists(imp_p):
                series.append(np.load(imp_p))
                continue
            p = os.path.join(rd, 'total_score_list.npy')
            if os.path.exists(p):
                arr = np.load(p)
                # 推断最小化/最大化: 读 experiment_summary.json
                experiment_root = self._experiment_root
                summary_path = os.path.join(experiment_root, 'experiment_summary.json')
                runner_type = 'SA'
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            runner_type = json.load(f).get('runner_type', 'SA').upper()
                    except Exception:
                        pass
                imp, _, _ = compute_normalized_improvement(
                    arr, minimize=(runner_type == 'SA'))
                series.append(imp)
        if not series:
            return None
        min_len = min(len(s) for s in series)
        aligned = np.stack([s[:min_len] for s in series], axis=0)
        if smooth_window > 1 and min_len > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.stack([
                np.convolve(s, kernel, mode='valid') for s in aligned
            ], axis=0)
            return smoothed
        return aligned

    def get_stability_series(
        self, exp_name: str, smooth_window: int = 1,
    ) -> Dict[str, np.ndarray]:
        """返回某实验下所有样本的收敛序列 {index: series_array}.

        仅返回 run 数 >= 2 的样本.
        """
        result: Dict[str, np.ndarray] = {}
        for idx in self.list_samples(exp_name):
            s = self.get_convergence_series(exp_name, idx, smooth_window=smooth_window)
            if s is not None and s.shape[0] >= 2:
                result[idx] = s
        return result

    def get_ablation_comparison_data(
        self,
        groups: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """A/B/C 三组对比数据: {group: {index: aggregate_runs_result}}."""
        if groups is None:
            groups = [g[0] for g in ABLATION_GROUPS]
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for g in groups:
            gd = os.path.join(self._experiment_root, g)
            if not os.path.isdir(gd):
                continue
            result[g] = {}
            for idx in _list_samples(gd):
                recompute_iter_to_95pct(gd, idx)
                result[g][idx] = aggregate_runs(gd, idx)
        return result

    def get_final_best_distribution(
        self, groups: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, List[float]]]:
        """箱线图数据: {sample: {group_label: [final_best_values]}}."""
        if groups is None:
            groups = [(g[0], g[2]) for g in ABLATION_GROUPS]
        else:
            groups = [(g, g) for g in groups]
        samples: set = set()
        for g, _ in groups:
            gd = os.path.join(self._experiment_root, g)
            if os.path.isdir(gd):
                samples.update(_list_samples(gd))

        sub: Dict[str, Dict[str, List[float]]] = {}
        for s in sorted(samples):
            sub[s] = {}
            for g, label in groups:
                exp_dir = os.path.join(self._experiment_root, g)
                if not os.path.isdir(exp_dir):
                    continue
                agg = aggregate_runs(exp_dir, s)
                vals = agg.get('final_best', {}).get('runs', [])
                if vals:
                    sub[s][label] = vals
        return sub

    # --- tornado / sensitivity ---
    def get_baseline_metric(self, baseline_name: str, metric: str) -> Optional[float]:
        """返回 baseline 的聚合指标值."""
        baseline_dir = os.path.join(self._experiment_root, baseline_name)
        extractor = _make_extractor(metric)
        return _aggregate_exp(baseline_dir, extractor)

    def get_knob_deviations(
        self, baseline_name: str = 'A_baseline',
        prefix: str = 'B_', metric: str = 'final_best',
    ) -> Dict[str, Dict[str, float]]:
        """Tornado 分析: 返回 {knob: {value_str: normalized_deviation}}.

        若 baseline 无效则返回空字典.
        """
        baseline_val = self.get_baseline_metric(baseline_name, metric)
        if baseline_val is None or abs(baseline_val) < 1e-12:
            return {}

        results: Dict[str, Dict[str, float]] = {}
        for name in self.list_experiments(prefix=prefix):
            if name == f'{prefix}baseline':
                continue
            rest = name[len(prefix):]
            if '=' not in rest:
                continue
            knob, val_str = rest.rsplit('=', 1)
            exp_dir = os.path.join(self._experiment_root, name)
            extractor = _make_extractor(metric)
            val = _aggregate_exp(exp_dir, extractor)
            if val is None:
                continue
            deviation = (val - baseline_val) / abs(baseline_val)
            results.setdefault(knob, {})[val_str] = deviation
        return results

    def get_knob_paired_data(
        self, baseline_name: str = 'A_baseline',
        prefix: str = 'B_', metric: str = 'final_best',
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Dict[str, float]]]]:
        """Tornado 配对数据: 返回 (baseline_by_index, perturbed_by_index).

        baseline_by_index  : {index: baseline_sample_mean}
        perturbed_by_index : {knob: {value_str: {index: perturbed_sample_mean}}}

        配对设计: 同一个 index 上, baseline 的样本内均值 vs 扰动的样本内均值.
        这样 baseline 不会被同一个 knob 的多个 value_str 重复累加.
        """
        baseline_dir = os.path.join(self._experiment_root, baseline_name)
        extractor = _make_extractor(metric)

        baseline_by_index: Dict[str, float] = {}
        for idx in _list_samples(baseline_dir):
            b_dirs = _list_run_dirs_for_sample(baseline_dir, idx)
            b_vals = [v for v in (extractor(rd) for rd in b_dirs) if v is not None]
            if b_vals:
                baseline_by_index[idx] = float(np.mean(b_vals))

        perturbed_by_index: Dict[str, Dict[str, Dict[str, float]]] = {}
        for name in self.list_experiments(prefix=prefix):
            if name == f'{prefix}baseline':
                continue
            rest = name[len(prefix):]
            if '=' not in rest:
                continue
            knob, val_str = rest.rsplit('=', 1)
            exp_dir = os.path.join(self._experiment_root, name)
            if not os.path.isdir(exp_dir):
                continue

            for idx in _list_samples(exp_dir):
                if idx not in baseline_by_index:
                    continue
                p_dirs = _list_run_dirs_for_sample(exp_dir, idx)
                p_vals = [v for v in (extractor(rd) for rd in p_dirs) if v is not None]
                if p_vals:
                    perturbed_by_index.setdefault(knob, {}).setdefault(val_str, {})[idx] = float(np.mean(p_vals))

        return baseline_by_index, perturbed_by_index

    def recompute_and_aggregate(self, exp_name: str, index: str) -> Dict[str, Any]:
        """重算 iter_to_95pct 并返回该样本的聚合指标."""
        exp_dir = os.path.join(self._experiment_root, exp_name)
        recompute_iter_to_95pct(exp_dir, index)
        return aggregate_runs(exp_dir, index)

    def get_sensitivity_data(
        self, prefix: str = 'S_', metric: str = 'final_best',
    ) -> Dict[str, Dict[float, List[float]]]:
        """Sensitivity 分析: 返回 {param_path: {value: [sample_means]}}."""
        extractor = _make_extractor(metric)
        param_data: Dict[str, Dict[float, List[float]]] = {}

        for name in self.list_experiments(prefix=prefix):
            rest = name[len(prefix):]
            if '=' not in rest:
                continue
            path, val_str = rest.rsplit('=', 1)
            try:
                val = float(val_str)
            except ValueError:
                continue

            exp_dir = os.path.join(self._experiment_root, name)
            sample_means: List[float] = []
            for idx in _list_samples(exp_dir):
                run_dirs = _list_run_dirs_for_sample(exp_dir, idx)
                vals = [v for v in (extractor(rd) for rd in run_dirs) if v is not None]
                if vals:
                    sample_means.append(float(np.mean(vals)))
            if sample_means:
                param_data.setdefault(path, {})[val] = sample_means
        return param_data


# ------------------------------------------------------------
# Stability 模式专用：runs / per_sample / summary 三层 CSV
# ------------------------------------------------------------
def _load_representative_meta(rep_json_path: str) -> Tuple[Dict[str, int], List[Tuple[float, float]]]:
    """读取 representative_samples.json, 返回 (index→n_pred 映射, Q1-Q5 区间列表).

    quantile_ranges 在 JSON 中原 key 为 Q0–Q4, 这里转成按论文口径的 Q1–Q5 列表
    (列表第 i 项对应 Qi+1), 每项 (min_n_pred, max_n_pred).
    """
    with open(rep_json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    npred_map = {rec['index']: int(rec['n_pred']) for rec in meta.get('all_records', [])}
    q_raw = meta.get('quantile_ranges', {})
    ranges: List[Tuple[float, float]] = []
    for i in range(5):
        # 兼容两种 key: "Q0"-"Q4" 或 "0"-"4"
        for key in (f'Q{i}', str(i)):
            if key in q_raw:
                ranges.append((float(q_raw[key]['min']), float(q_raw[key]['max'])))
                break
    return npred_map, ranges


def _classify_level(n_pred: int, q_ranges: List[Tuple[float, float]]) -> str:
    """按 n_pred 落入 quantile_ranges 第几档, 返回 'Q1'-'Q5'; 找不到返回 ''."""
    if not q_ranges or n_pred is None:
        return ''
    # 区间右开左闭外加最后一档闭合; 容差按 <= max 处理避免边界丢失
    for i, (lo, hi) in enumerate(q_ranges):
        if n_pred <= hi:
            # 排除小于第一档下界的极端样本
            if i == 0 and n_pred < lo:
                return ''
            return f'Q{i + 1}'
    return f'Q{len(q_ranges)}'  # 大于最后一档上界时归到最后


def _write_runs_csv_core(experiment_root: str,
                         rep_json_path: str,
                         groups: List[Tuple[str, str]],
                         save_path: Optional[str],
                         sort_key,
                         label: str) -> str:
    """扫描指定实验组的所有 run, 落盘扁平 runs.csv (stability / ablation 共用)."""
    # 1) 读 runner_type
    runner_type = 'SA'
    summary_path = os.path.join(experiment_root, 'experiment_summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                runner_type = json.load(f).get('runner_type', 'SA').upper()
        except Exception as e:
            print(f"  [{label}] 读取 experiment_summary.json 失败, 默认 SA: {e}")

    # 2) 读 n_pred / level 映射
    if not os.path.exists(rep_json_path):
        print(f"  [{label}] {rep_json_path} 不存在, n_pred/level 列将留空")
        npred_map: Dict[str, int] = {}
        q_ranges: List[Tuple[float, float]] = []
    else:
        npred_map, q_ranges = _load_representative_meta(rep_json_path)

    # 3) 扫描所有 group / run / sample
    rows: List[Dict[str, Any]] = []
    for group_name, config_label in groups:
        exp_dir = _resolve_group_dir(experiment_root, group_name)
        if not os.path.isdir(exp_dir):
            if len(groups) == 1:
                print(f"  [{label}] {group_name} 目录缺失: {exp_dir}")
                return ''
            continue
        for entry in sorted(os.listdir(exp_dir)):
            if not entry.startswith('run_'):
                continue
            try:
                r = int(entry.split('_')[1])
            except (IndexError, ValueError):
                continue
            run_dir = os.path.join(exp_dir, entry)
            if not os.path.isdir(run_dir):
                continue
            for index in sorted(os.listdir(run_dir)):
                sample_dir = os.path.join(run_dir, index)
                conv_path = os.path.join(sample_dir, 'convergence.json')
                if not os.path.exists(conv_path):
                    continue
                f_final = _read_conv_metric(conv_path, 'final_best')
                if f_final is None:
                    continue
                n_pred = npred_map.get(index, '')
                level = _classify_level(n_pred, q_ranges) if isinstance(n_pred, int) else ''
                rows.append({
                    'sample_id': index,
                    'n_pred': n_pred,
                    'level': level,
                    'algorithm': runner_type,
                    'config': config_label,
                    'run_idx': r + 1,
                    'F_final': f_final,
                    'ESV': _read_conv_metric(conv_path, 'es_score'),
                    'Runtime': _read_conv_metric(conv_path, 'wall_time_sec'),
                    'T95': _read_conv_metric(conv_path, 'iter_to_95pct'),
                })

    # 4) 落盘
    rows.sort(key=sort_key)
    ensure_folder(os.path.dirname(save_path))
    cols = ['sample_id', 'n_pred', 'level', 'algorithm', 'config', 'run_idx', 'F_final', 'ESV', 'Runtime', 'T95']
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) if r[c] is not None else '' for c in cols) + '\n')
    print(f"  {label} saved ({len(rows)} rows): {save_path}")
    return save_path


def write_stability_runs_csv(experiment_root: str,
                             rep_json_path: str,
                             save_path: Optional[str] = None) -> str:
    """扫描单次 stability pipeline 的所有 run, 落盘 stability_runs.csv."""
    save_path = save_path or os.path.join(experiment_root, 'plots', 'stability_runs.csv')
    return _write_runs_csv_core(
        experiment_root, rep_json_path,
        groups=[('A_guided', _DIR_TO_SHORT.get('A_guided', 'A'))],
        save_path=save_path,
        sort_key=lambda x: (x['sample_id'], x['run_idx']),
        label='stability_runs',
    )


def write_ablation_runs_csv(experiment_root: str,
                            rep_json_path: str,
                            save_path: Optional[str] = None) -> str:
    """扫描 ablation pipeline (A/B/C 三组) 的所有 run, 落盘 ablation_runs.csv."""
    save_path = save_path or os.path.join(experiment_root, 'plots', 'ablation_runs.csv')
    return _write_runs_csv_core(
        experiment_root, rep_json_path,
        groups=[(g[0], g[2]) for g in ABLATION_GROUPS],
        save_path=save_path,
        sort_key=lambda x: (x['sample_id'], x['config'], x['run_idx']),
        label='ablation_runs',
    )


def merge_stability_runs(runs_csv_paths: List[str]):
    """合并多个 stability_runs.csv (通常 SA+GA 各一份), 返回 pandas DataFrame.

    依赖 pandas; 若 pandas 不可用, 退化为 list[dict].
    """
    import pandas as pd
    frames = []
    for p in runs_csv_paths:
        if not os.path.exists(p):
            print(f"  [merge_stability] 文件不存在, 跳过: {p}")
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame(columns=['sample_id', 'n_pred', 'level', 'algorithm', 'config', 'run_idx', 'F_final', 'ESV', 'Runtime', 'T95'])
    df = pd.concat(frames, ignore_index=True)
    return df


def _calc_metric_stats(vals, n: int, t_crit_n5: float = 2.776):
    """对一组值计算 μ/σ/CV/95%CI 统计量.

    Parameters
    ----------
    vals : np.ndarray
        原始数值数组
    n : int
        样本量 (通常为重复运行次数)
    t_crit_n5 : float
        n=5 时的 t 临界值 (df=4, α=0.05 双侧). n≠5 时按 scipy 重算.

    Returns
    -------
    tuple[float, float, float, float, float]
        (mu, sigma, cv_pct, ci95_half, ci95_width_ratio_pct)
    """
    mu = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
    cv_pct = float(sigma / abs(mu) * 100) if mu != 0 else float('nan')
    if n >= 2:
        t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n != 5 else t_crit_n5
        ci95_half = t_crit * sigma / np.sqrt(n)
    else:
        ci95_half = 0.0
    ci95_width_ratio_pct = float(2 * ci95_half / abs(mu) * 100) if mu != 0 else float('nan')
    return mu, sigma, cv_pct, ci95_half, ci95_width_ratio_pct


def build_stability_per_sample_csv(runs_df,
                                   save_path: str,
                                   t_crit_n5: float = 2.776):
    """按 (sample_id, algorithm) 聚合 runs, 对 F_final / ESV / Runtime 分别输出 μ/σ/CV/95%CI.

    为同时量化目标函数、解质量与运行时间的稳定性, 内部复用 ``_calc_metric_stats``
    对三列指标计算相同的统计量, 输出列名用 F_/ES_/RT_ 前缀区分.

    Parameters
    ----------
    runs_df : pd.DataFrame
        原始 runs.csv, 至少含 sample_id / algorithm / F_final / ESV / Runtime 列.
    save_path : str
        输出 CSV 路径.
    t_crit_n5 : float
        n=5 时的 t 临界值, 用于 CI 计算.

    Returns
    -------
    pd.DataFrame
        每行一个 (sample_id, algorithm) 组合, 列:
        sample_id, n_pred, level, algorithm, n_runs,
        F_mu, F_sigma, F_cv_pct, F_ci95_half, F_ci95_width_ratio_pct,
        ES_mu, ES_sigma, ES_cv_pct, ES_ci95_half, ES_ci95_width_ratio_pct,
        RT_mu, RT_sigma, RT_cv_pct, RT_ci95_half, RT_ci95_width_ratio_pct.
    """
    import pandas as pd
    from scipy import stats as _stats

    if runs_df is None or runs_df.empty:
        print(f"  [per_sample] runs_df 为空, 跳过")
        return pd.DataFrame()

    nan = float('nan')
    rows = []
    grouped = runs_df.groupby(['sample_id', 'algorithm'], sort=False)
    for (sample_id, alg), g in grouped:
        n = len(g)
        f_vals = g['F_final'].astype(float).values
        rt_vals = g['Runtime'].astype(float).values

        f_mu, f_sigma, f_cv, f_ci_half, f_ci_ratio = _calc_metric_stats(f_vals, n, t_crit_n5)
        rt_mu, rt_sigma, rt_cv, rt_ci_half, rt_ci_ratio = _calc_metric_stats(rt_vals, n, t_crit_n5)

        # ESV 在历史数据中可能缺失或为 NaN, 过滤后退化样本量再交给 _calc_metric_stats
        if 'ESV' in g.columns:
            es_raw = g['ESV'].astype(float).values
            es_vals = es_raw[np.isfinite(es_raw)]
        else:
            es_vals = np.array([], dtype=float)
        if es_vals.size >= 2:
            es_mu, es_sigma, es_cv, es_ci_half, es_ci_ratio = _calc_metric_stats(
                es_vals, int(es_vals.size), t_crit_n5)
        elif es_vals.size == 1:
            es_mu, es_sigma, es_cv, es_ci_half, es_ci_ratio = float(es_vals[0]), 0.0, nan, 0.0, nan
        else:
            es_mu = es_sigma = es_cv = es_ci_half = es_ci_ratio = nan

        n_pred = int(g['n_pred'].iloc[0]) if g['n_pred'].notna().any() else ''
        level = g['level'].iloc[0] if g['level'].notna().any() else ''
        rows.append({
            'sample_id': sample_id,
            'n_pred': n_pred,
            'level': level,
            'algorithm': alg,
            'n_runs': n,
            'F_mu': f_mu, 'F_sigma': f_sigma, 'F_cv_pct': f_cv,
            'F_ci95_half': f_ci_half, 'F_ci95_width_ratio_pct': f_ci_ratio,
            'ES_mu': es_mu, 'ES_sigma': es_sigma, 'ES_cv_pct': es_cv,
            'ES_ci95_half': es_ci_half, 'ES_ci95_width_ratio_pct': es_ci_ratio,
            'RT_mu': rt_mu, 'RT_sigma': rt_sigma, 'RT_cv_pct': rt_cv,
            'RT_ci95_half': rt_ci_half, 'RT_ci95_width_ratio_pct': rt_ci_ratio,
        })

    out = pd.DataFrame(rows)
    # 排序: 按 n_pred 升序, 同 sample 内 SA 在前 GA 在后, 方便对照论文图
    if not out.empty:
        out = out.sort_values(['n_pred', 'sample_id', 'algorithm']).reset_index(drop=True)
    ensure_folder(os.path.dirname(save_path))
    out.to_csv(save_path, index=False)
    print(f"  stability_per_sample saved ({len(out)} rows): {save_path}")
    return out


def build_stability_summary_csv(per_sample_df,
                                save_path: str,
                                thresholds_pct=(1.0, 3.0, 5.0, 10.0)):
    """按 algorithm × metric 汇总 per_sample 的 CV 分布, 输出论文 tab:stability-summary.

    对 F(S) / ES / Runtime 三类 CV 做跨样本汇总, 生成 6 行:
    SA-F(S), SA-ES, SA-Runtime, GA-F(S), GA-ES, GA-Runtime.

    Parameters
    ----------
    per_sample_df : pd.DataFrame
        build_stability_per_sample_csv 的输出, 含 F_cv_pct / ES_cv_pct / RT_cv_pct 列.
    save_path : str
        输出 CSV 路径.
    thresholds_pct : tuple of float
        需要统计的 CV 阈值列表 (默认 1%, 3%, 5%, 10%).

    Returns
    -------
    pd.DataFrame
        列: algorithm, metric, n_samples, mean_cv_pct, median_cv_pct, max_cv_pct,
        cv_le_1pct_ratio_pct, cv_le_3pct_ratio_pct, cv_le_5pct_ratio_pct, cv_le_10pct_ratio_pct.
    """
    import pandas as pd

    if per_sample_df is None or per_sample_df.empty:
        print(f"  [summary] per_sample_df 为空, 跳过")
        return pd.DataFrame()

    rows = []
    for alg, g in per_sample_df.groupby('algorithm', sort=False):
        for metric_label, cv_col in [('F(S)', 'F_cv_pct'), ('ES', 'ES_cv_pct'), ('Runtime', 'RT_cv_pct')]:
            cv_series = g[cv_col].astype(float)
            # 过滤 NaN (ES 列在部分历史数据中可能全为 NaN)
            cv = cv_series.dropna().values
            if cv.size == 0:
                continue
            n_valid = cv.size
            row = {
                'algorithm': alg,
                'metric': metric_label,
                'n_samples': n_valid,
                'mean_cv_pct': float(np.mean(cv)),
                'median_cv_pct': float(np.median(cv)),
                'max_cv_pct': float(np.max(cv)),
            }
            for thr in thresholds_pct:
                n_le_thr = int(np.sum(cv <= thr))
                row[f'cv_le_{int(thr)}pct_ratio_pct'] = float(n_le_thr / n_valid * 100) if n_valid else 0.0
            rows.append(row)

    out = pd.DataFrame(rows)
    ensure_folder(os.path.dirname(save_path))
    out.to_csv(save_path, index=False)
    print(f"  stability_summary saved ({len(out)} rows): {save_path}")
    return out


# ------------------------------------------------------------
# DAR (Dimensional Attainment Rate) — 维度达标率矩阵
# ------------------------------------------------------------
_DAR_DIMS = ['food', 'carbon', 'water', 'culture']
_DAR_DIM_LABELS = {'food': '粮食供给', 'carbon': '碳固定', 'water': '水源涵养', 'culture': '文化服务'}


def _read_es_plans_from_perf(perf_path: str) -> Optional[Dict[str, Dict[str, float]]]:
    """从 performance.json 读取各维度的 best_plan 和 tgt_plan.

    Returns
    -------
    dict or None
        {dim: {'best': float, 'tgt': float}} 或 None (文件不存在/格式错误).
    """
    if not os.path.exists(perf_path):
        return None
    try:
        with open(perf_path, 'r', encoding='utf-8') as f:
            perf = json.load(f)
        es_data = perf.get('ES', {})
        out = {}
        for dim in _DAR_DIMS:
            entry = es_data.get(dim, {})
            best = entry.get('best_plan')
            tgt = entry.get('tgt_plan')
            if best is not None and tgt is not None:
                out[dim] = {'best': float(best), 'tgt': float(tgt)}
        return out if out else None
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def build_dar_detail_csv(exp_dir: str,
                         run_start: int = 0,
                         run_end: int = 4,
                         save_path: Optional[str] = None) -> 'pd.DataFrame':
    """Step 1-2: 逐 run 逐样本读取 performance.json, 输出 0/1 判断矩阵.

    对每个样本的每次 run, 判断各维度 best_plan > tgt_plan, 记为 1, 否则 0.

    Parameters
    ----------
    exp_dir : str
        实验目录 (如 results/ablation/<ts>/A_guided).
    run_start, run_end : int
        run 范围, 默认 0-4 (R=5).
    save_path : str or None
        输出 CSV 路径, 默认 <exp_dir>/plots/dar_detail.csv.

    Returns
    -------
    pd.DataFrame
        列: sample_id, run, food_beat, carbon_beat, water_beat, culture_beat.
    """
    import pandas as pd

    samples = _list_samples(exp_dir)
    if not samples:
        print(f"  [dar_detail] 未找到样本: {exp_dir}")
        return pd.DataFrame(columns=['sample_id', 'run'] + [f'{d}_beat' for d in _DAR_DIMS])

    rows = []
    for idx in samples:
        for r in range(run_start, run_end + 1):
            rd = os.path.join(exp_dir, f'run_{r}', idx)
            perf_path = os.path.join(rd, 'performance.json')
            plans = _read_es_plans_from_perf(perf_path)
            if plans is None:
                continue
            row = {'sample_id': idx, 'run': r}
            for dim in _DAR_DIMS:
                dim_data = plans.get(dim)
                if dim_data is not None:
                    row[f'{dim}_beat'] = int(dim_data['best'] > dim_data['tgt'])
                else:
                    row[f'{dim}_beat'] = ''
            rows.append(row)

    out = pd.DataFrame(rows)
    if save_path is not None:
        ensure_folder(os.path.dirname(save_path))
        out.to_csv(save_path, index=False)
        print(f"  dar_detail saved ({len(out)} rows): {save_path}")
    return out


def build_dar_matrix_csv(exp_dir: str,
                         rep_json_path: Optional[str] = None,
                         run_start: int = 0,
                         run_end: int = 4,
                         save_path: Optional[str] = None) -> 'pd.DataFrame':
    """Step 3-4: 聚合 0/1 判断矩阵为 S×4 DAR 矩阵.

    DAR_d = (1/R) * sum_{r} 1[best_plan_{r,d} > tgt_plan_d]

    每个维度值范围为 [0, 1], 步长 1/R.
    若提供 rep_json_path, 附加 n_pred / level 列.

    Parameters
    ----------
    exp_dir : str
        实验目录.
    rep_json_path : str or None
        representative_samples.json 路径, 用于获取 n_pred/level.
    run_start, run_end : int
        run 范围.
    save_path : str or None
        输出路径, 默认 <exp_dir>/plots/dar_matrix.csv.

    Returns
    -------
    pd.DataFrame
        列: sample_id, n_pred, level, n_runs, food_dar, carbon_dar, water_dar, culture_dar.
    """
    import pandas as pd

    # 先不落盘 detail, 先算矩阵; 最终统一目录落盘
    detail = build_dar_detail_csv(exp_dir, run_start, run_end, save_path=None)
    if detail.empty:
        return pd.DataFrame(
            columns=['sample_id', 'n_pred', 'level', 'n_runs']
            + [f'{d}_dar' for d in _DAR_DIMS])

    # 加载 n_pred / level 映射
    npred_map: Dict[str, int] = {}
    q_ranges: List[Tuple[float, float]] = []
    if rep_json_path and os.path.exists(rep_json_path):
        npred_map, q_ranges = _load_representative_meta(rep_json_path)

    rows = []
    for sid, g in detail.groupby('sample_id', sort=False):
        n_runs = len(g)
        if n_runs == 0:
            continue
        n_pred = npred_map.get(sid, '')
        level = _classify_level(n_pred, q_ranges) if isinstance(n_pred, int) else ''
        row = {
            'sample_id': sid,
            'n_pred': n_pred if n_pred != '' else '',
            'level': level,
            'n_runs': n_runs,
        }
        for dim in _DAR_DIMS:
            col = f'{dim}_beat'
            if col in g.columns:
                vals = g[col].dropna()
                row[f'{dim}_dar'] = round(float(vals.mean()), 4) if len(vals) > 0 else ''
            else:
                row[f'{dim}_dar'] = ''
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['n_pred', 'sample_id']).reset_index(drop=True)

    if save_path is not None or not detail.empty:
        save_path = save_path or os.path.join(exp_dir, 'plots', 'dar_matrix.csv')
        ensure_folder(os.path.dirname(save_path))
        out.to_csv(save_path, index=False)
        print(f"  dar_matrix saved ({len(out)} rows): {save_path}")

        # 将 detail 也落盘到 matrix 同目录, 方便后续合并
        detail_save_path = os.path.join(os.path.dirname(save_path), 'dar_detail.csv')
        detail.to_csv(detail_save_path, index=False)
        print(f"  dar_detail saved ({len(detail)} rows): {detail_save_path}")
    return out


def build_dar_summary_csv(dar_matrix: 'pd.DataFrame',
                          save_path: Optional[str] = None) -> 'pd.DataFrame':
    """Step 5: 跨样本 DAR 汇总统计.

    对每个维度计算 mean, std, min, max 并附加定性模式分类:
      - DAR ≥ 0.8 → 稳定达标型
      - DAR ≤ 0.2 → 系统未达标型
      - 其他     → 方向不确定型

    Parameters
    ----------
    dar_matrix : pd.DataFrame
        build_dar_matrix_csv 的输出.
    save_path : str or None
        输出路径.

    Returns
    -------
    pd.DataFrame
        列: dimension, dimension_label, mean_dar, std_dar, min_dar, max_dar, qualitative_pattern.
    """
    import pandas as pd

    if dar_matrix is None or dar_matrix.empty:
        print(f"  [dar_summary] dar_matrix 为空, 跳过")
        return pd.DataFrame(columns=[
            'dimension', 'dimension_label', 'mean_dar', 'std_dar',
            'min_dar', 'max_dar', 'qualitative_pattern'])

    rows = []
    for dim in _DAR_DIMS:
        col = f'{dim}_dar'
        if col not in dar_matrix.columns:
            continue
        vals = dar_matrix[col].dropna().astype(float).values
        if len(vals) == 0:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))

        if mean_val >= 0.8:
            pattern = '稳定达标型'
        elif mean_val <= 0.2:
            pattern = '系统未达标型'
        else:
            pattern = '方向不确定型'

        rows.append({
            'dimension': dim,
            'dimension_label': _DAR_DIM_LABELS.get(dim, dim),
            'mean_dar': round(mean_val, 4),
            'std_dar': round(std_val, 4),
            'min_dar': round(min_val, 4),
            'max_dar': round(max_val, 4),
            'qualitative_pattern': pattern,
        })

    out = pd.DataFrame(rows)
    if save_path is not None:
        ensure_folder(os.path.dirname(save_path))
        out.to_csv(save_path, index=False)
        print(f"  dar_summary saved ({len(out)} rows): {save_path}")
    return out


def build_dar_all(exp_dir: str,
                  rep_json_path: Optional[str] = None,
                  run_start: int = 0,
                  run_end: int = 4,
                  output_dir: Optional[str] = None) -> Dict[str, 'pd.DataFrame']:
    """一键生成 DAR 三张表 (detail / matrix / summary).

    内部 build_dar_matrix_csv 会顺带生成 dar_detail.csv,
    本函数仅额外单独调用 build_dar_summary_csv 完成汇总.
    """
    output_dir = output_dir or os.path.join(exp_dir, 'plots')
    ensure_folder(output_dir)

    matrix = build_dar_matrix_csv(
        exp_dir, rep_json_path, run_start, run_end,
        save_path=os.path.join(output_dir, 'dar_matrix.csv'),
    )
    summary = build_dar_summary_csv(
        matrix,
        save_path=os.path.join(output_dir, 'dar_summary.csv'),
    )
    # 读取已落盘的 detail (由 build_dar_matrix_csv 内部生成)
    import pandas as pd
    detail_path = os.path.join(output_dir, 'dar_detail.csv')
    detail = pd.read_csv(detail_path) if os.path.exists(detail_path) else pd.DataFrame()
    return {'detail': detail, 'matrix': matrix, 'summary': summary}
