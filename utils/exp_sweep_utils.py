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
    if metric == 'final_best':
        return lambda d: _read_conv_metric(os.path.join(d, 'convergence.json'), 'final_best')
    elif metric == 'wall_time_sec':
        return lambda d: _read_conv_metric(os.path.join(d, 'convergence.json'), 'wall_time_sec')
    elif metric == 'iter_to_95pct':
        return lambda d: _read_conv_metric(os.path.join(d, 'convergence.json'), 'iter_to_95pct')
    else:
        path = metric.split('.')
        return lambda d: _read_perf_metric(
            os.path.join(d, 'performance.json'), path)


_METRIC_KEYS_CONV = ['final_best', 'iter_to_95pct', 'wall_time_sec', 'total_iter', 'es_score']
_METRIC_KEYS_PERF = [
    'ES.food.best_ori_diff', 'ES.carbon.best_ori_diff',
    'ES.water.best_ori_diff', 'ES.culture.best_ori_diff',
    'ES.food.best_tgt_diff', 'ES.carbon.best_tgt_diff',
    'ES.water.best_tgt_diff', 'ES.culture.best_tgt_diff',
    'Change_area.real_metrics.iou', 'Change_area.real_metrics.f1_score',
    'Change_area.pred_metrics.iou', 'Change_area.pred_metrics.f1_score',
    'FINAL.FOM', 'FINAL.kappa',
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
    """用多次 final_best 均值作为参考最优, 重算每次 run 的 iter_to_95pct_repeat."""
    run_dirs = _list_run_dirs_for_sample(exp_dir, index)
    if len(run_dirs) < 2:
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

    if runner_type == 'SA':
        target = ref_best * 1.05 if ref_best >= 0 else ref_best * 0.95
        for rd in run_dirs:
            npy_p = os.path.join(rd, 'total_score_list.npy')
            conv_p = os.path.join(rd, 'convergence.json')
            if not (os.path.exists(npy_p) and os.path.exists(conv_p)):
                continue
            arr = np.load(npy_p)
            hit = (arr <= target).any()
            iter_repeat = int(np.argmax(arr <= target)) if hit else -1
            with open(conv_p, 'r', encoding='utf-8') as f:
                conv = json.load(f)
            conv['iter_to_95pct_repeat'] = iter_repeat
            conv['ref_best_repeat'] = ref_best
            with open(conv_p, 'w', encoding='utf-8') as f:
                json.dump(conv, f, indent=2, ensure_ascii=False)
    else:
        ref_best_fitness = -ref_best
        target = ref_best_fitness * 0.95 if ref_best_fitness >= 0 else ref_best_fitness * 1.05
        for rd in run_dirs:
            npy_p = os.path.join(rd, 'total_score_list.npy')
            conv_p = os.path.join(rd, 'convergence.json')
            if not (os.path.exists(npy_p) and os.path.exists(conv_p)):
                continue
            arr = np.load(npy_p)
            hit = (arr >= target).any()
            iter_repeat = int(np.argmax(arr >= target)) if hit else -1
            with open(conv_p, 'r', encoding='utf-8') as f:
                conv = json.load(f)
            conv['iter_to_95pct_repeat'] = iter_repeat
            conv['ref_best_repeat'] = ref_best
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
    """A vs B vs C 三组并排, 每个样本一行 (各组 mean±std 与相对 A 的 % 变化)."""
    per_group, samples = _load_ablation_data(experiment_root)
    groups = [g[0] for g in ABLATION_GROUPS]

    rows = []
    metrics = ['final_best', 'iter_to_95pct_repeat', 'wall_time_sec', 'es_score']
    for idx in sorted(samples):
        row = {'sample': idx}
        for g in groups:
            agg = per_group.get(g, {}).get(idx, {})
            for m in metrics:
                v = agg.get(m, {})
                row[f'{g}.{m}.mean'] = v.get('mean', '')
                row[f'{g}.{m}.std'] = v.get('std', '')
        a = per_group.get('A_guided', {}).get(idx, {})
        for g in ['B_unguided', 'C_random']:
            other = per_group.get(g, {}).get(idx, {})
            for m in metrics:
                am = a.get(m, {}).get('mean')
                om = other.get(m, {}).get('mean')
                if am and am != 0 and om is not None:
                    row[f'{g}.{m}.rel_to_A_pct'] = (om - am) / abs(am) * 100
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
    """A/B/C 每组按样本取均值, 对应 ablation.md 表2 (样本-配置均值)."""
    per_group, samples = _load_ablation_data(experiment_root)
    groups = [g[0] for g in ABLATION_GROUPS]

    rows = []
    metrics = ['final_best', 'iter_to_95pct', 'wall_time_sec', 'es_score']
    metric_map = {'final_best': 'Mean_F_S', 'iter_to_95pct': 'Mean_T95',
                  'wall_time_sec': 'Mean_Runtime', 'es_score': 'Mean_ESV'}
    for idx in sorted(samples):
        for g in groups:
            agg = per_group.get(g, {}).get(idx, {})
            row = {'Sample_ID': idx, 'Config': g.split('_')[0]}
            for m in metrics:
                v = agg.get(m, {})
                row[metric_map[m]] = v.get('mean', '')
            rows.append(row)

    save_path = save_path or os.path.join(experiment_root, 'plots', 'ablation_mean_table.csv')
    ensure_folder(os.path.dirname(save_path))
    if rows:
        keys = ['Sample_ID', 'Config', 'Mean_T95', 'Mean_Runtime', 'Mean_ESV', 'Mean_F_S']
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"  ablation_mean_table saved: {save_path}")
    return rows


def build_ablation_improvement_table(experiment_root: str,
                                     save_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """计算每个样本 A vs B / A vs C 的提升率, 对应 ablation.md 表3 (样本提升率)."""
    per_group, samples = _load_ablation_data(experiment_root)

    rows = []
    metrics = ['iter_to_95pct', 'wall_time_sec', 'es_score']
    metric_label = {'iter_to_95pct': 'T95', 'wall_time_sec': 'Runtime', 'es_score': 'ESV'}
    for idx in sorted(samples):
        a = per_group.get('A_guided', {}).get(idx, {})
        row = {'Sample_ID': idx}
        for g in ['B_unguided', 'C_random']:
            other = per_group.get(g, {}).get(idx, {})
            comp = 'AvsB' if g == 'B_unguided' else 'AvsC'
            for m in metrics:
                am = a.get(m, {}).get('mean')
                om = other.get(m, {}).get('mean')
                label = metric_label[m]
                if am is not None and om is not None and om != 0:
                    if m == 'es_score':
                        # ESV Improvement = (A - 对照) / 对照 * 100
                        row[f'{comp}_{label}_Imp_pct'] = (am - om) / abs(om) * 100
                    else:
                        # T95/Runtime Reduction = (对照 - A) / 对照 * 100
                        row[f'{comp}_{label}_Red_pct'] = (om - am) / abs(om) * 100
                else:
                    if m == 'es_score':
                        row[f'{comp}_{label}_Imp_pct'] = ''
                    else:
                        row[f'{comp}_{label}_Red_pct'] = ''
        rows.append(row)

    save_path = save_path or os.path.join(experiment_root, 'plots', 'ablation_improvement_table.csv')
    ensure_folder(os.path.dirname(save_path))
    if rows:
        keys = ['Sample_ID',
                'AvsB_T95_Red_pct', 'AvsB_Runtime_Red_pct', 'AvsB_ESV_Imp_pct',
                'AvsC_T95_Red_pct', 'AvsC_Runtime_Red_pct', 'AvsC_ESV_Imp_pct']
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

    metrics = ['iter_to_95pct', 'wall_time_sec', 'es_score']
    metric_label = {'iter_to_95pct': 'T95 Reduction', 'wall_time_sec': 'Runtime Reduction', 'es_score': 'ESV Improvement'}
    rows = []
    for g in ['B_unguided', 'C_random']:
        comp = 'A vs B' if g == 'B_unguided' else 'A vs C'
        for m in metrics:
            vals = []
            for idx in sorted(samples):
                a = per_group.get('A_guided', {}).get(idx, {})
                other = per_group.get(g, {}).get(idx, {})
                am = a.get(m, {}).get('mean')
                om = other.get(m, {}).get('mean')
                if am is not None and om is not None and om != 0:
                    if m == 'es_score':
                        vals.append((am - om) / abs(om) * 100)
                    else:
                        vals.append((om - am) / abs(om) * 100)
            if vals:
                rows.append({
                    'Algorithm': runner_type,
                    'Comparison': comp,
                    'Metric': metric_label[m],
                    'Mean': round(float(np.mean(vals)), 2),
                    'Range_min': round(float(np.min(vals)), 2),
                    'Range_max': round(float(np.max(vals)), 2),
                })

    save_path = save_path or os.path.join(experiment_root, 'plots', 'ablation_summary_table.csv')
    ensure_folder(os.path.dirname(save_path))
    if rows:
        keys = ['Algorithm', 'Comparison', 'Metric', 'Mean', 'Range_min', 'Range_max']
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
        groups=[('A_guided', 'A')],
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
                                cv_threshold_pct: float = 5.0):
    """按 algorithm × metric 汇总 per_sample 的 CV 分布, 输出论文 tab:stability-summary.

    对 F(S) / ES / Runtime 三类 CV 做跨样本汇总, 生成 6 行:
    SA-F(S), SA-ES, SA-Runtime, GA-F(S), GA-ES, GA-Runtime.

    Parameters
    ----------
    per_sample_df : pd.DataFrame
        build_stability_per_sample_csv 的输出, 含 F_cv_pct / ES_cv_pct / RT_cv_pct 列.
    save_path : str
        输出 CSV 路径.
    cv_threshold_pct : float
        CV 稳定阈值 (默认 5%).

    Returns
    -------
    pd.DataFrame
        列: algorithm, metric, n_samples, mean_cv_pct, max_cv_pct, cv_le_5pct_ratio_pct.
    """
    import pandas as pd

    if per_sample_df is None or per_sample_df.empty:
        print(f"  [summary] per_sample_df 为空, 跳过")
        return pd.DataFrame()

    rows = []
    for alg, g in per_sample_df.groupby('algorithm', sort=False):
        n_samples = len(g)
        for metric_label, cv_col in [('F(S)', 'F_cv_pct'), ('ES', 'ES_cv_pct'), ('Runtime', 'RT_cv_pct')]:
            cv_series = g[cv_col].astype(float)
            # 过滤 NaN (ES 列在部分历史数据中可能全为 NaN)
            cv = cv_series.dropna().values
            if cv.size == 0:
                continue
            n_valid = cv.size
            n_le_thr = int(np.sum(cv <= cv_threshold_pct))
            rows.append({
                'algorithm': alg,
                'metric': metric_label,
                'n_samples': n_valid,
                'mean_cv_pct': float(np.mean(cv)),
                'max_cv_pct': float(np.max(cv)),
                f'cv_le_{int(cv_threshold_pct)}pct_ratio_pct': float(n_le_thr / n_valid * 100) if n_valid else 0.0,
            })

    out = pd.DataFrame(rows)
    ensure_folder(os.path.dirname(save_path))
    out.to_csv(save_path, index=False)
    print(f"  stability_summary saved ({len(out)} rows): {save_path}")
    return out
