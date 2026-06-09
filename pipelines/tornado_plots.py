# tornado / sensitivity 实验结果可视化: tornado 图 + 一维线图
import csv
import json
import os
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utils.io_utils import ensure_folder
from utils.exp_sweep_utils import ExperimentResultStore, aggregate_runs, _calc_metric_stats


TORNADO_PERFORMANCE_METRICS = [
    'ES.food.best_tgt_diff',
    'ES.carbon.best_tgt_diff',
    'ES.water.best_tgt_diff',
    'ES.culture.best_tgt_diff',
    'Change_area.real_metrics.iou',
    'Change_area.real_metrics.f1_score',
]

SENSITIVITY_METRICS = [
    'final_best',
    'iter_to_95pct',
    'wall_time_sec',
    'ES.food.best_ori_diff',
    'ES.carbon.best_ori_diff',
    'ES.water.best_ori_diff',
    'ES.culture.best_ori_diff',
]

METRIC_DIRECTIONS = {
    'final_best': 'max',
    'iter_to_95pct': 'min',
    'wall_time_sec': 'min',
    'ES.food.best_ori_diff': 'max',
    'ES.carbon.best_ori_diff': 'max',
    'ES.water.best_ori_diff': 'max',
    'ES.culture.best_ori_diff': 'max',
    'ES.food.best_tgt_diff': 'max',
    'ES.carbon.best_tgt_diff': 'max',
    'ES.water.best_tgt_diff': 'max',
    'ES.culture.best_tgt_diff': 'max',
    'Change_area.real_metrics.iou': 'max',
    'Change_area.real_metrics.f1_score': 'max',
    'Change_area.pred_metrics.iou': 'max',
    'Change_area.pred_metrics.f1_score': 'max',
    'FINAL.FOM': 'max',
    'FINAL.kappa': 'max',
}

CONVERGENCE_METRICS = ['final_best', 'iter_to_95pct', 'wall_time_sec', 'total_iter', 'es_score']
RUN_PERFORMANCE_METRICS = [
    'ES.food.best_ori_diff',
    'ES.carbon.best_ori_diff',
    'ES.water.best_ori_diff',
    'ES.culture.best_ori_diff',
    'ES.food.best_tgt_diff',
    'ES.carbon.best_tgt_diff',
    'ES.water.best_tgt_diff',
    'ES.culture.best_tgt_diff',
    'Change_area.real_metrics.iou',
    'Change_area.real_metrics.f1_score',
    'Change_area.pred_metrics.iou',
    'Change_area.pred_metrics.f1_score',
    'FINAL.FOM',
    'FINAL.kappa',
]


# ============================================================
# 0. 通用读取 / 写表工具
# ============================================================
def _metric_short_name(metric: str) -> str:
    """将 metric 名称压缩为适合图表显示的短名."""
    return (metric.replace('ES.', '')
            .replace('.best_ori_diff', '')
            .replace('.best_tgt_diff', '')
            .replace('Change_area.', '')
            .replace('FINAL.', ''))


def _read_summary(experiment_root: str) -> Dict[str, Any]:
    """读取 experiment_summary.json; 缺失或损坏时返回空字典."""
    summary_path = os.path.join(experiment_root, 'experiment_summary.json')
    if not os.path.exists(summary_path):
        return {}
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  [warn] 读取 experiment_summary.json 失败: {e}")
        return {}


def _read_runner_type(experiment_root: str) -> str:
    """读取实验算法类型, 默认 SA."""
    return str(_read_summary(experiment_root).get('runner_type', 'SA')).upper()


def _read_nested(data: Dict[str, Any], dotted: str) -> Optional[float]:
    """从嵌套 dict 中按点号路径读取 float."""
    cur: Any = data
    for key in dotted.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _get_run_idx(run_dir: str) -> int:
    """从 run 目录路径中提取 1-based run_idx."""
    run_idx = 0
    for part in run_dir.replace('\\', '/').split('/'):
        if part.startswith('run_'):
            try:
                run_idx = int(part.split('_')[1])
            except (IndexError, ValueError):
                pass
    return run_idx + 1


def _read_run_metrics(run_dir: str) -> Dict[str, Any]:
    """读取单次 run 的 convergence/performance 指标."""
    rec: Dict[str, Any] = {}
    conv_path = os.path.join(run_dir, 'convergence.json')
    if os.path.exists(conv_path):
        with open(conv_path, 'r', encoding='utf-8') as f:
            conv = json.load(f)
        for metric in CONVERGENCE_METRICS:
            rec[metric] = conv.get(metric)

    perf_path = os.path.join(run_dir, 'performance.json')
    if os.path.exists(perf_path):
        with open(perf_path, 'r', encoding='utf-8') as f:
            perf = json.load(f)
        for metric in RUN_PERFORMANCE_METRICS:
            rec[metric] = _read_nested(perf, metric)
    return rec


def _write_csv(save_path: str, rows: List[Dict[str, Any]], cols: List[str], label: str) -> str:
    """用 DictWriter 写 CSV, 统一处理目录和空值."""
    ensure_folder(os.path.dirname(save_path))
    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, '') if row.get(c) is not None else '' for c in cols})
    print(f"  {label} saved ({len(rows)} rows): {save_path}")
    return save_path


def _format_range(lo: Any, hi: Any) -> str:
    """格式化区间列."""
    if lo == '' or hi == '' or lo is None or hi is None:
        return ''
    return f"{lo} - {hi}"


# ============================================================
# 1. Tornado: B_* 相对 A_baseline 的扰动
# ============================================================
def plot_tornado(store: ExperimentResultStore,
                 baseline_name: str = 'A_baseline',
                 save_path: Optional[str] = None,
                 metric: str = 'ES.carbon.best_tgt_diff') -> Dict[str, Dict[str, float]]:
    """画 tornado 图: 每个 B_* knob 相对 baseline 的归一化扰动幅度.

    默认批处理只使用方案层 performance 指标, 避免 final_best / 收敛时间等
    受目标函数缩放或优化器控制参数直接影响的伪敏感指标.
    """
    experiment_root = store.experiment_root
    save_path = save_path or os.path.join(experiment_root, 'plots', f'tornado_{metric}.png')
    ensure_folder(os.path.dirname(save_path))

    baseline_val = store.get_baseline_metric(baseline_name, metric)
    if baseline_val is None or abs(baseline_val) < 1e-12:
        print(f"警告: baseline 【{baseline_name}】 metric={metric} 无效或为 0, 无法归一化")
        return {}

    results = store.get_knob_deviations(baseline_name, prefix='B_', metric=metric)
    if not results:
        print(f"警告: 无任何 B_* 实验数据可用, 跳过 tornado 绘图")
        return {}

    knob_widths = []
    for knob, vmap in results.items():
        devs = list(vmap.values())
        knob_widths.append((knob, min(devs), max(devs), max(devs) - min(devs)))
    knob_widths.sort(key=lambda t: t[3])

    knobs_sorted = [k[0] for k in knob_widths]
    mins = [k[1] for k in knob_widths]
    maxs = [k[2] for k in knob_widths]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(knobs_sorted))))
    y_pos = np.arange(len(knobs_sorted))
    ax.barh(y_pos, np.array(maxs) - np.array(mins), left=mins,
            color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(knobs_sorted, fontsize=8)
    ax.set_xlabel(f'normalized deviation of {metric}\n(perturbed - baseline) / |baseline|')
    ax.set_title(f'Tornado: plan-level sensitivity\nbaseline={baseline_name} ({metric}={baseline_val:.4f})')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Tornado saved: {save_path}")

    summary_path = os.path.join(os.path.dirname(save_path), f'tornado_{metric}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'baseline': baseline_val, 'metric': metric, 'results': results}, f, indent=2, ensure_ascii=False)
    return results


def plot_tornado_composite(store: ExperimentResultStore,
                           baseline_name: str = 'A_baseline',
                           save_dir: Optional[str] = None,
                           metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """跨方案层指标综合敏感度排名.

    仅使用 performance.json 中由最终方案重算的指标, 不使用 final_best / T95 /
    wall_time 等会被目标函数缩放或运行控制参数直接影响的指标.
    """
    experiment_root = store.experiment_root
    save_dir = save_dir or os.path.join(experiment_root, 'plots')
    ensure_folder(save_dir)

    metrics = metrics or TORNADO_PERFORMANCE_METRICS
    metric_weights = [(metric, 1 / len(metrics)) for metric in metrics]

    all_knobs: set = set()
    knob_widths_by_metric: Dict[str, Dict[str, float]] = {}

    for metric, _ in metric_weights:
        baseline_val = store.get_baseline_metric(baseline_name, metric)
        if baseline_val is None or abs(baseline_val) < 1e-12:
            print(f"  [composite] metric={metric} baseline 无效, 跳过")
            continue
        deviations = store.get_knob_deviations(baseline_name, prefix='B_', metric=metric)
        if not deviations:
            continue
        knob_widths_by_metric[metric] = {}
        for knob, vmap in deviations.items():
            devs = list(vmap.values())
            if not devs:
                continue
            width = max(devs) - min(devs)
            knob_widths_by_metric[metric][knob] = width
            all_knobs.add(knob)

    if not all_knobs:
        print("  [composite] 无可用数据, 跳过")
        return {}

    composite: Dict[str, float] = {}
    for knob in all_knobs:
        score = 0.0
        for metric, weight in metric_weights:
            score += knob_widths_by_metric.get(metric, {}).get(knob, 0.0) * weight
        composite[knob] = score

    ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 100)
    print("  Tornado 跨方案层指标综合敏感度排名 (performance metrics 等权)")
    print("=" * 100)
    header_metrics = [m[0] for m in metric_weights]
    print(f"{'排名':<4} {'参数':<42} {'综合':>8}", end='')
    for metric in header_metrics:
        print(f"  {_metric_short_name(metric):>12}", end='')
    print()
    print("-" * (60 + 15 * len(header_metrics)))

    for rank, (knob, score) in enumerate(ranked, 1):
        print(f"{rank:<4} {knob:<42} {score:>8.4f}", end='')
        for metric in header_metrics:
            width = knob_widths_by_metric.get(metric, {}).get(knob, 0.0)
            print(f"  {width:>12.4f}", end='')
        print()

    summary = {
        'baseline_name': baseline_name,
        'metrics_used': header_metrics,
        'weights': {name: weight for name, weight in metric_weights},
        'ranking': [
            {
                'rank': rank,
                'parameter': knob,
                'composite_score': round(score, 6),
                'per_metric_widths': {
                    metric: round(knob_widths_by_metric.get(metric, {}).get(knob, 0.0), 6)
                    for metric in header_metrics
                },
            }
            for rank, (knob, score) in enumerate(ranked, 1)
        ],
    }
    summary_path = os.path.join(save_dir, 'tornado_composite_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nTornado composite summary saved: {summary_path}")

    csv_path = os.path.join(save_dir, 'tornado_composite_ranking.csv')
    csv_cols = ['rank', 'parameter', 'composite_score'] + header_metrics
    rows = []
    for rank, (knob, score) in enumerate(ranked, 1):
        row = {'rank': rank, 'parameter': knob, 'composite_score': round(score, 6)}
        for metric in header_metrics:
            row[metric] = round(knob_widths_by_metric.get(metric, {}).get(knob, 0.0), 6)
        rows.append(row)
    _write_csv(csv_path, rows, csv_cols, 'tornado_composite_ranking.csv')

    knobs_sorted = [k for k, _ in ranked]
    scores = [s for _, s in ranked]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(knobs_sorted))))
    y_pos = np.arange(len(knobs_sorted))
    ax.barh(y_pos, scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(knobs_sorted, fontsize=8)
    ax.set_xlabel('composite sensitivity score\n(equal-weight plan-level performance metrics)')
    ax.set_title('Tornado: composite sensitivity ranking\n(performance metrics only)')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    for i, score in enumerate(scores):
        ax.text(score + 0.002, i, f'{score:.4f}', va='center', fontsize=7)
    fig.tight_layout()
    chart_path = os.path.join(save_dir, 'tornado_composite.png')
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Tornado composite chart saved: {chart_path}")

    return composite


# ============================================================
# 1.5 Tornado 中间数据表 (对标 docs/ablation/tornado.md)
# ============================================================
def _list_tornado_experiments(store: ExperimentResultStore) -> List[str]:
    """返回所有 tornado 实验目录名 (A_baseline + B_*), 按名称排序."""
    names = store.list_experiments()
    ordered = []
    if 'A_baseline' in names:
        ordered.append('A_baseline')
    ordered.extend(sorted([name for name in names if name.startswith('B_')]))
    return ordered


def write_tornado_runs_csv(store: ExperimentResultStore,
                           save_path: Optional[str] = None) -> str:
    """Tornado 表1: 原始运行级数据 (sample × experiment × run).

    表内保留 convergence 字段作为原始记录, 但 tornado 默认敏感度排序只使用
    performance.json 中的方案层指标.
    """
    experiment_root = store.experiment_root
    save_path = save_path or os.path.join(experiment_root, 'plots', 'tornado_runs.csv')
    runner_type = _read_runner_type(experiment_root)

    rows: List[Dict[str, Any]] = []
    for exp_name in _list_tornado_experiments(store):
        for idx in sorted(store.list_samples(exp_name)):
            for run_dir in store.list_run_dirs(exp_name, idx):
                row = {
                    'sample_id': idx,
                    'experiment': exp_name,
                    'run_idx': _get_run_idx(run_dir),
                    'algorithm': runner_type,
                }
                row.update(_read_run_metrics(run_dir))
                rows.append(row)

    rows.sort(key=lambda r: (r['sample_id'], r['experiment'], r['run_idx']))
    cols = ['sample_id', 'experiment', 'run_idx', 'algorithm'] + CONVERGENCE_METRICS + RUN_PERFORMANCE_METRICS
    return _write_csv(save_path, rows, cols, 'tornado_runs.csv')


def build_tornado_mean_table(store: ExperimentResultStore,
                             save_path: Optional[str] = None) -> str:
    """Tornado 表2: 样本-配置均值 (sample × experiment)."""
    experiment_root = store.experiment_root
    save_path = save_path or os.path.join(experiment_root, 'plots', 'tornado_mean_table.csv')

    rows: List[Dict[str, Any]] = []
    for exp_name in _list_tornado_experiments(store):
        exp_dir = os.path.join(experiment_root, exp_name)
        for idx in sorted(store.list_samples(exp_name)):
            agg = aggregate_runs(exp_dir, idx)
            if not agg:
                continue
            row = {'sample_id': idx, 'experiment': exp_name}
            for metric in CONVERGENCE_METRICS + RUN_PERFORMANCE_METRICS:
                data = agg.get(metric, {})
                row[f'{metric}_mean'] = data.get('mean', '')
                row[f'{metric}_std'] = data.get('std', '')
            rows.append(row)

    rows.sort(key=lambda r: (r['sample_id'], r['experiment']))
    cols = ['sample_id', 'experiment']
    for metric in CONVERGENCE_METRICS + RUN_PERFORMANCE_METRICS:
        cols.extend([f'{metric}_mean', f'{metric}_std'])
    return _write_csv(save_path, rows, cols, 'tornado_mean_table.csv')


def build_tornado_deviation_table(store: ExperimentResultStore,
                                  baseline_name: str = 'A_baseline',
                                  save_path: Optional[str] = None,
                                  metrics: Optional[List[str]] = None) -> str:
    """Tornado 表3: 参数-取值归一化偏差表 (long format)."""
    experiment_root = store.experiment_root
    save_path = save_path or os.path.join(experiment_root, 'plots', 'tornado_deviation_table.csv')
    metrics = metrics or TORNADO_PERFORMANCE_METRICS

    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        baseline_val = store.get_baseline_metric(baseline_name, metric)
        if baseline_val is None or abs(baseline_val) < 1e-12:
            continue
        deviations = store.get_knob_deviations(baseline_name, prefix='B_', metric=metric)
        if not deviations:
            continue
        for knob, vmap in deviations.items():
            for val_str, dev in vmap.items():
                rows.append({
                    'metric': metric,
                    'parameter': knob,
                    'value': val_str,
                    'baseline': baseline_val,
                    'normalized_deviation': dev,
                })

    rows.sort(key=lambda r: (r['metric'], r['parameter'], str(r['value'])))
    cols = ['metric', 'parameter', 'value', 'baseline', 'normalized_deviation']
    return _write_csv(save_path, rows, cols, 'tornado_deviation_table.csv')


def build_tornado_ranking_table(store: ExperimentResultStore,
                                baseline_name: str = 'A_baseline',
                                save_path: Optional[str] = None,
                                metrics: Optional[List[str]] = None) -> str:
    """Tornado 表4: 参数敏感度排名表 (per performance metric)."""
    experiment_root = store.experiment_root
    save_path = save_path or os.path.join(experiment_root, 'plots', 'tornado_ranking_table.csv')
    metrics = metrics or TORNADO_PERFORMANCE_METRICS

    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        baseline_val = store.get_baseline_metric(baseline_name, metric)
        if baseline_val is None or abs(baseline_val) < 1e-12:
            continue
        deviations = store.get_knob_deviations(baseline_name, prefix='B_', metric=metric)
        if not deviations:
            continue
        knob_widths = []
        for knob, vmap in deviations.items():
            devs = list(vmap.values())
            if not devs:
                continue
            mn, mx = min(devs), max(devs)
            width = mx - mn
            knob_widths.append((knob, mn, mx, width, vmap))
        knob_widths.sort(key=lambda t: t[3], reverse=True)

        for rank, (knob, mn, mx, width, vmap) in enumerate(knob_widths, 1):
            if width >= 0.30:
                sensitivity = 'high'
            elif width >= 0.10:
                sensitivity = 'medium'
            else:
                sensitivity = 'low'
            rows.append({
                'metric': metric,
                'rank': rank,
                'parameter': knob,
                'min_deviation': round(mn, 6),
                'max_deviation': round(mx, 6),
                'range_width': round(width, 6),
                'sensitivity': sensitivity,
                'perturbed_values': ';'.join(sorted(vmap.keys())),
            })

    rows.sort(key=lambda r: (r['metric'], r['rank']))
    cols = ['metric', 'rank', 'parameter', 'min_deviation', 'max_deviation',
            'range_width', 'sensitivity', 'perturbed_values']
    return _write_csv(save_path, rows, cols, 'tornado_ranking_table.csv')


def _run_tornado_postprocess(store: ExperimentResultStore,
                             baseline_name: str = 'A_baseline') -> None:
    """执行 tornado 表格、per-metric 图、composite 输出."""
    print("\n--- Tornado 数据表 ---")
    for func, kwargs in [
        (write_tornado_runs_csv, {}),
        (build_tornado_mean_table, {}),
        (build_tornado_deviation_table, {'baseline_name': baseline_name}),
        (build_tornado_ranking_table, {'baseline_name': baseline_name}),
    ]:
        try:
            func(store, **kwargs)
        except Exception as e:
            print(f"  {func.__name__} 失败: {e}")

    print("\n--- Tornado 绘图 ---")
    for metric in TORNADO_PERFORMANCE_METRICS:
        try:
            plot_tornado(store, baseline_name=baseline_name, metric=metric)
        except Exception as e:
            print(f"Tornado [{metric}] 失败: {e}")

    print("\n--- Tornado 综合排名 ---")
    try:
        plot_tornado_composite(store, baseline_name=baseline_name)
    except Exception as e:
        print(f"Tornado composite 失败: {e}")


# ============================================================
# 2. Sensitivity: 一维线搜索图 + 表格
# ============================================================
def _parse_sensitivity_name(exp_name: str) -> Optional[Tuple[str, str, float]]:
    """解析 S_<param_path>=<value> 实验名."""
    if not exp_name.startswith('S_'):
        return None
    rest = exp_name[len('S_'):]
    if '=' not in rest:
        return None
    param_path, value_str = rest.rsplit('=', 1)
    try:
        value_float = float(value_str)
    except ValueError:
        return None
    return param_path, value_str, value_float


def _list_sensitivity_experiments(store: ExperimentResultStore) -> List[Tuple[str, str, str, float]]:
    """返回解析后的 S_* 实验: (exp_name, param_path, value_str, value_float)."""
    parsed = []
    for exp_name in store.list_experiments(prefix='S_'):
        item = _parse_sensitivity_name(exp_name)
        if item is None:
            continue
        param_path, value_str, value_float = item
        parsed.append((exp_name, param_path, value_str, value_float))
    parsed.sort(key=lambda x: (x[1], x[3]))
    return parsed


def write_sensitivity_runs_csv(store: ExperimentResultStore,
                               save_path: Optional[str] = None) -> str:
    """Sensitivity 表1: 原始运行级数据 (sample × experiment × run)."""
    experiment_root = store.experiment_root
    save_path = save_path or os.path.join(experiment_root, 'plots', 'sensitivity_runs.csv')
    runner_type = _read_runner_type(experiment_root)

    rows: List[Dict[str, Any]] = []
    for exp_name, param_path, value_str, value_float in _list_sensitivity_experiments(store):
        for idx in sorted(store.list_samples(exp_name)):
            for run_dir in store.list_run_dirs(exp_name, idx):
                row = {
                    'sample_id': idx,
                    'experiment': exp_name,
                    'param_path': param_path,
                    'param_value': value_float,
                    'param_value_raw': value_str,
                    'run_idx': _get_run_idx(run_dir),
                    'algorithm': runner_type,
                }
                row.update(_read_run_metrics(run_dir))
                rows.append(row)

    rows.sort(key=lambda r: (r['param_path'], r['param_value'], r['sample_id'], r['run_idx']))
    cols = ['sample_id', 'experiment', 'param_path', 'param_value', 'param_value_raw',
            'run_idx', 'algorithm'] + CONVERGENCE_METRICS + RUN_PERFORMANCE_METRICS
    return _write_csv(save_path, rows, cols, 'sensitivity_runs.csv')


def build_sensitivity_sample_mean_table(store: ExperimentResultStore,
                                        save_path: Optional[str] = None) -> str:
    """Sensitivity 表2: 每个样本在每个参数取值下的 run 内均值."""
    experiment_root = store.experiment_root
    save_path = save_path or os.path.join(experiment_root, 'plots', 'sensitivity_sample_mean_table.csv')

    rows: List[Dict[str, Any]] = []
    metrics = CONVERGENCE_METRICS + RUN_PERFORMANCE_METRICS
    for exp_name, param_path, value_str, value_float in _list_sensitivity_experiments(store):
        exp_dir = os.path.join(experiment_root, exp_name)
        for idx in sorted(store.list_samples(exp_name)):
            agg = aggregate_runs(exp_dir, idx)
            if not agg:
                continue
            row = {
                'sample_id': idx,
                'experiment': exp_name,
                'param_path': param_path,
                'param_value': value_float,
                'param_value_raw': value_str,
            }
            n_runs = 0
            for metric in metrics:
                data = agg.get(metric, {})
                row[f'{metric}_mean'] = data.get('mean', '')
                row[f'{metric}_std'] = data.get('std', '')
                if not n_runs and data.get('n'):
                    n_runs = data.get('n')
            row['n_runs'] = n_runs
            rows.append(row)

    rows.sort(key=lambda r: (r['param_path'], r['param_value'], r['sample_id']))
    cols = ['sample_id', 'experiment', 'param_path', 'param_value', 'param_value_raw', 'n_runs']
    for metric in metrics:
        cols.extend([f'{metric}_mean', f'{metric}_std'])
    return _write_csv(save_path, rows, cols, 'sensitivity_sample_mean_table.csv')


def _collect_sensitivity_trend_rows(store: ExperimentResultStore,
                                    metrics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """统一收集 sensitivity 跨样本趋势统计, 供表3/表4/图复用."""
    metrics = metrics or SENSITIVITY_METRICS
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        param_data = store.get_sensitivity_data(prefix='S_', metric=metric)
        for param_path, value_map in param_data.items():
            for value, sample_means in value_map.items():
                vals = np.asarray(sample_means, dtype=float)
                if len(vals) == 0:
                    continue
                mu, sigma, cv_pct, ci95_half, ci95_width_ratio_pct = _calc_metric_stats(vals, len(vals))
                rows.append({
                    'metric': metric,
                    'param_path': param_path,
                    'param_value': float(value),
                    'n_samples': len(vals),
                    'cross_sample_mean': mu,
                    'cross_sample_std': sigma,
                    'mu_minus_sigma': mu - sigma,
                    'mu_plus_sigma': mu + sigma,
                    'cv_pct': cv_pct,
                    'ci95_half': ci95_half,
                    'ci95_width_ratio_pct': ci95_width_ratio_pct,
                })
    rows.sort(key=lambda r: (r['metric'], r['param_path'], r['param_value']))
    return rows


def build_sensitivity_trend_table(store: ExperimentResultStore,
                                  metrics: Optional[List[str]] = None,
                                  save_path: Optional[str] = None,
                                  trend_rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Sensitivity 表3: 跨样本 μ±σ 线图数据."""
    experiment_root = store.experiment_root
    save_path = save_path or os.path.join(experiment_root, 'plots', 'sensitivity_trend_table.csv')
    rows = trend_rows if trend_rows is not None else _collect_sensitivity_trend_rows(store, metrics)
    cols = ['metric', 'param_path', 'param_value', 'n_samples', 'cross_sample_mean',
            'cross_sample_std', 'mu_minus_sigma', 'mu_plus_sigma', 'cv_pct',
            'ci95_half', 'ci95_width_ratio_pct']
    _write_csv(save_path, rows, cols, 'sensitivity_trend_table.csv')
    return rows


def _classify_sensitivity_trend(values: List[float], means: List[float], flat_threshold: float = 0.01) -> str:
    """粗略分类 sensitivity 曲线形态."""
    if len(values) <= 1:
        return 'single_point'
    median = float(np.median(means))
    denom = max(abs(median), 1e-12)
    relative_range = (max(means) - min(means)) / denom
    if relative_range < flat_threshold:
        return 'flat'

    diffs = np.diff(means)
    tol = max(abs(median), 1.0) * 1e-6
    signs = np.where(diffs > tol, 1, np.where(diffs < -tol, -1, 0))
    nonzero = [int(s) for s in signs if s != 0]
    if not nonzero:
        return 'flat'
    if all(s >= 0 for s in nonzero):
        return 'monotonic_increasing'
    if all(s <= 0 for s in nonzero):
        return 'monotonic_decreasing'
    if 1 in nonzero and -1 in nonzero:
        first_neg = nonzero.index(-1) if -1 in nonzero else len(nonzero)
        last_pos = len(nonzero) - 1 - nonzero[::-1].index(1) if 1 in nonzero else -1
        if last_pos < first_neg:
            return 'inverted_U'
        first_pos = nonzero.index(1) if 1 in nonzero else len(nonzero)
        last_neg = len(nonzero) - 1 - nonzero[::-1].index(-1) if -1 in nonzero else -1
        if last_neg < first_pos:
            return 'U_shaped'
    return 'mixed'


def _select_optimal_interval(points: List[Tuple[float, float]],
                             direction: str,
                             tolerance_ratio: float = 0.01,
                             flat_threshold: float = 0.01) -> Dict[str, Any]:
    """根据 value→mean 趋势选择最优取值点和连续最优区间."""
    if not points:
        return {}

    points = sorted(points, key=lambda x: x[0])
    values = [p[0] for p in points]
    means = [p[1] for p in points]
    mu_min, mu_max = min(means), max(means)
    median = float(np.median(means))
    relative_range = (mu_max - mu_min) / max(abs(median), 1e-12)
    trend_type = _classify_sensitivity_trend(values, means, flat_threshold=flat_threshold)

    if relative_range < flat_threshold:
        best_idx = len(values) // 2
        selected = list(range(len(values)))
        recommendation = '曲线平坦, 参数不敏感, 可固定默认值或中位扫描值'
    else:
        if direction == 'min':
            best_idx = int(np.argmin(means))
            best_mean = means[best_idx]
            threshold = best_mean + abs(best_mean) * tolerance_ratio
            selected = [i for i, y in enumerate(means) if y <= threshold]
        else:
            best_idx = int(np.argmax(means))
            best_mean = means[best_idx]
            threshold = best_mean - abs(best_mean) * tolerance_ratio
            selected = [i for i, y in enumerate(means) if y >= threshold]
        recommendation = '按 1% 阈值选择最优连续区间'

    selected = set(selected)
    lo_idx = hi_idx = best_idx
    while lo_idx - 1 in selected:
        lo_idx -= 1
    while hi_idx + 1 in selected:
        hi_idx += 1

    if lo_idx == hi_idx and len(values) > 1:
        candidates = []
        if lo_idx > 0:
            candidates.append(lo_idx - 1)
        if hi_idx < len(values) - 1:
            candidates.append(hi_idx + 1)
        if candidates:
            best_mean = means[best_idx]
            neighbor = min(candidates, key=lambda i: abs(means[i] - best_mean))
            lo_idx = min(lo_idx, neighbor)
            hi_idx = max(hi_idx, neighbor)

    interval_means = means[lo_idx:hi_idx + 1]
    return {
        'best_value': values[best_idx],
        'best_mean': means[best_idx],
        'interval_low': values[lo_idx],
        'interval_high': values[hi_idx],
        'interval_mean_min': min(interval_means),
        'interval_mean_max': max(interval_means),
        'scan_low': values[0],
        'scan_high': values[-1],
        'scan_mean_min': mu_min,
        'scan_mean_max': mu_max,
        'relative_range': relative_range,
        'trend_type': trend_type,
        'recommendation': recommendation,
    }


def build_sensitivity_optimal_interval_table(
    store: ExperimentResultStore,
    metrics: Optional[List[str]] = None,
    trend_rows: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[str] = None,
    tolerance_ratio: float = 0.01,
    flat_threshold: float = 0.01,
) -> List[Dict[str, Any]]:
    """Sensitivity 表4: 每个参数每个 metric 的最优取值区间."""
    experiment_root = store.experiment_root
    save_path = save_path or os.path.join(experiment_root, 'plots', 'sensitivity_optimal_interval_table.csv')
    trend_rows = trend_rows if trend_rows is not None else _collect_sensitivity_trend_rows(store, metrics)

    grouped: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for row in trend_rows:
        grouped.setdefault((row['metric'], row['param_path']), []).append(
            (float(row['param_value']), float(row['cross_sample_mean']))
        )

    rows: List[Dict[str, Any]] = []
    for (metric, param_path), points in sorted(grouped.items()):
        direction = METRIC_DIRECTIONS.get(metric, 'max')
        interval = _select_optimal_interval(points, direction, tolerance_ratio, flat_threshold)
        if not interval:
            continue
        rows.append({
            'metric': metric,
            'param_path': param_path,
            'direction': direction,
            'scan_range': _format_range(interval['scan_low'], interval['scan_high']),
            'optimal_interval': _format_range(interval['interval_low'], interval['interval_high']),
            'best_value': interval['best_value'],
            'best_mean': interval['best_mean'],
            'interval_low': interval['interval_low'],
            'interval_high': interval['interval_high'],
            'interval_mean_min': interval['interval_mean_min'],
            'interval_mean_max': interval['interval_mean_max'],
            'scan_mean_min': interval['scan_mean_min'],
            'scan_mean_max': interval['scan_mean_max'],
            'relative_range': interval['relative_range'],
            'trend_type': interval['trend_type'],
            'recommendation': interval['recommendation'],
        })

    cols = ['metric', 'param_path', 'direction', 'scan_range', 'optimal_interval',
            'best_value', 'best_mean', 'interval_low', 'interval_high',
            'interval_mean_min', 'interval_mean_max', 'scan_mean_min', 'scan_mean_max',
            'relative_range', 'trend_type', 'recommendation']
    _write_csv(save_path, rows, cols, 'sensitivity_optimal_interval_table.csv')
    return rows


def build_sensitivity_metric_conflict_table(store: ExperimentResultStore,
                                            optimal_rows: Optional[List[Dict[str, Any]]] = None,
                                            save_path: Optional[str] = None) -> str:
    """Sensitivity 表5: 同一参数在不同 metric 下的推荐区间冲突分析."""
    experiment_root = store.experiment_root
    save_path = save_path or os.path.join(experiment_root, 'plots', 'sensitivity_metric_conflict_table.csv')
    optimal_rows = optimal_rows or []

    by_param: Dict[str, List[Dict[str, Any]]] = {}
    for row in optimal_rows:
        by_param.setdefault(row['param_path'], []).append(row)

    rows: List[Dict[str, Any]] = []
    for param_path, items in sorted(by_param.items()):
        lows = [float(item['interval_low']) for item in items if item.get('interval_low') != '']
        highs = [float(item['interval_high']) for item in items if item.get('interval_high') != '']
        has_overlap = bool(lows and highs and max(lows) <= min(highs))
        conflict_flag = 'no' if has_overlap else 'yes'
        conflict_note = 'recommended intervals overlap across metrics' if has_overlap else 'recommended intervals do not overlap across metrics'
        for item in items:
            rows.append({
                'param_path': param_path,
                'metric': item['metric'],
                'direction': item['direction'],
                'best_value': item['best_value'],
                'best_mean': item['best_mean'],
                'optimal_interval': item['optimal_interval'],
                'trend_type': item['trend_type'],
                'conflict_flag': conflict_flag,
                'conflict_note': conflict_note,
            })

    cols = ['param_path', 'metric', 'direction', 'best_value', 'best_mean',
            'optimal_interval', 'trend_type', 'conflict_flag', 'conflict_note']
    return _write_csv(save_path, rows, cols, 'sensitivity_metric_conflict_table.csv')


def plot_sensitivity_lines(store: ExperimentResultStore,
                           metric: str = 'final_best',
                           save_path: Optional[str] = None) -> None:
    """一维线搜索图: 扫描 S_* 实验, x=参数取值, y=metric 均值 ± 标准差."""
    experiment_root = store.experiment_root
    save_path = save_path or os.path.join(experiment_root, 'plots', f'sensitivity_{metric}.png')
    ensure_folder(os.path.dirname(save_path))

    param_data = store.get_sensitivity_data(prefix='S_', metric=metric)
    if not param_data:
        print(f"警告: 未找到 S_* 实验, 跳过 sensitivity 绘图 [metric={metric}]")
        return

    n_params = len(param_data)
    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)

    for ax, (param_path, vmap) in zip(axes.flat, sorted(param_data.items())):
        x_vals = sorted(vmap.keys())
        y_means = [np.mean(vmap[x]) for x in x_vals]
        y_stds = [np.std(vmap[x], ddof=1) if len(vmap[x]) > 1 else 0.0 for x in x_vals]
        ax.errorbar(x_vals, y_means, yerr=y_stds, marker='o', capsize=4, linestyle='-')
        ax.set_xlabel(param_path, fontsize=9)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(f'Sensitivity: {param_path}', fontsize=10)
        ax.grid(linestyle='--', alpha=0.4)

    for j in range(n_params, len(axes.flat)):
        axes.flat[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Sensitivity plot saved: {save_path}")


def run_sensitivity_plots(store: ExperimentResultStore,
                          metrics: Optional[List[str]] = None) -> None:
    """批量生成 sensitivity 一维线图."""
    metrics = metrics or SENSITIVITY_METRICS
    for metric in metrics:
        try:
            plot_sensitivity_lines(store, metric=metric)
        except Exception as e:
            print(f"Sensitivity [{metric}] 失败: {e}")


def run_sensitivity_postprocess(experiment_root: str) -> None:
    """Sensitivity 后处理总入口: 表1-表5 + 多指标线图."""
    store = ExperimentResultStore(experiment_root)
    if not _list_sensitivity_experiments(store):
        print("警告: 未找到 S_* sensitivity 实验目录, 跳过 sensitivity 后处理")
        return

    print("\n--- Sensitivity 数据表 ---")
    for func in [write_sensitivity_runs_csv, build_sensitivity_sample_mean_table]:
        try:
            func(store)
        except Exception as e:
            print(f"  {func.__name__} 失败: {e}")

    trend_rows: List[Dict[str, Any]] = []
    try:
        trend_rows = _collect_sensitivity_trend_rows(store, SENSITIVITY_METRICS)
        build_sensitivity_trend_table(store, trend_rows=trend_rows)
    except Exception as e:
        print(f"  sensitivity_trend_table 失败: {e}")

    optimal_rows: List[Dict[str, Any]] = []
    try:
        optimal_rows = build_sensitivity_optimal_interval_table(store, trend_rows=trend_rows)
    except Exception as e:
        print(f"  sensitivity_optimal_interval_table 失败: {e}")

    try:
        build_sensitivity_metric_conflict_table(store, optimal_rows=optimal_rows)
    except Exception as e:
        print(f"  sensitivity_metric_conflict_table 失败: {e}")

    print("\n--- Sensitivity 绘图 ---")
    run_sensitivity_plots(store, SENSITIVITY_METRICS)


# ============================================================
# 3. 后处理 mode 分流入口
# ============================================================
def _resolve_postprocess_mode(store: ExperimentResultStore, mode: str = 'auto') -> List[str]:
    """解析 tornado/sensitivity 后处理模式."""
    mode = (mode or 'auto').lower()
    if mode == 'tornado':
        return ['tornado']
    if mode == 'sensitivity':
        return ['sensitivity']
    if mode != 'auto':
        raise ValueError(f"未知 tornado_plots mode: {mode}; 可选 tornado/sensitivity/auto")

    summary_mode = str(_read_summary(store.experiment_root).get('mode', '')).lower()
    if summary_mode in ('tornado', 'sensitivity'):
        return [summary_mode]

    names = store.list_experiments()
    has_sensitivity = any(name.startswith('S_') for name in names)
    has_tornado = 'A_baseline' in names or any(name.startswith('B_') for name in names)
    if has_sensitivity and not has_tornado:
        return ['sensitivity']
    if has_tornado and not has_sensitivity:
        return ['tornado']
    if has_tornado and has_sensitivity:
        print("警告: 同时发现 tornado 与 sensitivity 目录, auto 模式将同时后处理")
        return ['tornado', 'sensitivity']
    print("警告: 未发现 A_baseline/B_* 或 S_* 实验目录, 无后处理分支可执行")
    return []


def run_tornado_plots(experiment_root: str,
                      baseline_name: str = 'A_baseline',
                      mode: str = 'auto') -> None:
    """批量生成 tornado/sensitivity 后处理结果.

    Parameters
    ----------
    experiment_root : str
        实验结果根目录.
    baseline_name : str
        tornado 基线实验名.
    mode : {'tornado', 'sensitivity', 'auto'}
        后处理模式. auto 优先读取 experiment_summary.json 中的 mode.
    """
    store = ExperimentResultStore(experiment_root)
    branches = _resolve_postprocess_mode(store, mode)
    if 'tornado' in branches:
        _run_tornado_postprocess(store, baseline_name=baseline_name)
    if 'sensitivity' in branches:
        run_sensitivity_postprocess(experiment_root)
