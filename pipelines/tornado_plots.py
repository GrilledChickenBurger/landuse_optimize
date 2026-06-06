# tornado / sensitivity 实验结果可视化: tornado 图 + 一维线图
import os
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from utils.io_utils import ensure_folder
from utils.exp_sweep_utils import ExperimentResultStore


# ============================================================
# 1. Tornado: B_* 相对 A_baseline 的扰动
# ============================================================
def plot_tornado(store: ExperimentResultStore,
                 baseline_name: str = 'A_baseline',
                 save_path: Optional[str] = None,
                 metric: str = 'final_best') -> Dict[str, Dict[str, float]]:
    """画 tornado 图: 每个 B_* knob 相对 baseline 的归一化扰动幅度.

    Parameters
    ----------
    store : ExperimentResultStore
        实验结果存储对象
    baseline_name : str
        baseline 实验名, 默认 'A_baseline'
    save_path : str
        保存路径; None 则保存到 experiment_root/plots/tornado.png
    metric : str
        'final_best' 或 performance.json 的嵌套路径

    Returns
    -------
    Dict[str, Dict[str, float]]
        原始偏差数据 {knob: {value_str: deviation}}.
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

    # 按区间宽度排序
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
    ax.set_title(f'Tornado: sensitivity to magic numbers\nbaseline={baseline_name} ({metric}={baseline_val:.4f})')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Tornado saved: {save_path}")

    summary_path = os.path.join(os.path.dirname(save_path), f'tornado_{metric}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        import json
        json.dump({'baseline': baseline_val, 'metric': metric, 'results': results}, f, indent=2, ensure_ascii=False)
    return results


def run_tornado_plots(experiment_root: str, baseline_name: str = 'A_baseline') -> None:
    """批量生成 tornado 图: 对多个 metric 分别调用 plot_tornado."""
    store = ExperimentResultStore(experiment_root)
    metrics = [
        'final_best',
        'iter_to_95pct',
        'wall_time_sec',
        'ES.food.best_ori_diff',
        'ES.carbon.best_ori_diff',
        'ES.water.best_ori_diff',
        'ES.culture.best_ori_diff',
    ]
    for metric in metrics:
        try:
            plot_tornado(store, baseline_name=baseline_name, metric=metric)
        except Exception as e:
            print(f"Tornado [{metric}] 失败: {e}")


# ============================================================
# 2. Sensitivity: 一维线搜索图
# ============================================================
def plot_sensitivity_lines(store: ExperimentResultStore,
                           metric: str = 'final_best',
                           save_path: Optional[str] = None) -> None:
    """一维线搜索图: 扫描 S_* 实验, x=参数取值, y=metric 均值 ± 标准差.

    Parameters
    ----------
    store : ExperimentResultStore
        实验结果存储对象
    metric : str
        指标名
    save_path : str, optional
        保存路径; None 则保存到 experiment_root/plots/sensitivity_<metric>.png
    """
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
