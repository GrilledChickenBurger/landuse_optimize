# ablation 实验结果可视化: convergence / LU 并排 / 稳定性叠图 / A/B/C 对比
import os
import json
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

from libs.const import COLOR_MAP

# ── 全局字体: 英文 Times New Roman Bold ──────────────────────────
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
# ──────────────────────────────────────────────────────────────────


# Q1-Q5 背景色映射 (浅色, 从冷到暖)
_LEVEL_COLORS = {
    'Q1': '#deebf7',  # 浅蓝
    'Q2': '#e5f5e0',  # 浅绿
    'Q3': '#fff7bc',  # 浅黄
    'Q4': '#fed9a6',  # 浅橙
    'Q5': '#fcbba1',  # 浅红
}
_LEVEL_ORDER = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

from utils.tools import ensure_folder, visualize_tensor_list
from utils.exp_sweep_utils import (
    ABLATION_GROUPS,
    build_stability_table, build_ablation_compare_table,
    build_ablation_mean_table, build_ablation_improvement_table, build_ablation_summary_table,
    ExperimentResultStore,
    merge_stability_runs, build_stability_per_sample_csv, build_stability_summary_csv,
)


# ============================================================
# 1. 收敛曲线: A_guided vs B_unguided (旧功能, 保留兼容)
# ============================================================
def plot_convergence(experiment_root: str,
                     save_folder: Optional[str] = None,
                     smooth_window: int = 20) -> None:
    """对每个样本画 A_guided vs B_unguided 的 total_score 收敛曲线 (滑动平均).
    每个样本可能有多 run, 此处取 run_0 的曲线作为代表.
    """
    save_folder = save_folder or os.path.join(experiment_root, 'plots')
    ensure_folder(save_folder)

    store = ExperimentResultStore(experiment_root)
    samples_g = set(store.list_samples('A_guided'))
    samples_u = set(store.list_samples('B_unguided'))
    common = sorted(samples_g & samples_u)
    if not common:
        print(f"警告: A_guided / B_unguided 无共同样本, 跳过")
        return

    def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
        if w <= 1 or len(arr) <= w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='valid')

    for idx in common:
        g_runs = store.list_run_dirs('A_guided', idx)
        u_runs = store.list_run_dirs('B_unguided', idx)
        if not (g_runs and u_runs):
            continue
        g_npy = os.path.join(g_runs[0], 'total_score_list.npy')
        u_npy = os.path.join(u_runs[0], 'total_score_list.npy')
        if not (os.path.exists(g_npy) and os.path.exists(u_npy)):
            continue
        g_arr = np.load(g_npy)
        u_arr = np.load(u_npy)
        g_sm = _smooth(g_arr, smooth_window)
        u_sm = _smooth(u_arr, smooth_window)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.arange(len(g_sm)), g_sm, label='guided', color='tab:blue')
        ax.plot(np.arange(len(u_sm)), u_sm, label='unguided', color='tab:orange')
        g_conv_path = os.path.join(g_runs[0], 'convergence.json')
        if os.path.exists(g_conv_path):
            with open(g_conv_path, 'r', encoding='utf-8') as f:
                g_best = json.load(f).get('final_best')
            if g_best is not None:
                target = float(g_best) * 0.95 if float(g_best) < 0 else float(g_best) * 1.05
                ax.axhline(target, linestyle='--', color='gray', alpha=0.6, label=f'95% best={target:.3f}')
        ax.set_xlabel('iteration (smoothed)')
        ax.set_ylabel('total_score (lower=better)')
        ax.set_title(f'Convergence: {idx}')
        ax.legend()
        ax.grid(linestyle='--', alpha=0.4)
        fig.tight_layout()
        out = os.path.join(save_folder, f'convergence_{idx}.png')
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Convergence saved: {out}")


# ============================================================
# 2. LU 并排: ori / A_guided.best / B_unguided.best (旧功能, 保留兼容)
# ============================================================
def plot_lu_side_by_side(experiment_root: str,
                         index: str,
                         save_folder: Optional[str] = None) -> None:
    """对指定样本画 ori_plan / guided best / unguided best 并排图."""
    from utils.dataset_utils import remove_lines  # noqa
    import rasterio

    save_folder = save_folder or os.path.join(experiment_root, 'plots')
    ensure_folder(save_folder)
    idx_num = index.split('_')[1] if '_' in index else index

    store = ExperimentResultStore(experiment_root)

    def _read_tif(path: str) -> Optional[torch.Tensor]:
        if not os.path.exists(path):
            return None
        try:
            with rasterio.open(path) as src:
                arr = src.read(1)
            return torch.tensor(arr, dtype=torch.float32)
        except Exception as e:
            print(f"读取 {path} 失败: {e}")
            return None

    def _first_run_plan_dir(exp_name: str) -> Optional[str]:
        runs = store.list_run_dirs(exp_name, index)
        return os.path.join(runs[0], f'plans_{idx_num}') if runs else None

    g_plan_dir = _first_run_plan_dir('A_guided')
    u_plan_dir = _first_run_plan_dir('B_unguided')
    ori = _read_tif(os.path.join(g_plan_dir, 'ori_plan.TIF')) if g_plan_dir else None
    g_best = _read_tif(os.path.join(g_plan_dir, 'best_plan.TIF')) if g_plan_dir else None
    u_best = _read_tif(os.path.join(u_plan_dir, 'best_plan.TIF')) if u_plan_dir else None

    tensor_dict = {}
    if ori is not None:
        tensor_dict['ori_plan'] = ori
    if g_best is not None:
        tensor_dict['guided_best'] = g_best
    if u_best is not None:
        tensor_dict['unguided_best'] = u_best
    if not tensor_dict:
        print(f"警告: 样本 {index} 无可读 TIF, 跳过")
        return

    save_path = os.path.join(save_folder, f'lu_compare_{index}.png')
    visualize_tensor_list(tensor_dict,
                          main_title=f'LU compare: {index}',
                          save_path=save_path,
                          color_map=COLOR_MAP,
                          figsize=(15, 5))
    print(f"LU compare saved: {save_path}")


# ============================================================
# 主入口 (旧一站式, 保留向后兼容)
# ============================================================
def run_all_plots(experiment_root: str,
                  baseline_name: str = 'A_guided',
                  indexes_for_lu: Optional[List[str]] = None) -> None:
    """[deprecated] 旧一站式入口: 跑 convergence + LU 并排."""
    try:
        plot_convergence(experiment_root)
    except Exception as e:
        print(f"convergence 失败: {e}")

    if indexes_for_lu:
        for idx in indexes_for_lu:
            try:
                plot_lu_side_by_side(experiment_root, idx)
            except Exception as e:
                print(f"LU compare [{idx}] 失败: {e}")


# ============================================================
# 多次重复稳定性与 A/B/C 消融可视化
# ============================================================
def plot_stability_convergence(store: ExperimentResultStore,
                               exp_name: str,
                               index: str,
                               save_path: str,
                               smooth_window: int = 50) -> None:
    """单个样本的多次运行收敛曲线叠图: 半透明 + 均值粗线 + ±1 std 阴影."""
    series = store.get_convergence_series(exp_name, index, smooth_window=smooth_window)
    if series is None or series.shape[0] < 2:
        print(f"  [stability_conv] {exp_name}/{index} run 数 <2, 跳过")
        return

    mean = series.mean(axis=0)
    std = series.std(axis=0, ddof=1) if series.shape[0] >= 2 else np.zeros_like(mean)
    x = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=(9, 5))
    for s in series:
        ax.plot(x, s, color='steelblue', alpha=0.3, linewidth=0.8)
    ax.plot(x, mean, color='navy', linewidth=2.0, label=f'mean (n={series.shape[0]})')
    ax.fill_between(x, mean - std, mean + std, color='steelblue', alpha=0.2, label='±1 std')
    ax.set_xlabel('iteration (smoothed)')
    ax.set_ylabel('total_score (lower=better)')
    ax.set_title(f'Stability: {index} (n={series.shape[0]} runs)')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    ensure_folder(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  stability_convergence saved: {save_path}")


def plot_stability_convergence_all_samples(store: ExperimentResultStore,
                                            exp_name: str,
                                            save_path: str,
                                            smooth_window: int = 50,
                                            runner_type: str = 'SA') -> None:
    """跨样本聚合稳定性收敛曲线: 同一张图中, 每个样本一条均值线 + ±1 std 阴影.

    SA 直接画 total_score; GA 自动取负, 统一为 "lower=better" 语义.
    """
    stability_data = store.get_stability_series(exp_name, smooth_window=smooth_window)
    if not stability_data:
        print(f"  [stability_conv_all] {exp_name} 无足够样本, 跳过")
        return

    sign = -1.0 if runner_type.upper() == 'GA' else 1.0
    per_sample = []
    for idx, series in stability_data.items():
        signed = series * sign
        mean = signed.mean(axis=0)
        std = signed.std(axis=0, ddof=1)
        per_sample.append((idx, mean, std, series.shape[0]))

    cmap = plt.get_cmap('tab10')
    n_runs_list = sorted({n for _, _, _, n in per_sample})
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (idx, mean, std, n) in enumerate(per_sample):
        color = cmap(i % 10)
        x = np.arange(len(mean))
        ax.plot(x, mean, color=color, linewidth=1.5, label=f'{idx} (n={n})')
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_xlabel('iteration (smoothed)')
    ax.set_ylabel('total_score (lower=better)')
    group_name = exp_name
    n_runs_str = ','.join(str(n) for n in n_runs_list)
    ax.set_title(f'Stability across samples: {group_name} '
                 f'({runner_type.upper()}, {len(per_sample)} samples × {n_runs_str} runs)')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    ensure_folder(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  stability_convergence_all saved: {save_path}")


def plot_ablation_convergence(store: ExperimentResultStore,
                              idx: str,
                              save_path: str,
                              smooth_window: int = 50) -> None:
    """A/B/C 三组单样本的平均收敛曲线对比, 标注 iter_to_95pct_repeat 竖线."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for g, label, _, color, ls in ABLATION_GROUPS:
        series = store.get_convergence_series(g, idx, smooth_window=smooth_window)
        if series is None:
            continue
        mean = series.mean(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, color=color, linestyle=ls, linewidth=2.0, label=label)

        agg = store.recompute_and_aggregate(g, idx)
        iter_repeat = agg.get('iter_to_95pct_repeat', {}).get('mean')
        if iter_repeat is not None and iter_repeat > 0:
            ax.axvline(iter_repeat, color=color, linestyle=ls, alpha=0.4, linewidth=1.0)

    ax.set_xlabel('iteration (smoothed)')
    ax.set_ylabel('total_score (lower=better)')
    ax.set_title(f'Ablation convergence A/B/C: {idx}')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    ensure_folder(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ablation_convergence saved: {save_path}")


def plot_boxplot_final(store: ExperimentResultStore,
                       save_path: str,
                       metric: str = 'final_best') -> None:
    """5 样本 × A/B/C 最终 F(S) 箱线图."""
    from utils.visualize_utils import grouped_box_plot_figure

    sub = store.get_final_best_distribution()
    if not sub:
        print(f"  [boxplot] 无数据, 跳过")
        return

    data = {f'{metric} by sample × group': sub}
    ensure_folder(os.path.dirname(save_path))
    grouped_box_plot_figure(
        data=data,
        title=f'Ablation final F(S) distribution',
        y_label=metric,
        save_path=save_path,
        figsize=(max(10, 2 * len(sub)), 6),
    )
    print(f"  boxplot saved: {save_path}")


def plot_stability_cv_line(per_sample_csv_or_df,
                            save_path: str,
                            metric: str = 'F(S)',
                            figsize=(12, 5)) -> None:
    """CV 散点图: N 个样本按 n_pred 升序, SA vs GA 两组散点.

    依照 stability.md 5.2: 无连线, 加 5% 参考虚线.
    X 轴用 S1..SN 短标签; 不同 level (Q1-Q5) 有不同背景色.

    Parameters
    ----------
    per_sample_csv_or_df : str or pd.DataFrame
        stability_per_sample.csv 路径或 DataFrame.
    save_path : str
        输出 PNG 路径.
    metric : str
        'F(S)' / 'ES' / 'Runtime'.
    figsize : tuple
        图片尺寸.
    """
    import pandas as pd

    if isinstance(per_sample_csv_or_df, str):
        if not os.path.isfile(per_sample_csv_or_df):
            print(f"  [cv_scatter] CSV 不存在: {per_sample_csv_or_df}, 跳过")
            return
        df = pd.read_csv(per_sample_csv_or_df)
    else:
        df = per_sample_csv_or_df

    # 根据指标选列和标签
    if metric == 'F(S)':
        cv_col = 'F_cv_pct'
        ylabel = r'CV$_{F(S)}$ (%)'
        title = r'SA vs GA stability: CV of $F(S)$ across samples'
    elif metric == 'ES':
        cv_col = 'ES_cv_pct'
        ylabel = r'CV$_{\mathrm{ES}}$ (%)'
        title = r'SA vs GA stability: CV of ES across samples'
    else:
        cv_col = 'RT_cv_pct'
        ylabel = r'CV$_{\mathrm{Runtime}}$ (%)'
        title = r'SA vs GA stability: CV of Runtime across samples'

    if df.empty or 'algorithm' not in df.columns or cv_col not in df.columns:
        print(f"  [cv_scatter] 缺 algorithm 或 {cv_col} 列, 跳过")
        return

    # 按 n_pred 升序, 每个 algorithm 各自独立排序
    sa = df[df['algorithm'].str.upper() == 'SA'].sort_values('n_pred').copy()
    ga = df[df['algorithm'].str.upper() == 'GA'].sort_values('n_pred').copy()

    if sa.empty and ga.empty:
        print(f"  [cv_scatter] 无有效数据, 跳过")
        return

    # 用 SA 顺序作为 x 轴骨架 (若 SA 为空则用 GA)
    primary = sa if not sa.empty else ga
    n = len(primary)
    x_positions = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)
    x_label = r'Sample (sorted by $n_{\mathrm{pred}}$)'
    markersize = 9

    # ── 按 level 画背景色带 ──────────────────────────────────────
    has_level = 'level' in primary.columns
    if has_level:
        for i in range(n):
            lv = primary['level'].iloc[i]
            color = _LEVEL_COLORS.get(str(lv), '#f5f5f5')
            ax.axvspan(i - 0.5, i + 0.5, facecolor=color, edgecolor='none',
                       zorder=0)

    # ── SA 散点 ──────────────────────────────────────────────────
    if not sa.empty:
        # SA 样本按 n_pred 排序后的位置
        sa_order = np.arange(len(sa))
        ax.scatter(sa_order, sa[cv_col].values,
                   color='tab:blue', label='SA', s=markersize**2,
                   edgecolors='navy', linewidth=0.5, zorder=4)

    # ── GA 散点 ──────────────────────────────────────────────────
    if not ga.empty:
        ga_order = np.arange(len(ga))
        ax.scatter(ga_order, ga[cv_col].values,
                   color='tab:orange', label='GA', s=markersize**2,
                   marker='s', edgecolors='darkorange', linewidth=0.5, zorder=4)

    # ── X 轴: S1..SN 短标签 ──────────────────────────────────────
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'S{i + 1}' for i in range(n)],
                       rotation=0, ha='center', fontsize=9)

    # ── Level 图例 ───────────────────────────────────────────────
    if has_level:
        from matplotlib.patches import Patch
        present_levels = [lv for lv in _LEVEL_ORDER
                          if str(lv) in primary['level'].values]
        if present_levels:
            legend_patches = [Patch(facecolor=_LEVEL_COLORS[lv], edgecolor='#cccccc',
                                    label=f'{lv} (n_pred ↑)') for lv in present_levels]
            leg1 = ax.legend(handles=legend_patches, loc='upper left',
                             title='Level', fontsize=8, title_fontsize=9,
                             framealpha=0.9)
            ax.add_artist(leg1)

    # ── SA/GA 图例 ───────────────────────────────────────────────
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.7, n - 0.3)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # CV=5% 阈值水平线
    ax.axhline(y=5.0, color='gray', linestyle=':', alpha=0.6, linewidth=1.0)
    ax.annotate('5%', xy=(0.98, 5.0), ha='right', va='bottom', fontsize=8,
                color='gray', xycoords=('axes fraction', 'data'))

    fig.tight_layout()
    ensure_folder(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  stability_cv_scatter saved: {save_path}")


# ============================================================
# stability 箱线图: 展示 N 个样本的 CV 分布 (stability.md §5.1)
# ============================================================
def plot_stability_boxplot(per_sample_csv_or_df,
                            save_path: str,
                            show_runtime: bool = True,
                            figsize=(10, 5)) -> None:
    """stability 箱线图: 横轴为指标 (F(S) / ES / 可选 Runtime), 每指标 SA vs GA 箱线, 叠加散点.

    数据来源为 stability_per_sample.csv (build_stability_per_sample_csv 的输出),
    箱线内是 N 个样本在各自指标上的 CV 值, 直观展示跨样本变异系数分布。

    Parameters
    ----------
    per_sample_csv_or_df : str or pd.DataFrame
        stability_per_sample.csv 路径或 DataFrame, 需含 F_cv_pct / ES_cv_pct / RT_cv_pct / algorithm 列.
    save_path : str
        输出 PNG 路径.
    show_runtime : bool
        是否在箱线图中显示 Runtime 维度, 默认 True.
    figsize : tuple
        图片尺寸.
    """
    import pandas as pd

    if isinstance(per_sample_csv_or_df, str):
        if not os.path.isfile(per_sample_csv_or_df):
            print(f"  [boxplot] CSV 不存在: {per_sample_csv_or_df}, 跳过")
            return
        df = pd.read_csv(per_sample_csv_or_df)
    else:
        df = per_sample_csv_or_df

    if df.empty:
        print(f"  [boxplot] 空数据, 跳过")
        return

    # 提取 SA / GA 子集
    sa = df[df['algorithm'].str.upper() == 'SA']
    ga = df[df['algorithm'].str.upper() == 'GA']

    sa_f_cv = sa['F_cv_pct'].astype(float).dropna().values
    ga_f_cv = ga['F_cv_pct'].astype(float).dropna().values
    sa_es_cv = sa['ES_cv_pct'].astype(float).dropna().values if 'ES_cv_pct' in sa.columns else np.array([])
    ga_es_cv = ga['ES_cv_pct'].astype(float).dropna().values if 'ES_cv_pct' in ga.columns else np.array([])
    sa_rt_cv = sa['RT_cv_pct'].astype(float).dropna().values
    ga_rt_cv = ga['RT_cv_pct'].astype(float).dropna().values

    # 动态构建指标列表
    metrics_info = []
    metrics_info.append(('F(S)', sa_f_cv, ga_f_cv, '综合目标函数变异系数'))
    if sa_es_cv.size > 0 or ga_es_cv.size > 0:
        metrics_info.append(('ES', sa_es_cv, ga_es_cv, 'ES变异系数'))
    if show_runtime:
        metrics_info.append(('Runtime', sa_rt_cv, ga_rt_cv, '运行时间变异系数'))

    n_metrics = len(metrics_info)
    x_positions = list(range(n_metrics))

    # 样本数取各指标最大值
    n_samples = max(max(len(m[1]), len(m[2])) for m in metrics_info)

    # 箱线位置: SA 在左 (-0.18), GA 在右 (+0.18)
    sa_offset = -0.18
    ga_offset = +0.18
    sa_positions = [x + sa_offset for x in x_positions]
    ga_positions = [x + ga_offset for x in x_positions]

    sa_data = [m[1] for m in metrics_info]
    ga_data = [m[2] for m in metrics_info]

    fig, ax = plt.subplots(figsize=figsize)

    bp_sa = ax.boxplot(
        sa_data, positions=sa_positions, widths=0.28,
        patch_artist=True,
        boxprops=dict(facecolor='tab:blue', alpha=0.45),
        medianprops=dict(color='navy', linewidth=1.5),
        flierprops=dict(marker='', markersize=0),  # 隐藏离群点, 由散点替代
    )
    bp_ga = ax.boxplot(
        ga_data, positions=ga_positions, widths=0.28,
        patch_artist=True,
        boxprops=dict(facecolor='tab:orange', alpha=0.45),
        medianprops=dict(color='darkorange', linewidth=1.5),
        flierprops=dict(marker='', markersize=0),
    )

    # 叠加散点 (每个样本一个点)
    rng = np.random.RandomState(42)
    for pos, vals in zip(sa_positions, sa_data):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.06, 0.06, len(vals))
        ax.scatter(np.full_like(vals, pos, dtype=float) + jitter, vals,
                   color='tab:blue', alpha=0.7, s=30, zorder=3, edgecolors='white', linewidth=0.3)
    for pos, vals in zip(ga_positions, ga_data):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.06, 0.06, len(vals))
        ax.scatter(np.full_like(vals, pos, dtype=float) + jitter, vals,
                   color='tab:orange', alpha=0.7, s=30, zorder=3, edgecolors='white', linewidth=0.3)

    # 5% 参考线
    ax.axhline(y=5.0, color='gray', linestyle=':', alpha=0.6, linewidth=1.0)
    ax.annotate('5%', xy=(0.98, 5.0), ha='right', va='bottom', fontsize=9,
                color='gray', xycoords=('axes fraction', 'data'))

    # x 轴: 中文标签, 水平不旋转
    x_labels = [m[3] for m in metrics_info]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=11)
    ax.set_ylabel('变异系数 CV (%)')
    ax.set_title(f'SA vs GA stability: CV distribution across {n_samples} samples')

    # 图例 (手动构造)
    ax.legend(
        [bp_sa['boxes'][0], bp_ga['boxes'][0]],
        ['SA', 'GA'],
        loc='upper right',
    )
    ax.grid(linestyle='--', alpha=0.4, axis='y')

    fig.tight_layout()
    ensure_folder(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  stability_boxplot saved: {save_path}")


# ------------------------------------------------------------
# 模式选择器
# ------------------------------------------------------------
def _resolve_ablation_mode(experiment_root: str, mode: str = 'auto') -> List[str]:
    """解析后处理模式, 返回激活的分支列表.

    Parameters
    ----------
    experiment_root : str
        实验结果目录.
    mode : str
        'stability' / 'ablation' / 'auto' (默认).

    Returns
    -------
    list of str
        激活分支: ['stability'], ['ablation'] 或 ['stability', 'ablation'].
    """
    mode = (mode or 'auto').lower()
    if mode == 'stability':
        return ['stability']
    if mode == 'ablation':
        return ['ablation']
    if mode != 'auto':
        raise ValueError(f"Unknown ablation_plots mode: {mode}; "
                         f"choose stability/ablation/auto")

    # auto: 优先读 experiment_summary.json 的 mode 字段
    summary_path = os.path.join(experiment_root, 'experiment_summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            summary_mode = str(summary.get('mode', '')).lower()
            if summary_mode == 'stability':
                return ['stability']
            if summary_mode in ('ablation', 'full'):
                return ['stability', 'ablation']
        except Exception:
            pass

    # fallback: 扫目录结构
    has_B = os.path.isdir(os.path.join(experiment_root, 'B_unguided'))
    has_C = os.path.isdir(os.path.join(experiment_root, 'C_random'))
    has_A = os.path.isdir(os.path.join(experiment_root, 'A_guided'))

    if has_B or has_C:
        return ['stability', 'ablation']
    if has_A:
        return ['stability']
    return ['stability']


def _build_combined_folder_path(experiment_root: str,
                                 sibling_experiment_roots: List[str]) -> Optional[str]:
    """构建 SA+GA 合并输出目录路径.

    合并目录位于所有 experiment_root 同级:
    ``<parent>/<root1>+<root2>_combined/``

    Returns
    -------
    str or None
        合并目录路径; sibling 为空时返回 None.
    """
    if not sibling_experiment_roots:
        return None

    all_roots = [experiment_root] + list(sibling_experiment_roots)
    normalized_roots = [root.rstrip('/\\') for root in all_roots]
    parents = {os.path.abspath(os.path.dirname(root)) for root in normalized_roots}
    if len(parents) != 1:
        raise ValueError(f"合并目录要求所有 experiment_root 位于同一父目录: {all_roots}")

    parent = os.path.dirname(normalized_roots[0])
    root_names = [os.path.basename(root) for root in normalized_roots]
    combined_name = f"{'+'.join(root_names)}_combined"
    return os.path.join(parent, combined_name)


def _run_combined_stability(experiment_root: str,
                             sibling_experiment_roots: List[str],
                             show_runtime: bool = True,
                             mode: str = 'auto') -> None:
    """SA+GA 合并后处理: 产出合并目录下的 CSV + 图表 + combined_meta.json.

    Parameters
    ----------
    experiment_root : str
        主 (通常为 SA) 的 ablation 结果目录.
    sibling_experiment_roots : list of str
        其他算法 (如 GA) 的 ablation 结果目录列表.
    show_runtime : bool
        箱线图 / 折线图是否包含 Runtime 维度.
    mode : str
        模式; ablation-only 时跳过 stability 合并.
    """
    import pandas as pd
    import datetime

    branches = _resolve_ablation_mode(experiment_root, mode)
    if 'stability' not in branches:
        print(f"  [combined] mode={mode} 未激活 stability 分支, 跳过合并")
        return

    combined_root = _build_combined_folder_path(experiment_root, sibling_experiment_roots)
    if combined_root is None:
        return

    combined_plots_dir = os.path.join(combined_root, 'plots')
    ensure_folder(combined_plots_dir)

    # 合并阶段只读取已完成的单算法输出, 不再隐式补跑 run_all_ablation_plots
    all_roots = [experiment_root] + list(sibling_experiment_roots)
    runs_csv_list = [os.path.join(root, 'plots', 'stability_runs.csv') for root in all_roots]
    missing = [p for p in runs_csv_list if not os.path.exists(p)]
    if missing:
        print(f"  [combined] 缺少单算法 stability_runs.csv, 跳过合并: {missing}")
        return

    existing = runs_csv_list

    try:
        df_runs = merge_stability_runs(existing)

        # 保存合并后的 runs.csv
        combined_runs_path = os.path.join(combined_plots_dir, 'stability_runs.csv')
        df_runs.to_csv(combined_runs_path, index=False)

        # per_sample / summary
        per_sample_path = os.path.join(combined_plots_dir, 'stability_per_sample.csv')
        df_ps = build_stability_per_sample_csv(df_runs, per_sample_path)

        if df_ps is not None and not df_ps.empty:
            build_stability_summary_csv(
                df_ps,
                os.path.join(combined_plots_dir, 'stability_summary.csv'),
            )
            # CV 折线图: F(S) + ES + (可选) Runtime
            plot_stability_cv_line(
                df_ps,
                os.path.join(combined_plots_dir, 'stability_cv_line_f.png'),
                metric='F(S)',
            )
            plot_stability_cv_line(
                df_ps,
                os.path.join(combined_plots_dir, 'stability_cv_line_es.png'),
                metric='ES',
            )
            if show_runtime:
                plot_stability_cv_line(
                    df_ps,
                    os.path.join(combined_plots_dir, 'stability_cv_line_rt.png'),
                    metric='Runtime',
                )
            # 箱线图
            plot_stability_boxplot(
                df_ps,
                os.path.join(combined_plots_dir, 'stability_boxplot.png'),
                show_runtime=show_runtime,
            )

        # 保存 combined_meta.json
        algorithms_present = sorted(df_runs['algorithm'].str.upper().unique().tolist()) if not df_runs.empty else []
        combined_meta = {
            'sa_root': experiment_root,
            'sibling_roots': sibling_experiment_roots,
            'algorithms_present': algorithms_present,
            'generated_at': datetime.datetime.now().isoformat(),
        }
        meta_path = os.path.join(combined_root, 'combined_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(combined_meta, f, indent=2, ensure_ascii=False)

        print(f"  [combined] SA+GA 合并输出保存: {combined_root}")
    except Exception as e:
        print(f"  [combined] SA+GA 合并处理失败: {e}")


def run_combined_stability_plots(experiment_root: str,
                                 sibling_experiment_roots: Optional[List[str]] = None,
                                 mode: str = 'auto',
                                 show_runtime: bool = True) -> None:
    """合并多个已完成单算法后处理的 stability_runs.csv, 输出合并图表与汇总表.

    合并阶段只读取各 ``experiment_root/plots/stability_runs.csv``，不再隐式补跑
    单算法后处理；调用方应先分别运行 ``run_all_ablation_plots``。
    """
    if not sibling_experiment_roots:
        print("  [combined] 未提供 sibling_experiment_roots, 跳过合并")
        return
    _run_combined_stability(
        experiment_root=experiment_root,
        sibling_experiment_roots=list(sibling_experiment_roots),
        show_runtime=show_runtime,
        mode=mode,
    )


# ------------------------------------------------------------
# 主入口
# ------------------------------------------------------------
def run_all_ablation_plots(experiment_root: str,
                           groups_for_stability: Optional[List[str]] = None,
                           sibling_experiment_roots: Optional[List[str]] = None,
                           mode: str = 'auto',
                           show_runtime: bool = True) -> None:
    """一站式消融后处理: 稳定性叠图、A/B/C 收敛对比、箱线图和统计表.

    Parameters
    ----------
    experiment_root : str
        主 (通常为 SA) 的 ablation 结果目录.
    groups_for_stability : list of str, optional
        稳定性分析的实验组名列表, 默认取自 ABLATION_GROUPS.
    sibling_experiment_roots : list of str, optional
        其他算法版本 (如 GA) 的 ablation 结果目录列表.
        用于合并 SA+GA 的 stability_runs.csv 以产出 CV 散点图、箱线图和汇总表.
        若提供, 合并产物将输出到新建的同级合并目录中.
    mode : str
        后处理模式: 'stability' 仅稳定性输出 / 'ablation' 仅消融对比 / 'auto' 自动检测.
    show_runtime : bool
        箱线图和折线图是否包含 Runtime 维度, 默认 True.
    """
    plots_dir = os.path.join(experiment_root, 'plots')
    ensure_folder(plots_dir)
    groups_for_stability = groups_for_stability or [g[0] for g in ABLATION_GROUPS]

    # 解析模式
    branches = _resolve_ablation_mode(experiment_root, mode)
    print(f"  [mode] 激活分支: {branches}")

    store = ExperimentResultStore(experiment_root)

    # 读取 runner_type, 与 recompute_iter_to_95pct 保持一致
    runner_type = 'SA'
    summary_path = os.path.join(experiment_root, 'experiment_summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                runner_type = json.load(f).get('runner_type', 'SA').upper()
        except Exception as e:
            print(f"  读取 experiment_summary.json 失败, 使用默认 SA: {e}")

    # ====== stability 分支 ======
    if 'stability' in branches:
        # 1) 跨样本聚合稳定性图
        for g in groups_for_stability:
            try:
                plot_stability_convergence_all_samples(
                    store, g,
                    os.path.join(plots_dir, f'stability_convergence_all_{g}.png'),
                    runner_type=runner_type,
                )
            except Exception as e:
                print(f"  stability_conv_all [{g}] 失败: {e}")

        # 2) 单样本稳定性详细图: 仅 A_guided
        if os.path.isdir(os.path.join(experiment_root, 'A_guided')):
            for idx in store.list_samples('A_guided'):
                try:
                    plot_stability_convergence(
                        store, 'A_guided', idx,
                        os.path.join(plots_dir, f'stability_convergence_{idx}.png'))
                except Exception as e:
                    print(f"  stability_conv [{idx}] 失败: {e}")

        # stability 表 (per-group)
        for g in groups_for_stability:
            try:
                build_stability_table(experiment_root, g)
            except Exception as e:
                print(f"  stability_table [{g}] 失败: {e}")

        # 单算法的 per_sample / summary / 图表 (写入 experiment_root/plots/)
        try:
            runs_csv = os.path.join(plots_dir, 'stability_runs.csv')
            if os.path.exists(runs_csv):
                import pandas as pd
                df_runs = pd.read_csv(runs_csv)
                per_sample_path = os.path.join(plots_dir, 'stability_per_sample.csv')
                df_ps = build_stability_per_sample_csv(df_runs, per_sample_path)

                if df_ps is not None and not df_ps.empty:
                    build_stability_summary_csv(
                        df_ps,
                        os.path.join(plots_dir, 'stability_summary.csv'),
                    )
                    plot_stability_cv_line(
                        df_ps,
                        os.path.join(plots_dir, 'stability_cv_line_f.png'),
                        metric='F(S)',
                    )
                    plot_stability_cv_line(
                        df_ps,
                        os.path.join(plots_dir, 'stability_cv_line_es.png'),
                        metric='ES',
                    )
                    if show_runtime:
                        plot_stability_cv_line(
                            df_ps,
                            os.path.join(plots_dir, 'stability_cv_line_rt.png'),
                            metric='Runtime',
                        )
                    plot_stability_boxplot(
                        df_ps,
                        os.path.join(plots_dir, 'stability_boxplot.png'),
                        show_runtime=show_runtime,
                    )
        except Exception as e:
            print(f"  单算法 stability per_sample/summary/cv/boxplot 失败: {e}")

    # ====== ablation 分支 ======
    if 'ablation' in branches:
        # A/B/C 收敛对比
        samples = set()
        for g in groups_for_stability:
            samples.update(store.list_samples(g))
        for idx in sorted(samples):
            try:
                plot_ablation_convergence(
                    store, idx,
                    os.path.join(plots_dir, f'ablation_convergence_{idx}.png'))
            except Exception as e:
                print(f"  ablation_conv [{idx}] 失败: {e}")

        # A/B/C 最终 F(S) 箱线图
        try:
            plot_boxplot_final(store, os.path.join(plots_dir, 'boxplot_final_f.png'), metric='final_best')
        except Exception as e:
            print(f"  boxplot 失败: {e}")

        # ablation 对比表
        try:
            build_ablation_compare_table(experiment_root)
        except Exception as e:
            print(f"  ablation_compare_table 失败: {e}")

        try:
            build_ablation_mean_table(experiment_root)
        except Exception as e:
            print(f"  ablation_mean_table 失败: {e}")

        try:
            build_ablation_improvement_table(experiment_root)
        except Exception as e:
            print(f"  ablation_improvement_table 失败: {e}")

        try:
            build_ablation_summary_table(experiment_root)
        except Exception as e:
            print(f"  ablation_summary_table 失败: {e}")

    # ====== SA+GA 合并 ======
    if sibling_experiment_roots:
        run_combined_stability_plots(
            experiment_root=experiment_root,
            sibling_experiment_roots=sibling_experiment_roots,
            mode=mode,
            show_runtime=show_runtime,
        )

    print(f"\n[run_all_ablation_plots] 完成, 输出目录: {plots_dir}")
