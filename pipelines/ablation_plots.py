# ablation 实验结果可视化: convergence / LU 并排 / 稳定性叠图 / A/B/C 对比
import os
import json
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── 全局字体: 英文 Times New Roman Bold + 中文回退 ──────────────
from utils.visualize_utils import setup_plot_fonts
setup_plot_fonts()
# ──────────────────────────────────────────────────────────────────


# Q1-Q5 背景色映射 (蓝→绿渐变, 总体淡雅)
_LEVEL_COLORS = {
    'Q1': '#e8f4f8',  # 很浅蓝
    'Q2': '#d4eaf0',  # 浅蓝
    'Q3': '#c8e6d8',  # 蓝绿过渡
    'Q4': '#bce4cc',  # 浅绿
    'Q5': '#b0dcc0',  # 稍深绿(仍淡)
}
_LEVEL_TEXT_COLORS = {
    'Q1': '#5a8aaa',
    'Q2': '#4a7a9a',
    'Q3': '#3a8a7a',
    'Q4': '#2a7a6a',
    'Q5': '#1a6a5a',
}
_LEVEL_ORDER = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']


def _draw_level_bands(ax, primary, n):
    """在 ax 上绘制 Q1-Q5 背景色带."""
    if 'level' not in primary.columns:
        return False
    for i in range(n):
        lv = primary['level'].iloc[i]
        color = _LEVEL_COLORS.get(str(lv), '#f5f5f5')
        ax.axvspan(i - 0.5, i + 0.5, facecolor=color, edgecolor='none', zorder=0)
    return True


def _draw_level_labels(ax, primary, n):
    """在背景色带垂直中心位置标注 Q1-Q5 文字."""
    if 'level' not in primary.columns:
        return
    y_min, y_max = ax.get_ylim()
    y_text = y_min + (y_max - y_min) * 0.6
    for lv in _LEVEL_ORDER:
        indices = [i for i in range(n) if str(primary['level'].iloc[i]) == lv]
        if indices:
            mid_x = np.mean(indices)
            text_color = _LEVEL_TEXT_COLORS.get(lv, '#555555')
            ax.text(mid_x, y_text, lv, ha='center', va='center',
                    fontsize=11, color=text_color, fontweight='bold',
                    alpha=0.7, zorder=5)


def _set_bottom_center_legend(artist, ncol=None, handles=None, labels=None):
    """统一图例风格：放在下方居中，x_label 之下，透明无边框。

    Parameters
    ----------
    artist : Axes 或 Figure
        需要添加图例的对象。Figure 时须显式传入 handles/labels。
    ncol : int, optional
        图例列数；默认根据条目数自动选择（最多 4 列）。
    handles, labels : list, optional
        用于 Figure.legend 的句柄与标签；对 Axes 可缺省，自动从 artist 获取。
    """
    if isinstance(artist, plt.Axes):
        handles, labels = handles or artist.get_legend_handles_labels()
        if not labels:
            return
        if ncol is None:
            ncol = min(len(labels), 4)
        artist.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, -0.14), ncol=ncol,
                      frameon=False)
    else:  # Figure
        if not labels:
            return
        if ncol is None:
            ncol = min(len(labels), 4)
        # 使用 figure 坐标，把图例放在底部留白区域内
        artist.legend(handles, labels, loc='lower center',
                      bbox_to_anchor=(0.5, 0.04), ncol=ncol,
                      frameon=False)


from utils.tools import ensure_folder
from utils.exp_sweep_utils import (
    ABLATION_GROUPS,
    _BASELINE_DIR,
    build_stability_table, build_ablation_compare_table,
    build_ablation_mean_table, build_ablation_improvement_table, build_ablation_summary_table,
    ExperimentResultStore,
    merge_stability_runs, build_stability_per_sample_csv, build_stability_summary_csv,
    build_dar_matrix_csv, build_dar_summary_csv,
)


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
    _set_bottom_center_legend(ax)
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
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
    _set_bottom_center_legend(ax, ncol=min(len(per_sample), 3))
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
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
    _set_bottom_center_legend(ax)
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    ensure_folder(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ablation_convergence saved: {save_path}")


def plot_ablation_convergence_normalized(store: ExperimentResultStore,
                                         idx: str,
                                         save_path: str,
                                         smooth_window: int = 1,
                                         max_iter: Optional[int] = None) -> None:
    """A/B/C 三组单样本的归一化改善百分比收敛曲线对比.

    Y 轴为 Normalized Improvement Percentage [0, 100], 95% 处画水平参考线,
    并在 A/C 曲线穿过 95% 处标注 T95_best 竖线. B 不收敛则不标注.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for g, label, _, color, ls in ABLATION_GROUPS:
        series = store.get_improvement_series(g, idx, smooth_window=smooth_window)
        if series is None:
            continue
        mean = series.mean(axis=0)
        std = series.std(axis=0, ddof=1) if series.shape[0] >= 2 else np.zeros_like(mean)
        x = np.arange(len(mean))
        if max_iter is not None:
            x = x[:max_iter]
            mean = mean[:max_iter]
            std = std[:max_iter]

        ax.plot(x, mean, color=color, linestyle=ls, linewidth=2.0, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        agg = store.recompute_and_aggregate(g, idx)
        iter_best = agg.get('iter_to_95pct_best', {}).get('mean')
        if iter_best is not None and 0 < iter_best < len(x):
            ax.axvline(iter_best, color=color, linestyle=ls, alpha=0.5, linewidth=1.2)
            ax.text(iter_best, 100.5, f'{int(iter_best)}',
                    color=color, fontsize=8, ha='center', va='bottom')

    ax.axhline(95.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='95%')
    ax.set_xlabel('iteration')
    ax.set_ylabel('normalized improvement (%)')
    ax.set_title(f'Ablation convergence (normalized): {idx}')
    ax.set_ylim(-5, 110)
    _set_bottom_center_legend(ax)
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    ensure_folder(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ablation_convergence_normalized saved: {save_path}")


def plot_ablation_convergence_normalized_all_samples(store: ExperimentResultStore,
                                                     save_path: str,
                                                     smooth_window: int = 1,
                                                     max_iter: Optional[int] = None) -> None:
    """所有样本的归一化改善百分比收敛曲线聚合到一张图.

    每个 group 画一条跨样本均值曲线 + ±1 std 阴影.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('black')
    for g, label, _, color, ls in ABLATION_GROUPS:
        all_means = []
        sample_counts = []
        max_len = 0
        per_sample = {}
        for idx in store.list_samples(g):
            series = store.get_improvement_series(g, idx, smooth_window=smooth_window)
            if series is None:
                continue
            mean = series.mean(axis=0)
            if max_iter is not None:
                mean = mean[:max_iter]
            per_sample[idx] = mean
            max_len = max(max_len, len(mean))
            sample_counts.append(series.shape[0])
        if not per_sample:
            continue

        # 对齐到 max_len
        aligned = []
        for m in per_sample.values():
            if len(m) < max_len:
                aligned.append(np.pad(m, (0, max_len - len(m)), mode='edge'))
            else:
                aligned.append(m[:max_len])
        mat = np.stack(aligned, axis=0)
        grand_mean = mat.mean(axis=0)
        grand_std = mat.std(axis=0, ddof=1)
        x = np.arange(max_len)

        ax.plot(x, grand_mean, color=color, linestyle=ls, linewidth=2.0,
                label=f'{label}')
        ax.fill_between(x, grand_mean - grand_std, grand_mean + grand_std,
                        color=color, alpha=0.15)

    ax.axhline(95.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='95%')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('归一化改善率 (%)')
    ax.set_title('A/B/C 三组归一化收敛曲线（跨样本聚合）')
    ax.set_ylim(-5, 110)
    _set_bottom_center_legend(ax)
    ax.grid(linestyle='--', alpha=0.25, color='#cccccc')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    ensure_folder(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ablation_convergence_normalized_all saved: {save_path}")


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
        ylabel = r'综合目标函数变异系数 CV (%)'
        title = r'SA 与 GA 稳定性对比：综合目标函数'
    elif metric == 'ES':
        cv_col = 'ES_cv_pct'
        ylabel = r'生态系统服务变异系数 CV (%)'
        title = r'SA 与 GA 稳定性对比：生态系统服务'
    else:
        cv_col = 'RT_cv_pct'
        ylabel = r'运行时间变异系数 CV (%)'
        title = r'SA 与 GA 稳定性对比：运行时间'

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
    x_label = '样本（按 n_pred 升序）'
    markersize = 9
    # 同指标内 SA/GA 使用相同 marker
    marker = 'o' if metric in ('F(S)', 'Runtime') else '^'

    # ── 背景色带 + Q1-Q5 标注 ────────────────────────────────────
    has_level = _draw_level_bands(ax, primary, n)
    # 先画散点让 autoscale 生效，再标注文字

    # ── SA 散点 ──────────────────────────────────────────────────
    if not sa.empty:
        sa_order = np.arange(len(sa))
        ax.scatter(sa_order, sa[cv_col].values,
                   color='tab:blue', label='SA', s=markersize**2,
                   marker=marker, edgecolors='navy', linewidth=0.5, zorder=4)

    # ── GA 散点 ──────────────────────────────────────────────────
    if not ga.empty:
        ga_order = np.arange(len(ga))
        ax.scatter(ga_order, ga[cv_col].values,
                   color='tab:orange', label='GA', s=markersize**2,
                   marker=marker, edgecolors='darkorange', linewidth=0.5, zorder=4)

    if has_level:
        _draw_level_labels(ax, primary, n)

    # ── X 轴: S1..SN 短标签 ──────────────────────────────────────
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'S{i + 1}' for i in range(n)], rotation=0, ha='center')

    # ── SA/GA 图例: 下方居中, 位于 x_label 之下 ─────────────────
    _set_bottom_center_legend(ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.7, n - 0.3)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # CV=5% 阈值水平线
    ax.axhline(y=5.0, color='gray', linestyle=':', alpha=0.6, linewidth=1.0)
    ax.annotate('5%', xy=(0.98, 5.0), ha='right', va='bottom',
                color='gray', xycoords=('axes fraction', 'data'))

    fig.subplots_adjust(bottom=0.20)
    ensure_folder(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  stability_cv_scatter saved: {save_path}")


# ============================================================
# stability 整合折线图: F(S) + ES 同图, 不同 marker 区分
# ============================================================
def plot_stability_cv_line_combined(per_sample_csv_or_df,
                                     save_path: str,
                                     figsize=None,
                                     dual_panel: bool = True) -> None:
    """整合 CV 散点图: 展示 F(S) 与 ES 的 SA/GA 对比.

    Parameters
    ----------
    dual_panel : bool
        True  -> 上下两个独立子图 (默认), 彻底避免双Y轴刻度误读;
        False -> 单图双Y轴 (旧布局), 左轴F(S)右轴ES.
    """
    import pandas as pd

    if isinstance(per_sample_csv_or_df, str):
        if not os.path.isfile(per_sample_csv_or_df):
            print(f"  [cv_combined] CSV 不存在: {per_sample_csv_or_df}, 跳过")
            return
        df = pd.read_csv(per_sample_csv_or_df)
    else:
        df = per_sample_csv_or_df

    if df.empty or 'algorithm' not in df.columns:
        print(f"  [cv_combined] 缺 algorithm 列, 跳过")
        return

    sa = df[df['algorithm'].str.upper() == 'SA'].sort_values('n_pred').copy()
    ga = df[df['algorithm'].str.upper() == 'GA'].sort_values('n_pred').copy()
    primary = sa if not sa.empty else ga
    n = len(primary)
    if n == 0:
        print(f"  [cv_combined] 无有效数据, 跳过")
        return

    x_positions = np.arange(n)
    if figsize is None:
        figsize = (12, 7) if dual_panel else (12, 5)

    # ── 创建画布 ─────────────────────────────────────────────────
    if dual_panel:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={'height_ratios': [3, 1.5], 'hspace': 0.08}
        )
        ax_f, ax_e = ax1, ax2          # F(S) 上图, ES 下图
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        ax_f, ax_e = ax1, ax2          # F(S) 左轴, ES 右轴

    axes_for_band = [ax1, ax2] if dual_panel else [ax1]

    # ── 背景色带 ─────────────────────────────────────────────────
    has_level = _draw_level_bands(ax1, primary, n)
    if dual_panel:
        _draw_level_bands(ax2, primary, n)

    # ── F(S) 散点 (统一用圆) ─────────────────────────────────────
    if not sa.empty and 'F_cv_pct' in sa.columns:
        ax_f.scatter(np.arange(len(sa)), sa['F_cv_pct'].values,
                     color='tab:blue', marker='o', s=80,
                     edgecolors='navy', linewidth=0.5, zorder=4,
                     label='SA 综合目标函数')
    if not ga.empty and 'F_cv_pct' in ga.columns:
        ax_f.scatter(np.arange(len(ga)), ga['F_cv_pct'].values,
                     color='tab:orange', marker='o', s=80,
                     edgecolors='darkorange', linewidth=0.5, zorder=4,
                     label='GA 综合目标函数')

    # ── ES 散点 (统一用三角) ─────────────────────────────────────
    if not sa.empty and 'ES_cv_pct' in sa.columns:
        ax_e.scatter(np.arange(len(sa)), sa['ES_cv_pct'].values,
                     color='tab:blue', marker='^', s=80,
                     edgecolors='navy', linewidth=0.5, zorder=4,
                     label='SA 生态系统服务')
    if not ga.empty and 'ES_cv_pct' in ga.columns:
        ax_e.scatter(np.arange(len(ga)), ga['ES_cv_pct'].values,
                     color='tab:orange', marker='^', s=80,
                     edgecolors='darkorange', linewidth=0.5, zorder=4,
                     label='GA 生态系统服务')

    # ── 在色带中间标注 Q1-Q5 ─────────────────────────────────────
    if has_level:
        _draw_level_labels(ax_f, primary, n)

    # ── X 轴 ─────────────────────────────────────────────────────
    ax_f.set_xticks(x_positions)
    ax_f.set_xticklabels([f'S{i + 1}' for i in range(n)],
                         rotation=0, ha='center')

    # ── 图例: 下方居中, 位于 x_label 之下 ───────────────────────
    lines_f, labels_f = ax_f.get_legend_handles_labels()
    lines_e, labels_e = ax_e.get_legend_handles_labels()
    _set_bottom_center_legend(fig, handles=lines_f + lines_e, labels=labels_f + labels_e)

    # ── 标签与标题 ───────────────────────────────────────────────
    if dual_panel:
        ax_f.set_ylabel('综合目标函数 CV (%)')
        ax_e.set_ylabel('生态系统服务 CV (%)')
        ax_e.set_xlabel('样本（按 n_pred 升序）')
        ax_f.set_title('SA 与 GA 稳定性对比：综合目标函数与生态系统服务')
        ax_f.set_xlim(-0.7, n - 0.3)
        ax_e.set_xlim(-0.7, n - 0.3)
        ax_f.grid(axis='y', linestyle='--', alpha=0.4)
        ax_e.grid(axis='y', linestyle='--', alpha=0.4)
        # 5% 阈值线仅上图
        ax_f.axhline(y=5.0, color='gray', linestyle=':', alpha=0.6, linewidth=1.0)
        ax_f.annotate('5%', xy=(0.98, 5.0), ha='right', va='bottom',
                      color='gray', xycoords=('axes fraction', 'data'))
        fig.subplots_adjust(bottom=0.20)
    else:
        ax1.set_xlabel('样本（按 n_pred 升序）')
        ax1.set_ylabel('综合目标函数 CV (%)', color='tab:blue')
        ax2.set_ylabel('生态系统服务 CV (%)', color='tab:orange')
        ax1.set_title('SA 与 GA 稳定性对比：综合目标函数与生态系统服务')
        ax1.set_xlim(-0.7, n - 0.3)
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        # 5% 阈值线
        ax1.axhline(y=5.0, color='gray', linestyle=':', alpha=0.6, linewidth=1.0)
        ax1.annotate('5%', xy=(0.98, 5.0), ha='right', va='bottom',
                      color='gray', xycoords=('axes fraction', 'data'))
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.20)

    ensure_folder(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  stability_cv_combined saved: {save_path}")


# ============================================================
# stability 箱线图: 展示 N 个样本的 CV 分布 (stability.md §5.1)
# ============================================================
def plot_stability_boxplot(per_sample_csv_or_df,
                            save_path: str,
                            show_runtime: bool = False,
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
    metrics_info.append(('F(S)', sa_f_cv, ga_f_cv, '综合目标函数'))
    if sa_es_cv.size > 0 or ga_es_cv.size > 0:
        metrics_info.append(('ES', sa_es_cv, ga_es_cv, '生态系统服务'))
    if show_runtime:
        metrics_info.append(('Runtime', sa_rt_cv, ga_rt_cv, '运行时间'))

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
    ax.annotate('5%', xy=(0.98, 5.0), ha='right', va='bottom',
                color='gray', xycoords=('axes fraction', 'data'))

    # x 轴: 中文标签, 水平不旋转
    x_labels = [m[3] for m in metrics_info]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel('变异系数 CV (%)')
    ax.set_title(f'SA 与 GA 稳定性：跨样本变异系数分布（{n_samples} 个样本）')

    # 图例 (透明背景, 下方居中)
    _set_bottom_center_legend(ax)
    ax.grid(linestyle='--', alpha=0.4, axis='y')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
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


def _get_algorithm_label(experiment_root: str) -> str:
    """从 experiment_summary.json 读取 runner_type 作为算法标识.

    找不到时返回目录名最后一段 (timestamp) 作为 fallback.
    """
    summary_path = os.path.join(experiment_root, 'experiment_summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f).get('runner_type', 'SA').upper()
        except Exception:
            pass
    return os.path.basename(experiment_root.rstrip('/\\'))


def _merge_dar_tables(all_roots: List[str],
                      combined_plots_dir: str) -> None:
    """合并 SA 和 GA 各自已生成的 DAR 三张表, 加 algorithm 列后拼接写入合并目录.

    对 dar_detail.csv / dar_matrix.csv / dar_summary.csv 分别处理:
    若某 root 下缺少对应文件则跳过该 root.
    """
    dar_files = ['dar_detail.csv', 'dar_matrix.csv', 'dar_summary.csv']
    for dar_file in dar_files:
        frames = []
        for root in all_roots:
            path = os.path.join(root, 'plots', dar_file)
            if not os.path.exists(path):
                print(f"  [combined] DAR 文件缺失, 跳过: {path}")
                continue
            try:
                import pandas as pd
                df = pd.read_csv(path)
                df.insert(0, 'algorithm', _get_algorithm_label(root))
                frames.append(df)
            except Exception as e:
                print(f"  [combined] 读取 DAR 文件失败: {path}: {e}")
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            save_path = os.path.join(combined_plots_dir, dar_file)
            combined.to_csv(save_path, index=False)
            print(f"  [combined] DAR 合并 ({len(frames)} 来源): {save_path}")
        else:
            print(f"  [combined] DAR 合并跳过: 无可用 {dar_file}")


def _run_combined_stability(experiment_root: str,
                             sibling_experiment_roots: List[str],
                             show_runtime: bool = False,
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
            # 整合折线图
            plot_stability_cv_line_combined(
                df_ps,
                os.path.join(combined_plots_dir, 'stability_cv_line_combined.png'),
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

        # 合并 DAR 三张表: 读 SA/GA 各自的 dar_*.csv, 加 algorithm 列后拼接
        _merge_dar_tables(
            all_roots=all_roots,
            combined_plots_dir=combined_plots_dir,
        )

        print(f"  [combined] SA+GA 合并输出保存: {combined_root}")
    except Exception as e:
        print(f"  [combined] SA+GA 合并处理失败: {e}")


def run_combined_stability_plots(experiment_root: str,
                                 sibling_experiment_roots: Optional[List[str]] = None,
                                 mode: str = 'auto',
                                 show_runtime: bool = False) -> None:
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
                           show_runtime: bool = False) -> None:
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

        # 2) 单样本稳定性详细图: 仅 baseline (完整方法)
        if os.path.isdir(os.path.join(experiment_root, _BASELINE_DIR)):
            for idx in store.list_samples(_BASELINE_DIR):
                try:
                    plot_stability_convergence(
                        store, _BASELINE_DIR, idx,
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
                    plot_stability_cv_line_combined(
                        df_ps,
                        os.path.join(plots_dir, 'stability_cv_line_combined.png'),
                    )
                    plot_stability_boxplot(
                        df_ps,
                        os.path.join(plots_dir, 'stability_boxplot.png'),
                        show_runtime=show_runtime,
                    )
        except Exception as e:
            print(f"  单算法 stability per_sample/summary/cv/boxplot 失败: {e}")

        # DAR (维度达标率矩阵): 仅对 baseline (完整方法) 计算
        try:
            baseline_dir = os.path.join(experiment_root, _BASELINE_DIR)
            if os.path.isdir(baseline_dir):
                rep_json_path = os.path.join(experiment_root, 'representative_samples.json')
                if not os.path.exists(rep_json_path):
                    rep_json_path = None
                build_dar_matrix_csv(
                    baseline_dir,
                    rep_json_path=rep_json_path,
                    run_start=0, run_end=4,
                    save_path=os.path.join(plots_dir, 'dar_matrix.csv'),
                )
                # 读取 dar_matrix 用于汇总 (避免重复读取 detail)
                import pandas as pd
                dar_matrix = pd.read_csv(os.path.join(plots_dir, 'dar_matrix.csv'))
                build_dar_summary_csv(
                    dar_matrix,
                    save_path=os.path.join(plots_dir, 'dar_summary.csv'),
                )
        except Exception as e:
            print(f"  DAR 矩阵生成失败: {e}")

    # ====== ablation 分支 ======
    if 'ablation' in branches:
        # A/B/C 收敛对比 (raw total_score)
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

        # A/B/C 归一化改善百分比收敛对比 (核心图)
        for idx in sorted(samples):
            try:
                plot_ablation_convergence_normalized(
                    store, idx,
                    os.path.join(plots_dir, f'ablation_convergence_norm_{idx}.png'),
                    smooth_window=1)
            except Exception as e:
                print(f"  ablation_conv_norm [{idx}] 失败: {e}")

        # 所有样本聚合归一化改善曲线
        try:
            plot_ablation_convergence_normalized_all_samples(
                store,
                os.path.join(plots_dir, 'ablation_convergence_norm_all.png'),
                smooth_window=1)
        except Exception as e:
            print(f"  ablation_conv_norm_all 失败: {e}")

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
