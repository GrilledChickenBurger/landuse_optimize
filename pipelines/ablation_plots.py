# ablation 实验结果可视化: convergence / LU 并排 / 稳定性叠图 / A/B/C 对比
import os
import json
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import torch

from libs.const import COLOR_MAP
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
                            figsize=(10, 5)) -> None:
    """CV 散点图: N 个样本按 n_pred 升序, SA vs GA 两组散点.

    依照 stability.md 5.2: 无连线, 加 5% 参考虚线.

    Parameters
    ----------
    per_sample_csv_or_df : str or pd.DataFrame
        stability_per_sample.csv 路径, 或由 merge + build 得到的 DataFrame
        (需含 F_cv_pct / RT_cv_pct 列).
    save_path : str
        输出 PNG 路径.
    metric : str
        'F(S)' 或 'Runtime', 选择画哪一列 CV.
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

    fig, ax = plt.subplots(figsize=figsize)
    x_label = r'Sample (sorted by $n_{\mathrm{pred}}$)'
    markersize = 8

    if not sa.empty:
        x_positions = np.arange(len(sa))
        ax.scatter(x_positions, sa[cv_col].values,
                   color='tab:blue', label='SA', s=markersize**2, zorder=3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sa['sample_id'].values, rotation=45, ha='right', fontsize=8)

    if not ga.empty:
        gx_positions = np.arange(len(ga))
        ax.scatter(gx_positions, ga[cv_col].values,
                   color='tab:orange', label='GA', s=markersize**2, marker='s', zorder=3)
        if sa.empty:
            ax.set_xticks(gx_positions)
            ax.set_xticklabels(ga['sample_id'].values, rotation=45, ha='right', fontsize=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(linestyle='--', alpha=0.4)

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
                            figsize=(10, 5)) -> None:
    """stability 箱线图: 横轴为两个指标, 每指标 SA vs GA 箱线, 叠加散点.

    数据来源为 stability_per_sample.csv (build_stability_per_sample_csv 的输出),
    箱线内是 N 个样本在各自指标上的 CV 值, 直观展示跨样本变异系数分布。

    Parameters
    ----------
    per_sample_csv_or_df : str or pd.DataFrame
        stability_per_sample.csv 路径或 DataFrame, 需含 F_cv_pct / RT_cv_pct / algorithm 列.
    save_path : str
        输出 PNG 路径.
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
    sa_rt_cv = sa['RT_cv_pct'].astype(float).dropna().values
    ga_f_cv = ga['F_cv_pct'].astype(float).dropna().values
    ga_rt_cv = ga['RT_cv_pct'].astype(float).dropna().values

    # 计算样本数 (SA/GA 各应相同)
    n_f = max(len(sa_f_cv), len(ga_f_cv))
    n_rt = max(len(sa_rt_cv), len(ga_rt_cv))

    # 箱线位置: x=0 处画 F(S), x=1 处画 Runtime
    # SA 箱线在左 (-0.18), GA 在右 (+0.18)
    sa_offset = -0.18
    ga_offset = +0.18
    sa_positions = [0 + sa_offset, 1 + sa_offset]
    ga_positions = [0 + ga_offset, 1 + ga_offset]

    fig, ax = plt.subplots(figsize=figsize)

    bp_sa = ax.boxplot(
        [sa_f_cv, sa_rt_cv], positions=sa_positions, widths=0.28,
        patch_artist=True,
        boxprops=dict(facecolor='tab:blue', alpha=0.45),
        medianprops=dict(color='navy', linewidth=1.5),
        flierprops=dict(marker='', markersize=0),  # 隐藏离群点, 由散点替代
    )
    bp_ga = ax.boxplot(
        [ga_f_cv, ga_rt_cv], positions=ga_positions, widths=0.28,
        patch_artist=True,
        boxprops=dict(facecolor='tab:orange', alpha=0.45),
        medianprops=dict(color='darkorange', linewidth=1.5),
        flierprops=dict(marker='', markersize=0),
    )

    # 叠加散点 (每个样本一个点)
    rng = np.random.RandomState(42)
    for pos, vals in zip([0 + sa_offset, 1 + sa_offset], [sa_f_cv, sa_rt_cv]):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.06, 0.06, len(vals))
        ax.scatter(np.full_like(vals, pos, dtype=float) + jitter, vals,
                   color='tab:blue', alpha=0.7, s=30, zorder=3, edgecolors='white', linewidth=0.3)
    for pos, vals in zip([0 + ga_offset, 1 + ga_offset], [ga_f_cv, ga_rt_cv]):
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
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['综合目标函数变异系数', '运行时间变异系数'], rotation=0, fontsize=11)
    ax.set_ylabel('变异系数 CV (%)')
    ax.set_title(f'SA vs GA stability: CV distribution across {n_f} samples')

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
# 主入口
# ------------------------------------------------------------
def run_all_ablation_plots(experiment_root: str,
                           groups_for_stability: Optional[List[str]] = None,
                           sibling_experiment_roots: Optional[List[str]] = None) -> None:
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
    """
    plots_dir = os.path.join(experiment_root, 'plots')
    ensure_folder(plots_dir)
    groups_for_stability = groups_for_stability or [g[0] for g in ABLATION_GROUPS]

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

    try:
        plot_boxplot_final(store, os.path.join(plots_dir, 'boxplot_final_f.png'), metric='final_best')
    except Exception as e:
        print(f"  boxplot 失败: {e}")

    for g in groups_for_stability:
        try:
            build_stability_table(experiment_root, g)
        except Exception as e:
            print(f"  stability_table [{g}] 失败: {e}")
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

    # ------------------------------------------------------------
    # stability 模式: 合并 SA+GA runs.csv → per_sample / summary / CV 散点图 / 箱线图
    # ------------------------------------------------------------
    try:
        runs_csv_list = [os.path.join(experiment_root, 'plots', 'stability_runs.csv')]
        for sib in (sibling_experiment_roots or []):
            runs_csv_list.append(os.path.join(sib, 'plots', 'stability_runs.csv'))

        existing = [p for p in runs_csv_list if os.path.exists(p)]
        if existing:
            df_runs = merge_stability_runs(existing)
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
                    os.path.join(plots_dir, 'stability_cv_line_rt.png'),
                    metric='Runtime',
                )
                plot_stability_boxplot(
                    df_ps,
                    os.path.join(plots_dir, 'stability_boxplot.png'),
                )
        else:
            print(f"  [stability per_sample/summary/cv_scatter/boxplot] 未找到 stability_runs.csv, 跳过")
    except Exception as e:
        print(f"  stability per_sample/summary/cv_scatter/boxplot 失败: {e}")

    print(f"\n[run_all_ablation_plots] 完成, 输出目录: {plots_dir}")
