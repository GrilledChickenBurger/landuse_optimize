# ablation 实验结果可视化: tornado / convergence / LU 并排
import os
import json
import glob
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import torch

from libs.const import COLOR_MAP
from utils.tools import ensure_folder, visualize_tensor_list


# ============================================================
# 0. 目录结构适配
# ------------------------------------------------------------
# 当前 ablation pipeline 落盘结构:
#   results/ablation/<ts>/<exp_name>/run_<r>/<index>/{convergence.json, ...}
# 本模块所有"按样本枚举"和"取某样本的多 run"操作都通过下面两个函数取目录,
# 不再直接 os.listdir(exp_dir) 把第一层当样本.
# ============================================================
def _list_samples(exp_dir: str) -> List[str]:
    """返回该实验目录下所有出现过的样本 index (去重 + 排序).

    扫描 <exp_dir>/run_*/<index>/. 兼容老结构 <exp_dir>/<index>/run_*/ (若 exp_dir 第一层就是 index).
    """
    if not os.path.isdir(exp_dir):
        return []
    samples = set()
    for entry in os.listdir(exp_dir):
        sub = os.path.join(exp_dir, entry)
        if not os.path.isdir(sub):
            continue
        if entry.startswith('run_'):
            # 新结构: run_r/<index>/
            for idx in os.listdir(sub):
                if os.path.isdir(os.path.join(sub, idx)):
                    samples.add(idx)
        else:
            # 老结构: <index>/run_r/ 或 <index>/<file>
            samples.add(entry)
    return sorted(samples)


def _list_run_dirs_for_sample(exp_dir: str, index: str) -> List[str]:
    """返回某样本所有 run 的目录绝对路径列表 (按 run_r 序号升序).

    新结构: [<exp_dir>/run_0/<index>, <exp_dir>/run_1/<index>, ...]
    兼容老结构: [<exp_dir>/<index>/run_0, ...] 或 [<exp_dir>/<index>] (n_rep=1)
    """
    if not os.path.isdir(exp_dir):
        return []
    # 优先尝试新结构
    runs_new = []
    for entry in sorted(os.listdir(exp_dir)):
        if not entry.startswith('run_'):
            continue
        sample_dir = os.path.join(exp_dir, entry, index)
        if os.path.isdir(sample_dir):
            runs_new.append(sample_dir)
    if runs_new:
        return runs_new
    # 回退老结构
    old_root = os.path.join(exp_dir, index)
    if not os.path.isdir(old_root):
        return []
    runs_old = sorted([d for d in os.listdir(old_root)
                       if d.startswith('run_') and os.path.isdir(os.path.join(old_root, d))])
    if runs_old:
        return [os.path.join(old_root, d) for d in runs_old]
    # n_rep=1 老结构, sample 目录直接含 convergence.json
    if os.path.exists(os.path.join(old_root, 'convergence.json')):
        return [old_root]
    return []


# ============================================================
# 1. Tornado: B_* 相对 A_guided (或 B_baseline) baseline 的扰动
# ============================================================
def _read_perf_metric(perf_path: str, metric_path: List[str]) -> Optional[float]:
    """从 performance.json 中按嵌套 key 路径读数值. 路径不存在或文件不存在返回 None."""
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


def _aggregate_exp(exp_dir: str, metric_extractor) -> Optional[float]:
    """对一个实验文件夹下所有样本的指标做平均.

    metric_extractor(run_dir) -> float|None, 其中 run_dir 是单次 run 的目录 (含 convergence.json).
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


def plot_tornado(ablation_root: str,
                 baseline_name: str = 'A_guided',
                 save_path: Optional[str] = None,
                 metric: str = 'final_best') -> Dict[str, Dict[str, float]]:
    """画 tornado 图: 每个 B_* knob 相对 baseline 的归一化扰动幅度.

    Parameters
    ----------
    ablation_root : str
        results/ablation/<ts>/ 目录
    baseline_name : str
        baseline 实验名 (B_baseline 或 A_guided)
    save_path : str
        保存路径; None 则保存到 ablation_root/plots/tornado.png
    metric : str
        'final_best' (来自 convergence.json) 或 'es.food.best_ori_diff' 等
        (来自 performance_score.json 的嵌套路径, 用 . 分隔)
    """
    save_path = save_path or os.path.join(ablation_root, 'plots', f'tornado_{metric}.png')
    ensure_folder(os.path.dirname(save_path))

    # 选择 metric 提取器
    if metric == 'final_best':
        extractor = lambda d: _read_conv_metric(os.path.join(d, 'convergence.json'), 'final_best')
    elif metric == 'wall_time_sec':
        extractor = lambda d: _read_conv_metric(os.path.join(d, 'convergence.json'), 'wall_time_sec')
    elif metric == 'iter_to_95pct':
        extractor = lambda d: _read_conv_metric(os.path.join(d, 'convergence.json'), 'iter_to_95pct')
    else:
        # 假定是 performance_score.json 的点号路径, e.g. 'ES.food.best_ori_diff'
        path = metric.split('.')
        extractor = lambda d: _read_perf_metric(
            os.path.join(d, 'performance.json'), path)

    # baseline
    baseline_dir = os.path.join(ablation_root, baseline_name)
    baseline_val = _aggregate_exp(baseline_dir, extractor)
    if baseline_val is None or abs(baseline_val) < 1e-12:
        print(f"警告: baseline 【{baseline_name}】 metric={metric} 无效或为 0, 无法归一化")
        return {}

    # 遍历所有 B_* 实验
    results: Dict[str, Dict[str, float]] = {}  # {knob_path: {value: deviation}}
    for name in sorted(os.listdir(ablation_root)):
        if not name.startswith('B_') or name == 'B_baseline':
            continue
        exp_dir = os.path.join(ablation_root, name)
        if not os.path.isdir(exp_dir):
            continue
        # name 形如 'B_tgt_function.lambda_ES_high=1500'
        rest = name[2:]  # 去掉 'B_'
        if '=' not in rest:
            continue
        knob, val_str = rest.rsplit('=', 1)
        val = _aggregate_exp(exp_dir, extractor)
        if val is None:
            continue
        deviation = (val - baseline_val) / abs(baseline_val)
        results.setdefault(knob, {})[val_str] = deviation

    # 按区间宽度排序
    knob_widths = []
    for knob, vmap in results.items():
        devs = list(vmap.values())
        knob_widths.append((knob, min(devs), max(devs), max(devs) - min(devs)))
    knob_widths.sort(key=lambda t: t[3])  # 升序; 画图时大值在顶

    if not knob_widths:
        print(f"警告: 无任何 B_* 实验数据可用, 跳过 tornado 绘图")
        return results

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

    # 同时落盘 summary json
    summary_path = os.path.join(os.path.dirname(save_path), f'tornado_{metric}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'baseline': baseline_val, 'metric': metric, 'results': results}, f, indent=2, ensure_ascii=False)
    return results


# ============================================================
# 2. 收敛曲线: A_guided vs A_unguided
# ============================================================
def plot_convergence(ablation_root: str,
                     save_folder: Optional[str] = None,
                     smooth_window: int = 20) -> None:
    """对每个样本画 A_guided vs A_unguided 的 total_score 收敛曲线 (滑动平均).
    每个样本可能有多 run, 此处取 run_0 的曲线作为代表.
    """
    save_folder = save_folder or os.path.join(ablation_root, 'plots')
    ensure_folder(save_folder)

    guided_dir = os.path.join(ablation_root, 'A_guided')
    unguided_dir = os.path.join(ablation_root, 'A_unguided')
    if not (os.path.isdir(guided_dir) and os.path.isdir(unguided_dir)):
        print(f"警告: A_guided 或 A_unguided 不存在, 跳过收敛曲线")
        return

    # 找两边都有的样本
    samples_g = set(_list_samples(guided_dir))
    samples_u = set(_list_samples(unguided_dir))
    common = sorted(samples_g & samples_u)
    if not common:
        print(f"警告: A_guided / A_unguided 无共同样本, 跳过")
        return

    def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
        if w <= 1 or len(arr) <= w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='valid')

    for idx in common:
        g_runs = _list_run_dirs_for_sample(guided_dir, idx)
        u_runs = _list_run_dirs_for_sample(unguided_dir, idx)
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
        # 标 95% 解水平线: 统一从 convergence.json 读 final_best (已统一为 tgt_func 口径)
        g_conv = _read_conv_metric(os.path.join(g_runs[0], 'convergence.json'), 'final_best')
        if g_conv is not None:
            g_best = float(g_conv)
            target = g_best * 0.95 if g_best < 0 else g_best * 1.05
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
# 3. LU 并排: ori / A_guided.best / A_unguided.best
# ============================================================
def plot_lu_side_by_side(ablation_root: str,
                         index: str,
                         save_folder: Optional[str] = None) -> None:
    """对指定样本画 ori_plan / guided best / unguided best 并排图. 复用 visualize_tensor_list.
    plan tif 在 <exp>/run_0/<index>/plans_<idx>/{ori_plan.TIF, best_plan.TIF} (取 run_0 作为代表).
    """
    from utils.dataset_utils import remove_lines  # noqa
    import rasterio  # 项目内已使用

    save_folder = save_folder or os.path.join(ablation_root, 'plots')
    ensure_folder(save_folder)
    idx_num = index.split('_')[1]

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
        runs = _list_run_dirs_for_sample(os.path.join(ablation_root, exp_name), index)
        return os.path.join(runs[0], f'plans_{idx_num}') if runs else None

    g_plan_dir = _first_run_plan_dir('A_guided')
    u_plan_dir = _first_run_plan_dir('A_unguided')
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
# 主入口
# ============================================================
def run_all_plots(ablation_root: str,
                  baseline_name: str = 'A_guided',
                  indexes_for_lu: Optional[List[str]] = None) -> None:
    """一站式: 跑 tornado (多 metric) + convergence + LU 并排."""
    metrics = ['final_best', 'wall_time_sec', 'iter_to_95pct']
    for m in metrics:
        try:
            plot_tornado(ablation_root, baseline_name=baseline_name, metric=m)
        except Exception as e:
            print(f"tornado [{m}] 失败: {e}")

    try:
        plot_convergence(ablation_root)
    except Exception as e:
        print(f"convergence 失败: {e}")

    if indexes_for_lu:
        for idx in indexes_for_lu:
            try:
                plot_lu_side_by_side(ablation_root, idx)
            except Exception as e:
                print(f"LU compare [{idx}] 失败: {e}")


# ============================================================
# ============================================================
# C 组随机约束 + 多次重复统计 + 稳定性/消融可视化
# ============================================================
# ============================================================
from scipy import ndimage
import random as _py_random
from collections import deque

T_CRITICAL_95_DF4 = 2.776  # t_{0.025, 4}, 用于 n=5 的 95% CI 半宽
ABLATION_GROUPS = [
    ('A_guided', 'A: guided', 'A_guided', 'tab:blue', '-'),
    ('B_unguided', 'B: unguided', 'B_unguided', 'tab:orange', '--'),
    ('C_random', 'C: random', 'C_random', 'tab:green', ':'),
]


def _resolve_group_dir(ablation_root: str, group_name: str) -> str:
    return os.path.join(ablation_root, group_name)


def make_random_connected_mask(pred_diff: torch.Tensor,
                               hard_restrict_base: torch.Tensor,
                               seed: int) -> torch.Tensor:
    """生成与 pred_diff 连通块数和面积都匹配的随机 0/1 mask, 限制在 hard_restrict_base 内.

    Parameters
    ----------
    pred_diff : [1,1,H,W] bool/0-1 tensor, 模型预测的变化区域
    hard_restrict_base : [1,1,H,W] bool/0-1 tensor, 可放置随机块的候选区域 (耕地红线 ∧ LU≠0)
    seed : 随机种子

    Returns
    -------
    mask : [1,1,H,W] bool tensor, 与 pred_diff 连通块数相同, 面积之和约等
    """
    rng = np.random.RandomState(seed)
    _py_random.seed(seed)

    pd_np = pred_diff.detach().cpu().numpy().squeeze().astype(bool)
    base_np = hard_restrict_base.detach().cpu().numpy().squeeze().astype(bool)

    # 1. 统计 pred_diff 的连通块面积 (8-邻域)
    structure = np.ones((3, 3), dtype=bool)
    labeled, n_blocks = ndimage.label(pd_np, structure=structure)
    if n_blocks == 0:
        return torch.zeros_like(pred_diff, dtype=torch.bool)
    areas = [int((labeled == i).sum()) for i in range(1, n_blocks + 1)]
    # 大块优先放, 避免后续无空间
    areas.sort(reverse=True)

    H, W = pd_np.shape
    out = np.zeros((H, W), dtype=bool)

    # 2. 对每块: 在 base_np 内随机选种子点, BFS 8-邻域扩张到目标面积
    for target_area in areas:
        # 候选种子: base_np 内且尚未被占
        candidates = np.argwhere(base_np & (~out))
        if len(candidates) == 0:
            break
        seed_idx = rng.randint(0, len(candidates))
        sy, sx = candidates[seed_idx]

        # BFS 扩张
        filled = 0
        queue = deque([(sy, sx)])
        visited_local = set([(sy, sx)])
        while queue and filled < target_area:
            y, x = queue.popleft()
            if not (0 <= y < H and 0 <= x < W):
                continue
            if out[y, x] or not base_np[y, x]:
                continue
            out[y, x] = True
            filled += 1
            # 8 邻域, 随机打乱顺序避免方向偏置
            neighbors = [(y+dy, x+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                         if not (dy == 0 and dx == 0)]
            _py_random.shuffle(neighbors)
            for ny, nx in neighbors:
                if (ny, nx) not in visited_local and 0 <= ny < H and 0 <= nx < W:
                    visited_local.add((ny, nx))
                    queue.append((ny, nx))

    mask_t = torch.tensor(out, dtype=torch.bool).reshape(1, 1, H, W)
    return mask_t


# ------------------------------------------------------------
# 统计聚合
# ------------------------------------------------------------
_METRIC_KEYS_CONV = ['final_best', 'iter_to_95pct', 'wall_time_sec', 'total_iter']
_METRIC_KEYS_PERF = [
    'ES.food.best_ori_diff', 'ES.carbon.best_ori_diff',
    'ES.water.best_ori_diff', 'ES.culture.best_ori_diff',
]


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


def _iter_run_dirs(exp_dir: str, index: str) -> List[str]:
    """返回某实验下某样本所有 run 的绝对路径. 兼容新老结构, 见 _list_run_dirs_for_sample."""
    return _list_run_dirs_for_sample(exp_dir, index)


def aggregate_runs(exp_dir: str, index: str) -> Dict[str, Dict[str, Any]]:
    """从 <exp_dir>/run_*/<index>/{convergence.json, performance.json} 聚合所有 run 的指标.
    返回 {metric: {mean, std, ci_95, runs}}.
    """
    run_dirs = _iter_run_dirs(exp_dir, index)
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
            ci_half = T_CRITICAL_95_DF4 * std / np.sqrt(len(vals))
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
    run_dirs = _iter_run_dirs(exp_dir, index)
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
    # 最小化问题: target = ref_best * 1.05 (ref_best>0) 或 * 0.95 (<0)
    target = ref_best * 1.05 if ref_best >= 0 else ref_best * 0.95

    for rd in run_dirs:
        npy_p = os.path.join(rd, 'total_score_list.npy')
        conv_p = os.path.join(rd, 'convergence.json')
        if not (os.path.exists(npy_p) and os.path.exists(conv_p)):
            continue
        arr = np.load(npy_p)
        hit = (arr <= target).any() if ref_best >= 0 else (arr <= target).any()
        iter_repeat = int(np.argmax(arr <= target)) if hit else -1
        with open(conv_p, 'r', encoding='utf-8') as f:
            conv = json.load(f)
        conv['iter_to_95pct_repeat'] = iter_repeat
        conv['ref_best_repeat'] = ref_best
        with open(conv_p, 'w', encoding='utf-8') as f:
            json.dump(conv, f, indent=2, ensure_ascii=False)


def build_stability_table(ablation_root: str,
                          exp_name: str,
                          save_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """跨样本聚合单个实验组的统计指标, 输出 csv."""
    exp_dir = _resolve_group_dir(ablation_root, exp_name)
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

    save_path = save_path or os.path.join(ablation_root, 'plots', f'stability_table_{exp_name}.csv')
    ensure_folder(os.path.dirname(save_path))
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"  stability_table saved: {save_path}")
    return rows


def build_ablation_compare_table(ablation_root: str,
                                 save_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """A vs B vs C 三组并排, 每个样本一行 (各组 mean±std 与相对 A 的 % 变化)."""
    groups = [g[0] for g in ABLATION_GROUPS]
    per_group: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    samples = set()
    for g in groups:
        gd = _resolve_group_dir(ablation_root, g)
        if not os.path.isdir(gd):
            continue
        per_group[g] = {}
        for idx in _list_samples(gd):
            recompute_iter_to_95pct(gd, idx)
            per_group[g][idx] = aggregate_runs(gd, idx)
            samples.add(idx)

    rows = []
    metrics = ['final_best', 'iter_to_95pct_repeat', 'wall_time_sec']
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

    save_path = save_path or os.path.join(ablation_root, 'plots', 'ablation_compare_table.csv')
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


# ------------------------------------------------------------
# 多次重复稳定性与 A/B/C 消融可视化
# ------------------------------------------------------------
def plot_stability_convergence(exp_dir: str,
                               index: str,
                               save_path: str,
                               smooth_window: int = 50) -> None:
    """单个样本的多次运行收敛曲线叠图: 5 条半透明 + 均值粗线 + ±1 std 阴影."""
    run_dirs = _iter_run_dirs(exp_dir, index)
    if len(run_dirs) < 2:
        print(f"  [stability_conv] {exp_dir}/{index} run 数 <2, 跳过")
        return
    series = []
    for rd in run_dirs:
        p = os.path.join(rd, 'total_score_list.npy')
        if os.path.exists(p):
            arr = np.load(p)
            series.append(arr)
    if not series:
        return
    min_len = min(len(s) for s in series)
    series_aligned = np.stack([s[:min_len] for s in series], axis=0)  # [n_runs, min_len]

    # 滑动平均
    if smooth_window > 1 and min_len > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.stack([np.convolve(s, kernel, mode='valid') for s in series_aligned], axis=0)
    else:
        smoothed = series_aligned

    mean = smoothed.mean(axis=0)
    std = smoothed.std(axis=0, ddof=1) if len(series) >= 2 else np.zeros_like(mean)
    x = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=(9, 5))
    for s in smoothed:
        ax.plot(x, s, color='steelblue', alpha=0.3, linewidth=0.8)
    ax.plot(x, mean, color='navy', linewidth=2.0, label=f'mean (n={len(series)})')
    ax.fill_between(x, mean - std, mean + std, color='steelblue', alpha=0.2, label='±1 std')
    ax.set_xlabel('iteration (smoothed)')
    ax.set_ylabel('total_score (lower=better)')
    ax.set_title(f'Stability: {index} (n={len(series)} runs)')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    ensure_folder(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  stability_convergence saved: {save_path}")


def plot_ablation_convergence(ablation_root: str,
                              idx: str,
                              save_path: str,
                              smooth_window: int = 50) -> None:
    """A/B/C 三组单样本的平均收敛曲线对比, 标注 iter_to_95pct_repeat 竖线."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for g, label, _, color, ls in ABLATION_GROUPS:
        exp_dir = _resolve_group_dir(ablation_root, g)
        if not os.path.isdir(exp_dir):
            continue
        run_dirs = _iter_run_dirs(exp_dir, idx)
        series = []
        for rd in run_dirs:
            p = os.path.join(rd, 'total_score_list.npy')
            if os.path.exists(p):
                series.append(np.load(p))
        if not series:
            continue
        min_len = min(len(s) for s in series)
        aligned = np.stack([s[:min_len] for s in series], axis=0)
        if smooth_window > 1 and min_len > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.stack([np.convolve(s, kernel, mode='valid') for s in aligned], axis=0)
        else:
            smoothed = aligned
        mean = smoothed.mean(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, color=color, linestyle=ls, linewidth=2.0, label=label)
        recompute_iter_to_95pct(exp_dir, idx)
        agg = aggregate_runs(exp_dir, idx)
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


def plot_boxplot_final(ablation_root: str,
                       save_path: str,
                       metric: str = 'final_best') -> None:
    """5 样本 × A/B/C 最终 F(S) 箱线图. 复用 visualize_utils.grouped_box_plot_figure."""
    from utils.visualize_utils import grouped_box_plot_figure

    groups = [(g[0], g[2]) for g in ABLATION_GROUPS]
    samples = set()
    for g, _ in groups:
        gd = _resolve_group_dir(ablation_root, g)
        if os.path.isdir(gd):
            samples.update(_list_samples(gd))
    samples = sorted(samples)

    # data 结构: {subplot_title: {group_label: {series_name: [values]}}}
    # 这里只有 1 个 subplot, group=样本, series=A/B/C
    sub = {}
    for s in samples:
        sub[s] = {}
        for g, label in groups:
            exp_dir = _resolve_group_dir(ablation_root, g)
            if not os.path.isdir(exp_dir):
                continue
            agg = aggregate_runs(exp_dir, s)
            vals = agg.get(metric, {}).get('runs', [])
            if vals:
                sub[s][label] = vals

    data = {f'{metric} by sample × group': sub}
    ensure_folder(os.path.dirname(save_path))
    grouped_box_plot_figure(
        data=data,
        title=f'Ablation final F(S) distribution',
        y_label=metric,
        save_path=save_path,
        figsize=(max(10, 2 * len(samples)), 6),
    )
    print(f"  boxplot saved: {save_path}")


# ------------------------------------------------------------
# 主入口
# ------------------------------------------------------------
def run_all_ablation_plots(ablation_root: str,
                           groups_for_stability: Optional[List[str]] = None) -> None:
    """一站式消融后处理: 稳定性叠图、A/B/C 收敛对比、箱线图和统计表."""
    plots_dir = os.path.join(ablation_root, 'plots')
    ensure_folder(plots_dir)
    groups_for_stability = groups_for_stability or [g[0] for g in ABLATION_GROUPS]

    a_dir = _resolve_group_dir(ablation_root, 'A_guided')
    if os.path.isdir(a_dir):
        for idx in _list_samples(a_dir):
            try:
                plot_stability_convergence(a_dir, idx,
                    os.path.join(plots_dir, f'stability_convergence_{idx}.png'))
            except Exception as e:
                print(f"  stability_conv [{idx}] 失败: {e}")

    samples = set()
    for g in groups_for_stability:
        gd = _resolve_group_dir(ablation_root, g)
        if os.path.isdir(gd):
            samples.update(_list_samples(gd))
    for idx in sorted(samples):
        try:
            plot_ablation_convergence(ablation_root, idx,
                os.path.join(plots_dir, f'ablation_convergence_{idx}.png'))
        except Exception as e:
            print(f"  ablation_conv [{idx}] 失败: {e}")

    try:
        plot_boxplot_final(ablation_root,
            os.path.join(plots_dir, 'boxplot_final_f.png'),
            metric='final_best')
    except Exception as e:
        print(f"  boxplot 失败: {e}")

    for g in groups_for_stability:
        try:
            build_stability_table(ablation_root, g)
        except Exception as e:
            print(f"  stability_table [{g}] 失败: {e}")
    try:
        build_ablation_compare_table(ablation_root)
    except Exception as e:
        print(f"  ablation_compare_table 失败: {e}")

    print(f"\n[run_all_ablation_plots] 完成, 输出目录: {plots_dir}")

