# -*- coding: utf-8 -*-
"""检查指定 run 范围内各样本的统计稳定性.

复用 utils/exp_sweep_utils.py 的指标读取逻辑, 输出每个样本的 mean/std/variance/CV%/CI95/min/max
以及可靠性判定.

用法:
    # 单实验目录
    python scripts/check_run_reliability.py results/tornado/20260606_033319/A_baseline --run-range 0-2
    python scripts/check_run_reliability.py results/ablation/20260610_041128/A_guided --run-range 0-4 --metrics final_best,es_score

    # 批量处理父目录下所有子实验目录（每个子目录内含 run_*）
    python scripts/check_run_reliability.py results/tornado/20260606_033319 --run-range 0-2
"""
import os
import sys
import json
import csv
import argparse
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.exp_sweep_utils import (
    _list_samples,
    _METRIC_KEYS_CONV,
    _METRIC_KEYS_PERF,
    _read_nested,
)


def _parse_run_range(run_range_str: str) -> Tuple[int, int]:
    """解析 '0-2' 或 '0-4' 格式的 run 范围."""
    parts = run_range_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"run-range 格式应为 'start-end', 得到: {run_range_str}")
    return int(parts[0]), int(parts[1])


def _list_run_dirs_for_sample_in_range(
    exp_dir: str, index: str, run_start: int, run_end: int
) -> List[str]:
    """返回指定 run 范围内某样本的 run 目录绝对路径列表."""
    run_dirs: List[str] = []
    for r in range(run_start, run_end + 1):
        rd = os.path.join(exp_dir, f'run_{r}', index)
        if os.path.isdir(rd):
            run_dirs.append(rd)
    return run_dirs


def aggregate_runs_in_range(
    exp_dir: str, index: str, run_start: int, run_end: int
) -> Dict[str, Dict[str, Any]]:
    """聚合某样本在指定 run 范围内的指标.

    逻辑与 utils/exp_sweep_utils.aggregate_runs 一致, 但只读取 run_start~run_end.
    """
    run_dirs = _list_run_dirs_for_sample_in_range(exp_dir, index, run_start, run_end)
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
        if not vals:
            continue
        mean = float(np.mean(vals))
        n = len(vals)
        if n >= 2:
            std = float(np.std(vals, ddof=1))
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            ci_half = t_crit * std / np.sqrt(n)
        else:
            std = 0.0
            ci_half = 0.0
        out[m] = {
            'mean': mean,
            'std': std,
            'variance': std ** 2,
            'cv_pct': abs(std / mean * 100.0) if mean != 0 else float('inf'),
            'ci95_lower': mean - ci_half,
            'ci95_upper': mean + ci_half,
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'n': n,
            'runs': vals,
        }
    return out


def classify_reliability(cv_pct: float, n: int) -> str:
    """根据 CV% 和 run 数判定可靠性.

    - n < 2:  insufficient_runs
    - cv <= 5%: highly_reliable
    - cv <= 10%: reliable
    - cv <= 20%: moderate
    - cv > 20%: unreliable
    """
    if n < 2:
        return 'insufficient_runs'
    if cv_pct <= 5.0:
        return 'highly_reliable'
    if cv_pct <= 10.0:
        return 'reliable'
    if cv_pct <= 20.0:
        return 'moderate'
    return 'unreliable'


def _has_run_dirs(path: str) -> bool:
    """检查目录是否直接包含 run_* 子目录."""
    if not os.path.isdir(path):
        return False
    for entry in os.listdir(path):
        if entry.startswith('run_') and os.path.isdir(os.path.join(path, entry)):
            return True
    return False


def _find_exp_subdirs(parent_dir: str) -> List[str]:
    """返回 parent_dir 下所有包含 run_* 子目录的实验子目录绝对路径."""
    subdirs: List[str] = []
    for entry in sorted(os.listdir(parent_dir)):
        sub = os.path.join(parent_dir, entry)
        if os.path.isdir(sub) and _has_run_dirs(sub):
            subdirs.append(sub)
    return subdirs


def build_reliability_rows(
    exp_dir: str,
    run_start: int,
    run_end: int,
    metrics_filter: Optional[List[str]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """为单个实验目录构建可靠性 CSV 行.

    返回: (rows, n_samples, n_empty)
    """
    samples = _list_samples(exp_dir)
    rows: List[Dict[str, Any]] = []
    n_empty = 0
    for idx in samples:
        agg = aggregate_runs_in_range(exp_dir, idx, run_start, run_end)
        if not agg:
            n_empty += 1
            continue
        for metric, stat in agg.items():
            if metrics_filter and metric not in metrics_filter:
                continue
            rows.append({
                'sample_id': idx,
                'metric': metric,
                'n_runs': stat['n'],
                'mean': round(stat['mean'], 6),
                'std': round(stat['std'], 6),
                'variance': round(stat['variance'], 6),
                'cv_pct': round(stat['cv_pct'], 4) if stat['cv_pct'] != float('inf') else 'inf',
                'ci95_lower': round(stat['ci95_lower'], 6),
                'ci95_upper': round(stat['ci95_upper'], 6),
                'min': round(stat['min'], 6),
                'max': round(stat['max'], 6),
                'reliability': classify_reliability(stat['cv_pct'], stat['n']),
            })
    return rows, len(samples), n_empty


def write_reliability_csv(
    output: str,
    rows: List[Dict[str, Any]],
    extra_keys: Optional[List[str]] = None,
) -> None:
    """将可靠性行写入 CSV."""
    base_keys = ['sample_id', 'metric', 'n_runs', 'mean', 'std', 'variance', 'cv_pct',
                 'ci95_lower', 'ci95_upper', 'min', 'max', 'reliability']
    keys = (extra_keys or []) + base_keys
    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def check_all_subdirs_reliability(
    parent_dir: str,
    run_range_str: str = '0-2',
    metrics_filter: Optional[List[str]] = None,
    output: Optional[str] = None,
) -> str:
    """批量检查 parent_dir 下所有子实验目录的可靠性.

    自动识别 parent_dir 下直接包含 run_* 子目录的文件夹, 分别统计后合并输出一个 CSV.
    返回最终输出文件路径.
    """
    if not os.path.isdir(parent_dir):
        raise NotADirectoryError(f"目录不存在: {parent_dir}")

    run_start, run_end = _parse_run_range(run_range_str)
    subdirs = _find_exp_subdirs(parent_dir)
    if not subdirs:
        raise NotADirectoryError(
            f"未在 {parent_dir} 的子目录中找到包含 run_* 的实验目录"
        )

    all_rows: List[Dict[str, Any]] = []
    print(f"父目录: {parent_dir}")
    print(f"发现实验子目录: {len(subdirs)}, run 范围: run_{run_start} ~ run_{run_end}")

    for exp_dir in subdirs:
        group_name = os.path.basename(exp_dir)
        rows, n_samples, n_empty = build_reliability_rows(
            exp_dir, run_start, run_end, metrics_filter
        )
        print(f"  [{group_name}] 样本数: {n_samples}, 有效记录: {len(rows)}, 空样本: {n_empty}")
        if n_empty:
            print(f"    [warn] {n_empty} 个样本在指定 run 范围内无数据")
        for r in rows:
            r['exp_group'] = group_name
        all_rows.extend(rows)

    output = output or os.path.join(
        parent_dir, f'all_groups_reliability_run{run_start}-{run_end}.csv'
    )
    write_reliability_csv(output, all_rows, extra_keys=['exp_group'])

    reliable_count = sum(
        1 for r in all_rows if r['reliability'] in ('highly_reliable', 'reliable')
    )
    print(f"\n总记录数: {len(all_rows)}")
    print(f"高可靠/可靠记录数: {reliable_count}")
    print(f"输出文件: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description='检查指定 run 范围内各样本的统计稳定性')
    parser.add_argument('exp_dir', help='实验目录或父目录, 如 results/tornado/20260606_033319/A_baseline 或 results/tornado/20260606_033319')
    parser.add_argument('--run-range', default='0-2',
                        help="run 范围, 如 '0-2' 或 '0-4' (默认: 0-2)")
    parser.add_argument('--metrics', default=None,
                        help='逗号分隔的指标列表, 默认输出所有可用指标')
    parser.add_argument('--output', default=None,
                        help='输出 CSV 路径, 默认单目录模式写入 <exp_dir>/reliability_run<start>-<end>.csv, '
                             '批量模式写入 <parent>/all_groups_reliability_run<start>-<end>.csv')
    args = parser.parse_args()

    exp_dir = args.exp_dir
    if not os.path.isdir(exp_dir):
        raise NotADirectoryError(f"目录不存在: {exp_dir}")

    run_start, run_end = _parse_run_range(args.run_range)
    metrics_filter = [m.strip() for m in args.metrics.split(',')] if args.metrics else None

    # 若 exp_dir 本身就是实验目录（直接含 run_*）, 走单目录模式；否则尝试批量处理子目录
    if _has_run_dirs(exp_dir):
        rows, n_samples, n_empty = build_reliability_rows(
            exp_dir, run_start, run_end, metrics_filter
        )
        print(f"实验目录: {exp_dir}")
        print(f"样本数: {n_samples}, run 范围: run_{run_start} ~ run_{run_end}")
        if n_empty:
            print(f"  [warn] {n_empty} 个样本在指定 run 范围内无数据")

        output = args.output or os.path.join(exp_dir, f'reliability_run{run_start}-{run_end}.csv')
        write_reliability_csv(output, rows)

        reliable_count = sum(
            1 for r in rows if r['reliability'] in ('highly_reliable', 'reliable')
        )
        print(f"\n总记录数: {len(rows)}")
        print(f"高可靠/可靠记录数: {reliable_count}")
        print(f"输出文件: {output}")
    else:
        check_all_subdirs_reliability(
            exp_dir,
            run_range_str=args.run_range,
            metrics_filter=metrics_filter,
            output=args.output,
        )


if __name__ == '__main__':
    main()
