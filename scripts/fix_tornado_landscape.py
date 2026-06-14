# -*- coding: utf-8 -*-
"""修复 tornado / sensitivity / ablation 参数扫描实验的 Landscape 指标缺失.

autodl_fix.py 是针对 SA/GA 单 runner 结果结构写的, 不兼容参数扫描实验的
run_* 层级. 本脚本遍历 <experiment_root>/<exp_name>/run_*/<index>/plans_<idx>
结构, 为每个样本补算 Landscape 指标并写回 performance.json.

用法:
    python scripts/fix_tornado_landscape.py results/tornado/20260611_023835

需要:
    - fragstats 已通过 frg_cmd 环境变量配置
    - libs/frg_model.fca 存在
    - RAW_DATA_ROOT 下有对应年份的 .tfw 文件
"""
import os
import sys
import json
import shutil
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.performance import only_landscape
from utils.dataset_utils import load_tiff
from utils.performance import calculate_landuse_area_statistics
from libs.const import RAW_DATA_ROOT


def _get_index_from_sample_dir(sample_dir: str) -> str:
    """从 .../10_1417_5/plans_1417 中提取 10_1417_5."""
    # sample_dir: <root>/<exp>/run_r/10_1417_5/plans_1417
    parts = sample_dir.replace('\\', '/').split('/')
    return parts[-2]


def _find_plan_dirs(experiment_root: str) -> List[str]:
    """递归查找所有 plans_<idx> 目录."""
    plan_dirs = []
    for root, dirs, _ in os.walk(experiment_root):
        for d in dirs:
            if d.startswith('plans_'):
                plan_dirs.append(os.path.join(root, d))
    return sorted(plan_dirs)


def _copy_tfw_for_sample(index: str, plan_dir: str) -> None:
    """从 RAW_DATA_ROOT 复制 .tfw 到 plan_dir 下的三份 TIF."""
    parts = index.split('_')
    if len(parts) < 2:
        print(f"  [warn] 无法解析 index: {index}")
        return
    year, idx = parts[0], parts[1]

    src_folder = os.path.join(RAW_DATA_ROOT, f"LU{year}")
    tfw_files = [f for f in os.listdir(src_folder)
                 if f.startswith(f"LU{year}_{idx}.") and f.endswith(".tfw")]
    if not tfw_files:
        print(f"  [warn] 未找到 {index} 的 TFW 文件")
        return
    if len(tfw_files) > 1:
        print(f"  [warn] {index} 找到多个 TFW, 使用第一个: {tfw_files[0]}")

    src = os.path.join(src_folder, tfw_files[0])
    for name in ('best_plan.tfw', 'ori_plan.tfw', 'tgt_plan.tfw'):
        dst = os.path.join(plan_dir, name)
        shutil.copy(src, dst)


def _calc_landscape(plan_dir: str, frg_model_path: str) -> dict:
    """调用 only_landscape 计算 Landscape 指标."""
    index = _get_index_from_sample_dir(plan_dir)
    idx = index.split('_')[1]
    sample_dir = os.path.dirname(plan_dir)
    tmp_folder = os.path.join(sample_dir, f"performance_{idx}")
    os.makedirs(tmp_folder, exist_ok=True)
    return only_landscape(src_folder=plan_dir, tmp_folder=tmp_folder,
                          frg_model_path=frg_model_path)


def _update_performance_json(plan_dir: str, metrics: dict) -> None:
    """把 Landscape.land 写回 performance.json."""
    sample_dir = os.path.dirname(plan_dir)
    perf_path = os.path.join(sample_dir, "performance.json")

    if not os.path.exists(perf_path):
        print(f"  [warn] performance.json 不存在: {perf_path}")
        return

    with open(perf_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data.setdefault('Landscape', {})['land'] = metrics

    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _fix_land_change(plan_dir: str) -> None:
    """同步补算 FINAL.best_ori / FINAL.best_tgt（若缺失）."""
    sample_dir = os.path.dirname(plan_dir)
    perf_path = os.path.join(sample_dir, "performance.json")
    if not os.path.exists(perf_path):
        return

    with open(perf_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果已有则跳过
    if data.get('FINAL', {}).get('best_ori') and data.get('FINAL', {}).get('best_tgt'):
        return

    try:
        ori = load_tiff(os.path.join(plan_dir, 'ori_plan.TIF'))
        best = load_tiff(os.path.join(plan_dir, 'best_plan.TIF'))
        tgt = load_tiff(os.path.join(plan_dir, 'tgt_plan.TIF'))
    except Exception as e:
        print(f"  [warn] 读取 TIF 失败, 跳过 landuse 变化补算: {e}")
        return

    data.setdefault('FINAL', {})['best_ori'] = calculate_landuse_area_statistics(
        ori, 'ori_plan', best, 'best_plan')
    data['FINAL']['best_tgt'] = calculate_landuse_area_statistics(
        tgt, 'tgt_plan', best, 'best_plan')

    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def fix_experiment_landscape(experiment_root: str,
                             frg_model_path: str = "libs/frg_model.fca") -> None:
    """修复整个实验目录下所有样本的 Landscape 指标."""
    plan_dirs = _find_plan_dirs(experiment_root)
    print(f"找到 {len(plan_dirs)} 个 plans 目录待修复")

    for i, plan_dir in enumerate(plan_dirs, 1):
        index = _get_index_from_sample_dir(plan_dir)
        print(f"\n[{i}/{len(plan_dirs)}] {index}")

        # 1. 复制 TFW
        _copy_tfw_for_sample(index, plan_dir)

        # 2. 计算 Landscape
        try:
            metrics = _calc_landscape(plan_dir, frg_model_path)
        except Exception as e:
            print(f"  [error] Landscape 计算失败: {e}")
            continue

        # 3. 写回 performance.json
        _update_performance_json(plan_dir, metrics)
        print(f"  已更新 Landscape: {list(metrics.keys())}")

        # 4. 可选: 补算 landuse 变化
        _fix_land_change(plan_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/fix_tornado_landscape.py <experiment_root>")
        sys.exit(1)

    experiment_root = sys.argv[1]
    if not os.path.isdir(experiment_root):
        raise NotADirectoryError(f"目录不存在: {experiment_root}")

    fix_experiment_landscape(experiment_root)
    print("\n修复完成。请重新运行 tornado_plots 后处理以更新 Landscape 相关图表。")
