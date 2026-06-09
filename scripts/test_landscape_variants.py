# 临时脚本: 测试 cal_landscape_score 三种变体在 SA 上的表现
# 从 representative_samples.json 的 all_records 中每个 n_pred 等级选一个样本,
# 分别用三种 landscape score 函数各跑一次 SA, 比较结果。
#
# 用法:
#   cd 到项目根目录, 运行:
#   python scripts/test_landscape_variants.py
import os
import sys
import json
import copy
import random
import time
from datetime import datetime
from typing import List, Dict, Any

import torch
import numpy as np
from torch.utils.data import DataLoader

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner import create_runner
from datasets.base_dataloader import get_dataset
from utils.calculator_utils import train_maxent
from utils.tools import get_yaml, ensure_folder, _set_by_path
from utils.config_utils import load_score_cfg
from pipelines.experiment import process_batch
from libs.const import apply_const_overrides, restore_const_defaults, PIXEL_NUM_THRESHOLD_ISCHANGE

# Landscape score 三种变体
import calculator.tgt_function_cal as tgt_module
from calculator.cal_landscape_score import (
    cal_landscape_score as cal_landscape_original,
    cal_landscape_score_v2_minimal,
    cal_landscape_score_v2,
)

# 变体名称 → 函数映射
LANDSCAPE_VARIANTS: Dict[str, Any] = {
    "original":   cal_landscape_original,
    "v2_minimal": cal_landscape_score_v2_minimal,
    "v2":         cal_landscape_score_v2,
}


def select_samples(json_path: str) -> List[Dict[str, Any]]:
    """从 representative_samples.json 的 all_records 中每个级别选一个样本.

    级别划分依据文件内 quantile_ranges (5 个 n_pred 区间).
    排除 n_pred == 0 的记录 (SA 会跳过无变化样本).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_records = data['all_records']
    quantile_ranges = data['quantile_ranges']

    selected: List[Dict[str, Any]] = []
    used_indices: set = set()

    for level_str in sorted(quantile_ranges.keys(), key=int):
        level_info = quantile_ranges[level_str]
        lo, hi = level_info['min'], level_info['max']
        level = int(level_str)

        for record in all_records:
            idx = record['index']
            n_pred = record['n_pred']

            if idx in used_indices:
                continue
            if n_pred == 0:
                continue
            if n_pred < lo or n_pred > hi:
                continue

            record_copy = dict(record)
            record_copy['level'] = level
            selected.append(record_copy)
            used_indices.add(idx)
            print(f"  Level {level}: 选中 {idx} (n_pred={n_pred}, n_lu_types={record['n_lu_types']})")
            break
        else:
            print(f"  [warn] Level {level}: 未找到匹配样本 (range [{lo}, {hi}])")

    return selected


def run_single_sa(
    opt_cfg: Any,
    score_cfg: Any,
    device: torch.device,
    runner_config: Any,
    type_config_base: Any,
    weight_dict: Dict[str, float],
    use_landscape: bool,
    data_dict: Dict[str, torch.Tensor],
    index: str,
    save_path: str,
    variant_name: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """以指定 landscape 变体运行单次 SA, 返回收敛摘要.

    Parameters
    ----------
    variant_name : str
        "original" / "v2_minimal" / "v2"
    seed : int
        确定性种子, 确保同一样本三个变体的随机初始条件一致
    """
    # 保存原始函数引用, 确保 run 结束后恢复
    original_fn = tgt_module.cal_landscape_score

    try:
        # Monkey-patch: 替换 cal_tgt_function 内部引用的 landscape 函数
        tgt_module.cal_landscape_score = LANDSCAPE_VARIANTS[variant_name]

        # 深拷贝配置, 避免跨变体污染
        score_cfg_copy = copy.deepcopy(score_cfg)
        type_config = copy.deepcopy(type_config_base)

        apply_const_overrides(score_cfg_copy)

        # 确定性种子 (与 experiment_base 保持一致)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        runner = create_runner(
            runner_type=runner_config.type,
            config=type_config,
            common_config=runner_config.common_params,
            model_config=opt_cfg.model,
            device=device,
            data_dict=data_dict,
            index=index,
            save_path=save_path,
            weight_dict=weight_dict,
            use_landscape=use_landscape,
            guided=True,
            disable_early_stop=False,
            score_cfg=score_cfg_copy,
        )

        t0 = time.perf_counter()
        best_plan = runner.run()
        wall_time = time.perf_counter() - t0

        # 收集结果
        result: Dict[str, Any] = {
            'variant': variant_name,
            'index': index,
            'wall_time_sec': round(wall_time, 2),
            'best_score': float(runner.best_score),
            'best_score_iter': runner.best_score_iter,
            'total_iter': len(runner.total_score_list),
            'pred_diff_pixelnum': runner.pred_diff_pixelnum,
            'skipped': runner.pred_diff_pixelnum < PIXEL_NUM_THRESHOLD_ISCHANGE,
        }

        if runner.total_score_list:
            result['total_score_initial'] = float(runner.total_score_list[0])
            result['total_score_final']   = float(runner.total_score_list[-1])
        if runner.change_score_list:
            result['change_score_initial'] = float(runner.change_score_list[0])
            result['change_score_final']   = float(runner.change_score_list[-1])
        if runner.es_score_list:
            result['es_score_initial'] = float(runner.es_score_list[0])
            result['es_score_final']   = float(runner.es_score_list[-1])
        if runner.landscape_score_list:
            result['landscape_score_initial'] = float(runner.landscape_score_list[0])
            result['landscape_score_final']   = float(runner.landscape_score_list[-1])

        # 判断是否最优解有实际变化 (非原始方案)
        optimized = not torch.equal(best_plan, runner.ori_plan) if not result['skipped'] else False
        result['optimized'] = optimized

        return result

    finally:
        # 恢复原始函数, 防止跨变体泄漏
        tgt_module.cal_landscape_score = original_fn
        restore_const_defaults()
        # 释放 GPU 缓存, 避免 15 次运行累积显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_summary(all_results: List[Dict[str, Any]]) -> None:
    """打印对比摘要表格."""
    print(f"\n{'=' * 110}")
    print("对比摘要: 三种 Landscape Score 变体 × 5 样本")
    print(f"{'=' * 110}")

    header = (f"{'Sample':<16} {'Level':>5} {'n_pred':>6} "
              f"{'Variant':<12} {'Best':>12} "
              f"{'LS_init':>10} {'LS_final':>10} "
              f"{'Iter':>7} {'Time':>8} {'Skip':>5}")
    print(header)
    print("-" * 110)

    for r in all_results:
        sample_idx = r['index'].split('_')[1] if '_' in r['index'] else r['index']
        ls_init = f"{r.get('landscape_score_initial', 0):.2f}" if r.get('landscape_score_initial') is not None else 'N/A'
        ls_final = f"{r.get('landscape_score_final', 0):.2f}" if r.get('landscape_score_final') is not None else 'N/A'
        print(f"{sample_idx:<16} {r['sample_level']:>5} {r['n_pred']:>6} "
              f"{r['variant']:<12} {r['best_score']:>12.4f} "
              f"{ls_init:>10} {ls_final:>10} "
              f"{r['total_iter']:>7} {r['wall_time_sec']:>7.1f}s {'Y' if r['skipped'] else '':>5}")

    print("-" * 110)

    # 按样本分组, 计算 v2 相对 original 的变化百分比
    print(f"\n{'=' * 110}")
    print("Landscape Score 最终值对比 (v2 变体相对于 original 的变化)")
    print(f"{'=' * 110}")

    samples = sorted(set(r['index'] for r in all_results),
                     key=lambda s: next(r['sample_level'] for r in all_results if r['index'] == s))
    for sample in samples:
        sample_results = [r for r in all_results if r['index'] == sample]
        orig = next((r for r in sample_results if r['variant'] == 'original'), None)

        print(f"\n  {sample} (level={sample_results[0]['sample_level']}, n_pred={sample_results[0]['n_pred']}):")
        for r in sample_results:
            ls_final = r.get('landscape_score_final')
            ls_str = f"{ls_final:.4f}" if ls_final is not None else 'N/A'
            skipped = ' [SKIPPED]' if r['skipped'] else ''

            if orig and not r['skipped'] and not orig['skipped']:
                orig_ls = orig.get('landscape_score_final')
                if orig_ls and orig_ls != 0 and ls_final is not None:
                    delta_pct = (ls_final - orig_ls) / abs(orig_ls) * 100
                    print(f"    {r['variant']:<12}: LS={ls_str}, best={r['best_score']:.4f}, "
                          f"ΔLS={delta_pct:+.1f}%{skipped}")
                    continue
            print(f"    {r['variant']:<12}: LS={ls_str}, best={r['best_score']:.4f}{skipped}")

    print(f"\n{'=' * 110}")


def main() -> None:
    opt_cfg_path = os.path.join("configs", "optimizerConfig.yaml")
    score_cfg_path = os.path.join("configs", "score_cfg.yaml")
    rep_json = os.path.join("_data", "representative_samples.json")

    print("=" * 60)
    print("Landscape Score 变体对比测试")
    print("  对比: original vs v2_minimal vs v2")
    print("  样本: 每个 n_pred 等级各 1 个 (共 5 个)")
    print("  优化器: SA (guided=True, early_stop=False)")
    print("=" * 60)

    # ============================================================
    # 1. 选择样本
    # ============================================================
    print("\n[1] 选择测试样本...")
    selected_samples = select_samples(rep_json)
    print(f"  共选中 {len(selected_samples)} 个样本\n")

    if not selected_samples:
        print("[error] 未选中任何样本, 退出")
        return

    # ============================================================
    # 2. 加载配置
    # ============================================================
    print("[2] 加载配置...")
    opt_cfg = get_yaml(opt_cfg_path)
    score_cfg = load_score_cfg(score_cfg_path)

    # 为 v2 变体补充缺失的配置字段 (lambda_neighbor / lambda_patch / lambda_distance)
    v2_defaults = {
        'lambda_neighbor': 1.0,
        'lambda_patch': 1.0,
        'lambda_distance': 1.0,
    }
    for field, default_val in v2_defaults.items():
        if not hasattr(score_cfg.landscape, field):
            _set_by_path(score_cfg, f'landscape.{field}', default_val)
            print(f"  添加默认配置: landscape.{field} = {default_val}")

    print(f"  opt_cfg: {opt_cfg_path}")
    print(f"  score_cfg: {score_cfg_path}")

    # ============================================================
    # 3. 设备 + 数据 + MaxEnt
    # ============================================================
    print("\n[3] 初始化...")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and opt_cfg.device.use_gpu else "cpu"
    )
    print(f"  设备: {device}")

    dataset = get_dataset(type="optimize", data_config=opt_cfg.data)
    dataloader = DataLoader(
        dataset,
        batch_size=opt_cfg.data.batch_size,
        shuffle=opt_cfg.data.shuffle,
    )
    print(f"  数据集: {len(dataloader)} 个样本")

    maxent_path = opt_cfg.data.maxent_model_path.format(year=opt_cfg.data.year)
    train_maxent(opt_cfg.data.root, opt_cfg.data.year, maxent_path)

    runner_config = opt_cfg.runner
    use_landscape = runner_config.use_landscape
    type_config_base = getattr(opt_cfg, runner_config.type)
    scenario = runner_config.scenario if runner_config.scenario != 'all' else 'default'
    weight_dict = vars(getattr(runner_config.weight, scenario))

    print(f"  runner 类型: {runner_config.type}")
    print(f"  scenario: {scenario}, ES权重: {weight_dict}")
    print(f"  use_landscape: {use_landscape}")
    print(f"  SA T_start={type_config_base.T_start}, T_end={type_config_base.T_end}, "
          f"cooling_rate={type_config_base.cooling_rate}")

    # ============================================================
    # 4. 遍历运行
    # ============================================================
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_save = os.path.join("results", "landscape_test", run_ts)
    ensure_folder(root_save)
    print(f"\n  结果保存至: {root_save}")

    # 构建 index → data_pack 查找表 (避免多次遍历 dataloader)
    index_to_datapack: Dict[str, Any] = {}
    for data_pack in dataloader:
        idx = data_pack[5][0]
        index_to_datapack[idx] = data_pack

    all_results: List[Dict[str, Any]] = []
    variant_names = ["original", "v2_minimal", "v2"]

    for s_idx, sample in enumerate(selected_samples):
        index = sample['index']
        print(f"\n{'=' * 60}")
        print(f">>> 样本 {s_idx + 1}/{len(selected_samples)}: {index} "
              f"(level={sample['level']}, n_pred={sample['n_pred']}, "
              f"n_lu_types={sample['n_lu_types']})")
        print(f"{'=' * 60}")

        if index not in index_to_datapack:
            print(f"  [warn] 样本 {index} 不在 dataloader 中, 跳过")
            continue

        data_pack = index_to_datapack[index]
        data_dict = process_batch(data_pack, opt_cfg)

        # 确定性种子: 基于 index 的 hash, 三个变体共享
        base_seed = abs(hash(index)) & 0x7FFFFFFF

        for variant_name in variant_names:
            print(f"\n  --- Variant: {variant_name} ---")

            save_path = os.path.join(root_save, index, variant_name)
            ensure_folder(save_path)

            result = run_single_sa(
                opt_cfg=opt_cfg,
                score_cfg=score_cfg,
                device=device,
                runner_config=runner_config,
                type_config_base=type_config_base,
                weight_dict=weight_dict,
                use_landscape=use_landscape,
                data_dict=data_dict,
                index=index,
                save_path=save_path,
                variant_name=variant_name,
                seed=base_seed,
            )

            # 附加样本元数据
            result['sample_level'] = sample['level']
            result['n_pred'] = sample['n_pred']
            result['n_lu_types'] = sample['n_lu_types']
            result['seed'] = base_seed
            all_results.append(result)

            skipped_tag = " [SKIPPED - 无变化]" if result['skipped'] else ""
            print(f"     Best Score: {result['best_score']:.4f}, "
                  f"Iter: {result['total_iter']}, "
                  f"Time: {result['wall_time_sec']:.1f}s{skipped_tag}")

            if not result['skipped']:
                ls_init = result.get('landscape_score_initial')
                ls_final = result.get('landscape_score_final')
                print(f"     landscape: {ls_init:.4f} -> {ls_final:.4f}"
                      if ls_init is not None else
                      f"     landscape_final: {ls_final:.4f}")
                if result.get('change_score_initial') is not None:
                    print(f"     change: {result['change_score_initial']:.4f} -> {result['change_score_final']:.4f}")
                if result.get('es_score_initial') is not None:
                    print(f"     ES:     {result['es_score_initial']:.4f} -> {result['es_score_final']:.4f}")

    # ============================================================
    # 5. 保存并打印结果
    # ============================================================
    print(f"\n{'=' * 60}")
    print("[5] 保存结果...")

    comparison_path = os.path.join(root_save, "comparison.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  对比结果已保存至: {comparison_path}")

    # 也保存样本选择信息
    samples_path = os.path.join(root_save, "selected_samples.json")
    with open(samples_path, 'w', encoding='utf-8') as f:
        json.dump(selected_samples, f, indent=2, ensure_ascii=False)
    print(f"  样本选择信息已保存至: {samples_path}")

    print_summary(all_results)

    print(f"\n{'=' * 60}")
    print(f"测试完成! 共 {len(all_results)} 次运行")
    print(f"结果目录: {root_save}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
