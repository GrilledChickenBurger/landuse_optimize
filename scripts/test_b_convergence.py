# 临时脚本: 使用 SA + 实验B (仅数量限制) 目标函数测试收敛性
# 用法:
#   cd 到项目根目录, 运行:
#   python scripts/test_b_convergence.py              # 快速测试 (2样本, ~2000代/样本)
#   python scripts/test_b_convergence.py --full       # 完整收敛 (2样本, ~37000代/样本)
#   python scripts/test_b_convergence.py --samples 479 596  # 指定样本索引
#   python scripts/test_b_convergence.py --full --n 1        # 单样本完整收敛
import os
import sys
import copy
from datetime import datetime
from typing import List

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
from libs.const import apply_const_overrides, restore_const_defaults


def test_b_convergence(
    opt_cfg_path: str = "configs/optimizerConfig.yaml",
    score_cfg_path: str = "configs/score_cfg.yaml",
    sample_indices: List[str] = None,
    n_samples: int = 2,
    quick: bool = True,
):
    """测试实验B (仅数量限制) 目标函数的SA收敛性。

    Parameters
    ----------
    opt_cfg_path : str
        优化器配置文件路径
    score_cfg_path : str
        评分配置文件路径
    sample_indices : List[str] | None
        指定样本索引 (如 ['479', '596']), None 则自动取前 n_samples 个
    n_samples : int
        测试样本数 (sample_indices 为 None 时生效)
    """
    # ============================================================
    # 1. 加载配置
    # ============================================================
    print("=" * 60)
    print("实验B收敛性测试 — SA + 仅数量限制目标函数")
    print("=" * 60)

    opt_cfg = get_yaml(opt_cfg_path)
    score_cfg = load_score_cfg(score_cfg_path)

    # 覆写为实验B的真实控制变量: 目标函数 P_change 项切换为纯数量惩罚
    # (guided / disable_early_stop 通过 SA 构造函数传递, 不在 score_cfg 中)
    _set_by_path(score_cfg, "ablation.disable_change_score", True)
    apply_const_overrides(score_cfg)

    print(f"\n实验B配置覆盖:")
    print(f"  ablation.disable_change_score = {score_cfg.ablation.disable_change_score}")
    print(f"  quantity_penalty_C = {score_cfg.ablation.quantity_penalty_C}")
    print(f"  quantity_penalty_delta = {score_cfg.ablation.quantity_penalty_delta}")
    print(f"  quantity_penalty_power = {score_cfg.ablation.quantity_penalty_power}")

    # ============================================================
    # 2. 设备 + 数据 + MaxEnt
    # ============================================================
    device = torch.device(
        "cuda" if torch.cuda.is_available() and opt_cfg.device.use_gpu else "cpu"
    )
    print(f"\n设备: {device}")

    dataset = get_dataset(type="optimize", data_config=opt_cfg.data)
    dataloader = DataLoader(
        dataset,
        batch_size=opt_cfg.data.batch_size,
        shuffle=opt_cfg.data.shuffle,
    )
    print(f"数据集大小: {len(dataloader)} 个样本")

    # MaxEnt
    maxent_path = opt_cfg.data.maxent_model_path.format(year=opt_cfg.data.year)
    train_maxent(opt_cfg.data.root, opt_cfg.data.year, maxent_path)

    # ============================================================
    # 3. SA 配置 (实验B)
    # ============================================================
    runner_config = opt_cfg.runner
    use_landscape = runner_config.use_landscape
    type_config = copy.deepcopy(getattr(opt_cfg, runner_config.type))

    if quick:
        # 快速测试模式: 使用默认 T_end + 更快的冷却率 -> ~2000 迭代
        type_config.T_end = 0.1
        type_config.cooling_rate = 0.997
        print(f"\n[快速测试模式] T_end={type_config.T_end}, cooling_rate={type_config.cooling_rate}")
    else:
        # 完整收敛模式: 极端降温 -> ~37000 迭代
        type_config.T_end = 1e-6
        print(f"\n[完整收敛模式] T_end={type_config.T_end}, cooling_rate={type_config.cooling_rate}")

    scenario = runner_config.scenario if runner_config.scenario != 'all' else 'default'
    weight_dict = vars(getattr(runner_config.weight, scenario))
    print(f"\nscenario: {scenario}, ES权重: {weight_dict}")
    print(f"use_landscape: {use_landscape}")
    print(f"SA T_start={type_config.T_start}, T_end={type_config.T_end}, "
          f"cooling_rate={type_config.cooling_rate}, "
          f"max_noimprove_iter={type_config.max_noimprove_iter}")

    # ============================================================
    # 4. 结果目录
    # ============================================================
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_save = os.path.join("results", "test_b_convergence", run_ts)
    ensure_folder(root_save)
    print(f"\n结果保存至: {root_save}")

    # ============================================================
    # 5. 遍历样本运行SA
    # ============================================================
    tested = 0
    for i, data_pack in enumerate(dataloader):
        index = data_pack[5][0]

        # 按指定索引过滤
        if sample_indices:
            idx_part = index.split('_')[1] if '_' in index else index
            if idx_part not in sample_indices:
                continue
        elif tested >= n_samples:
            break

        print(f"\n{'=' * 60}")
        print(f">>> 样本 {index} ({tested + 1}/{n_samples if not sample_indices else len(sample_indices)})")
        print(f"{'=' * 60}")

        data_dict = process_batch(data_pack, opt_cfg)

        # 构造 SA runner (实验B参数)
        runner = create_runner(
            runner_type=runner_config.type,
            config=type_config,
            common_config=runner_config.common_params,
            model_config=opt_cfg.model,
            device=device,
            data_dict=data_dict,
            index=index,
            save_path=root_save,
            weight_dict=weight_dict,
            use_landscape=use_landscape,
            guided=False,                     # B实验: 无空间引导
            disable_early_stop=True,           # B实验: 禁用早停
            score_cfg=score_cfg,
        )

        # 运行 SA
        best_plan = runner.run()

        # 阈值检查: 若 pred_diff_pixelnum < PIXEL_NUM_THRESHOLD_ISCHANGE, SA 会直接返回 ori_plan
        from libs.const import PIXEL_NUM_THRESHOLD_ISCHANGE
        if runner.pred_diff_pixelnum < PIXEL_NUM_THRESHOLD_ISCHANGE:
            print(f"\n  [跳过] 样本 {index} pred_diff_pixelnum={runner.pred_diff_pixelnum} "
                  f"< 阈值 {PIXEL_NUM_THRESHOLD_ISCHANGE}, SA 未进入优化")
            tested += 1
            continue

        # 打印收敛摘要
        print(f"\n--- 收敛摘要 [{index}] ---")
        print(f"  pred_diff_pixelnum (target): {runner.pred_diff_pixelnum}")
        print(f"  总迭代次数: {len(runner.total_score_list)}")
        print(f"  最优目标函数值: {runner.best_score:.4f} (出现于 iter {runner.best_score_iter})")
        print(f"  初始目标函数值: {runner.total_score_list[0]:.4f}")
        print(f"  最终目标函数值: {runner.total_score_list[-1]:.4f}")
        print(f"  目标函数变化: {runner.total_score_list[-1] - runner.total_score_list[0]:.4f}")
        if runner.change_score_list:
            print(f"  初始 change_score(数量惩罚): {runner.change_score_list[0]:.4f}")
            print(f"  最终 change_score(数量惩罚): {runner.change_score_list[-1]:.4f}")
        if runner.es_score_list:
            print(f"  初始 ES_score: {runner.es_score_list[0]:.4f}")
            print(f"  最终 ES_score: {runner.es_score_list[-1]:.4f}")
        if runner.landscape_score_list:
            print(f"  初始 landscape_score: {runner.landscape_score_list[0]:.4f}")
            print(f"  最终 landscape_score: {runner.landscape_score_list[-1]:.4f}")

        # 检查数量限制是否进入死区 (变化像素数接近 target)
        pred_diff_pixelnum = runner.pred_diff_pixelnum
        best_diff = (best_plan != runner.ori_plan).sum().item()
        ratio = best_diff / max(pred_diff_pixelnum, 1)
        print(f"\n  最佳方案变化像素数: {best_diff}")
        print(f"  变化比例 (best/target): {ratio:.4f} (目标接近 1.0)")
        if abs(ratio - 1) <= score_cfg.ablation.quantity_penalty_delta:
            print(f"  ✓ 变化数量在软死区 (±{score_cfg.ablation.quantity_penalty_delta*100:.0f}%) 内!")
        else:
            print(f"  ✗ 变化数量在软死区外 (偏差: {abs(ratio-1)*100:.1f}%)")

        tested += 1

    # ============================================================
    # 6. 清理
    # ============================================================
    restore_const_defaults()
    print(f"\n{'=' * 60}")
    print(f"测试完成! 共测试 {tested} 个样本")
    print(f"结果保存至: {root_save}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SA + 实验B (仅数量限制) 收敛性测试")
    parser.add_argument("--full", action="store_true", help="完整收敛模式 (极端降温 T_end=1e-6)")
    parser.add_argument("--samples", type=str, nargs="+", default=None,
                        help="指定样本索引, 如: --samples 479 596")
    parser.add_argument("--n", type=int, default=2, help="测试样本数 (默认2, --samples 指定时忽略)")
    args = parser.parse_args()

    test_b_convergence(
        quick=not args.full,
        sample_indices=args.samples,
        n_samples=args.n if not args.samples else len(args.samples),
    )
