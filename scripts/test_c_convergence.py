# 临时脚本: 使用 SA + 实验C (随机连通块约束) 目标函数测试收敛性
# 用法:
#   cd 到项目根目录, 运行:
#   python scripts/test_c_convergence.py              # 快速测试 (2样本, ~2000代/样本)
#   python scripts/test_c_convergence.py --full       # 完整收敛 (2样本, ~37000代/样本)
#   python scripts/test_c_convergence.py --samples 479 596  # 指定样本索引
#   python scripts/test_c_convergence.py --full --n 1        # 单样本完整收敛
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
from utils.runner_utils import load_ANN_model, get_pred_diff
from utils.ablation_utils import make_random_connected_mask
from utils.tools import get_yaml, ensure_folder, _set_by_path
from utils.config_utils import load_score_cfg
from pipelines.experiment import process_batch
from libs.const import apply_const_overrides, restore_const_defaults, DEFAULT_CONDITION_LIST


def test_c_convergence(
    opt_cfg_path: str = "configs/optimizerConfig.yaml",
    score_cfg_path: str = "configs/score_cfg.yaml",
    sample_indices: List[str] = None,
    n_samples: int = 2,
    quick: bool = True,
):
    """测试实验C (随机连通块约束) 目标函数的SA收敛性。

    实验C: 用与 ANN pred_diff 连通块数/面积匹配的随机 0/1 mask 替代空间引导。
    约束 (random_restrict_mask) 和目标 (change_target) 同时替换为随机 mask，
    但 change_score 的空间引导机制仍保持启用（与实验B的纯数量惩罚不同）。

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
    quick : bool
        True 为快速测试 (~2000代/样本), False 为完整收敛 (~37000代/样本)
    """
    # ============================================================
    # 1. 加载配置
    # ============================================================
    print("=" * 60)
    print("实验C收敛性测试 — SA + 随机连通块约束目标函数")
    print("=" * 60)

    opt_cfg = get_yaml(opt_cfg_path)
    score_cfg = load_score_cfg(score_cfg_path)

    # 实验C 不需要 disable_change_score (与 B 的关键区别)
    # C 保持空间引导, 但引导来源是随机连通块 mask 而非 ANN 预测
    apply_const_overrides(score_cfg)

    print(f"\n实验C配置覆盖:")
    print(f"  ablation.disable_change_score = {score_cfg.ablation.disable_change_score}  (应为 False)")
    print(f"  guided = False (由 SA 构造参数控制)")
    print(f"  disable_early_stop = True (由 SA 构造参数控制)")
    print(f"  random_restrict_mask = 随机连通块 mask (与 pred_diff 块数/面积匹配, 位置随机)")

    # ============================================================
    # 2. 设备 + 数据 + MaxEnt + ANN 模型
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

    # ANN 模型 (加载一次, 所有样本共用)
    in_ch = len(opt_cfg.data.factor_list) + 6  # 6 类简化 LU 的 one-hot
    ann_model = load_ANN_model(
        opt_cfg.model.type, opt_cfg.model.model_path,
        in_ch, 1, device,
        getattr(opt_cfg.model, opt_cfg.model.type, None))
    print(f"ANN 模型: {opt_cfg.model.type}, 输入通道: {in_ch}")

    # ============================================================
    # 3. SA 配置 (实验C)
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
    root_save = os.path.join("results", "test_c_convergence", run_ts)
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

        # --------------------------------------------------------
        # 5a. 生成 ANN pred_diff + 随机连通块 mask
        # --------------------------------------------------------
        year = index.split('_')[0] if '_' in index else ''
        pred_diff = get_pred_diff(
            ann_model, data_dict['ori_plan'], data_dict['factors'], device,
            year=year, condition_list=DEFAULT_CONDITION_LIST,
            normalize=True,
            sigmoid_threshold=score_cfg.runner_utils.sigmoid_threshold)
        print(f"  pred_diff 预测变化像素数: {pred_diff.sum().item()}")

        # 硬约束区域: redline ∧ (LU≠0), 对齐到 pred_diff 所在设备
        redline = data_dict['restrict']
        lu_nonzero = (data_dict['ori_plan'] != 0)
        base_hard = torch.logical_and(
            redline.to(pred_diff.device), lu_nonzero.to(pred_diff.device))

        # 确定性种子 (与 ablation pipeline 一致: abs(hash(index)) & 0xFFFFFFFF)
        base_seed = (abs(hash(index)) & 0xFFFFFFFF)
        random_mask = make_random_connected_mask(pred_diff, base_hard, seed=base_seed)
        random_mask = random_mask.to(pred_diff.device)  # make_random_connected_mask 返回 CPU tensor
        print(f"  random_mask 变化像素数: {random_mask.sum().item()}")

        # 对齐校验
        ra_pred = torch.logical_and(base_hard, pred_diff.bool()).sum().item()
        ra_rand = torch.logical_and(base_hard, random_mask.bool()).sum().item()
        if ra_pred != ra_rand:
            print(f"  [warn] random_mask 像素数对齐失败 (RA∧pred={ra_pred}, RA∧random={ra_rand}); "
                  f"该样本 C 与 A/B 不可比")
        else:
            print(f"  ✓ random_mask 像素数对齐成功 (RA∧pred = RA∧random = {ra_pred})")

        # --------------------------------------------------------
        # 5b. 构造 SA runner (实验C参数)
        # --------------------------------------------------------
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
            guided=False,                     # C实验: 无ANN空间引导
            disable_early_stop=True,           # C实验: 禁用早停
            score_cfg=score_cfg,
            random_restrict_mask=random_mask,  # C实验核心: 随机连通块约束 + 目标
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
        print(f"  实际变化像素数 (random_mask): {random_mask.sum().item()}")
        print(f"  总迭代次数: {len(runner.total_score_list)}")
        print(f"  最优目标函数值: {runner.best_score:.4f} (出现于 iter {runner.best_score_iter})")
        print(f"  初始目标函数值: {runner.total_score_list[0]:.4f}")
        print(f"  最终目标函数值: {runner.total_score_list[-1]:.4f}")
        print(f"  目标函数变化: {runner.total_score_list[-1] - runner.total_score_list[0]:.4f}")
        if runner.change_score_list:
            print(f"  初始 change_score(空间引导): {runner.change_score_list[0]:.4f}")
            print(f"  最终 change_score(空间引导): {runner.change_score_list[-1]:.4f}")
        if runner.es_score_list:
            print(f"  初始 ES_score: {runner.es_score_list[0]:.4f}")
            print(f"  最终 ES_score: {runner.es_score_list[-1]:.4f}")
        if runner.landscape_score_list:
            print(f"  初始 landscape_score: {runner.landscape_score_list[0]:.4f}")
            print(f"  最终 landscape_score: {runner.landscape_score_list[-1]:.4f}")

        # 最佳方案变化像素覆盖分析
        pred_diff_pixelnum = runner.pred_diff_pixelnum
        best_diff = (best_plan != runner.ori_plan).sum().item()
        ratio = best_diff / max(pred_diff_pixelnum, 1)

        # 计算方案变化与随机 mask 的重叠
        if random_mask is not None:
            overlap = torch.logical_and(
                (best_plan != runner.ori_plan), random_mask.bool().squeeze()
            ).sum().item()
            overlap_ratio = overlap / max(random_mask.sum().item(), 1)
            print(f"\n  最佳方案变化像素数: {best_diff}")
            print(f"  变化比例 (best/target): {ratio:.4f}")
            print(f"  变化与 random_mask 重叠: {overlap} ({overlap_ratio:.4%})")

            # C 组关键指标: 方案变化区域与随机 mask 的重叠度
            # SA 被约束只能改 random_mask 内的像素, 所以重叠度应接近 100%
            if overlap_ratio >= 0.95:
                print(f"  ✓ 方案变更高度约束在随机 mask 范围内 (重叠 {overlap_ratio:.1%})")
            else:
                print(f"  [注意] 方案变更与随机 mask 重叠仅 {overlap_ratio:.1%}, 可能约束未生效")
        else:
            print(f"\n  最佳方案变化像素数: {best_diff}")
            print(f"  变化比例 (best/target): {ratio:.4f}")

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
    parser = argparse.ArgumentParser(description="SA + 实验C (随机连通块约束) 收敛性测试")
    parser.add_argument("--full", action="store_true", help="完整收敛模式 (极端降温 T_end=1e-6)")
    parser.add_argument("--samples", type=str, nargs="+", default=None,
                        help="指定样本索引, 如: --samples 479 596")
    parser.add_argument("--n", type=int, default=2, help="测试样本数 (默认2, --samples 指定时忽略)")
    args = parser.parse_args()

    test_c_convergence(
        quick=not args.full,
        sample_indices=args.samples,
        n_samples=args.n if not args.samples else len(args.samples),
    )
