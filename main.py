import os
import sys

# 容器里 nproc 常远大于实际 CPU 配额(如 128 vs 16), torch/numpy 默认按 nproc 开线程会造成
# OpenMP 忙等自旋抢满配额并饿住 GPU; 该现象只出现在 Linux 容器, 本机核数=配额则无需封顶.
# 必须在导入 numpy/torch 之前设置环境变量; setdefault 保证已显式设置的值优先.
_IS_LINUX = sys.platform.startswith('linux')
if _IS_LINUX:
    for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_thread_env, "1")

import random

import torch

# 主进程线程数: Linux 容器取小值(避免与 DataLoader worker 抢配额, worker 内另行压到 1);
# 本机核数即配额, 直接用满. 两种情况都可用 TORCH_NUM_THREADS 覆盖.
_DEFAULT_THREADS = 4 if _IS_LINUX else (os.cpu_count() or 4)
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", _DEFAULT_THREADS)))
# 确定性配置: 默认开启可复现模式(关 benchmark + deterministic 卷积), LU_DETERMINISM=0 关闭.
# benchmark=True 会按输入尺寸自动挑最快卷积算法, 但同一代码不同 run 可能选到不同算法,
# 是跨 run 不可复现的主要来源之一; deterministic=True 强制确定性卷积实现(略慢但可复现).
_DETERMINISM = os.environ.get("LU_DETERMINISM", "1") == "1"
if _DETERMINISM:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    # 输入尺寸固定(128x128), 让 cudnn 自动挑选最快卷积算法
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
# 强制确定性算子: 有确定性实现的算子(如 interpolate 反向)强制走确定性实现;
# 无确定性实现的算子按 warn_only 只告警一次(不中断训练, 但该算子仍可能非确定).
# 可用 LU_DETERMINISTIC_ALGO=0 关闭(仅在排查性能问题时建议关闭).
if _DETERMINISM and os.environ.get("LU_DETERMINISTIC_ALGO", "1") == "1":
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

from pipelines.run_runner import pipeline
from pipelines.experiment import experiment_pipeline
from pipelines.train import training_pipeline,valid_pipeline, get_changed_indices
from pipelines.hybrid_train import run_hybrid_train_pipeline
from pipelines.threshold_sweep import threshold_sweep_pipeline
from pipelines.statistic import get_params
from pipelines.ablation import ablation_pipeline
from pipelines.tornado import tornado_pipeline
from pipelines.results_analysis import (
    comparison_pipeline,
    model_comparison_pipeline,
    sample_visualization_pipeline,
    single_experiment_stat_pipeline,
)
from utils.exp_sweep_utils import load_representative_idxs, get_latest_experiment_dir
from utils.tornado_utils import DEFAULT_SENSITIVITY_PARAMS


def run_statistic_pipeline():
    config_path = os.path.join("configs","statisticConfig.yaml")
    get_params(config_path)

def run_runner_pipeline():
    config_path = os.path.join("configs","optimizerConfig.yaml")
    is_processed = False
    processed_timestamp = "20251230_212323"
    idxs = [760,1903,2207,479,596,634]
    pipeline(config_path, idxs, is_processed, processed_timestamp)

def run_experiment_pipeline():
    config_path = os.path.join("configs","optimizerConfig.yaml")
    is_processed = False
    processed_timestamp = "20251230_212323"
    idxs = [479,596,760,634,1903,2207,]
    experiment_pipeline(config_path, idxs, is_processed, processed_timestamp)

def run_train_pipeline():
    config_path = os.path.join("configs","trainConfig.yaml")
    training_pipeline(config_path)

def run_valid_pipeline():
    config_path = os.path.join("configs","trainConfig.yaml")
    # model_checkpoint_path = os.path.join("checkpoints","attunet","20260212_045750","best_model.pth")
    # model_checkpoint_path = os.path.join("checkpoints","unet","20260212_155606","best_model.pth")
    # model_checkpoint_path = os.path.join("checkpoints","segformer","20260213_062110","best_model.pth")
    model_checkpoint_path = os.path.join("checkpoints","transunet","20260214_052453","best_model.pth")

    valid_pipeline(config_path, model_checkpoint_path, only_changed=False, datasets=["val"])

def run_threshold_sweep_pipeline():
    config_path = os.path.join("configs","trainConfig.yaml")
    threshold_sweep_pipeline(config_path)

def run_get_changed_indices_pipeline():
    config_path = os.path.join("configs","trainConfig.yaml")
    datasets = ["train", "val"]
    changed_indices = get_changed_indices(config_path, datasets)
    print(f"所有数据集共有 {len(changed_indices)} 个变化样本, 索引为: {changed_indices}")


def run_ablation_pipeline():
    """消融实验主线: 多次重复稳定性 + A/B/C 三组对比.
    mode 可选: 'stability' 只跑 baseline (完整方法); 'ablation' 跑 A/B/C 三组.
    样本来自 scripts/select_representative_samples.py 的输出.
    """
    ablation_pipeline(
        opt_cfg_path=os.path.join("configs","optimizerConfig.yaml"),
        score_cfg_path=os.path.join("configs","score_cfg.yaml"),
        idxs=load_representative_idxs(),
        mode='stability',
        n_repeats=5,
    )

def run_ablation_plots_pipeline(experiment_root: str = None,
                                 sibling_experiment_roots=None,
                                 mode: str = 'auto',
                                 show_runtime: bool = True,
                                 combined_only: bool = False):
    """对消融实验结果出图: 稳定性叠图 + A/B/C 三组对比 + 箱线图 + 表格.

    如需 SA+GA 合并出 CV 折线图/汇总表, 传 sibling_experiment_roots (GA 的结果目录列表).
    此时会先对每个 experiment_root 独立出图, 再将合并产物写入同级合并目录
    (如 ``<sa_ts>+<ga_ts>_combined/plots/``).

    Parameters
    ----------
    experiment_root : str, optional
        消融实验结果根目录 (主, 通常为 SA); None 则自动查找 results/ablation/ 下最新目录.
    sibling_experiment_roots : list of str, optional
        其他算法版本 (如 GA) 的结果目录列表, 用于合并出 CV 折线图与汇总.
    mode : str
        后处理模式: 'stability' 仅稳定性输出 / 'ablation' 仅消融对比 / 'auto' 自动检测.
    show_runtime : bool
        箱线图和折线图是否包含 Runtime 维度, 默认 True.
    combined_only : bool
        若为 True 且传入了 sibling_experiment_roots, 跳过单算法出图,
        直接从已有 stability_runs.csv 生成合并结果.
    """
    from pipelines.ablation_plots import run_all_ablation_plots, run_combined_stability_plots
    if experiment_root is None:
        experiment_root = get_latest_experiment_dir('ablation')

    sibling_experiment_roots = sibling_experiment_roots or []
    if not sibling_experiment_roots:
        run_all_ablation_plots(experiment_root=experiment_root,
                                sibling_experiment_roots=None,
                                mode=mode,
                                show_runtime=show_runtime)
        return

    if combined_only:
        run_combined_stability_plots(experiment_root=experiment_root,
                                      sibling_experiment_roots=sibling_experiment_roots,
                                      mode=mode,
                                      show_runtime=show_runtime)
        return

    roots_to_run = [experiment_root] + list(sibling_experiment_roots)
    for root in roots_to_run:
        print(f"\n[run_ablation_plots_pipeline] 单独处理: {root}")
        run_all_ablation_plots(experiment_root=root,
                                sibling_experiment_roots=None,
                                mode=mode,
                                show_runtime=show_runtime)

    run_combined_stability_plots(experiment_root=experiment_root,
                                  sibling_experiment_roots=sibling_experiment_roots,
                                  mode=mode,
                                  show_runtime=show_runtime)


def run_tornado_pipeline(max_samples: int = None):
    """Tornado 参数粗筛: 8 个核心参数, 1 baseline + 20 扰动 = 21 组, SA, n_repeats=2.

    Parameters
    ----------
    max_samples : int, optional
        最多使用的样本数; None 表示使用全部 idxs.
    """
    idxs = load_representative_idxs()
    if max_samples is not None:
        idxs = random.sample(idxs, min(max_samples, len(idxs)))

    tornado_pipeline(
        opt_cfg_path=os.path.join("configs", "optimizerConfig.yaml"),
        score_cfg_path=os.path.join("configs", "score_cfg.yaml"),
        idxs=idxs,
        mode='tornado',
        n_repeats=3,
    )


def run_sensitivity_pipeline(top_k_params=None, max_samples: int = None):
    """一维敏感度线搜索: 对 tornado 筛选出的 Top-K 参数做细粒度扫描.

    Parameters
    ----------
    top_k_params : list of dict, optional
        待扫描的参数规格列表; None 用 utils.tornado_utils.DEFAULT_SENSITIVITY_PARAMS.
    max_samples : int, optional
        最多使用的样本数; None 表示使用全部 idxs.
    """
    if top_k_params is None:
        top_k_params = DEFAULT_SENSITIVITY_PARAMS

    idxs = load_representative_idxs()
    if max_samples is not None:
        idxs = random.sample(idxs, min(max_samples, len(idxs)))

    tornado_pipeline(
        opt_cfg_path=os.path.join("configs", "optimizerConfig.yaml"),
        score_cfg_path=os.path.join("configs", "score_cfg.yaml"),
        idxs=idxs,
        mode='sensitivity',
        n_repeats=3,
        top_k_params=top_k_params,
    )


def run_tornado_plots_pipeline(experiment_root: str = None,
                               mode: str = 'auto',
                               baseline_name: str = 'A_baseline'):
    """对 tornado / sensitivity 结果出图和表格.

    Parameters:
    -----------
    experiment_root : str, optional
        实验结果根目录; None 则自动查找 results/tornado/ 下最新目录.
    mode : {'tornado', 'sensitivity', 'auto'}
        后处理模式. auto 优先读取 experiment_summary.json 中的 mode.
    baseline_name : str
        tornado 基线实验名.
    """
    from pipelines.tornado_plots import run_tornado_plots
    if experiment_root is None:
        experiment_root = get_latest_experiment_dir('tornado')
    run_tornado_plots(experiment_root, baseline_name=baseline_name, mode=mode)


def run_results_analysis_pipeline():
    sa_expr_path = "SA\\20260301_101241"
    ga_expr_path = "GA\\20260305_120735"
    single_experiment_stat_pipeline(ga_expr_path)
    sample_visualization_pipeline(ga_expr_path, mode="picked")
    comparison_pipeline(sa_expr_path, ga_expr_path)
    model_comparison_pipeline()

if __name__ == "__main__":
    # pass
    # run_train_pipeline()
    # run_valid_pipeline()
    run_hybrid_train_pipeline()
    # run_runner_pipeline()
    # run_experiment_pipeline()
    # run_ablation_pipeline()
    # run_ablation_plots_pipeline(
    #     experiment_root='results/ablation/20260607_210931',
    #     sibling_experiment_roots=['results/ablation/20260610_163941'],
    #     mode='stability',
    #     show_runtime=False,
    #     combined_only=True
    # )
    # run_results_analysis_pipeline()
    # run_tornado_pipeline()
    # run_sensitivity_pipeline()
    # run_tornado_plots_pipeline(mode='tornado',experiment_root='results/tornado/20260606_033319')
    # run_tornado_plots_pipeline(mode='tornado',experiment_root='results/tornado/20260611_023835')
    # run_tornado_plots_pipeline(mode='tornado',experiment_root='results/tornado/20260620_220209')
    # pass
