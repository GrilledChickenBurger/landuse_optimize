import os
import random
from pipelines.run_runner import pipeline
from pipelines.experiment import experiment_pipeline
from pipelines.train import training_pipeline,valid_pipeline, get_changed_indices
from pipelines.statistic import get_params
from pipelines.ablation import ablation_pipeline
from pipelines.tornado import tornado_pipeline
from pipelines.results_analysis import (
    comparison_pipeline,
    model_comparison_pipeline,
    sample_visualization_pipeline,
    single_experiment_stat_pipeline,
)

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

def run_get_changed_indices_pipeline():
    config_path = os.path.join("configs","trainConfig.yaml")
    datasets = ["train", "val"]
    changed_indices = get_changed_indices(config_path, datasets)
    print(f"所有数据集共有 {len(changed_indices)} 个变化样本, 索引为: {changed_indices}")


def _get_latest_experiment_dir(category: str = 'ablation') -> str:
    """自动查找 results/<category>/ 下最新的时间戳目录 (按目录名排序, 不受修改时间影响)."""
    experiment_root = os.path.join("results", category)
    if not os.path.exists(experiment_root):
        raise FileNotFoundError(f"未找到实验结果目录: {experiment_root}")
    dirs = [d for d in os.listdir(experiment_root) if os.path.isdir(os.path.join(experiment_root, d))]
    if not dirs:
        raise FileNotFoundError(f"实验结果目录为空: {experiment_root}")
    dirs.sort(reverse=True)  # 时间戳格式 yyyymmdd_HHMMSS 按字符串排序即按时间
    return os.path.join(experiment_root, dirs[0])


def run_ablation_pipeline():
    """消融实验主线: 多次重复稳定性 + A/B/C 三组对比.
    mode 可选: 'stability' 只跑 baseline (完整方法); 'ablation' 跑 A/B/C 三组.
    样本来自 scripts/select_representative_samples.py 的输出.
    """
    import json as _json
    # 从代表性样本文件读 idxs (如未生成则手动指定)
    rep_json = "_data/representative_samples.json"
    if os.path.exists(rep_json):
        with open(rep_json, 'r', encoding='utf-8') as f:
            idxs = _json.load(f)['idxs_for_pipeline']
    else:
        print(f"[warn] {rep_json} 不存在, 用默认 idxs. 请先跑 scripts/select_representative_samples.py")
        idxs = [479, 596, 760, 634, 1903]
    ablation_pipeline(
        opt_cfg_path=os.path.join("configs","optimizerConfig.yaml"),
        score_cfg_path=os.path.join("configs","score_cfg.yaml"),
        idxs=idxs,
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
        experiment_root = _get_latest_experiment_dir('ablation')

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
    import json as _json

    rep_json = "_data/representative_samples.json"
    if os.path.exists(rep_json):
        with open(rep_json, 'r', encoding='utf-8') as f:
            idxs = _json.load(f)['idxs_for_pipeline']
    else:
        print(f"[warn] {rep_json} 不存在, 用默认 idxs")
        idxs = [479, 596, 760, 634, 1903]

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
        待扫描的参数规格列表. 每项含 path, baseline, values, category.
    max_samples : int, optional
        最多使用的样本数; None 表示使用全部 idxs.
    """
    import json as _json

    if top_k_params is None:
        top_k_params = [
            {'path': 'tgt_function.lambda_ES_high', 'baseline': 1000,
             'values': [200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000],
             'category': 'score_cfg'},
            {'path': 'cooling_rate', 'baseline': 0.9995,
             'values': [0.9985, 0.9990, 0.9992, 0.9993, 0.9995, 0.9996, 0.9997, 0.9998, 0.9999],
             'category': 'opt_cfg'},
        ]

    rep_json = "_data/representative_samples.json"
    if os.path.exists(rep_json):
        with open(rep_json, 'r', encoding='utf-8') as f:
            idxs = _json.load(f)['idxs_for_pipeline']
    else:
        print(f"[warn] {rep_json} 不存在, 用默认 idxs")
        idxs = [479, 596, 760, 634, 1903]

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
        experiment_root = _get_latest_experiment_dir('tornado')
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
    # run_runner_pipeline()
    # run_experiment_pipeline()
    # run_ablation_pipeline()
    run_ablation_plots_pipeline(
        experiment_root='results/ablation/20260607_210931',
        sibling_experiment_roots=['results/ablation/20260608_185005'],
        mode='stability',
        show_runtime=True,
    )
    # run_tornado_pipeline(3)
    # run_sensitivity_pipeline()
    # run_tornado_plots_pipeline()
    pass
