import os

from pipelines.run_runner import pipeline
from pipelines.experiment import experiment_pipeline
from pipelines.train import training_pipeline,valid_pipeline, get_changed_indices
from pipelines.statistic import get_params
from pipelines.ablation import ablation_pipeline
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


def run_ablation_pipeline():
    """消融实验主线: 多次重复稳定性 + A/B/C 三组对比.
    mode 可选: 'stability' 只跑 A_guided; 'ablation' 跑 A/B/C 三组.
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
        idxs=[479],
        mode='stability',     # 或 'stability'
        n_repeats=2,
    )

def run_ablation_plots_pipeline():
    """对消融实验结果出图: 稳定性叠图 + A/B/C 三组对比 + 箱线图 + 表格."""
    from utils.ablation_plots import run_all_ablation_plots
    ablation_root = os.path.join("results","ablation","20260528_120000")   # 改为实际时间戳
    run_all_ablation_plots(ablation_root=ablation_root)


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
    run_ablation_pipeline()
    # run_ablation_plots_pipeline()
