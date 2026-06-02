import os
from typing import Iterable

from utils.analysis_plot_utils import (
    compare_sample_images,
    export_sa_ga_combined_area_data,
    plot_model_comparison_bar,
    plot_origin_target_lu_stackbar,
    plot_sa_ga_boxplots,
    plot_sa_ga_diff_bars,
    plot_single_experiment_es_bars,
    plot_single_experiment_es_distribution,
    plot_single_experiment_lu_bars,
    plot_single_experiment_lu_distribution,
    visualize_all_samples,
    visualize_picked_samples,
)


def single_experiment_stat_pipeline(expr_path: str,
                                    plot_types: Iterable[str] = None,
                                    comparison_type: str = "target"):
    """
    单个 SA/GA 实验结果的 LU 与 ES 统计出图。

    Parameters:
    -----------
    - expr_path: 实验结果路径, 可传 "SA\\timestamp"、"GA\\timestamp" 或 "results\\SA|GA\\timestamp"
    - plot_types: 需要生成的图表类型, 默认 ["lu_box", "es_box", "origin_target_lu"]
        可选: "lu_bars", "lu_box", "lu_hist", "es_bars", "es_box", "es_hist", "origin_target_lu"
    - comparison_type: 变化比例的比较基准, "origin" 或 "target"

    Returns:
    --------
    - None
    """
    data_folder = _resolve_result_folder(expr_path)
    figure_folder = _resolve_figure_folder(expr_path)
    area_save_folder = os.path.join(figure_folder, "area")
    es_save_folder = os.path.join(figure_folder, "es")
    plot_types = set(plot_types or ["lu_box", "es_box", "origin_target_lu"])

    if "lu_bars" in plot_types:
        plot_single_experiment_lu_bars(data_folder=data_folder, save_folder=area_save_folder)
    if "lu_box" in plot_types:
        plot_single_experiment_lu_distribution(
            data_folder=data_folder,
            save_folder=area_save_folder,
            plot_type="box",
            comparison_type=comparison_type)
    if "lu_hist" in plot_types:
        plot_single_experiment_lu_distribution(
            data_folder=data_folder,
            save_folder=area_save_folder,
            plot_type="hist",
            density=True,
            comparison_type=comparison_type)
    if "es_bars" in plot_types:
        plot_single_experiment_es_bars(data_folder=data_folder, save_folder=es_save_folder)
    if "es_box" in plot_types:
        plot_single_experiment_es_distribution(
            data_folder=data_folder,
            save_folder=es_save_folder,
            plot_type="box",
            comparison_type=comparison_type)
    if "es_hist" in plot_types:
        plot_single_experiment_es_distribution(
            data_folder=data_folder,
            save_folder=es_save_folder,
            plot_type="hist",
            density=True,
            comparison_type=comparison_type)
    if "origin_target_lu" in plot_types:
        plot_origin_target_lu_stackbar(data_folder=data_folder, save_folder=area_save_folder)


def sample_visualization_pipeline(expr_path: str,
                                  mode: str = "picked",
                                  compare_expr_path: str = None,
                                  **kwargs):
    """
    单个实验样本方案可视化, 或对比两个实验已生成的样本图。

    Parameters:
    -----------
    - expr_path: 主实验路径, 可传 "SA\\timestamp"、"GA\\timestamp" 或 "results\\SA|GA\\timestamp"
    - mode: 可视化模式
        - "picked": 按 ES 提升条件筛选样本并可视化
        - "all": 可视化该实验下所有样本
        - "sa_ga_pair": 对比 expr_path 与 compare_expr_path 下已生成的 all_samples 图片
    - compare_expr_path: mode="sa_ga_pair" 时的另一个实验路径
    - **kwargs: 透传参数, 如 min_improvement_threshold、min_positive_scenarios、min_other_threshold、layout、save_folder

    Returns:
    --------
    - picked/all 模式返回样本索引列表; sa_ga_pair 模式返回 None
    """
    data_folder = _resolve_result_folder(expr_path)
    figure_folder = _resolve_figure_folder(expr_path)

    if mode == "picked":
        return visualize_picked_samples(
            data_folder=data_folder,
            save_folder=os.path.join(figure_folder, "samples_new"),
            min_improvement_threshold=kwargs.get("min_improvement_threshold", 0.0),
            min_positive_scenarios=kwargs.get("min_positive_scenarios", 2),
            min_other_threshold=kwargs.get("min_other_threshold", -0.05),
            layout=kwargs.get("layout", (1, 6)))
    if mode == "all":
        return visualize_all_samples(
            data_folder=data_folder,
            save_folder=os.path.join(figure_folder, "all_samples"),
            layout=kwargs.get("layout", (1, 6)))
    if mode == "sa_ga_pair":
        if compare_expr_path is None:
            raise ValueError("compare_expr_path must be provided when mode='sa_ga_pair'")
        compare_sample_images(
            img_folder1=os.path.join(_resolve_figure_folder(expr_path), "all_samples"),
            img_folder2=os.path.join(_resolve_figure_folder(compare_expr_path), "all_samples"),
            desc1=kwargs.get("desc1", "SA优化结果"),
            desc2=kwargs.get("desc2", "GA优化结果"),
            save_folder=kwargs.get("save_folder", os.path.join("figures", "comparison_results")))
        return None
    raise ValueError(f"Unsupported sample visualization mode: {mode}")


def comparison_pipeline(sa_expr_path: str,
                        ga_expr_path: str,
                        plot_types: Iterable[str] = None,
                        save_folder: str = None,
                        comparison_type: str = "target"):
    """
    SA 与 GA 优化结果对比出图。

    Parameters:
    -----------
    - sa_expr_path: SA 实验结果路径, 可传 "SA\\timestamp" 或 "results\\SA\\timestamp"
    - ga_expr_path: GA 实验结果路径, 可传 "GA\\timestamp" 或 "results\\GA\\timestamp"
    - plot_types: 需要生成的对比内容, 默认 ["lu_diff_bars", "lu_box", "es_box"]
        可选: "lu_diff_bars", "lu_box", "es_box", "combined_area_json"
    - save_folder: 对比图保存目录, 默认 "figures\\comparison_results"
    - comparison_type: 箱线图变化比例的比较基准, "origin" 或 "target"

    Returns:
    --------
    - None
    """
    sa_folder = _resolve_result_folder(sa_expr_path)
    ga_folder = _resolve_result_folder(ga_expr_path)
    save_folder = save_folder or os.path.join("figures", "comparison_results")
    plot_types = set(plot_types or ["lu_diff_bars", "lu_box", "es_box"])

    if "lu_diff_bars" in plot_types:
        plot_sa_ga_diff_bars(
            sa_folder=sa_folder,
            ga_folder=ga_folder,
            save_dir=save_folder)
    if "lu_box" in plot_types:
        plot_sa_ga_boxplots(
            sa_folder=sa_folder,
            ga_folder=ga_folder,
            save_dir=save_folder,
            data_type="LU",
            comparison_type=comparison_type)
    if "es_box" in plot_types:
        plot_sa_ga_boxplots(
            sa_folder=sa_folder,
            ga_folder=ga_folder,
            save_dir=save_folder,
            data_type="ES",
            comparison_type=comparison_type)
    if "combined_area_json" in plot_types:
        export_sa_ga_combined_area_data(
            sa_folder=sa_folder,
            ga_folder=ga_folder,
            output_path=os.path.join(save_folder, "stacked_bar_data.json"))


def model_comparison_pipeline(json_path: str = "model_comparison_data.json",
                              save_dir: str = "figures",
                              save_name: str = "model_comparison_bar.png"):
    """
    变化检测模型指标对比出图。

    Parameters:
    -----------
    - json_path: 模型对比数据 JSON 路径, 默认 "model_comparison_data.json"
    - save_dir: 图像保存目录, 默认 "figures"
    - save_name: 输出图片文件名, 默认 "model_comparison_bar.png"

    Returns:
    --------
    - None
    """
    plot_model_comparison_bar(
        json_path=json_path,
        save_dir=save_dir,
        save_name=save_name)


def complete_pipeline(expr_path: str):
    single_experiment_stat_pipeline(expr_path)


def picker(expr_path: str, min_improvement_threshold=0.0,
           min_positive_scenarios=2, min_other_threshold=-0.05):
    return sample_visualization_pipeline(
        expr_path,
        mode="picked",
        min_improvement_threshold=min_improvement_threshold,
        min_positive_scenarios=min_positive_scenarios,
        min_other_threshold=min_other_threshold)


def all_samples(expr_path: str):
    return sample_visualization_pipeline(expr_path, mode="all")


def show_SA_GA(expr_path1: str, expr_path2: str):
    return sample_visualization_pipeline(expr_path1, mode="sa_ga_pair", compare_expr_path=expr_path2)


def _resolve_result_folder(expr_path: str):
    norm_path = os.path.normpath(expr_path)
    if os.path.isabs(norm_path) or norm_path.split(os.sep)[0] == "results":
        return expr_path
    return os.path.join("results", expr_path)


def _resolve_figure_folder(expr_path: str):
    norm_path = os.path.normpath(expr_path)
    if os.path.isabs(norm_path):
        return os.path.join("figures", os.path.basename(os.path.dirname(norm_path)), os.path.basename(norm_path))
    parts = norm_path.split(os.sep)
    if parts[0] == "results":
        parts = parts[1:]
    return os.path.join("figures", *parts)
