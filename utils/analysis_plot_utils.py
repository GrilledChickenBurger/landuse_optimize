import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from libs.const import (
    ALL_ES_TYPES,
    ALL_ES_TYPES_DICT,
    ALL_LU_LABELS_DICT,
    ALL_SCENARIOS,
    ALL_SCENARIOS_DICT,
    SA_GA_COLOR_MAP,
    SCENARIO_COLOR_MAP,
)
from utils.analysis_utils import (
    build_sa_ga_combined_area_data,
    build_sa_ga_total_area_data,
    cal_sample_area_change_ratio,
    cal_sample_es_change_ratio,
    cal_scenario_diff_ratio,
    cal_total_area,
    cal_total_es,
    get_best_samples,
    get_performance_metrics,
    list_sample_indices,
    load_or_compute_metric,
    parallel_show,
    visualize_best_samples,
    visualize_by_box,
    visualize_by_hist,
    visualize_by_stackbar,
    visualize_by_type_bars,
)
from utils.tools import ensure_folder
from utils.visualize_utils import (
    grouped_bar_figure,
    grouped_box_plot_figure,
    setup_paper_plot_fonts,
)

FONT = setup_paper_plot_fonts()


def plot_single_experiment_lu_bars(data_folder: str, save_folder: str, json_path: str = None):
    total_area_data = load_or_compute_metric(
        data_folder, "total_area.json", cal_total_area, json_path,
        aggregate_by_first_digit=True)
    visualize_by_stackbar(
        total_area_data,
        total_title="各情景土地利用总面积占比对比图",
        label_format="percentage",
        y_axis_ratio=0.9,
        save_path=os.path.join(save_folder, "bars", "total_area_stackbar.png"))
    visualize_by_type_bars(
        total_area_data,
        total_title="土地利用各情景总面积对比图",
        show_legend=False,
        y_axis_ratio=0.9,
        save_folder=os.path.join(save_folder, "bars"))


def plot_single_experiment_lu_distribution(data_folder: str, save_folder: str,
                                            json_path: str = None,
                                            comparison_type: str = "origin",
                                            plot_type: str = "box",
                                            density: bool = True):
    sample_area_ratio_data = load_or_compute_metric(
        data_folder, f"sample_area_{comparison_type}.json",
        cal_sample_area_change_ratio, json_path,
        aggregate_by_first_digit=True,
        comparison_type=comparison_type)
    if plot_type == "hist":
        visualize_by_hist(
            sample_area_ratio_data,
            comparison_type=comparison_type,
            density=density,
            bins=20,
            histtype="bar",
            remove_extremes=True,
            method="percentile",
            lower_bound=0.05,
            upper_bound=0.95,
            save_folder=os.path.join(save_folder, "hist", f"{comparison_type}_density_{density}"))
    elif plot_type == "box":
        visualize_by_box(
            sample_area_ratio_data,
            comparison_type=comparison_type,
            show_fliers=False,
            show_means=False,
            save_folder=os.path.join(save_folder, "box", comparison_type))
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}")


def plot_single_experiment_es_bars(data_folder: str, save_folder: str, json_path: str = None):
    total_es_data = load_or_compute_metric(
        data_folder, "total_es.json", cal_total_es, json_path)
    visualize_by_type_bars(
        total_es_data,
        total_title="各情景ES总价值对比图",
        show_legend=False,
        show_labels=False,
        y_axis_ratio=0.95,
        save_folder=os.path.join(save_folder, "bars"))


def plot_single_experiment_es_distribution(data_folder: str, save_folder: str,
                                            json_path: str = None,
                                            comparison_type: str = "origin",
                                            plot_type: str = "box",
                                            density: bool = True):
    sample_es_ratio_data = load_or_compute_metric(
        data_folder, f"sample_es_{comparison_type}.json",
        cal_sample_es_change_ratio, json_path,
        comparison_type=comparison_type)
    if plot_type == "hist":
        visualize_by_hist(
            sample_es_ratio_data,
            comparison_type=comparison_type,
            density=density,
            bins=20,
            histtype="bar",
            remove_extremes=True,
            method="percentile",
            lower_bound=0.05,
            upper_bound=0.95,
            save_folder=os.path.join(save_folder, "hist", f"{comparison_type}_density_{density}"))
    elif plot_type == "box":
        visualize_by_box(
            sample_es_ratio_data,
            comparison_type=comparison_type,
            show_fliers=False,
            show_means=False,
            save_folder=os.path.join(save_folder, "box", comparison_type))
        visualize_by_box(
            _transpose_scenario_type_data(sample_es_ratio_data, ALL_ES_TYPES, ALL_SCENARIOS),
            comparison_type=comparison_type,
            show_fliers=False,
            show_means=False,
            save_folder=os.path.join(save_folder, "box", f"{comparison_type}_new"))
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}")


def plot_origin_target_lu_stackbar(data_folder: str, save_folder: str, json_path: str = None):
    total_area_data = load_or_compute_metric(
        data_folder, "total_area.json", cal_total_area, json_path,
        aggregate_by_first_digit=True)
    filtered_data = {key: total_area_data[key] for key in ["origin", "target"] if key in total_area_data}
    visualize_by_stackbar(
        filtered_data,
        total_title="原始与目标土地利用总面积占比对比图",
        label_format="percentage",
        y_axis_ratio=0.9,
        save_path=os.path.join(save_folder, "bars", "origin_target_stackbar.png"))


def visualize_picked_samples(data_folder: str, save_folder: str,
                             min_improvement_threshold=0.0,
                             min_positive_scenarios=2,
                             min_other_threshold=-0.05,
                             layout=(1, 6)):
    metrics = get_performance_metrics(data_folder)
    best_samples, _ = get_best_samples(
        metrics=metrics,
        min_improvement_threshold=min_improvement_threshold,
        min_positive_scenarios=min_positive_scenarios,
        min_other_threshold=min_other_threshold)
    print(best_samples)
    visualize_best_samples(best_samples, data_folder, save_folder, layout=layout)
    return best_samples


def visualize_all_samples(data_folder: str, save_folder: str, layout=(1, 6)):
    indices = list_sample_indices(data_folder)
    visualize_best_samples(indices, data_folder, save_folder, layout=layout)
    return indices


def compare_sample_images(img_folder1: str, img_folder2: str,
                          desc1: str = "SA优化结果", desc2: str = "GA优化结果",
                          save_folder: str = "figures/comparison_results"):
    parallel_show(img_folder1, img_folder2, desc1, desc2, save_folder)


def plot_sa_ga_diff_bars(sa_folder: str, ga_folder: str, save_dir: str = "figures/comparison",
                         y_scale: str = "pct", signed_mode: bool = True,
                         gap_threshold: float = 50, save_dpi: int = 200):
    sa_raw, ga_raw = build_sa_ga_total_area_data(sa_folder, ga_folder)
    scenario_order = _sa_ga_scenario_order()
    display_names = _sa_ga_display_names()
    data, lu_directions, lu_ref_lines = cal_scenario_diff_ratio(
        sa_raw, ga_raw,
        scenario_order=scenario_order,
        scenario_display_names=display_names,
        y_scale=y_scale,
        signed_mode=signed_mode,
        gap_threshold=gap_threshold)

    scale_name = "ratio" if y_scale == "ratio" else "pct"
    signed_name = "signed" if signed_mode else "unsigned"
    save_path = os.path.join(save_dir, f"sa_ga_diff_{scale_name}_{signed_name}.png")
    ensure_folder(os.path.dirname(save_path))

    lu_names = list(data.keys())
    ncols, nrows = 2, (len(lu_names) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 12))
    axes = axes.flatten() if len(lu_names) > 1 else [axes]
    y_label = "变化量（百分点）" if y_scale == "pct" else "变化量（比率 vs 目标）"
    bar_label_fmt = "{:.1f}" if y_scale == "pct" else "{:.2f}"
    ytick_fmt = "{:.1f}" if y_scale == "pct" else "{:.2f}"
    series_colors = _sa_ga_diff_colors()

    for i, lu_name in enumerate(lu_names):
        grouped_bar_figure(
            data={lu_name: data[lu_name]},
            title=lu_name,
            y_label=y_label,
            series_colors=series_colors,
            bar_width=0.85,
            hline_positions=_diff_reference_lines(lu_ref_lines[lu_name], lu_directions[lu_name], signed_mode),
            ax=axes[i],
            show_legend=False,
            show_bar_labels=True,
            bar_label_fmt=bar_label_fmt,
            ytick_fmt=ytick_fmt,
            show_xticklabels=False)

    for j in range(len(lu_names), len(axes)):
        axes[j].set_visible(False)

    _add_sa_ga_diff_legend(fig, series_colors)
    fig.savefig(save_path, dpi=save_dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"SA vs GA diff bars saved to: {save_path}")


def plot_sa_ga_boxplots(sa_folder: str, ga_folder: str,
                        save_dir: str = "figures/comparison",
                        data_type: str = "LU",
                        comparison_type: str = "target",
                        layout_mode: str = "by_scenario",
                        save_dpi_combined: int = 200,
                        save_dpi_single: int = 150):
    config = _resolve_sa_ga_boxplot_config(data_type, comparison_type)
    data = _build_sa_ga_boxplot_data(sa_folder, ga_folder, config, layout_mode)
    save_folder = os.path.join(save_dir, config["save_subdir"])
    ensure_folder(save_folder)

    grouped_box_plot_figure(
        data=data,
        title="",
        y_label=config["y_label"],
        series_colors=SA_GA_COLOR_MAP,
        subplot_layout=(2, 2),
        figsize=(20, 13),
        showfliers=False,
        showmeans=False,
        unified_ylim=False,
        exclude_fliers_for_ylim=True,
        ylim_margin=0.05,
        hspace=0.2,
        wspace=0.25,
        save_path=os.path.join(save_folder, f"SA_vs_GA_{comparison_type}.png"),
        dpi=save_dpi_combined,
        legend_loc="bottom")
    print(f"Combined figure saved to: {save_folder}")

    for subplot_key in data.keys():
        grouped_box_plot_figure(
            data={subplot_key: data[subplot_key]},
            title=f"{subplot_key} - SA vs GA {config['y_label']}（对比 {comparison_type}）",
            y_label=config["y_label"],
            series_colors=SA_GA_COLOR_MAP,
            figsize=(8, 5),
            showfliers=False,
            showmeans=False,
            unified_ylim=False,
            exclude_fliers_for_ylim=True,
            ylim_margin=0.05,
            save_path=os.path.join(save_folder, f"{subplot_key}_{comparison_type}.png"),
            dpi=save_dpi_single,
            legend_loc="bottom")
    print(f"Individual figures saved to: {save_folder}")


def plot_model_comparison_bar(json_path: str = "model_comparison_data.json",
                              save_dir: str = "figures",
                              save_name: str = "model_comparison_bar.png",
                              save_dpi: int = 200):
    with open(json_path, "r", encoding="utf-8") as f:
        cmp_data = json.load(f)

    models = cmp_data["models"]
    metrics = cmp_data["metrics"]
    # 按 F1 从大到小重排模型, 各指标数组同步调整顺序
    order = sorted(range(len(models)), key=lambda i: metrics["F1"][i], reverse=True)
    models = [models[i] for i in order]
    metrics = {name: [vals[i] for i in order] for name, vals in metrics.items()}
    metric_names = list(metrics.keys())
    # 柱顶数值与图例字号从 paper 基准(23/24)下调, 避免过大压盖/超宽
    annot_prop = FONT['prop'](16)
    legend_prop = FONT['prop'](16)
    green_shades = plt.cm.Greens(np.linspace(0.85, 0.35, len(models)))
    bar_width = 0.80
    offsets = np.linspace(
        -bar_width / 2 + bar_width / (2 * len(models)),
        bar_width / 2 - bar_width / (2 * len(models)),
        len(models))
    single_bar_width = bar_width / len(models) * 0.92

    fig, ax = plt.subplots(figsize=(14.4, 6))
    for i, metric_name in enumerate(metric_names):
        values = metrics[metric_name]
        for k, model_name in enumerate(models):
            x_pos = i + offsets[k]
            val = values[k]
            ax.bar(x_pos, val, single_bar_width,
                   color=green_shades[k], alpha=0.85,
                   edgecolor='white', linewidth=0.5)
            ax.text(x_pos, val + 0.005, f"{val:.4f}",
                    ha='center', va='bottom', fontproperties=annot_prop)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontproperties=FONT['tick'])
    ax.margins(y=0.10)  # Y 轴自适应, 顶部留白避免柱顶数值被裁剪
    ax.set_ylabel("得分", fontproperties=FONT['label'])
    ax.grid(True, alpha=0.5, axis='y', color='gray', linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(
        handles=[Patch(facecolor=green_shades[k], alpha=0.85, edgecolor='white', label=models[k])
                 for k in range(len(models))],
        loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,
        prop=legend_prop, frameon=False)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    fig.savefig(save_path, dpi=save_dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Model comparison bar saved to: {save_path}")


def export_sa_ga_combined_area_data(sa_folder: str, ga_folder: str, output_path: str = "stacked_bar_data.json"):
    combined = build_sa_ga_combined_area_data(sa_folder, ga_folder)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=4)
    print(f"Exported {len(combined)} groups to {output_path}")
    return combined


def _transpose_scenario_type_data(data, type_keys, scenarios):
    return {
        type_key: {
            scenario: data.get(scenario, {}).get(type_key, [])
            for scenario in scenarios
        }
        for type_key in type_keys
    }


def _sa_ga_scenario_order():
    return [f"SA_{scenario}" for scenario in ALL_SCENARIOS] + [f"GA_{scenario}" for scenario in ALL_SCENARIOS]


def _sa_ga_display_names():
    names = {}
    for method in ["SA", "GA"]:
        for scenario in ALL_SCENARIOS:
            names[f"{method}_{scenario}"] = f"{method} {ALL_SCENARIOS_DICT.get(scenario, scenario)}"
    return names


def _sa_ga_diff_colors():
    blues = plt.cm.Blues(np.linspace(0.85, 0.35, len(ALL_SCENARIOS)))
    oranges = plt.cm.Oranges(np.linspace(0.85, 0.35, len(ALL_SCENARIOS)))
    colors = {}
    for i, scenario in enumerate(ALL_SCENARIOS):
        colors[f"SA {ALL_SCENARIOS_DICT.get(scenario, scenario)}"] = blues[i]
        colors[f"GA {ALL_SCENARIOS_DICT.get(scenario, scenario)}"] = oranges[i]
    colors["真实"] = "#dedede"
    return colors


def _diff_reference_lines(ref_val, direction, signed_mode):
    target_line = -ref_val if signed_mode and direction == "decrease" else ref_val
    return [
        {"y": 0, "linestyle": "-", "color": "#228B22", "linewidth": 1.2},
        {"y": target_line, "linestyle": "--", "color": "#DC143C", "linewidth": 1.0},
    ]


def _add_sa_ga_diff_legend(fig, series_colors):
    sa_names = [f"SA {ALL_SCENARIOS_DICT.get(s, s)}" for s in ALL_SCENARIOS]
    ga_names = [f"GA {ALL_SCENARIOS_DICT.get(s, s)}" for s in ALL_SCENARIOS]
    sa_handles = [plt.Rectangle((0, 0), 1, 1, fc=series_colors.get(name, "gray"), alpha=0.85)
                  for name in sa_names]
    ga_handles = [plt.Rectangle((0, 0), 1, 1, fc=series_colors.get(name, "gray"), alpha=0.85)
                  for name in ga_names]
    ga_handles.append(plt.Rectangle((0, 0), 1, 1, fc=series_colors["真实"], alpha=0.85))
    leg_sa = fig.legend(sa_handles, sa_names, ncol=4, loc='lower center',
                        bbox_to_anchor=(0.5, 0.04), frameon=False,
                        handlelength=1.2, handleheight=1.2, prop=FONT['legend'])
    fig.add_artist(leg_sa)
    fig.legend(ga_handles, ga_names + ["真实"], ncol=5, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               handlelength=1.2, handleheight=1.2, prop=FONT['legend'])


def _resolve_sa_ga_boxplot_config(data_type: str, comparison_type: str):
    normalized = data_type.upper()
    if normalized == "LU":
        return {
            "type_keys": [1, 2, 4, 5],
            "name_dict": ALL_LU_LABELS_DICT,
            "calc_fn": cal_sample_area_change_ratio,
            "calc_kwargs": {"aggregate_by_first_digit": True, "comparison_type": comparison_type},
            "cache_name": f"sample_area_{comparison_type}.json",
            "y_label": "土地利用面积变化比率",
            "save_subdir": "SA_GA_lu_boxplot",
        }
    if normalized == "ES":
        return {
            "type_keys": ALL_ES_TYPES,
            "name_dict": ALL_ES_TYPES_DICT,
            "calc_fn": cal_sample_es_change_ratio,
            "calc_kwargs": {"comparison_type": comparison_type},
            "cache_name": f"sample_es_{comparison_type}.json",
            "y_label": "生态系统服务量变化比率",
            "save_subdir": "SA_GA_es_boxplot",
        }
    raise ValueError(f"Unsupported data_type: {data_type}")


def _build_sa_ga_boxplot_data(sa_folder: str, ga_folder: str, config: dict, layout_mode: str):
    sa_ratios = load_or_compute_metric(sa_folder, config["cache_name"], config["calc_fn"], **config["calc_kwargs"])
    ga_ratios = load_or_compute_metric(ga_folder, config["cache_name"], config["calc_fn"], **config["calc_kwargs"])
    type_keys = config["type_keys"]
    name_dict = config["name_dict"]
    data = {}

    if layout_mode == "by_type":
        for key in type_keys:
            type_name = name_dict.get(key, str(key))
            data[type_name] = {}
            for scenario in ALL_SCENARIOS:
                data[type_name][ALL_SCENARIOS_DICT.get(scenario, scenario)] = {
                    "SA": sa_ratios.get(scenario, {}).get(key, []),
                    "GA": ga_ratios.get(scenario, {}).get(key, []),
                }
    elif layout_mode == "by_scenario":
        for scenario in ALL_SCENARIOS:
            scenario_label = ALL_SCENARIOS_DICT.get(scenario, scenario)
            data[scenario_label] = {}
            for key in type_keys:
                type_name = name_dict.get(key, str(key))
                data[scenario_label][type_name] = {
                    "SA": sa_ratios.get(scenario, {}).get(key, []),
                    "GA": ga_ratios.get(scenario, {}).get(key, []),
                }
    else:
        raise ValueError(f"Unsupported layout_mode: {layout_mode}")
    return data
