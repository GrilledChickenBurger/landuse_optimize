# 生成论文用图表，包括折线图、柱状图、热力图等
import os
from typing import Any, List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from utils.tools import ensure_folder


def setup_plot_fonts(base_size: float = 12,
                     title_size: float = 14,
                     label_size: float = 12,
                     tick_size: float = 10,
                     legend_size: float = 10,
                     annot_size: float = 9):
    """统一配置中英文混排字体：英文 Times New Roman Bold + 中文 SimHei/SimSun 回退。"""
    matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei', 'SimSun', 'serif']
    matplotlib.rcParams['font.weight'] = 'bold'
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
    matplotlib.rcParams['axes.titlesize'] = title_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['xtick.labelsize'] = tick_size
    matplotlib.rcParams['ytick.labelsize'] = tick_size
    matplotlib.rcParams['legend.fontsize'] = legend_size
    matplotlib.rcParams['figure.titlesize'] = title_size
    return {
        'base_size': base_size,
        'title_size': title_size,
        'label_size': label_size,
        'tick_size': tick_size,
        'legend_size': legend_size,
        'annot_size': annot_size,
    }


# 模块导入时自动应用默认字体配置
setup_plot_fonts()

def color_bar_figure(x_labels: List[str], 
                    y_values: List[float], 
                    y_label: str, 
                    title: str, 
                    show_labels: bool = True,
                    label_type: str = "edge",
                    color_setting: Any = [], 
                    show_legend: bool = True,
                    y_axis_ratio: float = 1.0,
                    width: float = 0.35,
                    figsize: Tuple[int, int] = (10, 6),
                    ax: plt.Axes = None):
    """
    生成带有自定义颜色和图例的条形图
    
    参数:
        x_labels: x轴标签列表（类别）
        y_values: y轴数值列表（条形高度）
        y_label: y轴标签
        title: 图表标题
        color_setting: 颜色列表，与x_labels长度相同
        show_labels: 是否显示数值标签，默认False
        label_type: 标签位置，可选"center"（居中）或"edge"（边缘），默认"center"
        legend: 是否显示图例，默认True
        y_axis_ratio: Y轴显示比例，0-1之间，默认1.0（占满整个Y轴）
        figsize: 图表大小，默认(10, 6)
        ax: 可选的matplotlib Axes对象，如果提供则在该axes上绘制
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    # 创建带有自定义颜色的条形图
    bars = ax.bar(x_labels, y_values, 
                    color=color_setting,
                    width=width)
    
    # 如果启用标签显示，添加数值标签
    if show_labels:
        ax.bar_label(bars, label_type=label_type, padding=2, fmt='%d')
    
    # 设置Y轴显示范围
    if y_axis_ratio < 1.0:
        max_height = max(y_values) if y_values else 0
        min_height = min(y_values) if y_values else 0
        y_max = max_height / y_axis_ratio
        # 不从0开始，而是从最小值附近开始，突出数据变化
        y_min = max(0, min_height * 0.9)  # 最小值乘以0.9，但确保不小于0
        ax.set_ylim(y_min, y_max)
    
    # 设置轴标签和标题
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # 创建包含唯一颜色条目的图例
    unique_colors = {}  # 存储唯一颜色的字典，键为索引，值为颜色
    for i, color in enumerate(color_setting):
        if color not in unique_colors.values():
            unique_colors[x_labels[i]] = color
    
    if show_legend:
        # # 创建图例句柄
        handles = []
        for label, color in unique_colors.items():
            handles.append(plt.Rectangle((0,0),1,1, fc=color, label=label))
        ax.legend(handles=handles)

    plt.tight_layout()
    
    return fig


def stacked_bar_figure(categories: List[str], 
                      data_dict: Dict[str, np.ndarray], 
                      title: str, 
                      y_label: str = "Count",
                      width: float = 0.5,
                      show_labels: bool = True,
                      label_type: str = "center",
                      label_format: str = "value",
                      colors: Dict[str, Any] = None,
                      y_axis_ratio: float = 1.0,
                      figsize: Tuple[int, int] = (10, 6),
                      ax: plt.Axes = None):
    """
    生成堆叠条形图
    
    参数:
        categories: 类别标签列表
        data_dict: 数据字典，键为堆叠部分的标签，值为对应类别的数值数组
        title: 图表标题
        y_label: y轴标签，默认"Count"
        width: 条形宽度，默认0.5
        show_labels: 是否显示数值标签，默认False
        label_type: 标签位置，可选"center"（居中）或"edge"（边缘），默认"center"
        label_format: 标签格式，可选"value"（实际值）或"percentage"（百分比），默认"value"
        colors: 颜色字典，键为堆叠部分的标签，值为对应的颜色，默认None（使用默认颜色）
        y_axis_ratio: Y轴显示比例，0-1之间，默认1.0（占满整个Y轴）
        figsize: 图表大小，默认(10, 6)
        ax: 可选的matplotlib Axes对象，如果提供则在该axes上绘制
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # 初始化底部位置
    bottom = np.zeros(len(categories))

    # 计算每个类别的总高度（用于百分比计算）
    total_values = np.zeros(len(categories))
    for values in data_dict.values():
        total_values += values
    
    # 绘制堆叠条形图
    for i, (label, values) in enumerate(data_dict.items()):
        # 如果有自定义颜色，使用自定义颜色，否则使用默认颜色
        color = colors[label] if colors and label in colors else None
        
        p = ax.bar(categories, values, width, label=label, bottom=bottom, color=color)
        
        # 如果启用标签显示，添加数值标签
        if show_labels:
            if label_format == "percentage":
                # 计算每个条形的百分比值
                percentages = []
                for j, value in enumerate(values):
                    if total_values[j] > 0:
                        percentage = (value / total_values[j]) * 100
                        percentages.append(f"{percentage:.1f}%")
                    else:
                        percentages.append("0.0%")
                ax.bar_label(p, labels=percentages, label_type=label_type)
            else:
                # 显示实际值
                ax.bar_label(p, label_type=label_type)
        
        bottom += values
    
    # 设置Y轴显示范围
    if y_axis_ratio < 1.0:
        max_height = np.max(total_values)
        y_max = max_height / y_axis_ratio
        ax.set_ylim(0, y_max)
    
    # 设置轴标签和标题
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=len(data_dict))
    
    plt.tight_layout()
    plt.show()
    
    return fig

def hist_figure(data: np.ndarray,
                bins: int = 10,
                title: str = "Histogram",
                y_label: str = "Frequency",
                color: str = "skyblue",
                edgecolor: str = "black",
                alpha: float = 0.7,
                density: bool = False,
                cumulative: bool = False,
                histtype: str = "bar",
                orientation: str = "vertical",
                rwidth: float = 0.8,
                log: bool = False,
                figsize: Tuple[int, int] = (10, 6),
                ax: plt.Axes = None,
                remove_extremes: bool = False,
                method: str = "percentile",
                lower_bound: float = 0.05,
                upper_bound: float = 0.95,
                std_threshold: float = 3.0):
    """
    生成直方图
    
    参数:
        data: 输入数据数组
        bins: 直方图的箱子数量或边界，默认10
        title: 图表标题，默认"Histogram"
        y_label: y轴标签，默认"Frequency"
        color: 直方图颜色，默认"skyblue"
        edgecolor: 直方图边框颜色，默认"black"
        alpha: 透明度，默认0.7
        density: 是否显示密度而不是频数，默认False
        cumulative: 是否显示累积直方图，默认False
        histtype: 直方图类型，可选"bar", "step", "stepfilled"，默认"bar"
        orientation: 方向，"vertical"或"horizontal"，默认"vertical"
        rwidth: 直方图相对宽度，默认0.8
        log: 是否使用对数刻度，默认False
        figsize: 图表大小，默认(10, 6)
        ax: 可选的matplotlib Axes对象，如果提供则在该axes上绘制
        remove_extremes: 是否移除极端值，默认False
        method: 极端值检测方法，可选"percentile"（百分位数）或"std"（标准差），默认"percentile"
        lower_bound: 当method="percentile"时，下界百分位数，默认0.05
        upper_bound: 当method="percentile"时，上界百分位数，默认0.95
        std_threshold: 当method="std"时，标准差阈值，默认3.0
    
    返回:
        matplotlib.figure.Figure: 生成的图表对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # 处理极端值过滤
    filtered_data = data.copy()
    if remove_extremes:
        if method == "percentile":
            lower_val = np.percentile(data, lower_bound * 100)
            upper_val = np.percentile(data, upper_bound * 100)
            filtered_data = data[(data >= lower_val) & (data <= upper_val)]
        elif method == "std":
            mean_val = np.mean(data)
            std_val = np.std(data)
            lower_val = mean_val - std_threshold * std_val
            upper_val = mean_val + std_threshold * std_val
            filtered_data = data[(data >= lower_val) & (data <= upper_val)]
        else:
            raise ValueError(f"不支持的极端值检测方法: {method}")
    
    # 绘制直方图
    n, bins, patches = ax.hist(filtered_data, 
                              bins=bins, 
                              color=color, 
                              edgecolor=edgecolor, 
                              alpha=alpha,
                              density=density,
                              cumulative=cumulative,
                              histtype=histtype,
                              orientation=orientation,
                              rwidth=rwidth,
                              log=log)
    
    # 设置轴标签和标题
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # 添加网格线
    ax.grid(True, alpha=0.3)
    
    # 如果使用对数刻度，设置合适的格式
    if log:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    return fig


def box_plot_figure(data_sets: List[List[float]],
                    labels: List[str] = None,
                    title: str = "Box Plot",
                    y_label: str = "Values",
                    colors: List[str] = None,
                    show_labels: bool = True,
                    show_means: bool = True,
                    show_fliers: bool = True,
                    notch: bool = False,
                    vert: bool = True,
                    patch_artist: bool = True,
                    widths: float = 0.5,
                    whis: float = 1.5,
                    figsize: Tuple[int, int] = (10, 6),
                    ax: plt.Axes = None,
                    subplot_label: str = None):
    """
    生成箱线图，支持多组数据对比
    
    参数:
        data_sets: 包含多组数据的列表，每个元素是一个数据列表
        labels: 每组数据的标签列表，与data_sets长度相同
        title: 图表标题，默认"Box Plot"
        y_label: y轴标签，默认"Values"
        colors: 每组数据的颜色列表，与data_sets长度相同
        show_labels: 是否显示标签，默认True
        show_means: 是否显示均值线，默认True
        show_fliers: 是否显示异常值，默认True
        notch: 是否显示缺口，默认False
        vert: 是否垂直显示，默认True
        patch_artist: 是否填充箱体颜色，默认True
        widths: 箱体宽度，默认0.5
        whis: 须线范围，默认1.5（IQR的倍数）
        figsize: 图表大小，默认(10, 6)
        ax: 可选的matplotlib Axes对象，如果提供则在该axes上绘制
        subplot_label: 子图标注文本，如果提供则在左上角显示标注
    
    返回:
        matplotlib.figure.Figure: 生成的图表对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # 如果没有提供标签，使用默认标签
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(data_sets))]
    
    # 如果没有提供颜色，使用默认颜色序列
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_sets)))
    elif len(colors) < len(data_sets):
        # 如果颜色数量不足，使用默认颜色补充
        default_colors = plt.cm.Set3(np.linspace(0, 1, len(data_sets)))
        colors = list(colors) + list(default_colors[len(colors):])
    
    # 创建箱线图
    box_plot = ax.boxplot(data_sets,
                         labels=labels if show_labels else [""] * len(data_sets),
                         showmeans=show_means,
                         showfliers=show_fliers,
                         notch=notch,
                         vert=vert,
                         patch_artist=patch_artist,
                         widths=widths,
                         whis=whis)
    
    # 设置箱体颜色
    if patch_artist:
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    # 设置中位数线颜色
    for median in box_plot['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # 设置均值线样式
    if show_means:
        for mean in box_plot['means']:
            mean.set_color('red')
            mean.set_linewidth(2)
            mean.set_marker('o')
            mean.set_markerfacecolor('red')
            mean.set_markersize(4)
    
    # 设置异常值样式
    if show_fliers:
        for flier in box_plot['fliers']:
            flier.set_marker('o')
            flier.set_markerfacecolor('red')
            flier.set_markeredgecolor('red')
            flier.set_alpha(0.5)
            flier.set_markersize(4)
    
    # 设置轴标签和标题
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # 添加网格线
    ax.grid(True, alpha=0.3, axis='y' if vert else 'x')
    
    # 如果垂直显示，设置x轴标签旋转
    if vert and show_labels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 添加子图标注
    if subplot_label:
        ax.annotate(
            subplot_label,
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))
    
    plt.tight_layout()
    
    return fig


def legend(color_map: Dict, 
           title: str = "图例",
           figsize: Tuple[int, int] = (6, 4),
           ncol: int = 1,
           loc: str = "center",
           fontsize: int = 12,
           frameon: bool = True,
           fancybox: bool = True,
           shadow: bool = False,
           framealpha: float = 1.0,
           facecolor: str = "white",
           edgecolor: str = "black",
           borderpad: float = 1.0,
           labelspacing: float = 1.0,
           handlelength: float = 2.0,
           handleheight: float = 0.7,
           handletextpad: float = 0.8,
           borderaxespad: float = 0.5,
           columnspacing: float = 2.0) -> plt.Figure:
    """
    根据颜色映射生成一个只包含图例的图表
    
    参数:
        color_map: 颜色映射字典，格式为 {标签: 颜色}
        title: 图例标题，默认"图例"
        figsize: 图表大小，默认(6, 4)
        ncol: 图例列数，默认1
        loc: 图例位置，默认"center"
        fontsize: 字体大小，默认12
        frameon: 是否显示图例边框，默认True
        fancybox: 是否使用圆角边框，默认True
        shadow: 是否显示阴影，默认False
        framealpha: 边框透明度，默认1.0
        facecolor: 背景颜色，默认"white"
        edgecolor: 边框颜色，默认"black"
        borderpad: 边框内边距，默认1.0
        labelspacing: 标签间距，默认1.0
        handlelength: 图例标记长度，默认2.0
        handleheight: 图例标记高度，默认0.7
        handletextpad: 标记与文本间距，默认0.8
        borderaxespad: 图例与坐标轴间距，默认0.5
        columnspacing: 列间距，默认2.0
        
    返回:
        plt.Figure: 包含图例的图表对象
    """
    # 创建空图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 隐藏坐标轴
    ax.axis('off')
    
    # 创建图例句柄
    handles = []
    labels = []
    
    for label, color in color_map.items():
        # 创建矩形图例标记
        handle = plt.Rectangle((0, 0), 1, 1, fc=color, ec='black', linewidth=0.5)
        handles.append(handle)
        labels.append(str(label))
    
    # 创建图例
    legend_obj = ax.legend(handles, labels,
                          loc=loc,
                          ncol=ncol,
                          fontsize=fontsize,
                          frameon=frameon,
                          fancybox=fancybox,
                          shadow=shadow,
                          framealpha=framealpha,
                          facecolor=facecolor,
                          edgecolor=edgecolor,
                          borderpad=borderpad,
                          labelspacing=labelspacing,
                          handlelength=handlelength,
                          handleheight=handleheight,
                          handletextpad=handletextpad,
                          borderaxespad=borderaxespad,
                          columnspacing=columnspacing)
    
    # 设置图例标题
    if title:
        legend_obj.set_title(title, prop={'size': fontsize + 2})
    
    # 调整布局
    plt.tight_layout()
    plt.show()
    return fig




def grouped_box_plot_figure(
        data: Dict[str, Dict[str, Dict[str, List[float]]]],
        title: str = "Grouped Box Plot",
        y_label: str = "Values",
        series_colors: Dict[str, Any] = None,
        subplot_layout: Tuple[int, int] = None,
        figsize: Tuple[int, int] = None,
        showfliers: bool = False,
        showmeans: bool = True,
        width: float = 0.55,
        hspace: float = 0.35,
        wspace: float = 0.25,
        unified_ylim: bool = True,
        exclude_fliers_for_ylim: bool = True,
        ylim_margin: float = 0.05,
        save_path: str = None,
        dpi: int = 200,
        fontsize: int = 11,
        title_fontsize: int = None,
        ylabel_fontsize: int = None,
        tick_fontsize: int = None,
        legend_fontsize: int = None,
        suptitle_fontsize: int = None,
        legend_loc: str = "subplot",
        legend_use_zh_font: bool = True,
        zh_font=None,
) -> plt.Figure:
    """
    legend_loc: "subplot" = 第一个子图右上角, "bottom" = 图底部居中, None = 不显示图例
    """
    subplot_titles = list(data.keys())
    n_subplots = len(subplot_titles)

    if n_subplots == 0:
        raise ValueError("data 为空")

    # 收集所有组标签和系列名（保持出现顺序）
    group_labels = []
    series_names = []
    for subplot_data in data.values():
        for g_label in subplot_data:
            if g_label not in group_labels:
                group_labels.append(g_label)
            for s_name in subplot_data[g_label]:
                if s_name not in series_names:
                    series_names.append(s_name)

    n_groups = len(group_labels)
    n_series = len(series_names)

    # 默认颜色
    if series_colors is None:
        default_colors = plt.cm.Set2(np.linspace(0, 1, max(n_series, 1)))
        series_colors = {name: default_colors[i] for i, name in enumerate(series_names)}

    # 子图布局
    if subplot_layout is None:
        ncols = min(3, n_subplots)
        nrows = (n_subplots + ncols - 1) // ncols
    else:
        nrows, ncols = subplot_layout

    if figsize is None:
        figsize = (ncols * 7, nrows * 6)

    # 统一纵坐标
    global_ylim = None
    if unified_ylim:
        all_vals = []
        for subplot_data in data.values():
            for group_data in subplot_data.values():
                for vals in group_data.values():
                    if not vals:
                        continue
                    arr = np.array(vals)
                    if exclude_fliers_for_ylim:
                        q1, q3 = np.percentile(arr, [25, 75])
                        iqr = q3 - q1
                        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        arr = arr[(arr >= lo) & (arr <= hi)]
                    all_vals.extend(arr.tolist())
        if all_vals:
            vmin, vmax = np.min(all_vals), np.max(all_vals)
            margin = (vmax - vmin) * ylim_margin
            global_ylim = (vmin - margin, vmax + margin)

    # 计算每个组内系列的位置偏移
    # 组中心: 0, 2, 4, 6, ...
    total_series_width = width * n_series
    offsets = np.linspace(-total_series_width / 2 + width / 2,
                          total_series_width / 2 - width / 2,
                          n_series) if n_series > 1 else [0.0]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 计算各元素实际字号（优先使用指定值，否则由 fontsize 派生）
    _title_fs = title_fontsize if title_fontsize is not None else fontsize + 3
    _ylabel_fs = ylabel_fontsize if ylabel_fontsize is not None else fontsize
    _tick_fs = tick_fontsize if tick_fontsize is not None else fontsize
    _legend_fs = legend_fontsize if legend_fontsize is not None else fontsize
    _suptitle_fs = suptitle_fontsize if suptitle_fontsize is not None else fontsize + 5

    for i, (subplot_title, subplot_data) in enumerate(data.items()):
        ax = axes[i]
        positions_all = []
        data_all = []
        colors_all = []

        for j, g_label in enumerate(group_labels):
            group_center = j * 2  # 0, 2, 4, ...
            group_data = subplot_data.get(g_label, {})
            for k, s_name in enumerate(series_names):
                vals = group_data.get(s_name, [])
                pos = group_center + offsets[k]
                positions_all.append(pos)
                data_all.append(vals)
                colors_all.append(series_colors.get(s_name, "lightgray"))

        bp = ax.boxplot(data_all, positions=positions_all, widths=width,
                        patch_artist=True, showfliers=showfliers,
                        showmeans=showmeans,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=4))

        for patch, color in zip(bp['boxes'], colors_all):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        # x 轴
        group_centers = [j * 2 for j in range(n_groups)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_labels, fontproperties=zh_font, fontsize=_tick_fs)

        ax.set_title(subplot_title, fontproperties=zh_font, fontsize=_title_fs, fontweight='bold')
        ax.set_ylabel(y_label, fontproperties=zh_font, fontsize=_ylabel_fs)
        ax.tick_params(axis='y', labelsize=_tick_fs)
        ax.grid(True, alpha=0.3, axis='y')

        if global_ylim is not None:
            ax.set_ylim(global_ylim)

    # 图例
    if legend_loc is not None and n_series > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, fc=series_colors.get(s, "lightgray"), alpha=0.75)
                   for s in series_names]
        _legend_zh = zh_font if legend_use_zh_font else None
        if legend_loc == "subplot":
            _legend_prop = _legend_zh.copy() if _legend_zh is not None else None
            if _legend_prop is not None:
                _legend_prop.set_size(_legend_fs - 1)
            axes[0].legend(handles, series_names, loc='upper right', prop=_legend_prop, fontsize=_legend_fs - 1)
        elif legend_loc == "bottom":
            _legend_prop = _legend_zh.copy() if _legend_zh is not None else None
            if _legend_prop is not None:
                _legend_prop.set_size(_legend_fs)
            fig.legend(handles, series_names, ncol=len(series_names), loc='lower center',
                       bbox_to_anchor=(0.5, -0.04), frameon=False,
                       handlelength=1.2, handleheight=1.2, prop=_legend_prop, fontsize=_legend_fs)

    # 隐藏多余子图
    for j in range(n_subplots, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=_suptitle_fs, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    if save_path:
        ensure_folder(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def grouped_bar_figure(
        data: Dict[str, Dict[str, float]],
        title: str = "Grouped Bar Chart",
        y_label: str = "ΔP",
        series_colors: Dict[str, Any] = None,
        figsize: Tuple[int, int] = None,
        bar_width: float = 0.7,
        hline_positions: List[Dict] = None,
        save_path: str = None,
        dpi: int = 200,
        ax: plt.Axes = None,
        show_legend: bool = True,
        show_bar_labels: bool = False,
        bar_label_fmt: str = "{:.2f}",
        fontsize: int = 12,
        title_fontsize: int = None,
        ylabel_fontsize: int = None,
        tick_fontsize: int = None,
        bar_label_fontsize: int = None,
        legend_fontsize: int = None,
        ylim_padding: float = 0.15,
        ytick_fmt: str = None,
        show_xticklabels: bool = True,
        zh_font=None,
) -> plt.Figure:
    """
    ...
    ytick_fmt: str
        Y轴刻度格式化，如 "{:.1f}"。None 则使用 matplotlib 默认
    show_xticklabels: bool
        是否显示 X 轴刻度标签，默认 True
        是否在柱状图顶部显示数值标签
    bar_label_fmt: str
        数值标签的格式化字符串
    fontsize: int
        基础字体大小，ylabel/xticklabels/title 在此基础上缩放
    """
    group_labels = list(data.keys())
    n_groups = len(group_labels)

    if n_groups == 0:
        raise ValueError("data 为空")

    # 收集所有系列名（保持出现顺序）
    series_names = []
    for group_data in data.values():
        for s_name in group_data:
            if s_name not in series_names:
                series_names.append(s_name)

    n_series = len(series_names)

    if n_series == 0:
        raise ValueError("data 中没有系列数据")

    # 默认颜色
    if series_colors is None:
        default_colors = plt.cm.Set2(np.linspace(0, 1, n_series))
        series_colors = {name: default_colors[i] for i, name in enumerate(series_names)}

    if ax is None:
        if figsize is None:
            figsize = (max(12, n_groups * 2.0), 7)
        fig, ax = plt.subplots(figsize=figsize)
        is_subplot = False
    else:
        fig = ax.figure
        is_subplot = True

    # 计算各元素实际字号（优先使用指定值，否则由 fontsize 派生）
    _title_fs = title_fontsize if title_fontsize is not None else fontsize + 4
    _ylabel_fs = ylabel_fontsize if ylabel_fontsize is not None else fontsize + 2
    _tick_fs = tick_fontsize if tick_fontsize is not None else fontsize
    _bar_label_fs = bar_label_fontsize if bar_label_fontsize is not None else fontsize - 3
    _legend_fs = legend_fontsize if legend_fontsize is not None else fontsize

    # 计算每个组内系列的位置偏移
    offsets = np.linspace(-bar_width / 2 + bar_width / (2 * n_series),
                          bar_width / 2 - bar_width / (2 * n_series),
                          n_series) if n_series > 1 else [0.0]
    single_bar_width = bar_width / n_series * 0.85

    bars_all = []
    for i, g_label in enumerate(group_labels):
        group_data = data[g_label]
        for k, s_name in enumerate(series_names):
            value = group_data.get(s_name, 0.0)
            pos = i + offsets[k]
            color = series_colors.get(s_name, "lightgray")
            bar = ax.bar(pos, value, single_bar_width, color=color, alpha=0.85,
                         edgecolor='white', linewidth=0.5)
            bars_all.append((bar, value))

    # 柱状图顶部数值标签
    if show_bar_labels:
        for bar_container, val in bars_all:
            if abs(val) < 0.001:
                continue
            for rect in bar_container:
                h = rect.get_height()
                va = 'bottom' if h >= 0 else 'top'
                offset = 2 if h >= 0 else -2
                ax.annotate(bar_label_fmt.format(val),
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, offset), textcoords='offset points',
                            ha='center', va=va, fontsize=_bar_label_fs)

    # x 轴
    ax.set_xticks(range(n_groups))
    if show_xticklabels:
        ax.set_xticklabels(group_labels, fontproperties=zh_font, fontsize=_tick_fs)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.5, n_groups - 0.5)

    # 水平参考线
    if hline_positions is None:
        hline_positions = [{"y": 0, "linestyle": "-", "color": "black", "linewidth": 0.8}]
    for hl in hline_positions:
        ax.axhline(y=hl["y"], color=hl.get("color", "black"),
                   linewidth=hl.get("linewidth", 0.8),
                   linestyle=hl.get("linestyle", "-"))

    ax.set_ylabel(y_label, fontproperties=zh_font, fontsize=_ylabel_fs)
    if title:
        ax.set_title(title, fontproperties=zh_font, fontsize=_title_fs, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=_tick_fs)
    if ytick_fmt is not None:
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: ytick_fmt.format(x)))

    # Y轴留白（纳入参考线位置，确保参考线可见）
    if ylim_padding > 0:
        all_values = [v for gd in data.values() for v in gd.values()]
        if hline_positions:
            for hl in hline_positions:
                all_values.append(hl["y"])
        if all_values:
            vmin, vmax = min(all_values), max(all_values)
            vrange = vmax - vmin if vmax != vmin else 1
            ax.set_ylim(vmin - vrange * ylim_padding,
                        vmax + vrange * ylim_padding)

    # 图例
    if show_legend and n_series > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, fc=series_colors.get(s, "lightgray"), alpha=0.85)
                   for s in series_names]
        _legend_prop = zh_font.copy() if zh_font is not None else None
        if _legend_prop is not None:
            _legend_prop.set_size(_legend_fs)
        ax.legend(handles, series_names, loc='upper left', bbox_to_anchor=(1.01, 1),
                  prop=_legend_prop, fontsize=_legend_fs, frameon=True)

    if not is_subplot:
        plt.tight_layout()
        if save_path:
            ensure_folder(os.path.dirname(save_path))
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        else:
            plt.show()

    return fig


if __name__ == "__main__":
    pass