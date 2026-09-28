# 生成论文用图表，包括折线图、柱状图、热力图等
import os
import sys
from typing import Any, List, Dict, Tuple, Callable
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

from utils.io_utils import ensure_folder


def _resolve_font_family() -> List[str]:
    """英文 Arial Bold + 中文黑体 SimHei，宽松回退。

    只保留系统实际安装的字体: 否则 matplotlib 每次按 FontProperties 查找都会对缺失
    字体逐个打印 findfont 警告(逐图逐文本刷屏, 日志暴涨), 且重复走字体扫描.
    """
    if sys.platform == 'win32':
        candidates = ['Arial', 'SimHei']
    else:
        candidates = ['Arial', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei']
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = [name for name in candidates if name in installed]
    family.append('sans-serif')
    return family


def setup_plot_fonts(base_size: float = 12) -> Dict[str, Any]:
    """统一字体与背景：中英文全部加粗 + 白色背景。

    返回 {'title': prop, 'label': prop, 'tick': prop, 'legend': prop, 'annot': prop, 'prop': factory}.
    """
    family = _resolve_font_family()

    matplotlib.rcParams['font.family'] = family
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['axes.facecolor'] = 'white'
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    matplotlib.rcParams['axes.linewidth'] = 1.0
    matplotlib.rcParams['text.color'] = 'black'
    matplotlib.rcParams['axes.labelcolor'] = 'black'
    matplotlib.rcParams['xtick.color'] = 'black'
    matplotlib.rcParams['ytick.color'] = 'black'

    def mkprop(size: float = None) -> FontProperties:
        p = FontProperties(family=family, weight='bold')
        p.set_size(size if size is not None else base_size)
        return p

    return {
        'title':  mkprop(base_size + 2),
        'label':  mkprop(base_size),
        'tick':   mkprop(base_size - 2),
        'legend': mkprop(base_size - 2),
        'annot':  mkprop(max(base_size - 3, 7)),
        'prop':   mkprop,   # 工厂函数，调用方可派生特定字号的 prop
    }


def setup_paper_plot_fonts(base_size: int = 26) -> Dict[str, Any]:
    return setup_plot_fonts(base_size=base_size)


def setup_large_plot_fonts() -> Dict[str, Any]:
    return setup_plot_fonts(base_size=24)


def setup_tornado_fonts() -> Dict[str, Any]:
    return setup_plot_fonts(base_size=14)


def setup_ablation_fonts() -> Dict[str, Any]:
    return setup_plot_fonts(base_size=13)

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
    FONT = setup_paper_plot_fonts()
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
    ax.set_ylabel(y_label, fontproperties=FONT['label'])
    ax.set_title(title, fontproperties=FONT['title'])

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
        ax.legend(handles=handles, prop=FONT['legend'])

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
    FONT = setup_paper_plot_fonts()
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
    ax.set_ylabel(y_label, fontproperties=FONT['label'])
    ax.set_title(title, fontproperties=FONT['title'])
    ax.legend(loc="upper right", ncol=len(data_dict), prop=FONT['legend'])

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
    FONT = setup_paper_plot_fonts()
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
    ax.set_ylabel(y_label, fontproperties=FONT['label'])
    ax.set_title(title, fontproperties=FONT['title'])

    # 添加网格线
    ax.grid(True, alpha=0.5, axis='y', color='gray', linestyle='--')

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
    FONT = setup_paper_plot_fonts()
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
    ax.set_ylabel(y_label, fontproperties=FONT['label'])
    ax.set_title(title, fontproperties=FONT['title'])

    # 添加网格线
    ax.grid(True, alpha=0.5, axis='y' if vert else 'x', color='gray', linestyle='--')

    # 如果垂直显示，设置x轴标签旋转
    if vert and show_labels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # 添加子图标注
    if subplot_label:
        ax.annotate(
            subplot_label,
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            verticalalignment='top', fontproperties=FONT['annot'],
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
    FONT = setup_paper_plot_fonts()
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
    legend_prop = FONT['prop'](fontsize)
    legend_title_prop = FONT['prop'](fontsize + 2)
    legend_obj = ax.legend(handles, labels,
                          loc=loc,
                          ncol=ncol,
                          prop=legend_prop,
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
        legend_obj.set_title(title, prop=legend_title_prop)

    # 调整布局
    plt.tight_layout()
    plt.show()
    return fig




# ── 箱线图配色常量 ──
_SA_COLOR   = '#1976D2'  # 蓝 - SA 方法
_GA_COLOR   = '#FB8C00'  # 橙 - GA 方法
_MEDIAN_COLOR = 'white'   # 中位数线
_MEAN_COLOR = '#212121'   # 深灰 (保留给均值标记，已弃用均值显示)
_FLIER_FILL = '#E53935'  # 红
_FLIER_EDGE = '#B71C1C'  # 深红
_TWO_SERIES_COLORS = ['#1976D2', '#FB8C00']


def _sig_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def grouped_box_plot_figure(
        data: Dict[str, Dict[str, Dict[str, List[float]]]],
        title: str = "Grouped Box Plot",
        y_label: str = "Values",
        series_colors: Dict[str, Any] = None,
        subplot_layout: Tuple[int, int] = None,
        figsize: Tuple[int, int] = None,
        showfliers: bool = False,
        showmeans: bool = False,
        width: float = 0.55,
        hspace: float = 0.35,
        wspace: float = 0.25,
        unified_ylim: bool = False,
        exclude_fliers_for_ylim: bool = True,
        ylim_margin: float = 0.05,
        save_path: str = None,
        dpi: int = 200,
        legend_loc: str = "subplot",
        show_scatter: bool = True,
        show_conn_lines: bool = True,
        show_significance: bool = True,
) -> plt.Figure:
    """分组箱线图 + 散点叠加 + 均值配对连线 + 显著性星号。

    legend_loc: "subplot" / "bottom" / None.
    双系列时自动使用 SA/GA 配色方案。
    """
    from scipy.stats import mannwhitneyu

    FONT = setup_paper_plot_fonts()

    subplot_titles = list(data.keys())
    n_subplots = len(subplot_titles)

    if n_subplots == 0:
        raise ValueError("data 为空")

    # 收集组标签 & 系列名
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

    # 默认颜色：2 系列自动用 SA/GA 配色
    if series_colors is None:
        if n_series == 2:
            series_colors = {series_names[0]: _SA_COLOR, series_names[1]: _GA_COLOR}
        else:
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
                    if not vals: continue
                    arr = np.array(vals)
                    if exclude_fliers_for_ylim:
                        q1, q3 = np.percentile(arr, [25, 75])
                        iqr = q3 - q1
                        arr = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
                    all_vals.extend(arr.tolist())
        if all_vals:
            vmin, vmax = np.min(all_vals), np.max(all_vals)
            margin = (vmax - vmin) * ylim_margin
            global_ylim = (vmin - margin, vmax + margin)

    # 组内偏移
    total_series_width = width * n_series
    offsets = np.linspace(-total_series_width / 2 + width / 2,
                          total_series_width / 2 - width / 2,
                          n_series) if n_series > 1 else [0.0]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    title_prop = FONT['title']
    ylabel_prop = FONT['label']
    tick_prop = FONT['tick']
    legend_prop = FONT['legend']
    suptitle_prop = FONT['prop'](FONT['title'].get_size() + 2)
    annot_prop = FONT['annot']

    for i, (subplot_title, subplot_data) in enumerate(data.items()):
        ax = axes[i]
        positions_all = []
        data_all = []
        colors_all = []

        for j, g_label in enumerate(group_labels):
            group_center = j * 2
            group_data = subplot_data.get(g_label, {})
            for k, s_name in enumerate(series_names):
                vals = group_data.get(s_name, [])
                pos = group_center + offsets[k]
                positions_all.append(pos)
                data_all.append(vals)
                colors_all.append(series_colors.get(s_name, "lightgray"))

        # 箱线图
        bp = ax.boxplot(data_all, positions=positions_all, widths=width,
                        patch_artist=True, showfliers=False,  # fliers 手动散点绘制
                        showmeans=showmeans,
                        meanprops=dict(marker='D', markerfacecolor=_MEAN_COLOR,
                                       markeredgecolor=_MEAN_COLOR, markersize=7, zorder=5))

        # 箱体: 边框/须/帽线 + 30% 透明度填充
        for patch, color in zip(bp['boxes'], colors_all):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.2)
        for whisker, color in zip(bp['whiskers'], [c for c in colors_all for _ in range(2)]):
            whisker.set_color(color)
            whisker.set_linewidth(1.2)
        for cap, color in zip(bp['caps'], [c for c in colors_all for _ in range(2)]):
            cap.set_color(color)
            cap.set_linewidth(1.2)
        for median in bp['medians']:
            median.set_color(_MEDIAN_COLOR)
            median.set_linewidth(1.8)

        # 异常值散点
        if showfliers:
            for k, vals in enumerate(data_all):
                if not vals: continue
                arr = np.array(vals)
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                fliers = arr[(arr < lo) | (arr > hi)]
                if len(fliers):
                    ax.scatter(np.full_like(fliers, positions_all[k], dtype=float),
                               fliers, s=20, facecolor=_FLIER_FILL,
                               edgecolor=_FLIER_EDGE, linewidth=0.5, zorder=4, alpha=0.8)

        # 散点叠加（非异常值）
        if show_scatter:
            for k, vals in enumerate(data_all):
                if not vals: continue
                arr = np.array(vals)
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                inliers = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
                if len(inliers):
                    jitter = np.random.default_rng(42 + i * 100 + k).uniform(-width * 0.18, width * 0.18, len(inliers))
                    ax.scatter(np.full_like(inliers, positions_all[k], dtype=float) + jitter,
                               inliers, s=12, facecolor=colors_all[k],
                               edgecolor='white', linewidth=0.3, alpha=0.45, zorder=3)

        # 显著性括号 + 星号（仅双系列）
        if n_series == 2 and (show_conn_lines or show_significance):
            for j, g_label in enumerate(group_labels):
                group_center = j * 2
                group_data = subplot_data.get(g_label, {})
                vals_a = group_data.get(series_names[0], [])
                vals_b = group_data.get(series_names[1], [])
                if not vals_a or not vals_b: continue
                pos_a = group_center + offsets[0]
                pos_b = group_center + offsets[1]

                # 确定可视区域范围
                if global_ylim is not None:
                    y_lim_upper = global_ylim[1]
                    y_range = global_ylim[1] - global_ylim[0]
                else:
                    y_lim_upper = ax.get_ylim()[1]
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

                # 须顶 = 排除离群值后的最大值 (q3+1.5*iqr 围栏内的 max)
                def _upper_whisker(v):
                    a = np.array(v)
                    q1, q3 = np.percentile(a, [25, 75])
                    fence = q3 + 1.5 * (q3 - q1)
                    within = a[a <= fence]
                    return float(np.max(within)) if len(within) > 0 else float(fence)
                w_top_a = _upper_whisker(vals_a)
                w_top_b = _upper_whisker(vals_b)
                y_bracket = max(w_top_a, w_top_b) + y_range * 0.04
                y_bracket = min(y_bracket, y_lim_upper - y_range * 0.03)

                if show_conn_lines:
                    tick_h = y_range * 0.012
                    # 水平横线
                    ax.plot([pos_a, pos_b], [y_bracket, y_bracket],
                            color=_MEAN_COLOR, linestyle='-', linewidth=1.0, alpha=0.6, zorder=2)
                    # 左/右竖线 tick
                    ax.plot([pos_a, pos_a], [y_bracket - tick_h, y_bracket],
                            color=_MEAN_COLOR, linewidth=1.0, alpha=0.6, zorder=2)
                    ax.plot([pos_b, pos_b], [y_bracket - tick_h, y_bracket],
                            color=_MEAN_COLOR, linewidth=1.0, alpha=0.6, zorder=2)

                if show_significance:
                    try:
                        _, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
                    except Exception:
                        p = 1.0
                    stars = _sig_stars(p)
                    mid_x = (pos_a + pos_b) / 2
                    ax.text(mid_x, y_bracket + y_range * 0.005, stars,
                            ha='center', va='bottom',
                            fontproperties=annot_prop, color=_MEAN_COLOR, zorder=6)

        # x 轴
        group_centers = [j * 2 for j in range(n_groups)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_labels, fontproperties=tick_prop)

        ax.set_title(subplot_title, fontproperties=title_prop)
        ax.set_ylabel(y_label, fontproperties=ylabel_prop)

        if global_ylim is not None:
            ax.set_ylim(global_ylim)
        else:
            # 无统一 ylim 时, 自动范围上下各扩展 8% 留呼吸空间
            ylo, yhi = ax.get_ylim()
            ypad = (yhi - ylo) * 0.08
            ax.set_ylim(ylo - ypad, yhi + ypad)

        ax.tick_params(axis='y', labelsize=FONT['tick'].get_size())
        ax.grid(True, alpha=0.5, axis='y', color='gray', linestyle='--')

    # 图例
    if legend_loc is not None and n_series > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, fc=series_colors.get(s, "lightgray"), alpha=0.3, ec=series_colors.get(s, "lightgray"), linewidth=1.2)
                   for s in series_names]
        if legend_loc == "subplot":
            axes[0].legend(handles, series_names, loc='upper right', prop=legend_prop)
        elif legend_loc == "bottom":
            fig.legend(handles, series_names, ncol=len(series_names), loc='lower center',
                       bbox_to_anchor=(0.5, -0.03), frameon=False,
                       handlelength=1.2, handleheight=1.2, prop=legend_prop)

    for j in range(n_subplots, len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontproperties=suptitle_prop, y=0.98)
    # tight_layout: 底部图例时预留少量下方空间
    if legend_loc == 'bottom':
        plt.tight_layout(rect=[0, 0.06, 1, 1])
    else:
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
        ylim_padding: float = 0.15,
        ytick_fmt: str = None,
        show_xticklabels: bool = True,
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
    """
    FONT = setup_paper_plot_fonts()

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

    title_prop = FONT['title']
    ylabel_prop = FONT['label']
    tick_prop = FONT['tick']
    annot_prop = FONT['annot']
    legend_prop = FONT['legend']

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
                            ha='center', va=va, fontproperties=annot_prop)

    # x 轴
    ax.set_xticks(range(n_groups))
    if show_xticklabels:
        ax.set_xticklabels(group_labels, fontproperties=tick_prop)
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

    ax.set_ylabel(y_label, fontproperties=ylabel_prop)
    if title:
        ax.set_title(title, fontproperties=title_prop)
    ax.grid(True, alpha=0.5, axis='y', color='gray', linestyle='--')
    ax.tick_params(axis='both', labelsize=FONT['tick'].get_size())
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
        ax.legend(handles, series_names, loc='upper left', bbox_to_anchor=(1.01, 1),
                  prop=legend_prop, frameon=True)

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