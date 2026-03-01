# packages
import elapid as ela

import os
import gc
import warnings
from pprint import pprint
from typing import List, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio import plot as rioplot
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

# paths
https = "https://earth-chris.github.io/images/research"
csv = f"{https}/ariolimax-ca.csv"
gpkg = f"{https}/ariolimax-ca.gpkg"
raster_names = [
    "ca-cloudcover-mean.tif",
    "ca-cloudcover-stdv.tif",
    "ca-leafareaindex-mean.tif",
    "ca-leafareaindex-stdv.tif",
    "ca-surfacetemp-mean.tif",
    "ca-surfacetemp-stdv.tif",
]
rasters = [f"{https}/{raster}" for raster in raster_names]
labels = ["CloudCoverMean", "CloudCoverStdv", "LAIMean", "LAIStdv", "SurfaceTempMean", "SurfaceTempStdv"]


# preferences
mpl.style.use('ggplot')
warnings.filterwarnings("ignore")
pair_colors = ['#FFCC02', '#00458C']


def get_tgt_points_data_from_csv(csv_path: str, show_plot: bool = False, raster_sample: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    从csv文件中读取样本点（物种出现点）的位置信息，并转换为GeoDataFrame格式\n
    Parameters:
        csv: str, csv文件路径, 包含样本点的位置信息
    Returns:
        tgt_points: GeoDataFrame, 样本点的位置信息
    """
    # 样本点（物种出现点），通常以.csv格式存储，使用pandas读取csv文件
    df = pd.read_csv(csv_path)
    # print("DataFrame")
    # print(df.head())

    # csv文件里通常有经纬度的信息，将其转换为GeoSeries
    geometry = ela.xy_to_geoseries(
        x = df['wgs84lon'],
        y = df['wgs84lat']
    )
    # print("\nGeoSeries")
    # print(geometry.head())

    # 将剩余信息和转换好的地理信息GeoSeries合并，形成GeoDataFrame格式的信息
    tgt_points = gpd.GeoDataFrame(df[["name"]], geometry=geometry)
    print("\nGeoDataFrame")
    print(tgt_points.head())

    # 可视化
    if show_plot:
        fig, ax = plt.subplots(figsize=(5,5))
        if raster_sample is not None:
            with rio.open(raster_sample, 'r') as src:
                rioplot.show(src, ax=ax, cmap="gray")
                profile = src.profile.copy()
                print("profile metadata")
                pprint(profile)
        # overlay the ariolimax data
        tgt_points.to_crs(src.crs).plot(column='adname', ax=ax, legend=True)
        plt.show()

    return tgt_points

def sample_bg_points_data(raster: str, count: int = 10_000, show_plot: bool = False) -> gpd.GeoDataFrame:
    """
    从TIF影像中随机采样10000个点作为背景点（物种不出现点）\n
    Parameters:
        raster: str, 要采样的影像路径
        count: int, 要采样的点的数量, 默认为10000
    Returns:
        bg_points: GeoDataFrame, 背景点的位置信息
    """
    # 在非nodata区域随机采样10000个点，作为背景点（物种不出现点）
    bg_points = ela.sample_raster(raster, count)
    # 有些点可能被重复采样，这里去掉重复的点
    # bg_points.drop_duplicates(inplace=True)

    # 可视化
    if show_plot:
        bg_points.plot(markersize=0.5)
        plt.show()

    return bg_points

def show_raster_img(tgt_points: gpd.GeoDataFrame, raster: Optional[str] = None) -> None:
    """
    展示TIF影像，并在其上画出样本点\n
    Parameters:
        tgt_points: GeoDataFrame, 样本点的位置信息
        raster: str, 要展示的影像路径
    """
    if raster is None:
        print("No raster provided, please provide a raster path")
        return
    fig, ax = plt.subplots(figsize=(6,6))
    with rio.open(raster, 'r') as src:
        profile = src.profile.copy()
        print("profile metadata")
        pprint(profile)
        # {'blockxsize': 512,
        # 'blockysize': 512,
        # 'compress': 'lzw',
        # 'count': 1,
        # 'crs': CRS.from_epsg(32610),
        # 'driver': 'GTiff',
        # 'dtype': 'float32',
        # 'height': 1040,
        # 'interleave': 'band',
        # 'nodata': -9999.0,
        # 'tiled': True,
        # 'transform': Affine(1000.0, 0.0, 380000.0,
        #     0.0, -1000.0, 4654000.0),
        # 'width': 938}
        
        # 展示TIF影像，黑色表示低值，白色表示高值
        rioplot.show(src, ax=ax, cmap="gray")
        
        # 在TIF影像上画出样本点
        # tgt_points.to_crs(src.crs).plot(column='adname', ax=ax, legend=True)

        plt.show()

def get_raster_data_from_points(tgt_points, bg_points, rasters, labels=None, show_plot=False):
    """
    根据样本点和背景点的位置，从栅格数据中获取对应影像数据.\n
    注意：需要先融合样本点和背景点，再获取栅格数据，否则会由于样本点和背景点的字段不一致，导致最终融合失败，失去栅格数据字段的信息。
    \n
    Parameters:
        tgt_points: GeoDataFrame, 样本点的位置信息
        bg_points: GeoDataFrame, 背景点的位置信息
        rasters: List, 包含了各栅格数据的路径
        labels: List, 包含了各栅格数据的标签
    Returns:
        merged: GeoDataFrame, 样本点的影像数据
    """
    # # 若存在多种目标，则选择所需的值作为目标
    # tgt = tgt_points[tgt_points['species'] == 'Ariolimax buttoni']
    # presence = ela.annotate(
    #     tgt.geometry,
    #     rasters,
    #     labels=labels,
    #     drop_na=True,
    # )

    # 可视化
    if show_plot:
        presence = ela.annotate(tgt_points, rasters, labels=labels, drop_na=True)
        print("\nGeoDataFrame presence")
        print(presence.head())

        background = ela.annotate(bg_points, rasters, labels=labels, drop_na=True)
        print("\nGeoDataFrame background")
        print(background.head())

        compare_bg_vs_tgt_data(presence, background, labels)

    merged = ela.stack_geometries(tgt_points, bg_points, add_class_label=True)
    merged = ela.annotate(merged, rasters, labels=labels, drop_na=True)
    print("\nGeoDataFrame merged")
    print(merged.head())
    return merged


def compare_bg_vs_tgt_data(presence: gpd.GeoDataFrame, background: gpd.GeoDataFrame, labels: List[str]) -> None:
    """
    比较样本点和背景点在六个环境变量（即栅格）上的数据，绘制相应的直方图\n
    Parameters:
        presence: GeoDataFrame, 样本点数据
        background: GeoDataFrame, 背景点数据
    """
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6,9))

    for label, ax in zip(labels, axs.ravel()):
        pvar = presence[label]
        bvar = background[label]
        ax.hist(
            [pvar, bvar],
            density=True,
            alpha=0.7,
            label=['presence', 'background'],
            color=pair_colors,
        )
        ax.set_title(label)
        
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, loc=(0.6, 0.9))
    plt.tight_layout()


def train_with_alldata(csv, rasters, labels, save_path, show_plot=False):
    """
    读取csv文件和栅格数据，训练模型，保存模型\n
    Parameters:
        csv: str, csv文件路径, 包含样本点的位置信息
        rasters: List[str], 包含了各栅格数据的路径
        labels: List[str], 包含了各栅格数据的标签
        save_path: str, 保存模型的路径
        show_plot: bool, 是否可视化过程
    """
    # 读取样本点数据
    tgt_points = get_tgt_points_data_from_csv(csv, show_plot=show_plot, raster_sample=rasters[0])
    print("Finished reading Target points data")

    # 随机采样10000个点作为背景点
    bg_points = sample_bg_points_data(rasters[0], show_plot=show_plot)
    print("Finished sampling Background points data")

    # 合并样本点和背景点，并获取对应的栅格数据
    merged_data = get_raster_data_from_points(tgt_points, bg_points, rasters, labels, show_plot=show_plot)
    print("Finished getting raster data")

    # 获取x (环境变量) 和 y (类别标签)
    x = merged_data.drop(columns=['geometry', 'class'])
    y = merged_data['class']

    # 设置模型参数
    beta = 2.0
    model = ela.MaxentModel(beta_multiplier=beta, use_sklearn=True)
    # pprint(maxent.get_params())  # 查看默认参数

    # 训练模型（使用全部数据）
    model.fit(x, y)
    print("Finished training model")

    # 预测训练集
    ypred = model.predict(x)
    # 预测结果评估
    auc = metrics.roc_auc_score(y, ypred)
    print(f"Training AUC score: {auc:0.3f}")

    # 保存模型
    ela.save_object(model, save_path)
    print(f"Model saved to {save_path}")



def train_with_split_data():
    # 读取样本点数据
    tgt_points = get_tgt_points_data_from_csv(csv)
    # 随机采样10000个点作为背景点
    bg_points = sample_bg_points_data(rasters[0])

    # 合并样本点和背景点，并获取对应的栅格数据
    merged_data = get_raster_data_from_points(tgt_points, bg_points, rasters)

    # 划分训练集和测试集
    # let's use a 50km grid
    grid_size = 50_000
    train, test = ela.checkerboard_split(merged_data, grid_size=grid_size)

    # we'll re-merge them for plotting purposes
    train['split'] = 'train'
    test['split'] = 'test'
    checker = ela.stack_geodataframes(train, test)
    checker.plot(column='split', markersize=0.75, legend=True)


    xtrain = train.drop(columns=['class', 'SampleWeight', 'split'])
    ytrain = train['class']
    sample_weight_train = train['SampleWeight']

    xtest = test.drop(columns=['class', 'SampleWeight', 'split'])
    ytest = test['class']
    sample_weight_test = test['SampleWeight']

    # 设置模型参数
    maxent = ela.MaxentModel()
    # pprint(maxent.get_params())  # 查看默认参数

    # 训练模型(仅使用训练集)
    maxent.fit(xtrain, ytrain, sample_weight=sample_weight_train)
    # 预测测试集
    ypred = maxent.predict(xtest)
    # 预测结果评估
    print(f"Unweighted checkerboard AUC score: {metrics.roc_auc_score(ytest, ypred):0.3f}")
    print(f"Weighted checkerboard AUC score: {metrics.roc_auc_score(ytest, ypred, sample_weight=sample_weight_test):0.3f}")
    
    # 保存模型
    ela.save_object(maxent, 'demo-maxent-model.ela')

def predict_and_save_raster(model_path, rasters, result_folder, result_name='maxent-predictions.tif'):
    """
    根据模型预测栅格数据，并保存结果到磁盘\n
    Parameters:
        model_path: str, 模型路径
        rasters: list[str], 包含了各栅格数据的路径
        result_folder: str, 保存结果的文件夹路径
        result_name: str, 保存结果的文件名
    """
    gc.collect()  # free up memory before loading the model
    # 加载模型
    model = ela.load_object(model_path)
    print(f"Loading model from {model_path} ...")
    if model is None:
        raise ValueError("Failed to load model from {}".format(model_path))

    # write the model predictions to disk
    result_name = result_name if result_name is not None else 'maxent-predictions.tif'
    assert result_name.endswith('.tif'), "Result file name must end with '.tif'"
    result_path = os.path.join(result_folder, result_name)

    ela.apply_model_to_rasters(model, rasters, result_path, transform="logistic")
    print(f"Model predictions saved to {result_path}")

    # and read into memory
    with rio.open(result_path, 'r') as src:
        pred = src.read(1, masked=True)

        # plot the suitability predictions
        fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=100)
        plot = ax.imshow(pred, vmin=0, vmax=1, cmap='GnBu')
        ax.set_title('$FJMS$ suitability')
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(plot, ax=ax, label="relative occurrence probability", pad=0.04)
        plt.tight_layout()
        plt.show()
    
def predict_and_return_arrays(model_path, raster_arrays, labels:list):
    """
    根据模型预测栅格数据，并返回结果数组\n
    Parameters:
        model_path: str, 模型路径
        raster_arrays: ndarray[nbands, nrows, ncols], 各栅格数据的数组
        labels: list[str], 包含了各栅格数据的标签
    Returns:
        pred: np.ndarray, 模型预测的结果数组
    """
    gc.collect()  # free up memory before loading the model
    # 加载模型
    model = ela.load_object(model_path)
    print(f"Loading model from {model_path} ...")
    if model is None:
        raise ValueError("Failed to load model from {}".format(model_path))
    
    nodata = 0.0
    LU_idx = labels.index('LU')
    nodata_idx = np.where(raster_arrays[LU_idx] == nodata)
    pred = ela.apply_model_to_array(model, raster_arrays, nodata, nodata_idx)
    print(pred)
    return pred

def train_pipeline(root, save_path):
    """
    从根目录读取训练数据，包括csv文件路径，栅格数据路径，训练模型并保存到相应路径\n
    Parameters:
        root: str, 训练数据根目录, 包含csv文件和栅格数据
        save_path: str, 保存模型的路径
    """
    # 读取csv文件路径
    csv_name = [f for f in os.listdir(root) if f.endswith('.csv')]
    csv_path = os.path.join(root, csv_name[0])
    # 读取栅格数据路径
    raster_names=[f for f in os.listdir(root) if f.endswith('.tif')]
    rasters = [os.path.join(root, raster) for raster in raster_names]
    # 读取标签
    labels = [name.split('.')[0] for name in raster_names]
    print(f"rasters: {raster_names}")
    # 训练模型并保存
    train_with_alldata(csv_path, rasters, labels=labels, save_path=save_path, show_plot=False)


# if __name__ == '__main__':
#     data_path="D:\\_programming\\_LAB_ASSIGMENTS\\_data\\valid_20\\_maxent"

#     csv_name = [f for f in os.listdir(data_path) if f.endswith('.csv')]
#     csv = os.path.join(data_path, csv_name[0])

#     raster_names=[f for f in os.listdir(data_path) if f.endswith('.tif')]
#     rasters = [os.path.join(data_path, raster) for raster in raster_names]
#     labels = [name.split('.')[0] for name in raster_names]

#     model_name = 'demo-maxent-model_sklearn.ela'
#     model_path = os.path.join(data_path, model_name)

#     train_with_alldata(csv, rasters,labels=labels, save_path=model_path, show_plot=False)

#     result_name = 'demo-maxent-predictions.tif'
#     result_folder = data_path + '\\' + 'output'
#     predict_and_save_raster(model_path, rasters, result_folder=result_folder, result_name=result_name)
