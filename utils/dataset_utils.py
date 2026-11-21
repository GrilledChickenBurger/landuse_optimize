# 数据集相关工具，包括训练模型所需数据集、随机优化算法运行所需数据集

import os
import copy
from numbers import Number
from typing import List, Dict
import numpy as np
# import rasterio
# from osgeo import gdal
from scipy.ndimage import label, generate_binary_structure
import torch
os.environ['PROJ_LIB'] = 'D:/anaconda3/envs/pytorch/Library/share/proj'
os.environ['GDAL_DATA'] = 'D:/anaconda3/envs/pytorch/Library/share'

from libs.const import *
from utils.tools import convert_to_nparray

def sort_key(file_path: str) -> int:
    """
    按文件名末尾编号排序（形如 ..._123.TIF -> 123）
    """
    parts = file_path.split("/")
    filename_parts = parts[-1].replace(".TIF", "").split("_")
    step_num = int(filename_parts[-1])
    return step_num

def remove_lines(area: np.ndarray, threshold: int = None) -> np.ndarray:
    """
    差异图里如果有大块的改变区域，则去掉其他细小的区域（训练前处理）

    Parameters:
    ----------
        area: [128, 128]，numpy 或 torch
        threshold: 阈值，默认 PIXEL_NUM_THRESHOLD_REMOVE
    """
    if isinstance(area, torch.Tensor):
        area = area.squeeze().numpy()
    threshold = PIXEL_NUM_THRESHOLD_REMOVE if threshold is None else int(threshold)
    new_area = copy.deepcopy(area)

    structure = generate_binary_structure(2, 1)
    labeled_array, num_features = label(new_area, structure=structure)
    for j in range(1, num_features + 1):
        s = np.sum(labeled_array == j)
        if s < threshold:
            new_area[labeled_array == j] = 0
    return new_area


# remove_lines_forbatch 的逐样本结果缓存: GT 是静态的, 同一张 0/1 图每 epoch 都会重算,
# 按 (threshold, dtype, 形状, 内容) 缓存可跳过重复的连通域标记; 输出与原实现逐位一致.
_REMOVE_LINES_CACHE = {}
_REMOVE_LINES_CACHE_MAX = 8192


def remove_lines_forbatch(area: torch.Tensor, threshold: int = None) -> torch.Tensor:
    """
    批量版小斑块移除（训练前处理）
    
    Parameters:
    ----------
        area: [batch size, 1, 128, 128]
        threshold: 阈值，默认 PIXEL_NUM_THRESHOLD_REMOVE
    """
    area = convert_to_nparray(area)
    batch_size = area.shape[0]
    threshold = PIXEL_NUM_THRESHOLD_REMOVE if threshold is None else int(threshold)
    new_area = copy.deepcopy(area)
    # 预计算结构体，避免循环中重复创建
    structure = generate_binary_structure(2, 1)
    for i in range(batch_size):
        single = np.ascontiguousarray(new_area[i][0])
        # dtype 必须进 key: 同形状不同 dtype 的数组可能字节内容相同, 否则会串类型
        key = (threshold, single.dtype.str, single.shape, single.tobytes())
        cached = _REMOVE_LINES_CACHE.get(key)
        if cached is None:
            arr = single.copy()
            labeled_array, num_features = label(arr, structure=structure)
            for j in range(1, num_features + 1):
                if np.sum(labeled_array == j) < threshold:
                    arr[labeled_array == j] = 0
            cached = arr
            if len(_REMOVE_LINES_CACHE) >= _REMOVE_LINES_CACHE_MAX:
                _REMOVE_LINES_CACHE.clear()
            _REMOVE_LINES_CACHE[key] = cached
        new_area[i][0] = cached
    assert area.shape == new_area.shape
    return torch.from_numpy(new_area)

def load_tiff(file_path: str) -> np.ndarray:
    """
    使用 GDAL 优先，rasterio备用的方式加载单张 TIFF 文件

    Parameters:
    ----------
    file_path: str, 文件路径

    Returns:
    ----------
    tif_array: np.ndarray, 加载的 TIFF 数组
    """
    # 先尝试使用 GDAL 读取
    try:
        from osgeo import gdal
        dataset = gdal.Open(file_path)
        if dataset is not None:
            tif_array = dataset.ReadAsArray().astype(np.float32)
            # print(f"✓ GDAL成功读取: {file_path}")
            return tif_array
    except ImportError:
        pass
        # print("警告: GDAL不可用，将尝试使用rasterio")
    except Exception as e:
        print(f"GDAL读取失败 {file_path}: {e}")
    
    # GDAL失败后尝试使用 rasterio
    try:
        import rasterio
        with rasterio.open(file_path) as src:
            tif_array = src.read(1)
            # print(f"✓ rasterio成功读取: {file_path}")
            return tif_array
    except Exception as e:
        print(f"rasterio读取失败 {file_path}: {e}")
    
    # 两种方法都失败，报错
    raise ValueError(f"❌ 无法读取文件: {file_path}")



def get_year_data(
    root: str, 
    key_list: List[str], 
    shape: List[int], 
    max_workers: int = 4, 
    use_parallel: bool = True) -> np.ndarray:
    """
    获取指定年份、因子列表的所有数据

    Parameters:
    ----------
    root: str, 数据根目录
    year: str, 年份
    key_list: List[str], 因子列表
    shape: List[int], 数据形状，[1, H, W]
    max_workers: int, 并行工作线程数
    use_parallel: bool, 是否使用并行处理
    
    Returns:
    ----------
    nparr: np.ndarray, 包含所有数据的 numpy.float32 数组，shape为(num, len(key_list), H, W)
    """
    def load_single_file(file_info):
        """加载单个文件"""
        filename, factor_idx, sample_idx = file_info
        
        try:
            # 使用封装好的 load_tiff 函数
            tif_array = load_tiff(filename)
            assert tif_array.shape == (shape[1], shape[2]), f"文件 {filename} 形状错误，期望 {shape[1], shape[2]}，实际 {tif_array.shape}"
            return factor_idx, sample_idx, tif_array
        except Exception as e:
            print(f"文件读取失败 {filename}: {e}")
            return None
    
    # 获取文件数量
    num = len(os.listdir(os.path.join(root, key_list[0])))
    nparr=np.empty((num, len(key_list), shape[1], shape[2]), dtype=np.float32)
    
    # 准备所有要加载的文件信息: 每个文件对应一个 (filename, factor_idx, sample_idx) 元组
    file_infos = []
    for i, key in enumerate(key_list):
        key_files_folder = os.path.join(root, key)
        key_files_sorted = sorted(
            [
                os.path.join(key_files_folder, f)
                for f in os.listdir(key_files_folder)
                if f.endswith(".TIF")
            ],
            key=sort_key,
        )
        print("Data: getting FACTOR {} from: {}.".format( key, key_files_folder))

        for idx, filename in enumerate(key_files_sorted):
            file_infos.append((filename, i, idx))
    
    if use_parallel and max_workers > 1:
        # 使用线程池并行加载
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(load_single_file, file_infos))
    else:
        # 串行加载（兼容原版）
        results = [load_single_file(info) for info in file_infos]
    
    # 将结果填充到数组中
    for factor_idx, sample_idx, tif_array in results:
        nparr[sample_idx, factor_idx, :, :] = tif_array 

    return nparr

def apply_norm(x: np.ndarray, name: str, params: Dict[str, Dict[str, float]], log: bool = False, mask: np.ndarray = None) -> np.ndarray:
    """
    统一归一化接口（训练 / 推理共用）

    Parameters:
    ----------
        x: np.ndarray, 待归一化的张量，[B, H, W]
        name: str, 张量名称，用于从params中获取对应参数
        params: Dict[str, Dict[str, float]], 归一化参数字典
        log: bool, 是否对输入取对数
            当name in ALL_LOG_FACTORS时，log=True
        mask: np.ndarray, [B, H, W], True = valid
    
    Returns:
    ----------
    x_norm: np.ndarray, 归一化后的张量，[B, H, W]
    """
    if log:
        x = np.log1p(x)

    miu = params[name]["miu"]
    eps = params[name]["epsilon"]

    x_norm = (x - miu) / eps

    if mask is not None:
        x_norm = np.where(mask, x_norm, np.zeros_like(x_norm))

    print(f"apply_norm: FACTOR {name}, log={log}, miu={miu:.4f}, epsilon={eps:.4f}")
    return x_norm


def one_hot_lu(lu_tensor: torch.Tensor) -> torch.Tensor:
    """
    将LU图转换为one-hot编码（6通道：耕地、林地、草地、水域、建设用地、未利用地）

    Parameters:
    ----------
        lu_tensor: torch.Tensor, 支持 [1, H, W] 或 [H, W] 形状
    
    Returns:
    ----------
        out: torch.Tensor, [6, H, W]
    """
    # 处理不同维度的输入
    if lu_tensor.dim() == 4:
        # [N, 1, H, W] -> [N, H, W]
        lu_tensor = lu_tensor.squeeze(1)
    elif lu_tensor.dim() == 3:
        # [1, H, W] -> [H, W]
        if lu_tensor.shape[0] == 1:
            lu_tensor = lu_tensor.squeeze(0)
    
    # 确保输入为2D或3D张量
    if lu_tensor.dim() == 2:
        # [H, W] -> [1, H, W]
        lu_tensor = lu_tensor.unsqueeze(0)
    elif lu_tensor.dim() == 3:
        # [N, H, W] 保持不变
        pass
    else:
        raise ValueError(f"不支持的输入维度: {lu_tensor.dim()}，期望2D或3D张量")
    
    N, H, W = lu_tensor.shape
    
    # 定义6个通道的LU类型映射
    lu_category_mapping = {
        0: [11, 12],    # 耕地
        1: [21, 22, 23, 24],  # 林地
        2: [31, 32, 33],      # 草地
        3: [41, 42, 43, 45, 46],  # 水域
        4: [51, 52, 53],      # 建设用地
        5: [65, 66]           # 未利用地
    }
    
    # 创建6通道的输出张量 [N, 6, H, W]
    out = torch.zeros(N, 6, H, W, device=lu_tensor.device)
    
    # 为每个通道设置对应的LU类型
    for channel, lu_types in lu_category_mapping.items():
        for lu_type in lu_types:
            out[:, channel][lu_tensor == lu_type] = 1
    
    # 确保输出形状为 [6, H, W]
    if N == 1:
        # [1, 6, H, W] -> [6, H, W]
        out = out.squeeze(0)
    
    return out
