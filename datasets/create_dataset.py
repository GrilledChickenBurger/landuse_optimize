# 重构后未经验证
import os
import json
import numpy as np
import shutil
from typing import List, Optional
from utils.dataset_utils import load_tiff
from utils.tools import ensure_folder
from libs.const import ALL_YEARS, ALL_FACTORS

def create_maxent_dataset(root: str, folder: str, rasters_list: List[str]) -> None:
    """
    root: 'AAAAA栅格分割/_overview', 包含合并后的栅格数据
    folder: '_data'
    rasters_list: ['LU', 'DL','PODU','WATER','CITY']
            landuse、距道路距离、坡度、距水源距离、距城镇距离
    """
    if rasters_list is None or len(rasters_list) == 0:
        print("No rasters to create dataset.")
        return
    years = ["10", "15", "20"]
    rasters_list.append("POI")  # 增加POI
    # 按年份获得LU+factor
    for i in range(3):
        year_folder = f"{folder}/valid_{years[i]}/_maxent"
        ensure_folder(year_folder)
        for f in rasters_list:
            if f == "PODU":
                factor_name = f
            else:
                factor_name = f"{f}{years[i]}"
            factor_filepath = [f for f in os.listdir(root) if f.startswith(factor_name)]
            print(f"getting {factor_name} from src:{root}/{factor_name}, {len(factor_filepath)} files in total.")
            if len(factor_filepath) == 0:
                print(f"No {factor_name} files found in {root}.")
                continue
            
            for idx, filepath in enumerate(factor_filepath):
                shutil.copy(os.path.join(root, filepath), year_folder)
            print(f"finish copy {factor_name} from src:{root}/{factor_name}, new files: {year_folder}.")

def create_SA_dataset(root: str, folder: str, shape: List[int], factor_list: Optional[List[str]] = None, ES_list: Optional[List[str]] = None, need_split: bool = True) -> None:
    """
    root: str, 源数据文件夹，包含所有因子和ES数据
    folder: str, 目标数据文件夹，用于存储处理后的数据集
    factor list: ['DEM','DL','NLMY','GDP','RK','RA','LU']
    ES list: ['RAIN','NJW','ET','PODU','WATER','CITY','GD']
            年均降水、年均温度、年均蒸散量、坡度、距水源距离、距城镇距离、距耕地距离
    """
    years = ALL_YEARS
    factor_list = factor_list if factor_list is not None else []
    ES_list = ES_list if ES_list is not None else []
    all_list = list(set(factor_list + ES_list))
    if(len(all_list) == 0):
        print("No factors or ES to create dataset.")
        return
    
    # 按年份获得LU+factor
    for i in range(3):
        year_folder = os.path.join(folder, f"valid_{years[i]}")
        for f in all_list:
            invalid_num = 0
            valid_root = os.path.join(root, f"LU{years[i]}")  # 用人均生产总值判断该idx是否有效
            if f == "DEM" or f == "RA" or f == "PODU":
                factor_root = os.path.join(root, f)
            else:
                factor_root = os.path.join(root, f"{f}{years[i]}")
            valid_filepath = [
                os.path.join(valid_root, f)
                for f in os.listdir(valid_root)
                if f.lower().endswith((".tif", ".tiff"))
            ]
            factor_filepath = [
                os.path.join(factor_root, f)
                for f in os.listdir(factor_root)
                if f.lower().endswith((".tif", ".tiff"))
            ]
            factor_file_nums = len(factor_filepath)
            print(
                f"getting {years[i]}'s {f} from src:{factor_root}, {factor_file_nums} files in total."
            )
            factor_folder = os.path.join(year_folder, f)
            ensure_folder(factor_folder)
            for idx, filepath in enumerate(factor_filepath):
                try:
                    valid_array = load_tiff(valid_filepath[idx])
                    if np.all(valid_array == 0):
                        invalid_num += 1
                        print(f"idx:{idx} is invalid, skip it.")
                        continue
                    image_array = load_tiff(filepath)
                    if image_array.shape == (shape[1], shape[2]):
                        shutil.copy(filepath, factor_folder)
                except Exception as e:
                    print(f"读取 {valid_filepath[idx]} 失败: {e}")
                    continue
            print(
                f"finish copy {years[i]}'s {f} from src:{factor_root}, new files: {factor_folder}."
            )
            print(
                f"valid_num:{factor_file_nums-invalid_num}, invalid_num:{invalid_num}, total_num:{factor_file_nums}."
            )
    if need_split:
        split_SA_dataset(folder)


def split_SA_dataset(folder: str, factor_list: Optional[List[str]] = None,split_ratio: float = 0.8, save_split_indices: bool = True) -> None:
    """
    将10 15 20三个年份的数据按照8:2的比例划分为训练集，验证集。
        folder: '_data'
        factor_list: 要分割的因子列表，默认所有因子
        split_ratio: 0.8
        save_split_indices: 是否保存分割索引，便于后续添加数据
    """

    def tmp_sort_key(file_idx):
        return int(file_idx)

    full_idx = sorted(
        [
            f.split("_")[-1].split(".")[0]
            for f in os.listdir(os.path.join(folder, "valid_10", "LU"))
            if f.lower().endswith((".tif", ".tiff"))
        ],
        key=tmp_sort_key,
    )
    np.random.shuffle(full_idx)
    train_idx = full_idx[: int(len(full_idx) * split_ratio)]
    valid_idx = full_idx[int(len(full_idx) * split_ratio) :]

    # 保存分割索引到文件
    if save_split_indices:
        split_info = {
            "train_indices": train_idx,
            "valid_indices": valid_idx,
            "split_ratio": split_ratio,
            "total_samples": len(full_idx)
        }
        split_indices_file = os.path.join(folder, "split_indices.json")
        with open(split_indices_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)
        print(f"分割索引已保存到: {split_indices_file}")

    years = ALL_YEARS
    factor_list = factor_list if factor_list is not None else ALL_FACTORS
    factor_list.append("LU")
    for i in range(3):
        dataset_year = os.path.join(folder, "train", f"valid_{years[i]}")
        for f in factor_list:
            dataset_factor = os.path.join(dataset_year, f)
            ensure_folder(dataset_factor)
            src_factor = os.path.join(folder, f"valid_{years[i]}", f)
            src_factor_path = [
                os.path.join(src_factor, f)
                for f in os.listdir(src_factor)
                if f.endswith(".TIF")
            ]
            for path in src_factor_path:
                idx = path.split(os.path.sep)[-1].split("_")[-1].split(".")[0]
                if idx in train_idx:
                    dataset_path = dataset_factor
                else:
                    dataset_path = dataset_factor.replace("train", "val")
                ensure_folder(dataset_path)
                shutil.copy(path, dataset_path)
            assert len(os.listdir(os.path.join(folder, "train", f"valid_{years[i]}", f))) == len(
                train_idx
            )
            assert len(os.listdir(os.path.join(folder, "val", f"valid_{years[i]}", f))) == len(
                valid_idx
            )
            print(
                f"finish split {years[i]}'s {f} dataset, train_num:{len(train_idx)}, valid_num:{len(valid_idx)}."
            )

