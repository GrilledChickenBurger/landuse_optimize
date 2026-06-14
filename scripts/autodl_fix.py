from typing import List
import os
import json
import numpy as np
import torch
import shutil
from utils.dataset_utils import load_tiff
from utils.performance import only_landscape, calculate_landuse_area_statistics
from libs.const import RAW_DATA_ROOT

def get_sample_list(folder_path: str) -> List[str]:
    """
    获取文件夹下所有样本的路径列表
    
    Parameters:
    ------------
        folder_path: str, 样本文件夹总路径, 样本保存的路径格式为 {timestamp}\\{scenario}\\{year_idx_5}

    Returns:
    --------
        sample_list: list, 所有样本的路径列表, 格式为
        [
            {timestamp}\\{scenario}\\{year_idx_5}\\plans_{idx},
            ...
        ]
    """
    sample_list = []
    for scenario in os.listdir(folder_path):
        for sample in os.listdir(os.path.join(folder_path, scenario)):
            year, idx, _ = sample.split('_')
            sample_list.append(os.path.join(folder_path, scenario, sample, f'plans_{idx}'))
    return sample_list
    
def copy_tfw_from_root(index: str, save_folder: str, 
                        raw_data_root: str = RAW_DATA_ROOT):
    """
    对每个样本文件夹，从根文件夹复制TFW文件到样本文件夹的plans_{idx}文件夹
    
    Parameters:
    ------------
        index: str, 样本文件夹索引, 格式为 {year_idx_5}
        save_folder: str, 样本文件夹路径, 格式为
        {timestamp}\\{scenario}\\{year_idx_5}\\plans_{idx},
        raw_data_root: str, 根文件夹路径, 栅含TFW文件, 默认值为RAW_DATA_ROOT
    """
        # 一个栅格完整的TIFF文件有6种, 使用fragstats分析TIF需要.tfw
    # 需要TFW文件的名称和TIF的名称完全对应，需要复制3份并重命名。
    year = index.split("_")[0]
    idx = index.split("_")[1]
    if raw_data_root is not None:
        other_tiff_folder = os.path.join(raw_data_root, f"LU{year}")
        other_tiff_path = [os.path.join(other_tiff_folder, f) for f in os.listdir(other_tiff_folder) if f.startswith(f"LU{year}_{idx}.") and f.endswith(".tfw")]
        assert len(other_tiff_path)==1
        for path in other_tiff_path:
            shutil.copy(path, os.path.join(save_folder, 'best_plan.tfw'))
            shutil.copy(path, os.path.join(save_folder, 'ori_plan.tfw'))
            shutil.copy(path, os.path.join(save_folder, 'tgt_plan.tfw'))
            print(f"已复制 {path} 到 {save_folder}")

def create_and_cal_metrics(folder_path: str, frg_model_path: str):
    """
    从folder_path中获取所有样本的路径列表，完成tfw文件的复制，计算每个样本的指标并更新performance_score.json文件
    
    Parameters:
    ------------
        folder_path: str, 样本文件夹总路径, 样本保存的路径格式为 {timestamp}\\{scenario}\\{year_idx_5}
        frg_model_path: str, FRG模型路径, 栅含模型参数
    """
    sample_list = get_sample_list(folder_path)
    for sample in sample_list:
        print(f"\n当前样本路径: {sample}")
        # sample: {timestamp}\\{scenario}\\{year_idx_5}\\plans_{idx}
        index = sample.split("\\")[-2]  # 格式为 {year_idx_5}
        copy_tfw_from_root(index, sample)

        # save_folder: {timestamp}\\{scenario}\\{year_idx_5}
        save_folder = os.path.dirname(sample)
        idx = index.split("_")[1]
        # tmp文件夹, 格式为 {timestamp}\\{scenario}\\{year_idx_5}\\performance_{idx}
        tmp_folder = os.path.join(save_folder, f"performance_{idx}")
        os.makedirs(tmp_folder, exist_ok=True)
        metrics = only_landscape(src_folder=sample, tmp_folder=tmp_folder,
                                frg_model_path=frg_model_path)
        
        # 读取现有的performance_score.json文件并更新Landscape.land部分
        # performance_score_path: {timestamp}\\{scenario}\\{year_idx_5}\\performance_score.json
        performance_score_path = os.path.join(save_folder, "performance_score.json")
        
        # 检查文件是否存在
        if os.path.exists(performance_score_path):
            with open(performance_score_path, 'r', encoding='utf-8') as f:
                performance_data = json.load(f)
            print(f"已读取 {performance_score_path} 文件")
        else:
            # 如果文件不存在，创建基本结构
            print(f"文件 {performance_score_path} 不存在，创建基本结构")
            performance_data = {
                "Change_area": {},
                "ES": {},
                "Landscape": {"land": {}},
                "FINAL": {}
            }
        
        # 更新Landscape.land部分
        performance_data["Landscape"]["land"] = metrics
        
        # 保存更新后的数据到原文件
        with open(performance_score_path, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, indent=4, ensure_ascii=False)
            print(f"已更新 {sample} 的 landscape 指标到 {performance_score_path}.\n")

def create_and_cal_lu_change(folder_path: str):
    """
    从folder_path中获取所有样本的路径列表，计算每个样本的landuse面积变化并更新performance_score.json文件
    
    Parameters:
    ------------
        folder_path: str, 样本文件夹总路径, 样本保存的路径格式为 {timestamp}\\{scenario}\\{year_idx_5}\\plans_{idx}
    """
    sample_list = get_sample_list(folder_path)

    for sample in sample_list:
        # sample: {timestamp}\\{scenario}\\{year_idx_5}\\plans_{idx}
        ori_plan = os.path.join(sample, 'ori_plan.TIF')
        best_plan = os.path.join(sample, 'best_plan.TIF')
        tgt_plan = os.path.join(sample, 'tgt_plan.TIF')
        print(f"\n当前样本路径: {sample}")

        try:
            ori_plan_tensor = load_tiff(ori_plan)
            best_plan_tensor = load_tiff(best_plan)
            tgt_plan_tensor = load_tiff(tgt_plan)
            print(f"ori_plan 路径: {ori_plan}")
            print(f"best_plan 路径: {best_plan}")
            print(f"tgt_plan 路径: {tgt_plan}")
            ori_best = calculate_landuse_area_statistics(ori_plan_tensor, 'ori_plan',best_plan_tensor, 'best_plan')
            tgt_best = calculate_landuse_area_statistics(tgt_plan_tensor, 'tgt_plan',best_plan_tensor, 'best_plan')
        except Exception as e:
            print(f"读取 {sample} 失败，跳过: {e}")
            continue
        
        save_folder = os.path.dirname(sample)
        performance_score_path = os.path.join(save_folder, "performance_score.json")
        # 读取现有的performance_score.json文件并更新FINAL.best_ori和FINAL.best_tgt部分
        # performance_score_path: {timestamp}\\{scenario}\\{year_idx_5}\\performance_score.json
        if os.path.exists(performance_score_path):
            with open(performance_score_path, 'r', encoding='utf-8') as f:
                performance_data = json.load(f)
            print(f"已读取 {performance_score_path} 文件")
        else:
            print(f"文件 {performance_score_path} 不存在，创建基本结构")
            performance_data = {
                "Change_area": {},
                "ES": {},
                "Landscape": {"land": {}},
                "FINAL": {}
            }
        # 更新FINAL.best_ori和FINAL.best_tgt部分
        performance_data["FINAL"]["best_ori"] = ori_best
        performance_data["FINAL"]["best_tgt"] = tgt_best
        # 保存更新后的数据到原文件
        with open(performance_score_path, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, indent=4, ensure_ascii=False)
            print(f"已更新 {sample} 的 landuse 变化指标到 {performance_score_path}.\n")



if __name__ == "__main__":
    # folder_path = f"results\\SA\\20260215_042116"
    folder_path = f"results\\SA\\20260301_101241"
    frg_model_path = "libs\\frg_model.fca"
    create_and_cal_lu_change(folder_path)
    create_and_cal_metrics(folder_path, frg_model_path)