<<<<<<< HEAD
import os
from typing import Dict, Any, List
import json
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

from calculator.cal_es_score import cal_es_value, get_es_cal_func_dict
from utils.tools import ensure_folder
from utils.dataset_utils import remove_lines

def calculate_FOM(ori, pred, true):
    """
    计算FOM系数

    args:
        original: [1,128,128], 原始区域
        predicted: [1,128,128], 未来预测值
        target: [1,128,128], 未来真实值
    return:
        FOM: float, 预测的FOM值
    """
    original = ori.squeeze(0)
    predicted = pred.squeeze(0)
    target = true.squeeze(0)
    
    # 计算A, B, C, D
    A = ((original != target) & (original == predicted) & (target != predicted)).sum().item()
    B = ((original != target) & (predicted == target)).sum().item()
    C = ((original != target) & (predicted != target) & (predicted != original)).sum().item()
    D = ((original == target) & (predicted != target) & (original != predicted)).sum().item()

    # 计算FOM
    FOM = B / (A + B + C + D) if (A + B + C + D) != 0 else 0
    
    return FOM

def calculate_kappa(pred, true):
    """
    计算kappa系数

    args:
        predicted: [1,128,128], 未来预测值
        target: [1,128,128], 未来真实值
    return:
        kappa: float, 预测的kappa值
    """
    pred_flat = pred.squeeze(0).flatten()
    tgt_flat = true.squeeze(0).flatten()

    conf_matrix = confusion_matrix(tgt_flat, pred_flat)
    n_classes = conf_matrix.shape[0]
    
    # 计算kappa系数
    total_pixels = np.sum(conf_matrix)
    p_o = np.trace(conf_matrix) / total_pixels
    p_e = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total_pixels ** 2)
    kappa = (p_o - p_e) / (1 - p_e)

    return kappa

###################################################################
#####################  performance: 变化区域  #####################
###################################################################

def calculate_OA(pred, true):
    """
    计算OA (Overall Accuracy)、UA (User's Accuracy) 和 PA (Producer's Accuracy)
    
    参数:
        pred: 预测结果的张量，形状为[batch_size, 1, H, W]，只包含0和1
        target: 真实标签的张量，形状为[batch_size, 1, H, W]，只包含0和1
    
    返回:
        miou: 所有样本的IoU平均值，形状为[batch_size]
    """
    predicted = pred.squeeze(0)
    target = true.squeeze(0)

    TP = ((predicted == target) & (target != 0)).sum().item()
    FP = ((predicted != target) & (predicted != 0)).sum().item()
    TN = ((predicted == target) & (target == 0)).sum().item()
    FN = ((predicted != target) & (predicted == 0)).sum().item()
    
    # 计算总体准确率 (OA)
    total_pixels = TP + FP + TN + FN
    OA = (TP + TN) / total_pixels if total_pixels > 0 else 0
    # 计算用户准确率 (UA) 和生产准确率 (PA)
    # 用户准确率（UA）: 对于正类的总体精确度
    UA = TP / (TP + FP) if (TP + FP) > 0 else 0
    # 生产者准确率（PA）: 对于正类的真实识别率
    PA = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return OA, UA, PA

def calculate_accuracy(pred: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    """
    计算准确率，公式为：TP+TN/(TP+TN+FP+FN)\n
    Parameters:
        pred: 预测结果的张量，形状为[batch_size, 1, H, W]，只包含0和1
        target: 真实标签的张量，形状为[batch_size, 1, H, W]，只包含0和1
        epsilon: 用于防止除以零的小数
    Returns:
        accuracy: 所有样本的准确率平均值，形状为[batch_size]
    """
    pred = pred.view(-1)
    target = target.view(-1)

    tp = torch.sum((pred == 1) & (target == 1)).float()
    tn = torch.sum((pred == 0) & (target == 0)).float()

    acc = (tp + tn) / (pred.numel() + epsilon)

    return acc, tp, tn

def calculate_iou_01(prediction, target):
    """
    计算IOU (Intersection over Union)
    
    参数:
        pred: 预测结果的张量，形状为[batch_size, 1, H, W]，只包含0和1
        target: 真实标签的张量，形状为[batch_size, 1, H, W]，只包含0和1
        epsilon: 用于防止除以零的小数
    
    返回:
        miou: 所有样本的IoU平均值，形状为[batch_size]
    """
    epsilon=1e-6
    # 计算交集
    intersection = torch.logical_and(target, prediction)
    # 计算并集
    union = torch.logical_or(target, prediction)
    union = torch.clamp(union, min=epsilon)
    # 计算IoU
    iou_score = torch.sum(intersection.float()) / torch.sum(union.float())
    return iou_score.item()


def calculate_miou(pred, target, epsilon=1e-6):
    """
    计算MIOU (Mean Intersection over Union)
    
    参数:
        pred: 预测结果的张量，形状为[batch_size, 1, H, W]，只包含0和1
        target: 真实标签的张量，形状为[batch_size, 1, H, W]，只包含0和1
        epsilon: 用于防止除以零的小数
    
    返回:
        miou: 所有样本的IoU平均值，形状为[batch_size]
    """
    # 计算交集
    intersection = (pred & target).float().sum((1, 2, 3))
    
    # 计算并集
    union = (pred | target).float().sum((1, 2, 3))
    
    # 计算IoU
    # iou = intersection / (union + epsilon)

    iou_per_sample = torch.zeros_like(intersection)
    # 对于非全0张量，计算IoU
    non_zero_union_mask = union > 0
    iou_per_sample[non_zero_union_mask] = intersection[non_zero_union_mask] / (union[non_zero_union_mask] + epsilon)
    
    # 对于全0张量，设置IoU为1
    zero_union_mask = ~non_zero_union_mask
    iou_per_sample[zero_union_mask] = 1.0

    
    return iou_per_sample

###################################################################
###################  performance: ES  ###################
###################################################################

def calculate_es_score(ori_plan: torch.Tensor,
                       best_plan: torch.Tensor,
                       tgt_plan: torch.Tensor,
                       info_dict: dict) -> dict:
    """
    计算ES分数
    Parameters:
        ori_plan: tensor, [1,1,128,128], 原始方案
        best_plan: tensor, [1,1,128,128], 最佳方案
        tgt_plan: tensor, [1,1,128,128], 未来真实值
        info_dict: dict, 包含LU类别的信息
    Returns:
        es_score: dict, 包含每个LU类别的ES分数
    """
    best_es_dict = cal_es_value(best_plan, info_dict, get_es_cal_func_dict())
    ori_es_dict = cal_es_value(ori_plan, info_dict, get_es_cal_func_dict())
    tgt_es_dict = cal_es_value(tgt_plan, info_dict, get_es_cal_func_dict())

    es_score_dict = {}
    for type in best_es_dict.keys():
        es_score_dict[type] = {
            "best_plan": float(best_es_dict[type]),
            "tgt_plan": float(tgt_es_dict[type]),
            "ori_plan": float(ori_es_dict[type]),
            "best_ori_diff": (float(best_es_dict[type]) - float(ori_es_dict[type]))/float(ori_es_dict[type]),
            "best_tgt_diff": (float(best_es_dict[type]) - float(tgt_es_dict[type]))/float(tgt_es_dict[type]),
        }
    return es_score_dict


###################################################################
###################  performance: 景观格局函数  ###################
###################################################################

def create_frg_csv(src_folder, save_path):
    """创建一个批处理文件, 能够批量导入TIF调用fragstats。
    对单个index, 有3个文件需要调用fragstats: ori, best, tgt
    """
    print("Creating batch_csv for Fragstats...")
    index = src_folder.split("_")[-1]
    tif_paths = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith((".TIF",".tif"))]

    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        # 数据  
        for p in tif_paths:
            # 文件包括8个字段
            data_row:List=[p,'x',999,'x','x',1,'x','IDF_GeoTIFF']
            csv_writer.writerow(data_row)
    print(f"Done. Result is saved in {save_path}, TIF files num: {len(tif_paths)}.")

def frg_cmd(model_path, batchcsv_path, output_folder):
    """用命令行直接调用 fragstats软件, 分析batchcsv_path里的所有文件"""
    output_path = f"{output_folder}\\scores"
    frg_flag = 0
    while frg_flag != 1:
        frg_path=os.getenv('frg_cmd')
        command = f'"{frg_path}\\frg_cmd" -m {model_path} -b {batchcsv_path} -o {output_path}'
        os.system(command) # 命令行执行
        # subprocess.call(command, shell=True)

        # 检查输入输出条目数是否一致
        with open(batchcsv_path, 'r', newline='', encoding='utf-8') as batch_file:  
            csv_reader_input = csv.reader(batch_file)  
            batch_lines = sum(1 for line in csv_reader_input)  # 计算行数      

        with open(output_path + ".land", 'r', newline='', encoding='utf-8') as output_file:  
            csv_reader_output = csv.reader(output_file)  
            output_lines = sum(1 for line in csv_reader_output)-1  # 计算行数，-1是去掉第一行字段

        if batch_lines == output_lines: # 如果输入输出条目数匹配
            frg_flag = 1
            print("data match!")
        else:
            print("data do not match! Restart!")

    print(f"Done. {batchcsv_path} is completed.")

def remove_csv_space(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  
        csv_reader = csv.reader(file)  
        rows = list(csv_reader)  # 读取所有行  
  
    with open(file_path, 'w', newline='', encoding='utf-8') as file:  
        csv_writer = csv.writer(file)  
        for row in rows:  
            cleaned_row = [cell.strip() for cell in row]  # 去掉每个单元格的空格  
            csv_writer.writerow(cleaned_row)  

def get_cls_landscape_index(frg_output_folder, index_list):
    cls_file_path = [os.path.join(frg_output_folder, f) for f in os.listdir(frg_output_folder) if f.endswith(".class")]
    assert len(cls_file_path)==1
    # 由于fragstats的后缀默认是.class, 因此需要重命名
    old_name = cls_file_path[0].split("\\")[-1]
    new_name = "_".join(old_name.split(".")) + ".csv"
    cls_new_path = f"{frg_output_folder}\\{new_name}"
    if not os.path.exists(cls_new_path):
        shutil.copy(cls_file_path[0], cls_new_path)
        remove_csv_space(cls_new_path)
    
    cls_results = {}
    with open(cls_new_path, mode='r', encoding='utf-8', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # 使用 LID 作为字典的键
            plan_name = row['LID'].split("\\")[-1].split('.')[0]
            if plan_name not in cls_results:
                cls_results[plan_name]={}
            for idx in index_list:
                if idx not in cls_results[plan_name]:
                    cls_results[plan_name][idx]={}
                cls_name = row['TYPE']                
                cls_results[plan_name][idx].update({cls_name: row[idx]})
    return cls_results

def get_land_landscape_index(result_folder, index_list):
    cls_file_path = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith(".land")]
    assert len(cls_file_path)==1
    # 由于fragstats的后缀默认是.class, 因此需要重命名
    old_name = cls_file_path[0].split("\\")[-1]
    new_name = "_".join(old_name.split(".")) + ".csv"
    cls_new_path = f"{result_folder}\\{new_name}"
    if not os.path.exists(cls_new_path):
        shutil.copy(cls_file_path[0], cls_new_path)
        remove_csv_space(cls_new_path)
    
    land_results = {}
    with open(cls_new_path, mode='r', encoding='utf-8', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # 使用 LID 作为字典的键
            plan_name = row['LID'].split("\\")[-1].split('.')[0]
            if plan_name not in land_results:
                land_results[plan_name]={}
            for idx in index_list:              
                land_results[plan_name].update({idx: row[idx]})
    return land_results


def write_performance_score(best_plan,ori_plan,tgt_plan, 
                            pred_diff, 
                            info_dict,
                            src_folder, frg_model_path,
                            output_folder):
    """
    评估方案, 评估项: 变化区域IOU、ES、Landscape\n
    Parameters:
    ----------
        best_plan: torch.tensor, 最优方案
        ori_plan: torch.tensor, 原始方案
        tgt_plan: torch.tensor, 目标方案
        pred_diff: torch.tensor, 1.ATTUNET模型预测变化区域
        info_dict: dict, 原始方案的信息字典
        src_folder: str, 原始方案的文件夹
        frg_model_path: str, fragstats模型路径
        output_folder: str, 输出文件夹
    """
    print("================= 评估方案中，评估项：变化区域IOU、ES、Landscape =================")
    ensure_folder(output_folder)
    performance_score = {'Change_area':{},
                         'ES':{},
                         'Landscape':{},
                         'FINAL':{},
                         }
    # 1: 和change_area的IOU
    new_diff = torch.where(best_plan != ori_plan, torch.tensor(1), torch.tensor(0))
    real_diff = torch.where(tgt_plan != ori_plan, torch.tensor(1), torch.tensor(0))
    real_diff = torch.from_numpy(remove_lines(real_diff))
    # 生成方案的变化区域和 1.ATTUNET模型预测变化区域 2.真实变化区域 的IOU
    performance_score['Change_area']['pred_IOU_score']=calculate_iou_01(new_diff, pred_diff)
    performance_score['Change_area']['real_IOU_score']=calculate_iou_01(new_diff, real_diff)
    OA, UA, PA = calculate_OA(new_diff, real_diff)
    performance_score['Change_area']['OA'] = OA
    performance_score['Change_area']['UA'] = UA
    performance_score['Change_area']['PA'] = PA

    # 2: es
    es_value_dict = calculate_es_score(ori_plan, best_plan, tgt_plan, info_dict)
    performance_score['ES'] = es_value_dict
    
    # 3. landscape
    cls_list = ['LPI','LSI','AI','ED']
    land_list = ['LPI','LSI','AI','ED','SHEI','SHDI','CONTAG']
    create_frg_csv(src_folder, f'{output_folder}\\batch.csv')
    frg_cmd(frg_model_path, f'{output_folder}\\batch.csv', output_folder)
    # cls_dict = get_cls_landscape_index(output_folder, cls_list)
    land_dict = get_land_landscape_index(output_folder, land_list)
    # performance_score['Landscape']['class'] = cls_dict
    performance_score['Landscape']['land'] = land_dict
    
    # FINAL: FOM, kappa
    performance_score['FINAL']['FOM'] = calculate_FOM(ori_plan, best_plan, tgt_plan)
    performance_score['FINAL']['kappa'] = calculate_kappa(best_plan, tgt_plan)

    output_file = f'{output_folder}\\performance_score.json'
    with open(output_file,'w') as file:
        json.dump(performance_score, file, indent=4)  


    print(f"Done. The performance scores has been saved to {output_folder}.\n")
    return performance_score



if __name__=="__main__":
    pass
=======
import os
from typing import Dict, Any, List
import json
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

from calculator.cal_es_score import cal_es_value, get_es_cal_func_dict
from utils.tools import ensure_folder
from utils.dataset_utils import remove_lines

def calculate_FOM(ori, pred, true):
    """
    计算FOM系数

    args:
        original: [1,128,128], 原始区域
        predicted: [1,128,128], 未来预测值
        target: [1,128,128], 未来真实值
    return:
        FOM: float, 预测的FOM值
    """
    original = ori.squeeze(0)
    predicted = pred.squeeze(0)
    target = true.squeeze(0)
    
    # 计算A, B, C, D
    A = ((original != target) & (original == predicted) & (target != predicted)).sum().item()
    B = ((original != target) & (predicted == target)).sum().item()
    C = ((original != target) & (predicted != target) & (predicted != original)).sum().item()
    D = ((original == target) & (predicted != target) & (original != predicted)).sum().item()

    # 计算FOM
    FOM = B / (A + B + C + D) if (A + B + C + D) != 0 else 0
    
    return FOM

def calculate_kappa(pred, true):
    """
    计算kappa系数

    args:
        predicted: [1,128,128], 未来预测值
        target: [1,128,128], 未来真实值
    return:
        kappa: float, 预测的kappa值
    """
    pred_flat = pred.squeeze(0).flatten()
    tgt_flat = true.squeeze(0).flatten()

    conf_matrix = confusion_matrix(tgt_flat, pred_flat)
    n_classes = conf_matrix.shape[0]
    
    # 计算kappa系数
    total_pixels = np.sum(conf_matrix)
    p_o = np.trace(conf_matrix) / total_pixels
    p_e = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total_pixels ** 2)
    kappa = (p_o - p_e) / (1 - p_e)

    return kappa

###################################################################
#####################  performance: 变化区域  #####################
###################################################################

def calculate_OA(pred, true):
    """
    计算OA (Overall Accuracy)、UA (User's Accuracy) 和 PA (Producer's Accuracy)
    
    参数:
        pred: 预测结果的张量，形状为[batch_size, 1, H, W]，只包含0和1
        target: 真实标签的张量，形状为[batch_size, 1, H, W]，只包含0和1
    
    返回:
        miou: 所有样本的IoU平均值，形状为[batch_size]
    """
    predicted = pred.squeeze(0)
    target = true.squeeze(0)

    TP = ((predicted == target) & (target != 0)).sum().item()
    FP = ((predicted != target) & (predicted != 0)).sum().item()
    TN = ((predicted == target) & (target == 0)).sum().item()
    FN = ((predicted != target) & (predicted == 0)).sum().item()
    
    # 计算总体准确率 (OA)
    total_pixels = TP + FP + TN + FN
    OA = (TP + TN) / total_pixels if total_pixels > 0 else 0
    # 计算用户准确率 (UA) 和生产准确率 (PA)
    # 用户准确率（UA）: 对于正类的总体精确度
    UA = TP / (TP + FP) if (TP + FP) > 0 else 0
    # 生产者准确率（PA）: 对于正类的真实识别率
    PA = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return OA, UA, PA

def calculate_accuracy(pred: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    """
    计算准确率，公式为：TP+TN/(TP+TN+FP+FN)\n
    Parameters:
        pred: 预测结果的张量，形状为[batch_size, 1, H, W]，只包含0和1
        target: 真实标签的张量，形状为[batch_size, 1, H, W]，只包含0和1
        epsilon: 用于防止除以零的小数
    Returns:
        accuracy: 所有样本的准确率平均值，形状为[batch_size]
    """
    pred = pred.view(-1)
    target = target.view(-1)

    tp = torch.sum((pred == 1) & (target == 1)).float()
    tn = torch.sum((pred == 0) & (target == 0)).float()

    acc = (tp + tn) / (pred.numel() + epsilon)

    return acc, tp, tn

def calculate_iou_01(prediction, target):
    """
    计算IOU (Intersection over Union)
    
    参数:
        pred: 预测结果的张量，形状为[batch_size, 1, H, W]，只包含0和1
        target: 真实标签的张量，形状为[batch_size, 1, H, W]，只包含0和1
        epsilon: 用于防止除以零的小数
    
    返回:
        miou: 所有样本的IoU平均值，形状为[batch_size]
    """
    epsilon=1e-6
    # 计算交集
    intersection = torch.logical_and(target, prediction)
    # 计算并集
    union = torch.logical_or(target, prediction)
    union = torch.clamp(union, min=epsilon)
    # 计算IoU
    iou_score = torch.sum(intersection.float()) / torch.sum(union.float())
    return iou_score.item()


def calculate_miou(pred, target, epsilon=1e-6):
    """
    计算MIOU (Mean Intersection over Union)
    
    参数:
        pred: 预测结果的张量，形状为[batch_size, 1, H, W]，只包含0和1
        target: 真实标签的张量，形状为[batch_size, 1, H, W]，只包含0和1
        epsilon: 用于防止除以零的小数
    
    返回:
        miou: 所有样本的IoU平均值，形状为[batch_size]
    """
    # 计算交集
    intersection = (pred & target).float().sum((1, 2, 3))
    
    # 计算并集
    union = (pred | target).float().sum((1, 2, 3))
    
    # 计算IoU
    # iou = intersection / (union + epsilon)

    iou_per_sample = torch.zeros_like(intersection)
    # 对于非全0张量，计算IoU
    non_zero_union_mask = union > 0
    iou_per_sample[non_zero_union_mask] = intersection[non_zero_union_mask] / (union[non_zero_union_mask] + epsilon)
    
    # 对于全0张量，设置IoU为1
    zero_union_mask = ~non_zero_union_mask
    iou_per_sample[zero_union_mask] = 1.0

    
    return iou_per_sample

###################################################################
###################  performance: ES  ###################
###################################################################

def calculate_es_score(ori_plan: torch.Tensor,
                       best_plan: torch.Tensor,
                       tgt_plan: torch.Tensor,
                       info_dict: dict) -> dict:
    """
    计算ES分数
    Parameters:
        ori_plan: tensor, [1,1,128,128], 原始方案
        best_plan: tensor, [1,1,128,128], 最佳方案
        tgt_plan: tensor, [1,1,128,128], 未来真实值
        info_dict: dict, 包含LU类别的信息
    Returns:
        es_score: dict, 包含每个LU类别的ES分数
    """
    best_es_dict = cal_es_value(best_plan, info_dict, get_es_cal_func_dict())
    ori_es_dict = cal_es_value(ori_plan, info_dict, get_es_cal_func_dict())
    tgt_es_dict = cal_es_value(tgt_plan, info_dict, get_es_cal_func_dict())

    es_score_dict = {}
    for type in best_es_dict.keys():
        ori_value = float(ori_es_dict[type])
        tgt_value = float(tgt_es_dict[type])
        
        # 处理除零情况
        best_ori_diff = (float(best_es_dict[type]) - ori_value) / ori_value if ori_value != 0 else 0.0
        best_tgt_diff = (float(best_es_dict[type]) - tgt_value) / tgt_value if tgt_value != 0 else 0.0
        
        es_score_dict[type] = {
            "best_plan": float(best_es_dict[type]),
            "tgt_plan": tgt_value,
            "ori_plan": ori_value,
            "best_ori_diff": best_ori_diff,
            "best_tgt_diff": best_tgt_diff,
        }
    return es_score_dict


###################################################################
###################  performance: 景观格局函数  ###################
###################################################################

def create_frg_csv(src_folder, save_path):
    """创建一个批处理文件, 能够批量导入TIF调用fragstats。
    对单个index, 有3个文件需要调用fragstats: ori, best, tgt
    """
    print("Creating batch_csv for Fragstats...")
    index = src_folder.split("_")[-1]
    tif_paths = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if f.endswith((".TIF",".tif"))]

    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        # 数据  
        for p in tif_paths:
            # 文件包括8个字段
            data_row:List=[p,'x',999,'x','x',1,'x','IDF_GeoTIFF']
            csv_writer.writerow(data_row)
    print(f"Done. Result is saved in {save_path}, TIF files num: {len(tif_paths)}.")

def frg_cmd(model_path, batchcsv_path, output_folder):
    """用命令行直接调用 fragstats软件, 分析batchcsv_path里的所有文件"""
    output_path = os.path.join(output_folder, "scores")
    frg_flag = 0
    while frg_flag != 1:
        frg_path=os.getenv('frg_cmd')
        command = f'"{frg_path}\\frg_cmd" -m {model_path} -b {batchcsv_path} -o {output_path}'
        os.system(command) # 命令行执行
        # subprocess.call(command, shell=True)

        # 检查输入输出条目数是否一致
        with open(batchcsv_path, 'r', newline='', encoding='utf-8') as batch_file:  
            csv_reader_input = csv.reader(batch_file)  
            batch_lines = sum(1 for line in csv_reader_input)  # 计算行数      

        with open(output_path + ".land", 'r', newline='', encoding='utf-8') as output_file:  
            csv_reader_output = csv.reader(output_file)  
            output_lines = sum(1 for line in csv_reader_output)-1  # 计算行数，-1是去掉第一行字段

        if batch_lines == output_lines: # 如果输入输出条目数匹配
            frg_flag = 1
            print("data match!")
        else:
            print("data do not match! Restart!")

    print(f"Done. {batchcsv_path} is completed.")

def remove_csv_space(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  
        csv_reader = csv.reader(file)  
        rows = list(csv_reader)  # 读取所有行  
  
    with open(file_path, 'w', newline='', encoding='utf-8') as file:  
        csv_writer = csv.writer(file)  
        for row in rows:  
            cleaned_row = [cell.strip() for cell in row]  # 去掉每个单元格的空格  
            csv_writer.writerow(cleaned_row)  

def get_cls_landscape_index(frg_output_folder, index_list):
    cls_file_path = [os.path.join(frg_output_folder, f) for f in os.listdir(frg_output_folder) if f.endswith(".class")]
    assert len(cls_file_path)==1
    # 由于fragstats的后缀默认是.class, 因此需要重命名
    old_name = cls_file_path[0].split("\\")[-1]
    new_name = "_".join(old_name.split(".")) + ".csv"
    cls_new_path = f"{frg_output_folder}\\{new_name}"
    if not os.path.exists(cls_new_path):
        shutil.copy(cls_file_path[0], cls_new_path)
        remove_csv_space(cls_new_path)
    
    cls_results = {}
    with open(cls_new_path, mode='r', encoding='utf-8', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # 使用 LID 作为字典的键
            plan_name = row['LID'].split("\\")[-1].split('.')[0]
            if plan_name not in cls_results:
                cls_results[plan_name]={}
            for idx in index_list:
                if idx not in cls_results[plan_name]:
                    cls_results[plan_name][idx]={}
                cls_name = row['TYPE']                
                cls_results[plan_name][idx].update({cls_name: row[idx]})
    return cls_results

def get_land_landscape_index(result_folder, index_list):
    cls_file_path = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith(".land")]
    assert len(cls_file_path)==1
    # 由于fragstats的后缀默认是.class, 因此需要重命名
    old_name = cls_file_path[0].split("\\")[-1]
    new_name = "_".join(old_name.split(".")) + ".csv"
    cls_new_path = f"{result_folder}\\{new_name}"
    if not os.path.exists(cls_new_path):
        shutil.copy(cls_file_path[0], cls_new_path)
        remove_csv_space(cls_new_path)
    
    land_results = {}
    with open(cls_new_path, mode='r', encoding='utf-8', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # 使用 LID 作为字典的键
            plan_name = row['LID'].split("\\")[-1].split('.')[0]
            if plan_name not in land_results:
                land_results[plan_name]={}
            for idx in index_list:              
                land_results[plan_name].update({idx: row[idx]})
    return land_results


def write_performance_score(best_plan: torch.Tensor,
                            ori_plan: torch.Tensor,
                            tgt_plan: torch.Tensor, 
                            pred_diff: torch.Tensor, 
                            info_dict: dict,
                            src_folder: str, 
                            output_folder: str,
                            landscape: bool = True,
                            frg_model_path: str = None,):
    """
    评估方案, 评估项: 变化区域IOU、ES、Landscape\n
    Parameters:
    ----------
        best_plan: torch.tensor, 最优方案
        ori_plan: torch.tensor, 原始方案
        tgt_plan: torch.tensor, 目标方案
        pred_diff: torch.tensor, 1.ATTUNET模型预测变化区域
        info_dict: dict, 原始方案的信息字典
        src_folder: str, 原始方案的文件夹
        frg_model_path: str, fragstats模型路径
        output_folder: str, 输出文件夹
    """
    print("================= 评估方案中，评估项：变化区域IOU、ES、Landscape =================")
    ensure_folder(output_folder)
    performance_score = {'Change_area':{},
                         'ES':{},
                         'Landscape':{},
                         'FINAL':{},
                         }
    # 1: 和change_area的IOU
    new_diff = torch.where(best_plan != ori_plan, torch.tensor(1), torch.tensor(0))
    real_diff = torch.where(tgt_plan != ori_plan, torch.tensor(1), torch.tensor(0))
    real_diff = torch.from_numpy(remove_lines(real_diff))
    # 生成方案的变化区域和 1.ATTUNET模型预测变化区域 2.真实变化区域 的IOU
    performance_score['Change_area']['pred_IOU_score']=calculate_iou_01(new_diff, pred_diff)
    performance_score['Change_area']['real_IOU_score']=calculate_iou_01(new_diff, real_diff)
    OA, UA, PA = calculate_OA(new_diff, real_diff)
    performance_score['Change_area']['OA'] = OA
    performance_score['Change_area']['UA'] = UA
    performance_score['Change_area']['PA'] = PA

    # 2: es
    es_value_dict = calculate_es_score(ori_plan, best_plan, tgt_plan, info_dict)
    performance_score['ES'] = es_value_dict
    
    # 3. landscape
    land_dict = {}
    if landscape:
        cls_list = ['LPI','LSI','AI','ED']
        land_list = ['LPI','LSI','AI','ED','SHEI','SHDI','CONTAG']
        create_frg_csv(src_folder, f'{output_folder}\\batch.csv')
        frg_cmd(frg_model_path, f'{output_folder}\\batch.csv', output_folder)
        # cls_dict = get_cls_landscape_index(output_folder, cls_list)
        land_dict = get_land_landscape_index(output_folder, land_list)
    # performance_score['Landscape']['class'] = cls_dict
    performance_score['Landscape']['land'] = land_dict
    
    # FINAL: FOM, kappa
    performance_score['FINAL']['FOM'] = calculate_FOM(ori_plan, best_plan, tgt_plan)
    performance_score['FINAL']['kappa'] = calculate_kappa(best_plan, tgt_plan)

    output_file = os.path.join(output_folder, "performance_score.json")
    with open(output_file,'w') as file:
        json.dump(performance_score, file, indent=4)  


    print(f"Done. The performance scores has been saved to {output_folder}.\n")
    return performance_score



if __name__=="__main__":
    pass
>>>>>>> feat: 服务器SA运行代码备份
