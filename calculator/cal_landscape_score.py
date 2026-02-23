import scipy.ndimage
import torch
import numpy as np

from utils.calculator_utils import get_numfeature_dict, get_perimeter

def cal_landscape_score(cur_plan: torch.Tensor, cur_diff: torch.Tensor, 
                            ori_plan: torch.Tensor,
                            info_dict: dict,
                            change_ratio: float = 0.6) -> torch.Tensor:
    """
    第3?项: 衡量当前更改区域里的所有连通区域和ori_plan对应同类区域之间的距离.\n
    Parameters:
        plan: tensor, [1,1,128,128]
        change_area: tensor, [1,1,128,128]
    """
    # 当前和原始方案里，每个LU类的连通区域数
    tmp_num_feature_list = get_numfeature_dict(cur_plan)
    ori_num_feature_list = info_dict["ori_num_feature_list"]

    cur_plan = cur_plan.squeeze().numpy()
    cur_diff = cur_diff.squeeze().numpy()
    ori_plan = ori_plan.squeeze().numpy()
    lu_types = np.unique(ori_plan[ori_plan!=0.0]).tolist()

    tmp_plan=np.copy(cur_plan)
    tmp_plan[tmp_plan ==0.0] = 100.0     # 把里面的0以100代替，方便后续检测再change_area里lu=0的位置
    change_types = cur_diff * tmp_plan  # change_types: 变化区域

    # pure score
    total_pure_score = 0.0
    ca_labels, ca_num_features = scipy.ndimage.label(cur_diff)
    # 检查变化区域里的每个联通区域
    for ca_i in range(1, ca_num_features + 1):
        # 提取当前连通区域的像素位置
        current_change_area = np.where(ca_labels == ca_i, 1, 0)
        current_change_types = change_types * current_change_area
        # 获取包括的类型数以及各类型的斑块数
        current_lu_types, counts = np.unique(current_change_types[current_change_types!=0],return_counts=True)
        current_lu_types = current_lu_types.tolist()
        current_num_feature_list = get_numfeature_dict(current_change_types, current_lu_types)
        # 统计总共斑块数
        sum_of_num_feature = sum(current_num_feature_list.values())
        # 变化区域里的每个联通区域，里面包括的类型数越少越好，该类型的斑块数越少越好
        pure_score = len(current_lu_types) * sum_of_num_feature
        # 在类型数和连通区域数都相同时，Shannon熵越低越好。
        total_count = np.sum(counts)
        probabilities = counts / total_count 
        shannon_entropy = -np.sum(probabilities * np.log2(probabilities)) 
        pure_score = pure_score * (shannon_entropy + 1)           
        total_pure_score += pure_score

    # class distance
    total_cls_distance = 0.0
    for lu in lu_types:
        if int(lu)==0:
            labels, num_features = scipy.ndimage.label(change_types == 100.0)
        else:
            labels, num_features = scipy.ndimage.label(change_types == lu)
        
        if np.all(labels==0):  # 若为全0张量，说明这个LU类没有发生改变
            continue

        # 找出ori_plan里lu_i的区域
        # ori_lui_area = (ori_plan == lu)
        # distance_map = scipy.ndimage.distance_transform_edt(ori_lui_area)
        distance_map = info_dict["ori_lu_distancemap_dict"][lu]

        cls_distance = 0.0
        # 对change_area里同类的所有连通区域
        for i in range(1, num_features + 1):
            # 提取当前连通区域的像素位置
            current_area = np.where(labels == i, 1, 0)
            area = np.sum(current_area)   
            perimeter = get_perimeter(current_area)
            ratio = perimeter/area  # 周长面积比,越接近1越好

            min_distance = np.min(distance_map * current_area)
            max_distance = np.max(distance_map * current_area)

            if min_distance == 0:
                cls_distance -= 10/ratio
            elif min_distance > 0:
                cls_distance += ((min_distance + max_distance) * ratio**2)
        # 该类LU增加的连通区域，希望越少越好
        ori_num_features = ori_num_feature_list[lu]
        tmp_num_features = tmp_num_feature_list[lu]
        d_num_features = tmp_num_features - ori_num_features + 1
        d_num_features = d_num_features if d_num_features>=1 else 1
        cls_distance = cls_distance + d_num_features**2 
        total_cls_distance += cls_distance

    total_score = total_cls_distance + total_pure_score *(0.3+change_ratio*0.2)
    return torch.tensor(total_score)