import scipy.ndimage
import torch
import numpy as np
from typing import Optional
from argparse import Namespace

from utils.calculator_utils import get_numfeature_dict, get_perimeter
from utils.tools import get_default_score_cfg

def cal_landscape_score(cur_plan: torch.Tensor, cur_diff: torch.Tensor,
                            ori_plan: torch.Tensor,
                            info_dict: dict,
                            change_ratio: float = 0.6,
                            score_cfg: Optional[Namespace] = None) -> torch.Tensor:
    """
    第3?项: 衡量当前更改区域里的所有连通区域和ori_plan对应同类区域之间的距离.\n
    Parameters:
        plan: tensor, [1,1,128,128]
        change_area: tensor, [1,1,128,128]
        score_cfg: Namespace, optional, 评分常数配置. None 时使用默认值
    """
    cfg = score_cfg if score_cfg is not None else get_default_score_cfg()
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

    total_score = total_cls_distance + total_pure_score *(cfg.landscape.pure_base + change_ratio * cfg.landscape.pure_change)
    return torch.tensor(total_score)

def cal_landscape_score_v2_minimal(cur_plan: torch.Tensor, cur_diff: torch.Tensor,
                            ori_plan: torch.Tensor,
                            info_dict: dict,
                            change_ratio: float = 0.6,
                            score_cfg: Optional[Namespace] = None) -> torch.Tensor:
    """
    第3项: 衡量当前更改区域里的所有连通区域和ori_plan对应同类区域之间的距离.
    [v2-MINIMAL-INJECT] 新增: 局部邻域聚合度惩罚 + 超线性碎片惩罚 + 面积加权距离惩罚

    文献基础:
    - Aerts et al. (2003): 邻域方法 (Layer 1)
    - Stewart et al. (2004): 超线性簇惩罚 (Layer 2)  
    - He et al. (2000): 聚合指数AI (Layer 2)
    """
    cfg = score_cfg if score_cfg is not None else get_default_score_cfg()

    # ===== 原有代码: 数据准备 =====
    tmp_num_feature_list = get_numfeature_dict(cur_plan)
    ori_num_feature_list = info_dict["ori_num_feature_list"]

    cur_plan = cur_plan.squeeze().numpy()
    cur_diff = cur_diff.squeeze().numpy()
    ori_plan = ori_plan.squeeze().numpy()
    lu_types = np.unique(ori_plan[ori_plan!=0.0]).tolist()

    tmp_plan=np.copy(cur_plan)
    tmp_plan[tmp_plan ==0.0] = 100.0     # 把里面的0以100代替，方便后续检测再change_area里lu=0的位置
    change_types = cur_diff * tmp_plan  # change_types: 变化区域

    # [v2-INSERT-1] 计算fragmentation超线性惩罚因子 (基于Stewart 2004)
    # 原理: 连通块数量越多，每个块的pure_score被放大的倍数越高
    # 旧: pure_score线性累加; 新: pure_score * (1 + 0.05 * sqrt(num_patches-1))
    ca_labels_for_frag, ca_num_features_for_frag = scipy.ndimage.label(cur_diff)
    frag_factor = 1.0 + 0.05 * np.sqrt(max(ca_num_features_for_frag - 1, 0))
    # 注: frag_factor范围 ~[1.0, 1.5]，对1块无影响，100块时=1.5，控制数量级不变

    # ===== 原有代码: pure_score计算 =====
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
        # [v2-MODIFY-1] 乘以fragmentation因子，使碎片方案的pure_score增加
        total_pure_score += pure_score * frag_factor  # 旧: total_pure_score += pure_score

    # [v2-INSERT-2] Layer 1: 局部邻域聚合度惩罚 (Aerts 2003)
    # 原理: 每个变化像素的8邻居中，同类像素越少惩罚越大
    # 孤立像素惩罚=8，完全被同类包围惩罚=0
    from scipy.ndimage import uniform_filter

    neighbor_penalty = 0.0
    change_mask = cur_diff > 0
    total_change_pixels = np.sum(change_mask)

    if total_change_pixels > 0:
        for lu in lu_types:
            lu_mask = (cur_plan == lu).astype(np.float32)
            # 3x3卷积: 统计8邻域中同类像素数量
            neighbor_sum = uniform_filter(lu_mask, size=3, mode='constant', cval=0.0) * 9.0 - lu_mask
            changed_lu_mask = change_mask & (cur_plan == lu)
            if np.any(changed_lu_mask):
                # 惩罚 = 8 - 同类邻居数 (最少0)
                lu_penalty = np.sum(np.maximum(0, 8 - neighbor_sum[changed_lu_mask]))
                neighbor_penalty += lu_penalty

        # 归一化: 除以总变化像素数，使neighbor_penalty与pure_score同数量级
        # 对于9830像素的典型场景，纯大块neighbor_penalty≈1-5，全碎片≈50-80
        neighbor_weight = getattr(cfg.landscape, 'neighbor_weight', 1.0)
        neighbor_penalty = neighbor_penalty / (total_change_pixels + 1e-6) * neighbor_weight

    # ===== 原有代码: cls_distance计算 (保留结构，微调公式) =====
    total_cls_distance = 0.0
    for lu in lu_types:
        if int(lu)==0:
            labels, num_features = scipy.ndimage.label(change_types == 100.0)
        else:
            labels, num_features = scipy.ndimage.label(change_types == lu)

        if np.all(labels==0):  # 若为全0张量，说明这个LU类没有发生改变
            continue

        distance_map = info_dict["ori_lu_distancemap_dict"][lu]

        cls_distance = 0.0
        # 对change_area里同类的所有连通区域
        for i in range(1, num_features + 1):
            # 提取当前连通区域的像素位置
            current_area = np.where(labels == i, 1, 0)
            area = np.sum(current_area)   
            perimeter = get_perimeter(current_area)
            ratio = perimeter/area  # 周长面积比

            min_distance = np.min(distance_map * current_area)
            max_distance = np.max(distance_map * current_area)

            if min_distance == 0:
                # [v2-MODIFY-2] 贴边奖励: 面积越大奖励越多，但capped避免无限累积
                # 旧: cls_distance -= 10/ratio (小碎片ratio大，奖励虽小但可累积)
                # 新: min(10/ratio, 3.0 * sqrt(area)) 使大块获得更高奖励上限
                cls_distance -= min(10.0 / ratio, 3.0 * np.sqrt(area))
            elif min_distance > 0:
                # [v2-MODIFY-3] 远离惩罚: 增加距离幂次和面积加权
                # 旧: cls_distance += ((min_distance + max_distance) * ratio**2)
                # 新: 距离^1.2 * sqrt(面积) * ratio^2，大块远离惩罚更重
                cls_distance += ((min_distance ** 1.2) * np.sqrt(area) * (ratio ** 2))

        # 该类LU增加的连通区域，希望越少越好
        ori_num_features = ori_num_feature_list[lu]
        tmp_num_features = tmp_num_feature_list[lu]
        d_num_features = tmp_num_features - ori_num_features + 1
        d_num_features = d_num_features if d_num_features>=1 else 1
        cls_distance = cls_distance + d_num_features**2 
        total_cls_distance += cls_distance

    # [v2-MODIFY-4] 最终组合: 加入neighbor_penalty
    # 旧: total_score = total_cls_distance + total_pure_score * (pure_base + change_ratio * pure_change)
    # 新: 增加 neighbor_penalty 项
    total_score = total_cls_distance + total_pure_score * (cfg.landscape.pure_base + change_ratio * cfg.landscape.pure_change) + neighbor_penalty
    return torch.tensor(total_score)

def cal_landscape_score_v2(cur_plan: torch.Tensor, cur_diff: torch.Tensor,
                            ori_plan: torch.Tensor,
                            info_dict: dict,
                            change_ratio: float = 0.6,
                            score_cfg: Optional[Namespace] = None) -> torch.Tensor:
    """
    景观紧凑性评分函数 v2 - 改进版

    目的: 在确定变化区域时，使得变化区域内部尽可能是大片且纯度较高的区域，
         一块变化区域内尽量不要多个种类，而且新增区域尽量靠近原始方案中的同类型区域。

    文献基础:
    [1] Aerts et al. (2003) - 邻域方法: 最大化每个单元格8邻域中相同土地利用类型的数量
    [2] Stewart et al. (2004) - 多目标紧凑性: 最小化簇数量、最大化最大簇、优化形状指数
    [3] Ligmann-Zielinska et al. (2008) - 空间优化生成技术: 密度约束
    [4] Cao et al. (2011, 2012) - 空间紧凑度评估方法比较
    [5] He et al. (2000) - 聚合指数(AI)量化空间格局
    [6] Janssen et al. (2008) - 土地利用规划中的多目标优化

    改进点:
    1. 增加邻域聚合度奖励 (Neighbor-based Aggregation Reward) - 参考 Aerts et al. (2003)
    2. 增加超线性连通块惩罚 (Super-linear Patch Penalty) - 参考 Stewart et al. (2004) 
    3. 增加面积加权归一化 (Area-weighted Normalization) - 避免大面积方案被过度惩罚
    4. 保留并改进距离-形状耦合惩罚 (Distance-Shape Coupled Penalty)

    三层架构:
    Layer 1: 局部邻域聚合度 - 每个变化像素与其邻居的同类聚合
    Layer 2: 全局斑块结构   - 连通块数量和聚合指数
    Layer 3: 空间位置偏差   - 变化区域与原始同类型区域的距离

    Parameters:
        cur_plan: tensor, [1,1,128,128] - 当前方案
        cur_diff: tensor, [1,1,128,128] - 变化区域
        ori_plan: tensor, [1,1,128,128] - 原始方案
        info_dict: dict - 预计算信息
        change_ratio: float - 变化比例
        score_cfg: Namespace - 评分配置

    Returns:
        torch.Tensor - landscape_score (越小越好)
    """
    cfg = score_cfg if score_cfg is not None else get_default_score_cfg()

    # ===== 数据准备 =====
    cur_plan_np = cur_plan.squeeze().numpy()
    cur_diff_np = cur_diff.squeeze().numpy()
    ori_plan_np = ori_plan.squeeze().numpy()

    # 获取变化像素的掩码
    change_mask = cur_diff_np > 0
    total_change_pixels = np.sum(change_mask)

    # 如果没有变化，返回0
    if total_change_pixels == 0:
        return torch.tensor(0.0)

    # 获取LU类型列表
    lu_types = np.unique(ori_plan_np[ori_plan_np != 0.0]).tolist()

    # 当前和原始方案里，每个LU类的连通区域数
    tmp_num_feature_list = get_numfeature_dict(cur_plan)
    ori_num_feature_list = info_dict["ori_num_feature_list"]

    # ===== LAYER 1: 局部邻域聚合度惩罚 (Local Neighborhood Aggregation Penalty) =====
    # 参考: Aerts et al. (2003) - "Using Linear Integer Programming for Multi-Site Land-Use Allocation"
    # 核心思想: 每个变化像素的8邻域中，相同LU类型的邻居越多，聚合度越高
    # 对于每个变化像素，统计其8邻域中有多少是不同类的（惩罚）

    # 使用scipy的uniform_filter来计算邻居数量
    from scipy.ndimage import uniform_filter

    neighbor_penalty = 0.0

    # 对每个LU类型分别计算邻域聚合度
    for lu in lu_types:
        # 当前方案中该LU类型的掩码
        lu_mask = (cur_plan_np == lu).astype(np.float32)

        # 8邻域卷积: 每个像素位置的8邻居中同类型数量
        # uniform_filter with size=3 gives average of 3x3 neighborhood
        # multiply by 9 to get sum, subtract center to get 8-neighbor sum
        neighbor_sum = uniform_filter(lu_mask, size=3, mode='constant', cval=0.0) * 9.0 - lu_mask

        # 只对变化区域中该LU类型的像素计算
        changed_lu_mask = change_mask & (cur_plan_np == lu)

        if np.any(changed_lu_mask):
            # 对于每个变化像素，邻居中不同类的数量 = 8 - 同类邻居数
            # neighbor_sum gives same-type neighbors, so penalty = max(0, 8 - neighbor_sum)
            # but cap at 0 since edge pixels have fewer neighbors
            lu_penalty = np.sum(np.maximum(0, 8 - neighbor_sum[changed_lu_mask]))
            neighbor_penalty += lu_penalty

    # 归一化: 除以总变化像素数和变化比例
    # 这样大面积变化的惩罚不会线性增长
    neighbor_penalty = neighbor_penalty / (total_change_pixels * change_ratio + 1e-6)

    # ===== LAYER 2: 全局斑块结构惩罚 (Global Patch Structure Penalty) =====
    # 参考: Stewart et al. (2004) - 最小化簇数量
    # 参考: He et al. (2000) - 聚合指数(AI)
    # 核心思想: 
    #   2a. 超线性惩罚连通块数量
    #   2b. 使用聚合指数(AI)衡量整体紧凑度

    # 2a: 计算总连通块数量（所有LU类合计）
    total_patches = 0
    for lu in lu_types:
        lu_change_mask = change_mask & (cur_plan_np == lu)
        if np.any(lu_change_mask):
            _, num_patches = scipy.ndimage.label(lu_change_mask)
            total_patches += num_patches

    # 超线性惩罚: patches^1.5（而非线性累加）
    # 这样10个碎片的惩罚是31.6，而1个大块是1.0，差距更大
    patch_count_penalty = (total_patches ** 1.5) if total_patches > 0 else 0.0

    # 2b: 计算聚合指数 (Aggregation Index, AI) - He et al. (2000)
    # AI = g_ii / g_max_ii
    # g_ii = 同类像素共享的边数 (4-connected)
    # g_max_ii = 理论最大共享边数 = 2 * n_pixels * (sqrt(n_pixels) - 1) / sqrt(n_pixels)
    # 对于方形格子，简化: g_max ≈ 2 * n_pixels - 2 * ceil(sqrt(n_pixels))

    total_shared_edges = 0
    total_max_edges = 0

    for lu in lu_types:
        lu_change_mask = change_mask & (cur_plan_np == lu)
        n_pixels = np.sum(lu_change_mask)

        if n_pixels > 0:
            # 计算4-连通共享边数
            # 水平共享边
            h_shared = np.sum(lu_change_mask[:-1, :] & lu_change_mask[1:, :])
            # 垂直共享边
            v_shared = np.sum(lu_change_mask[:, :-1] & lu_change_mask[:, 1:])
            shared_edges = h_shared + v_shared
            total_shared_edges += shared_edges

            # 理论最大共享边数 (假设正方形排列)
            side = int(np.ceil(np.sqrt(n_pixels)))
            max_edges = 2 * n_pixels - 2 * side if n_pixels > 1 else 0
            total_max_edges += max(max_edges, 0)

    # AI = 0 (完全分散) 到 1 (完全聚合)
    if total_max_edges > 0:
        ai = total_shared_edges / total_max_edges
        ai = min(ai, 1.0)  # cap at 1.0
    else:
        ai = 1.0  # 单个像素，视为完全聚合

    # fragmentation penalty = (1 - AI) * total_change_pixels
    fragmentation_penalty = (1.0 - ai) * np.sqrt(total_change_pixels)

    patch_penalty = patch_count_penalty + fragmentation_penalty

    # ===== LAYER 3: 空间位置偏差惩罚 (Spatial Location Deviation Penalty) =====
    # 保留原始思想: 变化区域应靠近原始同类型区域
    # 改进: 使用面积加权，大块影响更大；增加远离惩罚的幂次

    distance_penalty = 0.0

    for lu in lu_types:
        # 当前方案中该LU类型的变化掩码
        lu_change_mask = change_mask & (cur_plan_np == lu)

        if not np.any(lu_change_mask):
            continue

        # 获取预计算的距离图
        distance_map = info_dict["ori_lu_distancemap_dict"][lu]

        # 使用scipy的distance_transform_edt计算变化区域内部像素到原始区域的距离
        # distance_map already gives distance from each pixel to ori_lu area

        # 计算该LU类型的变化区域中的连通块
        labels, num_features = scipy.ndimage.label(lu_change_mask)

        lu_distance_penalty = 0.0

        for i in range(1, num_features + 1):
            patch_mask = (labels == i)
            area = np.sum(patch_mask)

            if area == 0:
                continue

            # 计算该patch的平均距离和最大距离
            patch_distances = distance_map[patch_mask]
            mean_distance = np.mean(patch_distances)
            max_distance = np.max(patch_distances)

            # 计算周长-面积比 (perimeter / area)
            perimeter = get_perimeter(patch_mask)
            ratio = perimeter / area if area > 0 else 0

            # 距离惩罚: 
            # - 如果patch与原始区域接触 (min_distance=0): 奖励，但与面积成正比
            # - 如果远离: 惩罚 = mean_distance * area * ratio^2
            # 改进: 使用面积加权，大块远离惩罚更重
            if mean_distance == 0:
                # 贴边奖励: 面积越大奖励越多，但 capped
                lu_distance_penalty -= min(5.0 * np.sqrt(area), 20.0)
            else:
                # 远离惩罚: 距离 * 面积权重 * 形状因子
                # 使用1.5次幂让远离的惩罚增长更快
                lu_distance_penalty += (mean_distance ** 1.5) * np.sqrt(area) * (ratio ** 2)

        # 连通块数量增加惩罚（该LU类）
        ori_num = ori_num_feature_list.get(lu, 0)
        tmp_num = tmp_num_feature_list.get(lu, 0)
        d_num = max(tmp_num - ori_num + 1, 1)

        # 超线性惩罚
        lu_distance_penalty += (d_num ** 2) * np.log(total_change_pixels + 1)

        distance_penalty += lu_distance_penalty

    # 归一化距离惩罚
    distance_penalty = distance_penalty / (total_change_pixels + 1e-6)

    # ===== 最终组合 =====
    # 三层加权组合
    total_score = (
        cfg.landscape.lambda_neighbor * neighbor_penalty +
        cfg.landscape.lambda_patch * patch_penalty +
        cfg.landscape.lambda_distance * distance_penalty
    )

    return torch.tensor(total_score)