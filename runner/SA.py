import os
import json
from argparse import Namespace
from typing import Dict, Any, List, Optional, Union, Tuple
import copy
import torch
import numpy as np
from libs.const import PIXEL_NUM_THRESHOLD_ISCHANGE, COLOR_MAP
from generator.sa_generator import generate_new_plan
from calculator.tgt_function_cal import cal_tgt_function
from utils.performance import write_performance_score
import utils.calculator_utils as cUtils
import utils.runner_utils as rUtils
import utils.dataset_utils as dUtils
import utils.tools as tUtils


class SimulatedAnnealing:
    """
    模拟退火优化器类
    用于土地利用规划优化
    """
    def __init__(     
            self,       
            config: Namespace,
            common_config: Namespace,
            model_config: Namespace,
            device: torch.device,
            data_dict: Dict[str, torch.Tensor],
            index: str,
            save_path: str,
            weight_dict: Optional[Dict[str, float]] = None,
            use_landscape: bool = False,
            ):
        """
        
        Parameters:
        ----------
        config : ConfigObject
            SA配置对象，包含SA所需的必要的参数
        common_config : Namespace
            SA和GA的公共配置对象，包含路径在内的公共参数
        model_config : Namespace
            模型配置对象，包含模型类型.type 模型路径.model_path和参数
        device : torch.device
            运行设备（CPU/GPU）
        data_dict : Dict[str, torch.Tensor]，包含所有驱动因子和生态系统服务数据的字典
            ori_plan : 原始年份土地利用情况，形状 [1, 1, 128, 128]
            final_plan : 目标年份土地利用【真实】情况，形状 [1, 1, 128, 128]
            factors : 原始年份驱动因子，形状 [1, factor_num, 128, 128]
            restrict : 硬性限制区域，形状 [1, 1, 128, 128], 0-1, 0为受限区域, 1为可修改区域
            ... : 其他数据, 如生态系统服务数据和驱动因子数据
        index : str, year_idx_delta, 如 '10_764_5'
            土地利用索引
        save_path : str
            结果保存路径
        weight_dict : Dict, optional
            权重类型，默认为 'default'对应的权重字典
        use_landscape : bool, optional
            是否使用景观破碎化约束，默认为 True
        """
        print("================= SA Initing... =================")
        
        # 初始化基本参数
        self.device = device
        self.ori_plan = data_dict['ori_plan']
        self.factors = data_dict['factors']
        self.final_plan = data_dict['final_plan']
        self.index = index
        self.es_weight_dict = weight_dict
        self.use_landscape = use_landscape
        
        # 模拟退火参数
        self.t_0 = config.T_start
        self.t_end = config.T_end
        self.cooling_rate = config.cooling_rate
        
        # 流程提前结束条件
        self.max_noimprove_iter = config.max_noimprove_iter  # 最大无改进迭代次数
        self.max_beststuck_iter = config.max_beststuck_iter  # 最优解停滞阈值
 
        # 先验信息
        self.year = index.split('_')[0]
        model_type = model_config.type
        type_config = getattr(model_config, model_type, None)
        ANN_model = rUtils.load_ANN_model(model_type, model_config.model_path, 
                                self.factors.shape[1]+6, 1, device,
                                type_config)
        self.pred_diff = rUtils.get_pred_diff(ANN_model, self.ori_plan, self.factors, self.device,
                                                year=self.year,
                                                condition_list= ["DEM", "DL", "GDP", "RK"] ,
                                                normalize=True)
        self.frg_model_path = common_config.frg_model_path  # fragstats模型路径

        # LU方案相关信息
        self.lu_labels = rUtils.get_lu_labels(self.ori_plan)

        # 生成新方案相关限制信息
        redline_restrict = data_dict['restrict']
        pred_grid_restrict = rUtils.get_grid_restrict_area(self.pred_diff)
        lu_zero_restrict = (self.ori_plan != 0)  # LU=0的地方不可改变
        self.hard_restrict = torch.logical_and(torch.logical_and(redline_restrict, pred_grid_restrict), lu_zero_restrict)
        self.cls_restrict = rUtils.get_cls_restrict_area(self.ori_plan)
        self.pred_diff_pixelnum = torch.count_nonzero(torch.logical_and(self.hard_restrict, self.pred_diff)).item()
        self.scale_minmax_dict = rUtils.get_scale_minmax_dict(self.pred_diff_pixelnum)

        # 目标函数计算相关信息
        self.pred_diff_distance_map = cUtils.get_pred_diff_distance_map(self.pred_diff)
        self.info_dict = cUtils.get_info_dict(
                                    self.ori_plan,
                                    self.pred_diff,
                                    data_dict,
                                    common_config.carbon_pool_path, 
                                    common_config.maxent_model_path.format(year=self.year),
                                    common_config.maxent_list)
        if self.use_landscape:
            ori_distance_map_dict = cUtils.get_lu_distancemap_dict(self.ori_plan)
            ori_num_feature_dict = cUtils.get_numfeature_dict(self.ori_plan)
            self.info_dict['ori_distance_map_dict'] = ori_distance_map_dict
            self.info_dict['ori_num_feature_dict'] = ori_num_feature_dict

        # 初始化当前最佳方案
        self.best_plan = self.ori_plan
        self.best_score = float('inf') # 初始化为无穷大，因为是最小化问题
        self.best_score_iter = 0  # 最优解出现的迭代代数

        # text, charts, logs
        self.total_score_list = []
        self.change_score_list = []
        self.landscape_score_list = []
        self.es_score_list = []
        self.es_dict:Dict[str,List[float]] = {'food':[],'carbon':[],'water':[],'culture':[]}
        self.text = ''
        self.print_frequency = config.print_frequency
        self.visualize_frequency = config.visualize_frequency
        print(f"Done. LU_idx: {self.index}, weight: {self.es_weight_dict}.\n")

        # save result
        self.save_path = save_path


    def update_score_lists(self, score_list:List):
        """
        更新记录目标函数项的列表 + 说明文字\n
        Parameters:
        ----------
            score_list: List, 包含目标函数项的列表
                total_score: tensor[float], 总分
                change_score: tensor[float], 目标函数项1, 和预测变化区域的接近程度得分
                es_score: tensor[float], 目标函数项2, 生态系统服务综合得分
                es_dict: Dict, 生态系统服务综合得分的字典
                landscape_score: tensor[float], 目标函数项3, 破碎化限制得分
        """
        total_score, change_score, es_score, es_dict, landscape_score = score_list
        self.total_score_list.append(total_score)
        self.change_score_list.append(-change_score)
        self.es_score_list.append(es_score)
        self.text = f'change_score={-change_score:.4f}, ES_score={-es_score:.4f}'
        for key, value in es_dict.items():
            self.es_dict[key].append(value)

        if self.use_landscape:
            self.landscape_score_list.append(-landscape_score)
            self.text = f'{self.text}, landscape_score={-landscape_score:.4f}'
        
        self.text = f'{self.text}.'


    def evaluate_new_plan(self, new_plan: torch.Tensor, new_diff: torch.Tensor, new_score_list: List[float], 
                        cur_plan: torch.Tensor, cur_diff: torch.Tensor, cur_score_list: List[float], cur_temp: float) -> tuple:
        """
        根据Metropolis准则判断是否接受新解\n
        Parameters:
        ----------
            new_plan: torch.Tensor, 新解土地利用方案
            new_diff: torch.Tensor, 新解与初始解的差异
            new_score_list: list, 新解的分数列表
            cur_plan: torch.Tensor, 当前解土地利用方案
            cur_diff: torch.Tensor, 当前解与初始解的差异
            cur_score_list: list, 当前解的分数列表
            cur_temp: float, 当前温度
        
        Returns:
        ----------
            cur_plan: torch.Tensor, 更新后的当前解土地利用方案
            cur_diff: torch.Tensor, 更新后的当前解与初始解的差异
            cur_score_list: list, 更新后的当前解的分数列表
            is_accepted: bool, 是否接受新解
        """
        cur_total_score = cur_score_list[0]
        new_total_score = new_score_list[0]
        delta_score = new_total_score - cur_total_score
        p = torch.clamp(torch.exp(-delta_score / cur_temp), min=1e-6).item()
        is_accepted = delta_score < 0 or torch.rand(1).item() < p

        # 接受新解则更新当前状态
        if is_accepted:
            cur_plan = new_plan.clone()
            cur_diff = new_diff.clone()
            cur_score_list = copy.deepcopy(new_score_list)
        return cur_plan, cur_diff, cur_score_list, is_accepted


    def run(self):
        """
        执行模拟退火优化过程
        
        Returns:
        ----------
        torch.Tensor
            优化后的最佳土地利用方案
        """
        if self.pred_diff_pixelnum < PIXEL_NUM_THRESHOLD_ISCHANGE:
            print("预测不发生改变，不进入模拟退火过程。")
            return self.ori_plan
        
        print(f"================= 【{self.index}】SA Running... =================")
        
        # 初始化温度
        cur_temp = self.t_0
        cur_plan = self.ori_plan.clone()
        cur_diff = torch.zeros_like(self.pred_diff)
        cur_score_list = rUtils.get_initial_score_list()
        cur_score = cur_score_list[0]
        # 迭代次数计数器
        iteration = 0
        noimprove_iter = 0
        result_folder = os.path.join(self.save_path, self.index)
        tUtils.ensure_folder(result_folder)
        
        # 模拟退火主循环
        while cur_temp > self.t_end:
            # 生成新解
            new_plan, d_diff = generate_new_plan(cur_plan, 
                                        self.ori_plan,
                                        T_ratio = cur_temp / self.t_0,
                                        noimprove_step=noimprove_iter,
                                        scale_minmax_dict=self.scale_minmax_dict,
                                        lu_labels=self.lu_labels,
                                        hard_restrict=self.hard_restrict,
                                        cls_restrict=self.cls_restrict)
            
            # 计算新解的分数
            new_diff = torch.tensor(new_plan != self.ori_plan, dtype=torch.float32)
            new_score_list = cal_tgt_function(new_plan, new_diff, 
                                            self.pred_diff,
                                            self.ori_plan,
                                            self.info_dict, 
                                            self.es_weight_dict,
                                            cur_temp,
                                            self.use_landscape)
            
            # 评估新解
            prev_score = cur_score_list[0]
            cur_plan, cur_diff, cur_score_list, is_accepted = self.evaluate_new_plan(
                                        new_plan, new_diff, new_score_list, 
                                        cur_plan, cur_diff, cur_score_list, 
                                        cur_temp)
            # 更新迭代次数计数器
            noimprove_iter = noimprove_iter + 1 if prev_score <= cur_score_list[0] else 0
            # 更新当前分数
            cur_score = cur_score_list[0]
            # 更新记录目标函数项的列表
            self.update_score_lists(cur_score_list)
            
            # 更新全局最优解（最小化问题）
            if cur_score < self.best_score:
                self.best_plan = cur_plan.clone()
                self.best_score = cur_score
                self.best_score_iter = iteration  # 更新最优解出现的迭代代数
                print(f"============ step {iteration}: New best score = {self.best_score:.4f} ==============")
            
            # 是否提前终止？
            if noimprove_iter >= self.max_noimprove_iter or \
                    iteration - self.best_score_iter >= self.max_beststuck_iter:
                print(f"SA提前终止，未改进连续{self.max_noimprove_iter}step 或 最优解停滞{self.max_beststuck_iter}step。")
                break

            # 打印进度
            if iteration % self.print_frequency == 0:
                print(f"\nStep {iteration} noimprove_iter: {noimprove_iter} beststuck_iter: {self.best_score_iter}")
                print(f"        温度: {cur_temp:.4f}, 当前分数={cur_score:.4f}, 最佳分数={self.best_score:.4f}")
            
            if iteration % self.visualize_frequency == 0:
                print(f"        change_score={cur_score_list[1]:.4f}, ES_score={cur_score_list[2]:.4f}, landscape_score={cur_score_list[4]:.4f}.")
                tensor_dict = {
                    'ori_plan': self.ori_plan,
                    'cur_plan': cur_plan,
                    'pred_diff': self.pred_diff,
                    'cur_diff': cur_diff,
                }
                tUtils.visualize_tensor_list(tensor_dict, self.text,
                                            save_path=os.path.join(result_folder, 'history', f'iter_{iteration}.png'),
                                            figsize=(20, 6),
                                            color_map=COLOR_MAP)
            # 降温
            cur_temp *= self.cooling_rate
            iteration += 1
        
        print(f"\n随机优化结束! 最佳分数: {self.best_score:.4f}")

        # 保存最佳方案
        idx = self.index.split("_")[1]
        best_diff = torch.from_numpy(dUtils.remove_lines(self.best_plan != self.ori_plan))
        self.best_plan = torch.where(best_diff==1, self.best_plan, self.ori_plan)

        rUtils.save_all_plans(self.ori_plan, self.best_plan, self.final_plan, self.index, 
                        os.path.join(result_folder, f'plans_{idx}'),
                        raw_data_root=None)
        # 写入性能指标
        scores = write_performance_score(self.best_plan, self.ori_plan, self.final_plan, self.pred_diff,
                                self.info_dict,
                                src_folder=os.path.join(result_folder, f'plans_{idx}'),
                                output_folder=os.path.join(result_folder, f'performance_{idx}'),
                                landscape=False,
                                frg_model_path=self.frg_model_path,
                                )               
        output_file = os.path.join(result_folder, 'performance_score.json')
        with open(output_file,'w') as file:
            json.dump(scores, file, indent=4)  
        # 绘制目标函数项的变化趋势图
        score_list_data = [self.total_score_list, self.change_score_list, self.es_score_list, self.landscape_score_list, self.es_dict ]
        rUtils.draw_score_list(score_list_data, ['total_score', 'change_score', 'ES_score', 'landscape_score'],
                                save_folder=os.path.join(result_folder, 'charts'))
        # 可视化最终方案
        tensor_dict = {
                    'ori_plan': self.ori_plan,
                    'best_plan': self.best_plan,
                    'final_plan': self.final_plan,
                    'pred_diff': self.pred_diff,
                    'opt_diff': (self.ori_plan != self.best_plan),
                    'real_diff': (self.ori_plan != self.final_plan),
                }
        tUtils.visualize_tensor_list(tensor_dict, self.text,
                                    save_path=os.path.join(result_folder, 'best_score.png'),
                                    color_map=COLOR_MAP)

        return self.best_plan


# 主函数（如果需要直接运行SA.py）
def main():
    # 这里可以放置直接运行SA的代码
    print("SA optimizer module")


if __name__ == "__main__":
    main()