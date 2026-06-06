import os
import json
import time
import functools
from argparse import Namespace
from typing import Dict, Any, List, Optional, Union, Tuple
import copy
import torch
import numpy as np
from libs.const import *
import libs.const as _libs_const
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
            *,
            guided: bool = True,
            disable_early_stop: bool = False,
            skip_threshold_check: bool = False,
            score_cfg: Optional[Namespace] = None,
            random_restrict_mask: Optional[torch.Tensor] = None,
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

        # ablation 开关
        self.guided = guided
        self.disable_early_stop = disable_early_stop
        self.skip_threshold_check = skip_threshold_check
        self.score_cfg = score_cfg if score_cfg is not None else tUtils.get_default_score_cfg()
        self.random_restrict_mask = random_restrict_mask

        # 模拟退火参数
        self.t_0 = config.T_start
        self.t_end = config.T_end
        self.cooling_rate = config.cooling_rate

        # 流程提前结束条件
        if self.disable_early_stop:
            self.max_noimprove_iter = float('inf')
            self.max_beststuck_iter = float('inf')
        else:
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
                                                normalize=True,
                                                sigmoid_threshold=self.score_cfg.runner_utils.sigmoid_threshold)
        self.frg_model_path = common_config.frg_model_path  # fragstats模型路径

        # LU方案相关信息
        self.lu_labels = rUtils.get_lu_labels(self.ori_plan)

        # 生成新方案相关限制信息
        redline_restrict = data_dict['restrict']
        lu_zero_restrict = (self.ori_plan != 0)  # LU=0的地方不可改变
        restrict_parts = [redline_restrict, lu_zero_restrict]
        if self.random_restrict_mask is not None:
            # C 组: 随机连通块约束 (优先于 grid_restrict, 与 guided 互斥)
            restrict_parts.append(self.random_restrict_mask.to(redline_restrict.device))
        elif self.guided:
            pred_grid_restrict = rUtils.get_grid_restrict_area(
                self.pred_diff, grid_num=self.score_cfg.runner_utils.grid_num)
            restrict_parts.append(pred_grid_restrict)
        self.hard_restrict = functools.reduce(torch.logical_and, restrict_parts)
        self.cls_restrict = rUtils.get_cls_restrict_area(self.ori_plan)

        # C 组: change_target 用 random_mask 替代 pred_diff (目标函数 P_change 项对齐随机约束位置)
        # 注意: 必须先于 pred_diff_pixelnum 赋值, 因为后者依赖 self.change_target
        if self.random_restrict_mask is not None:
            self.change_target = self.random_restrict_mask.to(self.pred_diff.dtype)
        else:
            self.change_target = self.pred_diff

        self.pred_diff_pixelnum = torch.count_nonzero(torch.logical_and(self.hard_restrict, self.change_target)).item()
        self.scale_minmax_dict = rUtils.get_scale_minmax_dict(self.pred_diff_pixelnum)

        # 目标函数计算相关信息: distance map 基于 change_target 生成
        self.pred_diff_distance_map = cUtils.get_pred_diff_distance_map(self.change_target)
        self.info_dict = cUtils.get_info_dict(
                                    self.ori_plan,
                                    self.change_target,
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
        if not self.skip_threshold_check and self.pred_diff_pixelnum < _libs_const.PIXEL_NUM_THRESHOLD_ISCHANGE:
            print("预测不发生改变，不进入模拟退火过程。")
            return self.ori_plan

        print(f"================= 【{self.index}】SA Running... =================")

        # 计时
        t0 = time.perf_counter()
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
                                            self.change_target,
                                            self.ori_plan,
                                            self.info_dict,
                                            self.es_weight_dict,
                                            self.use_landscape,
                                            score_cfg=self.score_cfg,
                                            change_scale_n=self.pred_diff_pixelnum)
            
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
        # 使用通用函数保存优化结果
        plan_folder = os.path.join(result_folder, f'plans_{idx}')
        tmp_folder = os.path.join(result_folder, f'performance_{idx}')
        perf_path = os.path.join(result_folder, 'performance.json')
        rUtils.save_plan_and_get_performance(
            best_plan=self.best_plan,
            ori_plan=self.ori_plan,
            final_plan=self.final_plan,
            pred_diff=self.pred_diff,
            info_dict=self.info_dict,
            index=self.index,
            plan_folder=plan_folder,
            tmp_folder=tmp_folder,
            perf_path=perf_path,
            frg_model_path=self.frg_model_path,
            raw_data_root=RAW_DATA_ROOT
        )
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

        # ablation 评估口径: 收敛速度记录
        self._dump_convergence(result_folder, t0)

        return self.best_plan

    def _dump_convergence(self, result_folder: str, t0: float) -> None:
        """保存 total_score_list 和收敛统计, 用于 ablation 实验 A 收敛速度对比"""
        wall_time = time.perf_counter() - t0
        score_arr = np.array([float(s) for s in self.total_score_list])
        np.save(os.path.join(result_folder, 'total_score_list.npy'), score_arr)
        # 最小化问题: 触及 1.05 * best_score 的首次步数 (best_score 为负时反向)
        best = float(self.best_score)
        if best < 0:
            target = best * 0.95   # |target| < |best|
        else:
            target = best * 1.05
        iter_to_95pct = int(np.argmax(score_arr <= target)) if (score_arr <= target).any() else -1

        # 取最优解出现时的 ES score (已保存在 es_score_list 中), 供原始记录表使用
        es_score = (float(self.es_score_list[self.best_score_iter])
                    if self.es_score_list else float('nan'))

        conv = {
            'iter_to_95pct': iter_to_95pct,
            'final_best': best,
            'es_score': es_score,
            'wall_time_sec': wall_time,
            'total_iter': len(score_arr),
            'guided': self.guided,
            'disable_early_stop': self.disable_early_stop,
        }
        with open(os.path.join(result_folder, 'convergence.json'), 'w', encoding='utf-8') as f:
            json.dump(conv, f, indent=2, ensure_ascii=False)


# 主函数（如果需要直接运行SA.py）
def main():
    # 这里可以放置直接运行SA的代码
    print("SA optimizer module")


if __name__ == "__main__":
    main()