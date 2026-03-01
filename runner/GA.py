import os
import random
import json
from argparse import Namespace
from typing import Dict, Any, List, Optional, Union, Tuple
import copy
import torch
import numpy as np
from libs.const import PIXEL_NUM_THRESHOLD_ISCHANGE, COLOR_MAP
from generator.ga_generator import initial_new_population, crossover, mutate
from calculator.tgt_function_cal import cal_tgt_function
from utils.performance import write_performance_score
import utils.calculator_utils as cUtils
import utils.runner_utils as rUtils
import utils.dataset_utils as dUtils
import utils.tools as tUtils

class GeneticAlgorithmStrategy:
    """
    遗传算法策略类
    用于定义遗传算法的选择、交叉和变异操作

    | 变体编号   | 父本选择方法 | 父本来源  | 精英保留比例 | 未交叉个体处理 | 精英是否变异 | 预期收敛速度 | 预期解的质量 |
    | ---------- | ------------ | --------- | ------------ | -------------- | ------------ | ------------ | ------------ |
    | GA1 (基准) | 轮盘赌       | 全部种群  | 10%          | 直接复制       | 否           | 中等         | 中等   |
    | GA2        | 锦标赛(k=2)  | 全部种群  | 10%          | 直接复制       | 否           | 稍快         | 较好   |
    | GA3        | 轮盘赌       | 前20%精英 | 10%          | 直接复制       | 否           | 快           | 较低   |
    | GA4        | 轮盘赌       | 全部种群  | 0%           | 直接复制       | 否           | 慢           | 可能高 |
    | GA5        | 轮盘赌       | 全部种群  | 50%          | 直接复制       | 否           | 很快         | 较低   |
    | GA6        | 轮盘赌       | 全部种群  | 10%          | 复制后变异     | 否           | 稍慢         | 较高   |
    | GA7        | 轮盘赌       | 全部种群  | 10%          | 直接复制       | 是           | 中等         | 不稳定 |
    """
    def __init__(self, strategy_dict: Any):
        self.config = vars(strategy_dict)
    
    def build(self):
        """根据配置装配策略函数字典, 包括选择、交叉和变异操作的函数.
        
        Returns:
        --------
            funcs: Dict[str, callable], 策略函数字典
            - 'get_parent_pool': 选择繁殖种群的函数
            - 'select_parents': 选择父本的函数
            - 'handle_non_crossover': 处理未交叉个体的函数
            - 'apply_elite_mutation': 处理精英个体|列表进行变异的函数
        """
        funcs = {}

        # 父本来源
        pool_type = self.config.get('parent_pool', 'all')
        if pool_type == 'all':
            funcs['get_parent_pool'] = self._parent_pool_all
        elif pool_type == 'elite':
            elite_ratio = self.config.get('elite_ratio_for_pool', 0.2)
            funcs['get_parent_pool'] = lambda pop, fit: self._parent_pool_elite(pop, fit, elite_ratio=elite_ratio)

        # 父本选择方法
        sel_type = self.config.get('parent_selection', 'roulette')
        if sel_type == 'roulette':
            funcs['select_parents'] = lambda pool, fit, **kwargs: self._roulette_select(pool, fit, **kwargs)
        elif sel_type == 'tournament':
            k = self.config.get('tournament_k', 2)
            funcs['select_parents'] = lambda pool, fit, **kwargs: self._tournament_select(pool, fit, k=k, **kwargs)

        # 未交叉处理
        nc_type = self.config.get('non_crossover_handling', 'copy')
        if nc_type == 'copy':
            funcs['handle_non_crossover'] = self._handle_non_crossover_copy
        elif nc_type == 'copy_mutate':
            funcs['handle_non_crossover'] = self._handle_non_crossover_copy_mutate

        # 精英变异标志（布尔值，直接存储）
        em_type = self.config.get('elite_mutation', False)
        if em_type:
            funcs['apply_elite_mutation'] = self._apply_elite_mutation
        else:
            funcs['apply_elite_mutation'] = self._apply_no_elite_mutation

        return funcs
    # ---------- 父本来源 ----------
    def _parent_pool_all(self, population: List[torch.Tensor], 
                           fitness_scores: List[float],
                           **kwargs) -> List[torch.Tensor]:
        """
        ## 选择繁殖种群: 全部种群
        直接使用全部种群作为繁殖种群，不进行任何筛选。
        
        Parameters:
        ----------
            population: List[torch.Tensor], 当前种群的土地利用方案列表
            fitness_scores: List[float], 当前种群每个方案的适应度列表
        
        Returns:
        ----------
            parent_population: List[int], 被选中的父母个体【索引】列表
        """
        return list(range(len(population)))
        
    def _parent_pool_elite(self, population: List[torch.Tensor],
                            fitness_scores: List[float], 
                            elite_ratio: float = 0.2,
                            **kwargs) -> List[torch.Tensor]:
        """
        ## 选择繁殖种群: 精英保留
        根据适应度值保留种群中适应度最高的个体，作为下一代的繁殖种群。
        
        Parameters:
        ----------
            population: List[torch.Tensor], 当前种群的土地利用方案列表
            fitness_scores: List[float], 当前种群每个方案的适应度列表
            elite_ratio: float, 精英保留比例，默认为0.2
        
        Returns:
        ----------
            parent_population: List[int], 被选中的父母个体【索引】列表
                列表的长度为`(len(population) * elite_ratio)`取整
        """
        elite_num = int(len(population) * elite_ratio)
        elite_indices = np.argsort(fitness_scores)[-elite_num:]  # 适应度越大越好
        return elite_indices.tolist()

    # ---------- 父本选择方法 ----------
    def _tournament_select(self, pool_indices: List[torch.Tensor], 
                            fitness_scores: List[float], 
                            k: int = 3,
                            offspring_size: int = None,
                            **kwargs) -> List[int]:
        """
        ## 选择繁殖种群: 锦标赛选择
        锦标赛选择在种群中随机选择n个个体，通过"N元锦标赛"选择适应度较高的个体作为繁殖个体，
        并将其保留到下一代种群中。
        
        Parameters:
        ----------
            pool_indices: List[int], 当前种群的土地利用方案【索引】列表
            fitness_scores: List[float], 当前种群每个方案的适应度列表
            k: int, 锦标赛中选择的个体数，默认为2
            offspring_size: int, 选择的个体数量，默认为当前父本种群大小
        
        Returns:
        ----------
            parent_population: List[int], 被选中的父母个体【索引】列表
                列表的长度为`offspring_size`
        """
        selected = []
        offspring_size = offspring_size or len(pool_indices)
        for _ in range(offspring_size):
            competitors = random.sample(pool_indices, k)
            winner = max(competitors, key=lambda x: fitness_scores[x])
            selected.append(winner)
        return selected

    def _roulette_select(self, pool_indices: List[int], 
                                 fitness_scores: List[float], 
                                 offspring_size: int = None,
                                 **kwargs
                                 ) -> List[int]:
        """
        ## 选择繁殖种群: 轮盘赌选择
        轮盘赌选择根据个体的适应度值分配选择概率，适应度越高的个体被选中的概率越大。
        
        Parameters:
        ----------
            pool_indices: List[int], 当前种群的土地利用方案【索引】列表
            fitness_scores: List[float], 当前种群每个方案的适应度列表
            offspring_size: int, 选择的个体数量，默认为当前父本种群大小
            
        Returns:
        -------
            parent_population: List[int], 被选中的父母个体【索引】列表
                列表的长度为`offspring_size`
        """
        selected = []
        offspring_size = offspring_size or len(pool_indices)
        
        total_fitness = sum([fitness_scores[i] for i in pool_indices])  # 计算适应度总和
        # 如果适应度总和为0，则使用均匀分布选择
        if total_fitness == 0:
            return random.choices(pool_indices, k=offspring_size)
        
        # 计算每个个体的选择概率
        probabilities = [fitness_scores[i] / total_fitness for i in pool_indices]
        selected = random.choices(pool_indices, weights=probabilities, k=offspring_size)
        
        return selected

    # ---------- 未交叉个体处理 ----------
    def _handle_non_crossover_copy(self, parent1: torch.Tensor, parent2: torch.Tensor, **kwargs):
        """直接复制父代作为子代（不交叉）"""
        return parent1.clone(), parent2.clone()

    def _handle_non_crossover_copy_mutate(self,parent1: torch.Tensor, parent2: torch.Tensor, mutate_func, **kwargs):
        """复制父代后对副本进行变异"""
        child1 = mutate_func(parent1.clone(), **kwargs)
        child2 = mutate_func(parent2.clone(), **kwargs)
        return child1, child2

    # ---------- 精英变异控制 ----------
    def _apply_elite_mutation(self, elite, mutate_func, **kwargs):
        """对精英个体/列表中的每个精英进行变异"""
        if isinstance(elite, list):
            return [mutate_func(e, **kwargs) for e in elite]
        else:
            return mutate_func(elite, **kwargs)

    def _apply_no_elite_mutation(self, elite, mutate_func, **kwargs):
        """不对非精英个体进行变异"""
        if isinstance(elite, list):
            return [e.clone() for e in elite]
        else:
            return elite.clone()


class GeneticAlgorithm:
    """
    遗传算法优化器类
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
            GA配置对象，包含GA所需的必要的参数
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
        print("================= GA Initing... =================")
        
        # 初始化基本参数
        self.device = device
        self.ori_plan = data_dict['ori_plan']
        self.factors = data_dict['factors']
        self.final_plan = data_dict['final_plan']
        self.index = index
        self.es_weight_dict = weight_dict
        self.use_landscape = use_landscape
        
        # 遗传算法参数
        self.ori_pop_size = config.ori_pop_size  # 初始种群大小
        self.final_pop_size = config.final_pop_size  # 最终种群大小
        self.max_gen = config.max_gen  # 最大迭代次数
        self.crossover_rate = config.crossover_rate  # 交叉率
        self.mutation_rate = config.mutation_rate  # 变异率
        self.elite_ratio = vars(config.strategy).get('elite_ratio', 0.1)  # 精英保留比例
        self.strategy = GeneticAlgorithmStrategy(config.strategy).build()  # 策略, 包括select, crossover, mutation的具体实现方式
        
        # 流程提前结束条件
        self.max_noimprove_gen = config.max_noimprove_gen  # 最大无改进迭代次数
        self.max_beststuck_gen = config.max_beststuck_gen  # 最优解停滞阈值
        
        # 包装mutate函数，预先传入固定参数
        self.mutate_func = self._create_mutate_wrapper()
        self.crossover_func = crossover

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
        # tUtils.visualize_tensor_list(
        #     tensor_dict={
        #         "DEM":self.factors[0,0],
        #         "DL":self.factors[0,1],
        #         "GDP":self.factors[0,2],
        #         "RK":self.factors[0,3],
        #         "ori_plan": self.ori_plan,
        #         "final_plan": self.final_plan,
        #         "pred_diff": self.pred_diff,
        #         'real_diff': torch.from_numpy(dUtils.remove_lines(self.final_plan != self.ori_plan))
        #     },
        #     save_path=f"{self.index}_visual.png",
        #     main_title=f"{self.index}",
        #     color_map='viridis',
        #     show=True,
        # )
        # LU方案相关信息
        self.lu_labels = rUtils.get_lu_labels(self.ori_plan)

        # 生成新方案相关限制信息
        redline_restrict = data_dict['restrict']
        pred_grid_restrict = rUtils.get_grid_restrict_area(self.pred_diff)
        self.hard_restrict = torch.logical_and(redline_restrict, pred_grid_restrict)
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

        # 初始化当前种群 + 当前最佳方案
        self.best_plan = self.ori_plan
        self.best_score = float('-inf') # 初始化为无穷小，因为是最大化问题
        self.best_score_generation = 0  # 记录最优解出现的代数
        self.pop_size = self.ori_pop_size  # 当前种群大小
        self.population = initial_new_population(
                                self.ori_pop_size, 
                                config.scale_ratio, 
                                self.ori_plan, 
                                self.scale_minmax_dict, 
                                self.lu_labels, 
                                self.cls_restrict, 
                                self.hard_restrict)
        self.pop_size_interpolator = tUtils.PiecewiseLinear(
                                    [0.2, 0.9], 
                                    [self.ori_pop_size, self.final_pop_size])
        self.pop_size_factor_interpolator = tUtils.PiecewiseLinear(
                                    [0, 30], 
                                    [1.0, 1.9])
        
        # text, charts, logs
        # 最佳个体级别统计信息
        self.total_score_list = []  # 最佳个体的总得分列表
        self.change_score_list = []  # 最佳个体的变化得分列表
        self.landscape_score_list = []  # 最佳个体的景观破碎化得分列表
        self.es_score_list = []  # 最佳个体的生态系统服务得分列表
        self.es_dict:Dict[str,List[float]] = {'food':[],'carbon':[],'water':[],'culture':[]}
        # 种群级别统计信息
        self.best_fitness_list = []  # 种群最佳适应度值列表
        self.avg_fitness_list = []  # 种群平均适应度值列表
        self.worst_fitness_list = []  # 种群最差适应度值列表
        self.std_fitness_list = []  # 种群适应度值标准差列表
        
        self.idi_text = ''  # 最佳个体的说明文字
        self.pop_text = ''  # 种群的说明文字
        self.print_frequency = config.print_frequency  # 每n步打印一次输出
        self.visualize_frequency = config.visualize_frequency  # 每n步可视化一次输出
        print(f"Done. LU_idx: {self.index}, weight: {self.es_weight_dict}.\n")

        # save result
        self.save_path = save_path


    def _create_mutate_wrapper(self):
        """
        创建mutate函数的包装器，预先传入固定参数
        
        Returns:
        --------
            wrapped_mutate: callable, 包装后的变异函数
            
        """
        def wrapped_mutate(individual: torch.Tensor, 
                            gen_ratio: float, 
                            noimprove_step: int = 0):
            """
            包装后的变异函数
            
            Parameters:
            ----------
                individual: torch.Tensor, 要变异的个体
                gen_ratio: float, 遗传代数比例
                noimprove_step: int, 目标函数未优化的步数，默认为0
            
            Returns:
            --------
                mutated_individual: torch.Tensor, 变异后的个体
            """
            if random.random() < self.mutation_rate:
                return mutate(
                    individual=individual,
                    T_ratio=max(0.0001, 1-gen_ratio),
                    ori_plan=self.ori_plan,
                    noimprove_step=noimprove_step,
                    scale_minmax_dict=self.scale_minmax_dict,
                    lu_labels=self.lu_labels,
                    cls_restrict=self.cls_restrict,
                    hard_restrict=self.hard_restrict
                )
            else:
                return individual.clone()
        
        return wrapped_mutate


    def update_score_lists(self, score_list:List, pop_stats_dict:Dict):
        """
        每次迭代后更新记录最佳个体目标函数信息 + 种群级别统计信息\n
        Parameters:
        ----------
            score_list: List, 包含最佳个体的目标函数项的列表
                total_score: tensor[float], 总分
                change_score: tensor[float], 目标函数项1, 和预测变化区域的接近程度得分
                es_score: tensor[float], 目标函数项2, 生态系统服务综合得分
                es_dict: Dict, 生态系统服务综合得分的字典
                landscape_score: tensor[float], 目标函数项3, 破碎化限制得分
            pop_stats_dict: Dict, 包含种群级别的统计信息
                best_fitness: float, 最佳适应度值
                worst_fitness: float, 最差适应度值
                avg_fitness: float, 平均适应度值
                std_fitness: float, 适应度值的标准差
        """
        # ----- 最佳个体级别统计信息 -----
        total_score, change_score, es_score, es_dict, landscape_score = score_list
        self.total_score_list.append(total_score)
        self.change_score_list.append(-change_score)
        self.es_score_list.append(es_score)
        for key, value in es_dict.items():
            self.es_dict[key].append(value)
        if self.use_landscape:
            self.landscape_score_list.append(-landscape_score)
        
        # ----- 种群级别统计信息 -----
        self.best_fitness_list.append(pop_stats_dict["best_fitness"])
        self.worst_fitness_list.append(pop_stats_dict["worst_fitness"])
        self.avg_fitness_list.append(pop_stats_dict["avg_fitness"])
        self.std_fitness_list.append(pop_stats_dict["std_fitness"])

    def update_text(self, score_list:List, pop_stats_dict:Dict):
        """
        更新最佳个体和种群的说明文字\n
        Parameters:
        ----------
            score_list: List, 包含最佳个体的目标函数项的列表
            pop_stats_dict: Dict, 包含种群级别的统计信息
        """
        total_score, change_score, es_score, es_dict, landscape_score = score_list
        self.idi_text = f'change_score={-change_score:.4f}, ES_score={-es_score:.4f}'
        idi_text_parts = [f'change_score={-change_score:.4f}', f'ES_score={-es_score:.4f}']
        if self.use_landscape:
            idi_text_parts.append(f'landscape_score={-landscape_score:.4f}')
        self.idi_text = ', '.join(idi_text_parts) + '.'

        pop_text_parts = [f'best_fitness={pop_stats_dict["best_fitness"]:.4f}',
                          f'worst_fitness={pop_stats_dict["worst_fitness"]:.4f}',
                          f'avg_fitness={pop_stats_dict["avg_fitness"]:.4f}',
                          f'std_fitness={pop_stats_dict["std_fitness"]:.4f}']
        self.pop_text = ', '.join(pop_text_parts) + '.'

    def quick_shot(self, population: List[torch.Tensor], 
                            fitness_list: List[float],
                            save_path: str) -> None:
        """
        快照，完整展示整个population\n
        Parameters:
        ----------
            population: List[torch.Tensor], 当前种群的土地利用方案列表
            fitness_list: List[float], 当前种群每个方案的适应度列表
        """
        sorted_indices = np.argsort(fitness_list)[::-1]  # 按适应度值从大到小排序
        tensor_dict = {
            'ori_plan': self.ori_plan,
            'pred_diff': self.pred_diff,
        }
        for idx in sorted_indices[:49]:
            title = f'fitness={fitness_list[idx]:.2f}'
            tensor_dict[title] = population[idx]
            diff = torch.from_numpy(dUtils.remove_lines(population[idx] != self.ori_plan))
            tensor_dict[title + ' (diff)'] = diff
        
        # 种群中可能有相同的个体，tensor_dict的长度可能小于49*2
        tUtils.visualize_tensor_list(tensor_dict, 
                                     main_title = self.idi_text + '\n' + self.pop_text,
                                     save_path=save_path, color_map=COLOR_MAP,
                                     layout=(len(tensor_dict)//10+1, 10))

    def cal_population_stats(self, fitness_list, generation):
        """
        计算【种群】级别的统计信息。
        数据为目标函数【总得分】列表，不涉及目标函数各项得分\n
        Parameters:
        ----------
            fitness_list: List[float], 当前种群每个方案的适应度列表
            generation: int, 当前迭代代数
        
        Returns:
        ----------
            stats: Dict, 包含种群级别的统计信息
                generation: int, 当前迭代代数
                best_fitness: float, 最佳适应度值
                worst_fitness: float, 最差适应度值
                avg_fitness: float, 平均适应度值
                std_fitness: float, 适应度值的标准差
                best_individual_idx: int, 最佳个体的索引
        """
        stats = {
            'generation': generation,
            'best_fitness': max(fitness_list),
            'worst_fitness': min(fitness_list),
            'avg_fitness': np.mean(fitness_list),
            'std_fitness': np.std(fitness_list),
            'best_individual_idx': np.argmax(fitness_list)
        }
        return stats

    def adaptive_pop_size(self, gen_ratio: float, noimprove_step: int = 0):
        """
        自适应种群大小，根据遗传代数比例gen_ratio和未优化步数noimprove_step调整种群大小\n
        Parameters:
        ----------
            gen_ratio: float, 遗传代数比例
            noimprove_step: int, 目标函数未优化的步数，默认为0
        """
        # 使用插值器计算
        base_size = self.pop_size_interpolator(gen_ratio)
        factor = self.pop_size_factor_interpolator(noimprove_step)
        
        # 综合计算并限制范围: 种群大小在[final_pop_size, ori_pop_size*2]之间
        self.pop_size = max(self.final_pop_size, min(self.ori_pop_size*2, int(base_size * factor)))

    def fitness(self, population: List[torch.Tensor], ) -> List:
        """
        评估当前种群的适应度, 返回得分列表 ,为后续选择父母、挑选精英个体做准备\n
        Parameters:
        ----------
            population: List[torch.Tensor], 当前种群的土地利用方案列表
        
        Returns:
        ----------
            fitness_list: List[...], 当前种群每个方案的适应度列表，每个元素的结构是
                -total_score: torch.Tensor[float], 总得分，越大越好
                -change_score: torch.Tensor[float], 目标函数项1, 和预测变化区域的接近程度得分
                -es_score: torch.Tensor[float], 目标函数项2, 生态系统服务综合得分
                -es_dict: Dict, 生态系统服务综合得分的字典
                -landscape_score: torch.Tensor[float], 目标函数项3, 破碎化限制得分
            
        """
        fitness_list = []
        for plan in population:
            diff = torch.tensor(plan != self.ori_plan, dtype=torch.float32)
            score_list_ori = cal_tgt_function(plan, diff, 
                                            self.pred_diff,
                                            self.ori_plan,
                                            info_dict=self.info_dict, 
                                            es_weight_dict=self.es_weight_dict,
                                            use_landscape=self.use_landscape)
            score_list = list(score_list_ori)
            score_list[0] = -score_list[0]  # 适应度为负总得分
            fitness_list.append(score_list)  
        return fitness_list


    def crossover_for_offspring(self, population: List[torch.Tensor], parent_indices: List[int], gen_ratio: float, noimprove_step: int):
        """
        基于父代索引进行交叉操作，生成子代种群\n
        Parameters:
        ----------
            population: List[torch.Tensor], 当前种群的土地利用方案列表
            parent_indices: List[int], 选择的父代索引列表
            gen_ratio: float, 遗传代数比例
            noimprove_step: int, 目标函数未优化的步数
        
        Returns:
        ----------
            offspring: List[torch.Tensor], 生成的子代种群
        """
        offspring = []
        for i in range(0, len(parent_indices)-1, 2):
            p1 = population[parent_indices[i]]
            p2 = population[parent_indices[i+1]]
            if random.random() < self.crossover_rate:
                # 交叉
                c1, c2 = self.crossover_func(p1, p2, gen_ratio=gen_ratio,
                                             hard_restrict=self.hard_restrict)
                c1 = self.mutate_func(c1, gen_ratio=gen_ratio, noimprove_step=noimprove_step)
                c2 = self.mutate_func(c2, gen_ratio=gen_ratio, noimprove_step=noimprove_step)
                offspring.extend([c1, c2])
            else:
                # 未交叉处理: mutate / no mutate
                c1, c2 = self.strategy['handle_non_crossover'](p1, p2, gen_ratio=gen_ratio,
                                                                noimprove_step=noimprove_step,
                                                                mutate_func=self.mutate_func,
                                                                )
                offspring.extend([c1, c2])
        
        return offspring


    def run(self):
        """执行遗传算法优化过程
        
        Returns:
        ----------
        best_plan: torch.Tensor 优化后的最佳土地利用方案
        """
        if self.pred_diff_pixelnum < PIXEL_NUM_THRESHOLD_ISCHANGE:
            print("预测不发生改变，不进入遗传算法过程。")
            return self.ori_plan
        
        print("================= GA Running... =================")
        
        # 初始化当前最佳方案
        cur_plan = self.ori_plan.clone()
        cur_score_list = rUtils.get_initial_score_list()
        cur_score_list[0] = -cur_score_list[0]  # 初始化为无穷大，因为是最小化问题
        cur_score = cur_score_list[0]
        # 迭代次数计数器
        gen = 0
        noimprove_step = 0  # 记录连续未改进迭代次数
        result_folder = os.path.join(self.save_path, self.index)
        tUtils.ensure_folder(result_folder)
        
        population = self.population
        # 遗传算法主循环
        for gen in range(self.max_gen):            
            # 计算当前种群的适应度
            score_list = self.fitness(population)
            fitness_list = [score[0].item() for score in score_list]
            pop_stat_dict = self.cal_population_stats(fitness_list, gen)
            # 找到最佳个体
            prev_score = cur_score  # 记录上一代最佳适应度
            cur_plan = population[np.argmax(fitness_list)]
            cur_score_list = score_list[np.argmax(fitness_list)]
            cur_score = cur_score_list[0].item()
            # 更新记录 - 传入种群适应度
            self.update_score_lists(cur_score_list, pop_stat_dict)
            self.update_text(cur_score_list, pop_stat_dict)
            noimprove_step = 0 if cur_score > prev_score else noimprove_step + 1
            
            # 是否提前终止？
            if noimprove_step >= self.max_noimprove_gen or \
                    gen - self.best_score_generation >= self.max_beststuck_gen:
                print(f"GA提前终止，未改进连续{self.max_noimprove_gen}代 或 最优解停滞{self.max_beststuck_gen}代。")
                break

            # --- 自适应种群大小 ---
            self.adaptive_pop_size(gen_ratio=gen/self.max_gen)

            # --- 选择父本池: all / elite ---
            pool_indices = self.strategy['get_parent_pool'](population, fitness_list)

            #  --- 选择父本: 锦标赛 / 轮盘赌 ---
            num_parents_needed = self.pop_size  # 为简化，生成与种群相同数量的子代
            parent_indices = self.strategy['select_parents'](pool_indices, fitness_list, 
                                                            offspring_size=num_parents_needed)
            
            # --- 生成子代 ---
            offspring = self.crossover_for_offspring(population, parent_indices, 
                                                    gen_ratio=gen/self.max_gen, 
                                                    noimprove_step=noimprove_step)
            
            # 确保子代数量与种群大小一致（可能多一个或少一个，此处简化处理）
            offspring = offspring[:self.pop_size]
            
            # --- 精英保留: 0 / 0.1 / 0.5 ---
            elite_num = int(self.pop_size * self.elite_ratio)
            if elite_num > 0:
                # 选出当前最优个体
                elite_indices = np.argsort(fitness_list)[-elite_num:]
                elites = [population[i].clone() for i in elite_indices]
                # --- 精英变异控制: true / false ---
                offspring[:elite_num] = self.strategy['apply_elite_mutation'](elites, gen_ratio=gen/self.max_gen,
                                                                    noimprove_step=noimprove_step,
                                                                    mutate_func=self.mutate_func,
                                                                    )                     
            population = offspring
            
            # 更新全局最优解（最大化问题）
            if cur_score > self.best_score:
                self.best_plan = cur_plan.clone()
                self.best_score = cur_score
                self.best_score_generation = gen  # 记录最优解出现的代数
                print(f"============ step {gen}: New best score = {self.best_score:.4f} ==============")
            
            # 打印进度
            if gen % self.print_frequency == 0:
                print(f"\nStep {gen} noimprove_step={noimprove_step} beststuck_gen={gen - self.best_score_generation}")
                print(f"        种群avg: {pop_stat_dict['avg_fitness']:.4f}, 最差: {pop_stat_dict['worst_fitness']:.4f}")
                print(f"        当前best: {cur_score:.4f}, 最佳best: {self.best_score:.4f}")
                print(f"        change_score={cur_score_list[1]:.4f}, ES_score={cur_score_list[2]:.4f}, landscape_score={cur_score_list[4]:.4f}.")
                
            if gen % self.visualize_frequency == 0:
                history_save_path=os.path.join(result_folder, 'history', f'iter_{gen}.png')
                self.quick_shot(population, fitness_list, history_save_path)
        
        print(f"\n随机优化结束! 最佳分数: {self.best_score:.4f}")

        # 保存最佳方案
        idx = self.index.split("_")[1]
        best_diff = torch.from_numpy(dUtils.remove_lines(self.best_plan != self.ori_plan))
        self.best_plan = torch.where(best_diff==1, self.best_plan, self.ori_plan)

        rUtils.save_all_plans(self.ori_plan, self.best_plan, self.final_plan, self.index, 
                        os.path.join(result_folder, f'plans_{idx}'),
                        'D:\\_programming\\_LAB_ASSIGMENTS\\AAAAA栅格分割')
        # 写入性能指标
        scores = write_performance_score(self.best_plan, self.ori_plan, self.final_plan, self.pred_diff,
                                self.info_dict,
                                src_folder=os.path.join(result_folder, f'plans_{idx}'),
                                frg_model_path=self.frg_model_path,
                                output_folder=os.path.join(result_folder, f'performance_{idx}'))               
        output_file = os.path.join(result_folder, 'performance_score.json')
        with open(output_file,'w') as file:
            json.dump(scores, file, indent=4)          
        # 绘制目标函数项的变化趋势图
        score_list_data = [
            self.total_score_list,  # 最佳个体总得分
            self.change_score_list,  # 最佳个体目标函数项1, 和预测变化区域的接近程度得分
            self.es_score_list,  # 最佳个体目标函数项2, 生态系统服务综合得分
            self.landscape_score_list,  # 最佳个体目标函数项3, 破碎化限制得分
            self.es_dict # 最佳个体ES得分细分
        ]
        rUtils.draw_score_list(score_list_data, ['total_score', 'change_score', 'ES_score', 'landscape_score'],
                                save_folder=os.path.join(result_folder, 'charts'))
        # 绘制种群级别的统计信息
        pop_stats_data = [
            self.best_fitness_list,  # 最佳适应度值
            self.worst_fitness_list,  # 最差适应度值
            self.avg_fitness_list,  # 平均适应度值
            self.std_fitness_list,  # 适应度值的标准差
        ]
        rUtils.draw_score_list_general(pop_stats_data, ['best_fitness', 'worst_fitness', 'avg_fitness', 'std_fitness'],
                                save_folder=os.path.join(result_folder, 'charts'), multi=True)
        
        # 可视化最终方案
        tensor_dict = {
                    'ori_plan': self.ori_plan,
                    'best_plan': self.best_plan,
                    'final_plan': self.final_plan,
                    'pred_diff': self.pred_diff,
                    'opt_diff': (self.ori_plan != self.best_plan),
                    'real_diff': (self.ori_plan != self.final_plan),
                }
        tUtils.visualize_tensor_list(tensor_dict, self.idi_text,
                                    save_path=os.path.join(result_folder, 'best_score.png'),
                                    color_map=COLOR_MAP)

        return self.best_plan



# 主函数（如果需要直接运行SA.py）
def main():
    # 这里可以放置直接运行SA的代码
    print("SA optimizer module")


if __name__ == "__main__":
    main()