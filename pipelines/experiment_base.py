# 参数扫描实验的公共基类 (ablation / tornado / sensitivity)
import os
import sys
import json
import copy
import glob
import random
from argparse import Namespace
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner import create_runner
from datasets.base_dataloader import get_dataset
from utils.calculator_utils import train_maxent
from utils.tools import get_yaml, _set_by_path
from utils.io_utils import ensure_folder
from libs.const import apply_const_overrides, restore_const_defaults
from pipelines.experiment import process_batch


def _safe_extract_idx(index: str) -> str:
    """安全地从 index 字符串中提取编号部分.

    兼容 'year_idx' 和 'year_idx_delta' 格式, 均返回 idx (即下划线分隔的第二段).
    """
    if '_' in index:
        parts = index.split('_')
        return parts[1] if len(parts) >= 2 else parts[0]
    return index


class BaseSweepPipeline:
    """参数扫描实验的公共基类.

    子类需实现 build_matrix()，可选覆写 apply_type_config_hook / get_run_seed /
    get_random_mask 以注入各自差异逻辑.

    Attributes
    ----------
    result_category : str
        实验结果保存子目录名 (ablation / tornado). 子类覆写此属性来决定保存位置.
    """

    result_category: str = 'ablation'  # 子类覆写: 'tornado'

    def __init__(
        self,
        opt_cfg_path: str,
        score_cfg_path: str,
        idxs: Optional[List[int]] = None,
        mode: str = 'ablation',
        n_repeats: int = 1,
        is_processed: bool = False,
        processed_timestamp: Optional[str] = None,
    ):
        if is_processed and not processed_timestamp:
            raise ValueError("断点续传必须提供 processed_timestamp")

        self.opt_cfg_path = opt_cfg_path
        self.score_cfg_path = score_cfg_path
        self.idxs = idxs
        self.mode = mode
        self.n_repeats = n_repeats
        self.is_processed = is_processed
        self.processed_timestamp = processed_timestamp

        # 1. 加载配置
        self.opt_cfg = get_yaml(opt_cfg_path)
        self.base_score_cfg = self._load_score_cfg(score_cfg_path)

        # 2. 设备
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.opt_cfg.device.use_gpu else "cpu"
        )

        # 3. 数据
        self.dataset = get_dataset(type="optimize", data_config=self.opt_cfg.data)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.opt_cfg.data.batch_size,
            shuffle=self.opt_cfg.data.shuffle,
        )

        # 4. MaxEnt 模型
        self.maxent_path = self.opt_cfg.data.maxent_model_path.format(year=self.opt_cfg.data.year)
        train_maxent(self.opt_cfg.data.root, self.opt_cfg.data.year, self.maxent_path)

        # 5. runner 通用配置
        self.runner_config = self.opt_cfg.runner
        self.use_landscape = self.runner_config.use_landscape
        self.type_config_base = getattr(self.opt_cfg, self.runner_config.type)
        self.scenario = self.runner_config.scenario if self.runner_config.scenario != 'all' else 'default'
        self.weight_dict = vars(getattr(self.runner_config.weight, self.scenario))

        # 6. 时间戳与根目录
        self.run_ts = processed_timestamp if is_processed else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root_save = os.path.join('results', self.result_category, self.run_ts)
        ensure_folder(self.root_save)

        print(f"================= 开始 {self.__class__.__name__} [mode={mode}] =================")
        print(f"Optimizer config: {opt_cfg_path}")
        print(f"Score config:     {score_cfg_path}")
        print(f"Using device: {self.device}")
        print(f"  结果根目录: {self.root_save}")
        print(f"  scenario: {self.scenario}, weight: {self.weight_dict}")

    def _load_score_cfg(self, score_cfg_path: str):
        from utils.tools import load_score_cfg
        return load_score_cfg(score_cfg_path)

    # ------------------------------------------------------------
    # 子类覆写点
    # ------------------------------------------------------------
    def build_matrix(self) -> List[Dict[str, Any]]:
        """子类必须实现：构造实验矩阵."""
        raise NotImplementedError

    def apply_type_config_hook(self, type_config: Namespace, exp: Dict[str, Any]) -> None:
        """子类可选覆写：在 type_config 深拷贝后应用额外设置."""
        pass

    def get_run_seed(self, exp: Dict[str, Any], index: str, r: int) -> Optional[int]:
        """子类可选覆写：返回确定性随机种子；None 表示不设置."""
        return None

    def get_random_mask(
        self,
        data_dict: Dict[str, torch.Tensor],
        index: str,
        exp: Dict[str, Any],
        r: int,
    ) -> Optional[torch.Tensor]:
        """子类可选覆写：返回该 run 的 random_restrict_mask."""
        return None

    # ------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------
    def _should_skip_sample(self, index: str) -> bool:
        """根据 idxs 过滤样本."""
        if self.idxs and len(self.idxs) > 0:
            idx_part = _safe_extract_idx(index)
            if idx_part not in [str(x) for x in self.idxs]:
                return True
        return False

    def _experiment_complete(self, exp_save_folder: str, exp: Dict[str, Any]) -> bool:
        """判断本进程负责的 (样本, run) 是否已全部产出 convergence.json.

        不能仅凭 manifest.json 判定: manifest 由任一先跑完该实验的进程写入, 当多个
        分片共享同一时间戳目录时, 其它分片会因此整实验跳过自己尚未计算的样本
        (导致结果大面积缺失). 这里改为按本进程 idxs × n_repeats 逐个核对结果文件.
        """
        if not self.idxs:
            return False
        n_rep = exp.get('n_repeats', self.n_repeats)
        wanted = {str(x) for x in self.idxs}
        for r in range(n_rep):
            run_dir = os.path.join(exp_save_folder, f'run_{r}')
            done = {
                _safe_extract_idx(os.path.basename(os.path.dirname(p)))
                for p in glob.glob(os.path.join(run_dir, '*', 'convergence.json'))
            }
            if not wanted.issubset(done):
                return False
        return True

    def _write_manifest(self, exp_save_folder: str, exp: Dict[str, Any]) -> None:
        manifest = {
            'name': exp['name'],
            'overrides': exp.get('overrides', {}),
            'flags': exp.get('flags', {}),
            'runner_type': self.runner_config.type,
            'scenario': self.scenario,
            'weight_dict': self.weight_dict,
        }
        if 'opt_overrides' in exp:
            manifest['opt_overrides'] = exp['opt_overrides']
        with open(os.path.join(exp_save_folder, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def _write_summary(self, matrix: List[Dict[str, Any]], extra_fields: Optional[Dict] = None) -> None:
        summary = {
            'timestamp': self.run_ts,
            'mode': self.mode,
            'result_category': self.result_category,
            'runner_type': self.runner_config.type,
            'idxs': self.idxs,
            'experiments': [e['name'] for e in matrix],
        }
        if extra_fields:
            summary.update(extra_fields)
        with open(os.path.join(self.root_save, 'experiment_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------
    def run(self) -> str:
        matrix = self.build_matrix()
        print(f"  实验数量: {len(matrix)} (mode={self.mode}, n_repeats={self.n_repeats})")

        for exp in matrix:
            exp_name = exp['name']
            exp_save_folder = os.path.join(self.root_save, exp_name)
            if self.is_processed and self._experiment_complete(exp_save_folder, exp):
                print(f"\n实验【{exp_name}】已完成, 跳过")
                continue

            print(f"\n================= 实验【{exp_name}】=================")
            print(f"  overrides: {exp.get('overrides', {})}")
            if exp.get('opt_overrides'):
                print(f"  opt_overrides: {exp['opt_overrides']}")
            print(f"  flags:     {exp.get('flags', {})}")

            # 构造 score_cfg
            score_cfg = copy.deepcopy(self.base_score_cfg)
            for dotted_path, v in exp.get('overrides', {}).items():
                _set_by_path(score_cfg, dotted_path, v)
            apply_const_overrides(score_cfg)

            n_rep = exp.get('n_repeats', 1)

            type_config = copy.deepcopy(self.type_config_base)
            self.apply_type_config_hook(type_config, exp)

            # 应用 opt_overrides；若属性不存在则跳过，避免 GA 模式下写入 SA 专属参数崩溃
            for dotted_path, v in exp.get('opt_overrides', {}).items():
                try:
                    _set_by_path(type_config, dotted_path, v)
                except AttributeError:
                    print(f"    [warn] opt_overrides '{dotted_path}' 在 {self.runner_config.type} 配置中不存在, 跳过")

            ensure_folder(exp_save_folder)

            for i, data_pack in enumerate(self.dataloader):
                index = data_pack[5][0]
                if self._should_skip_sample(index):
                    continue

                print(f"  -> 样本 {index} ({i+1}/{len(self.dataloader)}), n_repeats={n_rep}")
                data_dict = process_batch(data_pack, self.opt_cfg)

                for r in range(n_rep):
                    # 确定性种子
                    seed = self.get_run_seed(exp, index, r)
                    if seed is not None:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        random.seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)

                    # random mask（子类可按 run 生成）
                    random_mask = self.get_random_mask(data_dict, index, exp, r)

                    save_path = os.path.join(exp_save_folder, f'run_{r}')
                    sample_save = os.path.join(save_path, index)

                    if self.is_processed and os.path.exists(os.path.join(sample_save, 'convergence.json')):
                        print(f"     run {r} 已完成, 跳过")
                        continue

                    ensure_folder(sample_save)

                    runner = create_runner(
                        runner_type=self.runner_config.type,
                        config=type_config,
                        common_config=self.runner_config.common_params,
                        model_config=self.opt_cfg.model,
                        device=self.device,
                        data_dict=data_dict,
                        index=index,
                        save_path=save_path,
                        weight_dict=self.weight_dict,
                        use_landscape=self.use_landscape,
                        guided=exp.get('flags', {}).get('guided', True),
                        disable_early_stop=exp.get('flags', {}).get('disable_early_stop', False),
                        score_cfg=score_cfg,
                        random_restrict_mask=random_mask,
                    )
                    runner.run()

            self._write_manifest(exp_save_folder, exp)

            # 恢复全局常量默认值，防止跨实验状态泄漏
            restore_const_defaults()

        self._write_summary(matrix)
        print(f"\n================= {self.__class__.__name__} 结束 =================")
        print(f"  结果根目录: {self.root_save}")
        return self.root_save
