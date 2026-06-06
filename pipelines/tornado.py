# tornado / sensitivity pipeline: 参数粗筛 + 一维线搜索
import os
import sys
from typing import Dict, Any, List, Optional

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.experiment_base import BaseSweepPipeline
from utils.tornado_utils import build_experiment_matrix as _build_tornado_matrix


class TornadoPipeline(BaseSweepPipeline):
    """Tornado / Sensitivity 实验流水线."""

    result_category = 'tornado'

    def __init__(self, top_k_params: Optional[List[Dict[str, Any]]] = None, **kwargs):
        self.top_k_params = top_k_params
        super().__init__(**kwargs)

    def build_matrix(self) -> List[Dict[str, Any]]:
        return _build_tornado_matrix(
            self.mode, n_repeats=self.n_repeats, top_k_params=self.top_k_params
        )

    def get_run_seed(self, exp: Dict[str, Any], index: str, r: int) -> Optional[int]:
        """提供确定性随机种子，保证实验可复现."""
        exp_name = exp['name']
        return (
            (abs(hash(exp_name)) % (2 ** 31)) +
            (abs(hash(index)) % (2 ** 31)) +
            r
        )

    def _write_summary(self, matrix: List[Dict[str, Any]], extra_fields: Optional[Dict] = None) -> None:
        """tornado/sensitivity 的 summary 额外记录参数列表."""
        extra = {
            'parameters_tested': [
                {'name': e['name'], 'overrides': e.get('overrides', {}),
                 'opt_overrides': e.get('opt_overrides', {})}
                for e in matrix
            ],
        }
        super()._write_summary(matrix, extra_fields=extra)


def tornado_pipeline(opt_cfg_path: str,
                     score_cfg_path: str,
                     idxs: Optional[List[int]] = None,
                     mode: str = 'tornado',
                     n_repeats: int = 2,
                     is_processed: bool = False,
                     processed_timestamp: Optional[str] = None,
                     top_k_params: Optional[List[Dict[str, Any]]] = None) -> str:
    """Tornado / Sensitivity 实验流水线.

    Returns:
    --------
    str
        结果根目录路径
    """
    pipeline = TornadoPipeline(
        opt_cfg_path=opt_cfg_path,
        score_cfg_path=score_cfg_path,
        idxs=idxs,
        mode=mode,
        n_repeats=n_repeats,
        is_processed=is_processed,
        processed_timestamp=processed_timestamp,
        top_k_params=top_k_params,
    )
    return pipeline.run()


