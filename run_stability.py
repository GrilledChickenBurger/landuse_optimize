# -*- coding: utf-8 -*-
"""稳定性实验的并行启动入口 (不改动 main.py).

复用 main.py 顶部的线程与确定性设置 (直接 import main 触发), 然后调用
AblationPipeline 的 stability 模式: 10 样本 / default 场景 / 重复 5 遍.

使用隔离的 MaxEnt 配置 configs/optimizerConfig_stability.yaml,
新进程只会训练/读取 checkpoints/maxent-model_sklearn_<year>_stab.ela,
不影响当前主运行的 maxent-model_sklearn_<year>.ela.

用法:
    nohup ./.venv/bin/python -u run_stability.py > sa_y15_stability.log 2>&1 &
"""
import os
import sys

# 切到项目根目录 (以脚本所在目录为准), 保证相对路径 (_data/configs/results) 正确
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.append(ROOT)

import main  # noqa: E402,F401  触发 main.py 顶部: OMP 线程限制 + torch 确定性配置
from pipelines.ablation import ablation_pipeline  # noqa: E402
from utils.exp_sweep_utils import load_representative_idxs  # noqa: E402


def run() -> None:
    idxs = load_representative_idxs()
    print(f"[run_stability] 待优化样本 ({len(idxs)}): {idxs}")
    ablation_pipeline(
        opt_cfg_path=os.path.join("configs", "optimizerConfig_stability.yaml"),
        score_cfg_path=os.path.join("configs", "score_cfg.yaml"),
        idxs=idxs,
        mode="stability",
        n_repeats=5,
    )


if __name__ == "__main__":
    run()
