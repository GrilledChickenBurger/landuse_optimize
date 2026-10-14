# -*- coding: utf-8 -*-
"""稳定性实验 (SA) 分片启动入口: 每片 2 个样本 × 重复 5 次 (default 场景).

- 复用 main.py 顶部线程/确定性设置 (import main 触发).
- 使用 configs/optimizerConfig_stability.yaml (transunet/year15, vsp=0.8, T_start=100/T_end=50, chain_length=10).
- 按分片 id 隔离 MaxEnt 路径 (checkpoints/maxent-model_sklearn_15_stab_s<shard>.ela), 避免并发写同一 .ela.
- 所有分片写同一个 timestamp 根目录 (results/ablation/<ts>/A_guided/run_<r>/<样本>/), 靠 convergence.json 断点续跑.

用法(仓库根目录):
    nohup ./.venv/bin/python -u scripts/shards/run_stability_shard.py <shard_id 0-4> <timestamp> > stab_sa_shard<id>.log 2>&1 &
    (或直接运行 scripts/shards/launch_stability_5shards.sh)
"""
import os
import sys

# 仓库根目录 = 本脚本目录的上两级 (scripts/shards/ -> scripts/ -> repo)
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
sys.path.append(ROOT)

import yaml  # noqa: E402
import main  # noqa: E402,F401  触发 main.py 顶部: OMP 线程限制 + torch 确定性配置
from pipelines.ablation import ablation_pipeline  # noqa: E402

# 10 个稳定性样本索引 (5 片 × 每片 2 个)
IDXS = [1875, 639, 2439, 1833, 1956, 857, 2025, 1909, 531, 1506]
PER_SHARD = 2
N_REPEATS = 5

BASE_CFG = os.path.join("configs", "optimizerConfig_stability.yaml")
SCORE_CFG = os.path.join("configs", "score_cfg.yaml")


def run(shard: int, ts: str) -> None:
    start = shard * PER_SHARD
    idxs = IDXS[start:start + PER_SHARD]
    print(f"[run_stability_shard] shard={shard} ts={ts} idxs({len(idxs)})={idxs}")

    with open(BASE_CFG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 分片隔离 MaxEnt 路径 (data 与 common_params 必须一致)
    mp = f"checkpoints/maxent-model_sklearn_15_stab_s{shard}.ela"
    cfg["data"]["maxent_model_path"] = mp
    cfg["runner"]["common_params"]["maxent_model_path"] = mp

    tmp_cfg = f"/tmp/opt_stability_shard{shard}.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    ablation_pipeline(
        opt_cfg_path=tmp_cfg,
        score_cfg_path=SCORE_CFG,
        idxs=idxs,
        mode="stability",
        n_repeats=N_REPEATS,
        is_processed=True,
        processed_timestamp=ts,
    )


if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2])
