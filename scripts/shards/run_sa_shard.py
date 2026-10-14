# -*- coding: utf-8 -*-
"""SA 分片启动入口: 每片 10 个样本 (scenario=all → 4 个情景), 与 GA 分片模式完全一致.

- 复用 main.py 顶部线程/确定性设置 (import main 触发).
- 使用 configs/optimizerConfig_sa.yaml (transunet/year15, T_start=100, T_end=50, chain_length=10).
- 按分片 id 隔离 MaxEnt 路径 (checkpoints/maxent-model_sklearn_15_sa_s<shard>.ela), 避免并发写同一 .ela.
- 所有分片写同一个 timestamp 根目录 (results/SA/<ts>/<scenario>/), 靠 is_processed 断点续跑.

用法(仓库根目录):
    nohup ./.venv/bin/python -u scripts/shards/run_sa_shard.py <shard_id 0-5> <timestamp> > sa_shard<id>.log 2>&1 &
    (或直接运行 scripts/shards/launch_sa_6shards.sh)
"""
import os
import sys

# 仓库根目录 = 本脚本目录的上两级 (scripts/shards/ -> scripts/ -> repo)
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
sys.path.append(ROOT)

import yaml  # noqa: E402
import main  # noqa: E402,F401  触发 main.py 顶部: OMP 线程限制 + torch 确定性配置
from pipelines.run_runner import pipeline  # noqa: E402

# 60 个样本索引 (与 GA 分片完全一致; 6 片 × 每片 10 个)
IDXS = [
    1006, 1936, 1875, 1137, 2323, 2267, 1359, 639, 1822, 640,
    1539, 848, 2439, 2635, 2148, 630, 1262, 2625, 2083, 391,
    2010, 600, 2499, 1480, 1479, 2502, 1345, 857, 1698, 1857,
    829, 530, 2024, 2149, 655, 654, 2018, 1900, 1594, 824,
    1096, 598, 1533, 2387, 1512, 1962, 660, 2391, 1650, 531,
    1158, 1724, 1963, 2142, 1030, 2265, 1575, 1714, 1854, 1896,
]

BASE_CFG = os.path.join("configs", "optimizerConfig_sa.yaml")
PER_SHARD = 10


def run(shard: int, ts: str) -> None:
    start = shard * PER_SHARD
    idxs = IDXS[start:start + PER_SHARD]
    print(f"[run_sa_shard] shard={shard} ts={ts} idxs({len(idxs)})={idxs}")

    with open(BASE_CFG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 分片隔离 MaxEnt 路径 (data 与 common_params 必须一致)
    mp = f"checkpoints/maxent-model_sklearn_15_sa_s{shard}.ela"
    cfg["data"]["maxent_model_path"] = mp
    cfg["runner"]["common_params"]["maxent_model_path"] = mp

    tmp_cfg = f"/tmp/opt_sa_shard{shard}.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    pipeline(tmp_cfg, idxs, True, ts)


if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2])
