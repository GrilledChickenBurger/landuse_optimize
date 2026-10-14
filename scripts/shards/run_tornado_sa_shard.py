# -*- coding: utf-8 -*-
"""tornado SA 参数扰动 分片执行器 (收敛自 18307 /tmp/run_tornado_sa10_shard.py).

用法:
  python run_tornado_sa_shard.py <shard_id 0-9> <timestamp> [score_cfg]

要点:
  - base 配置 configs/optimizerConfig_stability.yaml (transunet/year15, vsp=0.8);
  - 10 样本 (= Set Y, 与稳定性/消融/通用扰动一致), 每片 1 个样本;
  - SA 专用参数扰动 (论文 3.5.3.3): 1 基线 + 10 扰动 = 11 组;
  - 分片专属 maxent 路径 + 关 GPU(走 CPU 推理); 仓库根由脚本位置推导; 临时 cfg 写 /tmp。
"""
import os
import sys

# 线程封顶: 必须在导入 numpy/torch 之前设置
if sys.platform.startswith("linux"):
    for _e in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_e, "1")

# 仓库根 = scripts/shards/ -> scripts/ -> repo
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", 1)))

import utils.tornado_utils as tu  # noqa: E402
from pipelines.tornado import tornado_pipeline  # noqa: E402

# 用法: python run_tornado_sa_shard.py <shard_id 0-9> <timestamp> [score_cfg]
shard = int(sys.argv[1])
ts = sys.argv[2]
score_cfg = sys.argv[3] if len(sys.argv) > 3 else os.path.join("configs", "score_cfg.yaml")

# 10 样本 (= Set Y), 每片 1 个样本
IDXS = [1875, 639, 2439, 1833, 1956, 857, 2025, 1909, 531, 1506]
idxs = [IDXS[shard]]

# base = SA 稳定性配置; 逐分片隔离 maxent 路径 + 关 GPU (CPU 推理)
base = open("configs/optimizerConfig_stability.yaml", encoding="utf-8").read()
base = base.replace("_{year}_stab.ela", "_{year}_sa_tor10_s%d.ela" % shard)
base = base.replace("use_gpu: true", "use_gpu: false")
out_cfg = "/tmp/opt_tornado_sa10_s%d.yaml" % shard
open(out_cfg, "w", encoding="utf-8").write(base)

# SA 专用参数扰动 (论文 3.5.3.3): 单因素扰动, 其余默认; 1 基线 + 10 扰动 = 11 组
tu._TORNADO_PARAMS = [
    ('cooling_rate', 0.9995, [0.995, 0.998, 0.9998], 'opt_cfg'),
    ('max_beststuck_iter', 3000, [1000, 6000], 'opt_cfg'),
    ('max_noimprove_iter', 1000, [500, 2000, 5000], 'opt_cfg'),
    ('T_start', 100, [80, 200], 'opt_cfg'),
]

print("=== 启动 SA tornado(10分片, 1样本/片, CPU) shard=%d idxs=%s ts=%s score=%s ===" %
      (shard, idxs, ts, score_cfg), flush=True)
root = tornado_pipeline(
    opt_cfg_path=out_cfg,
    score_cfg_path=score_cfg,
    idxs=idxs,
    mode="tornado",
    n_repeats=int(os.environ.get("TOR_REPEATS", 3)),
    is_processed=True,
    processed_timestamp=ts,
)
print("=== TORNADO SA10 shard %d DONE: %s ===" % (shard, root), flush=True)
