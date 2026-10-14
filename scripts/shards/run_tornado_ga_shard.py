# -*- coding: utf-8 -*-
"""tornado GA 参数扰动 分片执行器 (收敛自 18307 /tmp/run_tornado_ga10_shard.py).

用法:
  python run_tornado_ga_shard.py <shard_id 0-9> <timestamp> [score_cfg]

要点:
  - base 配置 configs/optimizerConfig_stability_ga.yaml, 并按论文 GA 基线覆盖
    (N_pop=20, p_c=0.5, p_m=0.5, G_stall=100);
  - 10 样本 (= Set Y), 每片 1 个样本;
  - GA 专用参数扰动 (论文 3.5.3.4): 1 基线 + 8 扰动 = 9 组;
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

import yaml  # noqa: E402
import utils.tornado_utils as tu  # noqa: E402
from pipelines.tornado import tornado_pipeline  # noqa: E402

# 用法: python run_tornado_ga_shard.py <shard_id 0-9> <timestamp> [score_cfg]
shard = int(sys.argv[1])
ts = sys.argv[2]
score_cfg = sys.argv[3] if len(sys.argv) > 3 else os.path.join("configs", "score_cfg.yaml")

# 10 样本 (= Set Y), 每片 1 个样本
IDXS = [1875, 639, 2439, 1833, 1956, 857, 2025, 1909, 531, 1506]
idxs = [IDXS[shard]]

# base = GA 稳定性配置; 论文 GA 基线 + 逐分片隔离 maxent 路径 + 关 GPU(CPU 推理)
with open("configs/optimizerConfig_stability_ga.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["device"]["use_gpu"] = False
cfg["GA"]["ori_pop_size"] = 20
cfg["GA"]["crossover_rate"] = 0.5
cfg["GA"]["mutation_rate"] = 0.5
cfg["GA"]["max_noimprove_gen"] = 100
mp = "checkpoints/maxent-model_sklearn_15_tor_ga10_s%d.ela" % shard
cfg["data"]["maxent_model_path"] = mp
cfg["runner"]["common_params"]["maxent_model_path"] = mp
out_cfg = "/tmp/opt_tornado_ga10_s%d.yaml" % shard
with open(out_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)

# GA 专用参数扰动 (论文 3.5.3.4): 单因素扰动, 其余保持默认; 1 基线 + 8 扰动 = 9 组
tu._TORNADO_PARAMS = [
    ('ori_pop_size', 20, [10, 30], 'opt_cfg'),
    ('crossover_rate', 0.5, [0.3, 0.7], 'opt_cfg'),
    ('mutation_rate', 0.5, [0.3, 0.7], 'opt_cfg'),
    ('max_noimprove_gen', 100, [50, 150], 'opt_cfg'),
]

print("=== 启动 GA tornado(10分片, 1样本/片, CPU) shard=%d idxs=%s ts=%s score=%s ===" %
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
print("=== TORNADO GA10 shard %d DONE: %s ===" % (shard, root), flush=True)
