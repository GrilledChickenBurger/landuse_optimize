# -*- coding: utf-8 -*-
"""tornado 通用参数扰动 (cm) 分片执行器 (收敛自 18307 /tmp/run_tornado_common_shard.py).

用法:
  python run_tornado_common_shard.py <shard_tag> <idx_csv> <timestamp> [score_cfg]

要点:
  - base 配置 configs/optimizerConfig_tornado_common.yaml;
  - 通用参数扰动矩阵: 1 基线 + 20 扰动 = 21 组 (全部 score_cfg 覆写);
  - 分片专属 maxent 路径 + 关 GPU(走 CPU 推理), 与 SA/GA 扰动口径一致;
  - 仓库根由脚本位置推导; 临时 cfg 写 /tmp。
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

# 用法: python run_tornado_common_shard.py <shard_tag> <idx_csv> <timestamp> [score_cfg]
shard_tag = sys.argv[1]
idxs = [int(x) for x in sys.argv[2].split(",")]
ts = sys.argv[3]
score_cfg = sys.argv[4] if len(sys.argv) > 4 else os.path.join("configs", "score_cfg.yaml")

# 每分片独立 maxent 路径 + 关 GPU(走 CPU 推理), 与 SA/GA 扰动实验保持一致口径
base = open("configs/optimizerConfig_tornado_common.yaml", encoding="utf-8").read()
base = base.replace("_{year}_tor_cm.ela", "_{year}_tor_cm_%s.ela" % shard_tag)
base = base.replace("use_gpu: true", "use_gpu: false")
out_cfg = "/tmp/opt_tornado_cm_%s.yaml" % shard_tag
open(out_cfg, "w", encoding="utf-8").write(base)

# 通用参数扰动矩阵: 1 基线 + 20 扰动 = 21 组 (全部 score_cfg 覆写)
tu._TORNADO_PARAMS = [
    ('tgt_function.lambda_ES_high', 1000, [500, 1500, 2000], 'score_cfg'),
    ('tgt_function.lambda_ES_low', 5000, [2500, 7500, 10000], 'score_cfg'),
    ('tgt_function.lambda_LS_high', 5, [20, 50, 100], 'score_cfg'),
    ('tgt_function.lambda_LS_low', 1, [2, 5, 10], 'score_cfg'),
    ('change_score.inside_reward_coef', 10, [5, 15, 20], 'score_cfg'),
    ('landscape.pure_base', 0.3, [0.1, 0.5], 'score_cfg'),
    ('landscape.pure_change', 0.2, [0.05, 0.1, 0.4], 'score_cfg'),
]

print("=== 启动 通用参数扰动 shard=%s idxs=%s ts=%s score=%s ===" %
      (shard_tag, idxs, ts, score_cfg), flush=True)
root = tornado_pipeline(
    opt_cfg_path=out_cfg,
    score_cfg_path=score_cfg,
    idxs=idxs,
    mode="tornado",
    n_repeats=int(os.environ.get("TOR_REPEATS", 3)),
    is_processed=True,
    processed_timestamp=ts,
)
print("=== TORNADO CM shard %s DONE: %s ===" % (shard_tag, root), flush=True)
