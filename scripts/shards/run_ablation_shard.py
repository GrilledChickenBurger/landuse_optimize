# -*- coding: utf-8 -*-
"""消融 A/B/C 的「样本 x 组」分片执行器 (收敛自 18307 /tmp/run_ablation_g.py).

用法:
  python run_ablation_shard.py <tag> <groups_csv> <idx_csv> <timestamp> [score_cfg]

示例:
  python run_ablation_shard.py a1875 A 1875 20261014_120000
  python run_ablation_shard.py s1b1833 B 1833 20261014_120000
  # λ×200 消融 (用第 5 个参数指定 score_cfg):
  python run_ablation_shard.py a1875 A 1875 20261014_120000 configs/_score_cfg_full_x200.yaml

要点:
  1) 支持只跑指定组(A/B/C 子集) -> 可把 (样本 x 组) lane 分到多台机器;
  2) 统一 maxent: 禁止分片内重新训练(背景点无种子随机, 会导致各分片模型不同),
     改为复用预置的同一份模型副本, 保证跨 lane / 跨机器可比;
  3) 线程封顶 / 关 GPU / 独立分片配置; 仓库根由脚本位置推导(不再硬编码),
     临时 cfg 写 /tmp, 不落到仓库。
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

import pipelines.experiment_base as ebase  # noqa: E402
import pipelines.ablation as abl  # noqa: E402

# 用法: <tag> <groups_csv> <idx_csv> <timestamp> [score_cfg]
tag = sys.argv[1]
groups = set(g for g in sys.argv[2].split(",") if g)
idxs = [int(x) for x in sys.argv[3].split(",") if x]
ts = sys.argv[4]
score_cfg = sys.argv[5] if len(sys.argv) > 5 else os.path.join("configs", "score_cfg.yaml")
N_REPEATS = int(os.environ.get("ABL_REPEATS", 3))

# ---- 1) 统一 maxent: 禁止分片内训练(复用预置的同一模型副本) ----
ebase.train_maxent = lambda *a, **k: None

# ---- 2) 组子集: 只保留指定组 ----
_orig_matrix = abl.build_experiment_matrix


def _patched_matrix(mode='ablation', n_repeats=1):
    return [e for e in _orig_matrix(mode, n_repeats) if e.get('group') in groups]


abl.build_experiment_matrix = _patched_matrix

# ---- 3) 独立分片配置: 分片专属 maxent 路径 + 关 GPU(走 CPU 推理) ----
base = open("configs/optimizerConfig_ablation.yaml", encoding="utf-8").read()
base = base.replace("_{year}_ablation.ela", "_{year}_ablation_%s.ela" % tag)
base = base.replace("use_gpu: true", "use_gpu: false")
out_cfg = "/tmp/opt_ablation_%s.yaml" % tag
open(out_cfg, "w", encoding="utf-8").write(base)

print("=== 启动 SA 消融 shard=%s groups=%s idxs=%s ts=%s repeats=%d score=%s ===" %
      (tag, sorted(groups), idxs, ts, N_REPEATS, score_cfg), flush=True)
root = abl.ablation_pipeline(
    opt_cfg_path=out_cfg,
    score_cfg_path=score_cfg,
    idxs=idxs,
    mode="ablation",
    n_repeats=N_REPEATS,
    is_processed=True,
    processed_timestamp=ts,
)
print("=== ABLATION shard %s DONE: %s ===" % (tag, root), flush=True)
