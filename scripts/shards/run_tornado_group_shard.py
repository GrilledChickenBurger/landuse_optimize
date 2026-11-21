# -*- coding: utf-8 -*-
"""tornado 参数扰动 —— 按「组」分片执行器.

用法:
  python run_tornado_group_shard.py <sa|ga|cm> <sample_idx|all> <group_name> <timestamp> [score_cfg]

每片 = 1 个实验组 × n_repeats(TOR_REPEATS):
  - sample_idx = 具体样本号 -> 只跑该样本 × 该组;
  - sample_idx = all        -> 跑 Set Y 全部 10 样本 × 该组 (组级分片).

配置口径复用 run_tornado_{sa,ga,common}_shard.py(分片专属 maxent 路径 / 关 GPU /
GA 基线覆盖 / CM_LAMBDA_SCALE 缩放); 不改变任何既有代码或结果格式.
is_processed=True 时按 run_{r}/<样本>/convergence.json 自动跳过已完成的 run,
因此同样的时间戳 + TOR_REPEATS=3 只补跑缺失的 run_1 / run_2.
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
from pipelines.tornado import TornadoPipeline  # noqa: E402

# 用法: python run_tornado_group_shard.py <sa|ga|cm> <sample_idx|all> <group_name> <timestamp> [score_cfg]
family = sys.argv[1]
sample_arg = sys.argv[2]
group = sys.argv[3]
ts = sys.argv[4]
score_cfg = sys.argv[5] if len(sys.argv) > 5 else os.path.join("configs", "score_cfg.yaml")

# 10 样本 (= Set Y, 与稳定性/消融/通用扰动一致)
IDXS = [1875, 639, 2439, 1833, 1956, 857, 2025, 1909, 531, 1506]
if sample_arg == "all":                 # 组级分片: 1 组 × 全部 10 样本
    idxs = list(IDXS)
    stok = "all"
else:
    sample = int(sample_arg)
    if sample not in IDXS:
        raise SystemExit("sample %d 不在 Set Y 中" % sample)
    idxs = [sample]
    stok = str(sample)

n_repeats = int(os.environ.get("TOR_REPEATS", 3))
safe = group.replace("=", "_").replace(".", "_")
tag = "%s_%s_%s" % (family, stok, safe)

# ---- 按族构造 cfg / 扰动矩阵 (与既有分片脚本逐字对齐) ----
if family == "sa":
    base = open("configs/optimizerConfig_stability.yaml", encoding="utf-8").read()
    # 每分片唯一 maxent 路径 (train_maxent 每次都重训覆盖, 唯一化避免并发写冲突;
    # 训练无随机种子, 各进程模型一致)
    base = base.replace("_{year}_stab.ela", "_{year}_sa_tor10_%s_%s.ela" % (stok, safe))
    base = base.replace("use_gpu: true", "use_gpu: false")
    out_cfg = "/tmp/opt_torngrp_%s.yaml" % tag
    open(out_cfg, "w", encoding="utf-8").write(base)
    tu._TORNADO_PARAMS = [
        ('cooling_rate', 0.9995, [0.995, 0.998, 0.9998], 'opt_cfg'),
        ('max_beststuck_iter', 3000, [1000, 6000], 'opt_cfg'),
        ('max_noimprove_iter', 1000, [500, 2000, 5000], 'opt_cfg'),
        ('T_start', 100, [80, 200], 'opt_cfg'),
    ]
elif family == "ga":
    import yaml
    with open("configs/optimizerConfig_stability_ga.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["device"]["use_gpu"] = False
    cfg["GA"]["ori_pop_size"] = 20
    cfg["GA"]["crossover_rate"] = 0.5
    cfg["GA"]["mutation_rate"] = 0.5
    cfg["GA"]["max_noimprove_gen"] = 100
    mp = "checkpoints/maxent-model_sklearn_15_tor_ga10_%s_%s.ela" % (stok, safe)
    cfg["data"]["maxent_model_path"] = mp
    cfg["runner"]["common_params"]["maxent_model_path"] = mp
    out_cfg = "/tmp/opt_torngrp_%s.yaml" % tag
    with open(out_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    tu._TORNADO_PARAMS = [
        ('ori_pop_size', 20, [10, 30], 'opt_cfg'),
        ('crossover_rate', 0.5, [0.3, 0.7], 'opt_cfg'),
        ('mutation_rate', 0.5, [0.3, 0.7], 'opt_cfg'),
        ('max_noimprove_gen', 100, [50, 150], 'opt_cfg'),
    ]
elif family == "cm":
    _SC = float(os.environ.get("CM_LAMBDA_SCALE", "1"))

    def _sc(v):
        return int(round(v * _SC))

    base = open("configs/optimizerConfig_tornado_common.yaml", encoding="utf-8").read()
    base = base.replace("_{year}_tor_cm.ela", "_{year}_tor_cm_cm%s_%s.ela" % (stok, safe))
    base = base.replace("use_gpu: true", "use_gpu: false")
    out_cfg = "/tmp/opt_torngrp_%s.yaml" % tag
    open(out_cfg, "w", encoding="utf-8").write(base)
    tu._TORNADO_PARAMS = [
        ('tgt_function.lambda_ES_high', _sc(1000), [_sc(500), _sc(1500), _sc(2000)], 'score_cfg'),
        ('tgt_function.lambda_ES_low', _sc(5000), [_sc(2500), _sc(7500), _sc(10000)], 'score_cfg'),
        ('tgt_function.lambda_LS_high', _sc(5), [_sc(20), _sc(50), _sc(100)], 'score_cfg'),
        ('tgt_function.lambda_LS_low', _sc(1), [_sc(2), _sc(5), _sc(10)], 'score_cfg'),
        ('change_score.inside_reward_coef', 10, [5, 15, 20], 'score_cfg'),
        ('landscape.pure_base', 0.3, [0.1, 0.5], 'score_cfg'),
        ('landscape.pure_change', 0.2, [0.05, 0.1, 0.4], 'score_cfg'),
    ]
else:
    raise SystemExit("family 必须是 sa / ga / cm")


class GroupPipeline(TornadoPipeline):
    """只跑单个实验组的 tornado 流水线 (多分片共享同一时间戳目录)."""

    def __init__(self, only_group=None, **kwargs):
        self._only = only_group
        super().__init__(**kwargs)

    def build_matrix(self):
        matrix = super().build_matrix()
        picked = [e for e in matrix if e['name'] == self._only]
        if not picked:
            raise SystemExit("组名 %r 不在矩阵中" % self._only)
        return picked

    def _write_summary(self, matrix, extra_fields=None):
        # 多分片共享同一 ts 目录, 空实现以避免互相覆盖根目录 experiment_summary.json
        return


print("=== 启动 tornado 组级分片 family=%s sample_or_all=%s group=%s ts=%s repeats=%d ===" %
      (family, stok, group, ts, n_repeats), flush=True)
pipeline = GroupPipeline(
    opt_cfg_path=out_cfg,
    score_cfg_path=score_cfg,
    idxs=idxs,
    mode="tornado",
    n_repeats=n_repeats,
    is_processed=True,
    processed_timestamp=ts,
    only_group=group,
)
root = pipeline.run()
print("=== TORNADO GROUP shard %s DONE: %s ===" % (tag, root), flush=True)
