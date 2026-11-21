# -*- coding: utf-8 -*-
"""全量对照: 60 样本 x 4 情景, 同一进程内依次跑 lambda x200 与 x20 两档 (收敛自 40108 run_full_lambda.py).

用法: nohup ./.venv/bin/python -u scripts/shards/run_full_lambda.py <ga|sa> <shard 0-29> <ts_x200> <ts_x20> > full_<algo>_s<id>.log 2>&1 &
固定 data.year="15", valid_sample_prob=0.8; 强制 use_gpu=False (GPU 只能容纳 ~10 片, 会 OOM).
临时 opt cfg 写 /tmp, 不落仓库。
"""
import os, sys, time, random
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT); sys.path.append(ROOT)
import yaml
BASE_SCORE = yaml.safe_load(open(os.path.join("configs", "score_cfg.yaml"), encoding="utf-8"))
LEVELS = {
    "x200": {"lambda_ES_low": 1000000, "lambda_ES_high": 200000, "lambda_LS_low": 200, "lambda_LS_high": 1000},
    "x20":  {"lambda_ES_low": 100000,  "lambda_ES_high": 20000,  "lambda_LS_low": 20,  "lambda_LS_high": 100},
}
import main  # noqa
import utils.config_utils as cu
import pipelines.run_runner as rr
from pipelines.run_runner import pipeline
rr.train_maxent = lambda *a, **k: None
IDXS = [1006,1936,1875,1137,2323,2267,1359,639,1822,640,
        1539,848,2439,2635,2148,630,1262,2625,2083,391,
        2010,600,2499,1480,1479,2502,1345,857,1698,1857,
        829,530,2024,2149,655,654,2018,1900,1594,824,
        1096,598,1533,2387,1512,1962,660,2391,1650,531,
        1158,1724,1963,2142,1030,2265,1575,1714,1854,1896]
PER_SHARD = 2
BASE_CFG = {"ga": "configs/optimizerConfig_ga.yaml", "sa": "configs/optimizerConfig_sa.yaml"}
MAXENT = "checkpoints/maxent-model_sklearn_15.ela"
YEAR = "15"
VALID_SAMPLE_PROB = 0.8
USE_GPU = False

def build_score(lam):
    sc = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE_SCORE.items()}
    sc["tgt_function"] = dict(BASE_SCORE["tgt_function"])
    sc["tgt_function"].update(lam)
    return sc

def run(algo, shard, ts_map):
    idxs = IDXS[shard*PER_SHARD: shard*PER_SHARD+PER_SHARD]
    print("[full] algo=%s shard=%d idxs=%s ts=%s" % (algo, shard, idxs, ts_map), flush=True)
    for lvl, lam in LEVELS.items():
        cu._DEFAULT_SCORE_CFG_DICT = build_score(lam)
        with open("/tmp/_score_cfg_full_%s.yaml" % lvl, "w", encoding="utf-8") as f:
            yaml.safe_dump(cu._DEFAULT_SCORE_CFG_DICT, f, allow_unicode=True)
        cfg = yaml.safe_load(open(BASE_CFG[algo], encoding="utf-8"))
        cfg["data"]["year"] = YEAR
        cfg["data"]["maxent_model_path"] = MAXENT
        cfg["device"]["use_gpu"] = USE_GPU
        cfg["runner"]["common_params"]["maxent_model_path"] = MAXENT
        cfg["runner"]["common_params"]["valid_sample_prob"] = VALID_SAMPLE_PROB
        tmp = "/tmp/opt_%s_full_%s_s%d.yaml" % (algo, lvl, shard)
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        print("[full] %s %s lambda=%s year=%s use_gpu=%s vsp=%s ts=%s"
              % (algo, lvl, lam, cfg["data"]["year"], cfg["device"]["use_gpu"],
                 cfg["runner"]["common_params"]["valid_sample_prob"], ts_map[lvl]), flush=True)
        time.sleep(random.uniform(0, 15))
        pipeline(tmp, idxs, True, ts_map[lvl])
        print("[full] DONE %s %s ts=%s" % (algo, lvl, ts_map[lvl]), flush=True)

if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), {"x200": sys.argv[3], "x20": sys.argv[4]})
