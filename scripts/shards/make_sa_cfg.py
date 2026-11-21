# -*- coding: utf-8 -*-
"""由 configs/optimizerConfig_ga.yaml 生成 SA 专用配置 configs/optimizerConfig_sa.yaml.

SA 关键设置: type=SA, scenario=all, T_start=100, T_end=50, chain_length=10, valid_sample_prob=0.8.
用法(仓库根目录): ./.venv/bin/python -u scripts/shards/make_sa_cfg.py
"""
import os
import sys

# 仓库根目录 = 本脚本目录的上两级 (scripts/shards/ -> scripts/ -> repo)
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
sys.path.append(ROOT)

import yaml  # noqa: E402

src = os.path.join("configs", "optimizerConfig_ga.yaml")
dst = os.path.join("configs", "optimizerConfig_sa.yaml")

with open(src, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["runner"]["type"] = "SA"
cfg["runner"]["scenario"] = "all"
cfg["runner"]["common_params"]["valid_sample_prob"] = 0.8

mp = "checkpoints/maxent-model_sklearn_{year}_sa.ela"
cfg["data"]["maxent_model_path"] = mp
cfg["runner"]["common_params"]["maxent_model_path"] = mp

cfg["SA"]["T_start"] = 100
cfg["SA"]["T_end"] = 50            # 5-7 min/run (T_start=100 时约 7.3 min)
cfg["SA"]["cooling_rate"] = 0.9995
cfg["SA"]["chain_length"] = 10     # GA 配置的 SA 段缺此项, 不补会退化为 chain_length=1

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)

print("written:", dst)
print("runner.type =", cfg["runner"]["type"], "| scenario =", cfg["runner"]["scenario"])
print("SA =", {k: cfg["SA"][k] for k in
               ["T_start", "T_end", "cooling_rate", "chain_length",
                "max_noimprove_iter", "max_beststuck_iter", "save_path"]})
print("valid_sample_prob =", cfg["runner"]["common_params"]["valid_sample_prob"])
print("maxent template =", mp)
print("model =", cfg["model"]["type"], cfg["model"]["model_path"], "| year =", cfg["data"]["year"])
