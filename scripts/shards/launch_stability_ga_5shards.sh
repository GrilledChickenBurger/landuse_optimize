#!/bin/bash
# 启动 5 个 GA 稳定性分片, 共用一个 timestamp 根目录 (results/ablation/<ts>/, mode=stability)
# 注意: 与 SA 稳定性必须使用不同 timestamp, 否则同一样本会相互覆盖
# 仓库根目录由本脚本位置推导, 可在任意当前目录下执行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1
TS=$(date +%Y%m%d_%H%M%S)
echo "GA stability 5-shard launch, TS=$TS"
for i in 0 1 2 3 4; do
  nohup ./.venv/bin/python -u scripts/shards/run_stability_ga_shard.py "$i" "$TS" > "stab_ga_shard${i}.log" 2>&1 &
  echo "launched shard $i -> stab_ga_shard${i}.log pid $!"
done
echo "all launched; GA stability TS=$TS"
