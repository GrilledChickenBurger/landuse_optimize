#!/bin/bash
# 启动 6 个 GA 分片, 共用一个 timestamp 根目录 (results/GA/<ts>/)
# 仓库根目录由本脚本位置推导, 可在任意当前目录下执行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1
TS=$(date +%Y%m%d_%H%M%S)
echo "GA 6-shard launch, TS=$TS"
for i in 0 1 2 3 4 5; do
  nohup ./.venv/bin/python -u scripts/shards/run_ga_shard.py "$i" "$TS" > "ga_shard${i}.log" 2>&1 &
  echo "launched shard $i -> ga_shard${i}.log pid $!"
done
echo "all launched"
