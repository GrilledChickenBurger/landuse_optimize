#!/bin/bash
# 启动 6 个 SA 分片, 共用一个 timestamp 根目录 (results/SA/<ts>/), 与 GA 分片模式一致
# 仓库根目录由本脚本位置推导, 可在任意当前目录下执行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1
TS=$(date +%Y%m%d_%H%M%S)
echo "SA 6-shard launch, TS=$TS"
for i in 0 1 2 3 4 5; do
  nohup ./.venv/bin/python -u scripts/shards/run_sa_shard.py "$i" "$TS" > "sa_shard${i}.log" 2>&1 &
  echo "launched shard $i -> sa_shard${i}.log pid $!"
done
echo "all launched"
