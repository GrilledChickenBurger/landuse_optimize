#!/bin/bash
# 通用分片启动器: 逐行执行 tasks 文件中的命令, 并发闸门 max_parallel.
#
# 用法: bash scripts/shards/launch_pool.sh <tasks_file> <max_parallel>
#
# tasks_file 每行 = 一条完整命令 (不含 nohup/重定向), 例如:
#   ./.venv/bin/python -u scripts/shards/run_ablation_shard.py a1875 A 1875 20261014_120000 configs/_score_cfg_full_x200.yaml
#
# 日志写到 <tasks_file 同级>/logs_<ts>/task<NN>.log
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

TASKS="${1:?usage: launch_pool.sh <tasks_file> <max_parallel>}"
P="${2:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$(dirname "$TASKS")/logs_${TS}"
mkdir -p "$LOG_DIR"

i=0
while IFS= read -r cmd || [ -n "$cmd" ]; do
  cmd="$(echo "$cmd" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  i=$((i + 1))
  printf -v n "%02d" "$i"
  echo "[launch_pool] task ${n}: ${cmd}"
  nohup bash -c "$cmd" > "$LOG_DIR/task${n}.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do wait -n; done
done < "$TASKS"

wait
echo "[launch_pool] ALL DONE ($i tasks) logs=$LOG_DIR"
