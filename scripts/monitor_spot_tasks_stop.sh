#!/bin/bash
# 停掉闲时任务看护循环。
#
# 注意顺序：先放 STOP 哨兵再杀进程。哨兵有两个作用 —— 让 python 下一轮自行退出，
# 以及万一 wrapper 抢在被杀之前重新拉起 python，新进程也会立刻看到哨兵而退出。
#
# 这个脚本只停看护，不动训练任务本身。若接下来要人工 cancel 训练任务，
# STOP 哨兵会一直留在那里，重新启动看护时 start 脚本会清掉它。
set -uo pipefail

REPO=/vepfs-mlp2/c20250601/251105016/project/dllm_test
MON_DIR="$REPO/output/_monitor"
PID_FILE="$MON_DIR/monitor.pid"

mkdir -p "$MON_DIR"
touch "$MON_DIR/STOP"
echo "已放置 STOP 哨兵：$MON_DIR/STOP"

if [ -f "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        echo "已终止 wrapper pid=$pid"
    fi
    rm -f "$PID_FILE"
fi

pkill -f "monitor_spot_tasks.py" 2>/dev/null && echo "已终止看护 python 进程" || true

sleep 2
if pgrep -f "[m]onitor_spot_tasks.py" >/dev/null; then
    echo "警告：仍有看护进程存活，请手动检查 pgrep -af monitor_spot_tasks" >&2
    exit 1
fi
echo "看护已停止。重新启动： bash scripts/monitor_spot_tasks_start.sh"
