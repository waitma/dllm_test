#!/bin/bash
# 只停 no-ESMC v3 explore 看护循环，不 cancel 任何作业。
set -uo pipefail

REPO=/vepfs-mlp2/c20250601/251105016/project/dllm_test
MON_DIR="$REPO/output/immune_v3_explore_queue012/_monitor"
PID_FILE="$MON_DIR/monitor.pid"

mkdir -p "$MON_DIR"
touch "$MON_DIR/STOP"
echo "已放置全局 STOP 哨兵：$MON_DIR/STOP"

if [ -f "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        echo "已终止 wrapper pid=$pid"
    fi
    rm -f "$PID_FILE"
fi

pkill -f "V3_EXPLORE_MONITOR_DIR=$MON_DIR" 2>/dev/null || true
# 只杀看护 python 进程本身，避免误伤含脚本路径的其它命令行。
pkill -f "[p]ython3 scripts/monitor_v3_explore_spot.py" 2>/dev/null && echo "已终止 v3 explore 看护 python" || true

sleep 2
if pgrep -f "[p]ython3 scripts/monitor_v3_explore_spot.py" >/dev/null; then
    echo "警告：仍有 v3 explore 看护进程存活，请手动检查 pgrep -af monitor_v3_explore_spot" >&2
    exit 1
fi
echo "v3 explore 看护已停止。重新启动： bash scripts/monitor_v3_explore_spot_start.sh"
echo "本脚本没有 cancel 任何任务。"
