#!/bin/bash
# 只停 v2 immune_v6 评测看护循环，不动评测作业、更不动两条训练。
# 默认停 20260928 看护：bash scripts/monitor_v2_eval_spot_stop.sh
# 停旧 32 条看护：bash scripts/monitor_v2_eval_spot_stop.sh 20260927
set -uo pipefail

REPO=/project_c20250601/251105016/project/dllm_test
EVAL_DATE="${1:-20260928}"
MON_DIR="$REPO/output/downstream_generation/v2_immune_v6_eval_${EVAL_DATE}/_monitor"
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

# 只杀带对应 MONITOR_DIR 的 python；没有环境变量的旧进程一并清掉。
pkill -f "V2_EVAL_MONITOR_DIR=$MON_DIR" 2>/dev/null || true
if [ "$EVAL_DATE" = "20260928" ]; then
    pkill -f "scripts/monitor_v2_eval_spot.py" 2>/dev/null && echo "已终止评测看护 python" || true
fi

sleep 2
if pgrep -f "[m]onitor_v2_eval_spot.py" >/dev/null; then
    echo "警告：仍有评测看护进程存活，请手动检查 pgrep -af monitor_v2_eval_spot" >&2
    exit 1
fi
echo "评测看护已停止。重新启动： bash scripts/monitor_v2_eval_spot_start.sh $EVAL_DATE"
echo "本脚本没有 cancel 任何评测或训练任务。"
