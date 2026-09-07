#!/bin/bash
# 后台常驻启动闲时任务看护（脱离终端，关掉 shell 也继续跑）。
#
# 两层保护：
#   1. PID 文件互斥 —— 重复启动会让同一个终态任务被重提两次，所以先查有没有活着的实例。
#   2. wrapper 自愈 —— python 若因未捕获异常非 0 退出，10 秒后自动拉起；exit 0
#      （STOP 哨兵或全部 Success）才算正常收工，wrapper 一起退出。
set -uo pipefail

REPO=/vepfs-mlp2/c20250601/251105016/project/dllm_test
MON_DIR="$REPO/output/_monitor"
PID_FILE="$MON_DIR/monitor.pid"
OUT_LOG="$MON_DIR/nohup.out"

mkdir -p "$MON_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    echo "看护已在运行 pid=$(cat "$PID_FILE")，未重复启动。"
    echo "要重启： bash scripts/monitor_spot_tasks_stop.sh && bash scripts/monitor_spot_tasks_start.sh"
    exit 0
fi

# 上一次可能是被 STOP 哨兵停的；不清掉的话新进程会立刻退出。
if [ -f "$MON_DIR/STOP" ]; then
    echo "清除上次留下的 STOP 哨兵"
    rm -f "$MON_DIR/STOP"
fi

setsid nohup bash -c '
    REPO="$1"; MON_DIR="$2"; PID_FILE="$3"
    echo $$ > "$PID_FILE"
    cd "$REPO" || exit 1
    while true; do
        python3 scripts/monitor_spot_tasks.py
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "$(date -u +%FT%TZ) [INFO] 看护正常退出（STOP 哨兵或全部 Success）" >> "$MON_DIR/monitor.log"
            break
        fi
        echo "$(date -u +%FT%TZ) [ERROR] 看护异常退出 rc=$rc，10 秒后自动拉起" >> "$MON_DIR/monitor.log"
        sleep 10
    done
    rm -f "$PID_FILE"
' _ "$REPO" "$MON_DIR" "$PID_FILE" >> "$OUT_LOG" 2>&1 &

sleep 3
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    echo "看护已启动 pid=$(cat "$PID_FILE")"
    echo "  日志：      $MON_DIR/monitor.log"
    echo "  状态：      $MON_DIR/state.json"
    echo "  停止：      bash scripts/monitor_spot_tasks_stop.sh"
    echo "  人工 cancel 前务必先 touch $MON_DIR/STOP，否则任务会被自动重提回来"
else
    echo "启动失败，看 $OUT_LOG" >&2
    exit 1
fi
