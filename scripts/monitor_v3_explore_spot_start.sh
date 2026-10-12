#!/bin/bash
# 后台常驻：看护 queue012 no-ESMC v3 explore 目录全部 YAML（含后续提交）。
# 同一 PID 文件互斥，禁止第二份 daemon。不碰 ESMC / 旧 1M / fusion。
set -uo pipefail

REPO=/vepfs-mlp2/c20250601/251105016/project/dllm_test
MON_DIR="$REPO/output/immune_v3_explore_queue012/_monitor"
PID_FILE="$MON_DIR/monitor.pid"
OUT_LOG="$MON_DIR/nohup.out"

mkdir -p "$MON_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    echo "v3 explore 看护已在运行 pid=$(cat "$PID_FILE")，未重复启动。"
    echo "要重启： bash scripts/monitor_v3_explore_spot_stop.sh && bash scripts/monitor_v3_explore_spot_start.sh"
    echo "注意：stop 只停看护，不会 cancel 作业。"
    exit 0
fi

if [ -f "$MON_DIR/STOP" ]; then
    echo "清除上次留下的全局 STOP 哨兵（保留 STOP.<key|task_id>）"
    rm -f "$MON_DIR/STOP"
fi

setsid nohup bash -c '
    REPO="$1"; MON_DIR="$2"; PID_FILE="$3"
    echo $$ > "$PID_FILE"
    cd "$REPO" || exit 1
    export V3_EXPLORE_MONITOR_DIR="$MON_DIR"
    while true; do
        python3 scripts/monitor_v3_explore_spot.py
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "$(date -u +%FT%TZ) [INFO] v3 explore 看护正常退出（全局 STOP）" >> "$MON_DIR/monitor.log"
            break
        fi
        echo "$(date -u +%FT%TZ) [ERROR] v3 explore 看护异常退出 rc=$rc，10 秒后自动拉起" >> "$MON_DIR/monitor.log"
        sleep 10
    done
    rm -f "$PID_FILE"
' _ "$REPO" "$MON_DIR" "$PID_FILE" >> "$OUT_LOG" 2>&1 &

sleep 3
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    echo "v3 explore 看护已启动 pid=$(cat "$PID_FILE")"
    echo "  台账：      $MON_DIR/ledger.json"
    echo "  监视列表：  $MON_DIR/watch_list.json"
    echo "  日志：      $MON_DIR/monitor.log"
    echo "  停止看护：  bash scripts/monitor_v3_explore_spot_stop.sh"
    echo "  人工 cancel 作业前务必先 touch 哨兵，否则 Killed 会被重提："
    echo "      touch $MON_DIR/STOP"
    echo "      touch $MON_DIR/STOP.01_synthetic_ready.A0"
    echo "      touch $MON_DIR/STOP.<task_id>"
else
    echo "启动失败，看 $OUT_LOG" >&2
    exit 1
fi
