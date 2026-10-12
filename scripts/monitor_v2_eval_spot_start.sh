#!/bin/bash
# 后台常驻：只看护当前 submissions.json 里的 v2 immune_v6 官方闲时评测。
# 默认看护 20260928 no-ESMC 150000 的 16 条。要看护旧 32 条：
#   bash scripts/monitor_v2_eval_spot_start.sh 20260927
# 不碰训练 t-20260923185956-wsvpq / t-20260924185250-w22zf。
#
# 两层保护：
#   1. PID 文件互斥 —— 重复启动会把同一终态评测重提两次。
#   2. wrapper 自愈 —— python 非 0 退出则 10 秒后拉起；exit 0
#      （全局 STOP 或全部 Success）才收工。
set -uo pipefail

REPO=/vepfs-mlp2/c20250601/251105016/project/dllm_test
EVAL_DATE="${1:-20260928}"
SUBMISSIONS="$REPO/output/downstream_generation/v2_immune_v6_eval_${EVAL_DATE}/submissions.json"
MON_DIR="$REPO/output/downstream_generation/v2_immune_v6_eval_${EVAL_DATE}/_monitor"
PID_FILE="$MON_DIR/monitor.pid"
OUT_LOG="$MON_DIR/nohup.out"

mkdir -p "$MON_DIR"

if [ ! -f "$SUBMISSIONS" ]; then
    echo "找不到 $SUBMISSIONS" >&2
    exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    echo "评测看护已在运行 pid=$(cat "$PID_FILE") date=$EVAL_DATE，未重复启动。"
    echo "要重启： bash scripts/monitor_v2_eval_spot_stop.sh $EVAL_DATE && bash scripts/monitor_v2_eval_spot_start.sh $EVAL_DATE"
    echo "注意：stop 只停看护，不会 cancel 评测或训练作业。"
    exit 0
fi

if [ -f "$MON_DIR/STOP" ]; then
    echo "清除上次留下的全局 STOP 哨兵（保留 STOP.<TaskName|task_id>）"
    rm -f "$MON_DIR/STOP"
fi

setsid nohup bash -c '
    REPO="$1"; MON_DIR="$2"; PID_FILE="$3"; SUBMISSIONS="$4"
    echo $$ > "$PID_FILE"
    cd "$REPO" || exit 1
    export V2_EVAL_SUBMISSIONS="$SUBMISSIONS"
    export V2_EVAL_MONITOR_DIR="$MON_DIR"
    while true; do
        /vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python scripts/monitor_v2_eval_spot.py
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "$(date -u +%FT%TZ) [INFO] 评测看护正常退出（全局 STOP 或全部 Success）" >> "$MON_DIR/monitor.log"
            break
        fi
        echo "$(date -u +%FT%TZ) [ERROR] 评测看护异常退出 rc=$rc，10 秒后自动拉起" >> "$MON_DIR/monitor.log"
        sleep 10
    done
    rm -f "$PID_FILE"
' _ "$REPO" "$MON_DIR" "$PID_FILE" "$SUBMISSIONS" >> "$OUT_LOG" 2>&1 &

sleep 3
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    echo "评测看护已启动 pid=$(cat "$PID_FILE") date=$EVAL_DATE"
    echo "  台账：      $SUBMISSIONS"
    echo "  日志：      $MON_DIR/monitor.log"
    echo "  状态：      $MON_DIR/state.json"
    echo "  停止看护：  bash scripts/monitor_v2_eval_spot_stop.sh $EVAL_DATE"
    echo "  人工 cancel 评测前务必先 touch 哨兵，否则 Killed 会被重提："
    echo "      touch $MON_DIR/STOP"
    echo "      touch $MON_DIR/STOP.<TaskName>"
    echo "      touch $MON_DIR/STOP.<task_id>"
    echo "  训练作业不在看护范围内，不要对它们放哨兵/cancel。"
else
    echo "启动失败，看 $OUT_LOG" >&2
    exit 1
fi
