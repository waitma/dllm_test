#!/bin/bash
# 在 tmux 里跑本地长任务（预处理、语料重建、本地评测）。断连/关终端都不中断。
#
# 用法：
#   bash scripts/run_in_tmux.sh <任务名> <命令> [参数...]
#
# 例（重跑 v6 prepared shards）：
#   bash scripts/run_in_tmux.sh prep_v6 \
#       /vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python \
#       scripts/data/preprocess_immune_dataset.py \
#       --config configs/data/immune_v6_binding_only.yaml \
#       --output-dir data/prepared/immune_v6_binding_only --overwrite
#
# 看状态：
#   tmux attach -t <任务名>              # 进去看，Ctrl-b d 脱离（不要 Ctrl-c）
#   tail -f logs/<任务名>/run.log
#   ls logs/<任务名>/                    # started / done / failed 哨兵 + exit_code
#
# 提供五件事：
#   1. tmux 托管：断连不中断；pane 设 remain-on-exit，跑完仍在，退出码可回看。
#   2. flock 单实例互斥：同名任务不会并发跑两遍。预处理并发会互相覆盖 shard。
#   3. 哨兵文件 started/done/failed + exit_code：轮询状态不必解析日志。
#   4. 行缓冲（stdbuf -oL -eL）：Python 重定向到文件默认是块缓冲，不解开的话
#      日志在跑完前一直是空的，无法判断是在算还是卡死（2026-09-19 踩过，
#      当时只能靠 /proc/<pid>/fd 看它在读哪个文件才确认没卡）。
#   5. TMPDIR 兜到 vepfs 并断言存在：/ 是 overlay（20G，长期 95% 满），
#      临时文件落上去会把根盘写爆；且 TMPDIR 指向不存在的目录时 Python 的
#      tempfile 会静默回退 /tmp，不报错（2026-09-19 踩过，当时传了
#      .../251105016/tmp，该目录并不存在）。所以这里必须 mkdir -p 后再断言。
set -uo pipefail

REPO=/vepfs-mlp2/c20250601/251105016/project/dllm_test

if [ "$#" -lt 2 ]; then
    echo "用法: bash scripts/run_in_tmux.sh <任务名> <命令> [参数...]" >&2
    exit 2
fi

JOB="$1"; shift
case "$JOB" in
    *[!A-Za-z0-9_.-]*)
        echo "任务名只允许字母数字和 _ . -（会用作 tmux 会话名和目录名）: $JOB" >&2
        exit 2
        ;;
esac

LOG_DIR="$REPO/logs/$JOB"
LOG="$LOG_DIR/run.log"
LOCK="$LOG_DIR/lock"
mkdir -p "$LOG_DIR"

# tmux 会话名撞了就直接停手：另一个同名任务可能正在写同一批产物。
if tmux has-session -t "=$JOB" 2>/dev/null; then
    echo "tmux 会话 '$JOB' 已存在，未启动。" >&2
    echo "  查看：  tmux attach -t $JOB" >&2
    echo "  结束后：tmux kill-session -t $JOB" >&2
    exit 1
fi

# 上一轮的哨兵留着会让人误读状态，启动前清掉（日志用 >> 追加，保留历史）。
rm -f "$LOG_DIR/started" "$LOG_DIR/done" "$LOG_DIR/failed" "$LOG_DIR/exit_code"

# 把命令安全地拼成单个字符串：tmux 只收字符串，%q 保证带空格/引号的参数不被拆开。
CMD_STR="$(printf '%q ' "$@")"

# 真实的 TMPDIR（不是 /vepfs-mlp2/c20250601/251105016/tmp，那个不存在）。
RUN_TMPDIR="${TMPDIR:-/vepfs-mlp2/c20250601/251105016/conda/cache/tmp}"
mkdir -p "$RUN_TMPDIR"
if [ ! -d "$RUN_TMPDIR" ]; then
    echo "TMPDIR 不存在且无法创建: $RUN_TMPDIR" >&2
    exit 1
fi

# 内层脚本。flock 在这里拿锁而不是外层，锁要活到命令结束为止。
INNER=$(cat <<'INNER_EOF'
set -uo pipefail
cd "$REPO" || exit 1
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "$(date -u +%FT%TZ) [ABORT] 另一个 $JOB 实例持有锁，未启动" | tee -a "$LOG"
    exit 1
fi
export PYTHONPATH="$REPO"
export TMPDIR="$RUN_TMPDIR"
echo "$(date -u +%FT%TZ) START $JOB" | tee -a "$LOG"
echo "  cmd:    $CMD_STR" | tee -a "$LOG"
echo "  TMPDIR: $TMPDIR" | tee -a "$LOG"
date -u +%FT%TZ > "$LOG_DIR/started"
# stdbuf 解块缓冲，日志才能实时看；退出码要拿 PIPESTATUS，tee 会盖掉 $?。
set -o pipefail
stdbuf -oL -eL bash -c "$CMD_STR" 2>&1 | tee -a "$LOG"
rc="${PIPESTATUS[0]}"
echo "$rc" > "$LOG_DIR/exit_code"
if [ "$rc" -eq 0 ]; then
    date -u +%FT%TZ > "$LOG_DIR/done"
    echo "$(date -u +%FT%TZ) DONE $JOB rc=0" | tee -a "$LOG"
else
    date -u +%FT%TZ > "$LOG_DIR/failed"
    echo "$(date -u +%FT%TZ) FAILED $JOB rc=$rc" | tee -a "$LOG"
fi
INNER_EOF
)

tmux new-session -d -s "$JOB" \
    -e REPO="$REPO" \
    -e JOB="$JOB" \
    -e LOG="$LOG" \
    -e LOG_DIR="$LOG_DIR" \
    -e LOCK="$LOCK" \
    -e CMD_STR="$CMD_STR" \
    -e RUN_TMPDIR="$RUN_TMPDIR" \
    bash -c "$INNER"

# 跑完保留 pane，退出码能回看；否则会话直接消失，只剩日志。
tmux set-option -t "$JOB" remain-on-exit on

sleep 2
if tmux has-session -t "=$JOB" 2>/dev/null; then
    echo "已在 tmux 启动: $JOB"
    echo "  日志：  $LOG"
    echo "  进去看：tmux attach -t $JOB   （脱离是 Ctrl-b d，不要 Ctrl-c）"
    echo "  状态：  ls $LOG_DIR"
else
    echo "启动失败，看 $LOG" >&2
    exit 1
fi
