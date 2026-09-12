#!/usr/bin/env python3
"""闲时（抢占）训练任务看护循环：轮询状态，掉了就重提，重提后自动从最新 checkpoint 续跑。

为什么需要它：平台的 `RetryOptions`（YAML 里已配 InstanceReclaimed + MaxRetryTimes=50）
覆盖的是「实例被回收后自动重试」。但重试次数用尽、或任务因别的原因落到 Failed/Killed
终态之后，平台就不再管了，训练会永久停在那里。本脚本就是补这一层：只要「最新的同名任务」
进入终态异常，就用同一个 YAML 重新提交；YAML 的 entrypoint 里有 RESUME_ARG 逻辑，会自动
挑 OUTPUT_DIR 下编号最大的 checkpoint 续训，所以重提 == 从 last ckpt 接着跑。

三个必须知道的行为：

1. **人工 cancel 前务必先 `touch` STOP 哨兵**，否则脚本会把你 cancel 掉的任务重新提交回来。
       touch output/_monitor/STOP                     # 停掉整个循环
       touch output/_monitor/STOP.<TaskName>          # 只停某一条
2. **只看最新一条同名任务。** `volc ml_task list --name` 是模糊匹配且会返回历史终态任务
   （例如 2026-08-29 cancel 重提留下的 t-20260829135133-5np6m/Killed）。若不取最新，
   脚本会被历史记录骗到、无限重提。
3. **停滞只告警、不自动 kill。** Running 但 wandb 长时间没更新可能是 eval 在跑（每 1000 步
   一次、七源各 2000 行），误杀代价比漏报大，所以只写 WARN 到日志等人来看。

用法：
    python3 scripts/monitor_spot_tasks.py                # 前台跑（调试用）
    python3 scripts/monitor_spot_tasks.py --once         # 只跑一轮后退出（自检用）
    python3 scripts/monitor_spot_tasks.py --once --dry-run  # 一轮 + 不真提交
    bash scripts/monitor_spot_tasks_start.sh             # 后台常驻（实际用这个）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = "/vepfs-mlp2/c20250601/251105016/project/dllm_test"

# (TaskName, YAML 相对路径)。TaskName 必须与 YAML 里的 TaskName 完全一致 —— 脚本靠它精确比对。
TARGETS = [
    (
        "protein_esmc_llada270m_diffusion_immune_v3_spot_2m",
        "train_jobs/protein_esmc_llada270m_diffusion_immune_v3_spot_2m.yml",
    ),
    (
        "protein_esmc_llada270m_bert_immune_v5_8gpu_spot",
        "train_jobs/protein_esmc_llada270m_bert_immune_v5_8gpu_spot.yml",
    ),
]

POLL_INTERVAL = 600          # 轮询间隔，10 分钟
RESUBMIT_COOLDOWN = 900      # 两次重提之间的最小间隔，防止任务秒失败时刷一堆提交
MAX_RESUBMITS = 100          # 单个任务累计重提上限，超了就只告警等人介入
STALL_WARN_SECONDS = 3600    # Running 但 wandb 超过这么久没更新就告警

# volc 的状态枚举（见 `volc ml_task list --help`）：终态只有这三个，其余都还在流转中。
TERMINAL_OK = {"Success"}
TERMINAL_BAD = {"Failed", "Killed"}

MONITOR_DIR = os.path.join(REPO, "output", "_monitor")
STATE_PATH = os.path.join(MONITOR_DIR, "state.json")
LOG_PATH = os.path.join(MONITOR_DIR, "monitor.log")
STOP_PATH = os.path.join(MONITOR_DIR, "STOP")

# volc CLI 必须绕开代理，否则请求挂住（本仓库既有经验）。
PROXY_VARS = [
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY",
]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def log(level: str, msg: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z [{level}] {msg}"
    print(line, flush=True)
    os.makedirs(MONITOR_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def volc_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
    return env


def run_volc(args: list[str], timeout: int = 120) -> str:
    proc = subprocess.run(
        ["volc"] + args,
        cwd=REPO, env=volc_env(), timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return ANSI_RE.sub("", proc.stdout.decode("utf-8", "replace"))


def list_tasks(name: str) -> list[dict]:
    """取该名字下的全部任务（含终态），按 JobId 降序 —— JobId 形如 t-<YYYYMMDDHHMMSS>-xxxxx，
    时间戳在前，字符串降序即时间倒序。`--name` 是模糊匹配，故这里精确比对 JobName。"""
    out = run_volc([
        "ml_task", "list", "-n", name,
        "-s", "Queue,Staging,Running,Killing,Success,Failed,Killed,Initialized",
        "-o", "json", "--limit", "50",
    ])
    start = out.find("[")
    if start < 0:
        raise RuntimeError(f"volc list 未返回 JSON 数组: {out[:300]!r}")
    items = json.loads(out[start:])
    exact = [it for it in items if it.get("JobName") == name]
    return sorted(exact, key=lambda it: it.get("JobId", ""), reverse=True)


def submit(yaml_rel: str) -> str:
    out = run_volc(["ml_task", "submit", "--conf", yaml_rel], timeout=300)
    m = re.search(r"task_id=(\S+)", out)
    if not m:
        raise RuntimeError(f"提交未返回 task_id: {out[-500:]!r}")
    return m.group(1).strip()


def output_dir_of(yaml_rel: str) -> str | None:
    """从 YAML 的 entrypoint 里抠 OUTPUT_DIR=，避免依赖 pyyaml（本脚本刻意零第三方依赖）。"""
    path = os.path.join(REPO, yaml_rel)
    try:
        with open(path, encoding="utf-8") as fh:
            m = re.search(r"^\s*OUTPUT_DIR=(\S+)\s*$", fh.read(), re.M)
        return m.group(1) if m else None
    except OSError:
        return None


def latest_checkpoint(out_dir: str | None) -> str | None:
    """OUTPUT_DIR 下编号最大的 checkpoint-N（与 entrypoint 里 RESUME_ARG 的挑法一致：
    按数字序而非字典序，且排除 checkpoint-final）。"""
    if not out_dir or not os.path.isdir(out_dir):
        return None
    steps = []
    for entry in os.listdir(out_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", entry)
        if m:
            steps.append(int(m.group(1)))
    return f"checkpoint-{max(steps)}" if steps else None


def wandb_last_mtime(out_dir: str | None) -> float | None:
    """最近一次 wandb 写入时间。offline run 每 logging_steps=20 步就落盘，比 checkpoint
    （每 1000 步）灵敏得多，用它判断训练是否还在推进。"""
    if not out_dir:
        return None
    root = os.path.join(out_dir, "wandb")
    newest = None
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".wandb") or fn == "output.log":
                try:
                    mt = os.path.getmtime(os.path.join(dirpath, fn))
                except OSError:
                    continue
                if newest is None or mt > newest:
                    newest = mt
    return newest


def load_state() -> dict:
    try:
        with open(STATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    os.makedirs(MONITOR_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)


def check_one(name: str, yaml_rel: str, state: dict, dry_run: bool) -> None:
    st = state.setdefault(name, {"resubmits": 0, "last_resubmit_ts": 0, "history": []})

    if os.path.exists(f"{STOP_PATH}.{name}"):
        log("INFO", f"{name}: 存在 STOP.{name} 哨兵，跳过（人工接管中）")
        return
    if st.get("done"):
        return

    try:
        tasks = list_tasks(name)
    except Exception as exc:                                  # 网络抖动/CLI 异常都不该弄死循环
        log("ERROR", f"{name}: 查询失败（下一轮重试）: {exc}")
        return

    if not tasks:
        log("WARN", f"{name}: 未查到任何同名任务。不自动提交，请人工确认后手动 submit")
        return

    latest = tasks[0]
    job_id, status = latest.get("JobId"), latest.get("Status")
    out_dir = output_dir_of(yaml_rel)

    if job_id != st.get("job_id"):
        log("INFO", f"{name}: 追踪到任务 {job_id}（状态 {status}）")
        st["job_id"] = job_id
    st["status"] = status

    if status in TERMINAL_OK:
        log("INFO", f"{name}: {job_id} 已 {status}，训练完成，停止看护")
        st["done"] = True
        return

    if status not in TERMINAL_BAD:
        # Queue / Staging / Running / Killing / Initialized：还在流转，只在 Running 时查停滞
        detail = f"{name}: {job_id} {status}"
        if status == "Running":
            mt = wandb_last_mtime(out_dir)
            ckpt = latest_checkpoint(out_dir)
            if mt is None:
                detail += "，尚无 wandb 写入（刚起跑时正常：数据加载约需 5 分钟）"
            else:
                idle = time.time() - mt
                detail += f"，wandb {idle / 60:.0f} 分钟前更新，最新 ckpt={ckpt or '无'}"
                if idle > STALL_WARN_SECONDS:
                    log("WARN", f"{name}: {job_id} Running 但 wandb 已 {idle / 60:.0f} 分钟"
                                f" 未更新，疑似停滞（eval 阶段也会拉长间隔，故不自动处理，请人工看）")
        log("INFO", detail)
        return

    # 到这里 = 终态异常（Failed / Killed），需要重提
    ckpt = latest_checkpoint(out_dir)
    log("WARN", f"{name}: {job_id} 落到终态 {status}（End={latest.get('End')}），"
                f"最新 checkpoint={ckpt or '无 → 将从零开始'}")

    if st["resubmits"] >= MAX_RESUBMITS:
        log("ERROR", f"{name}: 累计重提已达上限 {MAX_RESUBMITS}，不再自动重提，请人工介入")
        return

    since = time.time() - st.get("last_resubmit_ts", 0)
    if since < RESUBMIT_COOLDOWN:
        log("INFO", f"{name}: 距上次重提仅 {since / 60:.1f} 分钟（冷却 {RESUBMIT_COOLDOWN / 60:.0f} 分钟），本轮跳过")
        return

    if dry_run:
        log("INFO", f"{name}: [dry-run] 本应重提 {yaml_rel}（会从 {ckpt or 'scratch'} 续跑）")
        return

    try:
        new_id = submit(yaml_rel)
    except Exception as exc:
        log("ERROR", f"{name}: 重提失败（下一轮重试）: {exc}")
        return

    st["resubmits"] += 1
    st["last_resubmit_ts"] = time.time()
    st["job_id"] = new_id
    st["history"].append({
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "dead_job": job_id, "dead_status": status,
        "resume_from": ckpt, "new_job": new_id,
    })
    log("INFO", f"{name}: 已重提 → {new_id}（第 {st['resubmits']} 次，从 {ckpt or 'scratch'} 续跑）")


def main() -> int:
    ap = argparse.ArgumentParser(description="闲时训练任务看护循环")
    ap.add_argument("--once", action="store_true", help="只跑一轮后退出（自检用）")
    ap.add_argument("--dry-run", action="store_true", help="检测到终态也不真提交，只记日志")
    ap.add_argument("--interval", type=int, default=POLL_INTERVAL, help=f"轮询间隔秒数（默认 {POLL_INTERVAL}）")
    args = ap.parse_args()

    os.makedirs(MONITOR_DIR, exist_ok=True)
    log("INFO", f"看护启动 pid={os.getpid()} 间隔={args.interval}s "
                f"dry_run={args.dry_run} 目标={[n for n, _ in TARGETS]}")
    log("INFO", f"人工 cancel 前请先 touch {STOP_PATH} 或 {STOP_PATH}.<TaskName>，否则会被自动重提")

    while True:
        if os.path.exists(STOP_PATH):
            log("INFO", f"检测到 STOP 哨兵（{STOP_PATH}），退出看护")
            return 0

        state = load_state()
        for name, yaml_rel in TARGETS:
            check_one(name, yaml_rel, state, args.dry_run)
        save_state(state)

        if all(state.get(n, {}).get("done") for n, _ in TARGETS):
            log("INFO", "所有目标任务均已 Success，退出看护")
            return 0
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("INFO", "收到中断，退出看护")
        sys.exit(130)
