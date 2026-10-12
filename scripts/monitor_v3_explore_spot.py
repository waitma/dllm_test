#!/usr/bin/env python3
"""看护 queue012 no-ESMC v3 explore（本目录全部 YAML，含后续提交）。

盯 train_jobs/immune_v3_explore_queue012/ 下所有 .yml（01_synthetic_ready、
02_real_data_gated、以及日后同目录新增）。task_id 每轮从 README ID 表、
PROJECT_PROCESS.md、ledger.json、YAML TaskName、submission_logs 发现。

不主动提交尚未提交的实验。Success 不重提；Running/Queue 不重复提交。
Killed 用同一 YAML 重提。Failed 默认重提，除非日志是明确代码 bug。
绝不提交 ESMC / fusion / 旧 1M。不 cancel 任何作业。
全部 Success 后继续运行，等待后续提交被发现。

人工 cancel 前必须先放 STOP 哨兵，否则 Killed 会被重提：
    touch output/immune_v3_explore_queue012/_monitor/STOP
    touch output/immune_v3_explore_queue012/_monitor/STOP.01_synthetic_ready.A3
    touch output/immune_v3_explore_queue012/_monitor/STOP.A3
    touch output/immune_v3_explore_queue012/_monitor/STOP.<task_id>
    touch output/immune_v3_explore_queue012/_monitor/STOP.<TaskName>

用法：
    python3 scripts/monitor_v3_explore_spot.py            # 前台
    python3 scripts/monitor_v3_explore_spot.py --once     # 只跑一轮
    bash scripts/monitor_v3_explore_spot_start.sh         # 后台（PID 互斥，禁止第二份）
    bash scripts/monitor_v3_explore_spot_stop.sh          # 只停看护
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
from pathlib import Path

REPO = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
EXPLORE_ROOT = REPO / "train_jobs/immune_v3_explore_queue012"
README = EXPLORE_ROOT / "README.md"
PROJECT_PROCESS = REPO / "PROJECT_PROCESS.md"
SUBMISSION_LOGS = REPO / "output/immune_v3_explore_queue012/submission_logs"
MONITOR_DIR = Path(
    os.environ.get(
        "V3_EXPLORE_MONITOR_DIR",
        str(REPO / "output/immune_v3_explore_queue012/_monitor"),
    )
)
LEDGER = MONITOR_DIR / "ledger.json"
STATE_PATH = MONITOR_DIR / "state.json"
LOG_PATH = MONITOR_DIR / "monitor.log"
STOP_PATH = MONITOR_DIR / "STOP"
FIRST_TICK_PATH = MONITOR_DIR / "first_tick.json"
EXPAND_TICK_PATH = MONITOR_DIR / "expand_v2_tick.json"
WATCH_LIST_PATH = MONITOR_DIR / "watch_list.json"
VOLC = REPO / "scripts/volc-no-proxy.sh"

PROTECTED = {
    "t-20260923185956-wsvpq",
    "t-20260924185250-w22zf",
    "t-20260929112158-4zp2q",
}
FORBIDDEN_SUBSTR = ("protein_esmc", "esmc_llada", "/esmc", "fusion")
EXPLORE_MARKERS = (
    "immune_v3_explore_queue012",
    "01_synthetic_ready",
    "02_real_data_gated",
    "immune-v3-a",
    "synthetic20",
    "gated-pilot10",
)

POLL_INTERVAL = 90
RESUBMIT_COOLDOWN = 180
MAX_RESUBMITS = 20

IN_FLIGHT = {"Queue", "Staging", "Running", "Initialized", "Killing"}
TERMINAL_OK = {"Success"}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
TASK_ID_RE = re.compile(r"t-\d{14}-[a-z0-9]+")
TASKNAME_RE = re.compile(r"^TaskName:\s*(.+)$", re.M)
README_TABLE_START = "## 看护当前 ID（自动更新）"
README_TABLE_END = "<!-- end v3-explore-monitor-table -->"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def log(level: str, msg: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z [{level}] {msg}"
    print(line, flush=True)
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, data) -> None:
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def run_volc(args: list[str], timeout: int = 180) -> str:
    proc = subprocess.run(
        ["bash", str(VOLC), *args],
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return ANSI_RE.sub("", proc.stdout.decode("utf-8", "replace"))


def parse_json_blob(text: str):
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [i for i in (start_obj, start_arr) if i >= 0]
    if not starts:
        raise RuntimeError(f"volc 未返回 JSON: {text[:400]!r}")
    return json.loads(text[min(starts) :])


def yaml_forbidden(path: str | Path) -> bool:
    low = str(path).lower()
    return any(s in low for s in FORBIDDEN_SUBSTR)


def parse_task_name(yaml_path: Path) -> str:
    text = yaml_path.read_text(encoding="utf-8")
    m = TASKNAME_RE.search(text)
    return m.group(1).strip() if m else yaml_path.stem


def catalog_yaml() -> list[dict]:
    """Every non-ESMC YAML currently under the explore tree."""
    items = []
    if not EXPLORE_ROOT.is_dir():
        return items
    for yml in sorted(EXPLORE_ROOT.rglob("*.yml")):
        if yaml_forbidden(yml):
            log("WARN", f"跳过禁止 YAML: {yml}")
            continue
        rel = yml.relative_to(EXPLORE_ROOT).as_posix()
        axis = yml.stem.upper()
        parent = "/".join(Path(rel).parts[:-1])
        key = f"{parent}.{axis}" if parent else axis
        items.append(
            {
                "key": key,
                "axis": axis,
                "group": rel.split("/", 1)[0],
                "yaml": str(yml.resolve()),
                "yaml_rel": rel,
                "task_name": parse_task_name(yml),
            }
        )
    return items


def infer_key(job: dict) -> str:
    if job.get("key"):
        return job["key"]
    yaml = job.get("yaml") or ""
    axis = (job.get("axis") or "").upper()
    if not axis:
        axis = Path(yaml).stem.upper() or "UNK"
    if "02_real_data_gated" in yaml:
        return f"02_real_data_gated.{axis}"
    if "01_synthetic_ready" in yaml:
        return f"01_synthetic_ready.{axis}"
    return f"unknown.{axis}"


def migrate_state(state: dict) -> dict:
    out = dict(state)
    for old in list(state):
        if re.fullmatch(r"A[0-4]", old):
            new = f"01_synthetic_ready.{old}"
            if new not in out:
                out[new] = dict(state[old])
    return out


def save_jobs(jobs: list[dict], extra: dict | None = None) -> None:
    payload = {
        "ts": utc_now(),
        "schema": 2,
        "jobs": jobs,
        "never_submit": ["ESMC", "fusion", "old 1M", "unsubmitted YAML (until an ID is discovered)"],
        "auto_discover": [
            "README ID table",
            "PROJECT_PROCESS.md",
            "ledger.json",
            "YAML TaskName",
            "submission_logs/",
        ],
        "protected": sorted(PROTECTED),
    }
    if extra:
        payload.update(extra)
    atomic_json(LEDGER, payload)


def get_task(task_id: str, cache: dict) -> dict:
    if task_id in cache:
        return cache[task_id]
    if task_id in PROTECTED:
        raise RuntimeError(f"拒绝把受保护训练 ID 当探索目标: {task_id}")
    out = run_volc(["ml_task", "get", "-i", task_id, "-o", "json"], timeout=120)
    data = parse_json_blob(out)
    if isinstance(data, list):
        data = data[0] if data else {}
    info = {
        "id": data.get("Id") or data.get("JobId") or task_id,
        "name": data.get("Name") or data.get("JobName") or "",
        "status": data.get("Status") or "",
        "start": data.get("Start") or data.get("LaunchTime") or "",
        "end": data.get("End") or data.get("EndTime") or "",
    }
    cache[task_id] = info
    return info


def fetch_logs(task_id: str, lines: int = 80) -> str:
    return run_volc(
        ["ml_task", "logs", "-t", task_id, "-i", "worker-0", "-l", str(lines)],
        timeout=90,
    )


def code_bug_reason(logs: str) -> str | None:
    """Failed 且日志是明确代码异常时跳过重提。OOM / 无 Traceback 不当成代码 bug。"""
    if "Traceback" not in logs:
        return None
    if "OutOfMemoryError" in logs or "CUDA out of memory" in logs:
        return None
    last = ""
    for line in logs.splitlines():
        s = line.strip()
        if re.match(r"^[A-Za-z][A-Za-z0-9_.]*(Error|Exception|Error:)", s) or s.startswith(
            "AssertionError"
        ):
            last = s[:240]
    if last:
        return last
    if "SyntaxError" in logs or "ModuleNotFoundError" in logs or "ImportError" in logs:
        return "import/syntax error in logs"
    return "Traceback without OOM"


def submit_yaml(yaml_path: str) -> str:
    yp = Path(yaml_path).resolve()
    root = EXPLORE_ROOT.resolve()
    if yaml_forbidden(yp):
        raise RuntimeError(f"拒绝提交禁止 YAML: {yp}")
    try:
        yp.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"只允许 explore 目录 YAML: {yp}") from exc
    out = run_volc(["ml_task", "submit", "--conf", str(yp), "-o", "json"], timeout=300)
    data = None
    try:
        data = parse_json_blob(out)
    except Exception:
        data = None
    new_id = None
    if isinstance(data, dict):
        new_id = data.get("Id") or data.get("JobId") or data.get("id")
    if not new_id:
        match = TASK_ID_RE.search(out)
        new_id = match.group(0) if match else None
    if not new_id:
        raise RuntimeError(f"提交未返回 task_id: {out[-800:]!r}")
    new_id = str(new_id).strip()
    if new_id in PROTECTED:
        raise RuntimeError(f"提交结果撞上受保护 ID: {new_id}")
    return new_id


def sentinel_hit(job: dict, current_id: str) -> str | None:
    names = {
        STOP_PATH,
        MONITOR_DIR / f"STOP.{job.get('key', '')}",
        MONITOR_DIR / f"STOP.{job.get('axis', '')}",
        MONITOR_DIR / f"STOP.{job.get('task_name', '')}",
        MONITOR_DIR / f"STOP.{current_id}",
        MONITOR_DIR / f"STOP.{job.get('original_task_id', '')}",
        MONITOR_DIR / f"STOP.{job.get('task_id', '')}",
    }
    for hist in job.get("history") or []:
        dead = hist.get("dead_job")
        if dead:
            names.add(MONITOR_DIR / f"STOP.{dead}")
    for path in names:
        if path and str(path).endswith("STOP."):
            continue
        if path.exists():
            return str(path)
    return None


def collect_text_ids(text: str, catalog: list[dict]) -> tuple[dict[str, set[str]], set[str]]:
    """Map context-bound IDs to keys; also return unbound IDs on explore-related lines."""
    by_key: dict[str, set[str]] = {}
    unbound: set[str] = set()
    yaml_map = {}
    names = {}
    for c in catalog:
        yaml_map[c["yaml_rel"]] = c["key"]
        yaml_map[f"train_jobs/immune_v3_explore_queue012/{c['yaml_rel']}"] = c["key"]
        yaml_map[c["yaml"]] = c["key"]
        names[c["task_name"]] = c["key"]
        names[c["key"]] = c["key"]
    for line in text.splitlines():
        ids = TASK_ID_RE.findall(line)
        if not ids:
            continue
        key = None
        for yrel, k in yaml_map.items():
            if yrel in line:
                key = k
                break
        if key is None:
            for name, k in names.items():
                if name and name in line:
                    key = k
                    break
        if key:
            by_key.setdefault(key, set()).update(ids)
        else:
            unbound.update(ids)
    return by_key, unbound


def explore_related(line: str) -> bool:
    return any(m in line for m in EXPLORE_MARKERS)


def bind_unbound_ids(
    unbound: set[str], catalog: list[dict], cache: dict
) -> dict[str, set[str]]:
    """YAML TaskName match: get() each candidate and bind if Name matches a catalog TaskName."""
    extra: dict[str, set[str]] = {}
    names = {c["task_name"]: c["key"] for c in catalog}
    for tid in sorted(unbound):
        if tid in PROTECTED:
            continue
        try:
            info = get_task(tid, cache)
        except Exception as exc:
            log("WARN", f"发现 ID {tid} 查询失败: {exc}")
            continue
        name = info.get("name") or ""
        if name in names:
            extra.setdefault(names[name], set()).add(tid)
        else:
            log("INFO", f"忽略非 explore TaskName: {tid} name={name!r}")
    return extra


def discovered_ids(catalog: list[dict], ledger_jobs: list[dict], cache: dict) -> dict[str, set[str]]:
    by_key: dict[str, set[str]] = {}

    def add(key: str, tid: str | None) -> None:
        if key and tid and TASK_ID_RE.fullmatch(tid):
            by_key.setdefault(key, set()).add(tid)

    for job in ledger_jobs:
        key = infer_key(job)
        add(key, job.get("task_id"))
        add(key, job.get("original_task_id"))
        for hist in job.get("history") or []:
            add(key, hist.get("dead_job"))
            add(key, hist.get("new_job"))

    texts = []
    if README.is_file():
        texts.append(("README", README.read_text(encoding="utf-8")))
    if PROJECT_PROCESS.is_file():
        pp = PROJECT_PROCESS.read_text(encoding="utf-8")
        pp_keep = "\n".join(ln for ln in pp.splitlines() if explore_related(ln))
        texts.append(("PROJECT_PROCESS", pp_keep))
    if SUBMISSION_LOGS.is_dir():
        blob = []
        for p in sorted(SUBMISSION_LOGS.glob("*")):
            if p.is_file() and p.stat().st_size < 2_000_000:
                try:
                    blob.append(p.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    pass
        texts.append(("submission_logs", "\n".join(blob)))

    unbound: set[str] = set()
    for label, text in texts:
        bound, extra_unbound = collect_text_ids(text, catalog)
        for key, ids in bound.items():
            by_key.setdefault(key, set()).update(ids)
        if label == "README":
            # README 全文 ID 都拿去用 TaskName 核对（体积小）。
            unbound.update(TASK_ID_RE.findall(text))
        else:
            unbound.update(extra_unbound)

    # Drop IDs already bound to any key.
    already = {tid for ids in by_key.values() for tid in ids}
    unbound -= already
    extra = bind_unbound_ids(unbound, catalog, cache)
    for key, ids in extra.items():
        by_key.setdefault(key, set()).update(ids)
    return by_key


def pick_watch_id(ids: set[str], cache: dict) -> tuple[str | None, str, dict | None]:
    if not ids:
        return None, "unsubmitted", None
    infos = []
    for tid in ids:
        if tid in PROTECTED:
            continue
        try:
            infos.append(get_task(tid, cache))
        except Exception as exc:
            log("WARN", f"get {tid} 失败: {exc}")
    if not infos:
        return None, "unsubmitted", None
    success = [i for i in infos if i["status"] in TERMINAL_OK]
    if success:
        success.sort(key=lambda x: x.get("end") or x.get("start") or "", reverse=True)
        return success[0]["id"], "success", success[0]
    inflight = [i for i in infos if i["status"] in IN_FLIGHT]
    if inflight:
        inflight.sort(key=lambda x: x.get("start") or "", reverse=True)
        return inflight[0]["id"], "in_flight", inflight[0]
    dead = [i for i in infos if i["status"] in {"Failed", "Killed"}]
    if dead:
        dead.sort(key=lambda x: x.get("end") or x.get("start") or "", reverse=True)
        return dead[0]["id"], "dead", dead[0]
    infos.sort(key=lambda x: x.get("start") or "", reverse=True)
    return infos[0]["id"], "other", infos[0]


def merge_jobs(catalog: list[dict], ledger_jobs: list[dict], found: dict[str, set[str]], cache: dict) -> list[dict]:
    by_key = {}
    for job in ledger_jobs:
        key = infer_key(job)
        job = dict(job)
        job["key"] = key
        by_key[key] = job
    merged = []
    for spec in catalog:
        job = by_key.get(spec["key"], {})
        job.update(
            {
                "key": spec["key"],
                "axis": spec["axis"],
                "group": spec["group"],
                "yaml": spec["yaml"],
                "yaml_rel": spec["yaml_rel"],
                "task_name": spec["task_name"],
            }
        )
        job.setdefault("history", [])
        job.setdefault("resubmit_count", 0)
        ids = set(found.get(spec["key"]) or [])
        if job.get("task_id"):
            ids.add(job["task_id"])
        chosen, reason, info = pick_watch_id(ids, cache)
        if chosen:
            if not job.get("original_task_id"):
                job["original_task_id"] = chosen
            if job.get("task_id") and job["task_id"] != chosen and info:
                # 发现更新的在途/成功 ID，切过去，避免对着旧 Killed 再提一份。
                log(
                    "INFO",
                    f"{spec['key']}: 采用发现的 ID {chosen} ({info.get('status')}) "
                    f"（原 ledger {job.get('task_id')}）",
                )
            job["task_id"] = chosen
            if info:
                job["status"] = info.get("status") or job.get("status")
                if info.get("name"):
                    job["task_name"] = info["name"]
        else:
            job["task_id"] = job.get("task_id") or None
            job["status"] = "unsubmitted"
            job["_pick"] = reason
        job["_pick"] = reason
        merged.append(job)
    return merged


def insert_dated_note(text: str, title: str, bullets: list[str]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    note = f"## {stamp} {title}\n\n" + "".join(f"- {b}\n" for b in bullets) + "\n"
    if text.startswith("# Project Process"):
        rest = text.split("\n", 1)[1].lstrip("\n")
        return f"# Project Process\n\n{note}{rest}"
    return note + text


def patch_active_table(text: str, jobs: list[dict], statuses: dict[str, str]) -> str:
    lines = text.splitlines(keepends=True)
    out = []
    skip_ids = set()
    new_rows = []
    for job in jobs:
        tid = job.get("task_id")
        if not tid:
            continue
        st = statuses.get(tid, job.get("status") or "")
        skip_ids.add(tid)
        skip_ids.add(job.get("original_task_id") or "")
        for hist in job.get("history") or []:
            if hist.get("dead_job"):
                skip_ids.add(hist["dead_job"])
            if hist.get("new_job"):
                skip_ids.add(hist["new_job"])
        if st in TERMINAL_OK or st in {"Failed", "Killed", "unsubmitted", "STOP"}:
            continue
        if st not in IN_FLIGHT and st:
            continue
        mark = f"{st}（看护 {utc_now()}）" if st else "Queue"
        steps = "pilot10" if "gated" in (job.get("group") or "") else "20"
        new_rows.append(
            f"| `{tid}` | `{job['task_name']}` | `queue012` 闲时 | 1 | {steps} | {mark} |\n"
        )
    inserted = False
    for line in lines:
        if line.startswith("| `t-") and any(f"`{sid}`" in line for sid in skip_ids if sid):
            if not inserted:
                out.extend(new_rows)
                inserted = True
            continue
        if line.startswith("Last updated:") and "看护" in line:
            line = (
                f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
                f"（queue012 v3 explore 看护）\n"
            )
        out.append(line)
    if not inserted and new_rows:
        rebuilt = []
        for line in out:
            rebuilt.append(line)
            if line.startswith("|---|") and any("Task ID" in x for x in rebuilt[-3:]):
                rebuilt.extend(new_rows)
                inserted = True
        out = rebuilt
    return "".join(out)


def patch_project_process(jobs: list[dict], statuses: dict[str, str], title: str | None, bullets: list[str] | None) -> None:
    text = PROJECT_PROCESS.read_text(encoding="utf-8")
    text = patch_active_table(text, jobs, statuses)
    if title and bullets:
        text = insert_dated_note(text, title, bullets)
    atomic_write(PROJECT_PROCESS, text)


def patch_readme(jobs: list[dict], statuses: dict[str, str]) -> None:
    if not README.is_file():
        return
    text = README.read_text(encoding="utf-8")
    rows = [
        "| key | YAML | task_id | Status |",
        "|---|---|---|---|",
    ]
    for job in jobs:
        tid = job.get("task_id")
        st = statuses.get(tid or "", job.get("status") or "unsubmitted")
        tid_cell = f"`{tid}`" if tid else "—"
        rows.append(
            f"| `{job['key']}` | `{job.get('yaml_rel') or job['yaml']}` | {tid_cell} | {st or '?'} |"
        )
    block = (
        f"{README_TABLE_START}\n\n"
        "看护：`bash scripts/monitor_v3_explore_spot_start.sh` / "
        "`bash scripts/monitor_v3_explore_spot_stop.sh`。"
        "盯本目录全部 YAML（含 `02_real_data_gated` 与日后新增）。"
        f"人工 cancel 前 `touch {STOP_PATH}` 或 `STOP.<key|axis|task_id|TaskName>`。\n\n"
        + "\n".join(rows)
        + f"\n\n{README_TABLE_END}\n\n"
    )
    if README_TABLE_START in text and README_TABLE_END in text:
        head = text.split(README_TABLE_START, 1)[0].rstrip() + "\n\n"
        tail = text.split(README_TABLE_END, 1)[1].lstrip("\n")
        text = head + block + tail
    elif README_TABLE_START in text:
        head = text.split(README_TABLE_START, 1)[0].rstrip() + "\n\n"
        rest = text.split(README_TABLE_START, 1)[1]
        nxt = re.search(r"\n## (?!看护当前 ID)", rest)
        tail = rest[nxt.start() + 1 :] if nxt else ""
        text = head + block + tail
    else:
        insert_at = text.find("## 合成 20-step")
        if insert_at < 0:
            insert_at = text.find("## 现在可以提交")
        if insert_at >= 0:
            text = text[:insert_at] + block + text[insert_at:]
        else:
            text = text.rstrip() + "\n\n" + block
    atomic_write(README, text)


def check_one(job: dict, state: dict, cache: dict) -> dict:
    key = job["key"]
    current_id = job.get("task_id")
    st = state.setdefault(
        key,
        {"resubmits": 0, "last_resubmit_ts": 0.0, "done": False, "skip_code_bug": False},
    )
    result = {
        "key": key,
        "axis": job.get("axis"),
        "task_id": current_id,
        "status": job.get("status") or "unsubmitted",
        "resubmitted": False,
        "skipped": None,
    }
    hit = sentinel_hit(job, current_id or "")
    if hit:
        result["status"] = "STOP"
        result["skipped"] = hit
        log("INFO", f"{key}: 哨兵 {hit}，不重提")
        return result
    if not current_id:
        result["status"] = "unsubmitted"
        result["skipped"] = "no_task_id_yet"
        log("INFO", f"{key}: 尚未发现 task_id，不主动提交")
        return result
    if st.get("skip_code_bug"):
        result["status"] = "Failed"
        result["skipped"] = f"code_bug:{st.get('code_bug')}"
        return result

    info = get_task(current_id, cache)
    status = info["status"]
    result["status"] = status
    result["name"] = info["name"]
    job["status"] = status
    log("INFO", f"{key}: {current_id} {status}")

    if status in TERMINAL_OK:
        st["done"] = True
        result["skipped"] = "success_no_resubmit"
        return result
    if status in IN_FLIGHT:
        result["skipped"] = "in_flight"
        return result
    if status not in {"Failed", "Killed"}:
        result["skipped"] = f"unhandled:{status}"
        log("WARN", f"{key}: 未处理状态 {status}")
        return result

    if status == "Failed":
        try:
            logs = fetch_logs(current_id)
        except Exception as exc:
            logs = f"(logs failed: {exc})"
        bug = code_bug_reason(logs)
        if bug:
            st["skip_code_bug"] = True
            st["code_bug"] = bug
            result["skipped"] = f"code_bug:{bug}"
            log("ERROR", f"{key}: Failed 判定为代码 bug，不重提：{bug}")
            return result

    if st["resubmits"] >= MAX_RESUBMITS:
        result["skipped"] = "max_resubmits"
        log("ERROR", f"{key}: 已达 MAX_RESUBMITS={MAX_RESUBMITS}，停止重提")
        return result
    elapsed = time.time() - float(st.get("last_resubmit_ts") or 0)
    if st["resubmits"] and elapsed < RESUBMIT_COOLDOWN:
        result["skipped"] = "cooldown"
        log("INFO", f"{key}: cooldown {int(RESUBMIT_COOLDOWN - elapsed)}s")
        return result

    new_id = submit_yaml(job["yaml"])
    hist = {
        "ts": utc_now(),
        "dead_job": current_id,
        "dead_status": status,
        "new_job": new_id,
    }
    job.setdefault("history", []).append(hist)
    job["task_id"] = new_id
    job["status"] = "Queue"
    job["resubmit_count"] = int(job.get("resubmit_count") or 0) + 1
    st["resubmits"] += 1
    st["last_resubmit_ts"] = time.time()
    st["done"] = False
    result.update(
        resubmitted=True,
        new_id=new_id,
        status="Queue",
        dead_job=current_id,
        reason=status,
        task_id=new_id,
    )
    log("INFO", f"{key}: {status} {current_id} → 重提 {new_id}（第 {st['resubmits']} 次）")
    return result


def tick(write_expand_note: bool) -> list[dict]:
    cache: dict = {}
    catalog = catalog_yaml()
    ledger = load_json(LEDGER, {})
    ledger_jobs = ledger.get("jobs") if isinstance(ledger, dict) else []
    if not isinstance(ledger_jobs, list):
        ledger_jobs = []
    found = discovered_ids(catalog, ledger_jobs, cache)
    jobs = merge_jobs(catalog, ledger_jobs, found, cache)
    state = migrate_state(load_json(STATE_PATH, {}))
    results = []
    resubmits = []
    statuses: dict[str, str] = {}
    for job in jobs:
        result = check_one(job, state, cache)
        results.append(result)
        tid = job.get("task_id")
        if tid:
            statuses[tid] = result["status"]
        if result.get("dead_job"):
            statuses[result["dead_job"]] = result.get("reason") or "Killed"
        if result.get("resubmitted"):
            resubmits.append(result)

    save_jobs(jobs)
    atomic_json(STATE_PATH, state)
    watch = [
        {
            "key": j["key"],
            "task_id": j.get("task_id"),
            "task_name": j.get("task_name"),
            "status": statuses.get(j.get("task_id") or "", j.get("status")),
            "yaml": j.get("yaml_rel"),
        }
        for j in jobs
    ]
    atomic_json(WATCH_LIST_PATH, {"ts": utc_now(), "watch": watch, "resubmits": resubmits})

    bullets = None
    title = None
    if resubmits:
        title = "Resubmitted no-ESMC v3 explore spot job(s)"
        bullets = [
            "只重提 explore 目录同一 YAML（含 `02_real_data_gated` 若已有 ID）；未提交 ESMC；未 cancel Running/Success；未主动提交尚无 ID 的实验。",
            f"queue012 Preemptible 1×ml.pni2.3xlarge。人工 cancel 前 touch `{STOP_PATH}` 或 `STOP.<key|task_id>`。",
        ]
        for item in resubmits:
            bullets.append(
                f"resubmit `{item['key']}` dead=`{item.get('dead_job')}` ({item.get('reason')}) "
                f"→ `{item['new_id']}`"
            )
    if write_expand_note:
        start_bullets = [
            "已把 **queue012 no-ESMC v3 explore 看护** 扩到本目录全部 YAML（`01_synthetic_ready` + `02_real_data_gated` + 日后新增），同一 daemon，无第二份。",
            "每轮从 README ID 表、PROJECT_PROCESS.md、ledger、YAML TaskName、`submission_logs/` 发现新 task_id；**不会**主动提交尚未提交的实验（含 gated / ESMC）。",
            "后续提交必须写入 README ID 表或 ledger（或 PROJECT_PROCESS 行内含 YAML/TaskName），否则看护发现不了、Killed 不会被重提。",
            f"启停：`bash scripts/monitor_v3_explore_spot_start.sh` / `bash scripts/monitor_v3_explore_spot_stop.sh`；日志 `{LOG_PATH}`。",
            f"人工 cancel 前放哨兵：全局 `{STOP_PATH}`；单条 `STOP.<key>` / `STOP.<task_id>` / `STOP.<TaskName>`。",
            "Success 不重提；Running 不重复提交；Killed 重提同一 YAML；Failed 默认重提，代码 bug 会记录并跳过。全部 Success 后看护继续跑，等后续 ID。",
            f"本轮重提 {len(resubmits)} 条。未碰旧 1M / fusion。",
        ]
        compact = []
        for r in results:
            mark = f" → {r['new_id']}" if r.get("resubmitted") else ""
            compact.append(f"{r['key']} `{r.get('task_id') or '—'}` {r.get('status')}{mark}")
        start_bullets.append("本轮：" + " · ".join(compact))
        if resubmits:
            start_bullets.extend(bullets[2:] if bullets else [])
        try:
            patch_project_process(
                jobs,
                statuses,
                "Expanded no-ESMC v3 explore spot monitor to all queue012 YAML",
                start_bullets,
            )
        except Exception as exc:
            log("ERROR", f"PROJECT_PROCESS 扩围更新失败: {exc}")
    elif resubmits:
        try:
            patch_project_process(jobs, statuses, title, bullets)
        except Exception as exc:
            log("ERROR", f"PROJECT_PROCESS 更新失败: {exc}")
    else:
        try:
            patch_project_process(jobs, statuses, None, None)
        except Exception as exc:
            log("ERROR", f"PROJECT_PROCESS Active 更新失败: {exc}")

    try:
        patch_readme(jobs, statuses)
    except Exception as exc:
        log("ERROR", f"README 更新失败: {exc}")

    if write_expand_note:
        atomic_json(
            EXPAND_TICK_PATH,
            {"ts": utc_now(), "results": results, "resubmits": resubmits, "watch": watch},
        )
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="no-ESMC v3 explore queue012 看护（全部 YAML）")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=POLL_INTERVAL)
    args = ap.parse_args()

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    readme = MONITOR_DIR / "README.txt"
    readme.write_text(
        "no-ESMC v3 explore queue012 monitor (all YAML under train_jobs/immune_v3_explore_queue012)\n\n"
        "STOP:\n"
        f"  touch {STOP_PATH}\n"
        f"  touch {MONITOR_DIR}/STOP.01_synthetic_ready.A0\n"
        f"  touch {MONITOR_DIR}/STOP.02_real_data_gated.A0\n"
        f"  touch {MONITOR_DIR}/STOP.<task_id>\n"
        f"  touch {MONITOR_DIR}/STOP.<TaskName>\n\n"
        "  bash scripts/monitor_v3_explore_spot_start.sh\n"
        "  bash scripts/monitor_v3_explore_spot_stop.sh\n"
        "Later submits: add task_id to README ID table or ledger.json.\n",
        encoding="utf-8",
    )

    log("INFO", f"v3 explore 看护启动 pid={os.getpid()} interval={args.interval}s schema=2")
    first = True
    while True:
        if STOP_PATH.exists():
            log("INFO", f"全局 STOP {STOP_PATH}，退出")
            return 0
        write_expand_note = first and not EXPAND_TICK_PATH.exists()
        tick(write_expand_note=write_expand_note)
        first = False
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("INFO", "中断，退出看护")
        sys.exit(130)
