#!/usr/bin/env python3
"""看护 32 条 v2 immune_v6 官方闲时评测：终态异常则用同一 YAML 重提。

只盯 submissions.json 里的评测 task_id 及其替换 ID。
绝不 cancel / 重提 / 改两条训练：
    t-20260923185956-wsvpq
    t-20260924185250-w22zf
也不复用 llada_train_watchdog_jobs.json 或 monitor_spot_tasks.py 的 TARGETS。

人工 cancel 前必须先放 STOP 哨兵，否则会被自动重提回来：
    touch output/downstream_generation/v2_immune_v6_eval_20260927/_monitor/STOP
    touch output/downstream_generation/v2_immune_v6_eval_20260927/_monitor/STOP.<TaskName>
    touch output/downstream_generation/v2_immune_v6_eval_20260927/_monitor/STOP.<task_id>

用法：
    python3 scripts/monitor_v2_eval_spot.py            # 前台
    python3 scripts/monitor_v2_eval_spot.py --once     # 只跑一轮
    bash scripts/monitor_v2_eval_spot_start.sh         # 后台常驻
    bash scripts/monitor_v2_eval_spot_stop.sh          # 只停看护，不动评测/训练作业
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
SUBMISSIONS = Path(
    os.environ.get(
        "V2_EVAL_SUBMISSIONS",
        str(REPO / "output/downstream_generation/v2_immune_v6_eval_20260928/submissions.json"),
    )
)
MONITOR_DIR = Path(
    os.environ.get(
        "V2_EVAL_MONITOR_DIR",
        str(REPO / "output/downstream_generation/v2_immune_v6_eval_20260928/_monitor"),
    )
)
STATE_PATH = MONITOR_DIR / "state.json"
LOG_PATH = MONITOR_DIR / "monitor.log"
STOP_PATH = MONITOR_DIR / "STOP"
FIRST_TICK_PATH = MONITOR_DIR / "first_tick.json"
PROJECT_PROCESS = REPO / "PROJECT_PROCESS.md"
RESULTS = REPO / "downstream/benchmark/RESULTS.md"
VOLC = REPO / "scripts/volc-no-proxy.sh"

PROTECTED_TRAINING = {
    "t-20260923185956-wsvpq",
    "t-20260924185250-w22zf",
}
SNAPSHOT_ROOTS = [
    REPO / "output/ab_eval_checkpoints/v2_esmc_immune_v6_15000_20260927",
    REPO / "output/ab_eval_checkpoints/v2_noesmc_immune_v6_110000_20260927",
    REPO / "output/ab_eval_checkpoints/v2_noesmc_immune_v6_150000_20260928",
    REPO / "output/protein_esmc_llada270m_diffusion_v2_immune_v6_2node4gpu/eval_snapshot_15000",
    REPO / "output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/eval_snapshot_110000",
    REPO / "output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/eval_snapshot_150000",
]
NEVER_STASH_PREFIXES = (
    str(REPO / "output/ab_eval_checkpoints/v2_esmc_immune_v6_15000_20260927"),
    str(REPO / "output/ab_eval_checkpoints/v2_noesmc_immune_v6_110000_20260927"),
    str(REPO / "output/ab_eval_checkpoints/v2_noesmc_immune_v6_150000_20260928"),
    str(REPO / "output/protein_esmc_llada270m_diffusion_v2_immune_v6_2node4gpu/eval_snapshot_15000"),
    str(REPO / "output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/eval_snapshot_110000"),
    str(REPO / "output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/eval_snapshot_150000"),
    str(REPO / "output/protein_esmc_llada270m_diffusion_v2_immune_v6_2node4gpu/checkpoint-"),
    str(REPO / "output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/checkpoint-"),
)

POLL_INTERVAL = 360
RESUBMIT_COOLDOWN = 180
MAX_RESUBMITS = 50

IN_FLIGHT = {"Queue", "Staging", "Running", "Initialized", "Killing"}
TERMINAL_OK = {"Success"}
TERMINAL_BAD = {"Failed", "Killed"}

T2T3_BENCH = {
    "t2a-v2": "tcr_clustering",
    "t2b-v2": "tcr_clustering_embed",
    "t3broad-v2": "tcr_representation",
    "t3deep-v2": "tcr_representation_paper6",
}
T4_NAMES = {
    "t4-benchmark14": "benchmark14",
    "t4-held20": "held20",
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
TASK_ID_RE = re.compile(r"t-\d{14}-[a-z0-9]+")


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
    return json.loads(text[min(starts):])


def get_task(task_id: str) -> dict:
    if task_id in PROTECTED_TRAINING:
        raise RuntimeError(f"拒绝查询训练任务作为评测目标: {task_id}")
    out = run_volc(["ml_task", "get", "-i", task_id, "-o", "json"], timeout=120)
    data = parse_json_blob(out)
    if isinstance(data, list):
        data = data[0] if data else {}
    return {
        "id": data.get("Id") or data.get("JobId") or task_id,
        "name": data.get("Name") or data.get("JobName") or "",
        "status": data.get("Status") or "",
        "queue": data.get("ResourceQueueId") or data.get("ResourceQueueName") or "",
        "start": data.get("LaunchTime") or data.get("Start") or "",
        "end": data.get("EndTime") or data.get("End") or "",
        "raw": data,
    }


def submit_yaml(yaml_path: str) -> str:
    if any(tid in yaml_path for tid in PROTECTED_TRAINING):
        raise RuntimeError(f"拒绝提交训练 YAML: {yaml_path}")
    out = run_volc(["ml_task", "submit", "--conf", yaml_path, "-o", "json"], timeout=300)
    data = None
    try:
        data = parse_json_blob(out)
    except Exception:
        data = None
    if isinstance(data, dict):
        new_id = data.get("Id") or data.get("JobId") or data.get("id")
        if new_id:
            new_id = str(new_id).strip()
            if new_id in PROTECTED_TRAINING:
                raise RuntimeError(f"提交结果撞上训练 ID，拒绝跟踪: {new_id}")
            return new_id
    match = re.search(r"task_id=(\S+)", out)
    if match:
        new_id = match.group(1).strip()
        if new_id in PROTECTED_TRAINING:
            raise RuntimeError(f"提交结果撞上训练 ID，拒绝跟踪: {new_id}")
        return new_id
    raise RuntimeError(f"提交未返回 task_id: {out[-800:]!r}")


def yaml_meta(yaml_path: str) -> dict:
    import yaml
    raw = Path(yaml_path).read_text(encoding="utf-8")
    text = yaml.safe_load(raw)["Entrypoint"]
    name = re.search(r"^TaskName:\s*(\S+)", raw, re.M)
    queue = re.search(r"^ResourceQueueName:\s*(\S+)", raw, re.M)
    preempt = re.search(r"^Preemptible:\s*(\S+)", raw, re.M)
    tag = re.search(r"TCR_EVAL_TAG=(\S+)", text)
    t1_tag = re.search(r"--tag\s+(\S+)", text)
    mkdirs = re.findall(r"mkdir -p\s+(\S+)", text)
    cleaned = []
    for item in mkdirs:
        path = item.strip().strip('"').strip("'").rstrip("\\")
        if path:
            cleaned.append(path)
    return {
        "task_name": name.group(1) if name else Path(yaml_path).stem,
        "queue": queue.group(1) if queue else "",
        "preemptible": (preempt.group(1).lower() == "true") if preempt else False,
        "tcr_tag": tag.group(1) if tag else "",
        "t1_tag": t1_tag.group(1) if t1_tag else "",
        "log_dirs": cleaned,
    }


def tcr_tag_for(job: dict, meta: dict) -> str:
    if meta.get("tcr_tag"):
        return meta["tcr_tag"]
    if job.get("model") == "esmc":
        return "tcr_v2_esmc_v6_15000_20260927"
    if str(job.get("step") or "") == "150000" or "150000" in str(job.get("yaml") or ""):
        return "tcr_v2_noesmc_v6_150000_20260928"
    return "tcr_v2_noesmc_v6_110000_20260927"


def unsafe_to_stash(path: Path) -> bool:
    resolved = str(path.resolve()) if path.exists() else str(path)
    for prefix in NEVER_STASH_PREFIXES:
        if resolved.startswith(prefix) or str(path).startswith(prefix):
            return True
    return False


def stash_path(path: Path, dead_job: str) -> str | None:
    if not path.exists():
        return None
    if unsafe_to_stash(path):
        log("ERROR", f"拒绝挪动受保护路径 {path}")
        return None
    dest = path.with_name(f"{path.name}.killed_{dead_job}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    shutil.move(str(path), str(dest))
    return str(dest)


def paths_to_stash(job: dict, meta: dict) -> list[Path]:
    """评测 runner 多数不能断点续跑；重提前挪开会触发 FileExistsError 的半成品。

    不进度安全（必须挪开，否则同一 YAML 会立刻 Failed）：
      T2/T3：`output.mkdir(exist_ok=False)` + preflight gate FileExistsError
      T4：`output.mkdir(exist_ok=False)`
    可保留（重提 = 同一快照的新鲜评测 / T1 特征缓存可续）：
      T1 embedding 缓存、AB CDR tee 日志
    pairing / specificity / gdp / m396：若还没有最终 metrics，挪开半成品以免脏写。
    快照与训练 checkpoint 永不挪动。
    """
    task = job.get("task", "")
    tag = tcr_tag_for(job, meta)
    out: list[Path] = []
    if task in T2T3_BENCH:
        out.append(REPO / "downstream/benchmark/outputs" / T2T3_BENCH[task] / f"ours_{tag}_runtime_v2")
        for log_dir in meta["log_dirs"]:
            out.append(Path(log_dir) / "gate.json")
    elif task in T4_NAMES:
        out.append(
            REPO / "downstream/benchmark/outputs/tcr_generation_bench/setting_B"
            / f"ours_{tag}_{T4_NAMES[task]}_pmhc_beta_v2"
        )
    elif task in {"pair-p0-cfg0", "specificity", "gdp_a1", "m396"}:
        for log_dir in meta["log_dirs"]:
            d = Path(log_dir)
            if task == "pair-p0-cfg0" and (d / "holdout500_n8.csv").exists():
                # The job validates the generation manifest and retries scoring.
                continue
            finals = [
                d / "holdout500_metrics.json",
                d / "metrics.json",
            ]
            blocking = [
                d / "holdout500_n8.csv",
                d / "run_manifest.json",
            ]
            if d.exists() and (not any(p.exists() for p in finals) or any(p.exists() for p in blocking)):
                if not any(p.exists() for p in finals):
                    out.append(d)
    return out


def resume_note(job: dict) -> str:
    task = job.get("task", "")
    if task in T2T3_BENCH or task in T4_NAMES:
        return (
            "same YAML re-submit of the same snapshot; runner is NOT checkpoint/progress-safe "
            "(mkdir exist_ok=False / gate FileExistsError). Incomplete runner outputs were stashed; "
            "this is a fresh eval, not a mid-run resume."
        )
    if task.startswith("t1-"):
        return (
            "same YAML re-submit of the same snapshot; T1 feature cache is kept so a resubmit "
            "can reuse embeddings if present. Not a training resume."
        )
    return (
        "same YAML re-submit of the same snapshot; AB log dirs use mkdir -p (not progress-safe). "
        "Incomplete pairing/specificity/gdp/m396 dirs without final metrics were stashed. Fresh eval."
    )


def confirm_snapshots() -> None:
    for root in SNAPSHOT_ROOTS:
        weights = root / "model.safetensors"
        if not weights.is_file():
            log("WARN", f"快照缺失（评测输入应保留）: {weights}")
            continue
        nlink = weights.stat().st_nlink
        log("INFO", f"快照仍在 {weights} nlink={nlink}")


def job_key(job: dict) -> str:
    return f"{job['model']}.{job['task']}"


def sentinel_hit(job: dict, meta: dict, current_id: str) -> str | None:
    names = {
        STOP_PATH,
        MONITOR_DIR / f"STOP.{meta['task_name']}",
        MONITOR_DIR / f"STOP.{current_id}",
        MONITOR_DIR / f"STOP.{job.get('task_id', '')}",
        MONITOR_DIR / f"STOP.{job_key(job)}",
    }
    for hist in job.get("history") or []:
        dead = hist.get("dead_job")
        if dead:
            names.add(MONITOR_DIR / f"STOP.{dead}")
    for path in names:
        if path.exists():
            return str(path)
    return None


def replace_task_id_in_text(text: str, old_id: str, new_id: str) -> str:
    if old_id in PROTECTED_TRAINING or new_id in PROTECTED_TRAINING:
        return text
    return text.replace(old_id, new_id)


def update_active_table(text: str, updates: list[dict]) -> str:
    """updates: {old_id, new_id, status, remove}"""
    wanted = {u["old_id"]: u for u in updates if u["old_id"] not in PROTECTED_TRAINING}
    lines = text.splitlines(keepends=True)
    out = []
    for line in lines:
        hit = None
        if line.startswith("|"):
            for old_id, upd in wanted.items():
                if f"`{old_id}`" in line:
                    hit = upd
                    break
        if hit:
            if hit.get("remove"):
                continue
            new_id = hit.get("new_id") or hit["old_id"]
            line = line.replace(f"`{hit['old_id']}`", f"`{new_id}`", 1)
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 7:
                parts[-2] = f" {hit['status']} "
                line = "|".join(parts) + "\n"
        out.append(line)
    joined = "".join(out)
    joined = re.sub(
        r"Last updated: [^\n]*v2 immune_v6[^\n]*",
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S+00:00')}（v2 immune_v6 闲时评测看护）",
        joined,
        count=1,
    )
    return joined


def insert_dated_note(text: str, title: str, bullets: list[str]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    note = f"## {stamp} {title}\n\n" + "".join(f"- {b}\n" for b in bullets) + "\n"
    if text.startswith("# Project Process"):
        rest = text.split("\n", 1)[1].lstrip("\n")
        return f"# Project Process\n\n{note}{rest}"
    return note + text


def patch_project_process(updates: list[dict], note_title: str | None, bullets: list[str] | None) -> None:
    text = PROJECT_PROCESS.read_text(encoding="utf-8")
    if updates:
        text = update_active_table(text, updates)
    if note_title and bullets:
        text = insert_dated_note(text, note_title, bullets)
    atomic_write(PROJECT_PROCESS, text)


def patch_results_ids(old_id: str, new_id: str) -> None:
    if not RESULTS.is_file() or old_id == new_id:
        return
    text = RESULTS.read_text(encoding="utf-8")
    new_text = replace_task_id_in_text(text, old_id, new_id)
    if new_text != text:
        atomic_write(RESULTS, new_text)


def maybe_fill_results(job: dict, meta: dict) -> None:
    """Success 且磁盘上已有真实指标时才填；缺文件就保持 pending，不编数字。"""
    extracted = extract_metrics(job, meta)
    if not extracted:
        log("INFO", f"{job_key(job)}: Success 但尚未找到完整指标文件，RESULTS 保持 pending")
        return
    text = RESULTS.read_text(encoding="utf-8")
    model = job["model"]
    task = job["task"]
    row_key = "Ours V6 ESMC 15000" if model == "esmc" else "Ours V6 no-ESMC 110000"
    changed = False

    def fill_pending_row(anchor: str, values: list[str], extra_suffix: str | None = None) -> str:
        nonlocal changed
        lines = text.splitlines(keepends=True)
        out = []
        in_section = False
        for line in lines:
            if line.startswith("### ") or line.startswith("## "):
                in_section = anchor in line or (in_section and line.startswith("####"))
            if in_section and line.startswith(f"| {row_key}"):
                parts = line.rstrip("\n").split("|")
                # parts[0] empty, [1] name, then metric cells, last maybe note
                cells = parts[1:-1]
                if cells and cells[0].strip().startswith(row_key.split()[0]):
                    # keep name cell; replace pending metric cells left-to-right
                    mi = 0
                    for i in range(1, len(cells)):
                        if mi >= len(values):
                            break
                        if "pending" in cells[i]:
                            cells[i] = f" {values[mi]} "
                            mi += 1
                    if extra_suffix and len(cells) >= 2:
                        last = cells[-1]
                        if "评测中" in last or last.strip() in {"", "|"}:
                            cells[-1] = f" {extra_suffix} "
                    new_line = "|" + "|".join(cells) + "|\n"
                    if new_line != line:
                        changed = True
                        line = new_line
            out.append(line)
        return "".join(out)

    # Keep this conservative: only fill when the helper returns a complete vector
    # and the matching RESULTS section still has pending cells.
    section_map = {
        "cdr-kong": ("§0.5 AB CDR infilling", extracted.get("values"), "Kong H1/H2/H3 AAR"),
        "cdr-sab23": ("SAb23", extracted.get("values"), None),
        "pair-p0-cfg0": ("pairing", extracted.get("values"), None),
        "specificity": ("Specificity", extracted.get("values"), None),
        "gdp_a1": None,  # GDP 主表结构特殊，只把数字记进 submissions
        "m396": ("m396", extracted.get("values"), None),
        "t1-cdr3b": ("T1", extracted.get("values"), None),
        "t1-others_cdr3b": ("T1", extracted.get("values"), None),
        "t1-cdr3ab": ("T1", extracted.get("values"), None),
        "t1-others_longab": ("T1", extracted.get("values"), None),
        "t2a-v2": ("T2-A", extracted.get("values"), None),
        "t2b-v2": ("T2-B", extracted.get("values"), None),
        "t3broad-v2": ("T3", extracted.get("values"), None),
        "t3deep-v2": ("T3", extracted.get("values"), None),
        "t4-benchmark14": ("T4", extracted.get("values"), None),
        "t4-held20": ("held20", extracted.get("values"), None),
    }
    spec = section_map.get(task)
    if spec and spec[1]:
        text = fill_pending_row(spec[0], spec[1], spec[2])
    if changed:
        atomic_write(RESULTS, text)
        log("INFO", f"{job_key(job)}: 已用真实指标回填 RESULTS.md")
    else:
        log("INFO", f"{job_key(job)}: 指标已落盘 {extracted.get('paths')}，RESULTS 表结构未自动对齐则保持 pending")


def extract_metrics(job: dict, meta: dict) -> dict | None:
    task = job["task"]
    model = job["model"]
    tag = tcr_tag_for(job, meta)
    ab_root = REPO / f"output/downstream_generation/ab_v2_{model}_v6_{'15000' if model == 'esmc' else '110000'}_20260927"

    def read_json(path: Path):
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def aar_from_log(path: Path) -> float | None:
        if not path.is_file():
            return None
        hits = re.findall(r"Average AAR all folds[:\s]+([0-9.]+)", path.read_text(encoding="utf-8", errors="replace"))
        if not hits:
            return None
        return float(hits[-1])

    if task in {"cdr-kong", "cdr-sab23"}:
        folder = ab_root / task
        vals = []
        paths = []
        for mode in ("cdrh1", "cdrh2", "cdrh3"):
            p = folder / f"{mode}.log"
            v = aar_from_log(p)
            if v is None:
                return None
            vals.append(f"{v:.2f}" if v > 1 else f"{v * 100:.2f}")
            paths.append(str(p))
        return {"values": vals, "paths": paths}

    if task == "pair-p0-cfg0":
        path = ab_root / "pair-p0-cfg0" / "holdout500_metrics.json"
        data = read_json(path)
        if not data:
            return None
        keys = ("im_aa", "validity", "diversity", "light_type_match")
        if not all(k in data or any(k in str(data).lower() for k in keys) for k in ()):
            pass
        picked = []
        for key in ("im", "IM", "im_aa", "validity", "diversity", "chain_type_match", "light_type_match"):
            if key in data and isinstance(data[key], (int, float)):
                picked.append(f"{data[key]:.6g}")
        if len(picked) < 3:
            return {"values": [], "paths": [str(path)], "raw": data}
        return {"values": picked, "paths": [str(path)], "raw": data}

    if task in {"specificity", "gdp_a1", "m396"}:
        folder = task + "-fv-v1" if task in {"specificity", "m396"} else task
        path = ab_root / folder / "metrics.json"
        data = read_json(path)
        if not data:
            return None
        return {"values": [], "paths": [str(path)], "raw": data}

    t1_map = {
        "t1-cdr3b": ("ours_tcr_v2_{m}_v6_{s}_20260927_cdr3b", "cdr3b"),
        "t1-others_cdr3b": ("ours_tcr_v2_{m}_v6_{s}_20260927_others_cdr3b", "others_cdr3b"),
        "t1-cdr3ab": ("ours_tcr_v2_{m}_v6_{s}_20260927_cdr3ab", "cdr3ab"),
        "t1-others_longab": ("ours_tcr_v2_{m}_v6_{s}_20260927_others_longab", "others_longab"),
    }
    if task in t1_map:
        step = "15000" if model == "esmc" else "110000"
        folder, track = t1_map[task]
        path = (
            REPO / "downstream/benchmark/outputs/tcr_binding_nm2025_retrained"
            / folder.format(m=model if model == "esmc" else "noesmc", s=step)
            / track / "AS" / "summary.json"
        )
        data = read_json(path)
        if not data:
            return None
        return {"values": [], "paths": [str(path)], "raw": data}

    if task in T2T3_BENCH:
        folder = T2T3_BENCH[task]
        name = "fewshot.json" if task.startswith("t3") else "metrics.json"
        path = REPO / "downstream/benchmark/outputs" / folder / f"ours_{tag}_runtime_v2" / name
        data = read_json(path)
        if not data:
            return None
        return {"values": [], "paths": [str(path)], "raw": data}

    if task in T4_NAMES:
        path = (
            REPO / "downstream/benchmark/outputs/tcr_generation_bench/setting_B"
            / f"ours_{tag}_{T4_NAMES[task]}_pmhc_beta_v2" / "metrics.json"
        )
        data = read_json(path)
        if not data:
            return None
        return {"values": [], "paths": [str(path)], "raw": data}
    return None


def load_jobs() -> list[dict]:
    data = load_json(SUBMISSIONS, {})
    jobs = list(data.get("jobs") or [])
    expected = int(data.get("expected_jobs") or 0)
    if expected and len(jobs) != expected:
        log("WARN", f"submissions.json jobs={len(jobs)}，期望 {expected}")
    elif not jobs:
        log("WARN", "submissions.json 没有 jobs")
    for job in jobs:
        tid = job.get("task_id")
        if tid in PROTECTED_TRAINING:
            raise RuntimeError(f"submissions.json 混入训练 ID，拒绝启动: {tid}")
    return jobs


def save_jobs(jobs: list[dict], extra: dict | None = None) -> None:
    data = load_json(SUBMISSIONS, {})
    data["jobs"] = jobs
    data["monitor"] = {
        "updated_at_utc": utc_now(),
        "log": str(LOG_PATH),
        "stop_global": str(STOP_PATH),
        "stop_per_job": str(MONITOR_DIR / "STOP.<TaskName|task_id|model.task>"),
        "resume_policy": "same YAML; T2/T3/T4 not progress-safe (stash then fresh eval); T1 cache kept",
    }
    if extra:
        data.update(extra)
    atomic_json(SUBMISSIONS, data)


def check_one(job: dict, state: dict, dry_run: bool) -> dict:
    key = job_key(job)
    meta = yaml_meta(job["yaml"])
    st = state.setdefault(key, {"resubmits": 0, "last_resubmit_ts": 0, "history": [], "done": False})
    current_id = job["task_id"]
    result = {
        "key": key,
        "task_id": current_id,
        "name": meta["task_name"],
        "status": "unknown",
        "resubmitted": False,
        "new_id": None,
        "reason": None,
        "skipped": None,
    }

    if current_id in PROTECTED_TRAINING:
        result["skipped"] = "protected_training"
        log("ERROR", f"{key}: 目标是训练 ID {current_id}，跳过且不重提")
        return result

    hit = sentinel_hit(job, meta, current_id)
    if hit:
        result["status"] = "STOP"
        result["skipped"] = hit
        log("INFO", f"{key}: 存在 STOP 哨兵 {hit}，跳过（人工接管）")
        return result
    if st.get("done"):
        result["status"] = st.get("status") or "Success"
        result["skipped"] = "done"
        return result

    try:
        info = get_task(current_id)
    except Exception as exc:
        log("ERROR", f"{key}: 查询失败（下一轮重试）: {exc}")
        result["skipped"] = f"get_failed:{exc}"
        return result

    status = info["status"]
    result["status"] = status
    result["name"] = info["name"] or meta["task_name"]
    job["last_status"] = status
    job["last_checked_utc"] = utc_now()
    if info["name"]:
        job["task_name"] = info["name"]
    st["job_id"] = current_id
    st["status"] = status

    if status in TERMINAL_OK:
        log("INFO", f"{key}: {current_id} Success，停止看护该条")
        st["done"] = True
        job["done"] = True
        extracted = extract_metrics(job, meta)
        if extracted:
            job["metrics_paths"] = extracted.get("paths")
            if extracted.get("raw") is not None:
                job["metrics_note"] = "real metrics file present; RESULTS filled only if row mapping succeeded"
            log("INFO", f"{key}: Success 且指标文件存在 {extracted.get('paths')}；RESULTS 不由看护自动改表，避免串行")
        else:
            job["metrics_note"] = "Success but metrics files not found yet; RESULTS stays pending"
        return result

    if status in IN_FLIGHT:
        log("INFO", f"{key}: {current_id} {status}（流转中，不重提；Queue 久候也不自动重提）")
        return result

    if status not in TERMINAL_BAD:
        log("WARN", f"{key}: {current_id} 异常终态 {status!r}，按 Failed/Killed 处理")

    log("WARN", f"{key}: {current_id} 终态 {status} End={info.get('end')}")
    if st["resubmits"] >= MAX_RESUBMITS:
        log("ERROR", f"{key}: 累计重提已达 {MAX_RESUBMITS}，不再自动重提")
        result["skipped"] = "max_resubmits"
        return result
    since = time.time() - float(st.get("last_resubmit_ts") or 0)
    if since < RESUBMIT_COOLDOWN:
        log("INFO", f"{key}: 距上次重提仅 {since / 60:.1f} 分钟（冷却 {RESUBMIT_COOLDOWN / 60:.0f}），本轮跳过")
        result["skipped"] = "cooldown"
        return result
    if dry_run:
        log("INFO", f"{key}: [dry-run] 本应重提 {job['yaml']}")
        result["skipped"] = "dry_run"
        return result

    stashed = []
    for path in paths_to_stash(job, meta):
        moved = stash_path(path, current_id)
        if moved:
            stashed.append(moved)
            log("INFO", f"{key}: 已挪开半成品 → {moved}")

    try:
        new_id = submit_yaml(job["yaml"])
    except Exception as exc:
        log("ERROR", f"{key}: 重提失败（下一轮重试）: {exc}")
        result["skipped"] = f"submit_failed:{exc}"
        return result

    note = resume_note(job)
    hist = {
        "ts": utc_now(),
        "dead_job": current_id,
        "dead_status": status,
        "new_job": new_id,
        "reason": status,
        "stashed": stashed,
        "resume_note": note,
    }
    job["dead_job"] = current_id
    job["reason"] = status
    job["resume_note"] = note
    job["original_task_id"] = job.get("original_task_id") or current_id
    job["task_id"] = new_id
    job.setdefault("history", []).append(hist)
    st["resubmits"] += 1
    st["last_resubmit_ts"] = time.time()
    st["job_id"] = new_id
    st["status"] = "submitted"
    st.setdefault("history", []).append(hist)
    result.update(resubmitted=True, new_id=new_id, reason=status, task_id=new_id)
    log("INFO", f"{key}: 已重提 → {new_id}（第 {st['resubmits']} 次；{note}）")
    return result


def write_first_tick(results: list[dict], resubmits: list[dict]) -> None:
    payload = {
        "ts": utc_now(),
        "jobs": [
            {
                "key": r["key"],
                "task_id": r["task_id"] if not r.get("resubmitted") else r.get("new_id") or r["task_id"],
                "tracked_id": r["task_id"],
                "name": r.get("name"),
                "status": r.get("status"),
                "resubmitted": r.get("resubmitted"),
                "new_id": r.get("new_id"),
                "reason": r.get("reason"),
                "skipped": r.get("skipped"),
            }
            for r in results
        ],
        "resubmits": resubmits,
        "training_untouched": True,
        "protected_training": sorted(PROTECTED_TRAINING),
    }
    atomic_json(FIRST_TICK_PATH, payload)


def tick(dry_run: bool, write_start_note: bool) -> list[dict]:
    confirm_snapshots()
    jobs = load_jobs()
    state = load_json(STATE_PATH, {})
    results = []
    table_updates = []
    resubmits = []
    for job in jobs:
        result = check_one(job, state, dry_run)
        results.append(result)
        status = result["status"]
        old_id = job.get("original_task_id") or job["task_id"]
        # check_one already mutated job["task_id"] on resubmit
        current_id = job["task_id"]
        tracked_before = result["task_id"] if not result.get("resubmitted") else (job.get("dead_job") or old_id)
        if result.get("resubmitted"):
            table_updates.append({
                "old_id": result.get("reason") and job["dead_job"],
                "new_id": current_id,
                "status": f"Queue（看护重提 {utc_now()}；dead={job['dead_job']} {job['reason']}）",
            })
            resubmits.append({
                "key": result["key"],
                "dead_job": job["dead_job"],
                "new_id": current_id,
                "reason": job["reason"],
                "yaml": job["yaml"],
            })
            try:
                patch_results_ids(job["dead_job"], current_id)
            except Exception as exc:
                log("ERROR", f"{result['key']}: RESULTS task_id 替换失败: {exc}")
        elif status in TERMINAL_OK:
            table_updates.append({
                "old_id": current_id,
                "new_id": current_id,
                "status": "Success",
                "remove": True,
            })
        elif status not in {"unknown", "STOP"}:
            table_updates.append({
                "old_id": current_id,
                "new_id": current_id,
                "status": f"{status}（看护 {utc_now()}）",
            })
        _ = tracked_before

    save_jobs(jobs, extra={"training_untouched": True})
    atomic_json(STATE_PATH, state)

    bullets = None
    title = None
    if resubmits:
        title = "Resubmitted v2 immune_v6 official eval spot job(s)"
        bullets = [
            "只重提评测 YAML，未 cancel / 改 / 重提训练 `t-20260923185956-wsvpq` 与 `t-20260924185250-w22zf`。",
            f"queue012 Preemptible 1×ml.pni2.3xlarge；c20250601 CreateCustomTask 仍不可用。",
            "人工 cancel 前先 touch "
            f"`{STOP_PATH}` 或 `{MONITOR_DIR}/STOP.<TaskName|task_id>`。",
        ]
        for item in resubmits:
            bullets.append(
                f"resubmit `{item['key']}` dead=`{item['dead_job']}` ({item['reason']}) → `{item['new_id']}` "
                f"via `{item['yaml']}`；T2/T3/T4 非断点续跑，半成品已 stash，同一快照新鲜评测。"
            )
    if write_start_note:
        counts: dict[str, int] = {}
        for r in results:
            counts[r.get("status") or "unknown"] = counts.get(r.get("status") or "unknown", 0) + 1
        start_bullets = [
            "已启动 **只看 32 条 v2 immune_v6 官方闲时评测** 的常驻看护，不复用 `monitor_spot_tasks.py` TARGETS / `llada_train_watchdog_jobs.json`。",
            f"启停：`bash scripts/monitor_v2_eval_spot_start.sh` / `bash scripts/monitor_v2_eval_spot_stop.sh`；日志 `{LOG_PATH}`。",
            f"**人工 cancel 前务必先放哨兵**，否则 Killed 会被重提：全局 `{STOP_PATH}`；单条 `{MONITOR_DIR}/STOP.<TaskName>` 或 `STOP.<task_id>` 或 `STOP.<model.task>`。",
            "重提 = 同一 YAML + 同一快照。T2/T3/T4 **不是** checkpoint 续跑（`exist_ok=False`），半成品会 stash 后新鲜评测；T1 保留 embedding 缓存。Queue 久候不自动重提。",
            f"首轮状态计数：{', '.join(f'{k}={v}' for k, v in sorted(counts.items()))}；本轮重提 {len(resubmits)} 条。",
            "训练 `t-20260923185956-wsvpq` 与 `t-20260924185250-w22zf` **未查询以外的任何操作**（看护循环不把它们当目标，不 cancel / 不重提）。",
        ]
        compact = []
        for r in results:
            mark = f" → {r['new_id']}" if r.get("resubmitted") else ""
            compact.append(f"`{r['key']}` `{r.get('task_id')}` {r.get('status')}{mark}")
        start_bullets.append("首轮逐条：" + " · ".join(compact))
        if resubmits:
            start_bullets.extend(bullets[3:] if bullets and len(bullets) > 3 else [])
        try:
            patch_project_process(table_updates, "Started v2 immune_v6 eval spot monitor", start_bullets)
        except Exception as exc:
            log("ERROR", f"PROJECT_PROCESS 首轮更新失败: {exc}")
    else:
        try:
            patch_project_process(table_updates, title, bullets)
        except Exception as exc:
            log("ERROR", f"PROJECT_PROCESS 更新失败: {exc}")

    if write_start_note:
        write_first_tick(results, resubmits)
    return results


def all_done(state: dict, jobs: list[dict]) -> bool:
    return all(state.get(job_key(j), {}).get("done") for j in jobs)


def main() -> int:
    ap = argparse.ArgumentParser(description="v2 immune_v6 官方闲时评测看护")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--interval", type=int, default=POLL_INTERVAL)
    args = ap.parse_args()

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    readme = MONITOR_DIR / "README.txt"
    if not readme.exists():
        readme.write_text(
            "v2 immune_v6 official eval spot monitor\n"
            "\n"
            "STOP (human cancel — required, or a Killed job will be resubmitted):\n"
            f"  touch {STOP_PATH}\n"
            f"  touch {MONITOR_DIR}/STOP.<TaskName>\n"
            f"  touch {MONITOR_DIR}/STOP.<task_id>\n"
            f"  touch {MONITOR_DIR}/STOP.<model.task>   # e.g. STOP.esmc.cdr-kong\n"
            "\n"
            "Start / stop the daemon only (does not cancel eval or training jobs):\n"
            "  bash scripts/monitor_v2_eval_spot_start.sh\n"
            "  bash scripts/monitor_v2_eval_spot_stop.sh\n"
            "\n"
            "Never touch training jobs t-20260923185956-wsvpq / t-20260924185250-w22zf.\n",
            encoding="utf-8",
        )

    log("INFO", f"评测看护启动 pid={os.getpid()} interval={args.interval}s dry_run={args.dry_run}")
    log("INFO", f"人工 cancel 前 touch {STOP_PATH} 或 {MONITOR_DIR}/STOP.<TaskName|task_id>")
    first = True
    while True:
        if STOP_PATH.exists():
            log("INFO", f"检测到全局 STOP（{STOP_PATH}），退出看护")
            return 0
        write_start_note = first and not FIRST_TICK_PATH.exists()
        results = tick(args.dry_run, write_start_note=write_start_note)
        first = False
        state = load_json(STATE_PATH, {})
        jobs = load_jobs()
        if all_done(state, jobs):
            log("INFO", f"{len(jobs)} 条评测均已 Success，退出看护")
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
