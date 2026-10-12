"""Audit and invoke pinned upstream antibody benchmark code, without reimplementing models.

Run from the repository root in conda pllm:
  python -m downstream.benchmark.ab_antigen.official audit --output /abs/audit
  python -m downstream.benchmark.ab_antigen.official probe --output /abs/new_probe

Probes exercise upstream command-line interfaces only. They are NOT inference,
benchmark scores, or evidence that a full experiment is reproducible.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
LOCK = Path(__file__).with_name("sources.lock.json")
SOURCE_ROOT = ROOT / "downstream/benchmark/baselines/ab_antigen_official"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def verify_source(spec: dict, source_root: Path = SOURCE_ROOT) -> dict:
    """Fail closed on a different commit, remote, dirty code, or altered entrypoint."""
    path = source_root / spec["id"]
    if not (path / ".git").exists():
        raise ValueError(f"Missing official checkout: {path}")
    actual = git(path, "rev-parse", "HEAD")
    if actual != spec["commit"]:
        raise ValueError(f"{spec['id']}: commit differs from lock: {actual}")
    remote = git(path, "remote", "get-url", "origin").removesuffix(".git").rstrip("/")
    if remote != spec["url"].removesuffix(".git").rstrip("/"):
        raise ValueError(f"{spec['id']}: unexpected origin: {remote}")
    dirty = git(path, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"{spec['id']}: upstream tracked files changed: {dirty}")
    # An untracked module can shadow an upstream import without changing HEAD.
    untracked = git(path, "ls-files", "--others", "--exclude-standard").splitlines()
    added_code = [name for name in untracked
                  if Path(name).suffix in {".py", ".so", ".pyd", ".sh"}]
    if added_code:
        raise ValueError(f"{spec['id']}: untracked executable code: {added_code}")
    for name, expected in spec["files"].items():
        candidate = path / name
        if not candidate.is_file() or digest(candidate) != expected:
            raise ValueError(f"{spec['id']}: missing or altered pinned file: {name}")
    return {"commit": actual, "origin": remote, "tracked_files_clean": True,
            "verified_files": len(spec["files"])}


def audit(spec: dict, source_root: Path = SOURCE_ROOT) -> dict:
    row = {"id": spec["id"], "scores_produced": False,
           "release": spec["release"], "protocol_status": spec["protocol_status"],
           "blockers": spec.get("blockers", [])}
    try:
        row.update(verify_source(spec, source_root))
        path = source_root / spec["id"]
        tracked = set(git(path, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
        row["required_files_absent_from_upstream_tree"] = [
            x for x in spec.get("required_upstream_files", []) if x not in tracked]
        row["syntax_errors"] = []
        for name in spec["files"]:
            if name.endswith(".py"):
                try:
                    ast.parse((path / name).read_text(), filename=name)
                except SyntaxError as exc:
                    row["syntax_errors"].append({"path": name, "error": str(exc)})
        row["status"] = "source_verified"
        row["static_checks_passed"] = not (
            row["syntax_errors"] or row["required_files_absent_from_upstream_tree"])
    except (ValueError, subprocess.SubprocessError, OSError) as exc:
        row.update(status="source_verification_failed", error=str(exc))
    return row


def probe(spec: dict, output: Path, *, source_root: Path = SOURCE_ROOT,
          python: str = sys.executable, timeout: int = 60) -> dict:
    """Run only reviewed --help commands; never fall back to a local implementation."""
    verification = verify_source(spec, source_root)
    results = []
    for index, recipe in enumerate(spec.get("cli_probes", [])):
        entry = recipe["entry"]
        if entry not in spec["files"] or not entry.endswith(".py"):
            raise ValueError(f"Unpinned probe entry: {entry}")
        path = source_root / spec["id"]
        command = [python, str(path / entry), "--help"]
        log = output / f"{spec['id']}_{index}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   MPLBACKEND="Agg")
        start = datetime.now(timezone.utc).isoformat()
        with log.open("w") as stream:
            try:
                done = subprocess.run(command, cwd=path, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=timeout)
                code = done.returncode
                status = "cli_help_passed" if code == 0 else "cli_help_failed"
            except subprocess.TimeoutExpired:
                code, status = None, "cli_help_timeout"
        results.append({"entry": entry, "command": command, "cwd": str(path),
                        "started_at": start, "returncode": code, "status": status,
                        "log": str(log), "log_sha256": digest(log)})
        verify_source(spec, source_root)
    return {"id": spec["id"], "verification": verification, "probes": results,
            "status": "probed" if results else "not_probed_no_safe_cli",
            "inference_executed": False, "scores_produced": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "probe"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", help="Source ID; default: all locked sources")
    parser.add_argument("--lock", type=Path, default=LOCK)
    args = parser.parse_args()
    args.output = args.output.resolve()
    lock = json.loads(args.lock.read_text())
    specs = lock["sources"]
    if args.source:
        unknown = set(args.source) - {s["id"] for s in specs}
        if unknown:
            parser.error(f"Unknown sources: {sorted(unknown)}")
        specs = [s for s in specs if s["id"] in args.source]
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for spec in specs:
        if args.action == "audit":
            result = audit(spec)
        else:
            try:
                result = probe(spec, args.output)
            except (ValueError, subprocess.SubprocessError, OSError) as exc:
                result = {"id": spec["id"], "status": "source_verification_failed",
                          "error": str(exc), "scores_produced": False}
        results.append(result)
        print(spec["id"], result["status"], flush=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "action": args.action, "lock_sha256": digest(args.lock),
              "python": sys.executable, "results": results,
              "inference_executed": False, "scores_produced": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(r["status"] == "source_verification_failed" for r in results):
        raise SystemExit(1)
    if any(r.get("static_checks_passed") is False for r in results):
        raise SystemExit(3)
    if any(p["status"] != "cli_help_passed" for r in results for p in r.get("probes", [])):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
