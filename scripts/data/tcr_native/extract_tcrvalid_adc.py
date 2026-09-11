#!/usr/bin/env python3
"""Resumable AIRR Data Commons pull for the TCR-VALID repertoire ID lists.

TCR-VALID published two unpaired unique-chain pools (TRA / TRB). This script
does **not** cartesian-product those pools. Pairing is only emitted when the
same repertoire has a shared nonempty ``cell_id`` with productive TRA and TRB.

Phases (all resumable):
  inventory  batched POST /repertoire metadata
  download   slim rearrangements (junction + V/J + cell_id), page size 1000
  pair       cell_id join inside each downloaded repertoire
  run        inventory -> download pair sources only -> pair

Default ``--mode pairs`` keeps only repertoires that can yield αβ pairs
(single-cell or physical TRA–TRB linkage). ImmuneCODE-style bulk is not
downloaded.

Writes under ``data/tcr_bulk_raw/tcrvalid/``. Does not touch the live v3
``tcr_repertoire`` 2.13M used by current checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DATA = PROJECT_ROOT / "data"
ID_DIR = DATA / "tcr_bulk_raw/tcrvalid_ids"
OUT_DIR = DATA / "tcr_bulk_raw/tcrvalid"
HOSTS = (
    "https://vdjserver.org/airr/v1",
    "https://ipa1.ireceptor.org/airr/v1",
    "https://covid19-1.ireceptor.org/airr/v1",
)
PAGE_SIZE = 1000  # VDJServer returns 0 rows at size=5000
FIELDS = [
    "repertoire_id",
    "sequence_id",
    "cell_id",
    "locus",
    "productive",
    "vj_in_frame",
    "stop_codon",
    "v_call",
    "j_call",
    "junction_aa",
    "cdr1_aa",
    "cdr2_aa",
    "cdr3_aa",
    "duplicate_count",
]
JUNCTION_MIN = 7
JUNCTION_MAX = 23


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


def _opener() -> urllib.request.OpenerDirector:
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener()


def adc_request(
    path: str,
    body: dict,
    *,
    timeout: int = 180,
    retries: int = 6,
    accept: str = "application/json",
) -> tuple[str, bytes]:
    payload = json.dumps(body).encode()
    last_err: Exception | None = None
    opener = _opener()
    for attempt in range(1, retries + 1):
        for host in HOSTS:
            url = f"{host.rstrip('/')}{path}"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": accept,
                    "User-Agent": "immune-llada-tcrvalid-extract/1.0",
                },
                method="POST",
            )
            try:
                with opener.open(req, timeout=timeout) as resp:
                    return host, resp.read()
            except Exception as exc:  # noqa: BLE001 — network, keep going
                last_err = exc
                log(f"warn {host}{path} attempt {attempt}: {exc}")
        time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"ADC request failed after retries: {path}: {last_err}")


def adc_json(path: str, body: dict, *, timeout: int = 180) -> tuple[str, dict]:
    host, raw = adc_request(path, body, timeout=timeout, accept="application/json")
    if not raw:
        return host, {}
    return host, json.loads(raw.decode())


def load_id_lists() -> tuple[set[str], set[str]]:
    def _read(name: str) -> set[str]:
        ids: set[str] = set()
        with (ID_DIR / name).open(newline="") as handle:
            for row in csv.DictReader(handle):
                rid = (row.get("repertoire_id") or "").strip()
                if rid:
                    ids.add(rid)
        return ids

    return _read("tra_repertoires.csv"), _read("trb_repertoires.csv")


def _pcr_loci(sample: dict) -> list[str]:
    loci: list[str] = []
    for item in sample.get("pcr_target") or []:
        if isinstance(item, dict) and item.get("pcr_target_locus"):
            loci.append(str(item["pcr_target_locus"]))
    return loci


def summarize_repertoire(rep: dict) -> dict:
    study = rep.get("study") or {}
    samples = rep.get("sample") or []
    sample = samples[0] if isinstance(samples, list) and samples else {}
    if not isinstance(sample, dict):
        sample = {}
    subject = rep.get("subject") or {}
    return {
        "repertoire_id": rep.get("repertoire_id"),
        "study_id": study.get("study_id"),
        "study_title": study.get("study_title"),
        "subject_id": subject.get("subject_id") if isinstance(subject, dict) else None,
        "single_cell": sample.get("single_cell"),
        "physical_linkage": sample.get("physical_linkage"),
        "pcr_loci": _pcr_loci(sample),
        "tissue": ((sample.get("tissue") or {}) or {}).get("label")
        if isinstance(sample.get("tissue"), dict)
        else sample.get("tissue"),
        "cell_isolation": sample.get("cell_isolation"),
        "library_generation_method": sample.get("library_generation_method"),
    }


def load_inventory(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get("repertoire_id")
            if rid:
                out[rid] = row
    return out


def write_inventory_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def phase_inventory(args: argparse.Namespace) -> dict[str, dict]:
    tra, trb = load_id_lists()
    all_ids = sorted(tra | trb)
    inv_path = OUT_DIR / "inventory.jsonl"
    have = load_inventory(inv_path)
    pending = [rid for rid in all_ids if rid not in have]
    log(
        f"inventory start unique={len(all_ids)} have={len(have)} "
        f"pending={len(pending)} overlap={len(tra & trb)}"
    )
    batch_size = args.inventory_batch
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        host, payload = adc_json(
            "/repertoire",
            {
                "filters": {
                    "op": "in",
                    "content": {"field": "repertoire_id", "value": batch},
                },
                "size": len(batch) + 5,
            },
            timeout=90,
        )
        found: dict[str, dict] = {}
        for rep in payload.get("Repertoire") or []:
            summary = summarize_repertoire(rep)
            rid = summary["repertoire_id"]
            if not rid:
                continue
            summary.update(
                {
                    "host": host,
                    "in_tra_list": rid in tra,
                    "in_trb_list": rid in trb,
                    "status": "ok",
                    "inventoried_at": utc_now(),
                }
            )
            found[rid] = summary
        for rid in batch:
            if rid in found:
                row = found[rid]
            else:
                row = {
                    "repertoire_id": rid,
                    "status": "missing",
                    "host": host,
                    "in_tra_list": rid in tra,
                    "in_trb_list": rid in trb,
                    "inventoried_at": utc_now(),
                }
            write_inventory_row(inv_path, row)
            have[rid] = row
        log(
            f"inventory batch {min(i + batch_size, len(pending))}/{len(pending)} "
            f"found={len(found)} missing={len(batch) - len(found)}"
        )
        time.sleep(args.sleep)
    write_inventory_table(have)
    n_sc = sum(1 for r in have.values() if r.get("single_cell") in (True, "true", "T"))
    n_miss = sum(1 for r in have.values() if r.get("status") == "missing")
    log(f"inventory done n={len(have)} single_cell={n_sc} missing={n_miss}")
    return have


def write_inventory_table(have: dict[str, dict]) -> None:
    path = OUT_DIR / "inventory.tsv"
    cols = [
        "repertoire_id",
        "status",
        "in_tra_list",
        "in_trb_list",
        "single_cell",
        "physical_linkage",
        "pcr_loci",
        "study_id",
        "study_title",
        "subject_id",
        "tissue",
        "cell_isolation",
        "library_generation_method",
        "host",
        "n_tra",
        "n_trb",
        "n_rows",
        "n_pairs_strict",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore", delimiter="\t")
        writer.writeheader()
        for rid in sorted(have):
            row = dict(have[rid])
            loci = row.get("pcr_loci")
            if isinstance(loci, list):
                row["pcr_loci"] = ",".join(loci)
            writer.writerow(row)


def is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "t", "1", "yes"}


def good_junction(seq: str) -> bool:
    if not seq or not (JUNCTION_MIN <= len(seq) <= JUNCTION_MAX):
        return False
    if seq[0] != "C" or seq[-1] not in "FW":
        return False
    bad = set("X*UBJOZ")
    return not any(ch in bad for ch in seq)


def done_path(rid: str) -> Path:
    return OUT_DIR / "rearrangements" / f"{rid}.done.json"


def tsv_path(rid: str) -> Path:
    return OUT_DIR / "rearrangements" / f"{rid}.tsv.gz"


def already_downloaded(rid: str) -> bool:
    marker = done_path(rid)
    gz = tsv_path(rid)
    if not marker.exists() or not gz.exists() or gz.stat().st_size == 0:
        return False
    try:
        meta = json.loads(marker.read_text())
    except json.JSONDecodeError:
        return False
    return bool(meta.get("complete"))


def facet_counts(rid: str) -> dict[str, int]:
    _host, payload = adc_json(
        "/rearrangement",
        {
            "filters": {"op": "=", "content": {"field": "repertoire_id", "value": rid}},
            "facets": "locus",
        },
        timeout=90,
    )
    counts: dict[str, int] = {}
    for item in payload.get("Facet") or []:
        locus = str(item.get("locus") or "")
        if locus:
            counts[locus] = int(item.get("count") or 0)
    return counts


def download_repertoire(rid: str, *, sleep_s: float) -> dict:
    if already_downloaded(rid):
        return json.loads(done_path(rid).read_text())
    dest = tsv_path(rid)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    n_rows = 0
    n_pages = 0
    n_cell = 0
    loci = Counter()
    offset = 0
    with gzip.open(tmp, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        while True:
            host, raw = adc_request(
                "/rearrangement",
                {
                    "filters": {
                        "op": "=",
                        "content": {"field": "repertoire_id", "value": rid},
                    },
                    "from": offset,
                    "size": PAGE_SIZE,
                    "format": "tsv",
                    "fields": FIELDS,
                },
                timeout=180,
                accept="text/tab-separated-values",
            )
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            if not lines:
                break
            reader = csv.DictReader(lines, delimiter="\t")
            page_rows = 0
            for row in reader:
                writer.writerow({k: row.get(k, "") for k in FIELDS})
                page_rows += 1
                n_rows += 1
                locus = (row.get("locus") or "").upper()
                if locus:
                    loci[locus] += 1
                if (row.get("cell_id") or "").strip():
                    n_cell += 1
            n_pages += 1
            if page_rows < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            if n_pages % 20 == 0:
                log(f"download {rid} pages={n_pages} rows={n_rows}")
            time.sleep(sleep_s)
    tmp.replace(dest)
    meta = {
        "repertoire_id": rid,
        "complete": True,
        "n_rows": n_rows,
        "n_pages": n_pages,
        "n_with_cell_id": n_cell,
        "n_tra": int(loci.get("TRA", 0)),
        "n_trb": int(loci.get("TRB", 0)),
        "bytes": dest.stat().st_size,
        "host": host,
        "finished_at": utc_now(),
    }
    done_path(rid).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def is_pair_source(row: dict) -> bool:
    """True only if this repertoire can yield cell_id or physically linked αβ pairs.

    ImmuneCODE overlap IDs are excluded: they sit on both TRA/TRB lists but
    almost never have ``cell_id``. Twin study PRJNA593622 is kept because
    ``physical_linkage=hetero_head-head`` already produced hundreds of
    thousands of 1:1 pairs from completed files.
    """
    if row.get("status") == "missing":
        return False
    if is_truthy(row.get("single_cell")):
        return True
    linkage = str(row.get("physical_linkage") or "").lower()
    return "head" in linkage


def select_download_ids(
    have: dict[str, dict],
    tra: set[str],
    trb: set[str],
    mode: str,
) -> list[str]:
    overlap = tra & trb

    def is_overlap_or_sc(rid: str) -> bool:
        row = have.get(rid) or {}
        if row.get("status") == "missing":
            return False
        if rid in overlap or is_truthy(row.get("single_cell")):
            return True
        loci = row.get("pcr_loci") or []
        return isinstance(loci, list) and "TRA" in loci and "TRB" in loci

    ids = [rid for rid in sorted(tra | trb) if (have.get(rid) or {}).get("status") != "missing"]
    if mode == "pairs":
        ids = [rid for rid in ids if is_pair_source(have.get(rid) or {})]
        # small single-cell libraries first; linked bulk twins after
        ids.sort(key=lambda rid: (0 if is_truthy((have.get(rid) or {}).get("single_cell")) else 1, rid))
        return ids
    if mode == "paired":
        ids = [rid for rid in ids if is_overlap_or_sc(rid)]
    elif mode == "unpaired":
        ids = [rid for rid in ids if not is_overlap_or_sc(rid)]
    elif mode != "all":
        raise ValueError(f"unknown mode {mode}")
    ids.sort(key=lambda rid: (0 if is_overlap_or_sc(rid) else 1, rid))
    return ids


def phase_download(args: argparse.Namespace, have: dict[str, dict]) -> None:
    tra, trb = load_id_lists()
    ids = select_download_ids(have, tra, trb, args.mode)
    pending = [rid for rid in ids if not already_downloaded(rid)]
    log(f"download mode={args.mode} selected={len(ids)} pending={len(pending)}")
    for i, rid in enumerate(pending, start=1):
        t0 = time.time()
        try:
            meta = download_repertoire(rid, sleep_s=args.sleep)
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR download {rid}: {exc}")
            fail = OUT_DIR / "rearrangements" / f"{rid}.fail.json"
            fail.write_text(json.dumps({"repertoire_id": rid, "error": str(exc), "at": utc_now()}, indent=2))
            time.sleep(args.sleep)
            continue
        row = have.get(rid) or {"repertoire_id": rid}
        row.update({k: meta.get(k) for k in ("n_rows", "n_tra", "n_trb", "n_with_cell_id")})
        have[rid] = row
        log(
            f"download {i}/{len(pending)} {rid} rows={meta.get('n_rows')} "
            f"TRA={meta.get('n_tra')} TRB={meta.get('n_trb')} "
            f"cell_id={meta.get('n_with_cell_id')} {time.time() - t0:.1f}s"
        )
        if args.pair_every and i % args.pair_every == 0:
            phase_pair(have)
        time.sleep(args.sleep)
    write_inventory_table(have)


def iter_rearrangement_rows(rid: str):
    path = tsv_path(rid)
    if not path.exists():
        return
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def phase_pair(have: dict[str, dict]) -> dict:
    pair_dir = OUT_DIR / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    strict_path = pair_dir / "pairs_strict.tsv"
    multi_path = pair_dir / "pairs_multi.tsv"
    cols = [
        "repertoire_id",
        "cell_id",
        "cdr3a",
        "cdr3b",
        "v_alpha",
        "j_alpha",
        "v_beta",
        "j_beta",
        "cdr1a",
        "cdr2a",
        "cdr1b",
        "cdr2b",
        "duplicate_count_a",
        "duplicate_count_b",
        "study_id",
        "pair_type",
    ]
    stats = Counter()
    seen_strict: set[tuple[str, str, str, str]] = set()
    with strict_path.open("w", newline="") as sf, multi_path.open("w", newline="") as mf:
        sw = csv.DictWriter(sf, fieldnames=cols, delimiter="\t")
        mw = csv.DictWriter(mf, fieldnames=cols, delimiter="\t")
        sw.writeheader()
        mw.writeheader()
        for rid, meta in sorted(have.items()):
            if not already_downloaded(rid):
                continue
            stats["repertoires_scanned"] += 1
            cells: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"TRA": [], "TRB": []})
            n_cell = 0
            for row in iter_rearrangement_rows(rid) or []:
                cell = (row.get("cell_id") or "").strip()
                if not cell:
                    continue
                n_cell += 1
                if not is_truthy(row.get("productive")):
                    continue
                if row.get("vj_in_frame") not in ("", None) and not is_truthy(row.get("vj_in_frame")):
                    continue
                junction = (row.get("junction_aa") or "").strip().upper()
                if not good_junction(junction):
                    continue
                locus = (row.get("locus") or "").upper()
                if locus not in {"TRA", "TRB"}:
                    continue
                cells[cell][locus].append(row)
            stats["rows_with_cell_id"] += n_cell
            study_id = (have.get(rid) or {}).get("study_id")
            n_strict = 0
            for cell_id, chains in cells.items():
                alphas = chains["TRA"]
                betas = chains["TRB"]
                if not alphas or not betas:
                    continue
                stats["cells_both_chains"] += 1
                pair_type = "strict" if len(alphas) == 1 and len(betas) == 1 else "multi"
                writer = sw if pair_type == "strict" else mw
                for alpha in alphas:
                    for beta in betas:
                        rec = {
                            "repertoire_id": rid,
                            "cell_id": cell_id,
                            "cdr3a": (alpha.get("junction_aa") or "").upper(),
                            "cdr3b": (beta.get("junction_aa") or "").upper(),
                            "v_alpha": alpha.get("v_call") or "",
                            "j_alpha": alpha.get("j_call") or "",
                            "v_beta": beta.get("v_call") or "",
                            "j_beta": beta.get("j_call") or "",
                            "cdr1a": alpha.get("cdr1_aa") or "",
                            "cdr2a": alpha.get("cdr2_aa") or "",
                            "cdr1b": beta.get("cdr1_aa") or "",
                            "cdr2b": beta.get("cdr2_aa") or "",
                            "duplicate_count_a": alpha.get("duplicate_count") or "",
                            "duplicate_count_b": beta.get("duplicate_count") or "",
                            "study_id": study_id or "",
                            "pair_type": pair_type,
                        }
                        key = (rid, cell_id, rec["cdr3a"], rec["cdr3b"])
                        if pair_type == "strict":
                            if key in seen_strict:
                                continue
                            seen_strict.add(key)
                            n_strict += 1
                        writer.writerow(rec)
                        stats[f"pairs_{pair_type}"] += 1
            row = have.get(rid) or {"repertoire_id": rid}
            row["n_pairs_strict"] = n_strict
            have[rid] = row
    summary = {
        "finished_at": utc_now(),
        **{k: int(v) for k, v in stats.items()},
        "unique_strict_keys": len(seen_strict),
        "strict_path": str(strict_path),
        "multi_path": str(multi_path),
    }
    (pair_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_inventory_table(have)
    log(
        f"pair done strict={summary.get('pairs_strict', 0)} "
        f"multi={summary.get('pairs_multi', 0)} "
        f"cells_both={summary.get('cells_both_chains', 0)}"
    )
    return summary


def write_status(have: dict[str, dict], extra: dict | None = None) -> None:
    tra, trb = load_id_lists()
    overlap = tra & trb
    downloaded = [rid for rid in have if already_downloaded(rid)]
    payload = {
        "updated_at": utc_now(),
        "n_inventory": len(have),
        "n_downloaded": len(downloaded),
        "n_overlap_ids": len(overlap),
        "n_single_cell": sum(1 for r in have.values() if is_truthy(r.get("single_cell"))),
        "proxy": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or "",
        **(extra or {}),
    }
    (OUT_DIR / "status.json").write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("inventory", "download", "pair", "run"))
    parser.add_argument(
        "--mode",
        choices=("pairs", "paired", "unpaired", "all"),
        default="pairs",
        help="pairs=single-cell or physically linked αβ only (default)",
    )
    parser.add_argument("--inventory-batch", type=int, default=40)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument(
        "--pair-every",
        type=int,
        default=1,
        help="rebuild pair tables every N newly downloaded repertoires (0=only at end)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rearrangements").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pairs").mkdir(parents=True, exist_ok=True)
    log(f"start phase={args.phase} mode={args.mode} out={OUT_DIR}")
    log(f"proxy={os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY') or 'NONE'}")

    have = load_inventory(OUT_DIR / "inventory.jsonl")
    if args.phase in {"inventory", "run"}:
        have = phase_inventory(args)
        write_status(have, {"phase": "inventory"})
    if args.phase in {"download", "run"}:
        # Refresh pairs from any already-finished gzip before more downloads.
        if any(already_downloaded(rid) for rid in have):
            phase_pair(have)
            write_status(have, {"phase": "pair_refresh"})
        if args.phase == "run":
            saved_mode = args.mode
            if saved_mode in {"pairs", "paired", "all"}:
                args.mode = "pairs" if saved_mode in {"pairs", "all"} else "paired"
                phase_download(args, have)
                phase_pair(have)
                write_status(have, {"phase": "pairs_download"})
            if saved_mode == "unpaired":
                args.mode = "unpaired"
                phase_download(args, have)
                write_status(have, {"phase": "unpaired_download"})
            args.mode = saved_mode
        else:
            phase_download(args, have)
            write_status(have, {"phase": "download"})
    if args.phase in {"pair", "run"}:
        phase_pair(have)
        write_status(have, {"phase": "pair"})
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
