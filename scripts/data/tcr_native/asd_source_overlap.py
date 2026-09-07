#!/usr/bin/env python3
"""READ-ONLY analysis: how much does ASD-antibody overlap each antibody benchmark
source (SAbDab CDR-infilling / FLAb / comp_chain OAS holdout) SEPARATELY, using
DIRECT train-vs-benchmark search (no connected-component transitive chaining).

Method: per bucket (heavy / light / CDR-H3 core), the benchmark seqs of all three
sources are pooled into ONE mmseqs query (headers tagged by source) and searched
against the ASD unique-sequence set as target (indexed once). An ASD sequence is
a DIRECT hit of source S at that bucket iff some S benchmark seq aligns to it at
>= threshold id / 0.80 cov. Thresholds: heavy/light 0.95, CDR-H3 core 0.80.
EXACT hits are byte-identical (normalized) set intersections.

Writes data/tcr_native/dataset/asd_source_overlap_report.json. Modifies nothing
else (no blocklist/dataset/loader/job changes).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decontam import DATA, MMSEQS_BIN, canon_core, is_valid_protein_sequence, normalize_sequence  # noqa: E402
from decontam_extra import ab_h3_core  # noqa: E402

PROJECT = DATA.parent
ASD_TRAIN = PROJECT / "downstream/asd/step6_final/antibody/train.csv"
OUT = DATA / "tcr_native/dataset/asd_source_overlap_report.json"
SOURCES = ("sabdab", "flab", "comp_chain")


# --------------------------------------------------------------------------- #
# per-source benchmark sub-banks (heavy / light / h3), nbbench excluded
# --------------------------------------------------------------------------- #

def _valid_chain(s: str) -> bool:
    return bool(s) and is_valid_protein_sequence(s) and len(s) >= 20


def build_source_banks() -> dict[str, dict[str, set[str]]]:
    banks = {s: {"heavy": set(), "light": set(), "h3": set()} for s in SOURCES}

    def add(src, h, l, h3):
        h = normalize_sequence(h)
        l = normalize_sequence(l)
        if _valid_chain(h):
            banks[src]["heavy"].add(h)
        if _valid_chain(l):
            banks[src]["light"].add(l)
        if h3:
            c = canon_core(h3, has_anchors=True)
            if c and len(c) >= 4:
                banks[src]["h3"].add(c)

    # SAbDab CDR-infilling
    for loop in ("cdrh1", "cdrh2", "cdrh3"):
        for fold in range(10):
            fp = DATA / f"downstream/cdr_infilling/sabdab/{loop}/fold_{fold}/test.json"
            if fp.is_file():
                for line in fp.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    h = d.get("heavy_chain_seq")
                    add("sabdab", h, d.get("light_chain_seq"),
                        d.get("cdrh3_seq") or ab_h3_core(h))
    # FLAb (no CDR3 column -> regex from heavy)
    flab_dir = DATA / "downstream/flab/flab_raw"
    if flab_dir.is_dir():
        for fp in sorted(flab_dir.glob("*.csv")):
            with fp.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    h = row.get("heavy")
                    add("flab", h, row.get("light"), ab_h3_core(h))
    # comp_chain OAS holdout
    p = DATA / "downstream/comp_chain/test_data_oas_holdout.csv"
    if p.is_file():
        with p.open(newline="") as fh:
            for row in csv.DictReader(fh):
                h = row.get("h_sequence") or row.get("cleaned_h_sequence")
                add("comp_chain", h, row.get("l_sequence") or row.get("cleaned_l_sequence"),
                    row.get("h_cdr3") or ab_h3_core(h))
    return banks


# --------------------------------------------------------------------------- #
# ASD unique keys
# --------------------------------------------------------------------------- #

def load_asd_keys() -> dict[str, set[str]]:
    heavy: set[str] = set()
    light: set[str] = set()
    h3: set[str] = set()
    csv.field_size_limit(2**31 - 1)
    with ASD_TRAIN.open(newline="") as fh:
        for row in csv.DictReader(fh):
            h = normalize_sequence(row.get("heavy_fv"))
            l = normalize_sequence(row.get("light_fv"))
            if _valid_chain(h):
                heavy.add(h)
                c = ab_h3_core(h)
                if c and len(c) >= 4:
                    h3.add(c)
            if _valid_chain(l):
                light.add(l)
    return {"heavy": heavy, "light": light, "h3": h3}


# --------------------------------------------------------------------------- #
# DIRECT search: source benchmark seqs (query, tagged) vs ASD uniques (target)
# --------------------------------------------------------------------------- #

def direct_hits(bucket: str, asd_keys: set[str], banks: dict[str, dict[str, set[str]]],
                min_id: float, threads: int, sens: float) -> dict[str, set[str]]:
    """Return {source -> set(ASD keys with a direct >=min_id / 0.80cov hit)}.

    ASD uniques are the QUERY (streamed), the pooled per-source benchmark bank is
    the small TARGET (tagged by source). Pairwise easy-search => no transitive
    connected-component chaining.
    """
    hits: dict[str, set[str]] = {s: set() for s in SOURCES}
    if not asd_keys:
        return hits
    tmp = Path(tempfile.mkdtemp(prefix=f"asdovl_{bucket}_"))
    try:
        asd_list = sorted(asd_keys)
        qry = tmp / "asd.fasta"
        with qry.open("w") as fh:
            for i, s in enumerate(asd_list):
                fh.write(f">a{i}\n{s}\n")
        tgt = tmp / "src.fasta"
        n_t = 0
        with tgt.open("w") as fh:
            for s in SOURCES:
                for j, seq in enumerate(sorted(banks[s][bucket])):
                    fh.write(f">{s}|{j}\n{seq}\n")
                    n_t += 1
        if n_t == 0:
            return hits
        res = tmp / "res.m8"
        cmd = [MMSEQS_BIN, "easy-search", str(qry), str(tgt), str(res), str(tmp / "tmp"),
               "--min-seq-id", str(min_id), "-c", "0.80", "--cov-mode", "0",
               "-e", "100", "-s", str(sens), "--threads", str(threads), "--max-seqs", "300"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(f"[mmseqs FAILED bucket={bucket} rc={proc.returncode}]\n")
            sys.stderr.write("STDERR tail:\n" + proc.stderr[-2500:] + "\n")
            raise RuntimeError(f"mmseqs easy-search failed for bucket {bucket}")
        with res.open() as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                qid = parts[0]
                src = parts[1].split("|", 1)[0]
                if src in hits and qid.startswith("a"):
                    hits[src].add(asd_list[int(qid[1:])])
        return hits
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    threads = 16
    print("loading ASD keys ...", flush=True)
    asd = load_asd_keys()
    print(f"ASD unique: heavy={len(asd['heavy'])} light={len(asd['light'])} h3={len(asd['h3'])}", flush=True)
    banks = build_source_banks()
    for s in SOURCES:
        print(f"bank[{s}]: heavy={len(banks[s]['heavy'])} light={len(banks[s]['light'])} h3={len(banks[s]['h3'])}", flush=True)

    # EXACT (byte-identical) intersections
    exact = {s: {b: len(asd[b] & banks[s][b]) for b in ("heavy", "light", "h3")} for s in SOURCES}

    # DIRECT search per bucket
    print("direct search heavy ...", flush=True)
    dh = direct_hits("heavy", asd["heavy"], banks, 0.95, threads, 4.0)
    print("direct search light ...", flush=True)
    dl = direct_hits("light", asd["light"], banks, 0.95, threads, 4.0)
    print("direct search h3 ...", flush=True)
    d3 = direct_hits("h3", asd["h3"], banks, 0.80, threads, 7.5)
    direct = {s: {"heavy": dh[s], "light": dl[s], "h3": d3[s]} for s in SOURCES}

    # per-source row removal (row removed if ANY of its buckets direct-hits that source)
    csv.field_size_limit(2**31 - 1)
    per_source_rows = {s: 0 for s in SOURCES}
    scenario_defs = {
        "a_all_three": ("sabdab", "flab", "comp_chain"),
        "b_drop_sabdab": ("flab", "comp_chain"),
        "c_drop_flab": ("sabdab", "comp_chain"),
        "d_sabdab_only": ("sabdab",),
        "e_flab_only": ("flab",),
    }
    scenario_rows = {k: 0 for k in scenario_defs}
    n_rows = 0
    with ASD_TRAIN.open(newline="") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            h = normalize_sequence(row.get("heavy_fv"))
            l = normalize_sequence(row.get("light_fv"))
            c = ab_h3_core(h)
            row_hit = {}
            for s in SOURCES:
                hit = ((_valid_chain(h) and h in direct[s]["heavy"])
                       or (_valid_chain(l) and l in direct[s]["light"])
                       or (c and c in direct[s]["h3"]))
                row_hit[s] = hit
                if hit:
                    per_source_rows[s] += 1
            for k, srcs in scenario_defs.items():
                if any(row_hit[s] for s in srcs):
                    scenario_rows[k] += 1

    def pct(n, d):
        return round(100.0 * n / d, 2) if d else 0.0

    report = {
        "schema_version": "asd_source_overlap.v1",
        "method": "DIRECT mmseqs easy-search (source bench = query, ASD uniques = target); "
                  "no connected-component transitive chaining",
        "thresholds": {"heavy": "0.95/0.80", "light": "0.95/0.80", "cdr3_h3": "0.80/0.80"},
        "nbbench": "excluded",
        "asd_rows": n_rows,
        "asd_unique": {b: len(asd[b]) for b in ("heavy", "light", "h3")},
        "bank_sizes": {s: {b: len(banks[s][b]) for b in ("heavy", "light", "h3")} for s in SOURCES},
        "per_source": {
            s: {
                "direct_heavy_hits": len(direct[s]["heavy"]),
                "direct_heavy_pct_of_unique": pct(len(direct[s]["heavy"]), len(asd["heavy"])),
                "direct_light_hits": len(direct[s]["light"]),
                "direct_light_pct_of_unique": pct(len(direct[s]["light"]), len(asd["light"])),
                "direct_h3_hits": len(direct[s]["h3"]),
                "direct_h3_pct_of_unique": pct(len(direct[s]["h3"]), len(asd["h3"])),
                "exact_heavy_hits": exact[s]["heavy"],
                "exact_light_hits": exact[s]["light"],
                "exact_h3_hits": exact[s]["h3"],
                "rows_removed_if_only_this_source": per_source_rows[s],
                "rows_removed_pct": pct(per_source_rows[s], n_rows),
                "rows_kept_if_only_this_source": n_rows - per_source_rows[s],
            }
            for s in SOURCES
        },
        "scenarios": {
            k: {"protected": list(v), "rows_removed": scenario_rows[k],
                "rows_removed_pct": pct(scenario_rows[k], n_rows),
                "rows_kept": n_rows - scenario_rows[k]}
            for k, v in scenario_defs.items()
        },
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
