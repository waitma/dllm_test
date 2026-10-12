"""Audit foundation-model exposure without modifying downstream splits.

Run in conda pllm, from the repository root:
  python scripts/data/audit_tcr_exposure.py --stage freeze
  python scripts/data/audit_tcr_exposure.py --stage near --workers 4
  python scripts/data/audit_tcr_exposure.py --stage scan --data-dir data/prepared/immune_v7_clean_r2

The short-sequence screen is exhaustive normalized Levenshtein similarity >=.8,
not MMseqs identity or a claim of biological novelty. Every support/query pool
is protected; supervised downstream adaptation remains a separate stage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.prepare_immune_v7 import (
    BioSeqRecord, _h3_core_from_chain, canon_core, observe, sha, write_json,
)

DEFAULT_WORK = ROOT / "data/dedup/tcr_exposure_20260930_r2"
INVENTORY = ROOT / "data/dedup/immune_v7_20260930"
SCOPE = "tcr_all_support_query_uptcr_v1"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
ROLES = ("cdr3a", "cdr3b", "fva", "fvb", "peptide", "ab", "ap", "bp", "abp", "bpmhc")


def sequence(value):
    value = str(value or "").strip().upper()
    return value if value and set(value) <= AA else ""


def compound(keys):
    for name, parts in {"ab": ("cdr3a", "cdr3b"), "ap": ("cdr3a", "peptide"),
                        "bp": ("cdr3b", "peptide"), "abp": ("cdr3a", "cdr3b", "peptide"),
                        "bpmhc": ("cdr3b", "peptide", "mhc")}.items():
        if all(keys.get(part) for part in parts):
            keys[name] = "|".join(keys[part] for part in parts)
    return keys


def benchmark_keys(row, kind):
    """Column provenance determines junction/core semantics, never end letters alone."""
    if kind == "uptcr":
        columns = dict(cdr3a="TRA.CDR3", cdr3b="TRB.CDR3", fva="FV_alpha",
                       fvb="FV_beta", peptide="Epitope", mhc="MHCseq")
    else:
        columns = dict(cdr3a="cdr3a", cdr3b="cdr3b", fva="tcra", fvb="tcrb",
                       peptide="peptide", mhc="mhc")
        if "epitope" in row and "peptide" not in row:
            columns["peptide"] = "epitope"
    keys, issues = {}, Counter()
    for role, col in columns.items():
        raw = str(row.get(col) or "").strip()
        s = sequence(raw)
        if raw and not s:
            # HLA allele names are not sequence keys; no guessed allele mapping.
            issues["non_sequence_" + role] += 1
        if s and role.startswith("cdr3") and kind != "immrep_core":
            if not (s.startswith("C") and s.endswith(("F", "W"))):
                issues["noncanonical_junction_" + role] += 1
            s = canon_core(s, has_anchors=True)
        if s:
            keys[role] = s
    return compound(keys), issues


def input_specs():
    """Explicit pools: no substring-based train/background exclusions."""
    bench = ROOT / "downstream/benchmark/data"
    specs = []
    up = ROOT / "output/research/uptcr_20260930/released_tables/data/finetune/abpm"
    for track in ("majority", "few-shot", "zero-shot"):
        for split in ("train", "valid", "test"):
            specs.append((f"uptcr/{track}/{split}", up / track / f"{split}.csv", "uptcr",
                          "support_or_adaptation" if split == "train" else split))
    for task, names in {
        "tcr_representation": ("train", "test"),
        "tcr_representation_paper6": ("target_binders", "background_pool"),
        "tcr_clustering": ("tcrs",), "tcr_clustering_embed": ("tcrs",),
        "tcr_binding": ("test",),
    }.items():
        for name in names:
            usage = ("support" if name == "train" and task == "tcr_representation" else
                     "support_query_universe" if task.endswith("paper6") else name)
            kind = "immrep_core" if task in {"tcr_representation", "tcr_binding"} else "generic"
            specs.append((f"{task}/{name}", bench / task / f"{name}.csv", kind, usage))
    for path in sorted((bench / "tcr_binding_nm2025").rglob("test.csv")):
        specs.append(("nm2025/" + str(path.relative_to(bench / "tcr_binding_nm2025")),
                      path, "generic", "test"))
    if not any(name.startswith("nm2025/") for name, *_ in specs):
        raise ValueError("Missing NM2025 test inputs")
    return specs


def frozen(work):
    registry = json.loads((work / "registry.json").read_text())
    if registry["scope_version"] != SCOPE or registry["script_sha256"] != sha(Path(__file__)):
        raise ValueError("Audit implementation/scope changed; use a new work directory")
    for item in registry["inputs"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Benchmark input changed: " + item["path"])
    if sha(work / "banks.json") != registry["banks_sha256"]:
        raise ValueError("Frozen audit banks changed")
    return registry, json.loads((work / "banks.json").read_text())


def freeze(work):
    if (work / "registry.json").exists():
        frozen(work)
        return
    banks, inputs = {}, []
    for name, path, kind, usage in input_specs():
        collected, issues, rows = defaultdict(set), Counter(), 0
        with path.open() as handle:
            for row in csv.DictReader(handle):
                keys, warnings = benchmark_keys(row, kind)
                issues.update(warnings)
                for role in ROLES:
                    if role in keys:
                        collected[role].add(keys[role])
                rows += 1
        banks[name] = {role: sorted(collected[role]) for role in ROLES}
        inputs.append(dict(cohort=name, path=str(path.resolve()), sha256=sha(path), kind=kind,
                           usage=usage, rows=rows, unique={k: len(v) for k, v in banks[name].items()},
                           field_issues=dict(issues)))
    write_json(work / "banks.json", banks)
    write_json(work / "registry.json", dict(scope_version=SCOPE, script_sha256=sha(Path(__file__)),
        created_at=datetime.now(timezone.utc).isoformat(), inputs=inputs,
        banks_sha256=sha(work / "banks.json"),
        policy="protect entire downstream support/adaptation/validation/query pools from foundation exposure; preserve official downstream rows",
        near="exhaustive global Levenshtein: 1-distance/max_length >= 0.80; implies length ratio >= 0.80",
        limitations=["full FV exact only; no full FV homology screen", "no inferred HLA mapping or parent-antigen identity",
                     "noncanonical junctions use documented endpoint normalization and are counted for review"] ))
    print(json.dumps({"cohorts": len(inputs), "rows": sum(i["rows"] for i in inputs)}), flush=True)


def inverted(banks):
    names = sorted(banks)
    index = {r: defaultdict(int) for r in ROLES}
    for i, name in enumerate(names):
        for role, values in banks[name].items():
            for value in values:
                index[role][value] |= 1 << i
    return names, index


def bit_indices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def near(work, workers):
    import numpy as np
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    import rapidfuzz
    _, banks = frozen(work)
    names, index = inverted(banks)
    inv = json.loads((INVENTORY / "inventory.json").read_text())
    db = sqlite3.connect(f"file:{INVENTORY / 'inventory.sqlite'}?mode=ro", uri=True)
    identity = dict(registry_sha256=sha(work / "registry.json"), inventory_sha256=sha(INVENTORY / "inventory.json"),
                    inventory_sqlite_sha256=sha(INVENTORY / "inventory.sqlite"),
                    source_manifest_sha256=inv["manifest_sha256"], rapidfuzz=rapidfuzz.__version__,
                    workers=workers, cutoff=.8, metric="normalized_Levenshtein", cohorts=names)
    write_json(work / "near_run.json", identity)
    for role in ("peptide", "cdr3a", "cdr3b"):
        out = work / f"near_{role}.jsonl"
        done = work / f"near_{role}.summary.json"
        if done.exists():
            previous = json.loads(done.read_text())
            if previous["identity"] != identity or previous["output_sha256"] != sha(out):
                raise ValueError("Stale near-match output")
            continue
        refs = sorted(index[role])
        ref_masks = [index[role][s] for s in refs]
        scanned = matched = 0
        start = time.monotonic()
        with out.with_suffix(".tmp").open("w") as handle:
            cursor = db.execute("SELECT sequence FROM sequences WHERE role=?", (role,))
            while batch := cursor.fetchmany(2048):
                seqs = [s for s, in batch]
                scores = process.cdist(seqs, refs, scorer=Levenshtein.normalized_similarity,
                                       score_cutoff=.8, workers=workers, dtype=np.float32)
                for i in np.flatnonzero(scores.max(axis=1) > 0):
                    positions = np.flatnonzero(scores[i] > 0)
                    mask = 0
                    for j in positions:
                        mask |= ref_masks[j]
                    best = int(scores[i].argmax())
                    handle.write(json.dumps(dict(sequence=seqs[i], cohorts=mask,
                        similarity=float(scores[i, best]), witness=refs[best])) + "\n")
                    matched += 1
                scanned += len(seqs)
                if scanned % (2048 * 50) == 0:
                    print(role, scanned, matched, round(time.monotonic() - start, 1), flush=True)
        out.with_suffix(".tmp").replace(out)
        write_json(done, dict(identity=identity, scanned=scanned, matched=matched,
                             elapsed_seconds=time.monotonic() - start, output_sha256=sha(out)))
        print(role, "complete", scanned, matched, flush=True)


def prepared_keys(row):
    keys, antigens, issues = {}, [], Counter()
    record = BioSeqRecord.from_prepared_dict(row)
    for chain in record.chains:
        role = chain.role
        if role not in {"tcr_alpha", "tcr_beta", "peptide", "mhc", "antigen"}:
            continue
        obs = observe(chain)
        if not obs.sequence:
            continue
        if role in {"tcr_alpha", "tcr_beta"}:
            short = "cdr3a" if role == "tcr_alpha" else "cdr3b"
            core = chain.regions.get("CDR3") or (obs.sequence if obs.scope == "core" else "")
            if not core and obs.scope == "junction":
                core = canon_core(obs.sequence, has_anchors=True)
            if not core:
                core = record.identifiers.get(short + "_core", "")
            if not core and obs.scope == "full":
                core = _h3_core_from_chain(obs.sequence)
            if sequence(core):
                keys[short] = core
            else:
                issues["missing_or_invalid_" + short] += 1
            if obs.scope == "full" and sequence(obs.sequence):
                keys["fva" if role == "tcr_alpha" else "fvb"] = obs.sequence
        elif role == "antigen":
            antigens.append(obs.sequence)
        elif sequence(obs.sequence):
            keys[role] = obs.sequence
        else:
            issues["invalid_" + role] += 1
    return compound(keys), antigens, issues


def scan(work, data_dir):
    """Rescan actual rows; count joint matches only when found in the same row."""
    registry, banks = frozen(work)
    names, exact = inverted(banks)
    near_index = {}
    for role in ("peptide", "cdr3a", "cdr3b"):
        report = json.loads((work / f"near_{role}.summary.json").read_text())
        if report["identity"]["registry_sha256"] != sha(work / "registry.json"):
            raise ValueError("Near screen belongs to another registry")
        path = work / f"near_{role}.jsonl"
        if sha(path) != report["output_sha256"]:
            raise ValueError("Near screen was modified")
        with path.open() as handle:
            near_index[role] = {r["sequence"]: r["cohorts"] for r in map(json.loads, handle)}
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_hash = report["identity"]["source_manifest_sha256"]
    if sha(manifest_path) != source_hash and manifest.get("decontamination", {}).get("source_manifest_sha256") != source_hash:
        raise ValueError("Dataset was not covered by the near-screen inventory")
    # Bind the source scan to the inventory, including v6 manifests without shard hashes.
    inv = json.loads((INVENTORY / "inventory.json").read_text())
    if sha(INVENTORY / "inventory.json") != report["identity"]["inventory_sha256"]:
        raise ValueError("Source inventory changed")
    expected_source = {path: digest for path, digest, *_ in inv["shards"]}
    out = work / data_dir.name
    out.mkdir(parents=True, exist_ok=True)
    counts, totals, issues, unique = defaultdict(Counter), Counter(), Counter(), defaultdict(set)
    ag_cache = {}
    shard_evidence = []
    started = time.monotonic()
    hit_path = out / "row_hits.jsonl"
    with hit_path.with_suffix(".tmp").open("w") as hits:
        for split, spec in manifest["splits"].items():
            for shard in spec["shards"]:
                path = data_dir / shard["path"]
                bucket = split + "/" + shard["source"]
                h, n = hashlib.sha256(), 0
                with path.open("rb") as handle:
                    for line in handle:
                        h.update(line)
                        if not line.strip():
                            continue
                        n += 1
                        row = json.loads(line)
                        keys, antigens, warnings = prepared_keys(row)
                        issues.update(warnings)
                        reasons, mask = {}, 0
                        for role in ROLES:
                            key = keys.get(role)
                            if not key:
                                continue
                            totals[bucket + "/observed_" + role] += 1
                            m = exact[role].get(key, 0)
                            if m:
                                reasons["exact_" + role] = m
                                for i in bit_indices(m):
                                    unique[(names[i], split, role)].add(key)
                        for role, lookup in near_index.items():
                            m = lookup.get(keys.get(role), 0)
                            if m:
                                reasons["edit80_" + role] = m
                        am = 0
                        for ag in antigens:
                            if ag not in ag_cache:
                                ag_mask = 0
                                for peptide, pm in exact["peptide"].items():
                                    if peptide in ag:
                                        ag_mask |= pm
                                ag_cache[ag] = ag_mask
                            am |= ag_cache[ag]
                        if am:
                            reasons["exact_epitope_in_antigen"] = am
                        for reason, m in reasons.items():
                            mask |= m
                            for i in bit_indices(m):
                                counts[(names[i], bucket)][reason] += 1
                        # Receptor-only and epitope-only exposure are distinct from same-pair exposure.
                        rm = reasons.get("edit80_cdr3a", 0) | reasons.get("edit80_cdr3b", 0) | reasons.get("exact_fva", 0) | reasons.get("exact_fvb", 0)
                        pm = reasons.get("edit80_peptide", 0) | am
                        for label, m in (("receptor_or_epitope", rm | pm), ("epitope_without_near_receptor", pm & ~rm)):
                            for i in bit_indices(m):
                                counts[(names[i], bucket)][label] += 1
                        totals[bucket + "/rows"] += 1
                        if mask:
                            totals[bucket + "/flagged_rows"] += 1
                            hits.write(json.dumps(dict(shard=shard["path"], row=n - 1, reasons=reasons)) + "\n")
                expected = shard.get("sha256") or expected_source.get(str(path.resolve()))
                if expected != h.hexdigest() or n != shard["records"]:
                    raise ValueError("Training shard changed or is not hash-bound: " + str(path))
                shard_evidence.append(dict(path=str(path.resolve()), rows=n, sha256=h.hexdigest()))
                write_json(out / "progress.json", dict(last_shard=shard["path"], totals=dict(totals), elapsed_seconds=time.monotonic()-started))
                print(data_dir.name, shard["path"], n, "done", flush=True)
    hit_path.with_suffix(".tmp").replace(hit_path)
    flagged = sum(v for k, v in totals.items() if k.endswith("/flagged_rows"))
    write_json(out / "report.json", dict(scope_version=SCOPE,
        status="overlap_found" if flagged else "review_required", created_at=datetime.now(timezone.utc).isoformat(),
        dataset_manifest_sha256=sha(manifest_path), registry_sha256=sha(work / "registry.json"),
        totals=dict(totals), field_issues=dict(issues), cohorts=names, shards=shard_evidence,
        counts=[dict(cohort=c, foundation_source_split=b, rows=dict(v)) for (c, b), v in sorted(counts.items())],
        unique_exact=[dict(cohort=c, foundation_split=s, role=r, count=len(v)) for (c,s,r),v in sorted(unique.items())],
        row_hits_sha256=sha(hit_path), elapsed_seconds=time.monotonic()-started,
        pending=["new_scope_foundation_cleanup", "review_noncanonical_benchmark_fields", "upstream_parent_antigen_provenance"],
        limitations=registry["limitations"]))
    print(data_dir.name, "COMPLETE", "flagged_rows", flagged, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "near", "scan"), required=True)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/prepared/immune_v7_clean_r2")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.work)
    elif args.stage == "near":
        near(args.work, args.workers)
    else:
        scan(args.work, args.data_dir.resolve())


if __name__ == "__main__":
    main()
