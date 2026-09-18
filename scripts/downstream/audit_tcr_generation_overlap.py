#!/usr/bin/env python3
"""Read-only TCRT5 reference audit against every manifest-listed prepared shard.

Run in pllm from the repository root:
  python scripts/downstream/audit_tcr_generation_overlap.py --out-dir /absolute/new/report
  python scripts/downstream/audit_tcr_generation_overlap.py --self-test

Only the new report directory is written. No dataset, blocklist or model changes.
Counts concern dataset membership, not proof a particular checkpoint saw a row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.tcr_native.common import cdr3_core

FIELDS = ("epitopes", "pmhc", "beta", "full_beta", "pair", "binding_pair", "pmhc_pair")
RESERVED = "RVRAYTYSK_HLA-A*03:01"


def norm(value):
    return "".join(str(value or "").split()).upper()


def fresh():
    return {name: set() for name in FIELDS}


class Auditor:
    def __init__(self, entries):
        self.entries = entries
        self.refs = []
        self.by_core = defaultdict(list)
        self.by_ep = defaultdict(list)
        for pmhc, entry in entries.items():
            ep = norm(entry["epitope"])
            self.by_ep[ep].append(pmhc)
            for ref in sorted(set(map(norm, entry["ref_binders"]))):
                core = cdr3_core(ref, has_anchors=True)
                if not core:
                    raise ValueError(f"Invalid reference {pmhc}: {ref}")
                index = len(self.refs)
                self.refs.append({"pmhc": pmhc, "epitope": ep, "junction": ref, "core": core})
                self.by_core[core].append(index)
        self.states = defaultdict(fresh)
        self.counters = defaultdict(Counter)
        self.examples = defaultdict(list)
        self.relations = defaultdict(lambda: defaultdict(set))

    def observe(self, row, split, source, location):
        key = f"{split}/{source}"
        state, count = self.states[key], self.counters[key]
        count["rows"] += 1
        # Manifest shard source is authoritative for loader membership. Prepared
        # row.source retains aliases (oas_paired / ots_paired / tcr_native).
        count[f"record_source:{row.get('source', '')}"] += 1
        if row.get("split") != split:
            raise ValueError(f"Shard/record identity mismatch: {location}")
        roles, chains = row["chain_roles"], row["chains"]
        if len(roles) != len(chains):
            raise ValueError(f"Role/chain mismatch: {location}")
        eps = {norm(seq) for role, seq in zip(roles, chains) if role in {"peptide", "epitope", "antigen"}}
        mhcs = {norm(seq) for role, seq in zip(roles, chains) if role.startswith("mhc")}
        target_eps = eps & self.by_ep.keys()
        state["epitopes"].update(target_eps)
        for ep in target_eps:
            for pmhc in self.by_ep[ep]:
                if norm(self.entries[pmhc].get("mhc_pseudo")) in mhcs:
                    state["pmhc"].add(pmhc)
        ids = row.get("identifiers") or {}
        id_ep = norm(ids.get("epitope"))
        if id_ep and id_ep not in eps:
            count["epitope_identifier_mismatch"] += 1
        beta_indices = [i for i, role in enumerate(roles) if role == "tcr_beta"]
        count["beta_rows"] += bool(beta_indices)
        if not beta_indices:
            return
        relation = (row.get("labels") or {}).get("relation", "unknown")
        for i in beta_indices:
            regions = (row.get("regions") or {}).get(str(i), {})
            core = norm(regions.get("CDR3"))
            sequence = norm(chains[i])
            id_core = norm(ids.get("cdr3b_core"))
            junction = ""
            if not core:
                # Real full chains can lack region annotation. Accept only a
                # nonempty annotated core that is present in the actual beta.
                if id_core and "X" not in id_core and id_core in sequence:
                    core = id_core
                    count["beta_core_from_identifier_verified_in_chain"] += 1
                    start = sequence.find(core)
                    end = start + len(core)
                    if start > 0 and sequence[start-1] == "C" and sequence[end:end+1] in {"F", "W"}:
                        junction = sequence[start-1:end+1]
                else:
                    count["beta_unresolved_core"] += 1
                    if len(self.examples[key + "/unresolved"]) < 5:
                        self.examples[key + "/unresolved"].append({"location": location, "identifiers": ids, "sequence": sequence})
                    continue
            if "X" in core:
                count["beta_incomplete_region"] += 1
                continue
            if id_core and id_core != core:
                count["beta_identifier_mismatch"] += 1
            elif not id_core:
                count["beta_missing_identifier_but_region_present"] += 1
            count["beta_complete_core"] += 1
            fr3, fr4 = norm(regions.get("FR3")), norm(regions.get("FR4"))
            if fr3.endswith("C") and fr4[:1] in {"F", "W"}:
                junction = fr3[-1:] + core + fr4[:1]
            for index in self.by_core.get(core, ()):
                ref = self.refs[index]
                state["beta"].add(index)
                if junction == ref["junction"]:
                    state["full_beta"].add(index)
                pair = ref["epitope"] in eps
                if pair:
                    state["pair"].add(index)
                    self.relations[key][relation].add(index)
                    if relation == "binding":
                        state["binding_pair"].add(index)
                    if norm(self.entries[ref["pmhc"]].get("mhc_pseudo")) in mhcs:
                        state["pmhc_pair"].add(index)
                exkey = f"{key}/{'pair' if pair else 'beta_only'}"
                if len(self.examples[exkey]) < 5:
                    self.examples[exkey].append({"location": location, "reference": ref,
                        "record_epitopes": sorted(eps), "record_core": core,
                        "record_junction": junction, "relation": relation,
                        "origin": row.get("metadata", {})})

    def summarize(self, names):
        indices = {i for i, ref in enumerate(self.refs) if ref["pmhc"] in names}
        eps = {norm(self.entries[p]["epitope"]) for p in names}
        result = {"n_targets": len(names), "n_references": len(indices),
                  "n_references_with_nonstandard_core": sum(bool(set(self.refs[i]["core"]) - set("ACDEFGHIKLMNPQRSTVWY")) for i in indices),
                  "n_unique_cores": len({self.refs[i]["core"] for i in indices}), "groups": {}}
        states = dict(self.states)
        for split in ("train", "valid"):
            states[split] = {f: set().union(*(s[f] for k, s in self.states.items() if k.startswith(split + "/"))) for f in FIELDS}
        for key, state in states.items():
            result["groups"][key] = {"epitope_targets": sum(norm(self.entries[p]["epitope"]) in state["epitopes"] for p in names),
                "unique_epitopes": len(eps & state["epitopes"]), "pmhc_targets": len(set(names) & state["pmhc"]),
                **{f + "_references": len(indices & state[f]) for f in FIELDS[2:]}}
        return result


def self_test():
    entries = {"A": {"epitope": "PEPTIDE", "mhc_pseudo": "MHCPSEUDO", "ref_binders": ["CASSFF"]}}
    def row(ep="PEPTIDE", split="train", core="ASSF", anchors=False):
        return {"split": split, "source": "test", "chain_roles": ["tcr_beta", "peptide", "mhc"],
            "chains": ["X" + core + "X", ep, "MHCPSEUDO"],
            "regions": {"0": {"CDR3": core, "FR3": "C" if anchors else "X", "FR4": "F" if anchors else "X"}},
            "identifiers": {"cdr3b_core": core, "epitope": ep}, "labels": {"relation": "binding"}}
    a = Auditor(entries)
    a.observe(row(ep="OTHER"), "train", "test", "test:1")
    a.observe(row(anchors=True, split="valid"), "valid", "test", "test:2")
    a.observe(row(anchors=True, split="valid"), "valid", "test", "test:3")
    stats = a.summarize(["A"])["groups"]
    assert stats["train"]["beta_references"] == 1 and stats["train"]["pair_references"] == 0
    assert stats["train"]["full_beta_references"] == 0
    assert stats["valid"]["binding_pair_references"] == 1 and stats["valid"]["pmhc_pair_references"] == 1
    assert stats["valid"]["full_beta_references"] == 1
    assert cdr3_core("CASSFF", has_anchors=True) == "ASSF"
    b = Auditor(entries)
    b.observe(row(core="XXXX"), "train", "test", "test:4")
    assert b.summarize(["A"])["groups"]["train"]["beta_references"] == 0
    r = row()
    r["regions"] = None
    r["chains"][0] = "LONGCASSFFTAIL"
    b.observe(r, "train", "test", "test:5")
    assert b.summarize(["A"])["groups"]["train"]["full_beta_references"] == 1
    r = row()
    r["identifiers"] = {}
    r["split"] = "valid"
    b.observe(r, "valid", "test", "test:6")
    assert b.summarize(["A"])["groups"]["valid"]["pair_references"] == 1
    print("self-test PASS: anchors, splits, pair/beta, junction, dedup, synthetic X, full-chain fallback, missing ID")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared-data-dir", type=Path, default=ROOT / "data/prepared/immune_v5_receptor_completion")
    ap.add_argument("--eval-json", type=Path, default=ROOT / "downstream/benchmark/data/tcr_generation_bench/eval_conditional.json")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out_dir is None:
        ap.error("--out-dir required")
    manifest_path = args.prepared_data_dir / "dataset_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    eval_bytes = args.eval_json.read_bytes()
    manifest, entries = json.loads(manifest_bytes), json.loads(eval_bytes)
    audit = Auditor(entries)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    shards = []
    # Validation first, then all training sources. No caps and no skipped shards.
    for split in ("valid", "train"):
        for shard in manifest["splits"][split]["shards"]:
            path = args.prepared_data_dir / shard["path"]
            before = path.stat()
            digest = hashlib.sha256()
            n = 0
            with path.open("rb") as handle:
                for lineno, line in enumerate(handle, 1):
                    digest.update(line)
                    if not line.strip():
                        continue
                    n += 1
                    audit.observe(json.loads(line), split, shard["source"], f"{path}:{lineno}")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or n != shard["records"]:
                raise ValueError(f"Changed shard or row-count mismatch: {path}, {n}")
            shards.append({**shard, "absolute_path": str(path), "bytes": before.st_size,
                           "mtime_ns": before.st_mtime_ns, "sha256": digest.hexdigest(), "scanned": n})
            print(f"{split}/{shard['source']}: {path.name} {n:,} rows; elapsed {time.monotonic()-started:.1f}s", flush=True)
    subsets = {cat: [p for p, e in entries.items() if e["category"] == cat] for cat in ("held20", "benchmark14")}
    subsets["paper_sparse13"] = [p for p in subsets["benchmark14"] if p != RESERVED]
    unseen_path = args.eval_json.parent / "bioseq_unseen_pmhc.json"
    unseen = set(json.loads(unseen_path.read_text()))
    subsets["legacy_common6"] = [p for p in subsets["paper_sparse13"] if p in unseen]
    block_path = ROOT / "data/tcr_native/dataset/t4_refbinder_blocklist.txt"
    block_bytes = block_path.read_bytes()
    blocked = set(block_bytes.decode().splitlines())
    integrity_errors = {key: {k: v for k, v in c.items() if "mismatch" in k or k == "beta_unresolved_core"} for key, c in audit.counters.items()}
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "prepared_root": str(args.prepared_data_dir),
        "eval_json": str(args.eval_json), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "eval_sha256": hashlib.sha256(eval_bytes).hexdigest(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": round(time.monotonic()-started, 2), "shards": shards,
        "definitions": {"beta": "Exact anchor-free CDR3beta core, irrespective of epitope or relation",
            "full_beta": "Exact junction reconstructed only when beta regions contain both C and F/W anchors",
            "pair": "Exact (epitope, beta core), ignoring MHC and including all relation labels",
            "binding_pair": "pair with record relation=binding", "pmhc_pair": "pair with identical MHC pseudo sequence in an MHC-role chain",
            "counts": "Reference entries (pMHC,junction), deduplicated within target; cores may collide or repeat across targets",
            "scope": "All selected prepared train/valid rows; not per-checkpoint sampler exposure, near-identity or encoder pretraining audit"},
        "counters": {k: dict(v) for k, v in audit.counters.items()}, "integrity_errors": integrity_errors,
        "summary": {k: audit.summarize(v) for k, v in subsets.items()},
        "per_pmhc": {p: audit.summarize([p]) for p in entries},
        "blocklist": {"path": str(block_path), "sha256": hashlib.sha256(block_bytes).hexdigest(),
            "uncovered_reference_pairs": [r for r in audit.refs if f"{r['core']}|{r['epitope']}" not in blocked]},
        "examples": dict(audit.examples),
        "matched_references": {key: {field: [audit.refs[i] for i in sorted(s[field])] for field in FIELDS[2:]} for key, s in audit.states.items()}}
    if manifest_path.read_bytes() != manifest_bytes or args.eval_json.read_bytes() != eval_bytes:
        raise ValueError("Manifest/eval changed during scan")
    with (args.out_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    for subset, stats in report["summary"].items():
        print(subset, json.dumps({k: stats["groups"][k] for k in ("train", "valid")}))
    print("integrity_errors", json.dumps(integrity_errors))
    print("uncovered_reference_pairs", len(report["blocklist"]["uncovered_reference_pairs"]))
    print("report", args.out_dir / "report.json")
    if any(any(c.values()) for c in integrity_errors.values()):
        raise SystemExit("Nonzero integrity errors; do not claim complete audit")


if __name__ == "__main__":
    main()
