"""Build sequence-first MHC dataset versions without replacing old corpora.

Run from the project root after activating pllm:
  python scripts/data/update_mhc_inputs.py
Use --verify-only in the production environment for independent loader/layout checks.
Unchanged shards are relative symlinks to the immutable parent; changed shards are
new files. Row membership, order, split, labels, and non-MHC chains are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dllm.pipelines.immune_llada.data.mhc import POLICY, PseudoTable, resolve_mhc
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, normalize_sequence
from dllm.pipelines.immune_llada.data.sources import row_to_record
from scripts.data.tcr_native.common import record_id

WORK = ROOT / "data/mhc_sequence_v1"
REF_SOURCE = ROOT / "output/research/uptcr_20260930/hla_reference/dtu_MHC_pseudo.dat"
REFERENCE = WORK / "reference/MHC_pseudo.dat"
FULL = ROOT / "data/tcr_pmhc_fulllength/records_train.jsonl"
RAW = {"trait": ROOT / "downstream/trait/step4_final",
       "tcr_native": ROOT / "data/tcr_native/dataset",
       "tcr_papers": ROOT / "data/tcr_papers_v2/dataset"}
VERSIONS = {"immune_v6_binding_only": "immune_v6_mhc_sequence_v1",
            "immune_v7_clean_r2": "immune_v7_mhc_sequence_v1"}
NEW_COLUMNS = ("mhc_seq_original", "mhc_chains_json", "mhc_resolution_json")
csv.field_size_limit(2**31 - 1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def identity(row, group):
    """Exact observed inputs plus old MHC and origin, independent of split/canvas."""
    seqs = defaultdict(list)
    for i, (role, sequence) in enumerate(zip(row["chain_roles"], row["chains"])):
        mask = row.get("chain_metadata", {}).get(str(i), {}).get("synthetic_residue_mask")
        if mask is not None:
            if len(mask) != len(sequence):
                raise ValueError("Bad synthetic mask")
            sequence = "".join(s for s, flag in zip(sequence, mask) if not flag)
        if sequence:
            seqs[role].append(sequence)
    payload = [group, row.get("metadata", {}).get("source", row["source"]),
               dict(seqs), row.get("labels", {}).get("relation")]
    return hashlib.sha256(dumps(payload).encode()).hexdigest()


def non_mhc(row):
    """Invariant checked for every output row, including all chain-indexed metadata."""
    items = []
    for i, (role, seq) in enumerate(zip(row["chain_roles"], row["chains"])):
        if role == "mhc":
            continue
        items.append([role, seq, row.get("regions", {}).get(str(i), {}),
                      row.get("chain_metadata", {}).get(str(i), {})])
    meta = {k: v for k, v in row.get("metadata", {}).items() if k != "mhc_update"}
    return [items, row.get("split"), row.get("labels"), row.get("identifiers"),
            row.get("source"), row.get("weight"), meta]


def replace_mhc(row, entries, info):
    result = deepcopy(row)
    kept = [i for i, role in enumerate(row["chain_roles"]) if role != "mhc"]
    result["chains"] = [e["sequence"] for e in entries] + [row["chains"][i] for i in kept]
    result["chain_roles"] = ["mhc"] * len(entries) + [row["chain_roles"][i] for i in kept]
    for field in ("regions", "chain_metadata"):
        old = row.get(field, {})
        result[field] = {str(j + len(entries)): old[str(i)] for j, i in enumerate(kept) if str(i) in old}
    for i, entry in enumerate(entries):
        result["chain_metadata"][str(i)] = deepcopy(entry["metadata"])
    result["task_type"] = "tcr_pmhc" if entries else "tcr_epitope"
    result.setdefault("metadata", {})["mhc_update"] = info
    if non_mhc(result) != non_mhc(row):
        raise AssertionError("Non-MHC content changed")
    return result


def legacy_full_index():
    result = {}
    with FULL.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            rid = record_id("full", i)
            mhcs = [seq for role, seq in zip(row["chain_roles"], row["chains"]) if role == "mhc"]
            pep = next(seq for role, seq in zip(row["chain_roles"], row["chains"]) if role == "peptide")
            result[rid] = (mhcs, pep, i)
    return result


def source_proposal(row, table, full):
    provided = []
    restored_index = None
    allele = row.get("mhc_allele_norm", "")
    if row.get("mhc_chains_json"):
        for entry in json.loads(row["mhc_chains_json"]):
            metadata = entry["metadata"]
            provided.append(dict(sequence=entry["sequence"],
                representation=metadata["mhc_representation"], origin=metadata["mhc_origin"],
                component=metadata.get("mhc_component", "mhc"), scope=metadata.get("sequence_scope")))
    elif row.get("source") == "tcr_pmhc_fulllength":
        sequences, pep, index = full[row["record_id"]]
        restored_index = index
        if normalize_sequence(row["epitope_seq"]) != pep:
            raise ValueError("Legacy record ID/peptide mismatch; cannot recover source MHC")
        for i, seq in enumerate(sequences):
            provided.append(dict(sequence=seq, representation="provided_sequence",
                                 origin="legacy_imgt_allele_resolution" if i == 0 else "legacy_B2M_reference",
                                 component="mhc_heavy" if i == 0 else "b2m", scope="full"))
    elif row.get("mhc_seq"):
        seq = row["mhc_seq"]
        # These legacy CSVs explicitly documented 34-aa pseudo inputs.
        provided = [dict(sequence=seq, representation="existing_pseudo34" if len(seq) == 34 else "provided_sequence",
                         origin="existing_source_csv")]
    out = resolve_mhc(provided=provided, allele=allele, table=table)
    if restored_index is not None:
        out["resolution"].update(restored_from=str(FULL), restored_source_row=restored_index,
                                 origin_note="Previously allele-resolved reference sequences, not experimental observations")
    return out


def build_sources():
    if (WORK / "source_report.json").exists():
        raise FileExistsError("Source build already exists; use --reuse-sources")
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REF_SOURCE, REFERENCE)
    table = PseudoTable(REFERENCE)
    full = legacy_full_index()
    report = {"policy": POLICY, "created_at": datetime.now(timezone.utc).isoformat(),
              "reference": {"path": str(REFERENCE), "sha256": table.sha256,
                            "url": "https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/data.tar.gz"},
              "full_source": {"path": str(FULL), "sha256": sha(FULL)}, "files": {}}
    for group, base in RAW.items():
        for split in ("train", "valid"):
            path = base / f"{split}.csv"
            output = WORK / "sources" / group / path.name
            if output.exists():
                raise FileExistsError(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            counts = Counter()
            before = sha(path)
            with path.open() as f, output.open("w", newline="") as out:
                reader = csv.DictReader(f)
                writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(list(reader.fieldnames) + list(NEW_COLUMNS))))
                writer.writeheader()
                for row in reader:
                    proposal = source_proposal(row, table, full)
                    seqs = proposal["chains"]
                    row["mhc_seq_original"] = row.get("mhc_seq", "")
                    row["mhc_seq"] = seqs[0]["sequence"] if seqs else ""
                    row["mhc_chains_json"] = dumps(seqs)
                    row["mhc_resolution_json"] = dumps(proposal["resolution"])
                    row["task_type"] = "tcr_pmhc" if seqs else "tcr_epitope"
                    if "tier" in row:
                        row["tier"] = ("A" if row.get("alpha_fv") or row.get("cdr3a") else "B") if seqs else "C"
                    writer.writerow(row)
                    counts["rows"] += 1
                    counts[proposal["resolution"]["status"]] += 1
                    counts["restored_long_mhc"] += int(row.get("source") == "tcr_pmhc_fulllength")
            if before != sha(path):
                raise ValueError("Raw source changed during build")
            report["files"][f"{group}/{split}"] = dict(input=str(path), input_sha256=before,
                output=str(output), output_sha256=sha(output), counts=dict(counts))
            print(group, split, dict(counts), flush=True)
    write_json(WORK / "source_report.json", report)


def build_lookup():
    """Index legacy inputs. Distinct sequence proposals remain separate candidates."""
    lookup = {}
    report = json.loads((WORK / "source_report.json").read_text())
    for label, info in report["files"].items():
        group, split = label.split("/")
        if sha(info["input"]) != info["input_sha256"] or sha(info["output"]) != info["output_sha256"]:
            raise ValueError("Source files changed")
        with open(info["input"]) as src, open(info["output"]) as dest:
            left, right = csv.DictReader(src), csv.DictReader(dest)
            for index, old in enumerate(left):
                new = next(right)
                record = row_to_record(group, old, split=split)
                if record is None:
                    continue
                key = identity(record.to_dict(), group)
                entries = json.loads(new["mhc_chains_json"])
                candidate = dumps(entries)
                if key not in lookup:
                    lookup[key] = {}
                variants = lookup[key]
                if candidate not in variants:
                    variants[candidate] = {"count": 0, "alleles": set(), "statuses": set(), "source_files": set()}
                item = variants[candidate]
                item["count"] += 1
                item["alleles"].add(old.get("mhc_allele_norm", ""))
                item["statuses"].add(json.loads(new["mhc_resolution_json"])["status"])
                item["source_files"].add(label)
            if next(right, None) is not None:
                raise ValueError("Updated source row count differs")
        print("indexed", label, len(lookup), flush=True)
    return lookup


def export_version(parent_name, name, lookup):
    parent = ROOT / "data/prepared" / parent_name
    output = ROOT / "data/prepared" / name
    if output.exists():
        raise FileExistsError(output)
    source_manifest = parent / "dataset_manifest.json"
    manifest_sha = sha(source_manifest)
    manifest = json.loads(source_manifest.read_text())
    new = deepcopy(manifest)
    output.mkdir(parents=True)
    counts = defaultdict(Counter)
    unresolved = Counter()
    ledger = (output / "mhc_unresolved.jsonl").open("w")
    for split, block in new["splits"].items():
        for shard in block["shards"]:
            source = parent / shard["path"]
            dest = output / shard["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            group = shard["source"]
            if group not in RAW:
                dest.symlink_to(os.path.relpath(source, dest.parent))
                continue
            stat = counts[f"{split}/{group}"]
            n = 0
            digest = hashlib.sha256()
            with source.open("rb") as f, dest.open("w") as out:
                for local_index, line in enumerate(f):
                    digest.update(line)
                    row = json.loads(line)
                    key = identity(row, group)
                    variants = lookup.get(key, {})
                    if not variants:
                        raise ValueError(f"No exact source match: {shard['path']}:{local_index}")
                    if len(variants) != 1:
                        # Preserve the old MHC when provenance cannot choose among supplied sequences.
                        entries = []
                        for role, seq in zip(row["chain_roles"], row["chains"]):
                            if role == "mhc":
                                entries.append(dict(sequence=seq, metadata={"mhc_policy": POLICY,
                                    "mhc_representation": "existing_pseudo34", "mhc_origin": "ambiguous_raw_match_preserved",
                                    "mhc_component": "mhc", "sequence_scope": "fragment"}))
                        info = dict(policy=POLICY, status="ambiguous_raw_match_preserved", candidate_count=len(variants))
                    else:
                        serialized, evidence = next(iter(variants.items()))
                        entries = json.loads(serialized)
                        info = dict(policy=POLICY, status="source_matched", source_match_count=evidence["count"],
                                    allele_candidates=sorted(evidence["alleles"]),
                                    resolution_statuses=sorted(evidence["statuses"]),
                                    source_files=sorted(evidence["source_files"]))
                    updated = replace_mhc(row, entries, info)
                    old_mhc = [s for role, s in zip(row["chain_roles"], row["chains"]) if role == "mhc"]
                    new_mhc = [e["sequence"] for e in entries]
                    stat["rows"] += 1
                    stat["with_mhc"] += bool(entries)
                    stat["without_mhc"] += not entries
                    stat["sequence_changed"] += old_mhc != new_mhc
                    stat["new_mhc_context"] += not old_mhc and bool(new_mhc)
                    stat["long_mhc_restored"] += any(len(s) > 34 for s in new_mhc)
                    stat["pseudo34_rows"] += len(new_mhc) == 1 and len(new_mhc[0]) == 34
                    stat["ambiguous_raw_match"] += len(variants) != 1
                    if not entries or len(variants) != 1:
                        reason = info.get("resolution_statuses", [info["status"]])
                        unresolved.update(reason)
                        ledger.write(dumps(dict(shard=shard["path"], row=local_index, identity=key, resolution=info)) + "\n")
                    out.write(dumps(updated) + "\n")
                    n += 1
            assert n == shard["records"], source
            if shard.get("sha256"):
                assert digest.hexdigest() == shard["sha256"], source
            shard["sha256"] = sha(dest)
            shard["parent_sha256"] = digest.hexdigest()
            print(name, shard["path"], n, flush=True)
    ledger.close()
    if sha(source_manifest) != manifest_sha:
        raise ValueError("Parent manifest changed")
    new["dataset_version"] = name
    new["created_at"] = datetime.now(timezone.utc).isoformat()
    new["mhc_update"] = dict(policy=POLICY, parent=str(parent), parent_manifest_sha256=manifest_sha,
        source_report=str(WORK / "source_report.json"), source_report_sha256=sha(WORK / "source_report.json"),
        reference_sha256=sha(REFERENCE), update_scope="MHC context and provenance only; row membership/order/splits/targets unchanged")
    if "decontamination" in new:
        new["decontamination"] = dict(status="audit_pending", parent_evidence=manifest["decontamination"],
            pending=["parent_current_scope_audit_pending", "updated_manifest_not_covered_by_parent_certificate"],
            non_mhc_rows_and_splits_preserved=True)
    # This is a data-format update, not a new decontamination certification.
    for filename in ("schema.json", "quarantine_keys.json"):
        src = parent / filename
        if src.exists():
            shutil.copyfile(src, output / filename)
    write_json(output / "dataset_manifest.json", new)
    write_json(output / "mhc_update_report.json", dict(status="exported_pending_layout_validation", policy=POLICY,
        counts={k: dict(v) for k, v in counts.items()}, unresolved=dict(unresolved),
        parent_manifest_sha256=manifest_sha, manifest_sha256=sha(output / "dataset_manifest.json"),
        invariants="Same row counts/order/splits/labels and exact non-MHC chain content+metadata; checked per row",
        script_sha256=sha(Path(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reuse-sources", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--parent", choices=list(VERSIONS), action="append")
    args = ap.parse_args()
    parents = args.parent or list(VERSIONS)
    if args.verify_only:
        from scripts.data.verify_mhc_inputs import verify
        for parent in parents:
            verify(ROOT / "data/prepared" / VERSIONS[parent])
        return
    if not args.reuse_sources:
        build_sources()
    lookup = build_lookup()
    for parent in parents:
        export_version(parent, VERSIONS[parent], lookup)


if __name__ == "__main__":
    main()
