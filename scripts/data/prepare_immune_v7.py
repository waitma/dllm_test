"""Versioned immune_v7 audit/cleaning. Run from the repository root:

python scripts/data/prepare_immune_v7.py --stage freeze
python scripts/data/prepare_immune_v7.py --stage inventory
python scripts/data/prepare_immune_v7.py --stage cluster --threads 8
python scripts/data/prepare_immune_v7.py --stage export

No old datasets/banks are overwritten. Frozen role clustering is an operational
similarity definition, not an exhaustive pairwise or biological novelty proof.
Every phase is resumable only against matching content hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/data/tcr_native"))
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, normalize_sequence
from dllm.pipelines.immune_llada.data.sources import _h3_core_from_chain
from dllm.pipelines.immune_llada.v3.layout import observe, ImmuneCollator
from scripts.data.dedup import build_downstream_banks as legacy
from decontam import canon_core

DEFAULT_WORK = ROOT / "data/dedup/immune_v7_20260930"
DEFAULT_INPUT = ROOT / "data/prepared/immune_v6_binding_only"
DEFAULT_OUTPUT = ROOT / "data/prepared/immune_v7_clean_r2"
MMSEQS = Path("/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/mmseqs")
RULES = {"heavy": .95, "h3": .8, "cdr3b": .8, "cdr3a": .8, "antigen": .7, "peptide": .8}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+"\n")
    os.replace(tmp, path)


def read_rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def duplicate_nanobody_header(row):
    """Recognize the exact repeated header artifact, not arbitrary short AAs."""
    return (str(row.get("seq","")).strip().lower()=="sequence"
            and str(row.get("label","")).strip().lower()=="sequence"
            and all(row.get(name)==name for name in ("CDR1","CDR2","CDR3")))


def covered_containment(sequence,reference,coverage=.8):
    """Exact containment must still satisfy the declared bidirectional coverage.

    A very short fragment is not evidence that every longer protein containing
    those letters is the same antigen; parent-protein identity needs provenance.
    """
    return (bool(sequence) and bool(reference)
            and min(len(sequence),len(reference))>=coverage*max(len(sequence),len(reference))
            and (sequence in reference or reference in sequence))


def freeze(work):
    work.mkdir(parents=True, exist_ok=True)
    if (work / "registry.json").exists():
        verify_registry(work)
        print("Frozen registry already exists and hashes match", flush=True)
        return
    inputs, missing = {}, set()
    old_csv, old_jsonl = legacy._read_csv, legacy._read_jsonl
    def track(reader):
        def wrapped(path):
            path = Path(path)
            if path.is_file():
                inputs[str(path.resolve())] = sha(path)
            else:
                missing.add(str(path))
            yield from reader(path)
        return wrapped
    legacy._read_csv, legacy._read_jsonl = track(old_csv), track(old_jsonl)
    old = legacy.Banks()
    try:
        legacy.build_antibody(old)
        legacy.build_nanobody(old)
        legacy.build_tcr(old)
    finally:
        legacy._read_csv, legacy._read_jsonl = old_csv, old_jsonl
    # The legacy nanobody reader prefers the masked `seq` over its native
    # `label`. Explicitly protect native references as well in NEW registries.
    native_path=ROOT/"data/nanobody_raw/nbbench/hf_data/CDRInfilling/test.csv"
    if native_path.is_file():
        inputs[str(native_path)]=sha(native_path)
        for row in old_csv(native_path):
            if duplicate_nanobody_header(row):
                continue
            native=normalize_sequence(row.get("label"))
            old.add_full(old.ab_heavy,native)
            core=_h3_core_from_chain(native)
            if core:
                old.ab_cdrh3.add(core)
    banks = {role: set() for role in (*RULES, "light", "pair")}
    banks["heavy"].update(old.ab_heavy)
    banks["light"].update(old.ab_light)
    # SAbDab infilling stores ANARCI loop cores, already anchor-free.
    banks["h3"].update(old.ab_cdrh3)
    banks["h3"].update(filter(None, map(_h3_core_from_chain, old.ab_heavy)))
    banks["cdr3b"].update(canon_core(s, has_anchors=True) for s in old.tcr_cdr3b)
    banks["cdr3a"].update(canon_core(s, has_anchors=True) for s in old.tcr_cdr3a)
    counts = {}
    root = ROOT / "data/downstream/ab_antigen_sequence_v1"
    for task in ("binding", "spr", "abbibench", "chimera"):
        path = root / f"{task}.jsonl"
        inputs[str(path)] = sha(path)
        splits = None
        if task in {"binding", "spr"}:
            split_path = root / f"{task}_splits.json"
            inputs[str(split_path)] = sha(split_path)
            splits = json.loads(split_path.read_text())
        selected, ag_selected = 0, 0
        for row in read_rows(path):
            held = task == "abbibench" or (task == "chimera" and row["split"] in {"val", "valid", "test"})
            if splits is not None:
                held = any(axis.get(row["id"]) in {"val", "valid", "test"} for axis in splits.values())
            if not held:
                continue
            selected += 1
            h, l = normalize_sequence(row["heavy"]), normalize_sequence(row["light"])
            banks["heavy"].add(h)
            banks["light"].add(l)
            banks["pair"].add(h+"|"+l)
            for heavy in (h, normalize_sequence(row.get("wt_heavy"))):
                if heavy:
                    banks["heavy"].add(heavy)
                    core = _h3_core_from_chain(heavy)
                    if core:
                        banks["h3"].add(core)
            # AbBiBench assays are NOT declared unseen-antigen by default.
            unseen_ag = task == "chimera" or (splits is not None and splits["antigen"].get(row["id"]) in {"val", "valid", "test"})
            if unseen_ag:
                banks["antigen"].add(normalize_sequence(row["antigen"]))
                ag_selected += 1
        counts[task] = {"protected_rows": selected, "unseen_antigen_rows": ag_selected}
    nm = ROOT / "downstream/benchmark/data/tcr_binding_nm2025/unseen"
    for path in sorted(nm.rglob("test.csv")):
        inputs[str(path)] = sha(path)
        with path.open() as handle:
            for row in csv.DictReader(handle):
                banks["peptide"].add(normalize_sequence(row.get("peptide")))
    for role in banks:
        banks[role].discard("")
        if not banks[role]:
            raise ValueError(f"Required protected bank is empty: {role}")
    registry = dict(schema_version="immune_v7.registry.v1", inputs=inputs,
                    missing_legacy_inputs=sorted(missing), legacy_provenance=old.provenance,
                    tasks=counts, counts={r:len(s) for r,s in banks.items()},
                    rules={r:{"identity":v,"coverage":.8,"coverage_mode":0} for r,v in RULES.items()},
                    limitations=["Legacy readers cover available registered immune benchmarks; missing inputs are unresolved, not clean.",
                                 "Source-protein accession mappings are not present in the prepared semantic corpus.",
                                 "Core-only input anchors are not imputed."])
    bank_path = work / "banks.json"
    write_json(bank_path, {r:sorted(s) for r,s in banks.items()})
    registry["banks_sha256"] = sha(bank_path)
    write_json(work / "registry.json", registry)
    print(json.dumps({"bank_counts":registry["counts"], "missing_legacy_inputs":len(missing)}), flush=True)


def verify_registry(work):
    registry = json.loads((work / "registry.json").read_text())
    if sha(work / "banks.json") != registry["banks_sha256"]:
        raise ValueError("Frozen banks changed")
    for path, expected in registry["inputs"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen benchmark input changed: {path}")
    return registry


def extract_keys(row):
    """Return role keys without changing model input sequence or inventing anchors."""
    record = BioSeqRecord.from_prepared_dict(row)
    keys = defaultdict(set)
    unresolved = []
    full = {}
    for chain in record.chains:
        obs = observe(chain)
        if not obs.sequence:
            continue
        role = chain.role
        if role in {"antibody_heavy", "nanobody_vhh"}:
            keys["heavy"].add(obs.sequence)
            core = record.identifiers.get("heavy_h3_core") or _h3_core_from_chain(obs.sequence)
            if core:
                keys["h3"].add(core)
            else:
                unresolved.append("missing_h3")
            full["heavy"] = obs.sequence
        elif role == "antibody_light":
            keys["light"].add(obs.sequence)
            full["light"] = obs.sequence
        elif role in {"tcr_alpha", "tcr_beta"}:
            bucket = "cdr3a" if role == "tcr_alpha" else "cdr3b"
            core = chain.regions.get("CDR3") or (obs.sequence if obs.scope == "core" else "")
            if not core and obs.scope == "junction":
                core = canon_core(obs.sequence, has_anchors=True)
            if not core:
                core = record.identifiers.get(bucket+"_core", "")
            if not core and obs.scope == "full":
                # Native full-Fv records can lack region columns. Use the same
                # conserved-junction motif extractor, never strip endpoint letters
                # from an already core-only fragment.
                core = _h3_core_from_chain(obs.sequence)
            if core:
                keys[bucket].add(core)
            else:
                unresolved.append("missing_"+bucket)
        elif role in {"antigen", "peptide"}:
            keys[role].add(obs.sequence)
    if full.get("heavy") and full.get("light"):
        keys["pair"].add(full["heavy"]+"|"+full["light"])
    return dict(keys), unresolved


def source_shards(input_dir):
    manifest = json.loads((input_dir / "dataset_manifest.json").read_text())
    return manifest, [s for split in manifest["splits"].values() for s in split["shards"]]


def audit_h3_coverage(work):
    """Resolve masked benchmark inputs only with already-protected source cores.

    A masked input is not a native full chain. Every occurrence must have an
    explicit source core (or native reference motif), already in the frozen
    effective bank. This changes no sequence, split, bank, or clustering.
    """
    registry = verify_registry(work)
    annotations = json.loads((work/"benchmark_h3_annotations.json").read_text())
    banks = json.loads((work/"effective_banks.json").read_text())
    unresolved, protected = set(annotations["unresolved"]), set(banks["h3"])
    evidence, failed = defaultdict(list), set()
    for filename in registry["inputs"]:
        path = Path(filename)
        if "nanobody_raw/nbbench/hf_data/" not in filename or path.name != "test.csv":
            continue
        with path.open() as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                seq = next((str(row[col]).replace("-", "") for col in ("seq","VHH_sequence","seq_nanobody") if row.get(col)), "")
                if not seq and path.parent.name == "CDRInfilling":
                    seq = str(row.get("label", ""))
                seq = normalize_sequence(seq)
                if seq not in unresolved:
                    continue
                core = normalize_sequence(row.get("CDR3")) or _h3_core_from_chain(normalize_sequence(row.get("label")))
                if core and core in protected:
                    evidence[seq].append(dict(source=filename,row=index,core=core))
                else:
                    failed.add(seq)
    covered = set(evidence)-failed
    # The legacy reader opens this one text file directly, bypassing CSV/JSONL
    # tracking. Bind it here and require that its content was already banked.
    extra = ROOT/"downstream/benchmark/data/tcr_generation/holdout_cdr3b.txt"
    extra_cores = {canon_core(normalize_sequence(s),has_anchors=True) for s in extra.read_text().splitlines() if s.strip()}
    if not extra_cores <= set(banks["cdr3b"]):
        raise ValueError("Unregistered TCR holdout content differs from frozen bank")
    report = dict(status="complete", registry_sha256=sha(work/"registry.json"),
                  annotations_sha256=sha(work/"benchmark_h3_annotations.json"),
                  effective_banks_sha256=sha(work/"effective_banks.json"), extra_inputs={str(extra):sha(extra)},
                  source_covered={s:evidence[s] for s in sorted(covered)}, unresolved=sorted(unresolved-covered))
    path = work/"benchmark_h3_coverage.json"
    if path.exists() and json.loads(path.read_text()) != report:
        raise ValueError("Source coverage evidence changed; use a new audit version")
    write_json(path,report)
    print(json.dumps({"source_covered":len(covered),"unresolved":len(report["unresolved"])}),flush=True)


def verify_h3_coverage(work):
    report = json.loads((work/"benchmark_h3_coverage.json").read_text())
    for key,filename in (("registry_sha256","registry.json"),("annotations_sha256","benchmark_h3_annotations.json"),
                         ("effective_banks_sha256","effective_banks.json")):
        if report[key] != sha(work/filename):
            raise ValueError("Stale source H3 coverage evidence")
    for path,expected in report["extra_inputs"].items():
        if sha(path) != expected:
            raise ValueError("Additional benchmark input changed")
    return report


def inventory(work, input_dir):
    verify_registry(work)
    db = sqlite3.connect(work / "inventory.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-524288")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("CREATE TABLE IF NOT EXISTS sequences(role TEXT, sequence TEXT, PRIMARY KEY(role,sequence)) WITHOUT ROWID")
    db.execute("CREATE TABLE IF NOT EXISTS shards(path TEXT PRIMARY KEY, sha256 TEXT, records INTEGER, unresolved TEXT, key_version INTEGER DEFAULT 1)")
    if "key_version" not in {r[1] for r in db.execute("PRAGMA table_info(shards)")}:
        db.execute("ALTER TABLE shards ADD COLUMN key_version INTEGER DEFAULT 1")
    manifest, shards = source_shards(input_dir)
    for shard in shards:
        path = input_dir / shard["path"]
        prior = db.execute("SELECT sha256,unresolved,key_version FROM shards WHERE path=?", (str(path),)).fetchone()
        if prior:
            if prior[0] != sha(path):
                raise ValueError(f"Prepared shard changed: {path}")
            if prior[2] >= 2 or not any(k.startswith("missing_cdr3") for k in json.loads(prior[1])):
                db.execute("UPDATE shards SET key_version=2 WHERE path=?", (str(path),))
                db.commit()
                continue
            db.execute("DELETE FROM shards WHERE path=?", (str(path),))
        h, n, missing, values = hashlib.sha256(), 0, Counter(), []
        with path.open("rb") as handle:
            for line in handle:
                h.update(line)
                if not line.strip():
                    continue
                n += 1
                keys, unresolved = extract_keys(json.loads(line))
                missing.update(unresolved)
                values.extend((role, seq) for role, sequences in keys.items() for seq in sequences)
                if n % 5000 == 0:
                    db.executemany("INSERT OR IGNORE INTO sequences VALUES (?,?)", values)
                    values.clear()
        db.executemany("INSERT OR IGNORE INTO sequences VALUES (?,?)", values)
        if n != shard["records"]:
            raise ValueError(f"Shard count mismatch: {path}")
        db.execute("INSERT INTO shards VALUES (?,?,?,?,?)", (str(path), h.hexdigest(), n, json.dumps(missing), 2))
        db.commit()
        print(json.dumps({"stage":"inventory","shard":shard["path"],"rows":n,"unresolved":missing}), flush=True)
    counts = dict(db.execute("SELECT role,COUNT(*) FROM sequences GROUP BY role"))
    rows = list(db.execute("SELECT path,sha256,records,unresolved FROM shards ORDER BY path"))
    write_json(work / "inventory.json", dict(status="complete", manifest_sha256=sha(input_dir / "dataset_manifest.json"),
               registry_sha256=sha(work / "registry.json"), counts=counts, shards=rows, rows=sum(r[2] for r in rows)))
    db.close()


def cluster(work, threads, reuse_work=None):
    verify_registry(work)
    inv = json.loads((work / "inventory.json").read_text())
    if inv["status"] != "complete" or inv["registry_sha256"] != sha(work / "registry.json"):
        raise ValueError("Incomplete or stale inventory")
    banks = json.loads((work / "banks.json").read_text())
    annotation_path = work / "benchmark_h3_annotations.json"
    annotations = json.loads(annotation_path.read_text())
    if annotations["status"] != "complete" or annotations["banks_sha256"] != sha(work / "banks.json"):
        raise ValueError("Benchmark H3 numbering incomplete or stale")
    banks["h3"] = sorted(set(banks["h3"]) | set(annotations["annotations"].values()))
    effective_path = work / "effective_banks.json"
    if effective_path.exists() and json.loads(effective_path.read_text()) != banks:
        raise ValueError("Effective banks changed; new audit version required")
    write_json(effective_path, banks)
    db = sqlite3.connect(work / "inventory.sqlite")
    folder = work / "clusters"
    folder.mkdir(exist_ok=True)
    for role, threshold in RULES.items():
        sequences = sorted({s for (s,) in db.execute("SELECT sequence FROM sequences WHERE role=?", (role,))} | set(banks[role]))
        key = digest(json.dumps([threshold, .8, sequences, banks[role]]))
        report_path = folder / f"{role}.json"
        mapping = folder / f"{role}.tsv"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get("input_sha256") == key and report.get("mapping_sha256") == sha(mapping):
                continue
            raise ValueError(f"Stale role cluster: {role}")
        if reuse_work is not None:
            prior_report=reuse_work/"clusters"/f"{role}.json"
            prior_mapping=reuse_work/"clusters"/f"{role}.tsv"
            if prior_report.exists() and prior_mapping.exists():
                prior=json.loads(prior_report.read_text())
                # Reuse only identical role sequence/bank/rule content, never
                # merely a filename, sequence count, or old status flag.
                if (prior.get("status")=="complete" and prior.get("input_sha256")==key
                        and prior.get("mapping_sha256")==sha(prior_mapping)):
                    shutil.copy2(prior_mapping,mapping)
                    prior.update(inventory_sha256=sha(work/"inventory.json"),effective_banks_sha256=sha(effective_path),
                                 reused_mapping=str(prior_mapping),reused_report_sha256=sha(prior_report))
                    write_json(report_path,prior)
                    print(json.dumps({"stage":"cluster_reuse","role":role,"sequences":len(sequences)}),flush=True)
                    continue
        fasta = folder / f"{role}.fasta"
        with fasta.open("w") as out:
            for i, sequence in enumerate(sequences):
                out.write(f">s{i}\n{sequence}\n")
        prefix = folder / f"{role}_mmseqs"
        command = [str(MMSEQS), "easy-linclust", str(fasta), str(prefix), str(folder / f"tmp_{role}"),
                   "--min-seq-id", str(threshold), "-c", "0.8", "--cov-mode", "0", "--cluster-mode", "1",
                   "--threads", str(threads), "--remove-tmp-files", "1"]
        print(json.dumps({"stage":"cluster","role":role,"sequences":len(sequences)}), flush=True)
        with (folder / f"{role}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        membership = {}
        with Path(str(prefix)+"_cluster.tsv").open() as handle:
            for line in handle:
                rep, member = line.strip().split("\t")
                if member in membership:
                    raise ValueError("Duplicate cluster membership")
                membership[member] = rep
        if set(membership) != {f"s{i}" for i in range(len(sequences))}:
            raise ValueError("MMseqs output is incomplete; no self-cluster fallback allowed")
        seq_cluster = {seq: membership[f"s{i}"] for i,seq in enumerate(sequences)}
        blocked = {seq_cluster[s] for s in banks[role]}
        with mapping.open("w") as out:
            for sequence in sequences:
                group = seq_cluster[sequence]
                out.write(f"{sequence}\t{role}:{group}\t{int(group in blocked)}\n")
        write_json(report_path, dict(status="complete", input_sha256=key, mapping_sha256=sha(mapping),
                   inventory_sha256=sha(work / "inventory.json"), effective_banks_sha256=sha(effective_path),
                   sequences=len(sequences), benchmark=len(banks[role]), clusters=len(set(seq_cluster.values())),
                   blocked_clusters=len(blocked), command=command, operational_definition="frozen_joint_linclust"))
    db.close()


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def join(self, keys):
        roots = sorted({self.find(k) for k in keys})
        if roots:
            for root in roots[1:]:
                self.parent[root] = roots[0]


def export(work, input_dir, output_dir):
    registry = verify_registry(work)
    verify_h3_coverage(work)
    inventory = json.loads((work / "inventory.json").read_text())
    if inventory["manifest_sha256"] != sha(input_dir / "dataset_manifest.json"):
        raise ValueError("Source manifest changed")
    for path, expected, _, _ in inventory["shards"]:
        if sha(path) != expected:
            raise ValueError(f"Source shard changed: {path}")
    if output_dir.exists():
        raise FileExistsError(f"Versioned export already exists: {output_dir}")
    mappings, blocked = {}, {}
    for role in RULES:
        mapping = work / "clusters" / f"{role}.tsv"
        report = json.loads((work / "clusters" / f"{role}.json").read_text())
        if (report["status"] != "complete" or sha(mapping) != report["mapping_sha256"]
                or report.get("inventory_sha256") != sha(work / "inventory.json")
                or report.get("effective_banks_sha256") != sha(work / "effective_banks.json")):
            raise ValueError(f"Incomplete/stale homology stage: {role}")
        mappings[role], blocked[role] = {}, set()
        with mapping.open() as handle:
            for line in handle:
                sequence, group, flag = line.rstrip().split("\t")
                mappings[role][sequence] = group
                if flag == "1":
                    blocked[role].add(sequence)
    banks = {k:set(v) for k,v in json.loads((work / "effective_banks.json").read_text()).items()}
    # Conservative edit-neighbour screen complements linclust for short epitopes.
    from rapidfuzz.distance import Levenshtein
    near_peptides = set()
    for sequence in mappings["peptide"]:
        for protected in banks["peptide"]:
            if min(len(sequence), len(protected)) >= .8*max(len(sequence), len(protected)) and Levenshtein.distance(sequence, protected) <= .2*max(len(sequence), len(protected)):
                near_peptides.add(sequence)
                break
    blocked["peptide"].update(near_peptides)
    epitope_containment = {ag for ag in mappings["antigen"] if any(ep in ag for ep in banks["peptide"])}
    antigen_containment = {ag for ag in mappings["antigen"] if any(covered_containment(ag,ref) for ref in banks["antigen"])}
    blocked["antigen"].update(antigen_containment)
    manifest, shards = source_shards(input_dir)
    components = Components()
    decisions, counts = {}, Counter()
    c_legacy, c_v3 = ImmuneCollator("A1"), ImmuneCollator("A4")

    def inspect(row):
        keys, missing = extract_keys(row)
        reasons = set("annotation."+m for m in missing)
        for role in RULES:
            if keys.get(role, set()) & blocked[role]:
                reasons.add("decontam."+role)
        if keys.get("pair", set()) & banks["pair"]:
            reasons.add("decontam.pair")
        if keys.get("antigen", set()) & epitope_containment:
            reasons.add("decontam.epitope_containment")
        nodes = [mappings[role][s] for role in ("heavy","h3","cdr3a","cdr3b") for s in keys.get(role, ())]
        if not nodes:
            reasons.add("annotation.no_receptor_family")
        return keys, reasons, nodes

    # Family graph uses BOTH receptor chains and core/heavy family links; never
    # concatenate independent cluster IDs into a tuple that permits leakage.
    for shard in shards:
        for row in read_rows(input_dir / shard["path"]):
            _, reasons, nodes = inspect(row)
            if not reasons:
                components.join(nodes)
        print(json.dumps({"stage":"family_graph","shard":shard["path"]}), flush=True)
    output_dir.mkdir(parents=True)
    write_json(output_dir / "quarantine_keys.json", {r:sorted(s) for r,s in blocked.items()})
    handles, shard_counts, hashes = {}, Counter(), {}
    seen = set()
    removal = (output_dir / "removals.jsonl").open("w")
    groups = defaultdict(set)
    eligibility_cache = {}
    try:
        for shard in shards:
            for offset, row in enumerate(read_rows(input_dir / shard["path"])):
                keys, reasons, nodes = inspect(row)
                semantic = BioSeqRecord.from_prepared_dict(row)
                signature = digest(json.dumps([sorted((c.role,observe(c).scope,observe(c).sequence) for c in semantic.chains),row.get("labels",{})], sort_keys=True))
                if signature in seen:
                    reasons.add("dedup.exact_semantic")
                if not reasons:
                    # Freeze common eligible records before any architecture run.
                    record = BioSeqRecord.from_prepared_dict(row)
                    layout_key = tuple((c.role,len(c.sequence), observe(c).scope,len(observe(c).sequence)) for c in record.chains)
                    if layout_key not in eligibility_cache:
                        try:
                            c_legacy([record]); c_v3([record])
                            eligibility_cache[layout_key] = None
                        except ValueError as exc:
                            eligibility_cache[layout_key] = str(exc)
                    if eligibility_cache[layout_key]:
                        reasons.add("budget.common_eligibility")
                if reasons:
                    counts.update(reasons)
                    removal.write(json.dumps(dict(shard=shard["path"],row=offset,reasons=sorted(reasons)))+"\n")
                    continue
                seen.add(signature)
                group = components.find(nodes[0])
                # 1% validation, consistent with the scale of the previous corpus.
                split = "valid" if int(digest(group)[:8],16) % 100 == 0 else "train"
                groups[split].add(group)
                source = shard["source"]
                index = shard_counts[split,source] // 100000
                rel = f"{split}/{source}-{index:05d}.jsonl"
                if rel not in handles:
                    (output_dir / split).mkdir(exist_ok=True)
                    handles[rel] = (output_dir / rel).open("wb")
                    hashes[rel] = hashlib.sha256()
                row["split"] = split
                row.setdefault("metadata", {})["v7_family_group"] = group
                raw = (json.dumps(row, separators=(",",":"))+"\n").encode()
                handles[rel].write(raw); hashes[rel].update(raw)
                shard_counts[split,source] += 1
            print(json.dumps({"stage":"export","shard":shard["path"],"kept":sum(shard_counts.values())}), flush=True)
    finally:
        removal.close()
        for handle in handles.values():
            handle.close()
    if groups["train"] & groups["valid"]:
        raise AssertionError("Family split overlap")
    new = {k:v for k,v in manifest.items() if k != "splits"}
    new["dataset_version"] = "immune_v7_clean_r2"
    new["splits"] = {s:{"records":sum(n for (sp,_),n in shard_counts.items() if sp==s),"shards":[]} for s in ("train","valid")}
    for rel in handles:
        split, filename = rel.split("/")
        source,index = filename.rsplit("-",1)
        count = min(100000,shard_counts[split,source]-int(index[:-6])*100000)
        new["splits"][split]["shards"].append(dict(path=rel,source=source,split=split,records=count,sha256=hashes[rel].hexdigest()))
    pending = ["upstream_source_protein_mapping_unavailable", "independent_export_rescan_pending", "benchmark_h3_annotation_coverage_review"]
    if registry["missing_legacy_inputs"]:
        pending.append("missing_legacy_benchmark_inputs")
    new["decontamination"] = dict(registry_sha256=sha(work / "registry.json"), status="audit_pending", pending=pending,
                                 protocol_revision="v7.2_bidirectional_containment",
                                 protocol="frozen_joint_linclust_plus_exact_epitope_edit_screen", source_manifest_sha256=inventory["manifest_sha256"],
                                 effective_banks_sha256=sha(work / "effective_banks.json"),
                                 h3_coverage_sha256=sha(work / "benchmark_h3_coverage.json"),
                                 quarantine_keys_sha256=sha(output_dir / "quarantine_keys.json"))
    write_json(output_dir / "dataset_manifest.json", new)
    write_json(output_dir / "cleaning_report.json", dict(kept={f"{s}/{src}":n for (s,src),n in shard_counts.items()},
               removed_reasons=counts, family_groups={s:len(v) for s,v in groups.items()}, training_ready=False, pending=pending))


def verify_export(work, output_dir):
    """Independent pass over the exact emitted stream; unresolved evidence stays red."""
    registry=verify_registry(work)
    coverage=verify_h3_coverage(work)
    manifest,shards=source_shards(output_dir)
    quarantine={k:set(v) for k,v in json.loads((output_dir/"quarantine_keys.json").read_text()).items()}
    banks={k:set(v) for k,v in json.loads((work/"effective_banks.json").read_text()).items()}
    if manifest["decontamination"]["quarantine_keys_sha256"]!=sha(output_dir/"quarantine_keys.json"):
        raise ValueError("Quarantine key file changed")
    if (manifest["decontamination"]["registry_sha256"] != sha(work/"registry.json")
            or manifest["decontamination"]["effective_banks_sha256"] != sha(work/"effective_banks.json")
            or manifest["decontamination"]["h3_coverage_sha256"] != sha(work/"benchmark_h3_coverage.json")):
        raise ValueError("Export benchmark evidence changed")
    # Banks are immutable for the whole pass. Building their large unions per
    # row makes verification scale with bank_size * corpus_size unnecessarily.
    protected={role:quarantine[role] | banks[role] for role in RULES}
    groups=defaultdict(set)
    overlaps,unresolved,rows,source_counts=Counter(),Counter(),Counter(),Counter()
    for shard in shards:
        path=output_dir/shard["path"]
        if sha(path)!=shard["sha256"]:
            raise ValueError(f"Export shard hash changed: {path}")
        count=0
        for row in read_rows(path):
            count+=1
            keys,missing=extract_keys(row)
            unresolved.update(missing)
            hits=[role for role in RULES if not protected[role].isdisjoint(keys.get(role,()))]
            if keys.get("pair",set()) & banks["pair"]:
                hits.append("pair")
            if any(ep in ag for ag in keys.get("antigen",()) for ep in banks["peptide"]):
                hits.append("epitope_containment")
            if hits:
                overlaps["rows"]+=1
                overlaps.update(hits)
            if row["split"]!=shard["split"]:
                raise ValueError("Export row/shard split mismatch")
            groups[row["split"]].add(row["metadata"]["v7_family_group"])
        if count!=shard["records"]:
            raise ValueError(f"Export row count differs: {path}")
        rows[shard["split"]]+=count
        source_counts[shard["split"],shard["source"]]+=count
        print(json.dumps({"stage":"verify","shard":shard["path"],"verified_rows":sum(rows.values()),
                          "forbidden_overlap_rows":overlaps["rows"]}),flush=True)
    annotations=json.loads((work/"benchmark_h3_annotations.json").read_text())
    if annotations["banks_sha256"] != sha(work/"banks.json"):
        raise ValueError("Benchmark annotation belongs to different frozen banks")
    pending=[]
    if manifest["decontamination"].get("protocol_revision")!="v7.2_bidirectional_containment":
        pending.append("obsolete_containment_protocol")
    unprotected_native=set()
    duplicate_headers=0
    for filename in registry["inputs"]:
        if filename.endswith("/nbbench/hf_data/CDRInfilling/test.csv"):
            with Path(filename).open() as handle:
                for row in csv.DictReader(handle):
                    if duplicate_nanobody_header(row):
                        duplicate_headers+=1
                        continue
                    native=normalize_sequence(row.get("label"))
                    if native and native not in banks["heavy"]:
                        unprotected_native.add(native)
    if unprotected_native: pending.append("unprotected_native_benchmark_reference")
    if registry["missing_legacy_inputs"]: pending.append("missing_legacy_benchmark_inputs")
    if annotations["status"]!="complete" or coverage["unresolved"]: pending.append("unresolved_benchmark_h3")
    if unresolved: pending.append("unresolved_training_receptor_annotation")
    if overlaps: pending.append("forbidden_overlap")
    overlap_groups=len(groups["train"] & groups["valid"])
    if overlap_groups: pending.append("family_split_overlap")
    if not rows["train"] or not rows["valid"]: pending.append("empty_split")
    expected_sources=set(manifest.get("sources",[])) or {s["source"] for s in shards}
    missing_source_splits=[f"{split}/{source}" for source in sorted(expected_sources) for split in ("train","valid")
                          if not source_counts[split,source]]
    if missing_source_splits: pending.append("empty_source_split")
    # Publish the verified state into the manifest before binding its hash.
    # Re-running verification is idempotent when the evidence is unchanged.
    manifest["decontamination"].update(status="passed" if not pending else "audit_pending",pending=pending)
    if manifest["decontamination"].get("protocol_revision")=="v7.2_bidirectional_containment":
        manifest["dataset_version"]="immune_v7_clean_r2"
    if "source_created_at" not in manifest:
        manifest["source_created_at"] = manifest.get("created_at")
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["decontamination"]["common_eligibility"] = dict(max_length=2048,full_capacity=256,fragment_capacity=64,
                                                               layouts=["legacy_fixed_canvas","observed_v3"])
    write_json(output_dir/"dataset_manifest.json",manifest)
    cleaning_path=output_dir/"cleaning_report.json"
    cleaning=json.loads(cleaning_path.read_text())
    cleaning.update(training_ready=not pending,pending=pending)
    write_json(cleaning_path,cleaning)
    certificate=dict(schema_version="immune_v7.audit.v1",status="passed" if not pending else "pending",
                     manifest_sha256=sha(output_dir/"dataset_manifest.json"),pending=pending,
                     forbidden_overlap_rows=overlaps["rows"],overlap_reasons=overlaps,
                     family_split_overlap=overlap_groups,rows=rows,benchmark_h3_unresolved=len(coverage["unresolved"]),
                     benchmark_h3_source_covered=len(coverage["source_covered"]),
                     unprotected_native_reference_heavy=len(unprotected_native),
                     benchmark_repeated_header_rows=duplicate_headers,
                     unprotected_native_reference_sha256=sorted(digest(s) for s in unprotected_native),
                     rows_by_source={f"{split}/{source}":count for (split,source),count in source_counts.items()},
                     missing_source_splits=missing_source_splits,
                     registry_sha256=sha(work/"registry.json"),effective_banks_sha256=sha(work/"effective_banks.json"),
                     scope="actual_prepared_training_and_validation; frozen operational clustering; no upstream parent-protein novelty claim")
    write_json(output_dir/"audit_certificate.json",certificate)
    print(json.dumps(certificate),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["freeze","inventory","cluster","coverage","export","verify"], required=True)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reuse-work",type=Path)
    args = parser.parse_args()
    if args.stage == "freeze": freeze(args.work)
    elif args.stage == "inventory": inventory(args.work,args.input)
    elif args.stage == "cluster": cluster(args.work,args.threads,args.reuse_work)
    elif args.stage == "coverage": audit_h3_coverage(args.work)
    elif args.stage == "export": export(args.work,args.input,args.output)
    else: verify_export(args.work,args.output)


if __name__ == "__main__":
    main()
