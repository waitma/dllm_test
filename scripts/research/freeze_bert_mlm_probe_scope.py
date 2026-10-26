"""Exclude protected test content from external BERT development probes.

After activating pllm, run with the training Python environment:
  USE_TF=0 python -B -m scripts.research.freeze_bert_mlm_probe_scope --index ABS
Creates a new index; the original index and all training shards stay immutable.
Uncovered negative examples are excluded instead of assuming a missing match
means no test overlap. Sources need >=16 examples/class in both probe splits.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.data.export_v7_binding_design import (
    audit_keys, read_pickle, reasons, unresolved_tcr, upstream_lookup)
from scripts.research.bert_mlm_data import sha, write_json


def freeze(index_path):
    original = json.loads(index_path.read_text())
    output = index_path.parent
    destination = output / "index_test_clean.json"
    if destination.exists():
        raise ValueError("Use a new probe version; never overwrite a frozen index")
    manifest = json.loads(Path(original["manifest"]).read_text())
    registry_path = Path(manifest["parent_rebuild"]["registry_path"])
    if sha(registry_path) != manifest["parent_rebuild"]["registry_sha256"]:
        raise ValueError("Protected test registry changed")
    work = registry_path.parent.parent / "no_negative_r1"
    match_path, coverage_path = work / "match_index.pkl", work / "candidate_keys.pkl"
    match_receipt = json.loads((work / "matching.json").read_text())
    candidate_receipt = json.loads((work / "candidates.json").read_text())
    if (sha(match_path) != manifest["parent_rebuild"]["match_index_sha256"]
            or sha(work/"candidates.json") != match_receipt["candidates_sha256"]
            or sha(coverage_path) != candidate_receipt["keys_sha256"]):
        raise ValueError("V7 test match or candidate coverage index changed")
    registry = json.loads(registry_path.read_text())
    upstream = Path(registry["upstream"])
    if sha(upstream) != registry["inputs"][str(upstream)]:
        raise ValueError("Upstream receptor key evidence changed")
    match, coverage, lookup = read_pickle(match_path), read_pickle(coverage_path), upstream_lookup()
    retained, counts, excluded = {}, {}, []
    for name in ("probe_train", "probe_valid"):
        reference = original["files"][name]
        if sha(reference["path"]) != reference["sha256"]:
            raise ValueError("Input development probe changed")
        retained[name], counts[name] = [], Counter()
        for number, line in enumerate(Path(reference["path"]).read_text().splitlines()):
            sample = json.loads(line)
            keys, _ = audit_keys(sample["record"], lookup)
            hits = {"unresolved_tcr": 1} if unresolved_tcr(keys) else reasons(keys, match)
            missing = [role for role in ("heavy", "h3", "antigen")
                       if keys.get(role) and keys[role] not in coverage["values"].get(role, set())]
            if not keys["paired_tcr"]:
                missing += [role for role in ("cdr3a", "cdr3b", "peptide")
                            if keys.get(role) and keys[role] not in coverage["near"].get(role, set())]
            if hits or missing:
                excluded.append(dict(file=name, row=number, source=sample["source_group"],
                                     reasons=hits, uncovered_keys=missing))
                continue
            retained[name].append(sample)
            counts[name][(sample["source_group"], sample["record"]["labels"]["relation"])] += 1
    sources = [source for source in original["probe_sources"] if all(
        counts[part][source, label] >= 16 for part in retained for label in ("binding", "nonbinding"))]
    if not sources:
        raise ValueError("No sufficiently populated, test-clean binary development probe")
    result = dict(original)
    result["files"] = dict(original["files"])
    final_counts = {}
    for name, samples in retained.items():
        selected = [sample for sample in samples if sample["source_group"] in sources]
        path = output / (name + "_test_clean.jsonl")
        with path.open("x") as stream:
            for sample in selected:
                stream.write(json.dumps(sample) + "\n")
        result["files"][name] = dict(path=str(path), sha256=sha(path), records=len(selected))
        final_counts[name] = dict(Counter(x["source_group"]+"/"+x["record"]["labels"]["relation"]
                                         for x in selected))
    audit_path = output / "probe_test_scope.json"
    audit = dict(status="passed", original_index=str(index_path), original_index_sha256=sha(index_path),
                 evidence={str(p): sha(p) for p in (registry_path, match_path, coverage_path, upstream,
                                                   work/"matching.json", work/"candidates.json")},
                 sources=sources, dropped_sources=sorted(set(original["probe_sources"])-set(sources)),
                 final_counts=final_counts, excluded=excluded,
                 rule="Apply published V7 exact/near/cluster rules; omit unsupported keys and sources "
                      "with fewer than 16 examples/class/split; inherited exact train isolation remains valid",
                 code_sha256=sha(Path(__file__)))
    write_json(audit_path, audit)
    result["files"]["probe_scope_audit"] = dict(path=str(audit_path), sha256=sha(audit_path))
    result.update(probe_sources=sources,
                  probe_protocol="External TCR papers/TRAIT binary development probes only; backbone frozen; "
                                 "V7 protected-test scope applied with verified candidate coverage, "
                                 "held-out receptor keys absent from every V7 train row",
                  probe_parent_index=str(index_path), probe_parent_index_sha256=sha(index_path))
    write_json(destination, result)
    print(json.dumps(dict(index=str(destination), sources=sources, counts=final_counts,
                          excluded_by_test_or_unverified_coverage=len(excluded)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    freeze(parser.parse_args().index.resolve())
