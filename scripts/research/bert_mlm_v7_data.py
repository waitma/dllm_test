"""Bind cleaned, MHC-completed V7 data to BERT without changing its rows.

After activating pllm, run with the training Python environment:
  USE_TF=0 python -B -m scripts.research.bert_mlm_v7_data --help
The older frozen binary development probe is evaluation-only: its negatives
never enter pretraining, and its held-out receptor keys are checked against V7.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from dllm.pipelines.immune_llada.fixed_chain.mlm import BertMLMCollator
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from scripts.research.bert_mlm_data import keep_smallest, scan_shard, sha, write_json

ROOT = Path(__file__).resolve().parents[2]
RELEASE_SHA = "a2c89f8f0ab4da2d7c03b5ced9fcffd04d545168ad1e8b734e36638bec3e33a9"
BINDING_SCHEMA = "immune.bert_mlm.v7_binding.v1"


def release_binding(data_dir):
    """Use the published V7 acceptance evidence and dataset-local model config."""
    manifest_path = data_dir / "dataset_manifest.json"
    if sha(manifest_path) != RELEASE_SHA:
        raise ValueError("Expected the published slots167/135 MHC-completed V7 release")
    manifest = json.loads(manifest_path.read_text())
    contract = manifest["reader_contract"]
    config_path = data_dir / contract["config_path"]
    config = json.loads(config_path.read_text())
    if (contract["layout"] != "immune.fixed_chain.f011.v1" or contract["a4_allowed"]
            or (config["first_chain_capacity"], config["second_chain_capacity"],
                config["full_capacity"], config["max_length"]) != (167, 135, 304, 2048)
            or sha(config_path) != contract["config_sha256"]):
        raise ValueError("V7 reader/config contract differs")
    evidence = {}
    for name in ("verification.json", "independent_acceptance.json", "reader_acceptance.json"):
        path = data_dir / name
        audit = json.loads(path.read_text())
        if audit["status"] != "passed" or audit["dataset_manifest_sha256"] != RELEASE_SHA:
            raise ValueError("V7 release acceptance is missing or stale: " + name)
        evidence[str(path)] = sha(path)
    reader = json.loads((data_dir / "reader_acceptance.json").read_text())
    if reader["config_sha256"] != sha(config_path):
        raise ValueError("Actual-reader acceptance used another model config")
    layout = ROOT / "dllm/pipelines/immune_llada/fixed_chain/layout.py"
    if reader["collator_sha256"] != sha(layout):
        raise ValueError("Fixed-chain layout changed since V7 reader acceptance")
    parent = Path(manifest["mhc_completion"]["parent"])
    parent_manifest = parent / "dataset_manifest.json"
    if sha(parent_manifest) != manifest["mhc_completion"]["parent_manifest_sha256"]:
        raise ValueError("Immutable V7 parent changed")
    evidence[str(parent_manifest)] = sha(parent_manifest)
    shards = []
    for split, specification in manifest["splits"].items():
        if sum(x["records"] for x in specification["shards"]) != specification["records"]:
            raise ValueError("V7 manifest shard totals differ")
        for entry in specification["shards"]:
            path = data_dir / entry["path"]
            resolved = path.resolve(strict=True)
            if resolved.parent.parent not in (data_dir, parent):
                raise ValueError("Shard resolves outside the release and its immutable parent")
            shards.append(dict(path=str(path), resolved_path=str(resolved), split=split,
                               source=entry["source"], records=entry["records"], sha256=entry["sha256"]))
    return dict(schema=BINDING_SCHEMA, data_dir=str(data_dir), manifest=str(manifest_path),
                manifest_sha256=RELEASE_SHA, config_path=str(config_path),
                config_sha256=sha(config_path), model=config, evidence=evidence,
                layout_sha256=sha(layout), shards=shards,
                records={s: v["records"] for s, v in manifest["splits"].items()})


def verify_binding(path, expected_sha, model):
    """Recheck release identity; callers separately hash every actual shard byte."""
    path = Path(path)
    if sha(path) != expected_sha:
        raise ValueError("V7 BERT data binding changed")
    bound = json.loads(path.read_text())
    if bound["schema"] != BINDING_SCHEMA or bound["model"] != model:
        raise ValueError("V7 binding/model mismatch")
    current = release_binding(Path(bound["data_dir"]))
    if current != bound:
        raise ValueError("V7 release metadata, config, evidence or symlink target changed")
    return bound


def prepare(data_dir, output, probe_index, workers):
    bound = release_binding(data_dir)
    old = json.loads(probe_index.read_text())
    probes = {}
    collator = BertMLMCollator(FixedChainConfig(**bound["model"]))
    probe_lengths = []
    for name in ("probe_train", "probe_valid"):
        reference = old["files"][name]
        if sha(reference["path"]) != reference["sha256"]:
            raise ValueError("Frozen external development probe changed")
        rows = [json.loads(s) for s in Path(reference["path"]).read_text().splitlines()]
        if len(rows) != reference["records"]:
            raise ValueError("Frozen probe count differs")
        for sample in rows:
            batch = collator([sample["record"]])  # Raises on overflow; never truncate.
            probe_lengths.append(int(batch["attention_mask"].sum()))
        probes[name] = rows
    held_keys = {row["receptor_key"] for row in probes["probe_valid"]}
    output.mkdir(parents=True, exist_ok=False)
    binding_path = output / "binding.json"
    write_json(binding_path, bound)
    manifest = json.loads(Path(bound["manifest"]).read_text())
    expected = {x["path"]: x for x in bound["shards"]}
    capacities = dict(antibody_heavy=167, tcr_beta=167, antibody_light=135, tcr_alpha=135)
    jobs = [(str(data_dir), dict(entry, split=split, forbid_nonbinding=True,
                               receptor_capacities=capacities), str(output), held_keys)
            for split, spec in manifest["splits"].items() for entry in spec["shards"]]
    results, validation, exposed = [], defaultdict(list), set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(scan_shard, job) for job in jobs]):
            item = future.result()
            target = expected[item["path"]]
            if (item["sha256"] != target["sha256"] or item["resolved_path"] != target["resolved_path"]
                    or item["original_records"] != target["records"]):
                raise ValueError("Actual V7 shard differs from its accepted manifest")
            if item["excluded"]:
                raise ValueError("Unexpected V7 record without observed receptor residues")
            if item["split"] == "train":
                exposed.update(item["overlap"])
            for score, text in item.pop("validation"):
                keep_smallest(validation[item["source"]], -score, json.loads(text), 128)
            item.pop("probe")
            item.pop("overlap")
            results.append(item)
            print(json.dumps(dict(shards=len(results), split=item["split"], records=item["records"])), flush=True)
    if exposed:
        raise ValueError("External development receptor identity occurs in V7 train")
    files = {}
    def save_rows(name, rows):
        path = output / (name + ".jsonl")
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        files[name] = dict(path=str(path), sha256=sha(path), records=len(rows))
    token_rows = [json.loads(text) for source in sorted(validation)
                  for _, text in sorted(validation[source])]
    lengths = [int(collator([x["record"]])["attention_mask"].sum()) for x in token_rows]
    save_rows("token_valid", token_rows)
    for name, rows in probes.items():
        save_rows(name, rows)
    counts = {s: sum(x["records"] for x in results if x["split"] == s) for s in ("train", "valid")}
    if counts != bound["records"]:
        raise ValueError("Full index coverage differs from V7 release")
    result = dict(schema="immune.bert_mlm.index.v1", binding_kind=BINDING_SCHEMA,
                  data_dir=str(data_dir), manifest=bound["manifest"], manifest_sha256=RELEASE_SHA,
                  binding=str(binding_path), binding_sha256=sha(binding_path),
                  shards=sorted(results, key=lambda x: (x["split"], x["path"])), files=files,
                  records=counts, probe_sources=old["probe_sources"],
                  probe_valid_candidates_exposed_to_train=0,
                  probe_protocol="Frozen external V5 train/valid binary diagnostic, evaluation-only; "
                                 "backbone receives no probe gradient; held-out observed receptor keys "
                                 "checked absent from every V7 train row; no benchmark test",
                  external_probe_origin=dict(index=str(probe_index), index_sha256=sha(probe_index),
                                             files={k: old["files"][k] for k in probes}),
                  full_shard_hashes_verified=len(results),
                  acceptance=dict(no_explicit_nonbinding=True, no_capacity_overflow=True,
                                  actual_mlm_collator_dev_records=len(token_rows)+sum(map(len, probes.values())),
                                  max_token_dev_length=max(lengths), max_external_probe_length=max(probe_lengths)),
                  sources={s: dict(Counter({r["source"]: sum(x["records"] for x in results
                                    if x["split"] == s and x["source"] == r["source"])
                                    for r in results if r["split"] == s})) for s in counts})
    write_json(output / "index.json", result)
    print(json.dumps(dict(status="passed", records=counts, dev_records={k: v["records"] for k, v in files.items()},
                          acceptance=result["acceptance"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-index", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.data_dir.resolve(), args.output.resolve(), args.probe_index.resolve(), args.workers)
