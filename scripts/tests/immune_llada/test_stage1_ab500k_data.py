"""CPU fixtures: activate pllm; USE_TF=0 python -B -m pytest THIS_FILE -q.

Exercise real parent-byte offsets, request construction, preserved row frequency,
holdout exclusion, sixteen-way partitioning and corruption rejection.
"""
from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from scripts.research.bert_mlm_data import sha
from scripts.research.bert_mlm_eval import read_samples
from scripts.research.stage1_ab500k_data import build_subset, validate_subset, validate_subset_files, _eligibility, _index


def pair(number, split):
    amino = "ACDEFGHIKLMNPQRSTVWY"
    return BioSeqRecord([BioSeqChain("ACDEFG" + amino[number], "antibody_heavy"),
                         BioSeqChain("LMNPQ" + amino[number], "antibody_light")],
                        "antibody", "oas_paired", split=split).to_dict()


def parent(tmp_path, repeat_train=1):
    root = tmp_path / "parent"
    root.mkdir()
    config = FixedChainConfig(d_model=32, n_layers=8, n_heads=4, mlp_hidden_size=64,
                              first_chain_capacity=16, second_chain_capacity=12, full_capacity=30)
    binding = root / "binding.json"
    binding.write_text(json.dumps(dict(model=asdict(config))))
    valid = [pair(i, "valid") for i in range(5)] + [pair(0, "valid")]
    bad_valid = pair(9, "valid")
    bad_valid["chain_roles"] = ["antibody_heavy", "antibody_heavy"]
    valid.append(bad_valid)
    train = [pair(i, "train") for i in range(5)] + [pair(6, "train"), pair(6, "train")]
    train += [pair(i, "train") for i in range(7, 10)]
    bads = []
    for kind in ("synthetic", "fragment", "role", "alphabet"):
        row = pair(11, "train")
        if kind == "synthetic":
            row["chain_metadata"] = {"0": {"synthetic_residue_mask": [1] + [0] * 6}}
        elif kind == "fragment":
            row["chain_metadata"] = {"1": {"sequence_scope": "cdr3"}}
        elif kind == "role":
            row["chain_roles"] = ["antibody_heavy", "antibody_heavy"]
        else:
            row["chains"][0] += "X"
        bads.append(row)
    train += bads
    train *= repeat_train
    shards = []
    def shard(source, split, name, rows):
        path = root / (name + ".jsonl")
        offsets = []
        with path.open("wb") as stream:
            for row in rows:
                offsets.append(stream.tell())
                stream.write((json.dumps(row) + "\n").encode())
        offset_path = root / (name + ".npy")
        np.save(offset_path, np.asarray(offsets, dtype=np.uint64))
        shards.append(dict(path=str(path), resolved_path=str(path.resolve()), sha256=sha(path),
                           source=source, split=split, records=len(rows), index=str(offset_path),
                           index_sha256=sha(offset_path)))
    # Non-OAS rows exercise source filtering and original global-row arithmetic.
    shard("asd_antibody", "train", "other_train", [pair(15, "train")])
    shard("oas", "train", "train_a", train[:7])
    shard("oas", "train", "train_b", train[7:])
    shard("asd_antibody", "valid", "other_valid", [pair(15, "valid")])
    shard("oas", "valid", "valid", valid)
    index = root / "index.json"
    index.write_text(json.dumps(dict(schema="immune.bert_mlm.index.v1", data_dir=str(root),
                                    binding=str(binding), binding_sha256=sha(binding), shards=shards,
                                    records=dict(train=len(train) + 1, valid=len(valid) + 1))))
    return index


def args(index, output, **kwargs):
    return Namespace(data_index=str(index), output=str(output), train_count=7, valid_count=3,
                     seed=42, pilot_count=5, **kwargs)


def test_real_offsets_strict_pairs_frequency_and_evaluator_compatibility(tmp_path):
    index = parent(tmp_path)
    out = tmp_path / "subset"
    subset = build_subset(args(index, out))
    report = validate_subset(out / "subset.json")
    assert report["status"] == "passed"
    assert validate_subset_files(out)["mode"] == "files_only"
    assert subset["counts"] == {"train": 7, "valid": 3}
    assert subset["stats"]["train_excluded_fixed_valid_exact_pair"] == 3
    assert subset["stats"]["selected_train_duplicate_pair_rows"] == 1
    assert subset["stats"]["valid_unique_eligible_pairs"] == 5
    assert subset["stats"]["train_excluded_synthetic"] == 1
    assert subset["stats"]["train_excluded_fragment"] == 1
    assert subset["stats"]["train_excluded_not_native_heavy_light"] == 1
    assert subset["stats"]["train_excluded_noncanonical_or_empty"] == 1
    train_rows = np.load(subset["train_rows"])
    assert min(train_rows) > 0 and len(set(train_rows.tolist())) == 7
    samples = read_samples(subset["files"]["token_valid"])
    assert [x["row_index"] for x in samples] == np.load(subset["valid_rows"]).tolist()
    lines = Path(subset["requests"]).read_bytes().splitlines(keepends=True)
    for worker, shard in enumerate(subset["request_shards"]):
        assert Path(shard["path"]).read_bytes().splitlines(keepends=True) == lines[worker::16]
    pilot = Path(subset["pilot_requests"]).read_bytes().splitlines(keepends=True)
    assert len(pilot) == 5 and all(line in lines for line in pilot)


def test_seeded_selection_repeats_with_reservoir_subsampling(tmp_path):
    index = parent(tmp_path, repeat_train=40)
    spec = args(index, tmp_path / "one")
    spec.train_count = 270
    one = build_subset(spec)
    spec.output = str(tmp_path / "two")
    spec.workers = 2
    two = build_subset(spec)
    for name in ("train_rows", "valid_rows", "requests", "token_valid", "pilot_requests"):
        assert Path(one[name]).read_bytes() == Path(two[name]).read_bytes()
    assert validate_subset(tmp_path / "two")["status"] == "passed"


def test_detects_rebound_wrong_row_and_modified_parent(tmp_path):
    index = parent(tmp_path)
    out = tmp_path / "subset"
    subset = build_subset(args(index, out))
    path = Path(subset["requests"])
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    row = json.loads(lines[0])
    row["row_index"] += 1
    lines[0] = (json.dumps(row) + "\n").encode()
    path.write_bytes(b"".join(lines))
    # Even rebinding the artifact digest must not hide incorrect source alignment.
    subset["requests_sha256"] = sha(path)
    subset["files"]["requests"]["sha256"] = sha(path)
    (out / "subset.json").write_text(json.dumps(subset))
    with pytest.raises(ValueError, match="row identity"):
        validate_subset(out)
    path.write_bytes(original)
    source = Path(json.loads(index.read_text())["shards"][1]["path"])
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source bytes"):
        validate_subset(out)


def test_insufficient_unique_holdout_and_training_rows_fail(tmp_path):
    index = parent(tmp_path)
    spec = args(index, tmp_path / "short_valid")
    spec.valid_count = 6
    with pytest.raises(ValueError, match="unique eligible valid"):
        build_subset(spec)
    spec = args(index, tmp_path / "short_train")
    spec.train_count = 8
    with pytest.raises(ValueError, match="eligible train rows"):
        build_subset(spec)


def test_explicit_empty_or_malformed_sequences_and_synthetic_provenance(tmp_path):
    index = parent(tmp_path)
    _, config = _index(index)
    row = pair(0, "train")
    row["chains"][0] = ""
    assert _eligibility(row, "train", config)[0] == "noncanonical_or_empty"
    row = pair(0, "train")
    row["chain_metadata"] = {"0": {"synthetic_residue_mask": [0, 1]}}
    with pytest.raises(ValueError, match="synthetic"):
        _eligibility(row, "train", config)


def test_fast_file_check_rejects_changed_worker_or_rebound_invalid_rows(tmp_path):
    index = parent(tmp_path)
    out = tmp_path / "subset"
    subset = build_subset(args(index, out))
    path = Path(subset["request_shards"][0]["path"])
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Worker request artifact changed"):
        validate_subset_files(out)
    path.write_bytes(original)
    rows_path = Path(subset["train_rows"])
    rows = np.load(rows_path)
    rows[1] = rows[0]
    np.save(rows_path, rows)
    subset["train_rows_sha256"] = sha(rows_path)
    subset["files"]["train_rows"]["sha256"] = sha(rows_path)
    (out / "subset.json").write_text(json.dumps(subset))
    with pytest.raises(ValueError, match="repeated"):
        validate_subset_files(out)
