"""CPU regressions: python -m pytest -q scripts/tests/immune_llada/test_tcr_binding_long.py.

Activate pllm first. Synthetic fixtures only; no GPU or benchmark copies.
"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer
from dllm.pipelines.immune_llada.data.profiles import fallback_region_profile, profile_digest
from downstream.benchmark.tcr_binding import run_retrained_ours as runner
from downstream.benchmark.tcr_binding.query_protocol import (
    LONG_INPUT_RECORD, TCRBindingQueryProtocol, observed_residue_pool,
)

PAIR = ("GILGFVFTL", "AQTCASSQETQYFGG", "DVVCAVRDSNYQLIWGG")


@pytest.fixture
def protocol(tmp_path):
    profile = fallback_region_profile()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"tcr_region_profile": {
        "profile": profile, "digest": profile_digest(profile),
    }}))
    return TCRBindingQueryProtocol(path, input_mode="longab")


def frame():
    return pd.DataFrame({
        "Epitope": [PAIR[0]], "LongB": [PAIR[1]], "LongA": [PAIR[2]],
        "CDR3B": ["CASSQETQYF"], "CDR3A": ["CAVRDSNYQLIW"],
        "Affinity": [1], "MHC": ["ignored"],
    })


def test_long_chains_are_verbatim_and_masked_with_all_residue_pool(protocol):
    record = protocol.record(*PAIR)
    assert record.chain_roles == ["peptide", "tcr_beta", "tcr_alpha"]
    assert [chain.sequence for chain in record.chains] == list(PAIR)
    assert not record.labels
    assert all(not chain.regions and not chain.metadata for chain in record.chains)
    tokenizer = GrammarTokenizer()
    batch = protocol.collate([record], GrammarBioSeqCollator(tokenizer),
                             decoder_mask_token_id=tokenizer.mask_token_id)
    assert batch["relation_target_mask"].sum() == 1
    assert batch["input_ids"][batch["relation_target_mask"]].item() == tokenizer.mask_token_id
    assert not batch["input_ids"].eq(tokenizer.special_id("<binding>")).any()
    assert "labels" not in batch
    assert batch["synthetic_residue_mask"].sum() == 0
    mask = batch["residue_mask"].bool() & batch["attention_mask"].bool()
    assert mask.sum() == sum(map(len, PAIR))
    hidden = torch.arange(mask.numel(), dtype=torch.float32).reshape(1, -1, 1)
    torch.testing.assert_close(observed_residue_pool(hidden, batch), hidden[mask].mean(0, keepdim=True))


@pytest.mark.parametrize("column,value", [
    ("LongA", None), ("LongB", "AA?"), ("CDR3A", None),
    ("CDR3B", "WWWW"), ("LongB", "CASSQETQYFCASSQETQYF"),
    ("LongB", "XXXX"), ("Epitope", "A-A"),
])
def test_bad_source_rows_fail_without_dropping(column, value):
    data = frame()
    data.loc[0, column] = value
    with pytest.raises(RuntimeError):
        runner._validate_frame(data, "source", "others_longab")


def test_missing_audit_column_fails():
    with pytest.raises(RuntimeError, match="CDR3A"):
        runner._validate_frame(frame().drop(columns="CDR3A"), "source", "others_longab")


def test_overlength_missing_chain_and_overlapping_cdr3_fail(protocol):
    protocol.max_length = sum(map(len, PAIR)) + 7
    with pytest.raises(ValueError, match="length"):
        protocol.record(*PAIR)
    with pytest.raises(ValueError, match="LongA"):
        protocol.record(*PAIR[:2])
    data = frame()
    data.loc[0, ["LongB", "CDR3B"]] = ["AAAAA", "AAA"]
    with pytest.raises(RuntimeError, match="exactly once"):
        runner._validate_frame(data, "source", "others_longab")


def test_data_selection_keys_and_isolation(protocol, monkeypatch, tmp_path, capsys):
    data = frame()
    calls = []
    def loader(**kwargs):
        calls.append(kwargs)
        return data.copy(), b"original", {"member": "original"}
    monkeypatch.setattr(runner, "load_official_train", loader)
    monkeypatch.setattr(runner, "load_official_test", loader)
    frames, provenance = runner.load_protocol("AS", "others_longab")
    assert len(calls) == 20 and all(c["track"] == "cdr3ab" for c in calls)
    assert provenance["input_mode"] == "longab"
    for splits in frames.values():
        for loaded in splits.values():
            pd.testing.assert_frame_equal(loaded, data)
    keys = runner.build_pair_universe(frames, "others_longab")
    assert keys == [PAIR]
    changed = data.copy()
    changed.loc[0, ["Affinity", "CDR3A", "MHC"]] = [0, "audit-only", "other"]
    assert runner._unique_pairs(changed, "others_longab") == set(keys)
    changed.loc[0, "LongA"] = "A" + PAIR[2]
    assert runner._unique_pairs(changed, "others_longab") != set(keys)
    old = TCRBindingQueryProtocol(protocol.provenance["completion_manifest"], input_mode="cdr3ab")
    assert old.sha256 != protocol.sha256
    assert protocol.provenance["audit_only_columns"] == ["CDR3B", "CDR3A"]
    monkeypatch.setattr(runner, "OUT_ROOT", tmp_path)
    roots = {runner._experiment_root("tag", "AS", track)
             for track in ("cdr3b", "cdr3ab", "others_cdr3b", "others_longab")}
    assert len(roots) == 4
    summary = runner.summarize("tag", "AS", feature_identity="new", track="others_longab")
    assert summary["input_columns"] == ["Epitope", "LongB", "LongA"]
    runner.write_rankings(summary, "AS")
    assert "no larger beta-only dataset ranking" in capsys.readouterr().out


def test_long_embedding_and_cache_round_trip(protocol, tmp_path, monkeypatch):
    calls = []
    tokenizer = GrammarTokenizer()
    def build(*args, **kwargs):
        calls.append(kwargs)
        model = torch.nn.Linear(1, 1)
        model._decoder_mask_token_id = 999
        return SimpleNamespace(
            model=model, hidden=2, feature_source="decoder", pool_mode="global",
            batch_size=2, tokenizer=tokenizer, device="cpu",
            collator=GrammarBioSeqCollator(tokenizer),
            _final_hidden=lambda batch: (batch["input_ids"].float().unsqueeze(-1).repeat(1, 1, 2), None),
        )
    monkeypatch.setitem(sys.modules, "common.model_api", SimpleNamespace(build_embedder=build))
    kwargs = dict(pairs=[PAIR], checkpoint=tmp_path / "fake", checkpoint_sha256="ckpt",
                  cache_dir=tmp_path / "features", embedding_batch_size=2, feature_chunk_size=1,
                  force=False, query_protocol=protocol, model_assets={})
    first, manifest = runner.extract_or_load_features(**kwargs)
    second, _ = runner.extract_or_load_features(**kwargs)
    np.testing.assert_array_equal(first, second)
    assert len(calls) == 1
    assert manifest["input_record"] == LONG_INPUT_RECORD
    assert list(pd.read_csv(tmp_path / "features/pairs.csv").columns) == ["Epitope", "LongB", "LongA"]
    other = TCRBindingQueryProtocol(protocol.provenance["completion_manifest"], input_mode="cdr3ab")
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        runner.extract_or_load_features(**{**kwargs, "query_protocol": other})
