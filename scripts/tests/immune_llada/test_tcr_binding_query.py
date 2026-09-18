"""CPU-only regression tests for runtime T1 inputs and cache isolation.

Run in pllm: ``python -m pytest -q scripts/tests/immune_llada/test_tcr_binding_query.py``.
No model weights, GPU jobs, or prepared benchmark copies are created.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dllm.pipelines.immune_llada.data import BioSeqChain, GrammarBioSeqCollator, GrammarTokenizer
from dllm.pipelines.immune_llada.data.profiles import fallback_region_profile, profile_digest
from dllm.pipelines.immune_llada.data.sources import _receptor_chains
from downstream.benchmark.tcr_binding.query_protocol import (
    INPUT_RECORD, POOLING, TCRBindingEmbedder, TCRBindingQueryProtocol,
    observed_residue_pool,
)
from downstream.benchmark.tcr_binding.run_retrained_ours import (
    _check_cache_manifest, _feature_identity, _require_provenance,
)


@pytest.fixture
def manifest(tmp_path):
    profile = fallback_region_profile()
    # A genuinely variable distribution tests order-independent seeding.
    profile["region_distributions"]["tcr_alpha"]["CDR3"] = {"7": 1, "13": 1, "16": 1}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"tcr_region_profile": {
        "profile": profile, "digest": profile_digest(profile),
    }}))
    return path


@pytest.fixture
def protocol(manifest):
    return TCRBindingQueryProtocol(manifest)


def test_observed_sequence_and_beta_first_are_preserved(protocol):
    record = protocol.record("GILGFVFTL", "CASSQETQYF")
    assert record.chain_roles == ["peptide", "tcr_beta", "tcr_alpha"]
    beta, alpha = record.chains[1:]
    assert beta.regions["CDR3"] == "ASSQETQY"
    observed = "".join(aa for aa, synthetic in zip(
        beta.sequence, beta.metadata["synthetic_residue_mask"],
    ) if not synthetic)
    assert observed == "CASSQETQYF"
    assert set(alpha.sequence) == {"X"}
    assert all(alpha.metadata["synthetic_residue_mask"])
    assert not record.labels


def test_runtime_completion_reuses_training_implementation(protocol):
    import hashlib
    beta = "CASSQETQYF"
    key = int.from_bytes(hashlib.sha256(beta.encode()).digest(), "big")
    expected = _receptor_chains(
        alpha_fv="", beta_fv="", alpha_cdr3="", beta_cdr3=beta,
        is_junction=True, profile=protocol.profile,
        source="tcr_t1_runtime", row_index=key,
    )
    assert protocol.record("GILGFVFTL", beta).chains[1:] == expected


def test_core_convention_is_explicit_and_keeps_terminal_anchors(manifest):
    query = TCRBindingQueryProtocol(manifest, cdr3_format="core")
    beta = query.record("GILGFVFTL", "CASSQETQYF").chains[1]
    assert beta.regions["CDR3"] == "CASSQETQYF"
    assert beta.metadata["observed_anchor_regions"] == {}


def test_completion_independent_of_epitope_and_call_order(protocol):
    first = protocol.record("GILGFVFTL", "CASSQETQYF")
    protocol.record("SLLMWITQV", "CASSLGQAYF")
    other = protocol.record("SLLMWITQV", "CASSQETQYF")
    assert first.chains[1:] == other.chains[1:]
    assert first == protocol.record("GILGFVFTL", "CASSQETQYF")


@pytest.mark.parametrize("bad", ["", "XXX", None, float("nan"), "AA?", "AA AA", "AA-AA"])
def test_invalid_inputs_are_rejected(protocol, bad):
    with pytest.raises(ValueError):
        protocol.record("GILGFVFTL", bad)
    with pytest.raises(ValueError):
        protocol.record(bad, "CASSQETQYF")


def test_masked_query_has_no_labels_and_keeps_encoder_sequences(protocol):
    tokenizer = GrammarTokenizer()
    collator = GrammarBioSeqCollator(tokenizer, max_sequence_length=512)
    record = protocol.record("GILGFVFTL", "CASSQETQYF")
    before = collator([record])
    after = protocol.collate([record], collator, decoder_mask_token_id=tokenizer.mask_token_id)
    query = after["relation_target_mask"]
    assert query.sum() == 1
    assert after["input_ids"][query].item() == tokenizer.mask_token_id
    assert not after["input_ids"].eq(tokenizer.special_id("<binding>")).any()
    assert "labels" not in after
    assert torch.equal(before["encoder_input_ids"], after["encoder_input_ids"])
    assert torch.equal(before["input_ids"][~query], after["input_ids"][~query])
    synthetic = after["synthetic_residue_mask"]
    assert not after["diffusion_loss_mask"][synthetic].any()
    assert not after["diffusion_eligible_mask"][synthetic].any()


@pytest.mark.parametrize("label", ["binding", "nonbinding"])
def test_gold_records_are_refused(protocol, label):
    record = protocol.record("GILGFVFTL", "CASSQETQYF")
    record.labels["relation"] = label
    with pytest.raises(ValueError, match="gold"):
        protocol.collate([record], GrammarBioSeqCollator(GrammarTokenizer()),
                         decoder_mask_token_id=32)


def test_pmhc_cannot_accidentally_mask_presentation(protocol):
    record = protocol.record("GILGFVFTL", "CASSQETQYF")
    record.chains.insert(0, BioSeqChain("Y" * 34, "mhc"))
    with pytest.raises(ValueError, match="layout"):
        protocol.collate([record], GrammarBioSeqCollator(GrammarTokenizer()),
                         decoder_mask_token_id=32)


def test_native_remap_collator_uses_decoder_space_mask(protocol):
    from examples.llada.protein_fusion_model import RemapCollator
    tokenizer = GrammarTokenizer()
    lookup = torch.arange(tokenizer.vocab_size) + 100
    collator = RemapCollator(GrammarBioSeqCollator(tokenizer), lookup)
    record = protocol.record("GILGFVFTL", "CASSQETQYF")
    batch = protocol.collate([record], collator, decoder_mask_token_id=999)
    assert batch["input_ids"][batch["relation_target_mask"]].item() == 999
    assert not batch["input_ids"].eq(lookup[tokenizer.special_id("<binding>")]).any()
    assert batch["encoder_input_ids"].max() < 100


def test_pool_ignores_synthetic_special_padding_and_query(protocol):
    batch = protocol.collate(
        [protocol.record("GILGFVFTL", "CASSQETQYF"),
         protocol.record("SLLMWITQV", "CASSLGQAYNEQFF")],
        GrammarBioSeqCollator(GrammarTokenizer()), decoder_mask_token_id=32,
    )
    observed = batch["residue_mask"] & ~batch["synthetic_residue_mask"]
    assert observed.sum(1).tolist() == [19, 23]
    hidden = torch.full((*observed.shape, 3), 12345.0)
    hidden[observed] = 2.0
    assert torch.equal(observed_residue_pool(hidden, batch), torch.full((2, 3), 2.0))


def test_adapter_frontend_is_batch_and_label_invariant(protocol):
    tokenizer = GrammarTokenizer()
    calls = []

    def hidden(batch):
        assert "labels" not in batch
        assert batch["input_ids"][batch["relation_target_mask"]].eq(999).all()
        calls.append(batch)
        # Deterministic residue-wise features allow batching/pool verification.
        return batch["input_ids"].float().unsqueeze(-1).repeat(1, 1, 2), None

    backbone = SimpleNamespace(
        model=SimpleNamespace(_decoder_mask_token_id=999), hidden=2,
        feature_source="decoder", pool_mode="global", batch_size=2,
        collator=GrammarBioSeqCollator(tokenizer), device="cpu", _final_hidden=hidden,
    )
    rows = [("GILGFVFTL", "CASSQETQYF", 0), ("SLLMWITQV", "CASSLGQAYF", 1)]
    embedder = TCRBindingEmbedder(backbone, protocol)
    def embed(data):
        return embedder.embed_pairs(beta=[row[1] for row in data],
                                    peptide=[row[0] for row in data])
    original = embed(rows)
    embedder._cache.clear()
    embedder.batch_size = 1
    flipped = embed([(pep, beta, 1-label) for pep, beta, label in reversed(rows)])
    np.testing.assert_array_equal(original, flipped[::-1])
    assert len(calls) == 3


def test_profile_and_format_and_length_invalidate_protocol(manifest):
    first = TCRBindingQueryProtocol(manifest)
    assert first.sha256 != TCRBindingQueryProtocol(manifest, cdr3_format="core").sha256
    assert first.sha256 != TCRBindingQueryProtocol(manifest, max_length=512).sha256
    data = json.loads(manifest.read_text())
    data["tcr_region_profile"]["profile"]["seed"] += 1
    data["tcr_region_profile"]["digest"] = profile_digest(data["tcr_region_profile"]["profile"])
    manifest.write_text(json.dumps(data))
    assert first.sha256 != TCRBindingQueryProtocol(manifest).sha256


def test_bad_profile_digest_rejected(manifest):
    data = json.loads(manifest.read_text())
    data["tcr_region_profile"]["digest"] = "wrong"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        TCRBindingQueryProtocol(manifest)


def test_no_truncation(manifest):
    with pytest.raises(ValueError, match="length"):
        TCRBindingQueryProtocol(manifest, max_length=50).record("GILGFVFTL", "CASSQETQYF")


@pytest.mark.parametrize("field", ["input_protocol_sha256", "model_assets_sha256", "pooling", "input_record"])
def test_old_or_changed_cache_rejected(field):
    manifest = {
        "checkpoint_sha256": "ckpt", "pairs_sha256": "pairs", "n_pairs": 2,
        "feature_source": "post_llada_final_hidden_state", "pooling": POOLING,
        "input_record": INPUT_RECORD, "input_protocol_sha256": "query",
        "model_assets_sha256": "assets", "feature_dim": 768,
    }
    args = {key: manifest[key] for key in (
        "checkpoint_sha256", "pairs_sha256", "n_pairs", "input_protocol_sha256",
        "model_assets_sha256",
    )}
    _check_cache_manifest(manifest, **args)
    assert _feature_identity(manifest)
    del manifest[field]
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        _check_cache_manifest(manifest, **args)


def test_predictions_and_heads_must_match_feature_identity():
    for artifact in ("head", "predictions", "fold metrics"):
        with pytest.raises(RuntimeError, match="provenance mismatch"):
            _require_provenance({}, {"feature_identity": "new"}, artifact)


def test_runner_feature_cache_round_trip_and_protocol_rejection(protocol, tmp_path, monkeypatch):
    import sys
    from downstream.benchmark.tcr_binding import run_retrained_ours as runner

    tokenizer = GrammarTokenizer()
    calls = []
    def build(*args, **kwargs):
        calls.append(kwargs)
        model = torch.nn.Linear(1, 1)
        model._decoder_mask_token_id = 999
        return SimpleNamespace(
            model=model, hidden=2, feature_source="decoder", pool_mode="global",
            batch_size=2, tokenizer=tokenizer,
            collator=GrammarBioSeqCollator(tokenizer), device="cpu",
            _final_hidden=lambda batch: (
                batch["input_ids"].float().unsqueeze(-1).repeat(1, 1, 2), None,
            ),
        )
    monkeypatch.setitem(sys.modules, "common.model_api", SimpleNamespace(build_embedder=build))
    kwargs = dict(
        pairs=[("GILGFVFTL", "CASSQETQYF"), ("SLLMWITQV", "CASSLGQAYF")],
        checkpoint=tmp_path / "fake_checkpoint", checkpoint_sha256="ckpt",
        cache_dir=tmp_path / "features", embedding_batch_size=2,
        feature_chunk_size=1, force=False, query_protocol=protocol, model_assets={},
    )
    first, manifest = runner.extract_or_load_features(**kwargs)
    second, _ = runner.extract_or_load_features(**kwargs)
    np.testing.assert_array_equal(first, second)
    assert len(calls) == 1  # cache hit must not reload model
    assert manifest["input_protocol_sha256"] == protocol.sha256
    changed = TCRBindingQueryProtocol(protocol.provenance["completion_manifest"], cdr3_format="core")
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        runner.extract_or_load_features(**{**kwargs, "query_protocol": changed})
    assert len(calls) == 1


def test_aggregator_refuses_historical_or_mixed_protocols(tmp_path, monkeypatch):
    from downstream.benchmark.tcr_binding import run_retrained_ours as runner
    monkeypatch.setattr(runner, "OUT_ROOT", tmp_path)
    path = runner._fold_dir("test", "AS", 1) / "seen_test" / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"fold": 1, "metrics": {}}))
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        runner.summarize("test", "AS", feature_identity="new")


def test_paired_cdr3_preserves_both_observed_chains(manifest):
    protocol = TCRBindingQueryProtocol(manifest, input_mode="cdr3ab")
    record = protocol.record("GILGFVFTL", "CASSQETQYF", "CAVRDSNYQLIW")
    assert record.chain_roles == ["peptide", "tcr_beta", "tcr_alpha"]
    for chain, expected in zip(record.chains[1:], ("CASSQETQYF", "CAVRDSNYQLIW")):
        observed = "".join(aa for aa, synthetic in zip(
            chain.sequence, chain.metadata["synthetic_residue_mask"],
        ) if not synthetic)
        assert observed == expected
        assert chain.regions["CDR3"] == expected[1:-1]
    batch = protocol.collate([record], GrammarBioSeqCollator(GrammarTokenizer()),
                             decoder_mask_token_id=999)
    assert batch["input_ids"][batch["relation_target_mask"]].item() == 999
    assert (batch["residue_mask"] & ~batch["synthetic_residue_mask"]).sum() == 31


@pytest.mark.parametrize("alpha", [None, "", "XXX"])
def test_paired_mode_refuses_missing_alpha(manifest, alpha):
    protocol = TCRBindingQueryProtocol(manifest, input_mode="cdr3ab")
    with pytest.raises(ValueError, match="CDR3A"):
        protocol.record("GILGFVFTL", "CASSQETQYF", alpha)


def test_track_is_part_of_protocol_identity(manifest):
    beta = TCRBindingQueryProtocol(manifest)
    paired = TCRBindingQueryProtocol(manifest, input_mode="cdr3ab")
    assert beta.sha256 != paired.sha256
    assert paired.provenance["input_columns"] == ["Epitope", "CDR3B", "CDR3A"]
    with pytest.raises(ValueError, match="beta-only"):
        beta.record("GILGFVFTL", "CASSQETQYF", "CAVRDSNYQLIW")


def test_same_beta_different_alpha_is_not_one_feature_key():
    import pandas as pd
    from downstream.benchmark.tcr_binding import run_retrained_ours as runner
    frame = pd.DataFrame({
        "Epitope": ["GILGFVFTL"] * 2, "CDR3B": ["CASSQETQYF"] * 2,
        "CDR3A": ["CAVRDSNYQLIW", "CAVGGGSNYQLIW"], "Affinity": [0, 1],
        "MHC": ["A01", "HLA-A*02:01"], "LongA": ["AAA", "BBB"],
    })
    keys = sorted(runner._unique_pairs(frame, "cdr3ab"))
    assert len(keys) == 2
    assert len(runner._unique_pairs(frame, "cdr3b")) == 1
    indices = runner.frame_feature_indices(frame, {key: i for i, key in enumerate(keys)}, "cdr3ab")
    assert set(indices) == {0, 1}
    changed = frame.copy()
    changed["MHC"], changed["LongA"], changed["Affinity"] = "ignored", "ignored", 0
    assert keys == sorted(runner._unique_pairs(changed, "cdr3ab"))
    changed.loc[0, "CDR3A"] = "CAVVVVNYQLIW"
    assert runner.pair_universe_sha256(keys) != runner.pair_universe_sha256(
        sorted(runner._unique_pairs(changed, "cdr3ab")))


def test_paired_embedder_cache_contains_alpha(manifest):
    protocol = TCRBindingQueryProtocol(manifest, input_mode="cdr3ab")
    tokenizer = GrammarTokenizer()
    backbone = SimpleNamespace(
        model=SimpleNamespace(_decoder_mask_token_id=999), hidden=1,
        feature_source="decoder", pool_mode="global", batch_size=2,
        collator=GrammarBioSeqCollator(tokenizer), device="cpu",
        _final_hidden=lambda batch: (batch["input_ids"].float().unsqueeze(-1), None),
    )
    embedder = TCRBindingEmbedder(backbone, protocol)
    peptide, beta = ["GILGFVFTL"] * 2, ["CASSQETQYF"] * 2
    alpha = ["CAVRDSNYQLIW", "CAWWWWWWWWWW"]
    values = embedder.embed_pairs(beta, alpha=alpha, peptide=peptide)
    assert len(embedder._cache) == 2
    assert values[0, 0] != values[1, 0]
    np.testing.assert_array_equal(values[::-1], embedder.embed_pairs(
        beta, alpha=alpha[::-1], peptide=peptide))


def test_official_others_members_are_separate_and_as_only():
    from downstream.benchmark.tcr_binding.retrained_protocol import (
        resolve_train_member, resolve_test_member,
    )
    assert "_others_seen/" in resolve_train_member(neg_source="AS", fold=1, track="cdr3ab")
    assert "_only_seen/" in resolve_train_member(neg_source="AS", fold=1)
    assert "_others_unseen/" in resolve_test_member(
        neg_source="AS", fold=1, eval_set="unseen_independent", track="cdr3ab")
    with pytest.raises(ValueError, match="AS"):
        resolve_train_member(neg_source="PS", fold=1, track="cdr3ab")


def test_others_outputs_do_not_mix_with_beta_only(monkeypatch, tmp_path):
    from downstream.benchmark.tcr_binding import run_retrained_ours as runner
    monkeypatch.setattr(runner, "OUT_ROOT", tmp_path)
    assert runner._experiment_root("same", "AS", "cdr3ab") != runner._experiment_root("same", "AS")
    # No matching others baseline table yet: must not use the beta-only table.
    runner.write_rankings({"track": "cdr3ab", "tag": "same"}, "AS")
    assert not list(tmp_path.iterdir())
