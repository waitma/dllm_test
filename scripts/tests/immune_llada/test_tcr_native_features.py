"""Run in pllm: python -m pytest scripts/tests/immune_llada/test_tcr_native_features.py -q."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dllm.pipelines.immune_llada.data import GrammarTokenizer, GrammarBioSeqCollator
from downstream.grammar.tcr_features import TCRRepresentationProtocol, TCRRuntimeEmbedder


def test_null_unknown_no_binding_target():
    p = TCRRepresentationProtocol(input_mode="cdr3ab")
    r = p.record("CASSQETQYF", "CAVRDTQYF")
    tok = GrammarTokenizer()
    b = p.collate([r], GrammarBioSeqCollator(tok))
    assert b["input_ids"].eq(tok.special_id("<unknown>")).sum() == 1
    assert not b["input_ids"].eq(tok.special_id("<binding>")).any()
    assert not b["relation_target_mask"].any()
    assert "labels" not in b
    assert r.chain_roles == ["tcr_beta", "tcr_alpha"]


def test_missing_alpha_is_explicit_unknown_and_not_observed():
    p = TCRRepresentationProtocol(input_mode="cdr3ab")
    r = p.record("CASSQETQYF", "")
    assert set(r.chains[1].sequence) == {"X"}
    assert all(r.chains[1].metadata["synthetic_residue_mask"])
    with pytest.raises(ValueError):
        p.record("CASSQETQYF", None)


def test_completion_and_pooling_order_invariant():
    tok = GrammarTokenizer()
    calls = []
    def forward(batch):
        calls.append(batch)
        x = batch["input_ids"].float()
        return torch.stack([x, x.square()], -1), None
    bb = SimpleNamespace(hidden=2, name="stub", device="cpu",
        collator=GrammarBioSeqCollator(tok), _final_hidden=forward)
    seqs = ["CASSQETQYF", "CASSLGQAYNEQFF"]
    a = TCRRuntimeEmbedder(backbone=bb, batch_size=2).embed(seqs)
    b = TCRRuntimeEmbedder(backbone=bb, batch_size=1).embed(seqs[::-1])[::-1]
    np.testing.assert_allclose(a, b)
    for batch in calls:
        assert not batch["relation_target_mask"].any()
    with pytest.raises(ValueError, match="epitope"):
        TCRRuntimeEmbedder(backbone=bb).embed_pairs(seqs, peptide=["TEST", "TEST"])


def test_labels_cannot_enter_protocol():
    p = TCRRepresentationProtocol()
    record = p.record("CASSQETQYF")
    record.labels["relation"] = "binding"
    with pytest.raises(ValueError):
        p.collate([record], GrammarBioSeqCollator(GrammarTokenizer()))


def test_multilabel_pair_not_counted_as_known_negative():
    import pandas as pd
    from scripts.downstream.run_tcr_native_repr import receptor_groups, multilabel_fewshot
    original = pd.DataFrame({"cdr3b": ["b1", "b1", "b2"], "cdr3a": ["a1", "a1", "a2"],
        "peptide": ["e1", "e2", "e3"]})
    grouped = receptor_groups(original)
    assert len(grouped) == 2 and grouped.peptide.iloc[0] == ("e1", "e2")
    train = pd.DataFrame({"peptide": [("e1", "e2"), ("e3",)]})
    scores = multilabel_fewshot(np.array([[0., 1.], [1., 0.]]), train, grouped, shots=(1,), seeds=2)
    assert scores["auroc_by_shot"][1]["mean"] == 1.0
    assert len(scores["episodes"]) == 6
