"""CPU checks for the CDR iteration runner and actual encoder boundary.

Run: python -m pytest scripts/tests/bioseq/test_cdr_iter_sweep.py -q
Uses tiny test networks; no pretrained weights or GPU are needed.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import build_generation_mask
from downstream.grammar.cdr_infill import build_antibody_record, _build_cdr_partial_mask
from examples.llada.protein_fusion_model import LLaDAEsmcFusion, RemapCollator
from scripts.downstream.run_ab_cdr_iter_sweep import generate, summarize_rows, target_encoder_slots


@pytest.mark.parametrize("role,target_string", [("heavy", "EFGH"), ("light", "MTQSP")])
@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_real_fusion_path_updates_cdr_encoder_without_hidden_reference(role, target_string, steps):
    tok = GrammarTokenizer()
    lookup = torch.arange(tok.vocab_size) + 100
    inverse = torch.full((256,), -1, dtype=torch.long)
    inverse[lookup] = torch.arange(tok.vocab_size)
    residue_ids = lookup[torch.tensor(tok.encode_residues("ACDEFGHIKLMNPQRSTVWYX"))]

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(tok.vocab_size, 8)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(256, 16)
            self.config = SimpleNamespace(vocab_size=256, pad_token_id=1)
            self.calls = 0

        def get_input_embeddings(self):
            return self.emb

        def forward(self, inputs_embeds, **kwargs):
            logits = torch.full((*inputs_embeds.shape[:2], 256), -100.0)
            logits[..., residue_ids[self.calls % len(residue_ids)]] = torch.arange(
                inputs_embeds.size(1)).float() / inputs_embeds.size(1)
            self.calls += 1
            return SimpleNamespace(logits=logits)

    record = build_antibody_record("ACDEFGHIK", "DIQMTQSPSS")
    batch = RemapCollator(GrammarBioSeqCollator(tok), lookup)([record])
    partial = _build_cdr_partial_mask(batch, tok, record, role, "CDR3", target_string)
    target = build_generation_mask(batch, partial)
    slots = target_encoder_slots(batch, target)
    assert len(slots) == len(target_string)
    assert {s[2] for s in slots} == ({0} if role == "heavy" else {1})
    original = batch["encoder_input_ids"].clone()
    outcomes = []
    for poisoned in (False, True):
        b = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        if poisoned:
            b["input_ids"][target] = int(residue_ids[-1])
            for r, col, chain, position in slots:
                b["encoder_input_ids"][r, chain, position] = inverse[residue_ids[-1]]
        model = LLaDAEsmcFusion(Decoder(), Encoder(), 8, 200, tok.mask_token_id,
                               residue_token_ids=residue_ids.tolist()).eval()
        model.register_buffer("llada_to_grammar_ids", inverse)
        output, trace = generate(model, b, partial, steps, audit=True)
        assert trace[0]["pending_mask"] == len(target_string)
        assert len(trace) <= steps
        if steps > 1:
            assert trace[-1]["generated_visible"] > 0
        assert not output[target].eq(200).any()
        assert not model.encoder._forward_pre_hooks
        outcomes.append((output, trace))
    assert torch.equal(outcomes[0][0], outcomes[1][0])
    assert outcomes[0][1] == outcomes[1][1]
    assert torch.equal(batch["encoder_input_ids"], original)


def test_macro_aggregation_is_not_pooled_sample_average():
    result = summarize_rows([{"fold": 0, "aar": .5}, {"fold": 0, "aar": .5}, {"fold": 1, "aar": 0.}])
    assert result["average_aar_all_folds"] == 25.
    assert result["fold_sd_ddof0"] == 25.
    assert np.isclose(result["average_aar"], 100 / 3)
    assert result["n_sequences"] == 3
    with pytest.raises(ValueError):
        summarize_rows([])
