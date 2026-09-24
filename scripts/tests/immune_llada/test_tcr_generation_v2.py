"""Run: python -m pytest scripts/tests/immune_llada/test_tcr_generation_v2.py -q."""
from types import SimpleNamespace

import pytest
import torch

from dllm.pipelines.immune_llada.data import GrammarTokenizer, GrammarBioSeqCollator
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import apply_decoder_corruption_to_encoder
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import BioSeqGenerateConfig, generate_bioseq
from downstream.grammar.tcr_generation_v2 import (
    FixedBetaGenerationProtocol, beta_record, beta_targets, residue_allowlist,
)
from downstream.grammar.tcr_generation import BioSeqTcrSampler
from downstream.grammar.tcr_generation_v5 import V5BetaSampler
from examples.llada.protein_fusion_model import RemapCollator

EP = "GILGFVFTL"
MHC = "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"


def bundle(monkeypatch, *, remapped=False):
    import downstream.grammar.tcr_generation as tcr
    import downstream.grammar.tcr_generation_v5 as v5
    tok = GrammarTokenizer()
    collator = GrammarBioSeqCollator(tok, fixed_receptor_lengths=True)
    model = SimpleNamespace(config=SimpleNamespace(fixed_receptor_lengths=True))
    if remapped:
        lookup = torch.arange(tok.vocab_size) + 100
        inverse = torch.full((int(lookup.max()) + 1,), -1, dtype=torch.long)
        inverse[lookup] = torch.arange(tok.vocab_size)
        model.llada_to_grammar_ids = inverse
        collator = RemapCollator(collator, lookup)
    model._fusion_eval_collator = collator
    for module in (tcr, v5):
        monkeypatch.setattr(module, "load_grammar_checkpoint", lambda *a, **k: (model, tok))
    return tok, model, collator


@pytest.mark.parametrize("epitope,mhc", [(None, None), (EP, None), (EP, MHC)])
def test_full_beta_masks_eos_and_hides_placeholder(epitope, mhc):
    tok = GrammarTokenizer()
    collator = GrammarBioSeqCollator(tok, fixed_receptor_lengths=True)
    hidden = []
    for placeholder in ("A", "G"):
        records = [beta_record(epitope, mhc, placeholder)]
        batch = collator(records)
        target = beta_targets(batch, records)
        assert target.sum() == 167
        assert (target & batch["chain_eos_mask"]).sum() == 1
        assert not (target & batch["fixed_context_mask"]).any()
        hidden.append((batch["input_ids"].masked_fill(target, tok.mask_token_id),
            apply_decoder_corruption_to_encoder(batch, target, tok.mask_token_id)))
    assert all(torch.equal(a, b) for a, b in zip(*hidden))


@pytest.mark.parametrize("remapped", [False, True])
@pytest.mark.parametrize("condition", ["none", "peptide", "pmhc"])
def test_generation_retains_failures_and_uses_no_length_prior(monkeypatch, remapped, condition):
    import downstream.grammar.tcr_generation as tcr
    from downstream.benchmark.tcr_generation_bench import scoring
    tok, model, collator = bundle(monkeypatch, remapped=remapped)
    monkeypatch.setattr(tcr, "_sample_lengths", lambda *a: pytest.fail("v2 sampled a length"))
    monkeypatch.setattr(scoring, "extract_cdr3b_anarci", lambda seqs: ["CASSQETQYF" if s else "" for s in seqs])
    calls = []

    def generate(model, batch, partial_mask, **kwargs):
        target = ~partial_mask & batch["attention_mask"]
        assert target.sum(1).eq(167).all()
        calls.append(target)
        tokens = batch["input_ids"].clone()
        if remapped:
            tokens = model.llada_to_grammar_ids[tokens]
        for row in range(len(tokens)):
            pos = torch.where(target[row])[0]
            tokens[row, pos] = tok.encode_residues("A")[0]
            if row == 0:
                seq = tok.encode_residues("ACASSQETQYFGAG")
                tokens[row, pos[:len(seq)]] = torch.tensor(seq)
                tokens[row, pos[len(seq):]] = tok.eos_token_id
        return tokens, torch.zeros_like(tokens).float()

    monkeypatch.setattr(tcr, "run_grammar_generate", generate)
    sampler = BioSeqTcrSampler("memory", device="cpu")
    if condition == "none":
        result = sampler.unconditional_cdr3b(3, batch_size=2)
    else:
        result = sampler.conditional_cdr3b(EP, 3, batch_size=2,
            mhc_pseudo=MHC if condition == "pmhc" else None)
    assert len(result) == 3 and result[1] == ""
    assert sampler.last_candidates[1]["failure_reason"] == "missing_eos"
    assert len(calls) == 2


def test_v5_dispatch_and_fixed_protocol_ignore_core_length(monkeypatch):
    bundle(monkeypatch, remapped=True)
    sampler = V5BetaSampler("memory", device="cpu")
    assert isinstance(sampler.protocol, FixedBetaGenerationProtocol)
    p = sampler.protocol
    assert p.lengths(0, "a", 3) == [166] * 3
    first = p.record(EP, MHC, 5, "first")
    assert first == p.record(EP, MHC, 82, "other")
    batch, target = p.collate([first], sampler.collator)
    assert target.sum() == 167


def test_v5_real_generation_keeps_failed_attempts(monkeypatch):
    import downstream.grammar.tcr_generation_v5 as v5
    from scripts.tests.immune_llada.test_v2_token_model import make_model
    tok, _, collator = bundle(monkeypatch)
    model = make_model(tok).eval()
    model._fusion_eval_collator = collator
    with torch.no_grad():
        model.decoder.head.weight.zero_()
        model.decoder.head.bias.zero_()
        model.decoder.head.bias[tok.eos_token_id] = 30
    monkeypatch.setattr(v5, "load_grammar_checkpoint", lambda *a, **k: (model, tok))
    sampler = V5BetaSampler("memory", device="cpu")
    result = sampler.generate(EP, MHC, "test", k=3, batch_size=2)
    assert len(result) == 3
    assert all(r["terminated_by_eos"] and not r["valid"] for r in result)
    assert all(r["failure_reason"] == "invalid_beta_tokens" for r in result)
    assert all(r["context_invariant"] and r["initial_targets_masked"] for r in result)


def test_single_infill_hidden_targets_do_not_affect_either_stream(monkeypatch):
    import downstream.grammar.tcr_generation as tcr
    tok, _, _ = bundle(monkeypatch)
    inputs = []

    def capture(model, batch, partial_mask, **kwargs):
        target = ~partial_mask & batch["attention_mask"]
        assert target.sum() == 3
        inputs.append((batch["input_ids"].masked_fill(target, tok.mask_token_id),
            apply_decoder_corruption_to_encoder(batch, target, tok.mask_token_id)))
        return batch["input_ids"], torch.zeros_like(batch["input_ids"]).float()

    monkeypatch.setattr(tcr, "run_grammar_generate", capture)
    sampler = BioSeqTcrSampler("memory", device="cpu")
    sampler.infill_cdr3b(["CAAAF"], 3)
    sampler.infill_cdr3b(["CGGGF"], 3)
    assert all(torch.equal(a, b) for a, b in zip(*inputs))


@pytest.mark.parametrize("remapped", [False, True])
def test_pair_infill_masks_only_window_and_translates_allowlist(monkeypatch, remapped):
    import downstream.grammar.tcr_generation as tcr
    tok, model, _ = bundle(monkeypatch, remapped=remapped)

    def copy(model, batch, partial_mask, allowed_token_ids, **kwargs):
        assert (~partial_mask & batch["attention_mask"]).sum() == 3
        assert partial_mask[batch["chain_eos_mask"]].all()
        assert tok.eos_token_id + (100 if remapped else 0) not in allowed_token_ids
        tokens = batch["input_ids"]
        if remapped:
            tokens = model.llada_to_grammar_ids[tokens]
        return tokens, torch.zeros_like(tokens).float()

    monkeypatch.setattr(tcr, "run_grammar_generate", copy)
    sampler = BioSeqTcrSampler("memory", device="cpu")
    row = sampler.infill_cdr3b_in_pair([("ACDE", "ACASSQETQYFGAG", "ASSQETQY")], 3)[0]
    assert row["length_ok"] and row["truth_window"] == row["pred_window"]
    assert row["length_condition"] == "reference_window"


@pytest.mark.parametrize("editing,self_correct", [(0., False), (.1, False), (0., True)])
def test_real_sampler_blocks_eos_for_infill_all_phases(editing, self_correct):
    from scripts.tests.immune_llada.test_v2_token_model import make_model
    from downstream.grammar.common import tcr_pair_record
    from downstream.grammar.masks import cdr3b_span_partial_mask
    tok = GrammarTokenizer()
    model = make_model(tok).eval()
    with torch.no_grad():
        model.decoder.head.weight.zero_()
        model.decoder.head.bias.zero_()
        model.decoder.head.bias[tok.eos_token_id] = 30
        model.decoder.head.bias[tok.encode_residues("A")[0]] = 20
    batch = GrammarBioSeqCollator(tok, fixed_receptor_lengths=True)([tcr_pair_record("ACDE", "CASSF")])
    partial = cdr3b_span_partial_mask(batch, 0, (1, 4))
    config = BioSeqGenerateConfig(max_iter=2, sampling_strategy="argmax",
        allowed_token_ids=residue_allowlist(model, tok), editing_threshold=editing,
        self_correct=self_correct, max_post_steps=2)
    tokens, _, history = generate_bioseq(model, batch, partial_mask=partial, config=config, return_history=True)
    target = ~partial & batch["attention_mask"]
    assert tokens[target].eq(tok.encode_residues("A")[0]).all()
    for state in history:
        assert torch.equal(state[partial], batch["input_ids"][partial])
