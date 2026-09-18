"""Run in protenix_abtcr after pllm: python -m pytest scripts/tests/immune_llada/test_tcr_generation_v5.py -q."""
import torch

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import apply_decoder_corruption_to_encoder
from downstream.grammar.tcr_generation_v5 import BetaGenerationProtocol

EP = "GILGFVFTL"
MHC = "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY"


def test_beta_core_only_and_unknown_alpha_cdr12():
    p = BetaGenerationProtocol()
    r = p.record(EP, MHC, 12, "candidate1")
    t = GrammarTokenizer()
    b, target = p.collate([r], GrammarBioSeqCollator(t))
    assert target.sum() == 12
    assert b["position_ids_chain"][target].eq(2).all()
    for chain in r.chains[2:]:
        for region in ("CDR1", "CDR2"):
            assert set(chain.regions[region]) == {"X"}
    assert set(r.chains[3].sequence) == {"X"}
    assert not b["relation_target_mask"].any()
    assert b["input_ids"].eq(t.special_id("<binding>")).sum() == 2
    assert "labels" not in b


def test_placeholder_hidden_from_both_streams():
    p = BetaGenerationProtocol()
    t = GrammarTokenizer()
    batches = [p.collate([p.record(EP, MHC, 12, "same", placeholder=c)],
        GrammarBioSeqCollator(t)) for c in ("A", "G")]
    hidden = []
    for b, target in batches:
        decoder = b["input_ids"].clone()
        decoder[target] = t.mask_token_id
        encoder = apply_decoder_corruption_to_encoder(b, target, t.mask_token_id)
        hidden.append((decoder, encoder))
    assert all(torch.equal(x, y) for x, y in zip(*hidden))


def test_length_prior_reproducible_without_reference_access():
    p = BetaGenerationProtocol()
    a = p.lengths(42, "target1", 100)
    p.lengths(42, "unrelated_target", 100)
    assert a == p.lengths(42, "target1", 100)
    assert set(a).issubset(set(map(int, p.profile["region_distributions"]["tcr_beta"]["CDR3"])))


def test_longest_profile_core_no_truncation():
    p = BetaGenerationProtocol()
    length = max(map(int, p.profile["region_distributions"]["tcr_beta"]["CDR3"]))
    r = p.record(EP, MHC, length, "longest")
    _, target = p.collate([r], GrammarBioSeqCollator(GrammarTokenizer()))
    assert target.sum() == length


def test_residue_decode_does_not_require_adapter_id_to_token():
    from downstream.grammar.tcr_generation_v5 import residue_lookup, AA
    tok = GrammarTokenizer()
    # Real HF adapter lacks a top-level id_to_token, and GrammarTokenizer.token
    # consequently returns <base:n>. Decoding must use its actual encoding map.
    tok.token = lambda i: f"<base:{i}>"
    lookup = residue_lookup(tok)
    assert set(lookup.values()) == AA
    assert "".join(lookup[i] for i in tok.encode_residues("CASSQETQYF")) == "CASSQETQYF"
    assert tok.mask_token_id not in lookup
