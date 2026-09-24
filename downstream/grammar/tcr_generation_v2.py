"""Fixed-canvas TCR generation helpers.

Run: python -m pytest scripts/tests/immune_llada/test_tcr_generation_v2.py -q
"""
from __future__ import annotations

import torch

from dllm.pipelines.immune_llada.data import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.data.grammar import (
    FIXED_HEAVY_DECODER_SLOTS, FIXED_LIGHT_DECODER_SLOTS,
)
from downstream.grammar.masks import tcr_generation_partial_mask
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import build_generation_mask

AA = "ACDEFGHIKLMNPQRSTVWY"
BETA_AA = FIXED_HEAVY_DECODER_SLOTS - 1
ALPHA_AA = FIXED_LIGHT_DECODER_SLOTS - 1


def beta_record(epitope=None, mhc=None, placeholder="A"):
    """Reference-free full beta target with an explicitly unknown alpha partner."""
    if placeholder not in AA or len(placeholder) != 1:
        raise ValueError("placeholder must be one canonical amino acid")
    if mhc and not epitope:
        raise ValueError("MHC conditioning requires an epitope")
    context = ([BioSeqChain(mhc, "mhc")] if mhc else [])
    context += [BioSeqChain(epitope, "peptide")] if epitope else []
    alpha = BioSeqChain("X" * ALPHA_AA, "tcr_alpha",
        metadata={"synthetic_residue_mask": [1] * ALPHA_AA})
    return BioSeqRecord(context + [BioSeqChain(placeholder * BETA_AA, "tcr_beta"), alpha],
        task_type="tcr", source="tcr_v2_beta_generation")


def beta_targets(batch, records):
    indices = [sum(c.role in {"mhc", "peptide"} for c in r.chains) for r in records]
    if len(set(indices)) != 1:
        raise ValueError("batch must share the same conditioning layout")
    partial = tcr_generation_partial_mask(batch, target_chain_indices=indices[0])
    targets = build_generation_mask(batch, partial)
    if not targets.sum(1).eq(FIXED_HEAVY_DECODER_SLOTS).all():
        raise ValueError("v2 beta generation must target every beta slot, including EOS")
    return targets


def decode_beta_candidates(tokens, batch, targets, tokenizer):
    """Keep every attempt, including missing EOS and failed CDR3 extraction."""
    from downstream.benchmark.tcr_generation_bench.scoring import extract_cdr3b_anarci, is_valid_cdr3b

    lookup = {tokenizer.encode_residues(aa)[0]: aa for aa in AA}
    results = []
    for row in range(len(tokens)):
        ids = tokens[row][targets[row]].tolist()
        terminated = tokenizer.eos_token_id in ids
        end = ids.index(tokenizer.eos_token_id) if terminated else len(ids)
        residues = ids[:end]
        valid_tokens = bool(residues) and all(i in lookup for i in residues)
        sequence = "".join(lookup.get(i, "X") for i in residues)
        results.append(dict(beta_sequence=sequence, terminated_by_eos=terminated,
            valid_beta_tokens=valid_tokens, beta_token_ids=ids))
    extracted = extract_cdr3b_anarci([
        r["beta_sequence"] if r["terminated_by_eos"] and r["valid_beta_tokens"] else ""
        for r in results
    ])
    for row, cdr3 in zip(results, extracted):
        valid = bool(cdr3) and is_valid_cdr3b(cdr3)
        row.update(sequence=cdr3 if valid else "", valid=valid,
            extraction_method="benchmark_anarci_with_regex_fallback",
            failure_reason=("missing_eos" if not row["terminated_by_eos"] else
                "invalid_beta_tokens" if not row["valid_beta_tokens"] else
                "cdr3_extraction_failed" if not valid else None),
            protocol="tcr_v2_full_beta_eos_then_cdr3")
    return results


def residue_allowlist(model, tokenizer):
    """Translate canonical residues to decoder IDs, including remapped checkpoints."""
    grammar_ids = torch.tensor([tokenizer.encode_residues(aa)[0] for aa in AA])
    inverse = getattr(model, "llada_to_grammar_ids", None)
    if inverse is None:
        return tuple(grammar_ids.tolist())
    ids = torch.where(torch.isin(inverse.cpu(), grammar_ids))[0]
    if ids.numel() != len(AA):
        raise ValueError("decoder must have exactly one token per canonical amino acid")
    return tuple(ids.tolist())


class FixedBetaGenerationProtocol:
    """V5 adapter interface with fixed budgets instead of sampled CDR3 lengths."""
    def __init__(self):
        self.provenance = dict(protocol_id="tcr_v2_full_beta_eos_then_cdr3",
            length_prior="none", reference_access="none_in_generator",
            targets="full_beta_including_eos", unknown_regions="fixed_X_alpha",
            extraction="benchmark_anarci_with_regex_fallback")

    def lengths(self, seed, target_key, n):
        return [BETA_AA] * n

    def record(self, epitope, mhc, core_length, sample_key, placeholder="A"):
        # Compatibility arguments cannot affect target length or content.
        return beta_record(epitope, mhc, placeholder)

    def collate(self, records, collator):
        if not records or any(r.chain_roles != ["mhc", "peptide", "tcr_beta", "tcr_alpha"]
                or r.labels or r != beta_record(r.chains[1].sequence, r.chains[0].sequence,
                r.chains[2].sequence[0]) for r in records):
            raise ValueError("v2 conditional records must be reference-free beta placeholders")
        batch = collator(records)
        batch.pop("labels", None)
        return batch, beta_targets(batch, records)
