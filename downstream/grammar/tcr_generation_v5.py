"""Runtime-v5 pMHC-conditioned CDR3beta generation; paired generation is deferred.

Run through scripts/downstream/run_tcr_native_generation.py --task preflight.
Only the beta CDR3 core is generated; universal C/F junction anchors are fixed.
Unknown FR/CDR1/2 and the entire alpha partner remain X throughout sampling.
No reference receptor sequence or reference length is accepted by this API.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from dllm.pipelines.immune_llada.data import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.data.sources import complete_tcr_chain
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import (
    BioSeqGenerateConfig, generate_bioseq, build_generation_mask,
)
from downstream.benchmark.tcr_binding.query_protocol import TCRBindingQueryProtocol, _sequence
from downstream.grammar.common import (
    load_grammar_checkpoint, build_eval_collator, _inverse_remap_llada_tokens,
)

REGIONS = ("FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4")
AA = set("ACDEFGHIKLMNPQRSTVWY")


def _model_is_fixed_v2(model) -> bool:
    """Return whether a loaded checkpoint uses the fixed-canvas contract."""

    config = getattr(model, "config", None)
    marker = getattr(config, "fixed_receptor_lengths", None)
    if marker is not None:
        return bool(marker)
    marker = getattr(model, "fixed_receptor_lengths", None)
    if marker is not None:
        return bool(marker)
    collator = getattr(model, "_fusion_eval_collator", None)
    marker = getattr(collator, "fixed_receptor_lengths", None)
    if marker is not None:
        return bool(marker)
    return bool(getattr(model, "_predict_eos", False))


def residue_lookup(tokenizer):
    """Invert actual residue encoding, independent of tokenizer adapter internals."""
    mapping = {}
    for aa in sorted(AA):
        ids = tokenizer.encode_residues(aa)
        if len(ids) != 1 or ids[0] in mapping:
            raise ValueError("expected one distinct grammar-space token per amino acid")
        mapping[int(ids[0])] = aa
    return mapping


class BetaGenerationProtocol:
    def __init__(self):
        base = TCRBindingQueryProtocol()
        self.profile = base.profile
        self.provenance = dict(base.provenance)
        self.provenance.update(protocol_id="tcr_v5_pmhc_beta_core_generation_v1",
            input_columns=["epitope", "mhc_pseudo"], input_mode="pmhc_beta_generation",
            input_record="mhc_peptide_completed_beta_alpha",
            recognition="fixed_binding_desired_generation_condition_not_prediction_label",
            pooling="none", cdr3_format="fixed_C_F_flanks_plus_generated_core",
            targets="beta_CDR3_core_only", unknown_regions="X_fixed_including_entire_alpha",
            length_prior="frozen_v5_training_profile_beta_CDR3_no_eval_lengths",
            reference_access="none_in_generator", seed=42, k=100, batch_size=8,
            max_iter=32, sampling_strategy="gumbel_argmax", cfg_scale=0., temperature=1.)
        self.provenance["greedy_control"] = "one_extra_argmax_at_training_modal_core_length"

    def lengths(self, seed, target_key, n):
        values = self.profile["region_distributions"]["tcr_beta"]["CDR3"]
        sizes = np.array(sorted(int(x) for x in values))
        weights = np.array([values[str(int(x))] for x in sizes], dtype=float)
        identity = hashlib.sha256(f"{seed}:{target_key}:beta_lengths".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(identity[:8], "big"))
        return rng.choice(sizes, size=n, p=weights / weights.sum()).tolist()

    def record(self, epitope, mhc, core_length, sample_key, placeholder="A"):
        epitope, mhc = _sequence(epitope, "epitope"), _sequence(mhc, "MHC pseudo")
        if len(mhc) != 34:
            raise ValueError("expected provided 34-residue MHC pseudo sequence")
        if placeholder not in AA or len(placeholder) != 1 or not 1 <= core_length <= 100:
            raise ValueError("invalid generation window")
        index = int.from_bytes(hashlib.sha256(str(sample_key).encode()).digest(), "big")
        common = dict(profile=self.profile, source="tcr_v5_beta_generate", row_index=index)
        beta = complete_tcr_chain("tcr_beta", core=placeholder * core_length,
            leading_anchor="C", trailing_anchor="F", **common)
        alpha = complete_tcr_chain("tcr_alpha", **common)
        record = BioSeqRecord([BioSeqChain(mhc, "mhc"), BioSeqChain(epitope, "peptide"), beta, alpha],
            task_type="tcr", source="tcr_v5_beta_generate")
        if sum(len(c.sequence) for c in record.chains) + 16 > 1024:
            raise ValueError("generation layout exceeds max length")
        return record

    def collate(self, records, collator):
        if any(r.chain_roles != ["mhc", "peptide", "tcr_beta", "tcr_alpha"] or r.labels for r in records):
            raise ValueError("unexpected generation layout or supplied labels")
        batch = collator(records)
        if any(x != "tcr_pmhc" for x in batch["grammar_names"]):
            raise ValueError("wrong grammar")
        batch.pop("labels", None)
        targets = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
        for row, record in enumerate(records):
            beta = record.chains[2]
            positions = torch.where(batch["residue_mask"][row] &
                batch["attention_mask"][row].bool() & batch["position_ids_chain"][row].eq(2))[0]
            if len(positions) != len(beta.sequence):
                raise ValueError("chain-to-decoder mapping/truncation error")
            start = sum(len(beta.regions[r]) for r in REGIONS[:5])
            targets[row, positions[start:start + len(beta.regions["CDR3"])]] = True
        if (targets & batch["synthetic_residue_mask"]).any() or batch["relation_target_mask"].any():
            raise ValueError("generation must not target unknown scaffold/relation")
        partial = ~targets
        if not torch.equal(build_generation_mask(batch, partial), targets):
            raise ValueError("sampler generation mask does not match CDR3 core")
        return batch, targets


class V5BetaSampler:
    def __init__(self, checkpoint, device="cuda"):
        self.device = torch.device(device)
        self.model, self.tokenizer = load_grammar_checkpoint(checkpoint, device=self.device)
        if _model_is_fixed_v2(self.model):
            raise ValueError(
                "tcr_generation_v5 is CDR3-only and incompatible with fixed_receptor_lengths v2"
            )
        self.collator = build_eval_collator(self.model, self.tokenizer)
        self.protocol = BetaGenerationProtocol()
        self.residue_tokens = residue_lookup(self.tokenizer)

    def sample_records(self, records, *, seed, max_iter=32, strategy="gumbel_argmax"):
        batch, target = self.protocol.collate(records, self.collator)
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        target = target.to(self.device)
        torch.manual_seed(seed)
        config = BioSeqGenerateConfig(max_iter=max_iter, sampling_strategy=strategy,
            temperature=1., cfg_scale=0.)
        with torch.no_grad():
            tokens, _, history = generate_bioseq(self.model, batch, partial_mask=~target,
                config=config, return_history=True)
        # Never allow a sampler revision to modify X, anchors, pMHC or structure.
        for state in history:
            if not torch.equal(state[~target], batch["input_ids"][~target]):
                raise ValueError("fixed generation context changed")
        if not history[0][target].eq(self.model.config.mask_token_id).all():
            raise ValueError("targets visible at step zero")
        if tokens[target].eq(self.model.config.mask_token_id).any():
            raise ValueError("unfinished generation")
        grammar_tokens = _inverse_remap_llada_tokens(self.model, tokens)
        results = []
        for row in range(len(records)):
            ids = grammar_tokens[row][target[row]].tolist()
            # Decode each token explicitly: do not silently strip invalid tokens.
            chars = [self.residue_tokens.get(int(i), f"<invalid:{i}>") for i in ids]
            core = "".join(chars)
            valid = all(len(c) == 1 and c in AA for c in chars)
            results.append({"sequence": "C" + core + "F", "valid": valid,
                "core_token_ids": ids, "core_length": len(ids)})
        return results, {"steps": len(history) - 1, "context_invariant": True,
            "initial_targets_masked": True, "target_tokens": target.sum(1).tolist()}

    def generate(self, epitope, mhc, target_key, *, k=100, batch_size=8, seed=42):
        lengths = self.protocol.lengths(seed, target_key, k)
        results = []
        for start in range(0, k, batch_size):
            records = [self.protocol.record(epitope, mhc, lengths[i], f"{seed}:{target_key}:{i}")
                for i in range(start, min(start + batch_size, k))]
            batch_seed = int.from_bytes(hashlib.sha256(f"{seed}:{target_key}:{start}".encode()).digest()[:4], "big")
            generated, checks = self.sample_records(records, seed=batch_seed)
            for i, item in enumerate(generated, start):
                results.append(dict(item, index=i, batch_seed=batch_seed, **checks))
        if len(results) != k:
            raise ValueError("candidate budget mismatch")
        return results
