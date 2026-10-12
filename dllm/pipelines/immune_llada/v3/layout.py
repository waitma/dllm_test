"""Anchor-preserving no-ESMC layouts; test with pytest scripts/tests/immune_llada/test_v3.py.

Only synthetic-mask provenance determines which residues may be removed.
Target capacity and attention visibility never depend on a hidden reference length.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ..data.grammar import GrammarBioSeqCollator, GrammarTokenizer
from ..data.records import BioSeqChain, BioSeqRecord, CHAIN_ROLE_TO_ID
from .vocabulary import make_tokenizer

SCOPES = {"unknown": 0, "full": 1, "junction": 2, "core": 3, "fragment": 4, "missing": 5}
RECEPTORS = frozenset({"antibody_heavy", "antibody_light", "nanobody_vhh", "tcr_alpha", "tcr_beta"})
ROLE_ORDER = ("mhc", "peptide", "antigen", "antibody_heavy", "antibody_light", "nanobody_vhh", "tcr_beta", "tcr_alpha")


@dataclass(frozen=True)
class Observation:
    sequence: str
    role: str
    scope: str
    original_indices: tuple[int, ...]


@dataclass(frozen=True)
class GenerationRequest:
    """Reference-free A4/V5 request, including unconditional chain generation."""
    conditions: tuple[BioSeqChain, ...]
    targets: tuple[tuple[str, str], ...]  # (role, explicit observation scope)

    def __post_init__(self):
        roles=[c.role for c in self.conditions]+[role for role,_ in self.targets]
        if not self.targets or len(set(roles))!=len(roles):
            raise ValueError("Generation needs unique condition/target roles")
        if any(role not in RECEPTORS or scope not in {"full","junction","core","fragment"} for role,scope in self.targets):
            raise ValueError("Invalid generation target role or scope")


def observe(chain: BioSeqChain) -> Observation:
    """Preserve observed anchors, including ones assigned to framework regions."""
    synthetic = chain.metadata.get("synthetic_residue_mask")
    if synthetic is None:
        scope = chain.metadata.get("sequence_scope", "full")
        scope = {"cdr3": "core", "cdr3_core": "core", "cdr3_junction": "junction", "fv": "full"}.get(scope, scope)
        if scope not in SCOPES:
            raise ValueError(f"Unknown sequence scope: {scope}")
        return Observation(chain.sequence, chain.role, scope, tuple(range(len(chain.sequence))))
    if len(synthetic) != len(chain.sequence) or any(v not in (0, 1, False, True) for v in synthetic):
        raise ValueError("Invalid synthetic residue provenance")
    indices = tuple(i for i, flag in enumerate(synthetic) if not flag)
    if not indices:
        return Observation("", chain.role, "missing", ())
    if len(indices) == len(chain.sequence):
        return Observation(chain.sequence, chain.role, "full", indices)
    if indices != tuple(range(indices[0], indices[-1] + 1)):
        raise ValueError("Discontinuous observations require separate fragment blocks; do not concatenate across gaps")
    anchors = chain.metadata.get("observed_anchor_regions", {})
    scope = "junction" if anchors.get("FR3") and anchors.get("FR4") else "core"
    if set(chain.metadata.get("observed_regions", [])) - {"CDR3"}:
        scope = "fragment"
    sequence = "".join(chain.sequence[i] for i in indices)
    if scope == "junction" and (not sequence.startswith(anchors["FR3"]) or not sequence.endswith(anchors["FR4"])):
        raise ValueError("Observed anchors disagree with synthetic mask")
    return Observation(sequence, chain.role, scope, indices)


def ordered_chains(record: BioSeqRecord) -> list[BioSeqChain]:
    unknown = set(record.chain_roles) - set(ROLE_ORDER)
    if unknown:
        raise ValueError(f"v3 immune suite requires explicit supported roles: {unknown}")
    return sorted(record.chains, key=lambda c: ROLE_ORDER.index(c.role))


class ImmuneCollator:
    """V5 uses fixed-capacity object grammar; A1-A3 retain legacy canvases.

    A4 uses observed spans and one terminal.

    mode='representation' uses true observed lengths. mode='generation' masks
    target slots without reading target residues into model input. Callers may
    specify target_roles for chain completion; the default targets all receptors.
    """

    def __init__(self, ablation="A4", *, max_length=2048, full_capacity=256, fragment_capacity=64,
                 mode="train", target_roles: Iterable[str] | None = None, generation_lengths: dict[str, int] | None = None):
        if ablation not in {"A0", "A1", "A2", "A3", "A4", "V5"}:
            raise ValueError("Unknown architecture control")
        if mode not in {"train", "generation", "representation"}:
            raise ValueError(mode)
        if min(full_capacity, fragment_capacity) < 2:
            raise ValueError("Capacity includes at least one residue and one terminal")
        self.ablation, self.max_length, self.mode = ablation, max_length, mode
        self.generation_lengths = generation_lengths
        if ablation == "V5" and generation_lengths is not None:
            raise ValueError("V5 uses full_capacity, not exact generation_lengths")
        self.full_capacity, self.fragment_capacity = full_capacity, fragment_capacity
        self.target_roles = RECEPTORS if target_roles is None else frozenset(target_roles)
        self.tokenizer = make_tokenizer(ablation)
        self.legacy_tokenizer = GrammarTokenizer()
        self.legacy = GrammarBioSeqCollator(self.legacy_tokenizer, max_sequence_length=max_length,
                                          max_protein_length=max_length, fixed_receptor_lengths=ablation != "V5")

    @classmethod
    def from_config(cls, config, **kwargs):
        """Use checkpoint layout consistently in training and downstream callers."""
        return cls(config.ablation, max_length=config.max_length,
                   full_capacity=config.full_capacity, fragment_capacity=config.fragment_capacity, **kwargs)

    def __call__(self, records):
        records = [BioSeqRecord.from_prepared_dict(r) if isinstance(r, dict) else r for r in records]
        if any(isinstance(r, GenerationRequest) for r in records) and (
            self.ablation not in {"A4", "V5"} or self.mode != "generation"
        ):
            raise ValueError("Reference-free requests require A4/V5 generation mode")
        if self.ablation == "V5":
            from .v5_canvas import canvas_row
            rows = [canvas_row(self, record) for record in records]
        elif self.ablation != "A4":
            return self._legacy(records)
        else:
            rows = [self._row(record) for record in records]
        size = max(len(row["input_ids"]) for row in rows)
        fill = {"input_ids": self.tokenizer.pad_token_id, "labels": -100, "chain_ids": -1}
        batch = {key: torch.tensor([row[key] + [fill.get(key, 0)] * (size-len(row[key])) for row in rows], dtype=torch.long)
                 for key in rows[0]}
        for key in ("attention_mask", "target_slot_mask", "residue_mask", "terminal_mask", "eligible_mask"):
            batch[key] = batch[key].bool()
        return batch

    def _legacy(self, records):
        if self.ablation == "V5":
            for record in records:
                for chain in record.chains:
                    synthetic = chain.metadata.get("synthetic_residue_mask")
                    if synthetic is not None and (
                        len(synthetic) != len(chain.sequence)
                        or any(flag not in (0, 1, False, True) for flag in synthetic)
                    ):
                        raise ValueError("Invalid synthetic residue provenance")
        batch = self.legacy(records)
        roles = torch.zeros_like(batch["input_ids"])
        scopes = torch.zeros_like(roles)
        chain_ids = batch["position_ids_chain"]
        for b, record in enumerate(records):
            chains = ordered_chains(record)
            ids = chain_ids[b][chain_ids[b] >= 0].unique(sorted=True).tolist()
            if len(ids) != len(chains):
                raise ValueError("Legacy renderer/semantic chain mapping disagrees")
            for uid, chain in zip(ids, chains):
                positions = chain_ids[b] == uid
                observed_tokens = batch["input_ids"][b][positions & batch["residue_mask"][b]]
                expected = torch.tensor(self.legacy_tokenizer.encode_residues(chain.sequence))
                if not torch.equal(observed_tokens, expected):
                    raise ValueError("Legacy chain order mismatch")
                roles[b, positions] = CHAIN_ROLE_TO_ID[chain.role]
                if self.ablation == "V5":
                    # Keep synthetic scaffolds and discontinuous observed regions intact.
                    scope = chain.metadata.get("sequence_scope", "full")
                    scope = {"cdr3": "core", "cdr3_core": "core", "cdr3_junction": "junction", "fv": "full"}.get(scope, scope)
                    scopes[b, positions] = SCOPES.get(scope, SCOPES["unknown"])
                else:
                    scopes[b, positions] = SCOPES[observe(chain).scope]
        eligible = batch["diffusion_eligible_mask"].bool()
        target_slots = batch["chain_slot_mask"].bool() & ~batch["fixed_context_mask"].bool()
        requested = torch.zeros_like(target_slots)
        for role in self.target_roles:
            requested |= roles == CHAIN_ROLE_TO_ID[role]
        target_slots &= requested
        eligible &= ~batch["chain_slot_mask"].bool() | requested
        if self.mode == "representation":
            target_slots.zero_()
            eligible.zero_()
        labels = batch["input_ids"].masked_fill(~eligible, -100)
        if self.ablation == "V5":
            # Preserve the prepared full-length scaffold and residue coordinates.
            # Only provenance-marked placeholders become masks; an observed X
            # remains an observed residue. These slots attend but carry no loss.
            synthetic_slots = batch["synthetic_residue_mask"].bool()
            batch["input_ids"] = batch["input_ids"].masked_fill(
                synthetic_slots, self.legacy_tokenizer.mask_token_id)
        if self.mode == "generation":
            batch["input_ids"] = batch["input_ids"].masked_fill(target_slots, self.legacy_tokenizer.mask_token_id)
            labels.fill_(-100)
        if self.ablation == "A0":
            remap=torch.tensor([self.tokenizer.token_to_id.get(self.legacy_tokenizer.token(i),-1)
                                for i in range(self.legacy_tokenizer.vocab_size)])
            batch["input_ids"]=remap[batch["input_ids"]]
            supervised=labels!=-100
            labels[supervised]=remap[labels[supervised]]
            if batch["input_ids"].lt(0).any() or labels[supervised].lt(0).any():
                raise ValueError("Legacy vocabulary is missing an emitted token")
        return dict(input_ids=batch["input_ids"], labels=labels, attention_mask=batch["attention_mask"],
                    role_ids=roles, scope_ids=scopes, chain_ids=chain_ids,
                    position_ids=batch["position_ids_inner"].clamp_min(0),
                    target_slot_mask=target_slots, eligible_mask=eligible,
                    residue_mask=batch["residue_mask"] & ~batch["synthetic_residue_mask"],
                    terminal_mask=((batch["input_ids"] == self.tokenizer.special_id("<protd>")) if self.ablation == "V5" else batch["chain_eos_mask"]) & eligible)

    def _row(self, record):
        row = {k: [] for k in ("input_ids", "labels", "attention_mask", "role_ids", "scope_ids", "chain_ids",
                              "position_ids", "target_slot_mask", "residue_mask", "terminal_mask", "eligible_mask")}
        tok = self.tokenizer

        def add(token, role=0, scope=0, chain=-1, pos=0, slot=False, residue=False, terminal=False, label=-100):
            values = (token, label, 1, role, scope, chain, pos, slot, residue, terminal, label != -100)
            for key, value in zip(row, values):
                row[key].append(int(value))

        if isinstance(record,GenerationRequest):
            observations=[observe(c) for c in record.conditions]+[Observation("",role,scope,()) for role,scope in record.targets]
            if any(o.role not in ROLE_ORDER for o in observations):
                raise ValueError("Unsupported generation condition role")
            observations.sort(key=lambda o:ROLE_ORDER.index(o.role))
            target_roles={role for role,_ in record.targets}
            labels,metadata={},{}
        else:
            observations=[observe(c) for c in ordered_chains(record)]
            target_roles=self.target_roles
            labels,metadata=record.labels,record.metadata
        marker = "<tcr>" if any(o.role.startswith("tcr_") for o in observations) else "<nb>" if any(o.role=="nanobody_vhh" for o in observations) else "<ab>"
        add(tok.special_id(marker))
        relation = labels.get("relation", "unknown")
        relation = f"<{relation}>"
        relation_id = tok.special_id(relation if relation in tok.token_to_id else "<unknown>")
        relation_supervised = self.mode == "train" and metadata.get("relation_supervised") and relation in {"<binding>", "<nonbinding>"}
        add(relation_id, label=relation_id if relation_supervised else -100)
        for uid, obs in enumerate(observations):
            role, scope = CHAIN_ROLE_TO_ID[obs.role], SCOPES[obs.scope]
            add(tok.special_id("<prots>"), role, scope, uid)
            if obs.scope == "missing":
                add(tok.special_id("<null>"), role, scope, uid, 1)
                add(tok.special_id("<protd>"), role, scope, uid, 2)
                continue
            target = obs.role in target_roles and self.mode != "representation"
            capacity = (self.full_capacity if obs.scope == "full" else self.fragment_capacity) if target else len(obs.sequence)+1
            if self.mode != "generation" and len(obs.sequence)+1 > capacity:
                raise ValueError(f"capacity_overflow:{obs.role}:{len(obs.sequence)+1}>{capacity}")
            residues = tok.encode_residues(obs.sequence) if not target or self.mode != "generation" else []
            content = residues + [tok.special_id("<protd>")]
            for j in range(capacity):
                is_residue = j < len(residues)
                is_end = j == len(residues)
                label = content[j] if target and self.mode == "train" and j < len(content) else -100
                token = tok.mask_token_id if (target and self.mode == "generation") or j >= len(content) else content[j]
                # Reference-derived masks below are labels/pooling metadata ONLY.
                # denoise() receives role/scope/chain/position and capacity mask.
                add(token, role, scope, uid, j+1, target, is_residue, is_end and self.mode != "generation", label)
        if len(row["input_ids"]) > self.max_length:
            raise ValueError(f"sequence_budget_overflow:{len(row['input_ids'])}>{self.max_length}")
        return row
