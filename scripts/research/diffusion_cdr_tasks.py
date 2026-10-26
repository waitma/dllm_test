"""Fixed-framework antibody CDR targets for the existing diffusion mask branch.

Activate pllm; USE_TF=0 python -B -m pytest
scripts/tests/immune_llada/test_diffusion_cdr_tasks.py -q.
Region annotations select losses only; they never enter encoder features.
"""
import math

import torch

from dllm.pipelines.immune_llada.data.records import BioSeqRecord, CHAIN_ROLE_TO_ID, REGION_ORDER

CDRS = ("H1", "H2", "H3", "L1", "L2", "L3")
SCOPES = ("chain", "cdr_single", "cdr_multi")
AB_ROLES = ("antibody_heavy", "antibody_light")


def default_cdr_tasks():
    return dict(mask_branch_ratios=dict(chain=.5, cdr_single=.25, cdr_multi=.25),
                full_mask_probability=.25,
                annotation="exact_complete_observed_antibody_regions",
                context="fixed_framework_and_unselected_cdrs",
                selection="single_uniform_or_uniform_multiple_without_replacement")


def validate_cdr_tasks(config):
    expected = default_cdr_tasks()
    if set(config) != set(expected) or set(config["mask_branch_ratios"]) != set(SCOPES):
        raise ValueError("Invalid CDR task fields")
    values = list(config["mask_branch_ratios"].values())
    if any(not math.isfinite(x) or x < 0 for x in values) or not math.isclose(sum(values), 1.):
        raise ValueError("CDR mask-branch ratios must sum to one")
    if not 0 <= config["full_mask_probability"] <= 1:
        raise ValueError("Invalid CDR full-mask probability")
    for name in ("annotation", "context", "selection"):
        if config[name] != expected[name]:
            raise ValueError("Unsupported CDR semantics: " + name)


def cdr_spans(record):
    """Return exact zero-based spans only when all present antibody chains qualify.

    Requires complete FR1..FR4/CDR1..CDR3 segmentation and real observations.
    No substring guesses, numbering inference, synthetic frameworks or TCR reuse.
    """
    if isinstance(record, dict):
        record = BioSeqRecord.from_prepared_dict(record)
    chains = [c for c in record.chains if c.role in AB_ROLES]
    if not chains or any(c.role.startswith("tcr_") for c in record.chains):
        return []
    spans = []
    for chain in chains:
        regions = chain.regions
        if (not all(regions.get(name) for name in REGION_ORDER)
                or "".join(regions[name] for name in REGION_ORDER) != chain.sequence
                or any(chain.metadata.get("synthetic_residue_mask", []))):
            return []
        offset = 0
        for name in REGION_ORDER:
            end = offset + len(regions[name])
            if name.startswith("CDR"):
                group = int(name[-1]) + (3 if chain.role == "antibody_light" else 0)
                spans.append((group, chain.role, offset, end))
            offset = end
    return spans


def annotate_cdrs(batch, records):
    """Use original sequence offsets on the fixed slots, never observed-only offsets."""
    ids = torch.zeros_like(batch["input_ids"])
    for row, record in enumerate(records):
        for group, role, start, end in cdr_spans(record):
            positions = (batch["target_slot_mask"][row]
                         & batch["role_ids"][row].eq(CHAIN_ROLE_TO_ID[role])).nonzero().flatten()
            chosen = positions[start:end]
            if len(chosen) != end-start or not batch["residue_mask"][row, chosen].all():
                raise ValueError("CDR positions do not match observed receptor slots")
            ids[row, chosen] = group
    return ids


def select_cdr_targets(batch, is_edit, generator, config, *, scope=None, target_cdrs=None):
    """Select whole CDR target regions; the caller then samples diffusion corruption."""
    bsz = len(batch["input_ids"])
    ids = batch["cdr_ids"]
    requested = torch.zeros(bsz, dtype=torch.long)
    if scope is not None and scope not in SCOPES:
        raise ValueError("Unknown mask scope")
    if target_cdrs is not None and (scope not in SCOPES[1:] or not target_cdrs
            or len(set(target_cdrs)) != len(target_cdrs) or any(n not in CDRS for n in target_cdrs)
            or (scope == "cdr_single" and len(target_cdrs) != 1)
            or (scope == "cdr_multi" and len(target_cdrs) < 2)):
        raise ValueError("Explicit targets must match the single/multiple CDR scope")
    if scope in SCOPES[1:] and is_edit.any():
        raise ValueError("CDR tasks belong only to the masking branch")
    available = [ids[row].unique().tolist() for row in range(bsz)]
    available = [[n for n in values if n] for values in available]
    eligible = torch.tensor([bool(values) for values in available])
    masking = ~is_edit.flatten()
    if scope is None:
        # Ineligible rows keep the existing chain tasks, without claiming a CDR example.
        chosen = (eligible & masking).nonzero().flatten()
        if len(chosen):
            probabilities = torch.tensor([config["mask_branch_ratios"][n] for n in SCOPES])
            requested[chosen] = torch.multinomial(probabilities, len(chosen), replacement=True,
                                                generator=generator)
    elif scope != "chain":
        if not eligible.all():
            raise ValueError("Explicit CDR task requires exact complete observed annotations")
        requested.fill_(SCOPES.index(scope))
    target = torch.zeros_like(ids, dtype=torch.bool)
    for row in requested.nonzero().flatten().tolist():
        values = available[row]
        if target_cdrs is not None:
            selected = [CDRS.index(name)+1 for name in target_cdrs]
            if not set(selected).issubset(values):
                raise ValueError("Requested CDR is absent from this antibody")
        else:
            count = 1 if requested[row] == 1 else int(torch.randint(2, len(values)+1, (), generator=generator))
            order = torch.randperm(len(values), generator=generator)[:count].tolist()
            selected = [values[i] for i in order]
        for group in selected:
            target[row] |= ids[row].eq(group)
    return target, requested, eligible
