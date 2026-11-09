"""CDR corruption for the current V10 variant; activate pllm and run test_v10_region_mlm.py.

Region annotations are training metadata only. Native cysteine targets remain
legal; only replacements in the whole-CDR mutation task exclude cysteine.
"""
import torch

from ..data.records import CHAIN_ROLE_TO_ID, REGION_ORDER, normalize_sequence
from .v9_bert import corrupt as mlm_corrupt

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
SPAN_POLICY = dict(
    mode_probabilities={'mlm': .70, 'span_mask': .15, 'span_mutation': .15},
    population='one_fully_observed_CDR1_2_3_on_H_L_VHH_TCRalpha_TCRbeta',
    annotation='all_seven_nonempty_regions_concatenate_exactly_to_sequence; cumulative_offsets; no_substring_guess',
    selection='uniform_eligible_chain_then_uniform_eligible_CDR; label_independent',
    missing_annotation='fallback_to_original_MLM',
    outside_span='clean_except_explicit_relations_always_MASK',
    mutation='AA20_excluding_C_and_original; every_selected_residue_changes; native_C_targets_retained',
    target='original_span_residues; nonbinding_relation_only; global_selected_token_mean',
    length='one_token_per_residue; fixed_length_no_sentinel',
)
MODE_NAMES = ('mlm', 'span_mask', 'span_mutation')
RECEPTOR_ROLES = frozenset(('antibody_heavy', 'antibody_light', 'nanobody_vhh', 'tcr_alpha', 'tcr_beta'))


def annotate_regions(row, record, tokenizer, *, enabled=True):
    """Map complete observed CDRs by original offsets, never observed-only offsets."""
    ids = [0] * len(row['input_ids'])
    if enabled:
        for chain in record.chains:
            if chain.role not in RECEPTOR_ROLES:
                continue
            parts = [normalize_sequence(chain.regions.get(name)) for name in REGION_ORDER]
            if not all(parts) or ''.join(parts) != chain.sequence:
                continue
            positions = {position: i for i, (role, position) in enumerate(zip(row['role_ids'], row['position_ids']))
                         if role == CHAIN_ROLE_TO_ID[chain.role]}
            cursor = 0
            for name, sequence in zip(REGION_ORDER, parts):
                end = cursor + len(sequence)
                if name.startswith('CDR') and all(aa in AMINO_ACIDS for aa in sequence):
                    mapped = [positions.get(j + 1) for j in range(cursor, end)]
                    expected = tokenizer.encode_residues(sequence)
                    if all(i is not None and row['residue_mask'][i] and row['attention_mask'][i]
                           and row['input_ids'][i] == token for i, token in zip(mapped, expected)):
                        for i in mapped:
                            ids[i] = int(name[-1])
                cursor = end
    row['cdr_region_ids'] = ids
    return row


def corrupt(batch, tokenizer, generator, *, mode=None):
    """Mix original MLM with one whole-CDR MASK/mutation; use only the supplied RNG.

    `mode` forces a branch for qualification, with the same annotation fallback.
    All returned tensors are [B,L], including metric-only mode/span metadata.
    """
    if mode is not None and mode not in MODE_NAMES:
        raise ValueError('Unknown region corruption mode: ' + str(mode))
    # Keep fallback and forced MLM exactly equivalent to the previous recipe.
    result = mlm_corrupt(batch, tokenizer, generator)
    clean = batch['input_ids']
    regions = batch.get('cdr_region_ids', torch.zeros_like(clean))
    result['corruption_mode'] = torch.zeros_like(clean)
    result['span_mask'] = torch.zeros_like(clean, dtype=torch.bool)
    legal = torch.tensor(tokenizer.encode_residues(AMINO_ACIDS.replace('C', '')), device=clean.device)
    lookup = torch.full((tokenizer.vocab_size,), -1, device=clean.device, dtype=torch.long)
    lookup[legal] = torch.arange(len(legal), device=clean.device)
    for row in range(clean.shape[0]):
        candidates = regions[row].gt(0) & batch['attention_mask'][row] & batch['residue_mask'][row]
        chains = batch['chain_ids'][row, candidates].unique(sorted=True)
        if not len(chains) or mode == 'mlm':
            continue
        if mode is None:
            draw = torch.rand((), generator=generator, device=clean.device).item()
            probabilities = SPAN_POLICY['mode_probabilities']
            chosen = (0 if draw < probabilities['mlm'] else
                      1 if draw < probabilities['mlm'] + probabilities['span_mask'] else 2)
        else:
            chosen = MODE_NAMES.index(mode)
        if chosen == 0:
            continue
        chain = chains[torch.randint(len(chains), (), generator=generator, device=clean.device)]
        chain_mask = candidates & batch['chain_ids'][row].eq(chain)
        available = regions[row, chain_mask].unique(sorted=True)
        region = available[torch.randint(len(available), (), generator=generator, device=clean.device)]
        selected = chain_mask & regions[row].eq(region)
        result['input_ids'][row] = clean[row]
        result['labels'][row].fill_(-100)
        result['noise_kind'][row].zero_()
        result['span_mask'][row] = selected
        result['corruption_mode'][row, batch['attention_mask'][row]] = chosen
        if chosen == 1:
            result['input_ids'][row, selected] = tokenizer.mask_token_id
        else:
            original = lookup[clean[row, selected]]
            replacement = torch.empty_like(original)
            # Native C has 19 possible replacements; all others have 18.
            for is_c in (False, True):
                mask = original.lt(0) if is_c else original.ge(0)
                values = torch.randint(len(legal) - (not is_c), (int(mask.sum()),),
                                       generator=generator, device=clean.device)
                if not is_c:
                    values += values.ge(original[mask]).long()
                replacement[mask] = legal[values]
            result['input_ids'][row, selected] = replacement
        result['noise_kind'][row, selected] = chosen
        targets = selected & batch['eligible_mask'][row]
        result['labels'][row, targets] = clean[row, targets]
        relation = batch['explicit_relation_mask'][row]
        result['input_ids'][row, relation] = tokenizer.mask_token_id
        result['labels'][row, relation] = clean[row, relation]
        result['noise_kind'][row, relation] = 1
    return result
