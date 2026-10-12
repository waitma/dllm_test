"""Fixed-capacity V5 object layout; run pytest scripts/tests/immune_llada/test_v5_canvas.py.

The target object has no reference-derived feature boundaries. Prepared synthetic
scaffolds retain their coordinates, but their residues never become labels.
"""
from dataclasses import replace

from ..data.records import BioSeqChain, BioSeqRecord, CHAIN_ROLE_TO_ID


def canvas_row(collator, item):
    from .layout import GenerationRequest, RECEPTORS, ordered_chains
    tok = collator.tokenizer
    generating = collator.mode == 'generation'
    if isinstance(item, GenerationRequest):
        if not generating:
            raise ValueError('GenerationRequest requires generation mode')
        target_roles = {role for role, _ in item.targets}
        chains = list(item.conditions) + [BioSeqChain('X', role) for role, _ in item.targets]
        roles = {c.role for c in chains}
        task = ('tcr' if any(r.startswith('tcr_') for r in roles) else
                'nanobody_antigen' if 'nanobody_vhh' in roles and 'antigen' in roles else
                'nanobody' if 'nanobody_vhh' in roles else
                'antibody_antigen' if 'antigen' in roles else 'antibody')
        record = BioSeqRecord(chains, task, 'generation_request')
    else:
        record = item
        target_roles = collator.target_roles & set(record.chain_roles)
    if not target_roles:
        raise ValueError('V5 canvas needs at least one target receptor role')
    # Replace hidden targets BEFORE rendering: no target sequence/metadata/length
    # may influence context features, budget validation or generation capacity.
    if generating:
        record = replace(record, chains=[BioSeqChain('X', c.role) if c.role in target_roles else c
                                        for c in record.chains])
    receptors = [c for c in ordered_chains(record) if c.role in RECEPTORS]
    roles = [c.role for c in receptors]
    supported = (['antibody_heavy'], ['antibody_light'], ['antibody_heavy', 'antibody_light'],
                 ['tcr_beta'], ['tcr_alpha'], ['tcr_beta', 'tcr_alpha'], ['nanobody_vhh'])
    if roles not in supported:
        raise ValueError(f'Unsupported V5 receptor roles/order: {roles}')
    first_target = next(i for i, c in enumerate(receptors) if c.role in target_roles)
    if any(c.role not in target_roles for c in receptors[first_target:]):
        raise ValueError('V5 variable-length generation supports whole objects or a target suffix; '
                         'a known suffix needs a separate conditioning protocol')
    rendered = collator._legacy([record])
    starts = (rendered['input_ids'][0] == tok.special_id('<prots>')).nonzero().flatten()
    prefix_length = int(starts[-1])
    row = {key: tensor[0, :prefix_length].long().tolist() for key, tensor in rendered.items()}
    # Context objects use explicit <prots>...<protd> boundaries, including
    # their markers. Internal dots do not create attention or RoPE boundaries.
    object_id, active, inner = -1, False, 0
    for index, token in enumerate(row['input_ids']):
        if token == tok.special_id('<prots>'):
            object_id += 1
            active, inner = True, 0
        row['chain_ids'][index] = object_id if active else -1
        row['position_ids'][index] = inner if active else 0
        if active:
            inner += 1
        if token == tok.special_id('<protd>'):
            active = False
    uid = object_id + 1
    marker = '<tcr>' if roles[0].startswith('tcr_') else '<nb>' if roles[0]=='nanobody_vhh' else '<ab>'

    def add(token, *, chain=-1, pos=0, slot=False, label=-100, residue=False, terminal=False):
        values = dict(input_ids=token, labels=label, attention_mask=1, role_ids=0,
                      scope_ids=0, chain_ids=chain, position_ids=pos,
                      target_slot_mask=slot, eligible_mask=label != -100,
                      residue_mask=residue, terminal_mask=terminal)
        for key in row:
            row[key].append(int(values[key]))

    # Object type is an explicit condition, not a generated answer boundary.
    add(tok.special_id('<prots>'), chain=uid, pos=0)
    add(tok.special_id(marker), chain=uid, pos=1)
    # Single alpha vs beta is a declared task identity, never a length boundary.
    row['role_ids'][-1] = CHAIN_ROLE_TO_ID[roles[0]]
    content = []
    known_prefix = 0
    if not generating or first_target:
        for index, chain in enumerate(receptors):
            if generating and index == first_target:
                break
            if index:
                content.append((tok.chain_separator_id(), False, False, index > first_target))
            synthetic = chain.metadata.get('synthetic_residue_mask', [0]*len(chain.sequence))
            for token, flag in zip(tok.encode_residues(chain.sequence), synthetic):
                content.append((token, bool(flag), True, index >= first_target))
        if generating:
            # The known prefix includes its trailing separator; its location is
            # declared context, while the end of the generated suffix is unknown.
            content.append((tok.chain_separator_id(), False, False, False))
            known_prefix = len(content)
        else:
            content.append((tok.special_id('<protd>'), False, False, True))
    if len(content) > collator.full_capacity or (generating and known_prefix >= collator.full_capacity):
        raise ValueError(f'object_capacity_overflow:{len(content)}>{collator.full_capacity}')
    # The entire unknown object, including its attended tail, stays one segment.
    # Never derive its feature boundary from the reference/predicted terminal.
    for j in range(collator.full_capacity):
        if j < len(content):
            token, synthetic, residue, supervised = content[j]
            label = token if collator.mode == 'train' and supervised and not synthetic else -100
            add(tok.mask_token_id if synthetic else token, chain=uid, pos=j+2, slot=True,
                label=label, residue=residue and not synthetic,
                terminal=token == tok.special_id('<protd>'))
        else:
            add(tok.mask_token_id, chain=uid, pos=j+2, slot=True)
    if len(row['input_ids']) > collator.max_length:
        raise ValueError(f'sequence_budget_overflow:{len(row["input_ids"])}>{collator.max_length}')
    return row
