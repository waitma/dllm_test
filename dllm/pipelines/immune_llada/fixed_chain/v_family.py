"""V-family auxiliary classification for V8; activate pllm and run its pytest.

Source labels are targets only. Negatives have relation-token and family loss,
with no sequence MLM targets and no corruption of sequence/context tokens.
The species-qualified vocabulary must be fitted on training records and frozen.
"""
from dataclasses import asdict
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from ..data.records import BioSeqRecord, CHAIN_ROLE_TO_ID
from .model import FixedChainConfig
from .mlm import selected_loss
from .unbinding_vhh import VHHUnbindingCollator, VHHUnbindingModel, corrupt, RECIPE as BASE_RECIPE

OBJECTIVE = 'binding_mlm_nonbinding_relation_only_vhh_v_family'
SCHEMA = 'immune.v8.v_family.v3'
RECIPE = dict(BASE_RECIPE, auxiliary='species_qualified_v_family',
              nonbinding='token_loss_relation_only; sequence_context_clean; family_auxiliary_enabled',
              auxiliary_population='all_observed_receptor_chains; missing_filtered_unseen_annotations_are_unknown',
              auxiliary_loss='global_eligible_chain_mean', pooling='observed_residue_mean_per_chain')
ROLES = ('antibody_heavy', 'antibody_light', 'tcr_beta', 'tcr_alpha')


def role_mask(batch, role):
    mask = batch['role_ids'].eq(CHAIN_ROLE_TO_ID[role])
    if role == 'antibody_heavy':
        mask |= batch['role_ids'].eq(CHAIN_ROLE_TO_ID['nanobody_vhh'])
    return mask & batch['residue_mask'] & batch['attention_mask']
SPECIES = {'human': 'human', 'homo sapiens': 'human', 'mouse': 'mouse',
           'mouse_balb/c': 'mouse', 'mouse_c57bl/6': 'mouse', 'mus musculus': 'mouse',
           'rat_sd': 'rat', 'rat': 'rat', 'rattus norvegicus': 'rat'}
PREFIXES = {'antibody_heavy': ('IGHV',), 'antibody_light': ('IGKV', 'IGLV'),
            'tcr_beta': ('TRBV',), 'tcr_alpha': ('TRAV',)}


def family_targets(record):
    """Return four optional class names and exclusion reasons; never infer genes."""
    record = BioSeqRecord.from_prepared_dict(record) if isinstance(record, dict) else record
    targets, reasons = [None] * 4, []
    species = SPECIES.get(str(record.metadata.get('species', '')).lower().strip())
    for slot, role in enumerate(ROLES):
        chains = [c for c in record.chains if c.role == role or
                  (role == 'antibody_heavy' and c.role == 'nanobody_vhh')]
        if not chains:
            continue
        reason = None
        if len(chains) != 1:
            reason = 'duplicate_role'
        elif chains[0].role == 'nanobody_vhh' or record.source not in ('oas_paired', 'ots_paired'):
            reason = 'unsupported_source'
        elif not species:
            reason = 'unknown_species'
        else:
            chain = chains[0]; meta = chain.metadata
            if any(meta.get('synthetic_residue_mask', [])) or 'X' in chain.sequence:
                reason = 'synthetic_or_incomplete'
            calls = {str(v).strip() for k, v in meta.items() if v and
                     (k.endswith('_v_call') or k.endswith('_v') or k == 'v_gene')}
            if not reason and len(calls) != 1:
                reason = 'missing_or_conflicting_call'
            if not reason:
                call = next(iter(calls))
                match = re.fullmatch(r'(IGHV|IGKV|IGLV|TRAV|TRBV)([0-9]+)[A-Z0-9/\-]*(?:\*[A-Z0-9]+)?', call)
                if not match:
                    reason = 'ambiguous_or_invalid_call'
                elif match[1] not in PREFIXES[role]:
                    reason = 'locus_role_conflict'
                else:
                    allowed = {'antibody_heavy': {'H'}, 'antibody_light': {'K', 'L'},
                               'tcr_beta': {'B'}, 'tcr_alpha': {'A'}}[role]
                    types = {str(v).upper() for k, v in meta.items() if k.endswith('anarci_type') and v}
                    if types and not types.issubset(allowed):
                        reason = 'anarci_role_conflict'
                    elif role == 'antibody_light' and meta.get('l_locus') and str(meta['l_locus']).upper() != match[1][2]:
                        reason = 'light_locus_conflict'
                    else:
                        targets[slot] = species + ':' + match[1] + match[2]
        if reason:
            reasons.append(role + ':' + reason)
            if len(chains) == 1:
                targets[slot] = 'unknown'
        if len(chains) == 1:
            synthetic = chains[0].metadata.get('synthetic_residue_mask', [])
            if synthetic and all(synthetic):
                targets[slot] = None
    return targets, reasons


def validate_spec(spec):
    if (not isinstance(spec, dict) or spec.get('schema') != SCHEMA
            or not spec.get('classes') or spec['classes'] != sorted(set(spec['classes']))
            or 'unknown' not in spec['classes']
            or not 0 < spec.get('loss_weight', 0) <= 1
            or any(c != 'unknown' and not re.fullmatch(r'(human|mouse|rat):(IGHV|IGKV|IGLV|TRAV|TRBV)[0-9]+', c)
                   for c in spec['classes'])):
        raise ValueError('Invalid frozen V-family specification')


class VFamilyCollator(VHHUnbindingCollator):
    def __init__(self, config=None, *, family_spec, hide_relation=False):
        super().__init__(config, hide_relation=hide_relation)
        validate_spec(family_spec)
        self.class_ids = {c: i for i, c in enumerate(family_spec['classes'])}

    def __call__(self, records):
        records = list(records)
        batch = super().__call__(records)
        batch['family_labels'] = torch.tensor([
            [-100 if c is None else self.class_ids.get(c, self.class_ids['unknown'])
             for c in family_targets(r)[0]] for r in records], dtype=torch.long)
        for slot, role in enumerate(ROLES):
            batch['family_labels'][~role_mask(batch, role).any(1), slot] = -100
        # Labels are never forwarded through feature_args into the encoder.
        return batch


class VFamilyModel(VHHUnbindingModel):
    def __init__(self, config, *, family_spec):
        super().__init__(config)
        validate_spec(family_spec)
        self.family_spec = json.loads(json.dumps(family_spec))
        self.family_loss_weight = family_spec['loss_weight']
        self.family_head = nn.Linear(config.d_model, len(family_spec['classes']))

    def forward(self, **batch):
        logits, hidden = self.denoise(**self.feature_args(batch), output_hidden_states=True)
        vectors = []
        present = []
        for role in ROLES:
            mask = role_mask(batch, role)
            vectors.append((hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1))
            present.append(mask.any(1))
        family_logits = self.family_head(torch.stack(vectors, dim=1))
        labels = batch['family_labels']
        if (labels.ne(-100) & ~torch.stack(present, dim=1)).any():
            raise ValueError('Family label without observed chain residues')
        selected = labels.ne(-100)
        family_loss = (F.cross_entropy(family_logits[selected].float(), labels[selected], reduction='sum')
                       if selected.any() else family_logits.sum() * 0)
        return dict(loss_sum=selected_loss(logits, batch['labels']), logits=logits,
                    family_loss_sum=family_loss, family_logits=family_logits,
                    family_correct=(family_logits.argmax(-1)[selected] == labels[selected]).sum())

    def save_research_config(self, directory):
        settings = dict(schema=SCHEMA, variant=OBJECTIVE, options=self.options,
                        base=asdict(self.config), recipe=RECIPE, family_spec=self.family_spec)
        (Path(directory) / 'research_config.json').write_text(json.dumps(settings, indent=2) + '\n')

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory / 'research_config.json').read_text())
        if (settings.get('schema') != SCHEMA or settings.get('variant') != OBJECTIVE
                or settings.get('recipe') != RECIPE
                or settings.get('options') != {'residual_scale': .1, 'segment_global': True}
                or settings['base'] != json.loads((directory / 'immune_config.json').read_text())):
            raise ValueError('V-family checkpoint contract changed')
        model = cls(FixedChainConfig(**settings['base']), family_spec=settings['family_spec'])
        if json.loads((directory / 'vocabulary.json').read_text()) != [model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError('V-family token vocabulary changed')
        model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
        return model
