"""Extend the V8 relation-only negative objective to explicit single-chain VHH.

Activate pllm; run USE_TF=0 python -B -m pytest
/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/scripts/tests/immune_llada/test_unbinding_vhh.py -q
Existing AB/TCR collation is unchanged. VHH uses its own role and <nb> marker,
the first-chain capacity, and no fabricated light chain. Old release hashes stay valid.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from ..data.records import BioSeqRecord, CHAIN_ROLE_TO_ID
from .model import FixedChainConfig, FixedChainModel
from .unbinding_mlm import UnbindingMLMCollator, UnbindingMLMModel, corrupt
from .unbinding_mlm import RECIPE as BASE_RECIPE

SCHEMA = 'immune.fixed_chain.unbinding_vhh_mlm.v1'
OBJECTIVE = 'binding_mlm_nonbinding_relation_only_vhh'
RECIPE = dict(BASE_RECIPE, supervision=OBJECTIVE,
              receptor_scope='antibody_H_L; TCR_alpha_beta; single_nanobody_vhh',
              vhh_layout='first_chain_capacity; role_nanobody_vhh; marker_nb; no_light_chain')


class VHHUnbindingCollator(UnbindingMLMCollator):
    """Reuse the existing slot renderer, then restore VHH type before masking."""

    def row(self, item):
        record = BioSeqRecord.from_prepared_dict(item) if isinstance(item, dict) else item
        if 'nanobody_vhh' not in record.chain_roles:
            return super().row(item)
        if (record.chain_roles.count('nanobody_vhh') != 1
                or any(role not in ('antigen', 'nanobody_vhh') for role in record.chain_roles)
                or record.chain_roles.count('antigen') > 1):
            raise ValueError('VHH requires one VHH and at most one antigen; no light chain')
        # This private view only reuses slot placement; neither the record nor
        # the encoder input is reclassified as a conventional antibody.
        proxy = replace(record, task_type='antibody_antigen' if 'antigen' in record.chain_roles else 'antibody',
            chains=[replace(c, role='antibody_heavy') if c.role == 'nanobody_vhh' else c
                    for c in record.chains])
        row = super().row(proxy)
        row['role_ids'] = [CHAIN_ROLE_TO_ID['nanobody_vhh'] if v == CHAIN_ROLE_TO_ID['antibody_heavy'] else v
                           for v in row['role_ids']]
        old, new = self.tokenizer.special_id('<ab>'), self.tokenizer.special_id('<nb>')
        row['input_ids'] = [new if v == old else v for v in row['input_ids']]
        row['labels'] = [new if v == old else v for v in row['labels']]
        return row


class VHHUnbindingModel(UnbindingMLMModel):
    """Same parameter shapes as BERT287000, with explicit VHH target validation."""

    @staticmethod
    def _validate_target_features(role_ids, chain_ids, position_ids, target_slot_mask):
        check_roles = role_ids.masked_fill(role_ids.eq(CHAIN_ROLE_TO_ID['nanobody_vhh']),
                                          CHAIN_ROLE_TO_ID['antibody_heavy'])
        FixedChainModel._validate_target_features(check_roles, chain_ids, position_ids, target_slot_mask)

    def receptor_vectors(self, batch):
        """Keep the four-slot probe interface; pool H/VHH, L, beta, alpha."""
        _, hidden = self.denoise(**self.feature_args(batch), output_hidden_states=True)
        pooled = []
        for roles in (('antibody_heavy', 'nanobody_vhh'), ('antibody_light',), ('tcr_beta',), ('tcr_alpha',)):
            role_mask = torch.zeros_like(batch['residue_mask'])
            for role in roles:
                role_mask |= batch['role_ids'].eq(CHAIN_ROLE_TO_ID[role])
            mask = role_mask & batch['residue_mask'] & batch['attention_mask']
            pooled.append((hidden.float() * mask.unsqueeze(-1)).sum(1)
                          / mask.sum(1, keepdim=True).clamp_min(1))
        return torch.cat(pooled, dim=-1)

    def save_research_config(self, directory):
        settings = dict(schema=SCHEMA, variant=OBJECTIVE, options=self.options,
                        base=asdict(self.config), recipe=RECIPE)
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
            raise ValueError('VHH checkpoint/recipe contract changed')
        model = cls(FixedChainConfig(**settings['base']))
        if json.loads((directory / 'vocabulary.json').read_text()) != [
                model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError('VHH vocabulary changed')
        model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
        return model
