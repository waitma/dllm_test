"""Independent G/S/L combinations, sharing the existing attention implementation.

Run the CPU integer-input contracts after activating pllm:
  CUDA_VISIBLE_DEVICES='' USE_TF=0 python -B -m pytest -q /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_factorial_architecture.py

The three digits in fGSL select dynamic head gating, fixed residual scaling,
and alternating segment-local/global attention. These are experimental
mechanisms, not a reproduction of a CV paper or an effect-size claim.
"""
from dataclasses import asdict
import json
from pathlib import Path
import re

import torch
from torch import nn

from ..research.model import ResearchModel
from ..research_next.model import ArchitectureBlock
from ..v3.model import ImmuneConfig
from .model import RefinedModel


SCHEMA = 'immune_architecture.factorial.v1'
FACTOR_ORDER = ('G', 'S', 'L')
VARIANTS = {
    'f000': {},
    'f001': {'local_radius': 32},
    'f010': {'residual_scale': .1},
    'f011': {'residual_scale': .1, 'local_radius': 32},
    'f100': {'head_gate': True},
    'f101': {'head_gate': True, 'local_radius': 32},
    'f110': {'head_gate': True, 'residual_scale': .1},
    'f111': {'head_gate': True, 'residual_scale': .1, 'local_radius': 32},
    'cv_segment_global': {'segment_global': True},
    'cv_late_fusion': {'chain_late_fusion': True},
    'fixed_gate_control': {'fixed_head_gate': .5},
}
LEGACY_EQUIVALENTS = {
    'f000': 'r20_refined_control',
    'f100': 'r21_refined_head_gate',
    'f010': 'r24_refined_fixed_scale',
    'f001': 'r26_refined_local_global',
    'cv_late_fusion': 'r25_refined_late_fusion',
    'fixed_gate_control': 'r22_refined_fixed_gate',
}


class FactorialModel(RefinedModel):
    def __init__(self, config, variant='f000'):
        if variant not in VARIANTS:
            raise ValueError('Unknown factorial architecture')
        ResearchModel.__init__(self, config, 'r00_a4_control')
        self.variant = variant
        self.options = dict(VARIANTS[variant])
        self.backbone.transformer.blocks = nn.ModuleList([
            ArchitectureBlock(block, self.options)
            for block in self.backbone.transformer.blocks])
        self.configure_sdpa_backend()

    def mask_key(self, layer):
        if self.options.get('segment_global') and layer % 2 == 0:
            return ('segment',)
        return super().mask_key(layer)

    def layer_allowed(self, layer, chain_ids, position_ids, attention_mask):
        if not self.options.get('segment_global') or layer % 2:
            return super().layer_allowed(layer, chain_ids, position_ids, attention_mask)
        valid = attention_mask.bool()
        # Prefix groups remain separate from segments; padding never becomes a
        # key. Discarded padding queries attend valid keys to avoid empty rows.
        same = chain_ids[:, :, None] == chain_ids[:, None, :]
        return (same & valid[:, None, :]) | (~valid[:, :, None] & valid[:, None, :])

    def load_common_parameters(self, reference):
        """Copy the complete shared state; preserve only target-only head gates.

        Both sides must use the same wrapped block layout and configuration.
        Unexpected missing parameters are errors, not silently skipped weights.
        """
        if not isinstance(reference, RefinedModel) or asdict(reference.config) != asdict(self.config):
            raise ValueError('Common parameter reference configuration/layout differs')
        source, target = reference.state_dict(), self.state_dict()
        common = sorted(source.keys() & target.keys())
        only_source, only_target = sorted(source.keys() - target.keys()), sorted(target.keys() - source.keys())
        gate_name = r'backbone\.transformer\.blocks\.\d+\.gate\.weight'
        if any(re.fullmatch(gate_name, name) is None for name in only_source + only_target):
            raise ValueError('Unexpected common parameter mismatch')
        if any(source[name].shape != target[name].shape or source[name].dtype != target[name].dtype
               for name in common):
            raise ValueError('Common parameter shape/dtype differs')
        # Validate everything before the first mutation; strict loading retains
        # candidate-only gate initialization without changing fixed buffers.
        target.update({name: source[name] for name in common})
        self.load_state_dict(target, strict=True)
        return dict(copied=common, retained_target_only=only_target, ignored_source_only=only_source)

    def save_research_config(self, directory):
        (Path(directory) / 'research_config.json').write_text(json.dumps(dict(
            schema=SCHEMA, factor_order=list(FACTOR_ORDER), variant=self.variant,
            options=self.options, base=asdict(self.config)), indent=2) + '\n')

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory / 'research_config.json').read_text())
        variant = settings.get('variant')
        if (settings.get('schema') != SCHEMA or variant not in VARIANTS
                or settings.get('factor_order') != list(FACTOR_ORDER)
                or settings.get('options') != VARIANTS[variant]):
            raise ValueError('Factorial architecture schema/options differ')
        if settings['base'] != json.loads((directory / 'immune_config.json').read_text()):
            raise ValueError('Factorial base configuration differs')
        model = cls(ImmuneConfig(**settings['base']), variant)
        vocabulary = [model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]
        if json.loads((directory / 'vocabulary.json').read_text()) != vocabulary:
            raise ValueError('Vocabulary changed')
        model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
        return model
