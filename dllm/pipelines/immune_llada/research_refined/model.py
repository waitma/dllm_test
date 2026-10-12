"""Reuse per-topology attention biases on the existing LLaDA270 backbone.

After activating pllm: python -m pytest scripts/tests/immune_llada/test_research_refined.py -q
The 768/8/12/3072 backbone is unchanged; compact protein vocabulary removes unused
text embedding/head parameters. Each release has distinct experiment identifiers.
"""
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path

import torch

from ..research_next.model import ArchitectureModel, VARIANTS as PREVIOUS_VARIANTS
from ..v3.model import ImmuneConfig


BASE_VARIANTS = {
    'r20_refined_control': 'r12_architecture_control',
    'r21_refined_head_gate': 'r13_head_output_gate',
    'r22_refined_fixed_gate': 'r14_fixed_half_attention',
    'r23_refined_layerscale': 'r15_layerscale',
    'r24_refined_fixed_scale': 'r16_fixed_residual_scale',
    'r25_refined_late_fusion': 'r17_chain_late_fusion',
    'r26_refined_local_global': 'r18_chain_local_global',
    'r27_refined_value_residual': 'r19_value_residual',
}
VARIANTS = {name: dict(PREVIOUS_VARIANTS[base]) for name, base in BASE_VARIANTS.items()}


class RefinedModel(ArchitectureModel):
    def __init__(self, config, variant='r20_refined_control'):
        if variant not in BASE_VARIANTS:
            raise ValueError('Unknown refined architecture')
        super().__init__(config, BASE_VARIANTS[variant])
        self.variant = variant
        self.configure_sdpa_backend()

    def configure_sdpa_backend(self):
        """Runtime-only policy; checkpoint tensors and architecture stay unchanged."""
        backend = os.environ.get('IMMUNE_SDPA_BACKEND', 'auto')
        if backend not in ('auto', 'math', 'efficient'):
            raise ValueError('IMMUNE_SDPA_BACKEND must be auto, math, or efficient')
        self.sdpa_backend = backend
        for block in self.backbone.transformer.blocks:
            block.sdpa_backend = backend
        logging.getLogger(__name__).info(
            'Refined SDPA policy=%s (auto: CUDA efficient with math fallback; CPU: math)', backend)

    def mask_key(self, layer):
        if 'local_radius' in self.options and layer % 2 == 0:
            return ('local', self.options['local_radius'])
        if self.options.get('chain_late_fusion') and layer < self.config.n_layers//2:
            return ('chain',)
        return ('global',)

    def denoise(self, input_ids, attention_mask, role_ids, scope_ids, chain_ids, position_ids,
                target_slot_mask, *, output_hidden_states=False):
        """Cache within one forward only; all checkpoint inputs remain explicit."""
        valid = attention_mask.bool()
        if not valid.any(1).all():
            raise ValueError('Each record needs at least one attended token')
        globally_unmasked = bool(valid.all())
        x = (self.backbone.transformer.wte(input_ids)+self.role_embeddings(role_ids)+
             self.scope_embeddings(scope_ids))
        x = self.backbone.transformer.emb_drop(x)
        same = ((chain_ids[:, :, None] == chain_ids[:, None, :]) &
                (chain_ids[:, :, None] >= 0) & (chain_ids[:, None, :] >= 0))
        # Repeated advanced indices make CUDA backward reduce B*S*S entries
        # into only two table rows. Broadcast selection keeps the same learned
        # bias while using an efficient reduction instead of IndexBackward.
        base_bias = torch.where(
            same[:, None], self.chain_bias[1][None, :, None, None],
            self.chain_bias[0][None, :, None, None])
        biases, first_values = {}, None
        for index, block in enumerate(self.backbone.transformer.blocks):
            key = self.mask_key(index)
            if key not in biases:
                if key == ('global',) and globally_unmasked:
                    biases[key] = base_bias
                else:
                    allowed = self.layer_allowed(index, chain_ids, position_ids, attention_mask)
                    biases[key] = base_bias.masked_fill(~allowed[:, None], torch.finfo(base_bias.dtype).min)
            bias = biases[key]
            if self.backbone.activation_checkpointing_strategy is not None and self.training:
                x, values = self.backbone._activation_checkpoint_fn(block, x, bias, position_ids, first_values)
            else:
                x, values = block(x, bias, position_ids, first_values)
            if index == 0 and self.options.get('value_residual'):
                first_values = values
        x = self.backbone.transformer.ln_f(x)
        logits = self.backbone.transformer.ff_out(x)
        logits = logits.masked_fill(target_slot_mask[:, :, None] & ~self.allowed_target[None, None], float('-inf'))
        return logits, x if output_hidden_states else None

    def save_research_config(self, directory):
        (Path(directory)/'research_config.json').write_text(json.dumps(dict(
            variant=self.variant, parent_variant=BASE_VARIANTS[self.variant], options=self.options,
            base=asdict(self.config), schema='immune_architecture.refined.v1'), indent=2)+'\n')

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory/'research_config.json').read_text())
        variant = settings['variant']
        if (settings['schema'] != 'immune_architecture.refined.v1' or variant not in VARIANTS
                or settings['parent_variant'] != BASE_VARIANTS[variant] or settings['options'] != VARIANTS[variant]):
            raise ValueError('Refined architecture checkpoint schema/options differ')
        model = cls(ImmuneConfig(**settings['base']), variant)
        if json.loads((directory/'vocabulary.json').read_text()) != [model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError('Vocabulary changed')
        model.load_state_dict(load_file(str(directory/'model.safetensors')), strict=True)
        return model
