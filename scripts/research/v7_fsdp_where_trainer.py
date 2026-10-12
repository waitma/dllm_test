"""Use the verified where expansion with the existing FSDP adapter.

Activate conda pllm, then use the independent run_v7_fsdp_where.py entrypoint.
This changes one bias expansion expression; parameter names and metrics stay
compatible with the original adapter. No SDPA backend flags are changed here.
"""
import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
BASE_ADAPTER = ROOT/'scripts/research/v7_fsdp_trainer.py'
spec = importlib.util.spec_from_file_location('_v7_where_original_adapter', BASE_ADAPTER)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
FSDPTrainer = base.FSDPTrainer
BoundedTrace = base.BoundedTrace


class ForwardMetricModel(base.ForwardMetricModel):
    """Frozen RefinedModel denoise with only the chain-bias expansion replaced."""

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
        base_bias = torch.where(same[:, None], self.chain_bias[1][None, :, None, None],
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
