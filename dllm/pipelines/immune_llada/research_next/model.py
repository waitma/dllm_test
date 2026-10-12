"""Architecture mechanisms with explicit controls and checkpoint-safe chain masks.

Run: python -m pytest scripts/tests/immune_llada/test_research_next.py -q
These adaptations are not full reproductions of the cited image/language models.
"""
from dataclasses import asdict
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..research.model import ResearchModel
from ..v3.model import ImmuneConfig


VARIANTS = {
    'r12_architecture_control': {},
    'r13_head_output_gate': {'head_gate': True},
    'r14_fixed_half_attention': {'fixed_head_gate': .5},
    'r15_layerscale': {'residual_scale': .1, 'learn_scale': True},
    'r16_fixed_residual_scale': {'residual_scale': .1},
    'r17_chain_late_fusion': {'chain_late_fusion': True},
    'r18_chain_local_global': {'local_radius': 32},
    'r19_value_residual': {'value_residual': True},
}


class ArchitectureBlock(nn.Module):
    def __init__(self, base, options):
        super().__init__()
        self.base = base
        self.options = dict(options)
        config = base.config
        self.gate = nn.Linear(config.d_model, config.n_heads, bias=False) if options.get('head_gate') else None
        if self.gate is not None:
            nn.init.normal_(self.gate.weight, std=.02)
        scale = options.get('residual_scale', 1.)
        for name in ('attention_scale', 'ffn_scale'):
            value = torch.full((config.d_model,), scale)
            if options.get('learn_scale'):
                self.register_parameter(name, nn.Parameter(value))
            else:
                self.register_buffer(name, value, persistent=False)

    def set_activation_checkpointing(self, strategy):
        self.base.set_activation_checkpointing(strategy)

    def forward(self, x, attention_bias, position_ids, first_values=None):
        """All forward-specific context is explicit, including the first-layer V."""
        block = self.base
        normalized = block.attn_norm(x)
        q, k, v = block.q_proj(normalized), block.k_proj(normalized), block.v_proj(normalized)
        original_v = v
        if self.options.get('value_residual') and first_values is not None:
            v = .5*(v+first_values)
        dtype = k.dtype
        if block.q_norm is not None and block.k_norm is not None:
            q, k = block.q_norm(q).to(dtype), block.k_norm(k).to(dtype)
        batch, length, width = q.shape
        heads = block.config.n_heads
        head_width = width//heads
        q = q.view(batch, length, heads, head_width).transpose(1, 2)
        k = k.view(batch, length, block.config.effective_n_kv_heads, head_width).transpose(1, 2)
        v = v.view(batch, length, block.config.effective_n_kv_heads, head_width).transpose(1, 2)
        if block.config.rope:
            q, k = block.rotary_emb(q, k, position_ids=position_ids)
        attention_bias = block._cast_attn_bias(attention_bias, dtype)
        # Scope the policy inside the block so checkpoint recomputation uses it
        # too, and restore the caller's global flags immediately afterwards.
        policy = getattr(self, 'sdpa_backend', None)
        backends = [SDPBackend.MATH]
        if q.is_cuda and policy in ('auto', 'efficient'):
            backends = [SDPBackend.EFFICIENT_ATTENTION]
            if policy == 'auto':
                backends.append(SDPBackend.MATH)
        context = sdpa_kernel(backends) if policy is not None else nullcontext()
        with context:
            attended = block._scaled_dot_product_attention(q, k, v, attn_mask=attention_bias,
                dropout_p=block.config.attention_dropout if self.training else 0., is_causal=False)
        attended = attended.transpose(1, 2).contiguous()
        if self.gate is not None:
            attended = attended*torch.sigmoid(self.gate(normalized)).unsqueeze(-1)
        elif 'fixed_head_gate' in self.options:
            attended = attended*self.options['fixed_head_gate']
        attended = block.attn_out(attended.reshape(batch, length, width))
        x = x+block.dropout(attended*self.attention_scale)
        normalized = block.ff_norm(x)
        feedforward = block.ff_out(block.act(block.ff_proj(normalized))*block.up_proj(normalized))
        return x+block.dropout(feedforward*self.ffn_scale), original_v


class ArchitectureModel(ResearchModel):
    def __init__(self, config, variant='r12_architecture_control'):
        if variant not in VARIANTS:
            raise ValueError('Unknown expanded architecture')
        super().__init__(config, 'r00_a4_control')
        self.variant = variant
        self.options = dict(VARIANTS[variant])
        self.backbone.transformer.blocks = nn.ModuleList([
            ArchitectureBlock(block, self.options) for block in self.backbone.transformer.blocks])

    def layer_allowed(self, layer, chain_ids, position_ids, attention_mask):
        valid = attention_mask.bool()
        length = chain_ids.shape[1]
        allowed = valid[:, None, :].expand(-1, length, -1)
        local = 'local_radius' in self.options and layer % 2 == 0
        isolated = self.options.get('chain_late_fusion') and layer < self.config.n_layers//2
        if local or isolated:
            # Prefix tokens stay in their own group. They cannot relay information
            # between two chains during the layers described as chain-isolated.
            same = chain_ids[:, :, None] == chain_ids[:, None, :]
            allowed = allowed & same
            if local:
                distance = (position_ids[:, :, None]-position_ids[:, None, :]).abs()
                allowed = allowed & ((distance <= self.options['local_radius']) |
                                     (chain_ids[:, :, None] < 0))
            # Padding queries are discarded; give them valid keys to avoid an
            # all-masked softmax. Padding keys remain inaccessible to valid queries.
            allowed = allowed | (~valid[:, :, None] & valid[:, None, :])
        return allowed

    def denoise(self, input_ids, attention_mask, role_ids, scope_ids, chain_ids, position_ids,
                target_slot_mask, *, output_hidden_states=False):
        if not attention_mask.bool().any(1).all():
            raise ValueError('Each record needs at least one attended token')
        x = (self.backbone.transformer.wte(input_ids)+self.role_embeddings(role_ids)+
             self.scope_embeddings(scope_ids))
        x = self.backbone.transformer.emb_drop(x)
        same = ((chain_ids[:, :, None] == chain_ids[:, None, :]) &
                (chain_ids[:, :, None] >= 0) & (chain_ids[:, None, :] >= 0))
        base_bias = self.chain_bias[same.long()].permute(0, 3, 1, 2).contiguous()
        first_values = None
        for index, block in enumerate(self.backbone.transformer.blocks):
            allowed = self.layer_allowed(index, chain_ids, position_ids, attention_mask)
            bias = base_bias.masked_fill(~allowed[:, None], torch.finfo(base_bias.dtype).min)
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
        (Path(directory)/'research_config.json').write_text(json.dumps(dict(variant=self.variant,
            options=self.options, base=asdict(self.config), schema='immune_architecture.v2'), indent=2)+'\n')

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory/'research_config.json').read_text())
        if settings['schema'] != 'immune_architecture.v2' or settings['options'] != VARIANTS[settings['variant']]:
            raise ValueError('Architecture checkpoint schema/options differ')
        model = cls(ImmuneConfig(**settings['base']), settings['variant'])
        if json.loads((directory/'vocabulary.json').read_text()) != [model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError('Vocabulary changed')
        model.load_state_dict(load_file(str(directory/'model.safetensors')), strict=True)
        return model
