"""A5 + head RMS normalization / unit-initialized gates: python -B -m architecture_attention.tests."""
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainBlock
from experiments.arch70m.models import build as original_build, HeadNorm

FACTORS = {'C0': (False, False), 'C1': (True, False),
           'C2': (False, True), 'C3': (True, True)}


class FactorBlock(FixedChainBlock):
    """Keep the parent math, adding normalization and/or a gate before W_o."""
    def __init__(self, original, normalize, gate):
        super().__init__(original.base)
        self.sdpa_backend = original.sdpa_backend
        width = self.base.config.d_model
        if normalize:
            self.q_head_norm = HeadNorm(width // self.base.config.n_heads)
            self.k_head_norm = HeadNorm(width // self.base.config.n_heads)
        if gate:
            self.output_gate = nn.Linear(width, self.base.config.n_heads)
            # 2*sigmoid(0) == 1: do not halve the initial attention residual.
            nn.init.zeros_(self.output_gate.weight)
            nn.init.zeros_(self.output_gate.bias)

    def forward(self, hidden_states, attention_bias, position_ids):
        b = self.base
        normalized = b.attn_norm(hidden_states)
        query, key, value = b.q_proj(normalized), b.k_proj(normalized), b.v_proj(normalized)
        dtype = key.dtype
        if b.q_norm is not None and b.k_norm is not None:
            query, key = b.q_norm(query).to(dtype), b.k_norm(key).to(dtype)
        n, length, width = query.shape
        heads, kv_heads = b.config.n_heads, b.config.effective_n_kv_heads
        hd = width // heads
        query = query.view(n, length, heads, hd).transpose(1, 2)
        key = key.view(n, length, kv_heads, hd).transpose(1, 2)
        value = value.view(n, length, kv_heads, hd).transpose(1, 2)
        if hasattr(self, 'q_head_norm'):
            query, key = self.q_head_norm(query), self.k_head_norm(key)
        query, key = b.rotary_emb(query, key, position_ids=position_ids)
        bias = b._cast_attn_bias(attention_bias, dtype)
        backends = [SDPBackend.MATH]
        if query.is_cuda and self.sdpa_backend in ('auto', 'efficient'):
            backends = [SDPBackend.EFFICIENT_ATTENTION]
            if self.sdpa_backend == 'auto': backends.append(SDPBackend.MATH)
        with sdpa_kernel(backends):
            attended = b._scaled_dot_product_attention(query, key, value, attn_mask=bias,
                dropout_p=b.config.attention_dropout if self.training else 0., is_causal=False)
        attended = attended.transpose(1, 2).contiguous()
        if hasattr(self, 'output_gate'):
            attended = attended * (2 * self.output_gate(normalized).sigmoid()).unsqueeze(-1)
        attended = b.attn_out(attended.reshape(n, length, width))
        hidden_states = hidden_states + b.dropout(attended * self.attention_scale)
        normalized = b.ff_norm(hidden_states)
        feedforward = b.ff_out(b.act(b.ff_proj(normalized)) * b.up_proj(normalized))
        return hidden_states + b.dropout(feedforward * self.ffn_scale)


def build(settings, variant='C0', seed=43):
    if variant == 'A0': return original_build(settings, 'A0', seed)
    if variant not in FACTORS: raise ValueError('Unknown factorial variant: ' + variant)
    model = original_build(settings, 'A5', seed)
    normalize, gate = FACTORS[variant]
    if normalize or gate:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 700001)
            for i, block in enumerate(model.backbone.transformer.blocks):
                model.backbone.transformer.blocks[i] = FactorBlock(block, normalize, gate)
    model.factorial_variant = variant
    return model
