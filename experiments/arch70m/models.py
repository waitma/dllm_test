"""Architecture-only V8 variants; python -B -m experiments.arch70m.tests."""
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainBlock, FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyModel
from dllm.pipelines.llada.models.modeling_llada import init_weights
from .common import VARIANTS

TAPER = (4608, 4464, 4032, 3408, 2736, 2112, 1680, 1536)

class HeadNorm(nn.Module):
    def __init__(self, width):
        super().__init__(); self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                * self.weight).to(x.dtype)

class AttentionBlock(FixedChainBlock):
    def __init__(self, original, variant):
        super().__init__(original.base); self.sdpa_backend = original.sdpa_backend
        self.variant = variant
        if variant in ('A1', 'A3'):
            self.q_head_norm = HeadNorm(self.base.config.d_model // self.base.config.n_heads)
            self.k_head_norm = HeadNorm(self.base.config.d_model // self.base.config.n_heads)
        if variant in ('A2', 'A3'):
            self.gate = nn.Linear(self.base.config.d_model, self.base.config.n_heads)
            nn.init.normal_(self.gate.weight, std=.02); nn.init.zeros_(self.gate.bias)

    def forward(self, hidden_states, attention_bias, position_ids):
        b = self.base; normalized = b.attn_norm(hidden_states)
        n, s, d = normalized.shape; h = b.config.n_heads; hd = d // h
        q = b.q_proj(normalized).view(n, s, h, hd).transpose(1, 2)
        k = b.k_proj(normalized).view(n, s, h, hd).transpose(1, 2)
        v = b.v_proj(normalized).view(n, s, h, hd).transpose(1, 2)
        if hasattr(self, 'q_head_norm'): q, k = self.q_head_norm(q), self.k_head_norm(k)
        q, k = b.rotary_emb(q, k, position_ids=position_ids)
        bias = b._cast_attn_bias(attention_bias, k.dtype)
        backends = [SDPBackend.MATH]
        if q.is_cuda and self.sdpa_backend in ('auto', 'efficient'):
            backends = [SDPBackend.EFFICIENT_ATTENTION]
            if self.sdpa_backend == 'auto': backends.append(SDPBackend.MATH)
        with sdpa_kernel(backends):
            attended = b._scaled_dot_product_attention(q, k, v, attn_mask=bias,
                dropout_p=b.config.attention_dropout if self.training else 0., is_causal=False)
        attended = attended.transpose(1, 2).contiguous()
        if hasattr(self, 'gate'): attended = attended * self.gate(normalized).sigmoid().unsqueeze(-1)
        elif self.variant == 'A7': attended = attended * .5
        hidden_states = hidden_states + b.dropout(b.attn_out(attended.reshape(n, s, d)) * self.attention_scale)
        normalized = b.ff_norm(hidden_states)
        return hidden_states + b.dropout(b.ff_out(b.act(b.ff_proj(normalized)) * b.up_proj(normalized)) * self.ffn_scale)

class ArchitectureModel(VFamilyModel):
    def mask_key(self, layer):
        return ('global',) if self.architecture == 'A5' else super().mask_key(layer)

    def layer_allowed(self, layer, chain_ids, position_ids, attention_mask):
        if self.architecture == 'A5':
            return attention_mask.bool()[:, None, :].expand(-1, chain_ids.shape[1], -1)
        return super().layer_allowed(layer, chain_ids, position_ids, attention_mask)

def build(settings, variant='A0', seed=42):
    """Construct the identical parent first; new parameters cannot shift its RNG."""
    if variant not in VARIANTS: raise ValueError(variant)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ArchitectureModel(FixedChainConfig(**settings['model']), family_spec=settings['family_prediction'])
        model.architecture = variant
        torch.manual_seed(seed + 700001)
        for i, block in enumerate(model.backbone.transformer.blocks):
            if variant in ('A1', 'A2', 'A3', 'A7'):
                model.backbone.transformer.blocks[i] = AttentionBlock(block, variant)
            elif variant == 'A4':
                for name in ('attention_scale', 'ffn_scale'):
                    value = getattr(block, name).clone(); delattr(block, name)
                    block.register_parameter(name, nn.Parameter(value))
            elif variant == 'A6':
                if model.config.n_layers != 8: raise ValueError('Taper requires eight layers')
                width = round(TAPER[i] * model.config.mlp_hidden_size / 3072)
                b = block.base; d = model.config.d_model
                for name, ni, no in (('ff_proj', d, width), ('up_proj', d, width), ('ff_out', width, d)):
                    layer = nn.Linear(ni, no, bias=getattr(b, name).bias is not None)
                    init_weights(b.config, layer, d=ni, layer_id=i if name == 'ff_out' else None)
                    setattr(b, name, layer)
        return model
