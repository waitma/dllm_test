"""Round-two models; CPU checks: python -B -m architecture_round2.tests."""
import copy
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from experiments.arch70m.models import ArchitectureModel
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig, FixedChainBlock
from dllm.pipelines.llada.models.modeling_llada import init_weights


def learn_scale(block, names):
    for name in names:
        value = getattr(block, name).clone(); delattr(block, name)
        block.register_parameter(name, nn.Parameter(value))


class HierarchicalBlock(FixedChainBlock):
    """Independent chain-local attention, global attention, then a smaller FFN."""
    def __init__(self, original, layer):
        super().__init__(original.base)
        self.sdpa_backend = original.sdpa_backend
        b = self.base; width = b.config.d_model
        if 4*width % 3: raise ValueError('Parameter matching requires d_model divisible by three.')
        self.local_norm = copy.deepcopy(b.attn_norm)
        for name in ('q_proj', 'k_proj', 'v_proj', 'attn_out'):
            source = getattr(b, name)
            module = nn.Linear(width, width, bias=source.bias is not None)
            init_weights(b.config, module, d=width, layer_id=layer if name == 'attn_out' else None)
            setattr(self, 'local_'+name, module)
        self.register_buffer('local_scale', torch.full((width,), .1), persistent=False)
        ffn_width = b.ff_proj.out_features-4*width//3
        if ffn_width < 1: raise ValueError('FFN budget is too small.')
        for name, n_in, n_out in (('ff_proj', width, ffn_width), ('up_proj', width, ffn_width),
                                 ('ff_out', ffn_width, width)):
            module = nn.Linear(n_in, n_out, bias=getattr(b, name).bias is not None)
            init_weights(b.config, module, d=n_in, layer_id=layer if name == 'ff_out' else None)
            setattr(b, name, module)

    def forward(self, hidden_states, attention_bias, position_ids):
        local_bias, global_bias = attention_bias
        b = self.base; x = self.local_norm(hidden_states)
        n, length, width = x.shape; heads = b.config.n_heads; head_dim = width//heads
        q, k, v = [getattr(self, 'local_'+name)(x).view(n, length, heads, head_dim).transpose(1, 2)
                   for name in ('q_proj', 'k_proj', 'v_proj')]
        q, k = b.rotary_emb(q, k, position_ids=position_ids)
        backends = [SDPBackend.MATH]
        if q.is_cuda and self.sdpa_backend in ('auto', 'efficient'):
            backends = [SDPBackend.EFFICIENT_ATTENTION]
            if self.sdpa_backend == 'auto': backends.append(SDPBackend.MATH)
        with sdpa_kernel(backends):
            local = b._scaled_dot_product_attention(q, k, v, attn_mask=b._cast_attn_bias(local_bias, k.dtype),
                dropout_p=b.config.attention_dropout if self.training else 0., is_causal=False)
        local = local.transpose(1, 2).contiguous().reshape(n, length, width)
        hidden_states = hidden_states+b.dropout(self.local_attn_out(local)*self.local_scale)
        return super().forward(hidden_states, global_bias, position_ids)


class RoundTwoModel(ArchitectureModel):
    def mask_key(self, layer):
        return ('hierarchical',) if self.round2_variant in ('B2', 'B3') else ('global',)

    def _build_attention_biases(self, chain_ids, position_ids, attention_mask):
        if self.round2_variant not in ('B2', 'B3'):
            return super()._build_attention_biases(chain_ids, position_ids, attention_mask)
        valid = attention_mask.bool()
        same = chain_ids[:, :, None].eq(chain_ids[:, None, :])
        protein_same = same & chain_ids[:, :, None].ge(0) & chain_ids[:, None, :].ge(0)
        base = torch.where(protein_same[:, None], self.chain_bias[1][None, :, None, None],
                           self.chain_bias[0][None, :, None, None])
        global_allowed = valid[:, None, :].expand(-1, chain_ids.shape[1], -1)
        local_allowed = (same & global_allowed) | (~valid[:, :, None] & global_allowed)
        minimum = torch.finfo(base.dtype).min
        return {('hierarchical',): (base.masked_fill(~local_allowed[:, None], minimum),
                                    base.masked_fill(~global_allowed[:, None], minimum))}


def build(settings, variant='B1', seed=43):
    if variant not in ('B0', 'B1', 'B2', 'B3'): raise ValueError(variant)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RoundTwoModel(FixedChainConfig(**settings['model']), family_spec=settings['family_prediction'])
        model.architecture = 'A5'; model.round2_variant = variant
        torch.manual_seed(seed+700001)
        for i, block in enumerate(model.backbone.transformer.blocks):
            if variant in ('B2', 'B3'):
                block = HierarchicalBlock(block, i); model.backbone.transformer.blocks[i] = block
            if variant in ('B1', 'B3'):
                names = ('attention_scale', 'ffn_scale')+ (('local_scale',) if variant == 'B3' else ())
                learn_scale(block, names)
        return model
