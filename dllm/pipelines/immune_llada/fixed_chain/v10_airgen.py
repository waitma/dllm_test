"""AirGen/MINT blocks with the unchanged V9 token, data and objective interfaces.

Activate pllm; run python -m pytest scripts/tests/immune_llada/test_v10_airgen.py.
No ESM weights or alphabet are loaded. The sequence layout and noisy targets are
exactly V9's; token+role+scope embeddings remain our input adapter. Each block
has separate intra/inter QKV, intra-only RoPE, one joint softmax, LayerNorm,
GELU and ordinary residuals. The tied RoBERTa LM head follows AirGen.
"""
from dataclasses import asdict
from functools import partial
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from ..data.records import CHAIN_ROLE_TO_ID
from ..v3.layout import SCOPES
from ..v3.vocabulary import make_tokenizer
from .full_token_mlm import excluded_ids
from .model import FixedChainConfig
from .v9_bert import V9Collator, V9Model, OBJECTIVE, RECIPE, corrupt
from .v_family import validate_spec

ARCHITECTURE = 'airgen_mint_v10'
SCHEMA = 'immune.fixed_chain.v10_airgen_bert.v1'
OPTIONS = dict(architecture=ARCHITECTURE, residual_scale=1.,
    attention='independent_intra_inter_qkv_joint_softmax_every_layer',
    position='intra_chain_rope10000; inter_chain_no_rope',
    normalization='layer_norm', activation='gelu', ffn_ratio=4,
    input_adapter='unchanged_token_plus_role_plus_scope; no_esm_token_dropout',
    output_adapter='airgen_roberta_tied_lm_head; unchanged_vocabulary_mask')
V10Collator = V9Collator


def validate_config(config):
    if (not isinstance(config, FixedChainConfig) or config.n_layers < 1
            or config.d_model % config.n_heads or (config.d_model // config.n_heads) % 2
            or config.mlp_hidden_size != 4 * config.d_model):
        raise ValueError('V10 requires even head width and the AirGen 4x FFN')


class AirGenBlock(nn.Module):
    """Exact MINT attention using a memory-efficient doubled-key SDPA mapping.

    Q=[Q_intra,Q_inter], K=[K_intra,0] or [0,K_inter], V=V_intra or V_inter.
    Every original key has exactly one allowed copy. This implements the same
    SINGLE softmax over intra/inter edges; separately normalized branches would
    change the model. No B*heads*length*length score tensor is materialized.
    """
    def __init__(self, config):
        super().__init__()
        validate_config(config)
        width = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = width // config.n_heads
        self.intra_q = nn.Linear(width, width)
        self.intra_k = nn.Linear(width, width)
        self.intra_v = nn.Linear(width, width)
        self.inter_q = nn.Linear(width, width)
        self.inter_k = nn.Linear(width, width)
        self.inter_v = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.attn_norm = nn.LayerNorm(width)
        self.ff_norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, config.mlp_hidden_size)
        self.fc2 = nn.Linear(config.mlp_hidden_size, width)
        self.register_buffer('inv_freq', 1. / (10000 ** (
            torch.arange(0, self.head_dim, 2).float() / self.head_dim)), persistent=False)
        self.sdpa_backend = os.environ.get('IMMUNE_SDPA_BACKEND', 'auto')
        for projection in (self.intra_q, self.intra_k, self.intra_v,
                           self.inter_q, self.inter_k, self.inter_v):
            nn.init.xavier_uniform_(projection.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def heads(self, value):
        return value.view(*value.shape[:2], self.n_heads, self.head_dim).transpose(1, 2)

    def rope(self, value, positions):
        # Positions reset at chain boundaries, exactly as in the V9 input.
        phase = positions.float()[..., None] * self.inv_freq.float()
        phase = torch.cat((phase, phase), -1)[:, None]
        left, right = value.chunk(2, -1)
        return (value * phase.cos() + torch.cat((-right, left), -1) * phase.sin()).to(value.dtype)

    def forward(self, hidden, allowed_copies, position_ids):
        normalized = self.attn_norm(hidden)
        qi = self.rope(self.heads(self.intra_q(normalized)), position_ids)
        ki = self.rope(self.heads(self.intra_k(normalized)), position_ids)
        qc = self.heads(self.inter_q(normalized))
        kc = self.heads(self.inter_k(normalized))
        vi = self.heads(self.intra_v(normalized))
        vc = self.heads(self.inter_v(normalized))
        query = torch.cat((qi, qc), -1)
        key = torch.cat((torch.cat((ki, torch.zeros_like(ki)), -1),
                         torch.cat((torch.zeros_like(kc), kc), -1)), -2)
        value = torch.cat((vi, vc), -2)
        policy = [SDPBackend.MATH]
        if query.is_cuda and self.sdpa_backend in ('auto', 'efficient'):
            policy = [SDPBackend.EFFICIENT_ATTENTION]
            if self.sdpa_backend == 'auto':
                policy.append(SDPBackend.MATH)
        with sdpa_kernel(policy):
            attended = F.scaled_dot_product_attention(query, key, value,
                attn_mask=allowed_copies, dropout_p=0., scale=self.head_dim ** -.5)
        attended = attended.transpose(1, 2).reshape_as(hidden)
        hidden = hidden + self.out_proj(attended)
        return hidden + self.fc2(F.gelu(self.fc1(self.ff_norm(hidden)), approximate='none'))


class AirGenBackbone(nn.Module):
    """Retain the training loop's activation-checkpointing and block interfaces."""
    block_class = AirGenBlock

    def __init__(self, config, vocabulary_size, pad_id):
        super().__init__()
        self.activation_checkpointing_strategy = None
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocabulary_size, config.d_model, padding_idx=pad_id),
            blocks=nn.ModuleList([self.block_class(config) for _ in range(config.n_layers)]),
            ln_f=nn.LayerNorm(config.d_model)))
        self.lm_dense = nn.Linear(config.d_model, config.d_model)
        self.lm_norm = nn.LayerNorm(config.d_model)
        self.lm_bias = nn.Parameter(torch.zeros(vocabulary_size))
        # Retain the small random token-embedding scale used by our input
        # adapter; there is no pretrained ESM embedding to inherit here.
        nn.init.normal_(self.transformer.wte.weight, std=.02)
        with torch.no_grad():
            self.transformer.wte.weight[pad_id].zero_()

    def set_activation_checkpointing(self, strategy):
        self.activation_checkpointing_strategy = strategy

    def forward(self, hidden, allowed_copies, position_ids, *block_args):
        for block in self.transformer.blocks:
            if self.training and self.activation_checkpointing_strategy is not None:
                hidden = checkpoint(block, hidden, allowed_copies, position_ids,
                                    *block_args, use_reentrant=False)
            else:
                hidden = block(hidden, allowed_copies, position_ids, *block_args)
        return hidden

    def logits(self, hidden):
        transformed = self.lm_norm(F.gelu(self.lm_dense(hidden), approximate='none'))
        # Tie by use, without duplicate state_dict aliases or unused parameters.
        return F.linear(transformed, self.transformer.wte.weight, self.lm_bias)


class V10Model(V9Model):
    """Same V9 loss, collator and public methods; only the backbone is replaced."""
    architecture = ARCHITECTURE
    schema = SCHEMA
    architecture_options = OPTIONS
    backbone_class = AirGenBackbone
    config_validator = staticmethod(validate_config)
    uses_scope_embeddings = True
    uses_family_prediction = True
    objective = OBJECTIVE
    recipe = RECIPE

    def __init__(self, config, *, family_spec=None):
        nn.Module.__init__(self)
        self.config_validator(config)
        if self.uses_family_prediction:
            validate_spec(family_spec)
        elif family_spec is not None:
            raise ValueError('This AirGen version has no family prediction')
        self.config, self.options = config, dict(self.architecture_options)
        self.variant = self.architecture
        self.tokenizer = make_tokenizer(config.ablation)
        self.terminal_id = self.tokenizer.eos_token_id
        if self.uses_family_prediction:
            self.family_spec = json.loads(json.dumps(family_spec))
            self.family_loss_weight = family_spec['loss_weight']
        self.backbone = self.backbone_class(config, self.tokenizer.vocab_size, self.tokenizer.pad_token_id)
        self.role_embeddings = nn.Embedding(len(CHAIN_ROLE_TO_ID), config.d_model, padding_idx=0)
        embeddings = [self.role_embeddings]
        if self.uses_scope_embeddings:
            self.scope_embeddings = nn.Embedding(len(SCOPES), config.d_model, padding_idx=0)
            embeddings.append(self.scope_embeddings)
        for embedding in embeddings:
            nn.init.normal_(embedding.weight, std=.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        if self.uses_family_prediction:
            self.family_head = nn.Linear(config.d_model, len(family_spec['classes']))
        allowed = torch.ones(self.tokenizer.vocab_size, dtype=torch.bool)
        allowed[list(excluded_ids(self.tokenizer))] = False
        self.register_buffer('allowed_target', allowed, persistent=False)

    def denoise(self, input_ids, attention_mask, role_ids, scope_ids, chain_ids,
                position_ids, target_slot_mask, *, output_hidden_states=False):
        self._validate_target_features(role_ids, chain_ids, position_ids, target_slot_mask)
        hidden = (self.backbone.transformer.wte(input_ids) + self.role_embeddings(role_ids)
                  + self.scope_embeddings(scope_ids))
        return self._denoise_embedded(hidden, attention_mask, chain_ids, position_ids,
                                      output_hidden_states=output_hidden_states)

    def _denoise_embedded(self, hidden, attention_mask, chain_ids, position_ids, *block_args,
                          output_hidden_states=False):
        """Shared MINT routing after the version-specific input adapter."""
        valid = attention_mask.bool()
        if not valid.any(1).all():
            raise ValueError('Each record needs at least one attended token')
        hidden = hidden * valid.unsqueeze(-1)
        # Prefix/grammar groups retain their existing chain_ids, including -1.
        same = chain_ids[:, :, None].eq(chain_ids[:, None, :])
        copies = torch.cat((same & valid[:, None, :], ~same & valid[:, None, :]), -1)[:, None]
        hidden = self.backbone(hidden, copies, position_ids, *block_args)
        hidden = self.backbone.transformer.ln_f(hidden)
        logits = self.backbone.logits(hidden).masked_fill(~self.allowed_target[None, None], float('-inf'))
        return logits, hidden if output_hidden_states else None

    def save_research_config(self, directory):
        settings = dict(schema=self.schema, variant=self.objective, options=self.options,
            base=asdict(self.config), recipe=self.recipe)
        if self.uses_family_prediction:
            settings['family_spec'] = self.family_spec
        (Path(directory) / 'research_config.json').write_text(json.dumps(settings, indent=2)+'\n')

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory / 'research_config.json').read_text())
        if (settings.get('schema') != cls.schema or settings.get('variant') != cls.objective
                or settings.get('recipe') != cls.recipe or settings.get('options') != cls.architecture_options
                or (not cls.uses_family_prediction and 'family_spec' in settings)
                or settings['base'] != json.loads((directory / 'immune_config.json').read_text())):
            raise ValueError('V10 AirGen checkpoint contract changed')
        model = cls(FixedChainConfig(**settings['base']), family_spec=settings.get('family_spec'))
        if json.loads((directory / 'vocabulary.json').read_text()) != [
                model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError('V10 vocabulary changed')
        model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
        return model


def components(family_spec=None):
    if family_spec is None:
        return V10Collator, V10Model, corrupt, RECIPE
    return partial(V10Collator, family_spec=family_spec), partial(V10Model, family_spec=family_spec), corrupt, RECIPE
