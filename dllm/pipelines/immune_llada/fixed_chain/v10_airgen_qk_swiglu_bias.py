"""V10 MINT with ESMC-style QK LayerNorm/SwiGLU and per-layer score bias.

Activate pllm; run USE_TF=0 python -B -m pytest /project_c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_v10_airgen_qk_swiglu_bias.py -q
This is a distinct checkpoint architecture, not an implicit upgrade of V10.
Tokenization and MINT routing stay V10's; training mixes MLM and whole-CDR noise.
The current variant has no family classification head, labels or loss.
The current input adapter uses token+role embeddings and has no scope feature.
Each layer broadcasts its post-attention type-token hidden state to the
corresponding receptor chains through an independent zero-initialized gate.
"""
import math
import os
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..data.records import BioSeqRecord, CHAIN_ROLE_TO_ID
from ..v3.vocabulary import make_tokenizer
from .model import FixedChainConfig
from .v10_airgen import AirGenBackbone, AirGenBlock, V10Model, OPTIONS as V10_OPTIONS
from .v9_bert import RECIPE as V9_RECIPE
from .region_mlm import SPAN_POLICY, annotate_regions, corrupt
from .unbinding_vhh import VHHUnbindingCollator
from .mlm import selected_loss

ARCHITECTURE = 'airgen_mint_v10_qk_swiglu_bias'
SCHEMA = 'immune.fixed_chain.v10_airgen_qk_swiglu_bias.v7'
OBJECTIVE = 'binding_mlm_nonbinding_relation_only_vhh_label_independent'
RECIPE = {k:v for k,v in V9_RECIPE.items()
          if k not in ('auxiliary', 'auxiliary_population', 'auxiliary_loss', 'pooling')}
RECIPE.update(supervision=OBJECTIVE, auxiliary='none',
              region_corruption=SPAN_POLICY,
              input_corruption='label_independent_70pct_MLM_15pct_whole_CDR_MASK_15pct_whole_CDR_mutation; '
                               'unavailable_regions_fall_back_to_MLM',
              selection_probability_scope='MLM_branch_only; span_branch_selects_every_residue_in_one_CDR',
              unlabelled='same_mixed_corruption_and_clean_token_targets_as_binding; no_fabricated_relation_labels',
              context_layout='preserve_single_context; multiple_antigens_with_distinct_physical_chain_ids',
              nonbinding='token_loss_relation_only; label_independent_sequence_corruption; no_family_auxiliary')
INPUT_FEATURES = ('input_ids', 'attention_mask', 'role_ids', 'chain_ids',
                  'position_ids', 'target_slot_mask', 'type_token_indices')
ROLE_TYPE_TOKENS = {
    CHAIN_ROLE_TO_ID['antibody_heavy']: '<ab>',
    CHAIN_ROLE_TO_ID['antibody_light']: '<ab>',
    CHAIN_ROLE_TO_ID['nanobody_vhh']: '<nb>',
    CHAIN_ROLE_TO_ID['tcr_alpha']: '<tcr>',
    CHAIN_ROLE_TO_ID['tcr_beta']: '<tcr>',
}
OPTIONS = dict(V10_OPTIONS, architecture=ARCHITECTURE,
    input_adapter='token_plus_role; no_scope_embedding_or_scope_ids; no_esm_token_dropout',
    qk_normalization='independent_intra_q_k_inter_q_k_layer_norm; full_d_model; before_heads_and_rope; gain_only',
    activation='swiglu', ffn_ratio=8/3, ffn_round_to=256, ffn_linear_bias=False,
    score_bias='intra_reference0; generic_inter_base; pair_and_recognition_offsets; '
               'per_layer_per_head; independent_across_layers; symmetric_relation_types; zero_initialized',
    score_bias_relations='pair=VH_VL_or_TCRalpha_TCRbeta; recognition=receptor_vs_antigen_peptide_mhc; '
                         'other_inter=base_only; chain_equality_routes_intra_first; labels_never_used',
    score_bias_implementation='seven_role_factorized_constant_qk_features; pad_to_multiple8; original_head_dim_scale',
    type_feature_feedback='post_attention_residual_before_ffn; corresponding_receptor_chain_slots; '
                          'affine_free_layer_norm_of_dynamic_marker_hidden; per_layer_scalar_gate_zero_initialized; '
                          'explicit_pre_corruption_type_token_indices',
    family_prediction='none; no_head_labels_loss_or_metrics')


class V10Collator(VHHUnbindingCollator):
    """Attach routing and training-only CDR annotations before input corruption."""
    def _layout_row(self, record):
        antigens = [c for c in record.chains if c.role == 'antigen']
        if len(antigens) < 2:
            return super().row(record)
        if any(c.role not in ('antigen', 'antibody_heavy', 'antibody_light', 'nanobody_vhh')
               for c in record.chains):
            raise ValueError('Multiple antigen chains currently require an AB/VHH receptor')
        receptors = [c for c in record.chains if c.role != 'antigen']
        # The legacy renderer accepts one antigen. Render each context privately,
        # combine its observed content, then shift receptor physical chain IDs.
        rows = [super(V10Collator, self).row(replace(record, chains=[ag, *receptors])) for ag in antigens]
        row = rows[0]
        end = row['input_ids'].index(self.tokenizer.special_id('<protd>'))
        additions = {key: [] for key in row}
        for chain, extra in enumerate(rows[1:], 1):
            stop = extra['input_ids'].index(self.tokenizer.special_id('<protd>'))
            for key in row:
                separator = (self.tokenizer.chain_separator_id() if key == 'input_ids' else
                             -100 if key == 'labels' else -1 if key == 'chain_ids' else
                             1 if key == 'attention_mask' else 0)
                values = extra[key][1:stop]
                if key == 'chain_ids':
                    values = [chain if value >= 0 else value for value in values]
                additions[key].extend([separator, *values])
        row['chain_ids'] = [value + len(antigens) - 1 if value > 0 else value for value in row['chain_ids']]
        for key in row:
            row[key] = row[key][:end] + additions[key] + row[key][end:]
        if len(row['input_ids']) > self.config.max_length:
            raise ValueError('sequence_budget_overflow')
        return row

    def row(self, item):
        record = BioSeqRecord.from_prepared_dict(item) if isinstance(item, dict) else item
        row = self._layout_row(record)
        marker_ids = {self.tokenizer.special_id(token): token for token in set(ROLE_TYPE_TOKENS.values())}
        latest, indices = {}, []
        for position, (token, role, chain) in enumerate(zip(row['input_ids'], row['role_ids'], row['chain_ids'])):
            if chain == -1 and token in marker_ids:
                latest[marker_ids[token]] = position
            marker = ROLE_TYPE_TOKENS.get(role) if chain >= 0 else None
            if marker is not None and marker not in latest:
                raise ValueError('Receptor chain has no preceding type token: '+marker)
            indices.append(latest[marker] if marker is not None else -1)
        row['type_token_indices'] = indices
        # Generation never exposes reference lengths or reference annotations.
        return annotate_regions(row, record, self.tokenizer, enabled=self.mode == 'train')

    def __call__(self, records):
        batch = super().__call__(records)
        batch.pop('scope_ids')
        # The inherited padder defaults new integer fields to zero. Padding
        # must never receive a feature from token zero.
        batch['type_token_indices'].masked_fill_(~batch['attention_mask'], -1)
        return batch


def swiglu_hidden_size(width):
    """Match ESMC's ceil-to-256 expansion of 8*d_model/3."""
    return math.ceil((8 / 3) * width / 256) * 256


def validate_config(config):
    if (not isinstance(config, FixedChainConfig) or config.n_layers < 1
            or config.d_model % config.n_heads or (config.d_model // config.n_heads) % 2
            or config.mlp_hidden_size != swiglu_hidden_size(config.d_model)):
        raise ValueError('V10 QK/SwiGLU/bias requires even head width and the ESMC rounded 8/3 FFN')


class AirGenQKSwiGLUBiasBlock(AirGenBlock):
    """Reuse MINT head/RoPE helpers with separately normalized Q/K and SwiGLU."""
    def __init__(self, config):
        nn.Module.__init__(self)
        validate_config(config)
        width = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = width // config.n_heads
        self.inter_score_bias = nn.Parameter(torch.zeros(config.n_heads))
        self.pair_score_bias = nn.Parameter(torch.zeros(config.n_heads))
        self.recognition_score_bias = nn.Parameter(torch.zeros(config.n_heads))
        self.type_feature_gate = nn.Parameter(torch.zeros(()))
        for branch in ['intra', 'inter']:
            for letter in 'qkv':
                setattr(self, branch+'_'+letter, nn.Linear(width, width))
            for letter in 'qk':
                setattr(self, branch+'_'+letter+'_norm', nn.LayerNorm(width, bias=False))
        self.out_proj = nn.Linear(width, width)
        self.attn_norm = nn.LayerNorm(width)
        self.ff_norm = nn.LayerNorm(width)
        # Exactly ESMC's FFN: LayerNorm -> Linear(2m, no bias) ->
        # SiLU(first half) * second half -> Linear(d_model, no bias).
        self.fc1 = nn.Linear(width, 2 * config.mlp_hidden_size, bias=False)
        self.fc2 = nn.Linear(config.mlp_hidden_size, width, bias=False)
        self.register_buffer('inv_freq', 1. / (10000 ** (
            torch.arange(0, self.head_dim, 2).float() / self.head_dim)), persistent=False)
        self.sdpa_backend = os.environ.get('IMMUNE_SDPA_BACKEND', 'auto')
        for branch in ['intra', 'inter']:
            for letter in 'qkv':
                nn.init.xavier_uniform_(getattr(self, branch+'_'+letter).weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def normalized_qk(self, normalized, positions):
        projected = {}
        for branch in ['intra', 'inter']:
            for letter in 'qk':
                name = branch+'_'+letter
                value = getattr(self, name)(normalized)
                value = getattr(self, name+'_norm')(value).to(value.dtype)
                value = self.heads(value)
                projected[name] = self.rope(value, positions) if branch == 'intra' else value
        return projected['intra_q'], projected['intra_k'], projected['inter_q'], projected['inter_k']

    def attention(self, qi, ki, qc, kc, vi, vc, allowed_copies, inter_score_bias, role_ids=None):
        # Exact role-factorized relation biases, without a trainable B*H*L*L
        # float mask or extra copies of keys/values. Only the inter key copy
        # carries these features; same physical chain always uses intra.
        query = torch.cat((qi, qc), -1)
        if role_ids is None:
            role_ids = torch.zeros(query.shape[0], query.shape[2], dtype=torch.long, device=query.device)
        heavy, light, alpha, beta = [role_ids.eq(CHAIN_ROLE_TO_ID[name]) for name in
            ('antibody_heavy', 'antibody_light', 'tcr_alpha', 'tcr_beta')]
        receptor = heavy | light | alpha | beta | role_ids.eq(CHAIN_ROLE_TO_ID['nanobody_vhh'])
        context = (role_ids.eq(CHAIN_ROLE_TO_ID['antigen']) | role_ids.eq(CHAIN_ROLE_TO_ID['peptide'])
                   | role_ids.eq(CHAIN_ROLE_TO_ID['mhc']))
        flags = torch.stack((heavy, light, alpha, beta, receptor, context), -1).to(query.dtype)[:, None]
        offsets = torch.stack((self.pair_score_bias,) * 4 + (self.recognition_score_bias,) * 2, -1)
        offsets = offsets.to(query.dtype)[None, :, None]
        base = inter_score_bias.to(query.dtype)[None, :, None, None].expand(*query.shape[:-1], 1)
        bias_query = torch.cat((base, offsets * flags), -1) * math.sqrt(self.head_dim)
        bias_key = torch.stack((torch.ones_like(heavy), light, heavy, beta, alpha, context, receptor), -1)
        bias_key = bias_key.to(query.dtype)[:, None].expand(*query.shape[:-1], 7)
        padding = -(query.shape[-1] + 7) % 8
        query = torch.cat((query, F.pad(bias_query, (0, padding))), -1)
        intra_key = F.pad(torch.cat((ki, torch.zeros_like(ki)), -1), (0, 7 + padding))
        inter_key = torch.cat((torch.zeros_like(kc), kc, F.pad(bias_key, (0, padding))), -1)
        key = torch.cat((intra_key, inter_key), -2)
        value = torch.cat((vi, vc), -2)
        policy = [SDPBackend.MATH]
        if query.is_cuda and self.sdpa_backend in ('auto', 'efficient'):
            policy = [SDPBackend.EFFICIENT_ATTENTION]
            if self.sdpa_backend == 'auto':
                policy.append(SDPBackend.MATH)
        with sdpa_kernel(policy):
            return F.scaled_dot_product_attention(query, key, value,
                attn_mask=allowed_copies, dropout_p=0., scale=self.head_dim ** -.5)

    def add_type_features(self, hidden, type_token_indices):
        """Broadcast a snapshot of dynamic marker features, never static embeddings."""
        if type_token_indices is None:
            # Standalone attention comparisons can omit routing. Keep the
            # gate connected with a zero gradient, including on DDP ranks.
            return hidden + self.type_feature_gate.to(hidden.dtype) * 0
        sources = type_token_indices.clamp_min(0).unsqueeze(-1).expand_as(hidden)
        features = hidden.gather(1, sources)
        features = F.layer_norm(features, (hidden.shape[-1],))
        features = features * type_token_indices.ge(0).unsqueeze(-1)
        return hidden + self.type_feature_gate.to(hidden.dtype) * features

    def forward(self, hidden, allowed_copies, position_ids, type_token_indices=None, role_ids=None):
        normalized = self.attn_norm(hidden)
        qi, ki, qc, kc = self.normalized_qk(normalized, position_ids)
        vi, vc = self.heads(self.intra_v(normalized)), self.heads(self.inter_v(normalized))
        attended = self.attention(qi, ki, qc, kc, vi, vc, allowed_copies, self.inter_score_bias, role_ids)
        attended = attended.transpose(1, 2).reshape_as(hidden)
        hidden = hidden + self.out_proj(attended)
        hidden = self.add_type_features(hidden, type_token_indices)
        gate, value = self.fc1(self.ff_norm(hidden)).chunk(2, -1)
        return hidden + self.fc2(F.silu(gate) * value)


class AirGenQKSwiGLUBiasBackbone(AirGenBackbone):
    block_class = AirGenQKSwiGLUBiasBlock


class V10QKSwiGLUBiasModel(V10Model):
    architecture = ARCHITECTURE
    schema = SCHEMA
    architecture_options = OPTIONS
    backbone_class = AirGenQKSwiGLUBiasBackbone
    config_validator = staticmethod(validate_config)
    uses_scope_embeddings = False
    uses_family_prediction = False
    objective = OBJECTIVE
    recipe = RECIPE

    def forward(self, **batch):
        logits, _ = self.denoise(**self.feature_args(batch))
        return dict(loss_sum=selected_loss(logits, batch['labels']), logits=logits)

    @staticmethod
    def feature_args(batch):
        return {name: batch[name] for name in INPUT_FEATURES}

    def denoise(self, input_ids, attention_mask, role_ids, chain_ids, position_ids,
                target_slot_mask, type_token_indices, *, output_hidden_states=False):
        self._validate_target_features(role_ids, chain_ids, position_ids, target_slot_mask)
        self._validate_type_features(type_token_indices, attention_mask, role_ids, chain_ids)
        hidden = self.backbone.transformer.wte(input_ids) + self.role_embeddings(role_ids)
        return self._denoise_embedded(hidden, attention_mask, chain_ids, position_ids, type_token_indices, role_ids,
                                      output_hidden_states=output_hidden_states)

    @staticmethod
    def _validate_type_features(indices, attention_mask, role_ids, chain_ids):
        if indices.shape != role_ids.shape or indices.dtype != torch.long:
            raise ValueError('type_token_indices must be int64 with the token layout shape')
        if ((indices < -1) | (indices >= role_ids.shape[1])).any():
            raise ValueError('type_token_indices is outside the token layout')
        receivers = torch.zeros_like(attention_mask, dtype=torch.bool)
        for role in ROLE_TYPE_TOKENS:
            receivers |= role_ids.eq(role)
        receivers &= attention_mask.bool() & chain_ids.ge(0)
        if not torch.equal(indices.ge(0), receivers):
            raise ValueError('type_token_indices must cover exactly the valid receptor chain slots')
        sources = indices.clamp_min(0)
        source_valid = (attention_mask.bool().gather(1, sources)
                        & chain_ids.gather(1, sources).eq(-1) & role_ids.gather(1, sources).eq(0))
        if (receivers & ~source_valid).any():
            raise ValueError('Type feature sources must be attended grammar positions')


def parameter_count(config):
    """Exact parameter contract, including per-layer biases and task adapters."""
    validate_config(config)
    width, middle = config.d_model, config.mlp_hidden_size
    vocabulary = make_tokenizer(config.ablation).vocab_size
    inputs = (vocabulary + len(CHAIN_ROLE_TO_ID)) * width
    blocks = config.n_layers * (7 * width**2 + 15 * width + 3 * width * middle)
    heads = width**2 + 5 * width + vocabulary
    return inputs + blocks + heads + config.n_layers * (3 * config.n_heads + 1)


def components(family_spec=None):
    if family_spec is not None:
        raise ValueError('The current AirGen variant has no family prediction')
    return V10Collator, V10QKSwiGLUBiasModel, corrupt, RECIPE
