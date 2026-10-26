"""Eight shared-trunk models for artificial integer-object joint tasks.

Activate conda pllm; run the artificial CPU checks at:
/project_c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_generic_joint_prototype.py
This module imports no receptor model, real corpus, pretrained weights, or trainer.
The optional EMA network is training-only; both inference tasks share one trunk.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .generic_joint_data import Inputs, ALPHABET, MASK, PAD


VARIANTS = ('g00_shared_joint', 'g01_task_norm', 'g02_late_adapters', 'g03_register4',
            'g04_bottleneck4', 'g05_bottleneck16', 'g06_self_alignment',
            'g07_alignment_compute_control')


@dataclass(frozen=True)
class ModelConfig:
    variant: str = VARIANTS[0]
    d_model: int = 192
    layers: int = 4
    heads: int = 6
    adapter_rank: int = 32

    def __post_init__(self):
        if (self.variant not in VARIANTS or type(self.d_model) is not int
                or self.d_model <= 0 or self.d_model % 2 or self.layers != 4
                or self.heads != 6 or self.d_model % self.heads or self.adapter_rank != 32):
            raise ValueError('Unsupported bounded joint-prototype configuration')


class SharedBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, hidden, allowed, valid):
        batch, length, width = hidden.shape
        qkv = self.qkv(self.norm_attention(hidden)).reshape(batch, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed[:, None], dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        hidden = hidden + self.attention_output(attended)
        hidden = hidden + self.mlp(self.norm_mlp(hidden))
        return hidden * valid[..., None]


class TaskAdapter(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden):
        return self.up(F.gelu(self.down(hidden)))


class JointModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Construct shared modules first so a fixed seed yields identical shared
        # initialization across methods. Optional modules come afterwards.
        width = config.d_model
        self.token_embedding = nn.Embedding(ALPHABET + 2, width, padding_idx=PAD)
        self.role_embedding = nn.Embedding(5, width, padding_idx=0)
        self.blocks = nn.ModuleList(SharedBlock(width, config.heads) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(width)
        self.token_head = nn.Linear(width, ALPHABET)
        self.relation_head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 4))
        self.task_norms = nn.ModuleDict()
        if config.variant == 'g01_task_norm':
            self.task_norms['understanding'] = self.final_norm
            self.task_norms['generation'] = nn.LayerNorm(width)
        self.adapters = nn.ModuleDict()
        if config.variant == 'g02_late_adapters':
            for task in ('understanding', 'generation'):
                for layer in (2, 3):
                    self.adapters[task + '_' + str(layer)] = TaskAdapter(width, config.adapter_rank)
        self.register_count = (16 if config.variant == 'g05_bottleneck16' else
                               4 if config.variant in ('g03_register4', 'g04_bottleneck4') else 0)
        self.registers = nn.Parameter(torch.randn(self.register_count, width) * 0.02)
        self.bottleneck = config.variant in ('g04_bottleneck4', 'g05_bottleneck16')

    def position_encoding(self, positions):
        width = self.config.d_model
        scales = torch.exp(torch.arange(0, width, 2, device=positions.device, dtype=torch.float32)
                           * (-math.log(10000.0) / width))
        angles = positions[..., None].float() * scales
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)

    def attention_layout(self, inputs):
        count, length = inputs.tokens.shape
        registers = torch.ones(count, self.register_count, device=inputs.valid.device, dtype=torch.bool)
        valid = torch.cat((registers, inputs.valid), dim=1)
        roles = F.pad(inputs.roles, (self.register_count, 0), value=0)
        size = length + self.register_count
        if self.bottleneck:
            allowed = roles[:, :, None].eq(roles[:, None, :])
            is_register = torch.arange(size, device=inputs.tokens.device) < self.register_count
            allowed = allowed | is_register[None, :, None] | is_register[None, None, :]
        else:
            allowed = torch.ones(count, size, size, device=inputs.valid.device, dtype=torch.bool)
        allowed = allowed & valid[:, None, :]
        return allowed, valid

    def forward(self, inputs, task):
        if not isinstance(inputs, Inputs) or task not in ('understanding', 'generation'):
            raise ValueError('Forward receives only the four input fields and a fixed task')
        hidden = (self.token_embedding(inputs.tokens) + self.role_embedding(inputs.roles)
                  + self.position_encoding(inputs.positions).to(self.token_embedding.weight.dtype))
        hidden = hidden * inputs.valid[..., None]
        registers = self.registers[None].expand(hidden.shape[0], -1, -1)
        hidden = torch.cat((registers, hidden), dim=1)
        allowed, valid = self.attention_layout(inputs)
        layers = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, allowed, valid)
            adapter = task + '_' + str(index)
            if adapter in self.adapters:
                hidden = (hidden + self.adapters[adapter](hidden)) * valid[..., None]
            layers.append(hidden[:, self.register_count:])
        tokens = hidden[:, self.register_count:]
        norm = self.task_norms[task] if task in self.task_norms else self.final_norm
        features = norm(tokens)
        summaries = []
        for role in (1, 2):
            chosen = inputs.roles.eq(role) & inputs.valid
            denominator = chosen.sum(1).clamp_min(1)[:, None]
            summaries.append((features * chosen[..., None]).sum(1) / denominator)
        return {'token_logits': self.token_head(features),
                'relation_logits': self.relation_head(torch.cat(summaries, dim=-1)),
                'features': features, 'layers': tuple(layers)}
