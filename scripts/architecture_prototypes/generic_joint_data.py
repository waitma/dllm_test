"""Generate mathematical integer objects; no corpus or sequence-file reader.

Activate conda pllm and run the artificial CPU checks:
USE_TF=0 CUDA_VISIBLE_DEVICES='' python -m pytest
/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_generic_joint_prototype.py -q
All model inputs are separate from labels, record IDs, and corruption metadata.
"""
from dataclasses import dataclass, replace
import hashlib
import math

import torch


ALPHABET = 16
MASK = 16
PAD = 17
TARGET_ROLE = 3
MASK_LEVELS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Inputs:
    tokens: torch.Tensor
    roles: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor

    def rows(self, start, end):
        return Inputs(*(value[start:end] for value in
                        (self.tokens, self.roles, self.positions, self.valid)))

    def permuted(self, indices):
        return Inputs(*(value[:, indices] for value in
                        (self.tokens, self.roles, self.positions, self.valid)))


@dataclass(frozen=True)
class Examples:
    clean: Inputs
    relation_labels: torch.Tensor
    token_labels: torch.Tensor
    record_ids: tuple
    segment_width: int


@dataclass(frozen=True)
class JointBatch:
    understanding: Inputs
    generation: Inputs
    teacher: Inputs
    relation_labels: torch.Tensor
    token_labels: torch.Tensor
    selected: torch.Tensor
    record_ids: tuple

    @property
    def records(self):
        return self.generation.tokens.shape[0]

    def rows(self, start, end):
        return JointBatch(self.understanding.rows(start, end), self.generation.rows(start, end),
                          self.teacher.rows(start, end), self.relation_labels[start:end],
                          self.token_labels[start:end], self.selected[start:end],
                          self.record_ids[start:end])


def make_examples(records, *, seed, split='train', lengths=(8, 16), distractor_probability=0.5):
    """B=(A+delta)%16, delta in 0..3; C=(A+B)%16, with independent distractors.

    Split and seed determine a local RNG. Record IDs never reach the model.
    Known segment lengths are part of this toy protocol, not learned stopping.
    """
    if (type(records) is not int or records <= 0 or type(seed) is not int
            or split not in ('train', 'validation')
            or not lengths or any(type(n) is not int or not 2 <= n <= 64 for n in lengths)
            or not 0 <= distractor_probability <= 1):
        raise ValueError('Invalid bounded artificial-object specification')
    salt = int.from_bytes(hashlib.sha256(split.encode()).digest()[:4], 'little')
    generator = torch.Generator().manual_seed(seed + salt)
    width = max(lengths)
    length = torch.tensor(lengths)[torch.arange(records) % len(lengths)]
    length = length[torch.randperm(records, generator=generator)]
    tokens = torch.randint(ALPHABET, (records, 4, width), generator=generator)
    delta = torch.randint(4, (records,), generator=generator)
    tokens[:, 1] = (tokens[:, 0] + delta[:, None]) % ALPHABET
    tokens[:, 2] = (tokens[:, 0] + tokens[:, 1]) % ALPHABET
    present = torch.ones(records, 4, dtype=torch.bool)
    present[:, 3] = torch.rand(records, generator=generator) < distractor_probability
    positions = torch.arange(width)[None, None, :].expand(records, 4, width).clone()
    valid = (positions < length[:, None, None]) & present[:, :, None]
    roles = torch.arange(1, 5)[None, :, None].expand(records, 4, width).clone()
    tokens = tokens.masked_fill(~valid, PAD)
    roles = roles.masked_fill(~valid, 0)
    positions = positions.masked_fill(~valid, 0)
    inputs = Inputs(*(value.reshape(records, -1) for value in (tokens, roles, positions, valid)))
    labels = inputs.tokens.masked_fill(~inputs.roles.eq(TARGET_ROLE), -100)
    return Examples(inputs, delta, labels, tuple((split, seed, n) for n in range(records)), width)


def corrupt(examples, *, seed, mask_levels=MASK_LEVELS, teacher_multiplier=0.5):
    if (not mask_levels or any(level not in MASK_LEVELS for level in mask_levels)
            or not 0 <= teacher_multiplier < 1):
        raise ValueError('Only the declared four-level artificial corruption is supported')
    clean = examples.clean
    if clean.tokens.device.type != 'cpu':
        raise ValueError('Corrupt on the CPU before moving any artificial batch')
    target = clean.roles.eq(TARGET_ROLE) & clean.valid
    selected = torch.zeros_like(target)
    teacher_selected = torch.zeros_like(target)
    generator = torch.Generator().manual_seed(seed)
    priorities = torch.rand(clean.tokens.shape, generator=generator)
    for row in range(clean.tokens.shape[0]):
        indices = torch.where(target[row])[0]
        indices = indices[torch.argsort(priorities[row, indices])]
        count = max(1, math.ceil(indices.numel() * mask_levels[row % len(mask_levels)]))
        selected[row, indices[:count]] = True
        teacher_selected[row, indices[:math.floor(count * teacher_multiplier)]] = True
    understanding = replace(clean, tokens=clean.tokens.masked_fill(target, MASK))
    generation = replace(clean, tokens=clean.tokens.masked_fill(selected, MASK))
    teacher = replace(clean, tokens=clean.tokens.masked_fill(teacher_selected, MASK))
    return JointBatch(understanding, generation, teacher, examples.relation_labels,
                      examples.token_labels, selected, examples.record_ids)


def segment_permutation(width, order=(2, 0, 3, 1)):
    if sorted(order) != [0, 1, 2, 3]:
        raise ValueError('Permute all four whole segments exactly once')
    indices = torch.cat([torch.arange(n * width, (n + 1) * width) for n in order])
    return indices, torch.argsort(indices)


def remove_condition(inputs, role=1):
    """Mask one necessary condition; unchanged labels are a dependency diagnostic."""
    if role not in (1, 2):
        raise ValueError('Only context roles can be removed')
    return replace(inputs, tokens=inputs.tokens.masked_fill(inputs.roles.eq(role), MASK))


def move_batch(batch, device):
    def move(view):
        return Inputs(*(value.to(device) for value in
                        (view.tokens, view.roles, view.positions, view.valid)))
    return JointBatch(move(batch.understanding), move(batch.generation), move(batch.teacher),
                      batch.relation_labels.to(device), batch.token_labels.to(device),
                      batch.selected.to(device), batch.record_ids)
