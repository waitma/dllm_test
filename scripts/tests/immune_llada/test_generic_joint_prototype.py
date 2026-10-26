"""Necessary semantic and gradient checks for eight artificial joint prototypes.

Activate conda pllm and run:
USE_TF=0 CUDA_VISIBLE_DEVICES='' python -m pytest
/project_c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_generic_joint_prototype.py -q
These small CPU cases check contracts; they cannot establish method improvements.
"""
from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch
from torch.nn import functional as F
import yaml

from scripts.architecture_prototypes.generic_joint_data import (
    Inputs, MASK, TARGET_ROLE, corrupt, make_examples, segment_permutation)
from scripts.architecture_prototypes.generic_joint_model import JointModel, ModelConfig, VARIANTS
from scripts.architecture_prototypes import generic_joint_experiment as experiment


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def system(variant):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return experiment.JointSystem(ModelConfig(variant=variant, d_model=48))


def batch(records=7):
    return corrupt(make_examples(records, seed=42, lengths=(4, 8)), seed=101)


def test_mathematical_labels_and_local_rng_are_independently_recomputable():
    rng = torch.random.get_rng_state().clone()
    examples = make_examples(12, seed=42, lengths=(4, 8))
    assert torch.equal(rng, torch.random.get_rng_state())
    duplicate = make_examples(12, seed=42, lengths=(4, 8))
    validation = make_examples(12, seed=42, split='validation', lengths=(4, 8))
    assert torch.equal(examples.clean.tokens, duplicate.clean.tokens)
    assert not set(examples.record_ids) & set(validation.record_ids)
    for row in range(12):
        a = examples.clean.tokens[row, examples.clean.roles[row].eq(1)]
        b = examples.clean.tokens[row, examples.clean.roles[row].eq(2)]
        c = examples.clean.tokens[row, examples.clean.roles[row].eq(3)]
        assert (b - a).remainder(16).eq(examples.relation_labels[row]).all()
        assert c.eq((a + b).remainder(16)).all()


def test_input_channel_has_no_label_id_or_corruption_metadata_and_teacher_is_nested():
    assert {field.name for field in fields(Inputs)} == {'tokens', 'roles', 'positions', 'valid'}
    value = batch()
    target = value.understanding.roles.eq(TARGET_ROLE) & value.understanding.valid
    assert value.understanding.tokens[target].eq(MASK).all()
    assert torch.equal(value.generation.tokens.eq(MASK), value.selected)
    teacher_masked = value.teacher.tokens.eq(MASK)
    assert not (teacher_masked & ~value.selected).any()
    assert (teacher_masked.sum(1) < value.selected.sum(1)).all()
    assert not (value.selected & ~target).any()


@pytest.mark.parametrize('variant', VARIANTS)
def test_joint_backward_reaches_both_heads_and_shared_trunk_without_teacher_gradients(variant):
    model = system(variant)
    value = batch()
    loss, observed = model.objective(value, window_records=value.records)
    assert torch.isfinite(loss) and observed['records'] == 7
    loss.backward()
    for parameter in (model.model.token_head.weight, model.model.relation_head[0].weight,
                      model.model.blocks[0].qkv.weight, model.model.role_embedding.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    if model.ema is not None:
        assert all(not p.requires_grad and p.grad is None for p in model.ema.parameters())
        assert observed['teacher_forwards'] == 1


@pytest.mark.parametrize('variant', VARIANTS)
def test_changing_target_labels_cannot_change_understanding_inputs_or_predictions(variant):
    model = system(variant).model.eval()
    examples = make_examples(3, seed=3, lengths=(4, 8))
    target = examples.clean.roles.eq(TARGET_ROLE)
    tokens = examples.clean.tokens.clone()
    tokens[target] = (tokens[target] + 7) % 16
    altered = replace(examples, clean=replace(examples.clean, tokens=tokens),
                      token_labels=examples.token_labels.masked_fill(target, 0),
                      relation_labels=(examples.relation_labels + 1) % 4)
    original_batch, altered_batch = corrupt(examples, seed=2), corrupt(altered, seed=2)
    assert torch.equal(original_batch.understanding.tokens, altered_batch.understanding.tokens)
    with torch.no_grad():
        original = model(original_batch.understanding, 'understanding')['relation_logits']
        changed = model(altered_batch.understanding, 'understanding')['relation_logits']
    torch.testing.assert_close(original, changed, atol=0, rtol=0)


@pytest.mark.parametrize('variant', VARIANTS)
def test_whole_segment_permutation_and_padding_preserve_semantic_outputs(variant):
    model = system(variant).model.eval()
    examples = make_examples(4, seed=8, lengths=(4, 8))
    value = corrupt(examples, seed=9).generation
    indices, inverse = segment_permutation(examples.segment_width)
    changed_padding = replace(value,
        tokens=value.tokens.masked_fill(~value.valid, 7),
        positions=value.positions.masked_fill(~value.valid, 31))
    with torch.no_grad():
        original = model(value, 'generation')
        permuted = model(value.permuted(indices), 'generation')
        padded = model(changed_padding, 'generation')
    torch.testing.assert_close(original['token_logits'], permuted['token_logits'][:, inverse], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(original['relation_logits'], permuted['relation_logits'], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(original['token_logits'][value.valid], padded['token_logits'][value.valid], atol=0, rtol=0)


@pytest.mark.parametrize('variant', ('g04_bottleneck4', 'g05_bottleneck16'))
def test_bottleneck_blocks_direct_exchange_but_allows_two_layer_exchange(variant):
    model = system(variant).model.eval()
    original = batch(2).generation
    tokens = original.tokens.clone()
    other_role = original.roles.eq(2)
    tokens[other_role] = (tokens[other_role] + 5) % 16
    changed = replace(original, tokens=tokens)
    with torch.no_grad():
        left = model(original, 'generation')
        right = model(changed, 'generation')
    chosen = original.roles.eq(1)
    torch.testing.assert_close(left['layers'][0][chosen], right['layers'][0][chosen], atol=0, rtol=0)
    assert (left['layers'][1][chosen] - right['layers'][1][chosen]).abs().max() > 1e-5


def test_all_methods_start_from_identical_shared_parameters():
    base = system(VARIANTS[0]).model
    names = dict(base.named_parameters())
    for variant in VARIANTS[1:]:
        candidate = dict(system(variant).model.named_parameters())
        for name, parameter in names.items():
            if name != 'registers':
                torch.testing.assert_close(parameter, candidate[name], atol=0, rtol=0)


@pytest.mark.parametrize('variant', ('g00_shared_joint', 'g06_self_alignment'))
def test_three_three_one_accumulation_matches_full_record_mean_and_gradients(variant):
    full = system(variant)
    accumulated = deepcopy(full)
    value = batch(7)
    loss, _ = full.objective(value, window_records=7)
    with torch.no_grad():
        classification = full.model(value.understanding, 'understanding')['relation_logits']
        generation = full.model(value.generation, 'generation')['token_logits']
        per_record = []
        for row in range(7):
            selected = value.selected[row]
            per_record.append(F.cross_entropy(generation[row, selected], value.token_labels[row, selected]))
        reference = F.cross_entropy(classification, value.relation_labels) + torch.stack(per_record).mean()
        if full.ema is not None:
            teacher = full.ema(value.teacher, 'generation')['layers'][3]
            student = full.alignment_projection(full.model(value.generation, 'generation')['layers'][1])
            individual = []
            for row in range(7):
                selected = value.selected[row]
                individual.append((1 - F.cosine_similarity(student[row, selected], teacher[row, selected], dim=-1)).mean())
            reference = reference + 0.1 * torch.stack(individual).mean()
    torch.testing.assert_close(loss.detach(), reference, atol=1e-6, rtol=1e-6)
    loss.backward()
    for begin, end in ((0, 3), (3, 6), (6, 7)):
        contribution, _ = accumulated.objective(value.rows(begin, end), window_records=7)
        contribution.backward()
    for (name, parameter), (other_name, other) in zip(full.named_parameters(), accumulated.named_parameters()):
        assert name == other_name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-6, rtol=2e-5)


def test_zero_alignment_control_preserves_teacher_compute_and_isolation():
    aligned, control = system('g06_self_alignment'), system('g07_alignment_compute_control')
    value = batch()
    left, lreceipt = aligned.objective(value, window_records=7)
    right, rreceipt = control.objective(value, window_records=7)
    assert lreceipt['teacher_forwards'] == rreceipt['teacher_forwards'] == 1
    assert lreceipt['alignment_contribution'] == pytest.approx(rreceipt['alignment_contribution'], abs=1e-7)
    assert float((left - right).detach()) == pytest.approx(0.1 * lreceipt['alignment_contribution'], abs=1e-6)
    assert sum(p.numel() for p in aligned.parameters()) == sum(p.numel() for p in control.parameters())
    old = next(aligned.ema.parameters()).clone()
    with torch.no_grad():
        next(aligned.model.parameters()).add_(1.0)
    aligned.update_ema()
    torch.testing.assert_close(next(aligned.ema.parameters()), old + 0.01, atol=2e-7, rtol=1e-6)


def test_iterative_generation_commits_all_positions_in_four_forwards_from_inputs_only():
    class ArithmeticOracle:
        def __init__(self):
            self.calls = 0

        def __call__(self, inputs, task):
            self.calls += 1
            logits = torch.zeros(*inputs.tokens.shape, 16)
            for row in range(inputs.tokens.shape[0]):
                for index in torch.where(inputs.roles[row].eq(3))[0]:
                    position = inputs.positions[row, index]
                    a = inputs.tokens[row, inputs.roles[row].eq(1) & inputs.positions[row].eq(position)][0]
                    b = inputs.tokens[row, inputs.roles[row].eq(2) & inputs.positions[row].eq(position)][0]
                    logits[row, index, int((a + b) % 16)] = 10
            return {'token_logits': logits}

    value = batch(5)
    oracle = ArithmeticOracle()
    generated, trace = experiment.iterative_generate(oracle, value.generation)
    target = value.generation.roles.eq(3)
    assert oracle.calls == 4 and len(trace) == 4
    assert trace[-1]['remaining_tokens'] == 0
    assert sum(item['committed_tokens'] for item in trace) == int(target.sum())
    assert generated[target].eq(value.token_labels[target]).all()


def test_fixed_plan_rejects_real_data_flag_before_model_or_output(tmp_path, monkeypatch):
    plan, _ = experiment.checked_plan(VARIANTS[0])
    plan['scope']['real_corpus_allowed'] = True
    (tmp_path/(VARIANTS[0] + '.yml')).write_text(yaml.safe_dump(plan))
    monkeypatch.setattr(experiment, 'PLAN_DIRECTORY', tmp_path)
    with pytest.raises(ValueError, match='artificial-only CPU contract'):
        experiment.checked_plan(VARIANTS[0])


def test_evaluation_packets_recount_token_metrics_and_disjoint_length_groups():
    result = experiment.evaluate(system('g00_shared_joint'), seed=17, records=7, lengths=(4, 8, 12))
    groups = result['groups']
    assert sum(groups['length_' + str(n)]['records'] for n in (4, 8, 12)) == 7
    assert groups['distractor']['records'] + groups['no_distractor']['records'] == 7
    for level, packets in result['mask_packets'].items():
        logits = torch.tensor([v for row in packets for v in row['logits']])
        labels = torch.tensor([v for row in packets for v in row['labels']])
        metric = result['mask_results'][level]
        assert metric['tokens'] == labels.numel()
        assert metric['correct'] == int(logits.argmax(-1).eq(labels).sum())
        assert metric['ce_sum'] == pytest.approx(float(F.cross_entropy(logits.double(), labels, reduction='sum')), abs=1e-5)
        assert sum(groups['length_' + str(n)]['mask_results'][level]['tokens'] for n in (4, 8, 12)) == metric['tokens']
    assert result['iterative_generation_tokens'] == sum(len(p['labels']) for p in result['generation_packets'])
    assert result['generation_trace'][-1]['remaining_tokens'] == 0
