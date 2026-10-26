"""CPU functional acceptance for the eight generic integer-object prototypes.

Activate conda pllm, then run from the project directory:
USE_TF=0 CUDA_VISIBLE_DEVICES='' python -m scripts.architecture_prototypes.generic_joint_experiment
--variant g00_shared_joint --cpu-steps 3 --output-dir
/project_c20250601/251105016/project/dllm_test/output/generic_cv_joint_cpu_example
This bounded CPU entry accepts no external data, checkpoint, weights, or GPU flag.
It writes small evidence only; it is not the planned 500-step method comparison.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .generic_joint_data import (ALPHABET, MASK, TARGET_ROLE, MASK_LEVELS, corrupt,
                                 make_examples, remove_condition, segment_permutation)
from .generic_joint_model import JointModel, ModelConfig, VARIANTS
from .record_loss import record_normalized_ce


ROOT = Path(__file__).resolve().parents[2]
PLAN_DIRECTORY = ROOT/'train_jobs/immune_cv_joint_exploration_20261001'
EXPECTED_METHODS = {
    'g00_shared_joint': ('shared_joint', {}),
    'g01_task_norm': ('task_norm', {'task_specific_final_norm': True}),
    'g02_late_adapters': ('late_adapters', {'adapter_rank': 32, 'last_layers': 2}),
    'g03_register4': ('register', {'global_registers': 4}),
    'g04_bottleneck4': ('bottleneck', {'global_registers': 4, 'cross_segment_attention': 'registers_only'}),
    'g05_bottleneck16': ('bottleneck', {'global_registers': 16, 'cross_segment_attention': 'registers_only'}),
    'g06_self_alignment': ('self_alignment', {'ema_decay': 0.99, 'alignment_weight': 0.1,
        'student_layer': 2, 'teacher_layer': 4, 'teacher_mask_ratio_multiplier': 0.5}),
    'g07_alignment_compute_control': ('self_alignment', {'ema_decay': 0.99, 'alignment_weight': 0.0,
        'student_layer': 2, 'teacher_layer': 4, 'teacher_mask_ratio_multiplier': 0.5}),
}


def checked_plan(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown fixed generic prototype')
    path = PLAN_DIRECTORY/(variant + '.yml')
    if path.is_symlink() or path != path.resolve():
        raise ValueError('Use the existing canonical fixed plan')
    plan = yaml.safe_load(path.read_text())
    expected_scope = {'data_kind': 'artificial_integer_objects_only', 'real_corpus_allowed': False,
        'biological_quality_claim': False, 'external_pretrained_encoder': False,
        'standalone_cuda_operator_test': False}
    method, options = EXPECTED_METHODS[variant]
    if (plan['schema'] != 'generic.multisegment_joint_experiment_plan.v1'
            or plan['variant'] != variant or plan['execution_enabled'] is not False
            or plan['scope'] != expected_scope or plan['method']['name'] != method
            or plan['method']['options'] != options
            or plan['training_plan']['checkpoint_save'] is not False
            or plan['prototype_model'] != {'d_model': 192, 'layers': 4, 'heads': 6,
                'alphabet_size': 16, 'shared_trunk': True,
                'joint_outputs': ['relation_classification', 'masked_token_prediction']}):
        raise ValueError('Plan differs from the fixed artificial-only CPU contract')
    return plan, ModelConfig(variant=variant)


class JointSystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = JointModel(config)
        self.ema = None
        self.alignment_projection = None
        self.alignment_weight = 0.0
        if config.variant in ('g06_self_alignment', 'g07_alignment_compute_control'):
            self.ema = deepcopy(self.model).requires_grad_(False).eval()
            self.alignment_projection = nn.Sequential(nn.LayerNorm(config.d_model),
                                                       nn.Linear(config.d_model, config.d_model, bias=False))
            self.alignment_weight = 0.1 if config.variant == 'g06_self_alignment' else 0.0

    def train(self, mode=True):
        super().train(mode)
        if self.ema is not None:
            self.ema.eval()
        return self

    @torch.no_grad()
    def update_ema(self):
        if self.ema is not None:
            for target, source in zip(self.ema.parameters(), self.model.parameters()):
                target.lerp_(source.detach(), 0.01)

    def objective(self, batch, *, window_records):
        understanding = self.model(batch.understanding, 'understanding')
        generation = self.model(batch.generation, 'generation')
        relation_per_record = F.cross_entropy(understanding['relation_logits'].float(),
                                             batch.relation_labels, reduction='none')
        generation_loss = record_normalized_ce(generation['token_logits'], batch.token_labels,
            batch.selected, batch.generation.valid, global_record_count=window_records)
        relation = relation_per_record.sum() / window_records
        alignment = relation.new_zeros(())
        teacher_forwards = 0
        if self.ema is not None:
            with torch.no_grad():
                teacher = self.ema(batch.teacher, 'generation')['layers'][3]
            student = self.alignment_projection(generation['layers'][1])
            distances = 1.0 - F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1)
            per_record = (distances * batch.selected).sum(1) / batch.selected.sum(1)
            alignment = per_record.sum() / window_records
            teacher_forwards = 1
        # G07 executes the same auxiliary graph with weight zero, rather than
        # silently removing the teacher pass or changing the joint objectives.
        loss = relation + generation_loss.contribution + self.alignment_weight * alignment
        return loss, {'relation_contribution': float(relation.detach()),
            'generation_contribution': float(generation_loss.contribution.detach()),
            'alignment_contribution': float(alignment.detach()), 'teacher_forwards': teacher_forwards,
            'selected_tokens': int(batch.selected.sum()), 'records': batch.records}


@torch.no_grad()
def iterative_generate(model, inputs, rounds=4):
    if rounds != 4:
        raise ValueError('The generic comparison fixes four generation forwards')
    view = replace(inputs, tokens=inputs.tokens.clone())
    initial_target = inputs.roles.eq(TARGET_ROLE) & inputs.valid
    view = replace(view, tokens=view.tokens.masked_fill(initial_target, MASK))
    trace = []
    for index in range(rounds):
        logits = model(view, 'generation')['token_logits'].float()
        probabilities = logits.softmax(-1)
        confidence, predictions = probabilities.max(-1)
        pending = initial_target & view.tokens.eq(MASK)
        committed = 0
        for row in range(view.tokens.shape[0]):
            positions = torch.where(pending[row])[0]
            remaining = positions.numel()
            number = (remaining + rounds - index - 1) // (rounds - index)
            selected = positions[torch.argsort(confidence[row, positions], descending=True)[:number]]
            view.tokens[row, selected] = predictions[row, selected]
            committed += selected.numel()
        trace.append({'forward': index + 1, 'committed_tokens': committed,
                      'remaining_tokens': int((initial_target & view.tokens.eq(MASK)).sum())})
    return view.tokens, trace


@torch.no_grad()
def evaluate(system, *, seed=91, records=8, lengths=(8, 16, 24)):
    system.eval()
    examples = make_examples(records, seed=seed, split='validation', lengths=lengths)
    batch = corrupt(examples, seed=seed, mask_levels=(1.0,))
    understanding = system.model(batch.understanding, 'understanding')
    relation_matches = understanding['relation_logits'].argmax(-1).eq(batch.relation_labels)
    relation_correct = int(relation_matches.sum())
    actual_lengths = batch.generation.roles.eq(TARGET_ROLE).sum(1)
    row_groups = {'length_' + str(n): actual_lengths.eq(n) for n in lengths}
    has_distractor = batch.generation.roles.eq(4).any(1)
    row_groups.update(distractor=has_distractor, no_distractor=~has_distractor)
    groups = {name: {'records': int(rows.sum()), 'relation_correct': int(relation_matches[rows].sum()),
                      'mask_results': {}} for name, rows in row_groups.items()}
    mask_results = {}
    mask_packets = {}
    for level in MASK_LEVELS:
        masked = corrupt(examples, seed=seed, mask_levels=(level,))
        output = system.model(masked.generation, 'generation')
        chosen = output['token_logits'][masked.selected].float()
        labels = masked.token_labels[masked.selected]
        # Retain the actual FP32 logits in packets, but compute reporting CE in
        # FP64 so different reduction paths do not lose one FP32 sum ULP.
        # The training objective above continues to use the original FP32 CE.
        losses = F.cross_entropy(chosen.double(), labels, reduction='none')
        ce = losses.sum()
        correct = int(chosen.argmax(-1).eq(labels).sum())
        count = labels.numel()
        mask_results[str(level)] = {'ce_sum': float(ce), 'tokens': count,
            'correct': correct, 'token_ce': float(ce) / count, 'token_accuracy': correct / count}
        mask_packets[str(level)] = [{'row': row,
            'logits': output['token_logits'][row, masked.selected[row]].float().tolist(),
            'labels': masked.token_labels[row, masked.selected[row]].tolist()}
            for row in range(records)]
        selected_rows = torch.arange(records)[:, None].expand_as(masked.selected)[masked.selected]
        for name, rows in row_groups.items():
            selected_group = rows[selected_rows]
            groups[name]['mask_results'][str(level)] = {
                'ce_sum': float(losses[selected_group].sum()), 'tokens': int(selected_group.sum()),
                'correct': int(chosen.argmax(-1).eq(labels)[selected_group].sum())}
    generated, trace = iterative_generate(system.model, batch.generation)
    removed, removed_trace = iterative_generate(system.model, remove_condition(batch.generation))
    target = examples.clean.roles.eq(TARGET_ROLE) & examples.clean.valid
    generated_correct = int(generated[target].eq(examples.token_labels[target]).sum())
    removed_correct = int(removed[target].eq(examples.token_labels[target]).sum())
    target_count = int(target.sum())
    for name, rows in row_groups.items():
        selected_group = target & rows[:, None]
        groups[name].update(iterative_generation_tokens=int(selected_group.sum()),
            iterative_generation_correct=int(generated[selected_group].eq(examples.token_labels[selected_group]).sum()),
            condition_removed_generation_correct=int(removed[selected_group].eq(examples.token_labels[selected_group]).sum()))
    indices, inverse = segment_permutation(examples.segment_width)
    original = system.model(batch.generation, 'generation')
    permuted = system.model(batch.generation.permuted(indices), 'generation')
    difference = float((original['token_logits'] - permuted['token_logits'][:, inverse]).abs().max())
    relation_difference = float((original['relation_logits'] - permuted['relation_logits']).abs().max())
    variance = float(original['features'][batch.generation.valid].float().var(dim=0, unbiased=False).mean())
    return {'relation_correct': relation_correct, 'relation_records': records,
        'relation_accuracy': relation_correct / records, 'mask_results': mask_results,
        'iterative_generation_correct': generated_correct, 'iterative_generation_tokens': target_count,
        'iterative_generation_accuracy': generated_correct / target_count,
        'condition_removed_generation_correct': removed_correct,
        'condition_removed_generation_gap': (generated_correct - removed_correct) / target_count,
        'condition_removed_interpretation': 'dependency_diagnostic_with_original_labels_not_counterfactual_accuracy',
        'segment_permutation_logit_difference': difference,
        'segment_permutation_relation_difference': relation_difference,
        'representation_variance': variance, 'generation_trace': trace,
        'condition_removed_trace': removed_trace, 'lengths': list(lengths),
        'groups': groups,
        'relation_packet': {'logits': understanding['relation_logits'].float().tolist(),
                            'labels': batch.relation_labels.tolist()},
        'mask_packets': mask_packets,
        'generation_packets': [{'row': row, 'length': int(actual_lengths[row]),
            'has_distractor': bool(has_distractor[row]),
            'labels': examples.token_labels[row, target[row]].tolist(),
            'generated': generated[row, target[row]].tolist(),
            'condition_removed': removed[row, target[row]].tolist()} for row in range(records)],
        'scope': 'untrained_or_short_CPU_functional_metrics_not_method_ranking'}


def run_cpu(variant, steps, output_dir):
    if (type(steps) is not int or not 1 <= steps <= 20
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized()):
        raise ValueError('Use bounded CPU-only functional steps with CUDA_VISIBLE_DEVICES empty')
    plan, config = checked_plan(variant)
    output = Path(output_dir)
    if (not output.is_absolute() or output != output.resolve() or output.exists()
            or output.parent != ROOT/'output' or not output.name.startswith('generic_cv_joint_cpu_')):
        raise ValueError('Use a new canonical project/output/generic_cv_joint_cpu_* directory')
    torch.set_num_threads(2)
    source_names = ('generic_joint_data.py', 'generic_joint_model.py', 'generic_joint_experiment.py', 'record_loss.py')
    sources = {str((Path(__file__).parent/name).relative_to(ROOT)):
               hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in source_names}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        system = JointSystem(config)
    output.mkdir()
    (output/'plan_snapshot.yml').write_text(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True))
    optimizer = torch.optim.AdamW((p for p in system.parameters() if p.requires_grad), lr=3e-4)
    initial = evaluate(system)
    windows = []
    start = time.monotonic()
    for step in range(steps):
        system.train()
        batch = corrupt(make_examples(8, seed=42 + step), seed=142 + step)
        optimizer.zero_grad(set_to_none=True)
        contributions = []
        for begin in range(0, batch.records, 3):
            micro = batch.rows(begin, min(begin + 3, batch.records))
            loss, receipt = system.objective(micro, window_records=batch.records)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite artificial joint loss')
            loss.backward()
            contributions.append(receipt)
        gradients = [p.grad for p in system.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Missing or nonfinite artificial gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0))
        tracked_names = ('token_head.weight', 'relation_head.0.weight', 'blocks.0.qkv.weight')
        tracked = {name: parameter for name, parameter in system.model.named_parameters()
                   if name in tracked_names}
        before = {name: parameter.detach().clone() for name, parameter in tracked.items()}
        optimizer.step()
        system.update_ema()
        changes = {name: float((parameter.detach() - before[name]).abs().max())
                   for name, parameter in tracked.items()}
        if any(change <= 0 for change in changes.values()):
            raise ValueError('A tracked shared or output parameter did not update')
        state_bytes = sum(tensor.numel() * tensor.element_size()
                          for state in optimizer.state.values() for tensor in state.values()
                          if isinstance(tensor, torch.Tensor))
        windows.append({'step': step + 1, 'actual_records': batch.records,
                        'micro_sizes': [v['records'] for v in contributions],
                        'preclip_gradient_norm': norm, 'parameter_max_abs_changes': changes,
                        'AdamW_state_parameters': len(optimizer.state),
                        'AdamW_state_bytes': state_bytes, 'contributions': contributions})
    elapsed = time.monotonic() - start
    final = evaluate(system)
    parameter_counts = {'trainable_parameters': sum(p.numel() for p in system.parameters() if p.requires_grad),
        'total_parameters_including_teacher': sum(p.numel() for p in system.parameters()),
        'teacher_parameters': 0 if system.ema is None else sum(p.numel() for p in system.ema.parameters())}
    result = {'schema': 'generic.joint_cpu_functional.v1', 'variant': variant,
        'status': 'cpu_functional_completed_not_GPU_or_method_acceptance',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sources,
        'model': config.__dict__, 'optimizer_windows': windows,
        'actual_records': sum(v['actual_records'] for v in windows),
        'actual_microsteps': sum(len(v['micro_sizes']) for v in windows),
        'optimizer_steps': steps, 'initial': initial, 'final': final,
        'parameter_counts': parameter_counts, 'CPU_training_seconds': elapsed,
        'gpu_seconds': None, 'GPU_memory': None, 'GPU_executed': False,
        'real_corpus_training_started': False, 'checkpoint_files': [],
        'biological_quality_claim': False, 'planned_500_step_method_experiment_completed': False}
    (output/'CPU_FUNCTIONAL.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--cpu-steps', type=int, default=3)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    result = run_cpu(args.variant, args.cpu_steps, args.output_dir)
    print(json.dumps({key: result[key] for key in
                     ('variant', 'status', 'optimizer_steps', 'actual_records', 'CPU_training_seconds')}))


if __name__ == '__main__':
    main()
