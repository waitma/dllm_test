"""Continue a completed LoRA run without resetting its data or optimizer progress.

Activate pllm/protenix_abtcr; run:
python -m pytest scripts/tests/immune_llada/test_post_training_continuation.py -q
The original checkpoint stays immutable. A new run records the extended schedule.
"""
import hashlib
import json
import math
from functools import partial
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def coverage_steps(source_lengths, batch):
    require(source_lengths and min(source_lengths) > 0 and batch > 0, 'Invalid coverage inputs')
    return math.ceil(max(source_lengths) * len(source_lengths) / batch)


def continuation_factor(step, start, end, warmup):
    """Use the cumulative optimizer step; the completed phase ended at zero LR."""
    require(0 < warmup < end - start, 'Invalid continuation warmup')
    elapsed = step - start
    if elapsed <= 0:
        return 0.0
    if elapsed < warmup:
        return elapsed / warmup
    progress = min(1.0, (elapsed - warmup) / (end - start - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class ContinuationScheduleMixin:
    """Native checkpoint loading still restores optimizer, scheduler and RNG."""
    continuation_plan = None

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.continuation_plan is None:
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is None:
            from torch.optim.lr_scheduler import LambdaLR
            plan = self.continuation_plan
            require(num_training_steps == plan['end_step'], 'Continuation horizon differs')
            # LambdaLR's state remains native and loadable from the prior cosine.
            self.lr_scheduler = LambdaLR(optimizer if optimizer is not None else self.optimizer,
                partial(continuation_factor, start=plan['start_step'], end=plan['end_step'],
                        warmup=plan['warmup_steps']))
            self._created_lr_scheduler = True
        return self.lr_scheduler


def read_plan(path, expected_sha):
    require(sha(path) == expected_sha, 'Continuation plan changed')
    plan = json.loads(Path(path).read_text())
    require(plan['schema'] == 'fixed_chain.post_training.coverage_continuation.v1',
            'Unknown continuation plan')
    parent = Path(plan['parent_checkpoint'])
    require(parent.is_absolute() and parent == parent.resolve(), 'Noncanonical parent checkpoint')
    require(sha(parent / 'formal_checkpoint.json') == plan['parent_sidecar_sha256'],
            'Parent checkpoint sidecar changed')
    require(sha(parent.parent / 'COMPLETE.json') == plan['parent_complete_sha256'],
            'Completed parent receipt changed')
    prior = json.loads((parent / 'run_provenance.json').read_text())
    sidecar = json.loads((parent / 'formal_checkpoint.json').read_text())
    complete = json.loads((parent.parent / 'COMPLETE.json').read_text())
    require(digest(prior) == sidecar['binding_sha256'] == plan['parent_binding_sha256'],
            'Parent binding changed')
    require(complete['status'] == 'completed' and complete['step'] == sidecar['step']
            == prior['schedule']['max_steps'] == plan['start_step'], 'Parent is not completed at the boundary')
    require(prior['post_training']['recipe'] == 'lora', 'Only confirmed LoRA continuation is supported')
    require(plan['end_step'] == coverage_steps(list(plan['source_lengths'].values()), plan['global_batch']),
            'Continuation must finish exactly one balanced coverage epoch')
    require(plan['global_batch'] == prior['global_batch']
            and plan['learning_rate'] == prior['schedule']['learning_rate'], 'Continuation batch/peak LR changed')
    continuation_factor(plan['start_step'], plan['start_step'], plan['end_step'], plan['warmup_steps'])
    return plan, prior


def validate_binding(plan, current, prior):
    """Permit a recorded schedule/source transition while keeping the experiment fixed."""
    changed = {'source_files', 'source_root', 'source_manifest_sha256', 'audit_execution',
               'schedule', 'continuation'}
    require({k: v for k, v in current.items() if k not in changed}
            == {k: v for k, v in prior.items() if k not in changed},
            'Continuation changed model/data/optimizer settings')
    require({k: v for k, v in current['audit_execution'].items() if k != 'source_manifest_sha256'}
            == {k: v for k, v in prior['audit_execution'].items() if k != 'source_manifest_sha256'},
            'Continuation changed the parent data audit')
    require(current['schedule'] == {**prior['schedule'], 'max_steps': plan['end_step'],
            'warmup_steps': plan['warmup_steps'], 'scheduler': 'continuation_warmup_cosine',
            'start_step': plan['start_step']}, 'Unexpected continuation schedule')
    old_root, new_root = Path(prior['source_root']), Path(current['source_root'])
    allowed = {'scripts/research/run_fixed_chain_post_training_formal.py',
               'scripts/research/launch_v5_direct.py'}
    for name, expected in prior['source_files'].items():
        old = Path(name)
        if old.is_relative_to(old_root):
            relative = str(old.relative_to(old_root))
            if relative not in allowed:
                require(current['source_files'].get(str(new_root / relative)) == expected,
                        'Unrelated frozen training source changed: ' + relative)
    index = json.loads(Path(current['post_training']['full_data_index']).read_text())
    lengths = {key: item['retained_rows'] for key, item in index['files'].items()}
    require(lengths == plan['source_lengths'], 'Continuation source counts changed')
