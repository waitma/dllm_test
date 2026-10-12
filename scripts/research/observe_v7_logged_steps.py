"""Read formal training progress without controlling training or loading weights.

Activate conda pllm, then run:
  python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/observe_v7_logged_steps.py

Only existing plan, receipt, state JSON and complete rank-0 log lines are read.
Saved steps and logged steps remain separate across segment boundaries.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
SUPERVISION = ROOT / 'output/v7_formal_supervision_20261005'
LAUNCHES = ROOT / 'output/v7_formal_eight_gpu_20261002/launch_20261005'
RUN = ROOT / 'output/v7_initial_training_20261005/run'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def learning_rate(step):
    if step < 1000:
        return 1e-4 * step / 1000
    return 1e-4 * 0.5 * (1 + math.cos(math.pi * (step - 1000) / (25042 - 1000)))


def infer_logged_steps(raw, native, prior_step, target_step):
    """Map complete log lines using the verified fixed schedule and saved history."""
    saved_step = native['global_step']
    require(0 <= prior_step <= saved_step <= target_step <= 25042, 'Segment/checkpoint range mismatch')
    require(native['logging_steps'] == 25 and native['max_steps'] == 25042, 'Logging/schedule changed')
    complete = raw[:raw.rfind(b'\n') + 1]
    rows = [ast.literal_eval(line) for line in complete.decode().splitlines()
            if line.startswith("{'loss':")]
    first_step = (prior_step // 25 + 1) * 25
    mapped = dict(zip(range(first_step, first_step + 25 * len(rows), 25), rows))
    require(not mapped or max(mapped) <= target_step, 'Log exceeds segment target')
    history = {row['step']: row for row in native['log_history'] if 'loss' in row}
    require(all(0 < step <= saved_step for step in history), 'Native training history exceeds checkpoint')
    for step, row in list(history.items()) + list(mapped.items()):
        require(math.isfinite(row['loss']) and math.isfinite(row['learning_rate']), 'Nonfinite training log')
        require(math.isclose(row['learning_rate'], learning_rate(step), rel_tol=1e-12, abs_tol=1e-16),
                'Missing/duplicated log or schedule mismatch at step ' + str(step))
    matched = []
    for step, row in history.items():
        if step <= prior_step:
            continue
        require(step in mapped, 'Saved history absent from current segment log')
        require(all(mapped[step][key] == row[key] for key in ('loss', 'epoch', 'learning_rate')),
                'Native history disagrees with current segment log')
        matched.append(step)
    if mapped:
        latest_step = max(mapped)
        latest = mapped[latest_step]
        source = 'current_segment_complete_log'
    else:
        require(saved_step == prior_step, 'Checkpoint advanced without observable training log')
        latest_step = max(history, default=None)
        latest = history.get(latest_step)
        source = 'saved_native_history' if latest is not None else 'no_training_log_yet'
    return dict(latest_logged_training_step=latest_step, latest_training_log=latest,
                latest_log_source=source, latest_completed_checkpoint_step=saved_step,
                current_segment_training_rows=len(rows), native_anchor_steps_matched=sorted(matched),
                complete_log_bytes=len(complete), complete_log_sha256=hashlib.sha256(complete).hexdigest(),
                all_learning_rates_checked=True)


def read_reference(path):
    require(path.is_file() and not path.is_symlink(), 'Regular metadata file required: ' + str(path))
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def observe():
    state, state_ref = read_reference(SUPERVISION / 'state.json')
    stage = Path(state['stage'])
    require(stage.parent == LAUNCHES and re.fullmatch(r'segment_[0-9]+', stage.name), 'Unknown stage')
    receipt, receipt_ref = read_reference(stage / 'enqueue_receipt.json')
    plan, plan_ref = read_reference(stage / 'plan.json')
    require(receipt['plan'] == plan_ref, 'Enqueued plan identity mismatch')
    prior, target = receipt['prior_step'], receipt['stop_after']
    require(prior == state['prior_step'] and target == state['target_step'], 'Supervisor stage mismatch')
    require(plan['work']['output_dir'] == str(RUN), 'Unexpected run directory')
    argv = plan['work']['driver_argv']
    expected = {'--max-steps': '25042', '--warmup-steps': '1000', '--learning-rate': '0.0001',
                '--stop-after': str(target), '--resume-from-checkpoint': str(RUN / f'checkpoint-{prior}')}
    for option, value in expected.items():
        require(argv.count(option) == 1 and argv[argv.index(option) + 1] == value, 'Plan changed: ' + option)
    checkpoints = sorted((path for path in RUN.glob('checkpoint-*')
                          if (path / 'formal_checkpoint.json').is_file()),
                         key=lambda path: int(path.name.split('-')[-1]))
    require(bool(checkpoints), 'No completed checkpoint metadata')
    checkpoint = checkpoints[-1]
    native, native_ref = read_reference(checkpoint / 'trainer_state.json')
    require(checkpoint.name == 'checkpoint-' + str(native['global_step']), 'Checkpoint name/state mismatch')
    log = stage / 'meeting/coordinator/rank0_child.log'
    raw = log.read_bytes() if log.is_file() else b''
    result = infer_logged_steps(raw, native, prior, target)
    result.update(at_utc=datetime.now(timezone.utc).isoformat(), stage=str(stage), prior_step=prior,
                  target_step=target, log_path=str(log), state=state_ref, plan=plan_ref,
                  receipt=receipt_ref, native_state=native_ref, model_execution=False)
    return result


if __name__ == '__main__':
    print(json.dumps(observe(), ensure_ascii=False, indent=2, allow_nan=False))
