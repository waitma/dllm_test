"""Freeze a two-segment LoRA continuation to the first balanced coverage epoch.

Activate pllm/protenix_abtcr; python -m scripts.research.prepare_post_training_epoch_continuation
    --name fixed_chain_lora_epoch1_20261015 --parent-checkpoint ABS --base-bundle ABS
Preparation verifies the immutable parent and CPU continuation tests. It does not enqueue.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from scripts.research.post_training_continuation import (
    require, sha, digest, coverage_steps, read_plan, validate_binding)

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--parent-checkpoint', type=Path, required=True)
    parser.add_argument('--base-bundle', type=Path, required=True)
    args = parser.parse_args()
    require(re.fullmatch('[a-z0-9_]+', args.name) is not None, 'Invalid run name')
    parent, base = args.parent_checkpoint.resolve(), args.base_bundle.resolve()
    tracking = ROOT / 'output' / args.name
    output = ROOT / 'output' / ('immune_' + args.name)
    bundles = [ROOT / 'output/v5_full_binding_20261007' / ('launch_' + args.name + f'_stage{i}')
               for i in (1, 2)]
    require(not any(p.exists() for p in [tracking, output, *bundles]), 'Preparation already exists')
    prior = json.loads((parent / 'run_provenance.json').read_text())
    sidecar = json.loads((parent / 'formal_checkpoint.json').read_text())
    require(sidecar['world_size'] == 8 and sidecar['rank_acknowledgments'] == list(range(8))
            and sidecar['binding_sha256'] == digest(prior), 'Incomplete parent checkpoint')
    require({p.name for p in parent.iterdir()} == set(sidecar['files']) | {'formal_checkpoint.json'},
            'Parent checkpoint inventory differs')
    for name, expected in sidecar['files'].items():
        require(sha(parent / name) == expected, 'Parent checkpoint changed: ' + name)
    # Do not import contemporary model/data changes into the completed experiment.
    old_source = Path(prior['source_root'])
    for name, expected in prior['source_files'].items():
        require(sha(name) == expected, 'Original frozen source changed: ' + name)
    index = json.loads(Path(prior['post_training']['full_data_index']).read_text())
    source_lengths = {key: value['retained_rows'] for key, value in index['files'].items()}
    end = coverage_steps(list(source_lengths.values()), prior['global_batch'])
    tracking.mkdir()
    plan = {'schema': 'fixed_chain.post_training.coverage_continuation.v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'parent_checkpoint': str(parent), 'parent_sidecar_sha256': sha(parent / 'formal_checkpoint.json'),
        'parent_complete_sha256': sha(parent.parent / 'COMPLETE.json'),
        'parent_binding_sha256': digest(prior), 'output_dir': str(output),
        'start_step': sidecar['step'], 'end_step': end, 'source_lengths': source_lengths,
        'global_batch': prior['global_batch'], 'warmup_steps': 200,
        'learning_rate': prior['schedule']['learning_rate'],
        'scheduler': 'continuation_warmup_cosine',
        'data_progress': 'Native Trainer skips previous cumulative batches in the unchanged deterministic sampler',
        'authorization': 'User requested completing the full LoRA epoch on the existing eight-GPU container'}
    plan_path = tracking / 'continuation_plan.json'
    write(plan_path, plan)
    read_plan(plan_path, sha(plan_path))
    source = bundles[0] / 'source'
    shutil.copytree(old_source, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('scripts/research/run_fixed_chain_post_training_formal.py',
                 'scripts/research/post_training_continuation.py',
                 'scripts/tests/immune_llada/test_post_training_continuation.py'):
        shutil.copy2(ROOT / name, source / name)
    for bundle in bundles:
        target = bundle / 'source/scripts/research/launch_v5_direct.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'scripts/research/launch_fixed_chain_post_training_formal.py', target)
    files = {str(p): sha(p) for p in source.rglob('*.py')}
    files.update({name: expected for name, expected in prior['source_files'].items()
                  if not Path(name).is_relative_to(old_source)})
    files[str(bundles[1] / 'source/scripts/research/launch_v5_direct.py')] = sha(
        bundles[1] / 'source/scripts/research/launch_v5_direct.py')
    manifest = bundles[0] / 'source_manifest.json'
    write(manifest, {'files': files})
    proposed = {**prior, 'source_root': str(source), 'source_files': files,
        'source_manifest_sha256': sha(manifest),
        'schedule': {**prior['schedule'], 'max_steps': end, 'warmup_steps': 200,
                     'scheduler': 'continuation_warmup_cosine', 'start_step': sidecar['step']}}
    validate_binding(plan, proposed, prior)
    old_launch = json.loads((base / 'direct_launch.json').read_text())
    argv = json.loads((base / 'launch_training_args.json').read_text())
    require(sha(base / 'launch_training_args.json') == old_launch['arguments_sha256'], 'Old launch changed')
    require(Path(argv[argv.index('--source-root')+1]) == old_source, 'Wrong base launch')
    changes = {'--source-root': source, '--source-manifest': manifest,
        '--source-manifest-sha256': sha(manifest), '--output-dir': output,
        '--max-steps': end, '--warmup-steps': 200}
    for key, value in changes.items():
        argv[argv.index(key)+1] = str(value)
    argv += ['--continuation-plan', str(plan_path), '--continuation-plan-sha256', sha(plan_path)]
    checkpoint_step = sidecar['step'] + 50
    tickets = []
    for stage, bundle in enumerate(bundles, 1):
        stage_args = list(argv)
        stage_args[stage_args.index('--stop-after')+1] = str(checkpoint_step if stage == 1 else 0)
        stage_args[stage_args.index('--resume-from-checkpoint')+1] = str(
            parent if stage == 1 else output / f'checkpoint-{checkpoint_step}')
        argfile = bundle / 'launch_training_args.json'
        write(argfile, stage_args)
        launcher = bundle / 'source/scripts/research/launch_v5_direct.py'
        launch = {**old_launch, 'schema': 'fixed_chain.coverage_continuation.launch.v1',
            'runtime_dir': str(tracking / f'stage{stage}_runtime'),
            'arguments_sha256': sha(argfile), 'launcher_sha256': sha(launcher),
            'initialization': 'native_full_state_resume_with_recorded_schedule_extension',
            'max_steps': end, 'purpose': 'Continue the same LoRA data stream through one full balanced epoch',
            'continuation_plan': str(plan_path), 'continuation_plan_sha256': sha(plan_path)}
        write(bundle / 'direct_launch.json', launch)
        tickets.append({'bundle': str(bundle), 'launcher_sha256': sha(launcher), 'purpose': launch['purpose']})
    test_paths = [source / 'scripts/tests/immune_llada' / name for name in (
        'test_post_training_continuation.py', 'test_fixed_chain_formal_post_training.py')]
    tests = subprocess.run([sys.executable, '-I', '-B', '-c',
        "import sys;sys.path.insert(0,sys.argv[1]);import pytest;raise SystemExit(pytest.main(sys.argv[2:]+['-q']))",
        str(source), *map(str, test_paths)], cwd='/tmp', capture_output=True, text=True)
    (tracking / 'isolated_tests.log').write_text(tests.stdout + tests.stderr)
    require(tests.returncode == 0, 'Frozen continuation tests failed')
    check = subprocess.run([sys.executable, '-I', '-B',
        str(source / 'scripts/research/launch_v5_direct.py'), '--bundle', str(bundles[0]), '--check-only'],
        cwd='/tmp', capture_output=True, text=True)
    (tracking / 'metadata_check.log').write_text(check.stdout + check.stderr)
    require(check.returncode == 0, 'Continuation metadata validation failed')
    # Explicitly inspect the scheduler boundary without rewriting either state file.
    import torch
    optimizer = torch.load(parent / 'optimizer.pt', map_location='cpu', weights_only=False)
    scheduler = torch.load(parent / 'scheduler.pt', map_location='cpu', weights_only=False)
    require(scheduler['last_epoch'] == sidecar['step'] and all(x == 0 for x in scheduler['_last_lr'])
            and all(g['lr'] == 0 for g in optimizer['param_groups']), 'Parent LR boundary is not zero')
    receipt = {'status': 'prepared_not_enqueued', 'output': str(output), 'plan': str(plan_path),
        'start_step': sidecar['step'], 'target_step': end, 'remaining_steps': end - sidecar['step'],
        'first_verification_step': checkpoint_step, 'tickets': tickets,
        'parent_inventory_verified': True, 'frozen_cpu_tests': 7, 'parent_lr': scheduler['_last_lr'],
        'stage_rule': 'Publish stage2 only after stage1 success and all-rank native restoration evidence'}
    write(tracking / 'prepared.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
