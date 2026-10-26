"""Dispatch frozen p02 experiments through the existing single-GPU workers.

Activate pllm. Called by the shared run_immune_research.py entry; prepare(argv)
is a read-only launch check. This module never submits or restarts resources.
Unrelated aliases delegate to the unchanged, SHA-bound p01 dispatcher.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
OWN = ROOT / 'output/v7_context_ablation_20261002'
POOL = ROOT / 'output/immune_research_pool_c20250601_20260930'
DATA = ROOT / 'output/v7_architecture_pilot_20261002/sample_context_plus20k'
FREEZE = OWN / 'source_snapshot_v1.json'
FREEZE_SHA = '0a88b6a1b38b85fdaa9491444286da6b1bd023c721ce4828d7f5ca0b0b0e53ab'
PREVIOUS = ROOT / 'scripts/research/pool_v7_pilot_dispatch.py'
PREVIOUS_SHA = '2e3ea24fe3992170e36a1b52548031bd234705c818647771c3b2306e42bf44e4'
METHODS = ('f000', 'f001', 'f010', 'f011', 'f100', 'f101', 'f110', 'f111',
           'cv_segment_global', 'cv_late_fusion', 'fixed_gate_control')
ALIASES = {f'p02_v7_{variant}_seed{seed}': (variant, seed)
           for variant in METHODS for seed in (42, 43)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--variant', choices=ALIASES, required=True)
    parser.add_argument('--mode', choices=('train',), required=True)
    parser.add_argument('--steps', type=int, choices=(3750,), required=True)
    for name in ('data-dir', 'profile-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.data_dir != DATA or args.profile_dir != POOL / 'profiles' / args.variant
            or args.output_dir.parent != POOL / 'runs'
            or not re.fullmatch(r'[0-9]+_' + re.escape(args.variant) + r'_[0-9a-f]{8}', args.output_dir.name)):
        raise ValueError('Require the registered subset and existing pool output paths')
    if sha(FREEZE) != FREEZE_SHA:
        raise ValueError('Frozen source manifest differs')
    frozen = json.loads(FREEZE.read_text())
    snapshot = Path(frozen['snapshot_root'])
    if (snapshot != OWN / 'source_snapshot_v1'
            or frozen['study'] != 'v7_context_factorial_120k_v1' or frozen['horizon'] != 3750):
        raise ValueError('Unexpected study source or schedule')
    for name, expected in frozen['source_sha256'].items():
        path = snapshot / name
        if path.is_symlink() or not path.resolve().is_relative_to(snapshot) or sha(path) != expected:
            raise ValueError('Frozen source changed: ' + name)
    for name, expected in frozen['evidence_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError('Bound compatibility evidence changed: ' + name)
    if sha(DATA / 'sample_manifest.json') != frozen['sample_manifest_sha256']:
        raise ValueError('Registered data manifest changed')
    variant, seed = ALIASES[args.variant]
    command = [sys.executable, '-I', '-B', '-u', str(snapshot / 'scripts/research/run_v7_context_ablation.py'),
               '--variant', variant, '--seed', str(seed), '--steps', '3750',
               '--data-dir', str(DATA), '--output-dir', str(args.output_dir)]
    return args, command


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == '--variant']
    selected += [value.split('=', 1)[1] for value in argv if value.startswith('--variant=')]
    if len(selected) != 1 or selected[0] not in ALIASES:
        if sha(PREVIOUS) != PREVIOUS_SHA:
            raise ValueError('Previous dispatcher source changed')
        return runpy.run_path(str(PREVIOUS))['main'](argv)
    args, command = prepare(argv)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not re.fullmatch('[0-7]', visible):
        raise ValueError('Require one GPU assigned by the existing worker')
    (OWN / 'dispatch').mkdir(exist_ok=True)
    with (OWN / 'dispatch' / (args.output_dir.name + '.json')).open('x') as handle:
        json.dump({'started': time.time(), 'job_id': args.output_dir.name,
                   'alias': args.variant, 'visible_gpu': visible, 'command': command,
                   'source_freeze_sha256': FREEZE_SHA, 'formal_training': False}, handle, indent=2)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    raise SystemExit(main())
