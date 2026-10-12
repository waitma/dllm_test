"""Dispatch the fixed V7 pilot through the already-running single-GPU pool.

Activate pllm; the existing pool invokes this via run_immune_research.py.
Importing this module only defines helpers. It never submits or restarts jobs.
Each pilot uses a frozen model source tree and the same sampled data manifest.
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

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN = ROOT / 'output/v7_architecture_pilot_20261002'
POOL = ROOT / 'output/immune_research_pool_c20250601_20260930'
LEGACY = ROOT / 'scripts/architecture_prototypes/gpu_pool_dispatcher_v3.py'
LEGACY_SHA = 'c1d1c4cf556815cd4dd51796e0db208fb460d97a8e1d4b0bd70e71d7b849f321'
FREEZE = OWN / 'source_snapshot_v1.json'
FREEZE_SHA = '33c12724a5008936416888348782734ccfd28eeae9307983b1eb26e707ccd9d7'
ALIASES = {f'p01_v7_{variant}_seed{seed}': (variant, seed)
           for variant in ('r20', 'r21', 'r24', 'r26') for seed in (42, 43)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--variant', required=True, choices=ALIASES)
    parser.add_argument('--mode', required=True, choices=('train',))
    parser.add_argument('--steps', required=True, type=int, choices=(3125,))
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--profile-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    if (args.data_dir != OWN / 'sample'
            or args.profile_dir != POOL / 'profiles' / args.variant
            or args.output_dir.parent != POOL / 'runs'
            or not re.fullmatch(r'[0-9]+_' + re.escape(args.variant) + r'_[0-9a-f]{8}', args.output_dir.name)):
        raise ValueError('Pilot must use the fixed sample and existing pool output paths')
    if sha(FREEZE) != FREEZE_SHA:
        raise ValueError('Frozen source manifest changed')
    frozen = json.loads(FREEZE.read_text())
    snapshot = Path(frozen['snapshot_root'])
    if snapshot != OWN / 'source_snapshot_v1':
        raise ValueError('Unexpected model source tree')
    for name, expected in frozen['source_sha256'].items():
        path = snapshot / name
        if path.is_symlink() or not path.resolve().is_relative_to(snapshot) or sha(path) != expected:
            raise ValueError('Frozen model source changed: ' + name)
    sample_sha = sha(args.data_dir / 'sample_manifest.json')
    layout = json.loads((OWN / 'sample_layout_preflight.json').read_text())
    parent = json.loads((OWN / 'parent_scope_preflight.json').read_text())
    if (sample_sha != frozen['sample_manifest_sha256']
            or layout.get('status') != 'passed' or layout.get('sample_manifest_sha256') != sample_sha
            or parent.get('status') != 'existing_parent_scope_evidence_verified'):
        raise ValueError('Sample/source compatibility evidence does not match')
    variant, seed = ALIASES[args.variant]
    command = [sys.executable, '-I', '-B', '-u',
               str(snapshot / 'scripts/research/run_v7_architecture_pilot.py'),
               '--variant', variant, '--seed', str(seed), '--steps', '3125',
               '--data-dir', str(args.data_dir), '--output-dir', str(args.output_dir)]
    return args, command


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == '--variant']
    selected += [value.split('=', 1)[1] for value in argv if value.startswith('--variant=')]
    if len(selected) != 1 or selected[0] not in ALIASES:
        if sha(LEGACY) != LEGACY_SHA:
            raise ValueError('Previous dispatcher source changed')
        return runpy.run_path(str(LEGACY))['main'](argv)
    args, command = prepare(argv)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not re.fullmatch('[0-7]', visible):
        raise ValueError('This pilot requires one GPU assigned by the existing worker')
    (OWN / 'dispatch').mkdir(exist_ok=True)
    with (OWN / 'dispatch' / (args.output_dir.name + '.json')).open('x') as handle:
        json.dump(dict(started=time.time(), job_id=args.output_dir.name, alias=args.variant,
                       visible_gpu=visible, command=command, source_freeze_sha256=FREEZE_SHA,
                       formal_training=False), handle, indent=2)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    raise SystemExit(main())
