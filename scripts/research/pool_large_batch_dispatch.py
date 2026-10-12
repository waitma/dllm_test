"""Run the bounded batch-capacity study on the next naturally free pool slot.

Activate conda pllm. The shared entry invokes main(argv) before ML imports.
prepare(argv) verifies the immutable code and bounded synthetic envelope only;
it does not start a process, write a plan or admit actual training data.
Other jobs delegate to the unchanged p02 dispatcher. No platform submissions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN = ROOT / 'output/v7_large_batch_ablation_20261002'
POOL = ROOT / 'output/immune_research_pool_c20250601_20260930'
FREEZE = OWN / 'source_snapshot_v1.json'
FREEZE_SHA = '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'
PREVIOUS = ROOT / 'scripts/research/pool_context_ablation_dispatch.py'
PREVIOUS_SHA = '0946d193e7f6e5bb080cc79447a72e09095c543df64fa44b36b9f1e98eb923d6'
ALIAS = 'p04_v7_capacity'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(path):
    require(path.is_absolute() and path == path.resolve()
            and not any(part.is_symlink() for part in (path, *path.parents)), 'Canonical nonsymlink path required')
    return path


def prepare(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--variant', required=True, choices=(ALIAS,))
    parser.add_argument('--mode', required=True, choices=('profile-synthetic',))
    parser.add_argument('--steps', required=True, type=int, choices=(72,))
    parser.add_argument('--profile-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    require(canonical(args.profile_dir) == POOL / 'profiles' / ALIAS
            and canonical(args.output_dir).parent == POOL / 'runs'
            and re.fullmatch(r'[0-9]+_' + ALIAS + r'_[0-9a-f]{8}', args.output_dir.name),
            'Require the existing pool and one explicit capacity job')
    require(sha(canonical(FREEZE)) == FREEZE_SHA, 'Batch study source manifest changed')
    frozen = json.loads(FREEZE.read_text())
    snapshot = canonical(Path(frozen['snapshot_root']))
    require(snapshot == OWN / 'source_snapshot_v1', 'Unexpected source snapshot')
    for name, expected in frozen['source_sha256'].items():
        path = canonical(snapshot / name)
        require(path.is_relative_to(snapshot) and sha(path) == expected, 'Frozen source changed: ' + name)
    for name, expected in frozen['evidence_sha256'].items():
        require(sha(canonical(Path(name))) == expected, 'Bound CPU evidence changed: ' + name)
    command = [sys.executable, '-I', '-B', '-u', str(snapshot / 'scripts/research/profile_v7_large_batch.py'),
               '--source-root', str(snapshot), '--output-dir', str(args.output_dir),
               '--variants', 'f000', 'f111', 'cv_segment_global', '--micro-batches', '4', '16', '32',
               '--global-batch', '32', '--warmup-steps', '2', '--measure-steps', '6',
               '--candidate-seconds', '180', '--max-seconds', '1800', '--seed', '42']
    return args, command


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == '--variant']
    selected += [value.split('=', 1)[1] for value in argv if value.startswith('--variant=')]
    if len(selected) != 1 or selected[0] != ALIAS:
        require(sha(PREVIOUS) == PREVIOUS_SHA, 'Previous p02 dispatcher changed')
        return runpy.run_path(str(PREVIOUS))['main'](argv)
    args, command = prepare(argv)
    require(re.fullmatch('[0-7]', os.environ.get('CUDA_VISIBLE_DEVICES', '')),
            'Use exactly one GPU assigned by the existing pool')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    raise SystemExit(main())
