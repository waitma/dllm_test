"""Connect the new eight-GPU allocation to the existing formal Trainer.

Activate conda pllm before preparing pool tickets. The existing pool invokes
main(original_argv) before ML imports. The resource-identity ticket only reads
its actual parent process. Formal tickets reuse the frozen dispatcher, slot
wrapper, transport, data gate and native Trainer without changing their checks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/v7_formal_eight_gpu_20261002'
POOL = ROOT/'output/immune_research_pool_c20250601_20261005'
BASE = OWN/'pool_launcher_source_v1/pool_formal_dispatch.py'
BASE_SHA = '21f2ffe71d23950207eb4c067b0cb26cf5dbb875a13e2ba15bdec3a7d3f5ce59'
PREVIOUS = ROOT/'output/v7_large_batch_ablation_20261002/launcher_source_v2/scripts/research/pool_large_batch_ablation_v2_dispatch.py'
PREVIOUS_SHA = 'b088ba3c19f58060a05393411cdf57a3abe8ecd7ba91508add62142896476cc3'
OBSERVER = ROOT/'output/v7_large_batch_ablation_20261002/pool_parent_source_v1/pool_parent_runtime_observation.py'
OBSERVER_SHA = '8d13c32c6c8e0b6ee65df4573a14e784947e98602e50c7ef10cf2459393e79c8'
CATALOG = OWN/'pool_job_catalog_20261005.json'
ALIASES = tuple(f'p06_v7_formal_member_{rank}' for rank in range(8))
IDENTITY_ALIAS = 'p06_v7_pool_identity'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound(path, expected):
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink()
            and sha(path) == expected, 'Bound entry source changed: '+str(path))
    result = runpy.run_path(str(path), run_name='_new_pool_bound_dependency')
    require(sha(path) == expected, 'Source changed while importing')
    return result


def formal_api():
    api = load_bound(BASE, BASE_SHA)
    runtime = api['prepare'].__globals__
    runtime.update(POOL=POOL, CATALOG=CATALOG, ALIASES=ALIASES,
                   PREVIOUS=PREVIOUS, PREVIOUS_SHA=PREVIOUS_SHA)
    return runtime


def observe_parent(argv):
    """Capture actual container process ownership; never import a model or torch."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--variant', choices=(IDENTITY_ALIAS,), required=True)
    parser.add_argument('--mode', choices=('profile-synthetic',), required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--profile-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.steps == 1 and args.profile_dir == POOL/'profiles'/IDENTITY_ALIAS,
            'Resource identity ticket envelope differs')
    output = args.output_dir
    require(output.is_absolute() and output == output.resolve() and output.parent == POOL/'runs'
            and re.fullmatch(r'[0-9]+_'+IDENTITY_ALIAS+r'_[0-9a-f]{8}', output.name),
            'Identity output must belong to the new pool')
    expected = dict(variant=IDENTITY_ALIAS, mode='profile-synthetic', steps=1)
    require(json.loads((POOL/'running'/f'{output.name}.json').read_text()) == expected,
            'Identity ticket is not actually held by the pool')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(re.fullmatch('[0-7]', mask) is not None, 'One actual assigned GPU mask required')
    observer = load_bound(OBSERVER, OBSERVER_SHA)
    runtime = observer['validate_parent_argv'].__globals__
    runtime['POOL'] = POOL
    api = observer['coordinator_api']()
    controller = api['process_identity'](os.getpid())
    parent = api['process_identity'](os.getppid())
    observer['validate_identities'](controller, parent)
    cwd = observer['read_parent_cwd'](parent['pid'])
    parent_argv = observer['read_parent_cmdline'](parent['pid'])
    resolved = observer['validate_parent_argv'](parent_argv, cwd)
    require(api['process_identity'](os.getpid()) == controller
            and api['process_identity'](os.getppid()) == parent
            and observer['read_parent_cmdline'](parent['pid']) == parent_argv,
            'Actual process identity changed during observation')
    payload = dict(schema='v7.pool_parent.runtime_observation.v1',
        observed_at=datetime.now(timezone.utc).isoformat(), job_id=output.name,
        alias=IDENTITY_ALIAS, pool_dir=str(POOL), controller=controller,
        pool_process=parent, pool_argv=parent_argv, pool_cwd=cwd,
        resolved_pool_source=resolved, visible_device=mask,
        source_sha256={str(path): sha(path) for path in
            (Path(__file__).resolve(), OBSERVER, observer['COORDINATOR'], Path(resolved))},
        source_scope='observed disk source bytes and actual process metadata',
        metadata_only=True, training_executed=False, GPU_profile_executed=False,
        actual_data_read=False, credentials_read=False)
    receipt = POOL/'logs'/f'{output.name}.pool_parent_runtime_observation.json'
    observer['publish'](receipt, payload, api)
    output.mkdir(exist_ok=False)
    observer['publish'](output/'COMPLETE.json', dict(status='resource_identity_observed',
        receipt=str(receipt), receipt_sha256=sha(receipt), training_executed=False,
        GPU_profile_executed=False, actual_data_read=False), api)
    print(json.dumps({'status': 'resource_identity_observed', 'receipt': str(receipt)}), flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = [argv[i+1] for i, value in enumerate(argv[:-1]) if value == '--variant']
    selected += [value.split('=', 1)[1] for value in argv if value.startswith('--variant=')]
    if selected == [IDENTITY_ALIAS]:
        return observe_parent(argv)
    if len(selected) == 1 and selected[0] in ALIASES:
        return formal_api()['main'](argv)
    return load_bound(PREVIOUS, PREVIOUS_SHA)['main'](argv)


if __name__ == '__main__':
    raise SystemExit(main())
