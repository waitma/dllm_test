"""Dispatch registered formal group tickets through the existing pool.

Activate conda pllm. The future shared entry calls main(argv) before ML imports;
prepare(argv) only reads a published catalog, bound plan and frozen wrapper.
This file never publishes a catalog, adds registry entries, queues or restarts
jobs. Ticket suffixes do not select GPUs: the slot wrapper observes its mask.
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
OWN = ROOT / 'output/v7_formal_eight_gpu_20261002'
POOL = ROOT / 'output/immune_research_pool_c20250601_20260930'
DATA = ROOT / 'data/prepared/immune_v7_test_scope_final_20261002'
CATALOG = OWN / 'pool_job_catalog.json'
PREVIOUS = ROOT / 'output/v7_large_batch_ablation_20261002/launcher_source_v1/pool_large_batch_ablation_dispatch.py'
PREVIOUS_SHA = 'dc3408ff7a8e74b7944b5ceb17ca18bdaf0baf9a226d270c9cb12fe99058aa0f'
ALIASES = tuple(f'p03_v7_formal_member_{index}' for index in range(8))


def require(value, message):
    if not value:
        raise ValueError(message)


def read_bytes(path):
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink()
            and path.is_file(), 'Expected a canonical regular file: ' + str(path))
    return path.read_bytes()


def reference(ref, *, wrapper=False):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'Explicit file reference required')
    path = Path(ref['path'])
    require(path.is_relative_to(OWN) and path != OWN, 'Formal file must belong to the registered output root')
    require(not wrapper or path.name == 'formal_pool_job.py', 'Only the frozen slot wrapper may execute')
    raw = read_bytes(path)
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Bound formal file changed: ' + str(path))
    return path, raw


def prepare(argv):
    """Resolve one explicitly catalogued ticket; no writes or execution."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--variant', choices=ALIASES, required=True)
    parser.add_argument('--mode', choices=('train',), required=True)
    parser.add_argument('--steps', type=int, required=True)
    for name in ('data-dir', 'profile-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume-from-checkpoint', type=Path)
    args = parser.parse_args(argv)
    for path in (args.data_dir, args.profile_dir, args.output_dir):
        require(path.is_absolute() and path == path.resolve()
                and not any(part.is_symlink() for part in (path, *path.parents)),
                'Pool data/output/profile paths must be canonical and nonsymlink')
    require(1 <= args.steps <= 5000 and args.data_dir == DATA
            and args.profile_dir == POOL / 'profiles' / args.variant
            and args.output_dir.parent == POOL / 'runs'
            and re.fullmatch(r'[0-9]+_' + re.escape(args.variant) + r'_[0-9a-f]{8}', args.output_dir.name),
            'Require the explicit formal parent, pool ticket and bounded segment')
    job = dict(variant=args.variant, mode=args.mode, steps=args.steps, data_dir=str(args.data_dir))
    if args.resume_from_checkpoint is not None:
        require(args.resume_from_checkpoint.is_absolute()
                and args.resume_from_checkpoint == args.resume_from_checkpoint.resolve(), 'Canonical resume path required')
        job['resume_from_checkpoint'] = str(args.resume_from_checkpoint)
    catalog = json.loads(read_bytes(CATALOG))
    require(catalog.get('schema') == 'v7.formal.pool_job_catalog.v1'
            and type(catalog.get('jobs')) is dict, 'Formal catalog is absent or has another schema')
    ref = catalog['jobs'].get(args.output_dir.name)
    require(ref is not None, 'Job ID has not been registered in the formal catalog')
    plan_path, raw = reference(ref)
    plan = json.loads(raw)
    require(plan.get('schema') == 'v7.formal.pool_group_plan.v1'
            and plan.get('pool_dir') == str(POOL)
            and len(plan.get('tickets', {})) == 8
            and plan['tickets'].get(args.output_dir.name) == job, 'Exact eight-ticket plan or job envelope differs')
    require(all(catalog['jobs'].get(ticket) == ref for ticket in plan['tickets']),
            'All eight tickets must refer to the same complete group plan')
    require(plan.get('work', {}).get('data_dir') == str(DATA), 'Formal parent differs from the bound driver work')
    wrapper, _ = reference(plan['wrapper'], wrapper=True)
    wrapper_api = runpy.run_path(str(wrapper), run_name='_bound_formal_slot_metadata')
    plan = wrapper_api['prepare'](plan)
    driver_path, _ = reference(plan['work']['driver'])
    require(driver_path == Path(plan['work']['source_root'])/'scripts/research/run_v7_formal.py',
            'Driver path differs from selected frozen source root')
    driver = runpy.run_path(str(driver_path), run_name='_bound_formal_driver_parse_only')
    wrapper_api['validate_segment'](plan, driver['parse_args'](plan['work']['driver_argv']),
                                    binding_digest=driver['digest'])
    command = [sys.executable, '-I', '-B', '-u', str(wrapper), '--plan', str(plan_path),
               '--plan-sha', ref['sha256'], '--job-id', args.output_dir.name]
    return args, command


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == '--variant']
    selected += [value.split('=', 1)[1] for value in argv if value.startswith('--variant=')]
    if len(selected) != 1 or selected[0] not in ALIASES:
        require(hashlib.sha256(read_bytes(PREVIOUS)).hexdigest() == PREVIOUS_SHA,
                'Frozen large-batch dispatcher source changed')
        return runpy.run_path(str(PREVIOUS))['main'](argv)
    args, command = prepare(argv)
    require(re.fullmatch('[0-7]', os.environ.get('CUDA_VISIBLE_DEVICES', '')),
            'Require one actual GPU mask from the existing pool')
    # exec retains the Popen PID and process group recorded by the live pool.
    os.execv(sys.executable, command)


if __name__ == '__main__':
    raise SystemExit(main())
