"""Check one privately authorized post-formal ablation, then exec its frozen CLI.

Activate conda pllm before a metadata check. A future SHA-bound pool entry calls
prepare(argv, catalog_path=ABSOLUTE_PATH, catalog_sha256=SHA,
        ledger_path=ABSOLUTE_PATH, ledger_sha256=SHA) without starting training.
Only main() execs a model; it is not a publisher, scheduler or retry mechanism.
Do not call main() before a separately reviewed post-formal publication.

Future catalog schema: v7.metrics_only.pool_catalog.v1; study and pool_dir are
strings. batch_plan, source_freeze, original_catalog, managed_local_plan,
adapter_source, formal_completion, pool_parent_observation and ledger are
{path, sha256} refs. formal_stage is the absolute final stage directory;
automatic_retry is false. jobs retain the existing reporter's five fields:
variant, seed, alias=p04_v7_<variant>_seed<seed>, job_id, job. job is exactly
{variant:alias, mode:train, steps:3750, data_dir:the registered subset}.

Ledger schema: v7.metrics_only.ablation_claims.v1; study, pool_dir, and cells.
cells is the complete 22-element list of {variant, seed, state, attempts}.
An admitted cell has state=reserved_pool and exactly one attempt containing
{transport:pool, pool_dir, job_id, lock_file:{path,sha256}}. Every other cell
has state=unclaimed, local_owned, completed or blocked. Existing original
attempts remain in original_catalog; this ledger describes new attempts only.
The existing empty lock file is shared with any future local launcher and is
held across exec. Neither function creates metadata, tickets or lock files.
Future local launchers must adopt this same lock before mixed local/pool use;
the existing managed-local preparation is not changed by this adapter.

This private file requires real original formal 25042 acceptance and actual
eight-rank cleanup. It reads manifests/certificates, never sequence rows.
The original runner still performs its full native data/source checks.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OLD = ROOT / 'output/v7_large_batch_ablation_20261002'
LOCAL = ROOT / 'output/v7_local_ablation_20261005'
POOL = ROOT / 'output/immune_research_pool_c20250601_20261005'
OLD_POOL = ROOT / 'output/immune_research_pool_c20250601_20260930'
DATA = ROOT / 'output/v7_architecture_pilot_20261002/sample_context_plus20k'
PLAN = OLD / 'batch_selection_v1.json'
PLAN_SHA = '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04'
ORIGINAL = OLD / 'jobs.json'
ORIGINAL_SHA = '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d'
FREEZE = LOCAL / 'source_snapshot_v1.json'
FREEZE_SHA = 'c840011ab6b01c8b6dd35e61d059e2b7206551749654cf67cc4ebe1c9562db0b'
MANAGED = LOCAL / 'managed_local_launch_plan_v1.json'
MANAGED_SHA = '7413c5fc9783ea2aa81dc37ebbdb5ba25588e09f76cc88190039e1a9de26970e'
SAMPLE_SHA = '1efa27ea19d10bc94143c3977c80dba8c3a29bad921c093ca7f24447635b9478'
PREFLIGHT_SHA = '21c640f870284cf7f09ebf8766db63ec7e7d311b8e94e0f2567798f53474a5f6'
LEGACY = OLD / 'launcher_source_v2/scripts/research/pool_large_batch_ablation_v2_dispatch.py'
LEGACY_SHA = 'b088ba3c19f58060a05393411cdf57a3abe8ecd7ba91508add62142896476cc3'
PARENT_API = OLD / 'pool_parent_source_v1/pool_parent_runtime_observation.py'
PARENT_API_SHA = '8d13c32c6c8e0b6ee65df4573a14e784947e98602e50c7ef10cf2459393e79c8'
FORMAL = ROOT / 'output/v7_formal_eight_gpu_20261002'
SUP = ROOT / 'output/v7_formal_supervision_20261005'
FORMAL_RUN = ROOT / 'output/v7_initial_training_20261005/run'
RESUME_API = FORMAL / 'source_snapshot_v3/scripts/research/formal_resume_observation.py'
RESUME_API_SHA = '0a0ca3baa74ae2567953db9fdebab01f031fe0a3ebb1f4ccd28405bf78fa4c97'
POOL_SOURCE = ROOT / 'scripts/immune_research_pool.py'
POOL_SOURCE_SHA = 'f3522eca58f7fedf145e8d61986d5124e0cdd9e2718c6062c53c94e563ba832f'
STUDY = 'v7_context_factorial_120k_large_batch_v1'
METHODS = ('f000', 'f001', 'f010', 'f011', 'f100', 'f101', 'f110', 'f111',
           'cv_segment_global', 'cv_late_fusion', 'fixed_gate_control')
ALIASES = {f'p04_v7_{v}_seed{s}': (v, s) for v in METHODS for s in (42, 43)}
MATH_SOURCES = ('dllm/pipelines/immune_llada/research_next/model.py',
    'dllm/pipelines/immune_llada/research_refined/model.py',
    'dllm/pipelines/immune_llada/research_refined/factorial.py',
    'dllm/pipelines/immune_llada/v3/layout.py', 'dllm/pipelines/immune_llada/v3/model.py',
    'dllm/pipelines/llada/models/modeling_llada.py', 'scripts/run_immune_architecture.py',
    'scripts/run_immune_refined.py', 'scripts/research/context_pilot_metrics.py')
DDP_ENV = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT')
RUN_ENV = dict(USE_TF='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    PYTHONDONTWRITEBYTECODE='1')


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve()
            and not any(p.is_symlink() for p in (path, *path.parents)),
            'Canonical nonsymlink path required: ' + str(path))
    return path


def sha(path):
    return hashlib.sha256(canonical(path).read_bytes()).hexdigest()


def read(path, expected=None):
    path = canonical(path)
    require(path.is_file(), 'Required metadata missing: ' + str(path))
    raw = path.read_bytes()
    if expected is not None:
        require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected)
                and hashlib.sha256(raw).hexdigest() == expected, 'Metadata SHA differs: ' + str(path))
    return json.loads(raw)


def bound(ref, path=None):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Explicit path/SHA reference required')
    if path is not None:
        require(ref['path'] == str(path), 'Reference path differs: ' + str(path))
    return read(Path(ref['path']), ref['sha256'])


def _api(path, expected):
    require(sha(path) == expected, 'Read-only helper source differs: ' + str(path))
    api = runpy.run_path(str(path), run_name='_post_formal_metadata_only')
    require(sha(path) == expected, 'Read-only helper changed during loading')
    return api


def _legacy_runtime():
    return _api(LEGACY, LEGACY_SHA)['prepare'].__globals__


def check_formal(catalog):
    complete = bound(catalog['formal_completion'], SUP / 'COMPLETE.json')
    require(complete.get('status') == 'actual_segment_passed' and complete.get('step') == 25042
            and complete.get('prior_step') == 25016, 'Formal 25042 acceptance is not established')
    require(not (SUP / 'STOP').exists(), 'Formal supervisor STOP prevents admission')
    stage = canonical(catalog['formal_stage'])
    require(stage.parent == FORMAL / 'launch_20261005'
            and re.fullmatch(r'segment_25042(?:_attempt(?:[2-9]|[1-9][0-9]+))?', stage.name), 'Unexpected final stage')
    state = read(stage / 'meeting/coordinator/state.json')
    require(state.get('terminal') == 'succeeded' and state.get('abort') is None,
            'Final eight-rank group is not terminal succeeded')
    results = complete.get('pool_results')
    require(isinstance(results, dict) and len(results) == 8
            and set(read(stage / 'enqueue_receipt.json')['job_ids']) == set(results),
            'Formal acceptance must bind the actual eight tickets')
    for job_id, ref in results.items():
        result = bound(ref, POOL / 'results' / (job_id + '.json'))
        require(result.get('job_id') == job_id and result.get('state') == 'succeeded'
                and type(result.get('returncode')) is int and result['returncode'] == 0,
                'Formal pool exit was not verified')
        require(not any((POOL / folder / (job_id + '.json')).exists() for folder in ('pending', 'running')),
                'A formal ticket remains active')
    for rank in range(8):
        receipt = read(stage / f'meeting/coordinator/rank{rank}_parent_receipt.json')
        cleanup = receipt['cleanup']
        require(receipt.get('rank') == rank and receipt.get('error') is None
                and type(cleanup.get('returncode')) is int and cleanup['returncode'] == 0
                and all(cleanup.get(k) is True for k in ('wait_observed', 'process_absent',
                    'process_group_gone', 'cleanup_completed')), 'Formal rank cleanup is incomplete')
    marker = bound(complete['marker'], FORMAL_RUN / 'COMPLETE.json')
    resume = bound(complete['resume'], FORMAL_RUN / 'RESUME_25016_TO_25042.json')
    checkpoint = complete['checkpoint']
    sidecar = bound(checkpoint['sidecar'], SUP / 'checkpoint_evidence/checkpoint-25042/formal_checkpoint.json')
    require(checkpoint.get('step') == 25042 and marker.get('status') == 'completed'
            and marker.get('step') == 25042 and marker.get('world_size') == 8
            and sidecar.get('step') == 25042 and sidecar.get('world_size') == 8
            and sidecar.get('rank_acknowledgments') == list(range(8))
            and resume.get('prior_step') == 25016 and resume.get('final_step') == 25042
            and sidecar.get('binding_sha256') == resume.get('binding_sha256') == marker.get('binding_sha256'),
            'Native final marker/resume/checkpoint binding differs')
    observer = _api(RESUME_API, RESUME_API_SHA)
    require(observer['complete_receipt'](resume['ranks']) == resume, 'Native resume receipt differs')
    prior = SUP / 'checkpoint_evidence/checkpoint-25016/formal_checkpoint.json'
    require(sha(prior) == resume['sidecar_sha256']
            and read(prior)['saved_parameter_digest'] != sidecar['saved_parameter_digest'],
            'Native prior checkpoint or resumed parameter change differs')


def check_data():
    sample = read(DATA / 'sample_manifest.json', SAMPLE_SHA)
    require(sample.get('schema') == 'v7.exploratory_uncertified.v1'
            and sample.get('status') == 'exploratory_uncertified'
            and sample.get('formal_training_ready') is False, 'Exploratory sample identity differs')
    parent = sample['parent']
    ref = parent['manifest']
    original = read(Path(ref['path']), ref['sha256'])
    require(ref.get('training_ready') is True and original.get('training_ready') is True,
            'Parent training-ready gate is not satisfied')
    for name in ('audit_certificate', 'downstream_scope_certificate'):
        entry = parent['certificates'][name]
        cert = read(Path(entry['path']), entry['sha256'])
        key = 'manifest_sha256' if name == 'audit_certificate' else 'dataset_manifest_sha256'
        require(cert.get('status') == entry.get('status') == 'passed'
                and cert.get('pending') == entry.get('pending') == [] and cert.get(key) == ref['sha256']
                and cert.get('forbidden_overlap_rows') == 0 and cert.get('family_split_overlap') == 0,
                'Parent certificate is not current/passed: ' + name)
    require(sample['files']['train']['records'] == 120000
            and sample['files']['valid']['records'] == 1024, 'Subset exposure protocol differs')


def check_sources_and_batch(catalog):
    require(catalog['source_freeze'] == dict(path=str(FREEZE), sha256=FREEZE_SHA)
            and catalog['batch_plan'] == dict(path=str(PLAN), sha256=PLAN_SHA), 'Frozen source/batch identity differs')
    legacy = _legacy_runtime()
    require(legacy['POOL'] == OLD_POOL and legacy['PLAN'] == PLAN
            and legacy['FREEZE'] == OLD / 'source_snapshot_v1.json', 'Original capacity namespace must remain unchanged')
    old_snapshot, old_sources = legacy['check_freeze'](canonical)
    frozen = bound(catalog['source_freeze'])
    snapshot = canonical(frozen['snapshot_root'])
    require(frozen.get('schema') == 'v7.metrics_only_ablation.source.v1'
            and snapshot == LOCAL / 'source_snapshot_v1'
            and frozen.get('base_snapshot') == str(legacy['FREEZE'])
            and frozen.get('base_snapshot_sha256') == legacy['FREEZE_SHA']
            and len(frozen['source_sha256']) == 569, 'Metrics-only freeze/base differs')
    for name, expected in frozen['source_sha256'].items():
        path = canonical(snapshot / name)
        require(path.is_relative_to(snapshot) and sha(path) == expected, 'Frozen source differs: ' + name)
    require(all(frozen['source_sha256'].get(n) == old_sources.get(n) for n in MATH_SOURCES),
            'Capacity-model mathematical sources differ')
    preflight = read(DATA.parent / 'context_augmentation_preflight.json', PREFLIGHT_SHA)
    require(preflight.get('status') == 'passed' and preflight.get('sample_manifest_sha256') == SAMPLE_SHA
            and preflight.get('layout_sha256') == frozen['source_sha256']['dllm/pipelines/immune_llada/v3/layout.py'],
            'Original context input preflight differs')
    runner = _api(snapshot / 'scripts/research/run_v7_large_batch_ablation.py',
                  frozen['source_sha256']['scripts/research/run_v7_large_batch_ablation.py'])
    plan = bound(catalog['batch_plan'])
    require((plan.get('micro_batch'), plan.get('grad_accum'), plan.get('global_batch'),
             plan.get('evaluation_micro_batch')) == (32, 1, 32, 4), 'Selected batch protocol differs')
    # Never rebind the legacy capacity function to the new pool/source root.
    legacy['check_capacity'](plan, old_snapshot, old_sources, canonical)
    return runner, snapshot


def check_claim(catalog, ledger, args, managed, original):
    require(catalog.get('automatic_retry') is False, 'Automatic retries are not authorized')
    rows = catalog.get('jobs')
    require(isinstance(rows, list) and rows, 'Explicit nonempty job catalog required')
    old_ids = {row['job_id'] for row in original['jobs']}
    jobs, cells = {}, set()
    for row in rows:
        alias, job_id = row['alias'], row['job_id']
        require(alias in ALIASES and ALIASES[alias] == (row['variant'], row['seed'])
                and type(row['seed']) is int and alias not in cells and job_id not in jobs
                and job_id not in old_ids and re.fullmatch(r'[0-9]+_' + re.escape(alias) + r'_[0-9a-f]{8}', job_id),
                'Unknown, duplicate or reused catalog job/cell')
        require(row['job'] == dict(variant=alias, mode='train', steps=3750, data_dir=str(DATA)),
                'Pool job training envelope differs')
        jobs[job_id], cells = row, cells | {alias}
    require(args.output_dir.name in jobs and jobs[args.output_dir.name]['alias'] == args.variant,
            'Current job is not the explicitly authorized cell')
    claims = ledger.get('cells')
    require(ledger.get('schema') == 'v7.metrics_only.ablation_claims.v1'
            and ledger.get('study') == STUDY and ledger.get('pool_dir') == str(POOL)
            and isinstance(claims, list) and len(claims) == 22, 'Full 22-cell new-attempt ledger required')
    by_cell = {}
    reserved = {}
    for item in claims:
        cell = item['variant'], item['seed']
        require(type(item['seed']) is int and cell in ALIASES.values() and cell not in by_cell
                and isinstance(item.get('attempts'), list), 'Duplicate/unknown ledger cell')
        by_cell[cell] = item
        if item.get('state') == 'reserved_pool':
            require(len(item['attempts']) == 1, 'Reserved cell must have one fresh exclusive attempt')
            attempt = item['attempts'][0]
            require(attempt.get('transport') == 'pool' and attempt.get('pool_dir') == str(POOL)
                    and attempt.get('job_id') in jobs and attempt['job_id'] not in reserved
                    and cell == (jobs[attempt['job_id']]['variant'], jobs[attempt['job_id']]['seed']),
                    'Cross-bound pool attempt claim')
            reserved[attempt['job_id']] = attempt
        else:
            require(item.get('state') in ('unclaimed', 'local_owned', 'completed', 'blocked'), 'Unknown claim state')
    require(set(reserved) == set(jobs), 'Catalog and unique reservations differ')
    row = jobs[args.output_dir.name]
    cell = row['variant'], row['seed']
    candidates = [m for m in managed['commands'] if (m['variant'], m['seed']) == cell]
    require(len(candidates) == 1, 'Cell is not an unstarted managed-local candidate')
    candidate = candidates[0]
    for path in (candidate['output_dir'], candidate['log_dir']):
        require(not canonical(path).exists(), 'A corresponding local attempt already exists')
    old_row = next(r for r in original['jobs'] if (r['variant'], r['seed']) == cell)
    old_result = OLD_POOL / 'results' / (old_row['job_id'] + '.json')
    if old_result.exists():
        require(read(old_result).get('state') != 'succeeded', 'Original cell already succeeded')
    for folder in ('results', 'finished', 'runs'):
        path = POOL / folder / (row['job_id'] if folder == 'runs' else row['job_id'] + '.json')
        require(not canonical(path).exists(), 'Fresh pool attempt identity already has evidence')
    for pool in (POOL, OLD_POOL):
        for folder in ('pending', 'running'):
            for path in (pool / folder).glob('*.json'):
                job = read(path)
                if pool == POOL and path.stem == row['job_id']:
                    require(job == row['job'], 'Actual ticket differs from authorized envelope')
                else:
                    require(job.get('variant') != args.variant, 'Cell already has an active pool attempt')
                require(not str(job.get('variant', '')).startswith('p06_v7_formal_member_'),
                        'Formal ticket remains active')
    lock = reserved[row['job_id']]['lock_file']
    lock_path = canonical(lock['path'])
    require(lock_path.parent == canonical(Path(catalog['ledger']['path'])).parent / 'cell_locks'
            and lock_path.name == f'{cell[0]}_seed{cell[1]}.lock'
            and lock_path.is_file() and lock_path.stat().st_size == 0 and sha(lock_path) == lock['sha256'],
            'Existing shared empty cell lock is not bound')
    return candidate, lock_path


def prepare(argv, *, catalog_path, catalog_sha256, ledger_path, ledger_sha256):
    """Read metadata/source bytes only; do not create directories or import ML."""
    flags = ('--variant', '--mode', '--steps', '--data-dir', '--profile-dir', '--output-dir')
    require(all(sum(v.split('=', 1)[0] == f for v in argv) == 1 for f in flags), 'Original options must appear once')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--variant', choices=ALIASES, required=True)
    parser.add_argument('--mode', choices=('train',), required=True)
    parser.add_argument('--steps', type=int, choices=(3750,), required=True)
    for name in ('data-dir', 'profile-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(canonical(args.data_dir) == DATA and canonical(args.profile_dir) == POOL / 'profiles' / args.variant
            and canonical(args.output_dir).parent == POOL / 'runs', 'Original data/profile/pool paths differ')
    catalog = read(catalog_path, catalog_sha256)
    ledger = read(ledger_path, ledger_sha256)
    require(catalog.get('schema') == 'v7.metrics_only.pool_catalog.v1' and catalog.get('study') == STUDY
            and catalog.get('pool_dir') == str(POOL)
            and catalog.get('ledger') == dict(path=str(canonical(ledger_path)), sha256=ledger_sha256),
            'Catalog study/pool/ledger differs')
    own = Path(__file__).resolve()
    require(catalog.get('adapter_source') == dict(path=str(own), sha256=sha(own)), 'Private adapter source is not bound')
    require(catalog['original_catalog'] == dict(path=str(ORIGINAL), sha256=ORIGINAL_SHA)
            and catalog['managed_local_plan'] == dict(path=str(MANAGED), sha256=MANAGED_SHA), 'Original launch plans differ')
    original, managed = bound(catalog['original_catalog']), bound(catalog['managed_local_plan'])
    check_formal(catalog)
    candidate, lock_path = check_claim(catalog, ledger, args, managed, original)
    runner, snapshot = check_sources_and_batch(catalog)
    check_data()
    command = list(candidate['training_argv'])
    require(command[:5] == [command[0], '-I', '-B', '-u', str(snapshot / 'scripts/research/run_v7_large_batch_ablation.py')]
            and Path(command[0]).is_absolute(), 'Managed command does not use the frozen isolated CLI')
    require(command.count('--output-dir') == 1, 'Managed command output is ambiguous')
    command[command.index('--output-dir') + 1] = str(args.output_dir)
    selected = runner['parse_args'](command[5:])
    runner['validate_parameters'](selected)
    require(selected.variant == ALIASES[args.variant][0] and selected.seed == ALIASES[args.variant][1]
            and selected.steps == 3750 and selected.eval_every == 250 and selected.micro_batch == 32
            and selected.grad_accum == 1 and selected.data_dir == DATA and selected.output_dir == args.output_dir
            and selected.batch_plan == PLAN and selected.batch_plan_sha256 == PLAN_SHA
            and selected.resume is False and selected.cpu_smoke is False and selected.save_checkpoints is False,
            'Managed CLI protocol must remain fresh and metrics-only')
    runner['checked_batch_plan'](selected)
    lock_stat = lock_path.stat()
    args.lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
    args.lock_path, args.catalog = lock_path, catalog
    return args, command


def check_worker(args, command):
    """Bind an actual child invocation; host prepare deliberately does not call this."""
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(re.fullmatch('[0-7]', mask), 'Require one physical CUDA mask assigned by the pool')
    require(sys.executable == command[0], 'Actual pool interpreter differs from the managed protocol')
    require(not any(k in os.environ for k in DDP_ENV), 'Independent jobs cannot inherit distributed rank variables')
    require(not any((POOL / name).exists() for name in ('STOP', 'STOP_NOW')), 'Pool is stopped')
    job_id = args.output_dir.name
    row = next(r for r in args.catalog['jobs'] if r['job_id'] == job_id)
    require(read(POOL / 'running' / (job_id + '.json')) == row['job'], 'Actual claimed ticket differs')
    expected = [sys.executable, '-u', str(ROOT / 'scripts/run_immune_research.py'), '--variant', args.variant,
        '--mode', 'train', '--steps', '3750', '--profile-dir', str(args.profile_dir),
        '--output-dir', str(args.output_dir), '--data-dir', str(DATA)]
    workers = [read(p) for p in (POOL / 'workers').glob('*.json')]
    owners = [w for w in workers if w.get('job_id') == job_id and w.get('state') == 'running']
    require(len(owners) == 1 and owners[0].get('device') == mask
            and owners[0].get('job') == row['job'] and owners[0].get('argv') == expected
            and owners[0].get('pid') in (None, os.getpid()), 'Actual worker ownership/command differs')
    ref = args.catalog['pool_parent_observation']
    parent = bound(ref)
    require(parent.get('schema') == 'v7.pool_parent.runtime_observation.v1'
            and parent.get('pool_dir') == str(POOL) and parent.get('resolved_pool_source') == str(POOL_SOURCE)
            and parent['source_sha256'].get(str(POOL_SOURCE)) == POOL_SOURCE_SHA
            and sha(POOL_SOURCE) == POOL_SOURCE_SHA, 'Observed pool source identity differs')
    observer = _api(PARENT_API, PARENT_API_SHA)
    api = observer['coordinator_api']()
    require(api['process_identity'](os.getppid()) == parent['pool_process']
            and observer['read_parent_cmdline'](os.getppid()) == parent['pool_argv']
            and observer['read_parent_cwd'](os.getppid()) == parent['pool_cwd'], 'Actual pool parent identity changed')


def main(argv=None, *, catalog_path, catalog_sha256, ledger_path, ledger_sha256):
    args, command = prepare(list(sys.argv[1:] if argv is None else argv), catalog_path=catalog_path,
        catalog_sha256=catalog_sha256, ledger_path=ledger_path, ledger_sha256=ledger_sha256)
    # Use an existing shared lock; keep its descriptor open in the execed child.
    descriptor = os.open(args.lock_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        opened, current = os.fstat(descriptor), args.lock_path.stat()
        require(opened.st_size == 0 and (opened.st_dev, opened.st_ino) == args.lock_identity
                == (current.st_dev, current.st_ino), 'Shared cell lock inode changed')
        require(sha(catalog_path) == catalog_sha256 and sha(ledger_path) == ledger_sha256,
                'Admission catalog/ledger changed before exec')
        args.catalog = read(catalog_path, catalog_sha256)
        check_claim(args.catalog, read(ledger_path, ledger_sha256), args,
                    bound(args.catalog['managed_local_plan']), bound(args.catalog['original_catalog']))
        check_worker(args, command)
        os.set_inheritable(descriptor, True)
        os.environ.update(RUN_ENV)  # CUDA_VISIBLE_DEVICES is deliberately preserved.
        os.execv(sys.executable, command)
    finally:
        os.close(descriptor)
