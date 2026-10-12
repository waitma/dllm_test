"""Prepare or publish missing original ablations after two epochs and evaluation.

Activate conda pllm; invoke the independently frozen copy with --action preview
for read-only inventory, or --action publish only after actual predecessor exit.
Example: python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/publish_v7_post_generation_ablation.py --action preview
No platform submission, container/worker restart, annotation, model modification
or automatic retry is performed. Publication uses the existing eight workers.
"""
import argparse
import ast
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import time
import uuid


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
LOCAL = ROOT/'output/v7_local_ablation_20261005'
BASE = LOCAL/'post_generation_pool_v1'
POOL = ROOT/'output/immune_research_pool_c20250601_20261005'
FORMAL = ROOT/'output/v7_formal_eight_gpu_20261002'
PARENT = FORMAL/'where_continuation_20261006'
EPOCH = FORMAL/'epoch2_continuation_20261006'
GENERATION = EPOCH/'generation_pool_v1'
PACKAGE = BASE/'package_v1.json'
ADAPTER = BASE/'source_v1/pool_metrics_only_ablation_v2_dispatch.py'
REPORTER = BASE/'source_v1/plot_v7_mixed_ablation.py'
SELECTION = LOCAL/'mixed_report_selection_v2.json'
CATALOG = BASE/'catalog.json'
LEDGER = LOCAL/'guarded_dispatch_v1/claims_post_generation_pool_v1.json'
PRIOR_LEDGER = LOCAL/'guarded_dispatch_v1/claims_f010_seed42_attempt1.json'
PUBLIC = ROOT/'scripts/run_immune_research.py'
REGISTRY = ROOT/'dllm/pipelines/immune_llada/research/model.py'
PLATFORM_API = ROOT/'output/v7_formal_supervision_20261005/source_v1/supervise_v7_formal_20261005.py'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def fixed():
    package = read(PACKAGE)
    require(package['schema'] == 'v7.post_generation.ablation_package.v1', 'Wrong private preparation package')
    for name, expected in package['files'].items():
        require(sha(name) == expected, 'Frozen post-generation dependency changed: '+name)
    return package


def live(directory):
    receipt = read(directory/'supervisor_launch.json')
    path = Path('/proc')/str(receipt['pid'])/'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] != 'Z' and int(fields[19]) == receipt['start_ticks']


def stopped():
    require(not any((path/name).exists() for path in (BASE, POOL, PARENT, EPOCH, GENERATION)
                    for name in ('STOP', 'STOP_NOW', 'SUPERVISION_STOPPED.json')), 'Explicit stop or supervision failure')


def released():
    workers = [read(p) for p in (POOL/'workers').glob('*.json')]
    return (len(workers) == 8 and {str(w['device']) for w in workers} == set(map(str, range(8)))
        and not list((POOL/'pending').glob('*.json')) and not list((POOL/'running').glob('*.json'))
        and all(w['state'] == 'standby_load' and 0 <= time.time()-w['time'] < 15 for w in workers))


def budget(platform, parent, task_count, current_time=None):
    """Conservative bound: every wave may consume the full existing job timeout."""
    task = platform['task']
    require(task['JobId'] == 't-20261005205851-lqfjg' and task['Status'] == 'Running', 'Original allocation is not Running')
    argv = parent['pool_argv']
    require(argv.count('--max-seconds') == argv.count('--job-seconds') == 1, 'Ambiguous original worker limits')
    maximum = int(argv[argv.index('--max-seconds')+1])
    per_job = int(argv[argv.index('--job-seconds')+1])
    require(maximum == 604200 and per_job == 86400, 'Original worker deadlines changed')
    started = datetime.fromisoformat(task['Start'].replace('Z', '+00:00')).timestamp()
    remaining = started+maximum-(time.time() if current_time is None else current_time)
    waves = math.ceil(task_count/8)
    required = waves*per_job+600
    require(remaining > required, 'Insufficient original allocation time for all selected job deadlines')
    return dict(remaining_seconds_lower_bound=remaining, required_seconds=required, waves=waves,
                per_job_limit_seconds=per_job, allocation_max_seconds=maximum,
                scope='Conservative admission bound, not a runtime forecast; original deadlines unchanged')


def inventory(api, reporter):
    """Re-read all original slots and exclude every observed or owned attempt."""
    report = reporter['collect'](SELECTION, 3750)
    managed = api['read'](api['MANAGED'], api['MANAGED_SHA'])
    candidates = {(r['variant'], r['seed']): r for r in managed['commands']}
    prior = read(PRIOR_LEDGER)
    prior_cells = {(r['variant'], r['seed']): r for r in prior['cells']}
    require(len(prior_cells) == len(report['runs']) == 22, 'Full original cell inventory required')
    selected, exclusions = [], []
    for run in report['runs']:
        cell = run['variant'], run['seed']
        if run['artifact_complete']:
            exclusions.append(dict(cell=list(cell), state='completed', reason=run['status']))
            continue
        if run['transport'] != 'pending':
            exclusions.append(dict(cell=list(cell), state='local_owned' if run['transport'] == 'local' else 'blocked',
                reason='Existing selected attempt: '+run['status']))
            continue
        require(cell in candidates and prior_cells[cell]['state'] == 'unclaimed', 'Pending cell lacks original fresh authorization')
        candidate = candidates[cell]
        observed = any(Path(candidate[key]).exists() for key in ('output_dir', 'log_dir'))
        alias = f'p04_v7_{cell[0]}_seed{cell[1]}'
        for folder in ('pending', 'running', 'results'):
            for path in (POOL/folder).glob('*.json'):
                value = read(path); spec = value.get('job', value)
                observed |= spec.get('variant') == alias
        if observed:
            exclusions.append(dict(cell=list(cell), state='blocked', reason='A newer local or pool attempt has evidence'))
        else:
            selected.append(cell)
    return report, selected, exclusions, prior


def formal_fields():
    return dict(formal_completion=ref(EPOCH/'COMPLETE.json'),
        formal_stage=str(EPOCH/'launches_v1/segment_50082'),
        generation_completion=ref(GENERATION/'COMPLETE.json'),
        generation_release=ref(GENERATION/'release.json'), generation_package=ref(GENERATION/'package_v1.json'))


def public_extension(before, aliases, adapter_sha, catalog_sha, ledger_sha):
    """Preserve all previous executable statements and existing alias metadata."""
    original = ast.parse(before.decode())
    doc = original.body[0]
    require(isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)
            and isinstance(doc.value.value, str), 'Original launcher docstring is required')
    prefix = f'''# Original-study ablations after actual training and generation completion.
if __name__ == '__main__':
    import sys as _abl_sys
    import runpy as _abl_runpy
    import hashlib as _abl_hashlib
    import json as _abl_json
    import time as _abl_time
    from pathlib import Path as _AblPath
    _abl_selected = [_abl_sys.argv[i+1] for i, value in enumerate(_abl_sys.argv[:-1]) if value == '--variant']
    if len(_abl_selected) == 1 and _abl_selected[0] in {tuple(aliases)!r}:
        _abl_entry = _AblPath({str(ADAPTER)!r})
        if _abl_hashlib.sha256(_abl_entry.read_bytes()).hexdigest() != {adapter_sha!r}:
            raise RuntimeError('Published post-generation ablation entry changed')
        _abl_receipt = _AblPath({str(BASE/'enqueue_receipt.json')!r})
        _abl_deadline = _abl_time.monotonic()+60
        while True:
            try:
                _abl_batch = _abl_json.loads(_abl_receipt.read_text())
            except FileNotFoundError:
                _abl_batch = {{}}
            if _abl_batch.get('state') == 'enqueued':
                if _abl_batch.get('catalog') != {{'path': {str(CATALOG)!r}, 'sha256': {catalog_sha!r}}}:
                    raise RuntimeError('Post-generation batch binding changed')
                break
            if _abl_time.monotonic() >= _abl_deadline:
                raise RuntimeError('Post-generation publication did not finish; no automatic retry')
            _abl_time.sleep(0.1)
        raise SystemExit(_abl_runpy.run_path(str(_abl_entry))['main'](_abl_sys.argv[1:],
            catalog_path=_AblPath({str(CATALOG)!r}), catalog_sha256={catalog_sha!r},
            ledger_path=_AblPath({str(LEDGER)!r}), ledger_sha256={ledger_sha!r}))
'''
    lines = before.decode().splitlines(keepends=True)
    after = ''.join(lines[:doc.end_lineno])+'\n'+prefix+''.join(lines[doc.end_lineno:])
    tree = ast.parse(after); del tree.body[1]
    require(ast.dump(tree, include_attributes=False) == ast.dump(original, include_attributes=False),
            'Earlier launcher statements were changed')
    compile(after, str(PUBLIC), 'exec')
    return after.encode()


def check_predecessors(api):
    stopped()
    require(not any(live(d) for d in (PARENT, EPOCH, GENERATION)), 'Training or generation supervisor is still live')
    fields = formal_fields()
    api['check_formal'](fields)
    return fields


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('preview', 'publish'), required=True)
    args = parser.parse_args(argv)
    package = fixed()
    api = runpy.run_path(str(ADAPTER), run_name='_post_generation_ablation_prepare')
    reporter = runpy.run_path(str(REPORTER), run_name='_post_generation_ablation_report')
    if args.action == 'preview':
        report, cells, exclusions, prior = inventory(api, reporter)
        print(json.dumps(dict(status='read_only_preview_not_admitted', selected_cells=cells,
            exclusions=exclusions, artifact_complete=sum(r['artifact_complete'] for r in report['runs'])), ensure_ascii=False))
        return 0
    # No publication file, lock, ledger or queue mutation precedes these gates.
    fields = check_predecessors(api)
    require(not (BASE/'publication_started.json').exists() and not CATALOG.exists() and not LEDGER.exists(),
            'Never replay a started or partly published batch')
    platform = runpy.run_path(str(PLATFORM_API), run_name='_post_ablation_platform')['platform']()
    release = read(GENERATION/'release.json')
    parent_ref = release['pool_parent_observation']; parent = api['bound'](parent_ref)
    require(parent['pool_process'] == release['pool_process'], 'Original pool identity differs')
    require(released(), 'Original eight workers have not all been released')
    require(sha(PUBLIC) == release['files'][str(PUBLIC)] and sha(REGISTRY) == release['files'][str(REGISTRY)],
            'Post-generation public entry changed before ablation publication')
    report, cells, exclusions, prior = inventory(api, reporter)
    require(cells, 'No unstarted eligible cells; do not create an empty batch')
    resource_budget = budget(platform, parent, len(cells))
    with ExitStack() as stack:
        publication = stack.enter_context((BASE/'publication.lock').open('a+'))
        fcntl.flock(publication, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks = {}
        for variant, seed in cells:
            lock = LEDGER.parent/'cell_locks'/f'{variant}_seed{seed}.lock'
            descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            stack.callback(os.close, descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            actual, current = os.fstat(descriptor), lock.stat()
            require(actual.st_size == 0 and (actual.st_dev, actual.st_ino) == (current.st_dev, current.st_ino),
                    'Existing shared cell lock changed')
            locks[(variant, seed)] = ref(lock)
        fresh, fresh_cells, fresh_exclusions, fresh_prior = inventory(api, reporter)
        require(fresh_cells == cells and fresh_exclusions == exclusions and fresh_prior == prior,
                'Cell ownership changed before publication')
        stopped(); require(released(), 'Pool became busy before publication')
        fixed(); api['check_formal'](fields)
        jobs = []
        for variant, seed in cells:
            alias = f'p04_v7_{variant}_seed{seed}'
            job_id = f'{time.time_ns()}_{alias}_{uuid.uuid4().hex[:8]}'
            jobs.append(dict(variant=variant, seed=seed, alias=alias, job_id=job_id,
                job=dict(variant=alias, mode='train', steps=3750, data_dir=str(api['DATA']))))
        book = json.loads(json.dumps(prior))
        by_cell = {(r['variant'], r['seed']): r for r in book['cells']}
        for item in exclusions:
            cell = tuple(item['cell']); row = by_cell[cell]
            if item['state'] in ('completed', 'blocked'):
                row.update(state=item['state'], exclusion_reason=item['reason'])
        for job in jobs:
            cell = job['variant'], job['seed']
            by_cell[cell].update(state='reserved_pool', attempts=[dict(transport='pool', pool_dir=str(POOL),
                job_id=job['job_id'], lock_file=locks[cell])])
        book.update(at_utc=now(), prior_ledger=ref(PRIOR_LEDGER), generation_completion=fields['generation_completion'])
        intent = dict(at_utc=now(), work_type='original_architecture_ablation', automatic_retry=False,
            package=ref(PACKAGE), platform=platform, resource_budget=resource_budget, selected_cells=cells,
            exclusions=exclusions, predecessor_evidence=fields, public_before=ref(PUBLIC), registry_unchanged=ref(REGISTRY))
        write_new(BASE/'publication_started.json', intent)
        write_new(BASE/'inventory_report.json', fresh)
        write_new(LEDGER, book)
        catalog = dict(schema='v7.metrics_only.pool_catalog.v2', study=api['STUDY'], pool_dir=str(POOL),
            batch_plan=dict(path=str(api['PLAN']), sha256=api['PLAN_SHA']),
            source_freeze=dict(path=str(api['FREEZE']), sha256=api['FREEZE_SHA']),
            original_catalog=dict(path=str(api['ORIGINAL']), sha256=api['ORIGINAL_SHA']),
            managed_local_plan=dict(path=str(api['MANAGED']), sha256=api['MANAGED_SHA']),
            adapter_source=ref(ADAPTER), ledger=ref(LEDGER), pool_parent_observation=parent_ref,
            automatic_retry=False, jobs=jobs, **fields)
        write_new(CATALOG, catalog)
        pool_api = runpy.run_path(str(ROOT/'scripts/immune_research_pool.py'), run_name='_same_existing_pool')
        kwargs = dict(catalog_path=CATALOG, catalog_sha256=sha(CATALOG), ledger_path=LEDGER, ledger_sha256=sha(LEDGER))
        for row in jobs:
            pool_api['validate'](row['job'])
            api['prepare'](pool_api['command'](POOL, row['job_id'], row['job'])[3:], **kwargs)
        before = PUBLIC.read_bytes()
        after = public_extension(before, [row['alias'] for row in jobs], sha(ADAPTER), sha(CATALOG), sha(LEDGER))
        (BASE/'public_before.py').write_bytes(before)
        (BASE/'public_after.py').write_bytes(after)
        updated_selection = json.loads(json.dumps(read(SELECTION)))
        for row in jobs:
            index = next(i for i, item in enumerate(updated_selection['cells'])
                         if (item['variant'], item['seed']) == (row['variant'], row['seed']))
            require(updated_selection['cells'][index]['transport'] == 'pending', 'Selected report cell is already owned')
            updated_selection['cells'][index] = dict(variant=row['variant'], seed=row['seed'], transport='pool',
                attempt_id=row['job_id'], source_cohort='local_metrics_only_v1', catalog=ref(CATALOG),
                pool_dir=str(POOL), job_id=row['job_id'])
        write_new(BASE/'mixed_report_selection.json', updated_selection)
        stopped(); require(released(), 'Pool became busy before entry replacement')
        require(sha(PUBLIC) == intent['public_before']['sha256'], 'Public entry changed while preparing')
        temporary = PUBLIC.with_name('.'+PUBLIC.name+'.'+uuid.uuid4().hex+'.tmp')
        with temporary.open('xb') as stream:
            stream.write(after)
        temporary.chmod(PUBLIC.stat().st_mode & 0o777); temporary.replace(PUBLIC)
        write_new(BASE/'release.json', dict(at_utc=now(), status='entry_published_not_yet_enqueued',
            catalog=ref(CATALOG), ledger=ref(LEDGER), public_entry=ref(PUBLIC), registry_unchanged=ref(REGISTRY),
            package=ref(PACKAGE), inventory=ref(BASE/'inventory_report.json'), resource_budget=resource_budget))
        receipt = dict(at_utc=now(), state='publishing', catalog=ref(CATALOG), enqueued_job_ids=[],
            all_job_ids=[row['job_id'] for row in jobs])
        for row in jobs:
            stopped()
            ticket = POOL/'pending'/(row['job_id']+'.json')
            require(not ticket.exists(), 'Duplicate queue ticket')
            pool_api['write_json'](ticket, row['job'])
            receipt['enqueued_job_ids'].append(row['job_id'])
            pool_api['write_json'](BASE/'enqueue_receipt.json', receipt)
    # Claiming workers wait in the stdlib entry until these admission locks
    # have actually closed, avoiding spurious LOCK_NB failures during enqueue.
    receipt['state'] = 'enqueued'; pool_api['write_json'](BASE/'enqueue_receipt.json', receipt)
    message = f'原八卡在两轮训练和生成完成后单次发布{len(jobs)}个缺失消融；120k/3750、metrics-only不变。原件 {BASE}/enqueue_receipt.json。'
    with (ROOT/'PROJECT_PROCESS.md').open('a') as stream:
        stream.write(f'\n\n## {now()} 后续消融实际入队\n\n{message}\n')
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
