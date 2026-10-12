"""Publish nine fixed CDR evaluations only after the actual two-epoch endpoint.

Activate conda pllm; the bounded generation supervisor calls this frozen script
once after both training supervisors exit successfully. No resource submit or
worker restart is used. Existing source is archived; only the public launcher
and AST alias registry are extended, after verifying they are outside the model
source manifest. The frozen training source, audit and checkpoints stay intact.
"""
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import runpy
import time
import uuid

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/v7_formal_eight_gpu_20261002'
EPOCH = OWN/'epoch2_continuation_20261006'
PARENT = OWN/'where_continuation_20261006'
BASE = EPOCH/'generation_pool_v1'
POOL = ROOT/'output/immune_research_pool_c20250601_20261005'
DATA = ROOT/'data/prepared/immune_v7_test_scope_final_20261002'
PUBLIC = ROOT/'scripts/run_immune_research.py'
REGISTRY = ROOT/'dllm/pipelines/immune_llada/research/model.py'
OLD_PUBLIC_SHA = '94908c1ea71dd46ee428a12c3843a0b39c1b1fdeb5a0f14ec00a088e5a466007'
OLD_REGISTRY_SHA = '538af1659ffadd58850eaca1f7e80993a1884b96920ed215bf428572b8776bdd'
DISPATCHER = BASE/'source_v1/pool_v7_generation_dispatch.py'
PACKAGE = BASE/'package_v1.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def atomic_bytes(path, value):
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(value)
    temporary.chmod(path.stat().st_mode & 0o777)
    temporary.replace(path)


def stops():
    require(not any(path.exists() for path in (BASE/'STOP', EPOCH/'STOP', PARENT/'STOP', POOL/'STOP', POOL/'STOP_NOW')),
            'Explicit stop prevents evaluation admission')


def released():
    workers = [read(path) for path in (POOL/'workers').glob('*.json')]
    return (len(workers) == 8 and {str(w['device']) for w in workers} == set(map(str, range(8)))
        and not list((POOL/'running').glob('*.json')) and not list((POOL/'pending').glob('*.json'))
        and all(w['state'] == 'standby_load' and 0 <= time.time()-w['time'] < 15 for w in workers))


def code_extension(public_bytes, registry_bytes, aliases, dispatcher, dispatcher_sha):
    """Pure source preparation: preserve every old executable statement."""
    public = public_bytes.decode()
    tree = ast.parse(public)
    doc = tree.body[0]
    require(isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str),
            'Existing public launcher must begin with its docstring')
    lines = public.splitlines(keepends=True)
    prefix = f'''# Post-training CDR entry; no model or CUDA imports before dispatch.
if __name__ == '__main__':
    import sys as _cdr_sys
    import runpy as _cdr_runpy
    import hashlib as _cdr_hashlib
    from pathlib import Path as _CdrPath
    _cdr_selected = [_cdr_sys.argv[i+1] for i, value in enumerate(_cdr_sys.argv[:-1]) if value == '--variant']
    if len(_cdr_selected) == 1 and _cdr_selected[0] in {tuple(aliases)!r}:
        _cdr_entry = _CdrPath({str(dispatcher)!r})
        if _cdr_hashlib.sha256(_cdr_entry.read_bytes()).hexdigest() != {dispatcher_sha!r}:
            raise RuntimeError('Published CDR evaluator changed')
        raise SystemExit(_cdr_runpy.run_path(str(_cdr_entry))['main'](_cdr_sys.argv[1:]))
'''
    new_public = ''.join(lines[:doc.end_lineno])+'\n'+prefix+''.join(lines[doc.end_lineno:])
    registry = registry_bytes.decode()
    require(registry.count('VARIANTS = {\n') == 1 and all(name not in registry for name in aliases),
            'Ambiguous registry or existing evaluation aliases')
    additions = ''.join(f"    {name!r}: {{'launcher_only': True, 'generation_evaluation_only': True}},\n" for name in aliases)
    new_registry = registry.replace('VARIANTS = {\n', 'VARIANTS = {\n'+additions, 1)
    old_ast = ast.dump(ast.parse(public), include_attributes=False)
    new_tree = ast.parse(new_public)
    del new_tree.body[1]
    require(ast.dump(new_tree, include_attributes=False) == old_ast, 'Original public executable statements changed')
    old_tree, new_tree = ast.parse(registry), ast.parse(new_registry)
    for node in new_tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VARIANTS' for t in node.targets):
            require([ast.literal_eval(k) for k in node.value.keys[:len(aliases)]] == list(aliases), 'Registry insertion differs')
            del node.value.keys[:len(aliases)]
            del node.value.values[:len(aliases)]
    require(ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False),
            'Existing registry or model code changed')
    compile(new_public, str(PUBLIC), 'exec')
    compile(new_registry, str(REGISTRY), 'exec')
    return new_public.encode(), new_registry.encode()


def main():
    stops()
    require(not (BASE/'release.json').exists() and not (BASE/'publication_started.json').exists(),
            'Do not repeat or repair a partially published task batch automatically')
    package = read(PACKAGE)
    for name, expected in package['files'].items():
        require(sha(Path(name)) == expected, 'Frozen generation dependency changed')
    for directory in (PARENT, EPOCH):
        launch = read(directory/'supervisor_launch.json')
        require(not (Path('/proc')/str(launch['pid'])).exists(), 'Training supervisor has not exited')
        require(not (directory/'SUPERVISION_STOPPED.json').exists(), 'Training supervision did not succeed')
    actual = read(EPOCH/'COMPLETE.json')
    require(actual['status'] == 'actual_world8_epoch2_continuation_passed' and actual['step'] == 50082
            and actual['epoch'] == 2 and actual['eight_native_restores_and_owned_exits'], 'Actual two-epoch completion required')
    launcher = runpy.run_path(str(EPOCH/'launcher_v1/launch_v7_epoch2_segment.py'), run_name='_eval_prior_cleanup')
    launcher['cleanup_checked'](EPOCH/'launches_v1/segment_50082', success=True)
    readiness = read(EPOCH/'EVALUATION_READY.json')
    require(readiness['checkpoint'] == actual['checkpoint'] and readiness['native_epoch'] == 2, 'Training eval pointer differs')
    platform_api = runpy.run_path(str(ROOT/'output/v7_formal_supervision_20261005/source_v1/supervise_v7_formal_20261005.py'), run_name='_eval_platform_only')
    platform = platform_api['platform']()
    require(platform['task']['Status'] == 'Running', 'Original allocation is not Running')
    require(released(), 'All eight original workers must be fresh and released')
    manifest = read(EPOCH/'source_manifest_v1.json')
    require(str(PUBLIC) not in manifest['files'] and str(REGISTRY) not in manifest['files'],
            'Cannot replace any source bound by the training checkpoint')
    require(sha(PUBLIC) == OLD_PUBLIC_SHA and sha(REGISTRY) == OLD_REGISTRY_SHA, 'Original public source changed')
    api = runpy.run_path(str(DISPATCHER), run_name='_generation_prepare')
    aliases = api['ALIASES']
    public, registry = code_extension(PUBLIC.read_bytes(), REGISTRY.read_bytes(), aliases, DISPATCHER, sha(DISPATCHER))
    jobs = []
    for alias, cell in aliases.items():
        job_id = f'{time.time_ns()}_{alias}_{uuid.uuid4().hex[:8]}'
        output = ROOT/'output/downstream_generation'/f'v7_f011_step50082_{cell[0]}_{cell[1]}'
        require(not output.exists(), 'An assigned generation result already exists')
        jobs.append(dict(job_id=job_id, cell=list(cell), evaluation_output=str(output),
            job=dict(variant=alias, mode='train', steps=1, data_dir=str(DATA))))
    original_identity = read(OWN/'entry_20261005/identity_enqueue.json')
    identity_path = POOL/'logs'/f'{original_identity["job_id"]}.pool_parent_runtime_observation.json'
    identity = read(identity_path)
    catalog = dict(schema='v7.cdr.pool_catalog.v1', work_type='generation_evaluation', optimizer_updates=0, jobs=jobs)
    release = dict(schema='v7.cdr.pool_release.v1', at_utc=datetime.now(timezone.utc).isoformat(),
        work_type='generation_evaluation', optimizer_updates=0, automatic_retry=False,
        training_acceptance=ref(EPOCH/'COMPLETE.json'), checkpoint=actual['checkpoint'],
        data_audit=ref(EPOCH/'parent_gate.stdout.json'), pool_process=identity['pool_process'],
        pool_parent_observation=ref(identity_path), pool_dir=str(POOL), files=dict(package['files']),
        compatibility_envelope='Existing worker mode=train/steps=1; no optimizer updates; actual inputs pinned separately')
    # A separate publication lock does not replace the existing worker/device locks.
    with (BASE/'publication.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stops()
        require(released(), 'Pool became busy before publication')
        write_new(BASE/'publication_started.json', dict(at_utc=release['at_utc'], platform=platform,
            old_public=ref(PUBLIC), old_registry=ref(REGISTRY), tasks=len(jobs)))
        (BASE/'public_before.py').write_bytes(PUBLIC.read_bytes())
        (BASE/'registry_before.py').write_bytes(REGISTRY.read_bytes())
        write_new(BASE/'catalog.json', catalog)
        release['catalog'] = ref(BASE/'catalog.json')
        release['files'].update({str(PUBLIC): hashlib.sha256(public).hexdigest(),
                                 str(REGISTRY): hashlib.sha256(registry).hexdigest()})
        atomic_bytes(REGISTRY, registry)
        atomic_bytes(PUBLIC, public)
        write_new(BASE/'release.json', release)
        api['completion'](release)
        pool = runpy.run_path(str(ROOT/'scripts/immune_research_pool.py'), run_name='_existing_pool_commands')
        for row in jobs:
            pool['validate'](row['job'])
            argv = pool['command'](POOL, row['job_id'], row['job'])[3:]
            api['prepare'](argv)
        stops()
        require(released(), 'Pool became busy before enqueue')
        receipt = dict(at_utc=datetime.now(timezone.utc).isoformat(), state='publishing',
            release=ref(BASE/'release.json'), enqueued_job_ids=[], all_job_ids=[r['job_id'] for r in jobs])
        for row in jobs:
            stops()
            ticket = POOL/'pending'/(row['job_id']+'.json')
            require(not ticket.exists(), 'Duplicate queue ticket')
            pool['write_json'](ticket, row['job'])
            receipt['enqueued_job_ids'].append(row['job_id'])
            pool['write_json'](BASE/'enqueue_receipt.json', receipt)
        receipt['state'] = 'enqueued'
        pool['write_json'](BASE/'enqueue_receipt.json', receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
