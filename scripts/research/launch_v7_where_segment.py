"""Prepare or enqueue a same-container where-DDP continuation segment.

Activate conda pllm. Use prepare --resume-checkpoint ABS --stop-after N, inspect
the resulting plan, then enqueue --stage ABS. The initial enqueue requires the
actual old-group cleanup receipt. Later enqueues require successful native
restore, eight exits, and the previous checkpoint. No resource submit/restart.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import time
import uuid

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/v7_formal_eight_gpu_20261002'
BASE = OWN/'where_continuation_20261006'
POOL = ROOT/'output/immune_research_pool_c20250601_20261005'
SOURCE = BASE/'source_snapshot_v1'
MANIFEST = BASE/'source_manifest_v1.json'
TRANSITION = BASE/'transition_v1.json'
RUN = BASE/'run'
DATA = ROOT/'data/prepared/immune_v7_test_scope_final_20261002'
TRANSPORT = OWN/'transport_preload_v1/formal_pool_transport.py'
WRAPPER = BASE/'wrapper_v1/formal_pool_job.py'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    require(path.is_file() and not path.is_symlink(), 'Regular metadata file required: '+str(path))
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def entry_api():
    publication = read(OWN/'entry_20261005/publication.json')
    for name, expected in publication['source_sha256'].items():
        require(sha(Path(name)) == expected, 'Original pool entry drift')
    return runpy.run_path(publication['frozen_entry']['path'], run_name='_where_original_entry')


def validate_plan(plan):
    driver = runpy.run_path(str(SOURCE/'scripts/research/run_v7_formal.py'), run_name='_where_driver_metadata')
    args = driver['parse_args'](plan['work']['driver_argv'])
    driver['verify_sources'](args)
    wrapper = runpy.run_path(str(WRAPPER), run_name='_where_wrapper_metadata')
    wrapper['prepare'](plan)
    interval = wrapper['validate_segment'](plan, args, binding_digest=driver['digest'])
    for item in (plan['wrapper'], plan['transport'], plan['pool_source'],
                 *[plan['work'][k] for k in ('selection', 'audit', 'driver', 'source_manifest')]):
        require(sha(Path(item['path'])) == item['sha256'], 'Plan dependency changed')
    audit = read(Path(plan['work']['audit']['path']))
    require(audit['status'] == 'passed' and audit['source_manifest_sha256'] == sha(MANIFEST),
            'Original source-bound parent audit required')
    require(all(sha(DATA/name) == value for name, value in audit['metadata'].items()), 'Audited parent metadata changed')
    return interval


def prepare(checkpoint, target):
    entry = entry_api()
    require(checkpoint.is_absolute() and checkpoint == checkpoint.resolve(), 'Canonical checkpoint required')
    prior = read(checkpoint/'trainer_state.json')['global_step']
    require(0 < prior < target <= 25042 and target-prior <= 5000, 'Invalid bounded continuation')
    transition = read(TRANSITION)
    require(checkpoint.parent == RUN or str(checkpoint) == transition['anchor_checkpoint'], 'Unbound resume owner')
    stage = BASE/'launches'/f'segment_{target}'
    require(not stage.exists(), 'Do not overwrite or replay a prepared segment')
    identity_id = read(OWN/'entry_20261005/identity_enqueue.json')['job_id']
    identity = read(POOL/'logs'/f'{identity_id}.pool_parent_runtime_observation.json')
    original = read(BASE/'initial_driver_argv.json')
    command = list(original)
    command[command.index('--stop-after')+1] = str(target)
    command[command.index('--resume-from-checkpoint')+1] = str(checkpoint)
    tickets = {}
    for alias in entry['ALIASES']:
        job = f'{time.time_ns()}_{alias}_{uuid.uuid4().hex[:8]}'
        tickets[job] = dict(variant=alias, mode='train', steps=target-prior,
                           data_dir=str(DATA), resume_from_checkpoint=str(checkpoint))
    choice = OWN/'launch_20261005/initial_model_choice.json'
    work = dict(selection=ref(choice), audit=ref(BASE/'parent_gate.stdout.json'),
        driver=ref(SOURCE/'scripts/research/run_v7_formal.py'), source_manifest=ref(MANIFEST),
        source_root=str(SOURCE), repo_root=str(ROOT), data_dir=str(DATA), output_dir=str(RUN),
        driver_argv=command, variant='f011', seed=42, micro_batch=32, grad_accum=1,
        global_batch=256, max_steps=25042)
    plan = dict(schema='v7.formal.pool_group_plan.v1', pool_dir=str(POOL),
        meeting_dir=str(stage/'meeting'), group_id=f'v7_where_world8_20261006_step{target}',
        pool_process=identity['pool_process'], pool_argv=identity['pool_argv'],
        pool_source=ref(ROOT/'scripts/immune_research_pool.py'), wrapper=ref(WRAPPER),
        transport=ref(TRANSPORT), work=work, tickets=tickets, join_seconds=300,
        startup_seconds=180, duration_seconds=80000, cleanup_margin_seconds=120, segment_steps=target)
    interval = validate_plan(plan)
    stage.mkdir(parents=True)
    write_new(stage/'plan.json', plan)
    write_new(stage/'enqueue_receipt.json', dict(state='prepared', at_utc=datetime.now(timezone.utc).isoformat(),
        plan=ref(stage/'plan.json'), prior_step=prior, stop_after=target, job_ids=list(tickets),
        enqueued_job_ids=[], formal_training_started=False, launcher=ref(Path(__file__).resolve())))
    print(json.dumps(dict(status='prepared_not_enqueued', stage=str(stage), interval=interval)), flush=True)


def cleanup_checked(stage, success):
    receipt = read(stage/'enqueue_receipt.json')
    require(receipt['state'] == 'enqueued' and len(receipt['job_ids']) == 8, 'Eight original tickets required')
    for job in receipt['job_ids']:
        result = read(POOL/'results'/f'{job}.json')
        require(result['state'] in ('succeeded', 'failed') and type(result['returncode']) is int,
                'Prior actual result is not terminal')
        if success:
            require(result['state'] == 'succeeded' and result['returncode'] == 0, 'Prior continuation failed')
    for rank in range(8):
        report = read(stage/f'meeting/coordinator/rank{rank}_parent_receipt.json')
        cleanup = report['cleanup']
        require(cleanup['cleanup_completed'] and cleanup['wait_observed']
                and cleanup['process_absent'] and cleanup['process_group_gone'], 'Owned rank cleanup incomplete')
        if success:
            require(report['error'] is None and cleanup['returncode'] == 0, 'Prior rank unsuccessful')
    return receipt


def enqueue(stage):
    require(stage.is_absolute() and stage == stage.resolve() and stage.parent == BASE/'launches', 'Wrong stage')
    receipt = read(stage/'enqueue_receipt.json')
    require(receipt['state'] == 'prepared' and not receipt['enqueued_job_ids']
            and not (stage/'admission_commit.json').exists(), 'Refusing duplicate publication')
    plan = read(stage/'plan.json')
    require(ref(stage/'plan.json') == receipt['plan'], 'Prepared plan changed')
    interval = validate_plan(plan)
    require(not (BASE/'STOP').exists() and not (POOL/'STOP').exists() and not (POOL/'STOP_NOW').exists(), 'Explicit stop present')
    require(not list((POOL/'running').glob('*.json')) and not list((POOL/'pending').glob('*.json')), 'Pool still busy')
    workers = [read(p) for p in (POOL/'workers').glob('*.json')]
    require(len(workers) == 8 and {str(w['device']) for w in workers} == set(map(str, range(8)))
            and all(w['state'] == 'standby_load' and 0 <= time.time()-w['time'] < 15 for w in workers),
            'Eight released fresh workers required')
    transition = read(TRANSITION)
    if plan['work']['driver_argv'][plan['work']['driver_argv'].index('--resume-from-checkpoint')+1] == transition['anchor_checkpoint']:
        stopped = read(BASE/'OLD_TRAINING_STOPPED.json')
        require(stopped['status'] == 'old_group_terminal_and_owned_processes_cleaned'
                and stopped['transition'] == ref(TRANSITION), 'Exact authorized stop receipt required')
        cleanup_checked(Path(stopped['old_stage']), success=False)
    else:
        previous_stage = BASE/'launches'/f"segment_{interval['prior_step']}"
        cleanup_checked(previous_stage, success=True)
        previous_marker = read(RUN/f"SEGMENT_{interval['prior_step']}.json")
        require(previous_marker['step'] == interval['prior_step'], 'Previous segment marker differs')
        previous_receipt = read(previous_stage/'enqueue_receipt.json')
        restore = read(RUN/f"RESUME_{previous_receipt['prior_step']}_TO_{interval['prior_step']}.json")
        native = runpy.run_path(str(SOURCE/'scripts/research/formal_resume_observation.py'), run_name='_native_resume_verify')
        require(native['complete_receipt'](restore['ranks']) == restore, 'Native eight-rank resume evidence differs')
    entry = entry_api()
    pool = runpy.run_path(str(ROOT/'scripts/immune_research_pool.py'), run_name='_existing_pool_api')
    catalog_path = entry['CATALOG'];catalog = read(catalog_path)
    require(not set(plan['tickets']) & set(catalog['jobs']), 'Ticket already registered')
    for job, spec in plan['tickets'].items():
        pool['validate'](spec);catalog['jobs'][job] = receipt['plan']
    pool['write_json'](catalog_path, catalog)
    for job, spec in plan['tickets'].items():
        entry['formal_api']()['prepare'](pool['command'](POOL, job, spec)[3:])
    write_new(stage/'admission_commit.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        state='publication_committed', transition=ref(TRANSITION), job_ids=list(plan['tickets'])))
    for job, spec in plan['tickets'].items():
        require(not (POOL/'pending'/f'{job}.json').exists(), 'Duplicate queue ticket')
        pool['write_json'](POOL/'pending'/f'{job}.json', spec)
        receipt['enqueued_job_ids'].append(job)
        pool['write_json'](stage/'enqueue_receipt.json', receipt)
    receipt['state'] = 'enqueued'
    receipt['at_utc'] = datetime.now(timezone.utc).isoformat()
    pool['write_json'](stage/'enqueue_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    a = commands.add_parser('prepare');a.add_argument('--resume-checkpoint', type=Path, required=True);a.add_argument('--stop-after', type=int, required=True)
    b = commands.add_parser('enqueue');b.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.resume_checkpoint, args.stop_after)
    else:
        enqueue(args.stage)
