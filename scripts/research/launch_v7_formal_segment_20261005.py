"""Prepare or enqueue one bounded segment in the already allocated eight-GPU pool.

Activate conda pllm. First segment:
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/launch_v7_formal_segment_20261005.py --stop-after 8 --enqueue
Subsequent segments additionally require --resume-step N. This never submits or
restarts a platform task, changes a data certificate, or overwrites a prior plan.
Use --attempt N for a fresh repaired attempt and --previous-stage ABSOLUTE_PATH
when the successful preceding segment is an explicitly named later attempt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import time
import uuid

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/v7_formal_eight_gpu_20261002'
POOL = ROOT/'output/immune_research_pool_c20250601_20261005'
OLD_POOL = ROOT/'output/immune_research_pool_c20250601_20260930'
SOURCE = OWN/'source_snapshot_v3'
MANIFEST = OWN/'source_manifest_v3.json'
MANIFEST_SHA = '992bc3c5718b71c379c7a2ee4f39d8e5866f974266500d2c2ec69d5ab4db700b'
DATA = ROOT/'data/prepared/immune_v7_test_scope_final_20261002'
RUN = ROOT/'output/v7_initial_training_20261005/run'
LAUNCHES = OWN/'launch_20261005'
HORIZON = 25042


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    require(path.is_file() and not path.is_symlink(), 'Regular metadata file required: '+str(path))
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def model_choice():
    catalog_path = ROOT/'output/v7_large_batch_ablation_20261002/jobs.json'
    catalog = read(catalog_path)
    rows = []
    for job in catalog['jobs']:
        if job['variant'] not in ('f001', 'f011'):
            continue
        directory = OLD_POOL/'runs'/job['job_id']
        summary, complete = read(directory/'summary.json'), read(directory/'COMPLETE.json')
        result = read(OLD_POOL/'results'/f'{job["job_id"]}.json')
        require(result['state'] == 'succeeded' and result['returncode'] == 0
                and complete['global_step'] == summary['step'] == 3750
                and summary['records'] == summary['unique_train_records'] == 120000,
                'Model selection requires actual complete matching pilot results')
        require(complete['summary_sha256'] == sha(directory/'summary.json')
                and complete['provenance_sha256'] == sha(directory/'run_provenance.json'),
                'Pilot evidence changed')
        provenance = read(directory/'run_provenance.json')
        require(provenance['micro_batch'] == 32 and provenance['grad_accum'] == 1
                and provenance['global_batch'] == 32 and provenance['variant'] == job['variant'],
                'Pilot comparison protocol differs')
        rows.append(dict(variant=job['variant'], seed=job['seed'],
            aa_ce=summary['validation']['token_pooled']['aa_loss'],
            full_mask_accuracy=summary['validation']['token_pooled']['mask_levels']['mask100']['aa_accuracy'],
            pilot_peak_allocated_gib=summary['peak_allocated_gib'],
            evidence=[ref(directory/name) for name in ('summary.json', 'COMPLETE.json', 'run_provenance.json')]
                + [ref(OLD_POOL/'results'/f'{job["job_id"]}.json')]))
    require({(x['variant'], x['seed']) for x in rows} ==
            {(variant, seed) for variant in ('f001', 'f011') for seed in (42, 43)}, 'Two complete seeds required')
    return dict(schema='v7.formal.model_choice.v1', status='selected', variant='f011',
        phase='initial_formal_version', final_ablation_winner=False, evidence=rows,
        rationale='User requested initial training now; completed matched pilots favor f011 reconstruction CE. Full matrix remains incomplete.',
        biological_quality_claim=False, remaining_ablation_count=14, catalog=ref(catalog_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-after', type=int, required=True)
    parser.add_argument('--resume-step', type=int, default=0)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--previous-stage', type=Path)
    parser.add_argument('--transport', type=Path)
    parser.add_argument('--enqueue', action='store_true')
    args = parser.parse_args()
    require(args.attempt >= 1, 'Positive distinct attempt required')
    stage = LAUNCHES/(f'segment_{args.stop_after}' + (f'_attempt{args.attempt}' if args.attempt > 1 else ''))
    require(not stage.exists(), 'Refusing to overwrite or replay a prior attempt')
    if args.attempt > 1:
        failed_stage = LAUNCHES/(f'segment_{args.stop_after}' + (f'_attempt{args.attempt-1}' if args.attempt > 2 else ''))
        failed = read(failed_stage/'enqueue_receipt.json')
        require(read(failed_stage/'meeting/coordinator/state.json')['terminal'] == 'aborted',
                'Previous attempt must be terminal aborted')
        for job_id in failed['job_ids']:
            require(read(POOL/'results'/f'{job_id}.json')['state'] == 'failed', 'Previous attempt is not fully terminal')
        for rank in range(8):
            cleanup = read(failed_stage/f'meeting/coordinator/rank{rank}_parent_receipt.json')['cleanup']
            require(cleanup['cleanup_completed'] and cleanup['process_absent'] and cleanup['process_group_gone'],
                    'Previous owned processes are not fully cleaned')
    require(0 <= args.resume_step < args.stop_after <= HORIZON
            and args.stop_after-args.resume_step <= 5000, 'Bounded additional update budget required')
    require(args.resume_step > 0 or args.stop_after == 8, 'First actual eight-rank segment must stop at eight')
    require(not list((POOL/'running').glob('*.json')) and not list((POOL/'pending').glob('*.json')),
            'All eight pool slots must be idle before preparing this segment')
    workers = [read(path) for path in (POOL/'workers').glob('*.json')]
    require(len(workers) == 8 and {str(x['device']) for x in workers} == set(map(str, range(8)))
            and all(x['state'] == 'standby_load' and 0 <= time.time()-x['time'] < 15 for x in workers),
            'Eight fresh actual standby workers required')
    entry_manifest = read(OWN/'entry_20261005/publication.json')
    for name, expected in entry_manifest['source_sha256'].items():
        require(sha(Path(name)) == expected, 'Published entry changed: '+name)
    entry = runpy.run_path(entry_manifest['frozen_entry']['path'], run_name='_new_pool_entry_prepare')
    identity_job = read(OWN/'entry_20261005/identity_enqueue.json')['job_id']
    identity_result = read(POOL/'results'/f'{identity_job}.json')
    require(identity_result['state'] == 'succeeded' and identity_result['returncode'] == 0,
            'Actual container identity capture must have completed')
    observation_path = POOL/'logs'/f'{identity_job}.pool_parent_runtime_observation.json'
    observation = read(observation_path)
    require(observation['pool_dir'] == str(POOL), 'Captured identity belongs to another container')
    for name, expected in observation['source_sha256'].items():
        require(sha(Path(name)) == expected, 'Identity capture source changed')
    if args.resume_step:
        previous_stage = args.previous_stage or LAUNCHES/f'segment_{args.resume_step}'
        require(previous_stage.is_absolute() and previous_stage == previous_stage.resolve()
                and previous_stage.parent == LAUNCHES, 'Explicit previous stage must belong to this launch series')
        previous = read(previous_stage/'enqueue_receipt.json')
        require(previous['stop_after'] == args.resume_step, 'Previous stage step differs')
        require(previous['state'] == 'enqueued' and len(previous['job_ids']) == 8, 'Prior eight-ticket segment required')
        for job_id in previous['job_ids']:
            result = read(POOL/'results'/f'{job_id}.json')
            require(result['state'] == 'succeeded' and result['returncode'] == 0, 'Prior rank/controller did not finish successfully')
        require(read(RUN/f'SEGMENT_{args.resume_step}.json')['step'] == args.resume_step, 'Actual prior segment missing')
    else:
        require(not RUN.exists(), 'Initial formal output already exists')
    driver = runpy.run_path(str(SOURCE/'scripts/research/run_v7_formal.py'), run_name='_frozen_formal_prepare')
    command = ['--source-root', str(SOURCE), '--repo-root', str(ROOT), '--data-dir', str(DATA),
        '--output-dir', str(RUN), '--source-manifest', str(MANIFEST), '--source-manifest-sha256', MANIFEST_SHA,
        '--variant', 'f011', '--seed', '42', '--micro-batch', '32', '--grad-accum', '1', '--global-batch', '256',
        '--max-steps', str(HORIZON), '--warmup-steps', '1000', '--learning-rate', '0.0001',
        '--eval-every', '1000', '--save-every', '1000', '--valid-per-source', '256', '--stop-after', str(args.stop_after)]
    checkpoint = RUN/f'checkpoint-{args.resume_step}'
    if args.resume_step:
        command += ['--resume-from-checkpoint', str(checkpoint)]
    parsed = driver['parse_args'](command)
    driver['verify_sources'](parsed)
    audit_path = OWN/'parent_gate_v3.stdout.json'
    audit = read(audit_path)
    require(audit['status'] == 'passed' and audit['source_manifest_sha256'] == MANIFEST_SHA,
            'Existing actual parent gate differs')
    for name, expected in audit['metadata'].items():
        require(sha(DATA/name) == expected, 'Current data metadata differs from passed audit')
    for name, expected in audit['executed_sources'].items():
        require(sha(Path(name)) == expected, 'Actual audit helper source changed')
    choice = model_choice()
    LAUNCHES.mkdir(exist_ok=True)
    choice_path = LAUNCHES/'initial_model_choice.json'
    if choice_path.exists():
        require(read(choice_path) == choice, 'Initial model choice/evidence changed')
    else:
        write_new(choice_path, choice)
    stage.mkdir(exist_ok=False)
    tickets = {}
    for alias in entry['ALIASES']:
        job_id = f'{time.time_ns()}_{alias}_{uuid.uuid4().hex[:8]}'
        tickets[job_id] = dict(variant=alias, mode='train', steps=args.stop_after-args.resume_step, data_dir=str(DATA))
        if args.resume_step:
            tickets[job_id]['resume_from_checkpoint'] = str(checkpoint)
    wrapper_path = OWN/'pool_launcher_source_v1/formal_pool_job.py'
    transport_path = args.transport or SOURCE/'scripts/research/formal_pool_transport.py'
    require(transport_path.is_absolute() and transport_path == transport_path.resolve()
            and transport_path.is_relative_to(OWN), 'Transport must be an explicit frozen local source')
    work = dict(selection=ref(choice_path), audit=ref(audit_path), driver=ref(SOURCE/'scripts/research/run_v7_formal.py'),
        source_manifest=ref(MANIFEST), source_root=str(SOURCE), repo_root=str(ROOT), data_dir=str(DATA),
        output_dir=str(RUN), driver_argv=command, variant='f011', seed=42,
        micro_batch=32, grad_accum=1, global_batch=256, max_steps=HORIZON)
    plan = dict(schema='v7.formal.pool_group_plan.v1', pool_dir=str(POOL), meeting_dir=str(stage/'meeting'),
        group_id=f'v7_f011_world8_20261005_step{args.stop_after}_attempt{args.attempt}', pool_process=observation['pool_process'],
        pool_argv=observation['pool_argv'], pool_source=ref(ROOT/'scripts/immune_research_pool.py'),
        wrapper=ref(wrapper_path), transport=ref(transport_path), work=work, tickets=tickets,
        join_seconds=300, startup_seconds=180, duration_seconds=7200 if args.stop_after <= 16 else 80000,
        cleanup_margin_seconds=120, segment_steps=args.stop_after)
    wrapper = runpy.run_path(str(wrapper_path), run_name='_frozen_wrapper_prepare')
    wrapper['prepare'](plan)
    wrapper['validate_segment'](plan, parsed, binding_digest=driver['digest'])
    plan_path = stage/'plan.json'
    write_new(plan_path, plan)
    write_new(stage/'source_identity.json', dict(launcher=ref(Path(__file__).resolve()), identity=ref(observation_path),
        plan=ref(plan_path), data_sampling='epoch_random_natural; context approximately 4.65%; no added negatives',
        schedule_horizon=HORIZON, nominal_epoch_exposures=HORIZON*256,
        batch='32x1x8=256; pilot micro32 passed; actual DDP memory remains to be measured'))
    pool_api = runpy.run_path(str(ROOT/'scripts/immune_research_pool.py'), run_name='_pool_enqueue_helpers')
    catalog_path = entry['CATALOG']
    catalog = read(catalog_path) if catalog_path.exists() else dict(schema='v7.formal.pool_job_catalog.v1', jobs={})
    require(not set(catalog['jobs']) & set(tickets), 'A ticket ID is already registered')
    for job_id, job in tickets.items():
        pool_api['validate'](job)
        catalog['jobs'][job_id] = ref(plan_path)
    pool_api['write_json'](catalog_path, catalog)
    for job_id, job in tickets.items():
        argv = pool_api['command'](POOL, job_id, job)[3:]
        entry['formal_api']()['prepare'](argv)
    receipt = dict(at_utc=datetime.now(timezone.utc).isoformat(), state='prepared', job_ids=list(tickets),
        enqueued_job_ids=[], plan=ref(plan_path), pool_dir=str(POOL), stop_after=args.stop_after,
        prior_step=args.resume_step, formal_training_started=False)
    pool_api['write_json'](stage/'enqueue_receipt.json', receipt)
    if args.enqueue:
        for job_id, job in tickets.items():
            require(not (POOL/'pending'/f'{job_id}.json').exists(), 'Refusing duplicate pending ticket')
            pool_api['write_json'](POOL/'pending'/f'{job_id}.json', job)
            receipt['enqueued_job_ids'].append(job_id)
            pool_api['write_json'](stage/'enqueue_receipt.json', receipt)
        receipt['state'] = 'enqueued'
        pool_api['write_json'](stage/'enqueue_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
