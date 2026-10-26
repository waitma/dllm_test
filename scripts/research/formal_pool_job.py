"""Join eight existing pool job slots before one formal transport publication.

Activate conda pllm. Artificial CPU checks: python -B /absolute/repo/scripts/
research/formal_pool_job.py --self-test. No listener/GPU/job/data read is used.
Future early dispatcher must exec this file (preserving the pool Popen PID):
python -I -B formal_pool_job.py --plan /absolute/plan.json --plan-sha SHA
--job-id ACTUAL_POOL_JOB_ID. No real plan/queue/registry is produced by this file.
The plan binds eight ticket IDs/specs, not ticket-to-GPU assignments. Actual
CUDA masks determine ranks. The legacy transport worker_process is the slot's
independent job/controller PID; all eight share the same real parent pool PID.
Ticket steps count new updates (at most 5000); segment_steps is the cumulative
driver endpoint. Resume accounting reads existing SHA-bound JSON metadata only.
"""
import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import signal
import socket
import sys
import time
import uuid

SCHEMA = 'v7.formal.pool_group_plan.v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Regular metadata file required')
    return json.loads(path.read_text())


def checked(ref):
    path = Path(ref['path'])
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink()
            and sha(path) == ref['sha256'], 'Bound file differs: '+str(path))
    return path


def prepare(spec):
    """Pure metadata construction; no actual plan or admission is written."""
    plan = copy.deepcopy(spec)
    require(plan.get('schema') == SCHEMA, 'Group plan schema differs')
    require(set(plan) == {'schema', 'pool_dir', 'meeting_dir', 'group_id', 'pool_process', 'pool_argv',
        'pool_source', 'wrapper', 'transport', 'work', 'tickets', 'join_seconds', 'startup_seconds',
        'duration_seconds', 'cleanup_margin_seconds', 'segment_steps'}, 'Unexpected plan fields')
    for key in ('pool_dir', 'meeting_dir'):
        require(Path(plan[key]).is_absolute() and '..' not in Path(plan[key]).parts, 'Absolute plan paths required')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', plan['group_id']) is not None, 'Group ID required')
    tickets = plan['tickets']
    require(type(tickets) is dict and len(tickets) == 8, 'Eight explicit ticket job IDs/specs required')
    require(all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', key) and type(value) is dict
                for key, value in tickets.items()), 'Ticket fields differ')
    require(type(plan['pool_argv']) is list and plan['pool_argv'], 'Actual pool command binding required')
    for key in ('join_seconds', 'startup_seconds', 'duration_seconds', 'cleanup_margin_seconds', 'segment_steps'):
        require(type(plan[key]) is int and plan[key] > 0, 'Positive bounded budgets required')
    require(plan['join_seconds'] <= 300 and plan['startup_seconds'] <= 180
        and plan['cleanup_margin_seconds'] >= 120 and plan['segment_steps'] <= plan['work']['max_steps'], 'Budget limits differ')
    require(plan['work']['variant'] is not None, 'Explicit selected model required')
    require(not Path(plan['work']['output_dir']).is_relative_to(Path(plan['pool_dir'])/'runs'),
            'Shared driver output must be separate from per-ticket pool runs')
    ticket_protocol(plan)
    return plan


def ticket_protocol(plan):
    """Eight pool tickets must describe the same bounded additional work."""
    common = None
    require(type(plan['tickets']) is dict and len(plan['tickets']) == 8, 'Eight tickets required')
    for ticket in plan['tickets'].values():
        require(type(ticket) is dict and {'variant', 'mode', 'steps', 'data_dir'} <= set(ticket)
                and set(ticket) <= {'variant', 'mode', 'steps', 'data_dir', 'resume_from_checkpoint'},
                'Unexpected ticket fields')
        require(ticket['mode'] == 'train' and type(ticket['steps']) is int and 1 <= ticket['steps'] <= 5000
                and ticket['data_dir'] == plan['work']['data_dir'], 'Ticket update budget/data differ')
        resume = ticket.get('resume_from_checkpoint')
        require(resume is None if 'resume_from_checkpoint' not in ticket else isinstance(resume, str) and bool(resume),
                'Resume must be absent or an explicit checkpoint path')
        protocol = (ticket['steps'], ticket['data_dir'], resume)
        require(common is None or protocol == common, 'Eight ticket steps/data/resume must be identical')
        common = protocol
    return common


def validate_segment(plan, args, *, binding_digest):
    """Bind ticket delta to the parsed frozen driver and existing checkpoint JSON.

    This is an early metadata check, not native optimizer/RNG restoration. The
    unchanged driver still validates every checkpoint byte and binary state.
    No step is inferred from a directory name and no checkpoint is written.
    """
    updates, data_dir, ticket_resume = ticket_protocol(plan)
    work, target = plan['work'], plan['segment_steps']
    require(type(target) is int and target == (args.stop_after or args.max_steps)
            and 0 < target <= args.max_steps == work['max_steps'], 'Cumulative driver target/horizon differs')
    for key in ('source_root', 'repo_root', 'data_dir', 'output_dir', 'variant', 'seed',
                'micro_batch', 'grad_accum', 'global_batch', 'max_steps'):
        require(str(getattr(args, key)) == str(work[key]), 'Driver work differs: '+key)
    require(str(args.source_manifest) == work['source_manifest']['path']
            and args.source_manifest_sha256 == work['source_manifest']['sha256'], 'Driver source manifest differs')
    actual_resume = None if args.resume_from_checkpoint is None else str(args.resume_from_checkpoint)
    require(ticket_resume == actual_resume, 'Ticket resume differs from actual frozen driver argument')
    prior, references = 0, {}
    if actual_resume is not None:
        checkpoint = Path(actual_resume)
        require(checkpoint.is_absolute() and checkpoint == checkpoint.resolve()
                and checkpoint.is_dir() and not checkpoint.is_symlink()
                and checkpoint.parent == Path(work['output_dir']), 'Resume must be an existing checkpoint in shared output')
        sidecar_path = checkpoint/'formal_checkpoint.json'
        references[str(sidecar_path)] = sha(sidecar_path)
        sidecar = read(sidecar_path)
        require(sidecar.get('schema') == 'v7.formal.native_checkpoint.v1' and sidecar.get('world_size') == 8
                and sidecar.get('writer_rank') == 0 and sidecar.get('rank_acknowledgments') == list(range(8)),
                'Existing native checkpoint sidecar differs')
        files = sidecar.get('files', {})
        metadata = {}
        for name in ('trainer_state.json', 'run_provenance.json'):
            expected = files.get(name)
            require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Missing checkpoint metadata SHA')
            path = checked({'path': str(checkpoint/name), 'sha256': expected})
            references[str(path)] = expected
            metadata[name] = read(path)
        prior = metadata['trainer_state.json'].get('global_step')
        binding = metadata['run_provenance.json']
        require(type(prior) is int and 0 < prior < target and sidecar.get('step') == prior,
                'Existing checkpoint step/sidecar/target disagree')
        # Native driver hashes use its own serialization, not the compact plan digest.
        require(sidecar.get('binding_sha256') == binding_digest(binding)
                and binding.get('source_manifest_sha256') == work['source_manifest']['sha256']
                and binding.get('world_size') == 8, 'Prior checkpoint binding differs')
        variants = {'r20': 'r20_refined_control', 'r21': 'r21_refined_head_gate',
                    'r24': 'r24_refined_fixed_scale', 'r26': 'r26_refined_local_global'}
        require(binding.get('variant') == variants.get(work['variant'], work['variant'])
                and all(str(binding.get(key)) == str(work[key]) for key in
                    ('source_root', 'repo_root', 'data_dir', 'seed', 'micro_batch', 'grad_accum', 'global_batch')),
                'Prior checkpoint model/data/batch differ')
        require(binding.get('schedule') == dict(max_steps=args.max_steps, warmup_steps=args.warmup_steps,
                learning_rate=args.learning_rate, scheduler='cosine', eval_every=args.eval_every, save_every=args.save_every),
                'Prior checkpoint fixed schedule differs')
        require(all(sha(Path(path)) == expected for path, expected in references.items()),
                'Resume metadata changed during verification')
    require(target-prior == updates, 'Ticket additional updates differ from target minus saved step')
    return dict(prior_step=prior, target_step=target, additional_updates=updates, checkpoint_metadata_sha256=references)


def atomic(path, value, coordinator):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        coordinator['_exclusive_json'](temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(directory, timeout=2.):
    deadline = time.monotonic()+timeout
    with (directory/'join.lock').open('a') as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                require(time.monotonic() < deadline, 'Slot publication lock deadline expired')
                time.sleep(.02)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def option(argv, name, default=None):
    values = [argv[index+1] for index, value in enumerate(argv[:-1]) if value == name]
    values += [value.split('=', 1)[1] for value in argv if value.startswith(name+'=')]
    require(len(values) <= 1, 'Ambiguous pool option '+name)
    require(values or default is not None, 'Missing pool option '+name)
    return values[0] if values else default


def invocation_source(argv, cwd):
    """Resolve the actual pool script against its observed /proc working directory."""
    require(type(argv) is list and argv.count('work') == 1, 'One pool work invocation required')
    index = argv.index('work')
    require(index >= 2 and all(isinstance(value, str) and value for value in argv),
            'Actual interpreter/script/action arguments required')
    cwd, script = Path(cwd), Path(argv[index-1])
    require(cwd.is_absolute() and cwd == cwd.resolve() and script.name == 'immune_research_pool.py',
            'Observed pool working directory/script differ')
    return (script if script.is_absolute() else cwd/script).resolve()


def resource_deadline(plan, claims, now_monotonic, now_boottime, ticks_per_second):
    """Process starts precede pool/job deadlines; include suspend time conservatively."""
    argv = plan['pool_argv']
    require(int(option(argv, '--expected-gpus', '8')) == 8, 'Pool must retain eight slots')
    require(Path(option(argv, '--pool-dir')).resolve() == Path(plan['pool_dir']), 'Pool command points elsewhere')
    maximum, job_seconds = int(option(argv, '--max-seconds', '259200')), int(option(argv, '--job-seconds', '7200'))
    require(min(maximum, job_seconds, ticks_per_second) > 0, 'Pool limits must be positive')
    starts = [plan['pool_process']['start_ticks']/ticks_per_second+maximum]
    starts += [claim['controller']['start_ticks']/ticks_per_second+job_seconds for claim in claims]
    return now_monotonic+min(starts)-now_boottime


def verify_claims(plan, claims, statuses, running, pending, now_mono, now_wall, alive, complete=False):
    """Validate actual ownership snapshots; ticket aliases never select rank."""
    expected = set(plan['tickets'])
    require(set(running) <= expected and set(pending) <= expected, 'Unrelated job present before group publication')
    require(not set(running) & set(pending), 'Job occurs in both running and pending')
    for job, spec in {**running, **pending}.items():
        require(spec == plan['tickets'][job], 'Pool ticket spec changed')
    require(alive(plan['pool_process']), 'Shared pool process is no longer the bound process')
    require(len({claim['rank'] for claim in claims}) == len(claims)
        and len({claim['job_id'] for claim in claims}) == len(claims)
        and len({claim['controller']['pid'] for claim in claims}) == len(claims), 'Duplicate slot/job/controller')
    for claim in claims:
        rank, job, process = claim['rank'], claim['job_id'], claim['controller']
        require(claim['plan_digest'] == digest(plan), 'Claim belongs to another plan')
        require(type(rank) is int and 0 <= rank < 8 and claim['mask'] == str(rank) and job in running, 'Claim rank/mask/job differs')
        require(claim['shared_pool'] == plan['pool_process'] and process['parent_pid'] == plan['pool_process']['pid']
            and process['parent_start_ticks'] == plan['pool_process']['start_ticks']
            and process['hostname'] == plan['pool_process']['hostname'] and process['boot_id'] == plan['pool_process']['boot_id']
            and process['process_group_id'] == process['pid'] and alive(process), 'Controller is not the actual pool Popen child')
        require(0 <= now_mono-claim['observed_monotonic'] <= 30, 'Claim freshness expired')
        status = statuses.get(claim['worker_id'], {})
        require(status.get('state') == 'running' and status.get('job_id') == job and status.get('pid') == process['pid']
            and str(status.get('device')) == claim['mask'] and status.get('job') == plan['tickets'][job], 'Pool worker current pid/job differs')
        require(type(status.get('heartbeat')) in (int, float) and 0 <= now_wall-status['heartbeat'] <= 30, 'Pool worker heartbeat is stale')
    if complete:
        require(len(claims) == 8 and set(running) == expected and not pending
            and {claim['rank'] for claim in claims} == set(range(8)), 'All eight slots must already hold this group; pending must be empty')
        require(len({claim['hardware']['uuid'] for claim in claims}) == 8, 'Actual GPU UUIDs are not unique')
    return sorted(claims, key=lambda item: item['rank'])


def pool_snapshot(plan):
    pool = Path(plan['pool_dir'])
    require(not (pool/'STOP').exists() and not (pool/'STOP_NOW').exists(), 'Pool stopped before publication')
    return ({path.stem: read(path) for path in (pool/'workers').glob('*.json')},
            {path.stem: read(path) for path in (pool/'running').glob('*.json')},
            {path.stem: read(path) for path in (pool/'pending').glob('*.json')})


def abort(directory, plan_sha, message, coordinator):
    try:
        coordinator['_exclusive_json'](directory/'ABORT.json', {'plan_sha256': plan_sha, 'error': str(message)[-2000:]})
    except FileExistsError:
        require(read(directory/'ABORT.json')['plan_sha256'] == plan_sha, 'Abort belongs to another plan')


def abort_fence(directory, plan_sha):
    if (directory/'ABORT.json').exists():
        record = read(directory/'ABORT.json')
        require(record['plan_sha256'] == plan_sha, 'Abort plan differs')
        raise ValueError('Slot rendezvous aborted: '+record['error'])


def build_intent(plan, claims, transport):
    members = [dict(rank=claim['rank'], worker_id=claim['worker_id'], job_id=claim['job_id'],
        gpu_identity=claim['hardware']['uuid'], visible_device=claim['mask'], hardware=claim['hardware'],
        worker_process=claim['controller']) for claim in sorted(claims, key=lambda item: item['rank'])]
    return transport['prepare'](dict(group_id=plan['group_id'], group_directory=str(Path(plan['meeting_dir'])/'coordinator'),
        hostname=plan['pool_process']['hostname'], boot_id=plan['pool_process']['boot_id'], members=members,
        work=plan['work'], transport_sources={**transport['FIXED'], plan['transport']['path']: plan['transport']['sha256']},
        startup_seconds=plan['startup_seconds'], heartbeat_seconds=30, duration_seconds=plan['duration_seconds']))


def publish(directory, plan, plan_sha, claims, transport, coordinator):
    """Caller holds join.lock and has just revalidated all eight real slots."""
    intent = build_intent(plan, claims, transport)
    now = time.monotonic()
    hard_deadline = resource_deadline(plan, claims, now, time.clock_gettime(time.CLOCK_BOOTTIME), os.sysconf('SC_CLK_TCK'))
    group_deadline = min(now+plan['duration_seconds'], hard_deadline-plan['cleanup_margin_seconds'])
    require(group_deadline > now+plan['startup_seconds'], 'Insufficient remaining pool/job resource budget')
    reservation = dict(schema='v7.formal.all_eight_pool_reservation.v1', status='reserved_all_eight',
        intent_sha256=transport['digest'](intent), hostname=intent['hostname'], boot_id=intent['boot_id'],
        members=intent['members'], reserved_monotonic=now, group_deadline_monotonic=group_deadline,
        previous_jobs_finished=True, exclusive=True)
    transport['validate_reservation'](intent, reservation, now)
    coordinator['_exclusive_json'](directory/'intent.json', intent)
    coordinator['_exclusive_json'](directory/'reservation.json', reservation)
    # The final commit record is the only publication point visible to waiters.
    commit = dict(schema='v7.formal.slot_publication.v1', plan_sha256=plan_sha,
        intent_sha256=sha(directory/'intent.json'), reservation_sha256=sha(directory/'reservation.json'),
        hard_deadline_monotonic=hard_deadline, published_monotonic=now,
        semantics='eight controllers already hold all pool slots; atomic group publication, not a global queue transaction')
    coordinator['_exclusive_json'](directory/'PUBLISHED.json', commit)
    return commit


def completion_evidence(plan, intent, status, group, shared, driver_digest):
    """Read actual successful group/segment/full-checkpoint evidence, without torch."""
    require(status.get('terminal') == 'succeeded' and status.get('abort') is None
        and set(status.get('exits', {})) == {str(rank) for rank in range(8)}, 'Eight successful rank exits required')
    for rank in range(8):
        exit_record = status['exits'][str(rank)]
        require(exit_record['returncode'] == 0 and exit_record['wait_observed'] is True
            and exit_record['cleanup_completed'] is True and exit_record['process_group_gone'] is True, 'Rank exit cleanup failed')
        parent = read(group/f'rank{rank}_parent_receipt.json')
        require(parent['error'] is None and parent['cleanup']['returncode'] == 0
            and parent['intent_sha256'] == digest(intent) and parent['parent'] == intent['members'][rank]['worker_process'], 'Actual controller supervision receipt differs')
    step, horizon = plan['segment_steps'], plan['work']['max_steps']
    marker = shared/('COMPLETE.json' if step == horizon else f'SEGMENT_{step}.json')
    segment, binding = read(marker), read(shared/'run_provenance.json')
    work = plan['work']
    variants = {'r20': 'r20_refined_control', 'r21': 'r21_refined_head_gate',
                'r24': 'r24_refined_fixed_scale', 'r26': 'r26_refined_local_global'}
    require(binding.get('variant') == variants.get(work['variant'], work['variant'])
        and binding.get('source_manifest_sha256') == work['source_manifest']['sha256']
        and all(str(binding.get(key)) == str(work[key]) for key in
            ('source_root', 'repo_root', 'data_dir', 'seed', 'micro_batch', 'grad_accum', 'global_batch'))
        and binding.get('schedule', {}).get('max_steps') == horizon, 'Shared model/source/data/budget differs')
    require(segment.get('step') == step and segment.get('world_size') == 8
        and segment.get('status') == ('completed' if step == horizon else 'paused')
        and segment.get('checkpoint') == f'checkpoint-{step}' and segment.get('binding_sha256') == driver_digest(binding), 'Shared driver segment evidence differs')
    checkpoint = shared/segment['checkpoint']
    sidecar = read(checkpoint/'formal_checkpoint.json')
    require(sidecar.get('step') == step and sidecar.get('binding_sha256') == segment['binding_sha256']
        and sidecar.get('schema') == 'v7.formal.native_checkpoint.v1' and sidecar.get('writer_rank') == 0
        and sidecar.get('world_size') == 8 and sidecar.get('rank_acknowledgments') == list(range(8)), 'Full checkpoint sidecar differs')
    required = {'model.safetensors', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'run_provenance.json',
                'immune_config.json', 'research_config.json', 'vocabulary.json', 'training_args.bin',
                *(f'rng_state_{rank}.pth' for rank in range(8))}
    files = sidecar['files']
    require(required <= files.keys() and all(not Path(name).is_absolute() and '..' not in Path(name).parts
        and (checkpoint/name).is_file() and not (checkpoint/name).is_symlink() and sha(checkpoint/name) == expected
        for name, expected in files.items()), 'Native checkpoint inventory/hash incomplete')
    actual = {str(path.relative_to(checkpoint)) for path in checkpoint.rglob('*') if path.is_file() and path != checkpoint/'formal_checkpoint.json'}
    require(actual == set(files) and not any(path.is_symlink() for path in checkpoint.rglob('*'))
        and read(checkpoint/'run_provenance.json') == binding, 'Checkpoint inventory/provenance changed')
    require(sidecar.get('rank_parameter_digests') == [{'rank': rank, **sidecar['saved_parameter_digest']} for rank in range(8)], 'Actual rank parameter evidence differs')
    return {'kind': 'formal_pool_segment_completion', 'segment_step': step, 'scheduler_horizon': horizon,
        'formal_training_complete': step == horizon, 'shared_marker': str(marker), 'shared_marker_sha256': sha(marker),
        'checkpoint_sidecar': str(checkpoint/'formal_checkpoint.json'), 'checkpoint_sidecar_sha256': sha(checkpoint/'formal_checkpoint.json'),
        'eight_rank_exits_verified': True, 'intent_sha256': digest(intent)}


def completion_ready(status, parent_receipts_present):
    # Coordinator may refresh terminal after copying an earlier seven-exit snapshot.
    return (status.get('terminal') == 'succeeded' and status.get('abort') is None
        and set(status.get('exits', {})) == {str(rank) for rank in range(8)}
        and len(parent_receipts_present) == 8 and all(parent_receipts_present))


def run(plan_path, plan_sha, job_id):
    plan = prepare(read(checked({'path': str(plan_path), 'sha256': plan_sha})))
    require(job_id in plan['tickets'], 'Unexpected ticket')
    checked(plan['wrapper'])
    require(Path(plan['wrapper']['path']) == Path(__file__).resolve(), 'Wrapper source identity differs')
    checked(plan['pool_source'])
    transport = runpy.run_path(str(checked(plan['transport'])), run_name='_bound_formal_transport')
    coordinator, _, hardware_api = transport['dependencies']()
    current = coordinator['process_identity']()
    require(coordinator['process_identity'](os.getppid()) == plan['pool_process'], 'Controller has another pool parent')
    actual_argv = Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
    require([value.decode() for value in actual_argv if value] == plan['pool_argv'], 'Actual pool invocation changed')
    actual_cwd = Path(f'/proc/{os.getppid()}/cwd').readlink()
    require(invocation_source(plan['pool_argv'], actual_cwd) == checked(plan['pool_source']),
            'Not the bound pool work process')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(re.fullmatch('[0-7]', mask) is not None, 'One actual pool physical GPU required')
    worker_id = f'{socket.gethostname()}_{mask}'.replace('/', '_')
    directory = Path(plan['meeting_dir'])
    require(directory == directory.resolve() and not directory.is_symlink(), 'Meeting path differs')
    directory.mkdir(parents=False, exist_ok=True)
    claim_path = directory/f'claim_{mask}.json'
    hardware = hardware_api['observe_gpu_hardware'](mask)['hardware']
    claim = dict(rank=int(mask), mask=mask, job_id=job_id, worker_id=worker_id,
        controller=current, shared_pool=plan['pool_process'], hardware=hardware,
        plan_digest=digest(plan), observed_monotonic=time.monotonic())
    started = time.monotonic()
    intent, commit = None, None
    old_handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(number, frame):
        raise RuntimeError('Pool interrupted controller: '+str(number))
    for number in old_handlers:
        signal.signal(number, interrupted)
    try:
        while commit is None:
            require(time.monotonic()-started <= plan['join_seconds'], 'Eight-slot join deadline expired')
            abort_fence(directory, plan_sha)
            with locked(directory):
                abort_fence(directory, plan_sha)
                if (directory/'PUBLISHED.json').exists():
                    commit = read(directory/'PUBLISHED.json')
                    break
                statuses, running, pending = pool_snapshot(plan)
                own_status = statuses.get(worker_id, {})
                # Pool writes the PID heartbeat just after Popen; allow that bounded startup race.
                if own_status.get('pid') != current['pid']:
                    require(own_status.get('job_id') == job_id and own_status.get('state') == 'running', 'Own slot does not hold this ticket')
                    candidate = None
                else:
                    if claim_path.exists():
                        require(read(claim_path)['controller'] == current, 'Slot is already claimed by another process')
                    claim['observed_monotonic'] = time.monotonic()
                    atomic(claim_path, claim, coordinator)
                    claims = [read(path) for path in directory.glob('claim_*.json')]
                    verify_claims(plan, claims, statuses, running, pending, time.monotonic(), time.time(), coordinator['_same_process_alive'])
                    candidate = build_intent(plan, claims, transport) if len(claims) == 8 else None
            if candidate is not None:
                # Potentially expensive source/audit evidence hashing occurs outside publication lock.
                driver = transport['verify_intent_files'](candidate)
                args = driver['parse_args'](plan['work']['driver_argv'])
                validate_segment(plan, args, binding_digest=driver['digest'])
                with locked(directory):
                    abort_fence(directory, plan_sha)
                    if (directory/'PUBLISHED.json').exists():
                        commit = read(directory/'PUBLISHED.json')
                    else:
                        statuses, running, pending = pool_snapshot(plan)
                        claims = [read(path) for path in directory.glob('claim_*.json')]
                        claims = verify_claims(plan, claims, statuses, running, pending, time.monotonic(), time.time(), coordinator['_same_process_alive'], complete=True)
                        require(build_intent(plan, claims, transport) == candidate, 'Slot identities changed during source verification')
                        commit = publish(directory, plan, plan_sha, claims, transport, coordinator)
            if commit is None:
                time.sleep(.1)
        require(commit['plan_sha256'] == plan_sha, 'Published plan differs')
        intent_path, reservation_path = directory/'intent.json', directory/'reservation.json'
        intent = read(checked({'path': str(intent_path), 'sha256': commit['intent_sha256']}))
        checked({'path': str(reservation_path), 'sha256': commit['reservation_sha256']})
        require(intent['members'][int(mask)]['worker_process'] == current and intent['members'][int(mask)]['job_id'] == job_id,
                'Published slot differs from this controller')
        os.environ.update(FORMAL_POOL_WORKER_ID=worker_id, FORMAL_POOL_JOB_ID=job_id)
        transport['member_parent'](intent_path, commit['intent_sha256'], int(mask), reservation_path, commit['reservation_sha256'])
        group = Path(intent['group_directory'])
        contract = transport['contract_for'](intent, coordinator)
        last_observed = time.monotonic()
        while True:
            abort_fence(directory, plan_sha)
            require(time.monotonic() < commit['hard_deadline_monotonic']-10, 'Group completion proof deadline expired')
            contract_sha = coordinator['contract_digest'](contract)
            try:
                status = coordinator['group_status'](group, contract_sha)
            except coordinator['StatusObservationUnavailable'] as unavailable:
                require(type(unavailable) is coordinator['StatusObservationUnavailable']
                    and unavailable.group_directory == str(group) and unavailable.contract_sha256 == contract_sha
                    and unavailable.operation == 'group_status' and unavailable.abort_on_lock_timeout is False,
                    'Post-exit status unavailability binding differs')
                coordinator['raise_if_aborted'](group, contract_sha)
                require(time.monotonic()-last_observed <= 30, 'Post-exit status freshness expired')
                time.sleep(.1)
                continue
            last_observed = time.monotonic()
            require(status['abort'] is None and status['terminal'] != 'aborted', 'Transport group failed')
            if completion_ready(status, [(group/f'rank{rank}_parent_receipt.json').is_file() for rank in range(8)]):
                break
            time.sleep(.1)
        # Pending jobs created after publication cannot displace the held slots; do not inspect them here.
        driver = transport['verify_intent_files'](intent)
        evidence = completion_evidence(plan, intent, status, group, Path(plan['work']['output_dir']), driver['digest'])
        run_dir = Path(plan['pool_dir'])/'runs'/job_id
        run_dir.mkdir(exist_ok=True)
        coordinator['_exclusive_json'](run_dir/'COMPLETE.json', {**evidence, 'job_id': job_id, 'rank': int(mask),
            'scope': 'this pool ticket and declared training segment; not an independent model run', 'plan_sha256': plan_sha})
    except BaseException as error:
        abort(directory, plan_sha, repr(error), coordinator)
        if intent is not None:
            transport['broadcast_abort'](intent, coordinator, repr(error))
        raise
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha')
    parser.add_argument('--job-id')
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    require(args.plan is not None and args.plan_sha and args.job_id, 'Future pool dispatcher must provide a bound plan and actual job ID')
    try:
        run(args.plan, args.plan_sha, args.job_id)
    except BaseException as error:
        # Early source/hardware failures must fence siblings even before our claim.
        try:
            plan = prepare(read(checked({'path': str(args.plan), 'sha256': args.plan_sha})))
            directory = Path(plan['meeting_dir'])
            require(directory == directory.resolve() and not directory.is_symlink(), 'Abort directory differs')
            directory.mkdir(parents=False, exist_ok=True)
            with (directory/'ABORT.json').open('x') as handle:
                json.dump({'plan_sha256': args.plan_sha, 'error': repr(error)[-2000:]}, handle)
        except Exception:
            pass
        raise


def self_test():
    import tempfile
    import unittest
    from unittest.mock import patch
    root = Path('/project_c20250601/251105016/project/dllm_test')
    transport_path = root/'output/v7_formal_eight_gpu_20261002/source_snapshot_v2/scripts/research/formal_pool_transport.py'
    transport = runpy.run_path(str(transport_path), run_name='_artificial_slot_transport')
    coordinator = transport['dependencies']()[0]
    boot = '11111111-1111-1111-1111-111111111111'
    def identity(pid, parent=90):
        return dict(pid=pid, parent_pid=parent, parent_start_ticks=100, process_group_id=pid,
                    start_ticks=100, hostname='fixture', boot_id=boot)
    def fixture(directory='/tmp/unused-formal-slot-fixture'):
        ref = lambda name: {'path': directory+'/'+name, 'sha256': 'a'*64}
        work = dict(selection=ref('selection.json'), audit=ref('audit.json'), driver=ref('driver.py'),
            source_manifest=ref('source_manifest.json'), source_root=directory+'/frozen', repo_root=directory+'/real',
            data_dir=directory+'/data', output_dir=directory+'/shared', driver_argv=['unused-artificial-argv'],
            variant='f000', seed=42, micro_batch=4, grad_accum=8, global_batch=256, max_steps=10000)
        tickets = {f'ticket{index}': {'variant': f'p03_v7_formal_member_{index}', 'mode': 'train', 'steps': 1000, 'data_dir': work['data_dir']} for index in range(8)}
        plan = prepare(dict(schema=SCHEMA, pool_dir=directory+'/pool', meeting_dir=directory+'/meeting',
            group_id='artificial-only', pool_process=identity(90, 1), pool_argv=['python', '/fake/pool.py', 'work',
                '--pool-dir', directory+'/pool', '--max-seconds', '2000', '--job-seconds', '900'],
            pool_source=ref('pool.py'), wrapper=ref('wrapper.py'), transport={'path': str(transport_path), 'sha256': sha(transport_path)},
            work=work, tickets=tickets, join_seconds=180, startup_seconds=180, duration_seconds=600,
            cleanup_margin_seconds=120, segment_steps=1000))
        claims, statuses = [], {}
        for rank in range(8):
            job, worker = f'ticket{7-rank}', f'fixture_{rank}'
            hardware = dict(uuid='GPU-'+str(rank)*8+'-'+str(rank)*4+'-'+str(rank)*4+'-'+str(rank)*4+'-'+str(rank)*12,
                            mask=str(rank), observed=True)
            claim = dict(rank=rank, mask=str(rank), job_id=job, worker_id=worker, controller=identity(100+rank),
                shared_pool=plan['pool_process'], hardware=hardware, plan_digest=digest(plan), observed_monotonic=100.)
            claims.append(claim)
            statuses[worker] = dict(state='running', job_id=job, pid=100+rank, device=str(rank), job=tickets[job], heartbeat=1000.)
        return plan, claims, statuses
    class Protocol(unittest.TestCase):
        def test_shared_parent_eight_actual_controller_ids_and_shuffled_tickets(self):
            plan, claims, statuses = fixture()
            verified = verify_claims(plan, claims, statuses, plan['tickets'], {}, 101., 1001., lambda _: True, complete=True)
            intent = build_intent(plan, verified, transport)
            self.assertEqual(intent['members'][0]['job_id'], 'ticket7')
            self.assertEqual(len({row['worker_process']['pid'] for row in intent['members']}), 8)
            self.assertEqual({row['worker_process']['parent_pid'] for row in intent['members']}, {90})

        def test_pid_alias_stale_or_shared_pool_pid_rejected(self):
            for mutation in ('pid', 'job', 'stale', 'shared_pid'):
                plan, claims, statuses = fixture()
                if mutation == 'pid': statuses['fixture_0']['pid'] = 999
                if mutation == 'job': statuses['fixture_0']['job_id'] = 'ticket0'
                if mutation == 'stale': claims[0]['observed_monotonic'] = 1.
                if mutation == 'shared_pid': claims[0]['controller'] = plan['pool_process']
                with self.assertRaises(ValueError):
                    verify_claims(plan, claims, statuses, plan['tickets'], {}, 101., 1001., lambda _: True, complete=True)

        def test_partial_or_unrelated_queue_cannot_publish(self):
            plan, claims, statuses = fixture()
            with self.assertRaises(ValueError):
                verify_claims(plan, claims[:7], statuses, plan['tickets'], {}, 101., 1001., lambda _: True, complete=True)
            with self.assertRaises(ValueError):
                verify_claims(plan, claims, statuses, plan['tickets'], {'other': {}}, 101., 1001., lambda _: True)

        def test_resource_deadline_uses_process_start_and_job_limit(self):
            plan, claims, _ = fixture()
            self.assertEqual(resource_deadline(plan, claims, 500., 600., 100.), 801.)

        def test_atomic_publication_is_only_after_all_slots_held(self):
            with tempfile.TemporaryDirectory() as temporary:
                plan, claims, statuses = fixture(temporary)
                directory = Path(plan['meeting_dir']); directory.mkdir()
                claims = verify_claims(plan, claims, statuses, plan['tickets'], {}, 101., 1001., lambda _: True, complete=True)
                with patch.dict(publish.__globals__, {'resource_deadline': lambda *args: time.monotonic()+1000}):
                    with locked(directory):
                        record = publish(directory, plan, 'a'*64, claims, transport, coordinator)
                self.assertEqual(read(directory/'PUBLISHED.json'), record)
                self.assertEqual(sha(directory/'intent.json'), record['intent_sha256'])
                self.assertFalse((directory/'coordinator').exists())
                self.assertEqual(read(directory/'reservation.json')['members'][0]['worker_process']['pid'], 100)

        def test_abort_fence_is_durable_in_artificial_directory(self):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                abort(directory, 'a'*64, 'artificial failure', coordinator)
                with self.assertRaises(ValueError):
                    abort_fence(directory, 'a'*64)

        def test_terminal_refresh_with_seven_exits_waits(self):
            status = {'terminal': 'succeeded', 'abort': None, 'exits': {str(rank): {} for rank in range(7)}}
            self.assertFalse(completion_ready(status, [True]*8))
            status['exits']['7'] = {}
            self.assertTrue(completion_ready(status, [True]*8))
            self.assertFalse(completion_ready(status, [True]*7+[False]))

        def test_missing_full_checkpoint_never_creates_slot_complete(self):
            with tempfile.TemporaryDirectory() as temporary:
                plan, claims, _ = fixture(temporary)
                group, shared = Path(temporary)/'group', Path(temporary)/'shared'
                group.mkdir(); shared.mkdir()
                intent = build_intent(plan, claims, transport)
                status = dict(terminal='succeeded', abort=None, exits={})
                for rank in range(8):
                    cleanup = dict(returncode=0, wait_observed=True, cleanup_completed=True, process_group_gone=True)
                    status['exits'][str(rank)] = cleanup
                    coordinator['_exclusive_json'](group/f'rank{rank}_parent_receipt.json', dict(error=None, cleanup=cleanup,
                        intent_sha256=digest(intent), parent=intent['members'][rank]['worker_process']))
                work = plan['work']
                binding = {key: work[key] for key in ('variant', 'source_root', 'repo_root', 'data_dir', 'seed', 'micro_batch', 'grad_accum', 'global_batch')}
                binding.update(source_manifest_sha256=work['source_manifest']['sha256'], schedule={'max_steps': 10000})
                coordinator['_exclusive_json'](shared/'run_provenance.json', binding)
                coordinator['_exclusive_json'](shared/'SEGMENT_1000.json', dict(step=1000, world_size=8,
                    status='paused', checkpoint='checkpoint-1000', binding_sha256=digest(binding)))
                checkpoint = shared/'checkpoint-1000'; checkpoint.mkdir()
                for name in ('model.safetensors', 'scheduler.pt', 'trainer_state.json', 'immune_config.json',
                             'research_config.json', 'vocabulary.json', 'training_args.bin',
                             *(f'rng_state_{rank}.pth' for rank in range(8))):
                    (checkpoint/name).write_text('artificial metadata fixture, not a model\n')
                coordinator['_exclusive_json'](checkpoint/'run_provenance.json', binding)
                parameter = {'sha256': 'a'*64, 'numel': 3}
                sidecar = dict(schema='v7.formal.native_checkpoint.v1', writer_rank=0, step=1000, world_size=8,
                    binding_sha256=digest(binding), rank_acknowledgments=list(range(8)),
                    saved_parameter_digest=parameter, rank_parameter_digests=[{'rank': rank, **parameter} for rank in range(8)])
                sidecar['files'] = {path.name: sha(path) for path in checkpoint.iterdir()}
                coordinator['_exclusive_json'](checkpoint/'formal_checkpoint.json', sidecar)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    completion_evidence(plan, intent, status, group, shared, digest)
                self.assertFalse((Path(plan['pool_dir'])/'runs'/'ticket7'/'COMPLETE.json').exists())
                (checkpoint/'optimizer.pt').write_text('artificial optimizer metadata\n')
                sidecar['files']['optimizer.pt'] = sha(checkpoint/'optimizer.pt')
                atomic(checkpoint/'formal_checkpoint.json', sidecar, coordinator)
                result = completion_evidence(plan, intent, status, group, shared, digest)
                self.assertFalse(result['formal_training_complete'])
                self.assertEqual(result['segment_step'], 1000)
    with patch.object(socket, 'socket', side_effect=AssertionError('No socket in artificial wrapper controls')):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Protocol))
    require(result.wasSuccessful(), 'New wrapper protocol checks failed')
    print('Artificial metadata/temp directories only; no real pool/data/plan, listener, child or GPU.', flush=True)


if __name__ == '__main__':
    main()
