"""Source-only transport for a future atomically reserved eight-worker group.

Activate conda pllm. CPU protocol only: python -B /absolute/repo/scripts/research/
formal_pool_transport.py --self-test. This does not listen, create groups or GPUs.
API prepare(spec)/validate_intent(intent) is pure. Future pool integration calls
member_parent(intent_path, intent_sha, rank, reservation_path, reservation_sha).
Only an external pool reservation of all eight idle workers admits that call;
this module never queues jobs, selects a winner, or creates a launch approval.
The private child CLI is for that parent only, never a standalone torchrun entry.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
V4 = ROOT/'output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4'
FIXED = {
    str(V4/'eight_rank_coordinator_v2.py'): 'dbaf3e8536370309a42fd2c33d23b020224b62d981451b2038387a21009e4001',
    str(V4/'gpu_worker_cold_rank_v1.py'): '4a469cd05a71056342d1e14d0a66a550fa4c1657e86166ca940c5656e3a9cbbe',
    str(V4/'gpu_worker_bootstrap_v1.py'): 'd8b2cf6cde74217602d40bd22f7b6d1568c7c865efe8472beff58021729adce0'}
SCHEMA = 'v7.formal.pool_intent.v1'
FINAL_MANIFEST = '28e3cbf8fb860c13b514a11d600579257a233fefe9f70ea070b042d04f8a424c'
PROCESS_KEYS = {'pid', 'parent_pid', 'parent_start_ticks', 'process_group_id', 'start_ticks', 'hostname', 'boot_id'}
VARIANTS = {'r20', 'r21', 'r24', 'r26', 'f000', 'f001', 'f010', 'f011', 'f100', 'f101', 'f110', 'f111',
            'cv_segment_global', 'cv_late_fusion', 'fixed_gate_control'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def absolute(value):
    path = Path(value)
    require(path.is_absolute() and str(path) == os.path.normpath(str(path)), 'Canonical absolute path required')
    return path


def reference(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Expected an explicit file reference')
    absolute(value['path'])
    require(re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None, 'Explicit SHA256 required')


def validate_intent(intent):
    """Structural validation only: no writes, imports, GPU observation or socket."""
    keys = {'schema', 'group_id', 'group_directory', 'hostname', 'boot_id', 'members', 'work',
            'transport_sources', 'startup_seconds', 'heartbeat_seconds', 'duration_seconds'}
    require(type(intent) is dict and set(intent) == keys and intent['schema'] == SCHEMA, 'Formal intent fields differ')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', intent['group_id']) is not None, 'Invalid group ID')
    absolute(intent['group_directory'])
    require(type(intent['hostname']) is str and bool(intent['hostname']), 'Host is required')
    require(re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', intent['boot_id']) is not None, 'Boot identity required')
    require(intent['heartbeat_seconds'] == 30, 'Original 30-second freshness cannot be extended')
    for key in ('startup_seconds', 'duration_seconds'):
        require(type(intent[key]) in (int, float) and math.isfinite(intent[key]) and intent[key] > 0, 'Finite bounded deadlines required')
    require(intent['startup_seconds'] <= 180 and intent['duration_seconds'] >= intent['startup_seconds'], 'Startup/whole budget differs')
    work = intent['work']
    require(set(work) == {'selection', 'audit', 'driver', 'source_manifest', 'source_root', 'repo_root', 'data_dir',
                         'output_dir', 'driver_argv', 'variant', 'seed', 'micro_batch', 'grad_accum', 'global_batch', 'max_steps'},
            'Formal work fields differ')
    for key in ('selection', 'audit', 'driver', 'source_manifest'):
        reference(work[key])
    for key in ('source_root', 'repo_root', 'data_dir', 'output_dir'):
        absolute(work[key])
    require(work['variant'] in VARIANTS, 'Winner/explicit model choice is still missing or unsupported')
    require(type(work['seed']) is int and work['seed'] >= 0, 'Explicit seed required')
    require(all(type(work[key]) is int and work[key] > 0 for key in ('micro_batch', 'grad_accum', 'global_batch', 'max_steps')),
            'Positive training budgets required')
    require(work['micro_batch']*work['grad_accum']*8 == work['global_batch'], 'World8 batch arithmetic differs')
    require(type(work['driver_argv']) is list and all(type(arg) is str for arg in work['driver_argv']), 'Explicit driver argv required')
    members = intent['members']
    require(type(members) is list and len(members) == 8, 'Exactly eight members required')
    for rank, member in enumerate(members):
        require(set(member) == {'rank', 'worker_id', 'job_id', 'gpu_identity', 'visible_device', 'hardware', 'worker_process'}
                and type(member['rank']) is int and member['rank'] == rank, 'Ordered rank records required')
        for key in ('worker_id', 'job_id'):
            require(type(member[key]) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', member[key]) is not None,
                    'Explicit worker/job ID required')
        require(re.fullmatch('[0-7]', member['visible_device']) is not None, 'One physical mask per worker required')
        process = member['worker_process']
        require(set(process) == PROCESS_KEYS and process['hostname'] == intent['hostname'] and process['boot_id'] == intent['boot_id'],
                'Worker host/boot binding differs')
        require(all(type(process[key]) is int and process[key] > 0 for key in PROCESS_KEYS-{'hostname', 'boot_id'}), 'Worker process identity differs')
        require(member['hardware']['uuid'] == member['gpu_identity'] and member['hardware']['mask'] == member['visible_device']
                and member['hardware']['observed'] is True, 'Actual hardware observation required')
    for key in ('worker_id', 'job_id', 'gpu_identity', 'visible_device'):
        require(len({member[key] for member in members}) == 8, 'Duplicate '+key)
    require(len({member['worker_process']['pid'] for member in members}) == 8, 'Duplicate actual worker process')
    sources = intent['transport_sources']
    require(set(sources) == {*FIXED, str(Path(__file__).resolve())} and all(sources[path] == value for path, value in FIXED.items()),
            'Transport dependencies differ; synthetic V4 gates are not reusable formal gates')
    require(all(re.fullmatch('[0-9a-f]{64}', value) is not None for value in sources.values()), 'Transport SHA required')
    return intent


def prepare(spec):
    """Return a checked in-memory intent; deliberately does not publish it."""
    require('schema' not in spec, 'prepare supplies its own schema')
    return validate_intent({'schema': SCHEMA, **copy.deepcopy(spec)})


def read_reference(ref):
    reference(ref)
    path = absolute(ref['path'])
    require(path == path.resolve() and not path.is_symlink() and path.is_file()
        and sha(path) == ref['sha256'], 'Evidence/source SHA drift: '+str(path))
    return json.loads(path.read_text())


def load_fixed(path, expected):
    require(sha(path) == expected, 'Fixed source differs')
    result = runpy.run_path(str(path), run_name='_formal_transport_dependency')
    require(sha(path) == expected, 'Fixed source changed while importing')
    return result


def dependencies():
    return tuple(load_fixed(path, expected) for path, expected in FIXED.items())


def verify_intent_files(intent):
    validate_intent(intent)
    for path, expected in intent['transport_sources'].items():
        require(sha(absolute(path)) == expected, 'Transport source drift')
    work = intent['work']
    selection, audit, manifest = (read_reference(work[key]) for key in ('selection', 'audit', 'source_manifest'))
    require(selection.get('schema') == 'v7.formal.model_choice.v1' and selection.get('status') == 'selected'
            and selection.get('variant') == work['variant'], 'Formal model selection is not finalized')
    require(audit.get('schema') == 'v7.formal.read_only_parent_check.v1' and audit.get('status') == 'passed'
            and audit.get('repo_root') == work['repo_root'] and audit.get('data_dir') == work['data_dir']
            and audit.get('source_manifest_sha256') == work['source_manifest']['sha256']
            and audit.get('metadata', {}).get('dataset_manifest.json') == FINAL_MANIFEST,
            'An actual passed final-parent audit bound to these sources is required')
    require(audit.get('executed_sources') and all(manifest['files'].get(path) == value
        and sha(path) == value for path, value in audit['executed_sources'].items()), 'Executed gate sources differ')
    require(all(sha(Path(work['data_dir'])/name) == value for name, value in audit['metadata'].items()), 'Audited parent metadata drift')
    driver_path = absolute(work['driver']['path'])
    require(driver_path == Path(work['source_root'])/'scripts/research/run_v7_formal.py'
        and manifest['files'].get(str(driver_path)) == work['driver']['sha256'], 'Frozen driver identity differs')
    driver = load_fixed(driver_path, work['driver']['sha256'])
    args = driver['parse_args'](work['driver_argv'])
    for key in ('source_root', 'repo_root', 'data_dir', 'output_dir', 'variant', 'seed', 'micro_batch', 'grad_accum', 'global_batch', 'max_steps'):
        require(str(getattr(args, key)) == str(work[key]), 'Driver argv differs: '+key)
    require(str(args.source_manifest) == work['source_manifest']['path']
        and args.source_manifest_sha256 == work['source_manifest']['sha256'], 'Driver source manifest differs')
    driver['verify_sources'](args)
    return driver


def contract_for(intent, coordinator):
    contract = dict(schema=coordinator['CONTRACT_SCHEMA'], scope='formal_v7_masked_reconstruction',
        group_id=intent['group_id'], world_size=8, hostname=intent['hostname'], boot_id=intent['boot_id'],
        startup_timeout_seconds=intent['startup_seconds'], heartbeat_timeout_seconds=30,
        work_binding_sha256=digest(intent), ranks=[{key: row[key] for key in
            ('rank', 'job_id', 'gpu_identity', 'visible_device')} for row in intent['members']])
    return coordinator['validate_contract'](contract)


def validate_reservation(intent, reservation, now):
    """External pool must atomically reserve all eight; idle claims alone are insufficient."""
    require(reservation.get('schema') == 'v7.formal.all_eight_pool_reservation.v1'
        and reservation.get('status') == 'reserved_all_eight' and reservation.get('intent_sha256') == digest(intent)
        and reservation.get('hostname') == intent['hostname'] and reservation.get('boot_id') == intent['boot_id']
        and reservation.get('members') == intent['members'], 'Whole-group pool reservation differs')
    require(0 <= now-reservation['reserved_monotonic'] <= 30, 'Pool reservation freshness exceeds 30 seconds')
    require(reservation.get('previous_jobs_finished') is True and reservation.get('exclusive') is True,
            'All workers must finish their previous jobs and be reserved together')
    require(type(reservation.get('group_deadline_monotonic')) in (int, float)
        and math.isfinite(reservation['group_deadline_monotonic'])
        and now+intent['startup_seconds'] < reservation['group_deadline_monotonic']
        <= reservation['reserved_monotonic']+intent['duration_seconds'], 'Finite original resource deadline required')
    return reservation


def environment_for(rank, mask, port):
    environment = dict(os.environ)
    for key in tuple(environment):
        if key in {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK',
                   'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'PYTHONPATH'} or key.startswith(
                   ('TORCHELASTIC_', 'OMPI_', 'PMI_', 'PMIX_', 'SLURM_', 'ACCELERATE_')):
            environment.pop(key)
    environment.update(CUDA_VISIBLE_DEVICES=mask, RANK=str(rank), WORLD_SIZE='8', LOCAL_RANK='0',
        LOCAL_WORLD_SIZE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), PYTHONDONTWRITEBYTECODE='1',
        USE_TF='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return environment


def check_abort(intent, coordinator):
    group = Path(intent['group_directory'])
    marker = group.with_name(group.name+'.transport_abort.json')
    if marker.exists():
        require(coordinator['_read_json'](marker).get('intent_sha256') == digest(intent), 'Abort marker belongs to another intent')
        raise ValueError('Formal transport has an irreversible abort marker')


def broadcast_abort(intent, coordinator, message):
    """Also fence peers that have not yet observed rank0's group creation."""
    group = Path(intent['group_directory'])
    marker = group.with_name(group.name+'.transport_abort.json')
    try:
        coordinator['_exclusive_json'](marker, {'intent_sha256': digest(intent), 'error': str(message)[-2000:]})
    except FileExistsError:
        require(coordinator['_read_json'](marker).get('intent_sha256') == digest(intent), 'Abort marker identity differs')
    if (group/'contract.json').is_file():
        contract = contract_for(intent, coordinator)
        coordinator['abort_group'](group, coordinator['contract_digest'](contract), str(message)[-2000:])


def member_parent(intent_path, intent_sha, rank, reservation_path, reservation_sha):
    """Future pool call only. Own one isolated child, its FD and complete cleanup."""
    intent = read_reference({'path': str(intent_path), 'sha256': intent_sha})
    validate_intent(intent)
    coordinator, cold, bootstrap = dependencies()
    try:
        check_abort(intent, coordinator)
        verify_intent_files(intent)
        reservation = validate_reservation(intent, read_reference({'path': str(reservation_path), 'sha256': reservation_sha}), time.monotonic())
        require(type(rank) is int and 0 <= rank < 8, 'Rank required')
        row = intent['members'][rank]
        current = coordinator['process_identity']()
        worker = row['worker_process']
        require(current['hostname'] == intent['hostname'] and current['boot_id'] == intent['boot_id']
            and all(coordinator['_same_process_alive'](member['worker_process']) for member in intent['members']), 'Worker host/boot/process changed')
        require(current == worker or (current['parent_pid'] == worker['pid'] and current['parent_start_ticks'] == worker['start_ticks']),
                'Pool caller is not the reserved worker or its direct job child')
        require(os.environ.get('FORMAL_POOL_WORKER_ID') == row['worker_id']
            and os.environ.get('FORMAL_POOL_JOB_ID') == row['job_id'], 'External pool worker/job assignment differs')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == row['visible_device'], 'Pool physical mask differs')
        require(bootstrap['observe_gpu_hardware'](row['visible_device'])['hardware'] == row['hardware'], 'Physical GPU observation differs')
    except BaseException:
        broadcast_abort(intent, coordinator, traceback.format_exc())
        raise
    group = absolute(intent['group_directory'])
    contract = contract_for(intent, coordinator)
    contract_sha = coordinator['contract_digest'](contract)
    listener, process, child_identity, observation, cleanup, error = None, None, None, None, None, None
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in handlers:
        signal.signal(number, cold['interrupted'])
    try:
        if rank == 0:
            check_abort(intent, coordinator)
            listener = cold['hold_loopback_socket']()
            coordinator['initialize_group'](group, contract, contract_sha)
            coordinator['_exclusive_json'](group/'endpoint.json', {'intent_sha256': digest(intent),
                'port': listener.getsockname()[1], 'owner_parent': current, 'listener_fd': listener.fileno()})
        deadline = min(time.monotonic()+intent['startup_seconds'], reservation['group_deadline_monotonic'])
        while not (group/'endpoint.json').is_file():
            check_abort(intent, coordinator)
            require(time.monotonic() < deadline, 'All-worker endpoint startup deadline expired')
            time.sleep(.05)
        endpoint = coordinator['_read_json'](group/'endpoint.json')
        require(endpoint['intent_sha256'] == digest(intent)
            and coordinator['_read_json'](group/'contract.json') == contract, 'Endpoint/group binding differs')
        coordinator['raise_if_aborted'](group, contract_sha)
        argv = [sys.executable, '-I', '-B', '-u', str(Path(__file__).resolve()), 'child',
            '--intent', str(intent_path), '--intent-sha', intent_sha, '--rank', str(rank)]
        if listener is not None:
            argv += ['--listen-fd', str(listener.fileno())]
        with (group/f'rank{rank}_child.log').open('x') as stream:
            process = subprocess.Popen(argv, env=environment_for(rank, row['visible_device'], endpoint['port']),
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=() if listener is None else (listener.fileno(),))
            child_identity = coordinator['_process_snapshot'](process.pid)[0]
            require(child_identity['parent_pid'] == os.getpid() and child_identity['parent_start_ticks'] == current['start_ticks']
                and child_identity['process_group_id'] == process.pid, 'Child ownership differs')
            if listener is not None:
                listener.close()
            state = coordinator['_read_json'](group/'state.json')
            observation = cold['supervise_process_observations'](process,
                observe_status=lambda: coordinator['group_status'](group, contract_sha),
                check_member_abort=lambda: check_abort(intent, coordinator), coordinator=coordinator,
                coordinator_directory=group, contract_sha256=contract_sha,
                startup_deadline_monotonic=state['startup_deadline_monotonic'], heartbeat_timeout_seconds=30,
                whole_deadline_monotonic=reservation['group_deadline_monotonic'])
    except BaseException as cause:
        observation = getattr(cause, 'observation_supervision', observation)
        error = traceback.format_exc()
        try:
            broadcast_abort(intent, coordinator, error)
        except BaseException:
            error += '\n'+traceback.format_exc()
    finally:
        for number in handlers:
            signal.signal(number, signal.SIG_IGN)
        try:
            if listener is not None:
                listener.close()
            if process is not None:
                cleanup = cold['reap_owned_process'](process)
                require(cleanup['cleanup_completed'], 'Child/descendants remain alive')
                state = coordinator['_read_json'](group/'state.json')
                registered = state['registrations'].get(str(rank))
                if registered:
                    require(registered['process'] == child_identity, 'Registered child differs')
                    receipt = {'process': child_identity, **{key: cleanup[key] for key in
                        ('returncode', 'wait_observed', 'normal_exit', 'cleanup_completed', 'process_group_gone')}}
                    coordinator['record_exit'](group, contract_sha, rank, receipt)
                else:
                    require(cleanup['returncode'] != 0, 'Successful child never registered')
        except BaseException:
            error = (error or '')+'\nCleanup: '+traceback.format_exc()
            try:
                broadcast_abort(intent, coordinator, error)
            except BaseException:
                error += '\n'+traceback.format_exc()
        finally:
            try:
                if group.is_dir():
                    coordinator['_exclusive_json'](group/f'rank{rank}_parent_receipt.json', {
                        'schema': 'v7.formal.rank_supervision.v1', 'rank': rank, 'intent_sha256': digest(intent),
                        'parent': current, 'child': child_identity, 'observation': observation, 'cleanup': cleanup, 'error': error})
            finally:
                for number, handler in handlers.items():
                    signal.signal(number, handler)
    require(error is None and cleanup is not None and cleanup['returncode'] == 0, 'Formal rank failed: '+str(error))
    return cleanup


def rank_child(intent_path, intent_sha, rank, listen_fd=None):
    intent = read_reference({'path': str(intent_path), 'sha256': intent_sha})
    driver = verify_intent_files(intent)
    coordinator, cold, bootstrap = dependencies()
    row = intent['members'][rank]
    group, contract = absolute(intent['group_directory']), contract_for(intent, coordinator)
    contract_sha = coordinator['contract_digest'](contract)
    endpoint = coordinator['_read_json'](group/'endpoint.json')
    require(endpoint['intent_sha256'] == digest(intent) and coordinator['_read_json'](group/'contract.json') == contract,
            'Child group binding differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == row['visible_device'], 'Child physical mask differs')
    require(bootstrap['observe_gpu_hardware'](row['visible_device'])['hardware'] == row['hardware'], 'Child actual GPU differs')
    current = coordinator['process_identity']()
    parent = coordinator['process_identity'](os.getppid())
    worker = row['worker_process']
    require(parent == worker or (parent['parent_pid'] == worker['pid'] and parent['parent_start_ticks'] == worker['start_ticks']),
            'Child is outside reserved worker process tree')
    require(os.environ.get('FORMAL_POOL_WORKER_ID') == row['worker_id']
        and os.environ.get('FORMAL_POOL_JOB_ID') == row['job_id'], 'Child job assignment differs')
    check_abort(intent, coordinator)
    if rank == 0:
        require(parent == endpoint['owner_parent'] and listen_fd == endpoint['listener_fd'], 'Rank0 held listener owner differs')
    else:
        require(listen_fd is None, 'Only rank0 receives a listener')
    coordinator['register_rank'](group, contract_sha, rank, row['job_id'], row['gpu_identity'], row['visible_device'], digest(intent))
    stop, pulse_errors = threading.Event(), []
    def pulse():
        while not stop.wait(.25):
            try:
                coordinator['heartbeat'](group, contract_sha, rank)
            except BaseException as exc:
                pulse_errors.append(repr(exc))
                try:
                    coordinator['abort_group'](group, contract_sha, repr(exc))
                finally:
                    return
    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    distributed, store = None, None
    try:
        coordinator['wait_startup'](group, contract_sha, rank, intent['startup_seconds'])
        require(not pulse_errors, 'Heartbeat failed before NCCL')
        verify_intent_files(intent)
        store, distributed = cold['initialize_explicit_store'](rank, endpoint['port'], listen_fd, 180.)
        driver['main'](intent['work']['driver_argv'])
        require(not pulse_errors, 'Heartbeat failed during training')
        distributed.barrier()
        stop.set()
        thread.join(timeout=5)
        require(not thread.is_alive() and not pulse_errors, 'Heartbeat cleanup failed')
        distributed.barrier()
    except BaseException:
        broadcast_abort(intent, coordinator, traceback.format_exc())
        raise
    finally:
        stop.set()
        thread.join(timeout=5)
        torch = sys.modules.get('torch')
        if torch is not None and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=['child'])
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--intent', type=Path)
    parser.add_argument('--intent-sha')
    parser.add_argument('--rank', type=int, choices=range(8))
    parser.add_argument('--listen-fd', type=int)
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    require(args.mode == 'child' and args.intent is not None and args.intent_sha and args.rank is not None, 'Explicit parent-owned child arguments required')
    rank_child(args.intent, args.intent_sha, args.rank, args.listen_fd)


def self_test():
    """Pure artificial metadata checks only; never instantiate a group or socket."""
    import unittest
    from unittest.mock import patch
    boot = '11111111-1111-1111-1111-111111111111'
    def fixture():
        ref = lambda name: {'path': '/tmp/formal_transport_fixture/'+name, 'sha256': 'a'*64}
        members = []
        for rank in range(8):
            uuid = 'GPU-'+str(rank)*8+'-'+str(rank)*4+'-'+str(rank)*4+'-'+str(rank)*4+'-'+str(rank)*12
            members.append(dict(rank=rank, worker_id=f'worker{rank}', job_id=f'job{rank}', gpu_identity=uuid,
                visible_device=str(rank), hardware=dict(uuid=uuid, mask=str(rank), observed=True),
                worker_process=dict(pid=100+rank, parent_pid=50, parent_start_ticks=1,
                    process_group_id=100+rank, start_ticks=10+rank, hostname='fixture-host', boot_id=boot)))
        spec = dict(group_id='artificial-contract-only', group_directory='/tmp/formal_transport_fixture/uncreated-group',
            hostname='fixture-host', boot_id=boot, members=members,
            transport_sources={**FIXED, str(Path(__file__).resolve()): 'a'*64},
            startup_seconds=180, heartbeat_seconds=30, duration_seconds=3600,
            work=dict(selection=ref('selection.json'), audit=ref('audit.json'), driver=ref('driver.py'),
                source_manifest=ref('source_manifest.json'), source_root='/tmp/formal_transport_fixture/frozen',
                repo_root='/tmp/formal_transport_fixture/real', data_dir='/tmp/formal_transport_fixture/data',
                output_dir='/tmp/formal_transport_fixture/run', driver_argv=['--artificial-unused-argv'],
                variant='f000', seed=42, micro_batch=4, grad_accum=8, global_batch=256, max_steps=10000))
        return prepare(spec)
    def reservation(intent):
        return dict(schema='v7.formal.all_eight_pool_reservation.v1', status='reserved_all_eight',
            intent_sha256=digest(intent), hostname=intent['hostname'], boot_id=boot, members=intent['members'],
            reserved_monotonic=100., group_deadline_monotonic=3700., previous_jobs_finished=True, exclusive=True)
    class Protocol(unittest.TestCase):
        def test_pure_prepare_does_not_mutate_spec(self):
            intent = fixture()
            original = copy.deepcopy(intent)
            self.assertEqual(validate_intent(intent), original)
            self.assertEqual(len(intent['members']), 8)

        def test_no_pending_model_choice(self):
            intent = fixture()
            intent['work']['variant'] = None
            with self.assertRaises(ValueError):
                validate_intent(intent)

        def test_unique_workers_jobs_and_gpus(self):
            for key in ('worker_id', 'job_id', 'gpu_identity', 'visible_device'):
                intent = fixture()
                intent['members'][1][key] = intent['members'][0][key]
                with self.assertRaises(ValueError):
                    validate_intent(intent)

        def test_batch_budget_and_original_freshness(self):
            for key, value in [('global_batch', 32), ('max_steps', 0)]:
                intent = fixture()
                intent['work'][key] = value
                with self.assertRaises(ValueError):
                    validate_intent(intent)
            intent = fixture()
            intent['heartbeat_seconds'] = 31
            with self.assertRaises(ValueError):
                validate_intent(intent)

        def test_no_partial_stale_or_busy_reservation(self):
            intent = fixture()
            self.assertEqual(validate_reservation(intent, reservation(intent), 110.)['status'], 'reserved_all_eight')
            for update in ({'members': intent['members'][:7]}, {'previous_jobs_finished': False},
                           {'exclusive': False}, {'reserved_monotonic': 79.}, {'boot_id': 'changed'}):
                with self.assertRaises(ValueError):
                    validate_reservation(intent, {**reservation(intent), **update}, 110.)

        def test_single_mask_environment_and_rank_zero_local(self):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '6', 'RANK': '99', 'ACCELERATE_USE_FSDP': 'true',
                    'PYTHONPATH': '/bad/source', 'FORMAL_POOL_JOB_ID': 'job6'}):
                env = environment_for(6, '6', 23456)
            self.assertEqual((env['RANK'], env['WORLD_SIZE'], env['LOCAL_RANK'], env['CUDA_VISIBLE_DEVICES']), ('6', '8', '0', '6'))
            self.assertNotIn('ACCELERATE_USE_FSDP', env)
            self.assertNotIn('PYTHONPATH', env)
            self.assertEqual(env['FORMAL_POOL_JOB_ID'], 'job6')

        def test_immutable_coordinator_accepts_generic_contract(self):
            coordinator, cold, bootstrap = dependencies()
            intent = fixture()
            contract = contract_for(intent, coordinator)
            self.assertEqual(contract['work_binding_sha256'], digest(intent))
            self.assertEqual(contract['scope'], 'formal_v7_masked_reconstruction')
            self.assertTrue(callable(cold['supervise_process_observations']))
            self.assertIn('training_admitted', cold['FALSE_FLAGS'])
            self.assertEqual(bootstrap['WORK_BINDING']['scope'], 'synthetic_fixture_only')

        def test_old_synthetic_schema_is_not_formal_intent(self):
            intent = fixture()
            intent['schema'] = 'generic.eight_worker_GPU_intent.v1'
            with self.assertRaises(ValueError):
                validate_intent(intent)
    with patch.object(socket, 'socket', side_effect=AssertionError('CPU protocol must not create a socket')):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Protocol))
    require(result.wasSuccessful(), 'New transport protocol failed')
    print('Artificial metadata only; no intent/reservation/group/listener/GPU/job created.', flush=True)


if __name__ == '__main__':
    main()
