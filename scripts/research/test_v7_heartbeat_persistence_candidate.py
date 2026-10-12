"""Test the unbound heartbeat candidate using two groups of eight CPU processes.

Activate conda pllm, then run once:
CUDA_VISIBLE_DEVICES='' python -I -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_heartbeat_persistence_candidate.py
Only artificial metadata is used. No model, dataset, GPU or job is accessed.
Evidence is retained and this script removes only its own temporary fixtures.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
BASE = ROOT / 'output/generic_eight_worker_group_integration_20261002'
OWN = ROOT / 'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/coordinator_candidate_v1'
SELF = Path(__file__).resolve()
HELPERS = BASE / 'actual_gpu_original_Trainer_v2/lock_transaction_controls_v1.py'
SOURCES = {
    'baseline': BASE / 'actual_gpu_original_Trainer_v4/eight_rank_coordinator_v2.py',
    'candidate': OWN / 'eight_rank_coordinator_v2.py',
}
HASHES = {
    'baseline': 'dbaf3e8536370309a42fd2c33d23b020224b62d981451b2038387a21009e4001',
    'candidate': 'ba3c595803974e47ee0a8a23cf8a5cc88f6f8d3eddeb4bc98a66fe9b81e7b724',
}
ML_NAMES = {'torch', 'transformers', 'accelerate', 'numpy', 'dllm'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def load(mode):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and sys.dont_write_bytecode,
            'Empty CUDA mask and -B are required')
    require(sha(HELPERS) == 'e301ccf21b3c2935ea6158b0d8ae9c9becc925e87f2894a49eb91585f02c5274',
            'Utility source changed')
    require(sha(SOURCES[mode]) == HASHES[mode], 'Coordinator source changed')
    helper = runpy.run_path(str(HELPERS), run_name='_utilities_only')
    module = runpy.run_path(str(SOURCES[mode]), run_name='_unbound_CPU_control')
    require(module['LOCK_TIMEOUT_SECONDS'] == 2., 'Lock timeout changed')
    require(not ML_NAMES & {name.split('.')[0] for name in sys.modules}, 'ML import prohibited')
    return module, helper


def child(mode, directory, rank):
    module, helper = load(mode)
    require(OWN / '_cpu_tmp' in directory.resolve().parents, 'Unowned fixture')
    read, write, wait = helper['read'], helper['write'], helper['wait_for']
    contract = read(directory / 'contract.json')
    require(contract['scope'] == 'CPU_control_metadata_only', 'Wrong fixture scope')
    digest = module['contract_digest'](contract)
    actual_fsync = os.fsync
    startup_fsync = []

    def count_startup(fd):
        startup_fsync.append(os.readlink('/proc/self/fd/' + str(fd)))
        return actual_fsync(fd)

    os.fsync = count_startup
    try:
        row = contract['ranks'][rank]
        registration = module['register_rank'](directory, digest, rank, row['job_id'],
            row['gpu_identity'], row['visible_device'], contract['work_binding_sha256'])
        token = module['wait_startup'](directory, digest, rank, 15.)
    finally:
        os.fsync = actual_fsync
    write(directory / f'ready_{rank}.json', dict(registration=registration, token=token,
                                               startup_fsync=startup_fsync))
    wait(lambda: (directory / 'go.json').exists(), 15., 'No test start')
    if rank in (0, 1):
        globals_ = module['heartbeat'].__globals__
        actual_write = globals_['_write_state']
        fsync_calls, writes = [], []
        injected = [False]

        def controlled_fsync(fd):
            path = os.readlink('/proc/self/fd/' + str(fd))
            fsync_calls.append(path)
            if rank == 0 and Path(path).name.startswith('.state_') and not injected[0]:
                injected[0] = True
                time.sleep(2.35)
            return actual_fsync(fd)

        def observed_write(target, state, **kwargs):
            writes.append(kwargs.get('durable', True))
            if rank == 0:
                write(directory / 'heartbeat_started.json', {'pid': os.getpid()})
            return actual_write(target, state, **kwargs)

        globals_['_write_state'] = observed_write
        os.fsync = controlled_fsync
        error = None
        if rank == 1:
            wait(lambda: (directory / 'heartbeat_started.json').exists(), 6., 'No first heartbeat')
            time.sleep(.05)
        started = time.monotonic()
        try:
            module['heartbeat'](directory, digest, rank)
        except module['CoordinationError'] as exc:
            error = str(exc)
        finally:
            os.fsync = actual_fsync
            globals_['_write_state'] = actual_write
        write(directory / f'heartbeat_{rank}.json', dict(error=error,
            elapsed_seconds=time.monotonic() - started, slow_fsync_injected=injected[0],
            fsync_calls=fsync_calls, write_durable_flags=writes))
        if mode == 'candidate' and rank == 0:
            wait(lambda: (directory / 'race_go.json').exists(), 8., 'No race start')
            race_flags, race_fsync = [], []

            def observed_fsync(fd):
                race_fsync.append(os.readlink('/proc/self/fd/' + str(fd)))
                return actual_fsync(fd)

            def publish_window(target, state, **kwargs):
                race_flags.append(kwargs.get('durable', True))
                result = actual_write(target, state, **kwargs)
                if len(race_flags) == 1:
                    write(directory / 'race_state_written.json', {'pid': os.getpid()})
                    wait(lambda: (directory / 'race_abort_written.json').exists(), 1.5,
                         'Independent abort publisher did not finish')
                return result

            globals_['_write_state'] = publish_window
            os.fsync = observed_fsync
            error = None
            try:
                module['heartbeat'](directory, digest, rank)
            except module['CoordinationError'] as exc:
                error = str(exc)
            finally:
                os.fsync = actual_fsync
                globals_['_write_state'] = actual_write
            write(directory / 'race_result.json', dict(error=error,
                write_durable_flags=race_flags, fsync_calls=race_fsync))
    elif mode == 'candidate' and rank == 2:
        wait(lambda: (directory / 'race_state_written.json').exists(), 10., 'No race window')
        abort_fsync = []

        def observed_abort_fsync(fd):
            abort_fsync.append(os.readlink('/proc/self/fd/' + str(fd)))
            return actual_fsync(fd)

        os.fsync = observed_abort_fsync
        try:
            module['_publish_abort_request'](directory, digest, 'synthetic concurrent abort')
        finally:
            os.fsync = actual_fsync
        write(directory / 'race_abort_written.json', dict(pid=os.getpid(), fsync_calls=abort_fsync))
    wait(lambda: (directory / 'release.json').exists(), 15., 'No cleanup release')
    require(not ML_NAMES & {name.split('.')[0] for name in sys.modules}, 'ML import prohibited')


def run():
    require(sys.executable == '/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python',
            'Activate pllm first')
    _, helper = load('candidate')
    read, write, wait = helper['read'], helper['write'], helper['wait_for']
    evidence = OWN / 'cpu_evidence_v1'
    evidence.mkdir(exist_ok=False)
    temporary_parent = OWN / '_cpu_tmp'
    temporary_parent.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='heartbeat_', dir=temporary_parent))
    inputs = {str(path): sha(path) for path in [SELF, HELPERS, *SOURCES.values()]}
    children, streams, commands, results, cleanup = [], [], [], [], []
    started, failure = time.monotonic(), None
    write(evidence / 'started_once.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        scope='CPU_metadata_only', source_sha256=inputs, maximum_seconds=75,
        groups=2, children_per_group=8, explicit_slow_fsync_seconds=2.35))
    try:
        trees = {name: {node.name: ast.dump(node, include_attributes=False)
                       for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
                 for name, path in SOURCES.items()}
        changed = {name for name in trees['baseline']
                   if trees['baseline'][name] != trees['candidate'].get(name)}
        require(changed == {'_exclusive_json', '_write_state', '_locked'}, 'Unexpected function changes')
        require(set(trees['candidate']) - set(trees['baseline']) == {'_heartbeat_snapshot_only'},
                'Unexpected added functions')
        for mode in ('baseline', 'candidate'):
            module, _ = load(mode)
            directory = temporary / mode
            contract = dict(schema=module['CONTRACT_SCHEMA'], scope='CPU_control_metadata_only',
                group_id=mode, world_size=8, hostname=socket.gethostname(), boot_id=module['_boot_id'](),
                startup_timeout_seconds=20., heartbeat_timeout_seconds=30.,
                work_binding_sha256=hashlib.sha256(b'CPU metadata only').hexdigest(),
                ranks=[dict(rank=i, job_id='CPU_job_' + str(i),
                            gpu_identity='CPU_metadata_not_observed_GPU_' + str(i),
                            visible_device=str(i)) for i in range(8)])
            digest = module['contract_digest'](contract)
            actual_fsync = os.fsync
            initialize_calls = []

            def count_initialize(fd):
                initialize_calls.append(os.readlink('/proc/self/fd/' + str(fd)))
                return actual_fsync(fd)

            os.fsync = count_initialize
            try:
                module['initialize_group'](directory, contract, digest)
            finally:
                os.fsync = actual_fsync
            require(len(initialize_calls) == 2, 'Initial metadata lost synchronous persistence')
            ranks = []
            for rank in range(8):
                log = evidence / f'{mode}_rank{rank}.log'
                stream = log.open('x')
                streams.append(stream)
                argv = [sys.executable, '-I', '-B', str(SELF), '--child', mode,
                        '--directory', str(directory), '--rank', str(rank)]
                process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, start_new_session=True)
                children.append(process); ranks.append(process)
                commands.append(dict(argv=argv, pid=process.pid, log=str(log)))
            wait(lambda: all((directory / f'ready_{rank}.json').exists() for rank in range(8)),
                 18., 'Eight CPU ranks did not start')
            initial = module['group_status'](directory, digest)
            require(initial['phase'] == 'started' and initial['abort'] is None, 'Startup failed')
            require(all(read(directory / f'ready_{rank}.json')['startup_fsync'] for rank in range(8)),
                    'Rank registration/barrier lost synchronous persistence')
            write(directory / 'go.json', {'go': True})
            wait(lambda: all((directory / f'heartbeat_{i}.json').exists() for i in (0, 1)),
                 8., 'Heartbeat observations incomplete')
            pulses = [read(directory / f'heartbeat_{i}.json') for i in (0, 1)]
            before_race = module['group_status'](directory, digest)
            race = None
            if mode == 'baseline':
                require(pulses[0]['slow_fsync_injected'], 'Baseline slow-fsync injection missing')
                require(pulses[0]['error'] and 'lock wait expired' in pulses[1]['error'],
                        'Baseline did not reproduce lock timeout and abort propagation')
                require(before_race['abort']['reason'] == 'Coordinator lock wait expired',
                        'Baseline did not persist timeout abort')
            else:
                require(all(p['error'] is None and p['fsync_calls'] == []
                            and p['write_durable_flags'] == [False] for p in pulses),
                        'Candidate heartbeat still synchronously persists or failed')
                require(before_race['abort'] is None, 'Candidate group aborted before race')
                write(directory / 'race_go.json', {'go': True})
                wait(lambda: (directory / 'race_result.json').exists(), 5., 'No race result')
                race = read(directory / 'race_result.json')
                publisher = read(directory / 'race_abort_written.json')
                require(race['error'] and race['write_durable_flags'] == [False, True],
                        'Concurrent abort did not convert final state to durable write')
                require(len(race['fsync_calls']) == len(publisher['fsync_calls']) == 1,
                        'Abort request/state persistence missing')
                require(module['group_status'](directory, digest)['abort']['reason'] ==
                        'synthetic concurrent abort', 'Durable concurrent abort was lost')
                race['publisher'] = publisher
                race['later_refusal'] = helper['expect_rejection'](
                    lambda: module['raise_if_aborted'](directory, digest), 'durably aborted')
            write(directory / 'release.json', {'release': True})
            exit_fsync = []

            def count_exit(fd):
                exit_fsync.append(os.readlink('/proc/self/fd/' + str(fd)))
                return actual_fsync(fd)

            os.fsync = count_exit
            try:
                for rank, process in enumerate(ranks):
                    require(process.wait(timeout=5.) == 0 and helper['group_gone'](process.pid),
                            'CPU child did not exit cleanly')
                    receipt = dict(process=read(directory / f'ready_{rank}.json')['registration']['process'],
                        returncode=process.returncode, wait_observed=True, normal_exit=True,
                        cleanup_completed=True, process_group_gone=True)
                    module['record_exit'](directory, digest, rank, receipt)
            finally:
                os.fsync = actual_fsync
            terminal = module['group_status'](directory, digest)
            require(terminal['terminal'] == 'aborted' and len(terminal['exits']) == 8,
                    'Expected abort or actual eight-child cleanup missing')
            require(len(exit_fsync) == 8, 'Exit updates lost synchronous persistence')
            require(terminal['metadata_only'] and all(terminal[key] is False for key in
                    ('execution_ready', 'gpu_accepted', 'data_accepted', 'training_accepted')),
                    'CPU test incorrectly admitted work')
            results.append(dict(mode=mode, heartbeats=pulses, concurrent_abort=race,
                initialization_fsync=initialize_calls, exit_fsync=exit_fsync,
                before_race=before_race, terminal=terminal, status='passed'))
            write(evidence / f'{mode}_result.json', results[-1])
    except BaseException:
        failure = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for process in children:
            try:
                cleanup.append(helper['reap'](process, time.monotonic() + 5.))
            except BaseException:
                cleanup.append(dict(pid=process.pid, error=traceback.format_exc()))
        for stream in streams:
            stream.close()
        write(evidence / 'raw_metadata_and_timings.json',
              {str(path.relative_to(temporary)): path.read_text()
               for path in temporary.rglob('*') if path.is_file() and path.suffix in {'.json', '.jsonl'}})
        if all(row.get('wait_observed') and row.get('process_group_gone') for row in cleanup):
            shutil.rmtree(temporary)
    unchanged = all(sha(Path(path)) == digest for path, digest in inputs.items())
    passed = failure is None and len(results) == 2 and unchanged and not temporary.exists()
    result = dict(at_utc=datetime.now(timezone.utc).isoformat(), status='passed' if passed else 'failed',
        failure=failure, elapsed_seconds=time.monotonic() - started, source_sha256=inputs,
        source_unchanged=unchanged, temporary_removed=not temporary.exists(), cleanup=cleanup,
        commands=commands, results=results, synthetic_only=True, gpu_tested=False,
        deployed=False, actual_incident_fsync_root_cause_proven=False,
        ML_imports=sorted(ML_NAMES & {name.split('.')[0] for name in sys.modules}))
    write(evidence / 'result.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'failure', 'elapsed_seconds',
                     'temporary_removed', 'deployed')}, ensure_ascii=False), flush=True)
    return 0 if passed and result['ML_imports'] == [] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', choices=tuple(SOURCES))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()

    def timeout_handler(signum, frame):
        raise TimeoutError('Bounded CPU control deadline reached')

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, 75.)
    if args.child:
        child(args.child, args.directory, args.rank)
    else:
        raise SystemExit(run())
