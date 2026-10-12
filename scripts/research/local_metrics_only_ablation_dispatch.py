"""Guard a future local metrics-only attempt with the shared cell/device locks.

Activate conda pllm before metadata checks. Call prepare(argv, *,
admission_path=ABSOLUTE_PATH, admission_sha256=SHA,
ledger_path=ABSOLUTE_PATH, ledger_sha256=SHA) with argv=['--variant', V,
'--seed', S]. prepare is read-only, imports no ML and never reads sequences.
main has the same interface and is reserved for a future reviewed tmux launch.
The publisher must verify fresh task/log identities before the existing tmux
runner, which clears sentinels; this guard is not that publisher.
CLI: original_training_python -I -B -u ABSOLUTE_GUARD_PATH
--admission-path PATH --admission-sha256 SHA --ledger-path PATH
--ledger-sha256 SHA --variant V --seed S. Do not run a real main now.

Immutable admission schema v7.local.metrics_only_admission.v1 contains study,
variant, seed, task_name, output_dir, log_dir, local_hostname, training_argv,
automatic_retry=false; path/SHA refs managed_local_plan, original_catalog,
source_freeze, batch_plan, guard_source, pool_adapter_source, runner, ledger,
local_device_lock; and prior_f000={launch_receipt:ref, completion:ref}.
The same v7.metrics_only.ablation_claims.v1 ledger must contain all 22 cells.
Exactly this cell is local_owned, with one attempt {transport:local, variant,
seed, task_name, output_dir, log_dir, lock_file:ref}. Other cells retain the
existing pool/completed/unclaimed/blocked states. cell_locks/<V>_seed<S>.lock
is shared with pool dispatch; ledger.parent/local_gpu0.lock serializes local
GPU0. Both are existing empty files, never created by this guard.

Future actual v7.local.tmux_launch.v2 records original training_argv and
runner/task_name/log_dir/started/command/source_manifest/source_manifest_sha256/
checkpoint_saving=false, plus admission, ledger, guard_source, guard_argv and
guard_execution path/SHA refs. Its command replaces only the original tmux
command's training tail with guard_argv(); it retains original env clearing.
v7.local.tmux_exit.v2 binds that launch/runner/task/command/started and actual
exit_code/done/failed, and repeats admission/ledger/guard_source/guard_argv/
guard_execution exactly from the launch. V1 readers need a separate v2 change.
guard_execution is only actual pre-exec admission evidence, never proof of
successful exec or training. Native COMPLETE and an actual exit remain needed.

No formal-completion gate is called: unrelated eight-rank formal jobs may run.
Current f000 must have its native complete epoch and an absent original PID;
these prove resource release and native artifacts, not a previous returncode.
No tasks, ledgers or listeners are created, retried or stopped by this file.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import runpy
import shlex
import socket
import sys


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
POOL_ADAPTER = ROOT / 'scripts/research/pool_metrics_only_ablation_dispatch.py'
POOL_ADAPTER_SHA = 'a9baf1390cdeeba315c7984c8602599e29033a356fd7649b82dcb206899a5fe8'
RUNNER = ROOT / 'scripts/run_in_tmux.sh'
RUNNER_SHA = '33dcb80073774e788dfdebc48d71b8de17b37d09fc1d7676433b0f805f0a5943'
F000_LAUNCH = ROOT / 'output/v7_local_ablation_20261005/launch_receipt.json'
F000_LAUNCH_SHA = '1d99c57ddb07cd1740b9d1697c6e1bb12beed62caac07bfc65262ab772d7e6be'


def _pool_api():
    """Load only the exact reviewed stdlib adapter, never its main()."""
    import hashlib
    require(POOL_ADAPTER.is_file() and not POOL_ADAPTER.is_symlink()
            and POOL_ADAPTER == POOL_ADAPTER.resolve()
            and hashlib.sha256(POOL_ADAPTER.read_bytes()).hexdigest() == POOL_ADAPTER_SHA,
            'Reviewed pool adapter source differs')
    api = runpy.run_path(str(POOL_ADAPTER), run_name='_local_metadata_helpers')
    require(api['sha'](POOL_ADAPTER) == POOL_ADAPTER_SHA, 'Pool adapter changed during loading')
    return api


def require(value, message):
    if not value:
        raise ValueError(message)


def _pid_present(pid):
    require(type(pid) is int and pid > 0, 'Actual positive local PID required')
    return Path(f'/proc/{pid}').exists()


def _empty_lock(pool, ref, path):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'} and ref['path'] == str(path),
            'Explicit existing lock path/SHA required')
    path = pool['canonical'](path)
    require(path.is_file() and path.stat().st_size == 0 and pool['sha'](path) == ref['sha256'],
            'Existing shared empty lock differs')
    stat = path.stat()
    return path, (stat.st_dev, stat.st_ino)


def _check_prior(pool, admission, frozen_runner):
    """Resource release evidence does not manufacture a previous exit code."""
    launch_ref, completion_ref = (admission['prior_f000'][k] for k in ('launch_receipt', 'completion'))
    require(launch_ref == dict(path=str(F000_LAUNCH), sha256=F000_LAUNCH_SHA),
            'Original f000 launch identity differs')
    launch = pool['bound'](launch_ref)
    prior_dir = pool['LOCAL'] / 'f000_seed42_attempt1'
    complete = pool['bound'](completion_ref, prior_dir / 'COMPLETE.json')
    require(launch.get('kind') == 'independent_fresh_attempt' and launch.get('checkpoint_saving') is False
            and launch.get('source_manifest') == str(pool['FREEZE'])
            and launch.get('source_manifest_sha256') == pool['FREEZE_SHA'], 'Original f000 provenance differs')
    snapshot = pool['bound'](admission['source_freeze'])['snapshot_root']
    require(launch['command'][1:5] == ['-I', '-B', '-u', str(Path(snapshot)
                / 'scripts/research/run_v7_large_batch_ablation.py')], 'Original f000 frozen CLI differs')
    previous_args = frozen_runner['parse_args'](launch['command'][5:])
    frozen_runner['validate_parameters'](previous_args)
    require(previous_args.variant == 'f000' and previous_args.seed == 42
            and previous_args.output_dir == prior_dir and previous_args.steps == 3750
            and previous_args.micro_batch == 32 and previous_args.grad_accum == 1
            and previous_args.eval_every == 250 and previous_args.resume is False
            and previous_args.save_checkpoints is False and previous_args.cpu_smoke is False
            and previous_args.data_dir == pool['DATA'] and previous_args.batch_plan == pool['PLAN']
            and previous_args.batch_plan_sha256 == pool['PLAN_SHA'], 'Original f000 protocol differs')
    require(not _pid_present(launch['pid']), 'Original f000 process is still present')
    require(not (prior_dir / 'FAILURE.json').exists(), 'Original f000 has failure evidence')
    summary_path, provenance_path = prior_dir / 'summary.json', prior_dir / 'run_provenance.json'
    summary = pool['read'](summary_path, complete['summary_sha256'])
    provenance = pool['read'](provenance_path, complete['provenance_sha256'])
    require(complete.get('status') == 'completed' and complete.get('global_step') == 3750
            and complete.get('kind') == provenance.get('kind') == 'exploratory_architecture_reconstruction'
            and complete.get('formal_training') is False and complete.get('biological_quality_claim') is False
            and summary.get('step') == 3750 and summary.get('records') == summary.get('subset_train_records') == 120000
            and summary.get('epoch') == 1 and summary.get('position') == 0
            and provenance.get('variant') == 'f000' and provenance.get('seed') == 42
            and provenance.get('checkpoint_saving') is False and provenance.get('scheduler_horizon') == 3750
            and provenance.get('micro_batch') == 32 and provenance.get('grad_accum') == 1
            and provenance.get('study', {}).get('name') == pool['STUDY'], 'f000 native completion/epoch differs')
    frozen = pool['bound'](admission['source_freeze'])
    recorded = provenance.get('source_sha256')
    require(isinstance(recorded, dict) and recorded and all(frozen['source_sha256'].get(k) == v
            for k, v in recorded.items()), 'f000 recorded provenance sources differ')


def _check_claim(pool, admission, ledger, candidate, ledger_path):
    require(ledger.get('schema') == 'v7.metrics_only.ablation_claims.v1'
            and ledger.get('study') == pool['STUDY'] and ledger.get('pool_dir') == str(pool['POOL'])
            and isinstance(ledger.get('cells'), list) and len(ledger['cells']) == 22,
            'Full shared 22-cell ledger required')
    wanted = admission['variant'], admission['seed']
    seen, owned = set(), []
    for cell in ledger['cells']:
        key = cell['variant'], cell['seed']
        require(type(cell['seed']) is int and key in pool['ALIASES'].values() and key not in seen
                and isinstance(cell.get('attempts'), list), 'Duplicate/unknown shared ledger cell')
        seen.add(key)
        require(cell.get('state') in ('local_owned', 'reserved_pool', 'unclaimed', 'completed', 'blocked'),
                'Unknown shared claim state')
        if cell['state'] == 'local_owned':
            owned.append(cell)
    require(len(owned) == 1 and (owned[0]['variant'], owned[0]['seed']) == wanted
            and len(owned[0]['attempts']) == 1, 'Another local cell is owned or selected claim is not exclusive')
    attempt = owned[0]['attempts'][0]
    expected = dict(transport='local', variant=wanted[0], seed=wanted[1],
        task_name=candidate['task_name'], output_dir=candidate['output_dir'], log_dir=candidate['log_dir'])
    require(all(attempt.get(k) == v for k, v in expected.items()), 'Local attempt/task/output/log claim differs')
    output = pool['canonical'](candidate['output_dir'])
    require(not output.exists(), 'Selected local output already exists')
    alias = f'p04_v7_{wanted[0]}_seed{wanted[1]}'
    original = pool['bound'](admission['original_catalog'])
    old_row = next(row for row in original['jobs'] if (row['variant'], row['seed']) == wanted)
    old_result = pool['OLD_POOL'] / 'results' / (old_row['job_id'] + '.json')
    if old_result.exists():
        require(pool['read'](old_result).get('state') != 'succeeded', 'Original cell already succeeded')
    require(not (pool['OLD_POOL'] / 'runs' / old_row['job_id'] / 'COMPLETE.json').exists(),
            'Original cell already has completion evidence')
    for parent in (pool['POOL'], pool['OLD_POOL']):
        for folder in ('pending', 'running'):
            for path in (parent / folder).glob('*.json'):
                require(pool['read'](path).get('variant') != alias, 'Selected cell already has an active pool attempt')
        for path in (parent / 'results').glob('*.json'):
            result = pool['read'](path)
            if result.get('job', {}).get('variant') == alias or result.get('variant') == alias:
                require(result.get('state') != 'succeeded', 'Selected pool cell already succeeded')
                require(parent != pool['POOL'], 'Selected cell already has a fresh pool result; no automatic retry')
    log = pool['canonical'](candidate['log_dir'])
    if log.exists():
        require(log.is_dir() and {p.name for p in log.iterdir()} <= {'lock', 'started', 'run.log'},
                'Existing local log identity contains previous attempt evidence')
    return _empty_lock(pool, attempt['lock_file'], Path(ledger_path).parent / 'cell_locks'
        / f'{wanted[0]}_seed{wanted[1]}.lock')


def guard_argv(args, *, admission_path, admission_sha256, ledger_path, ledger_sha256):
    """Construct only the future guard command; no shell evaluation or writes."""
    return [args.training_argv[0], '-I', '-B', '-u', str(Path(__file__).resolve()),
        '--admission-path', str(admission_path), '--admission-sha256', admission_sha256,
        '--ledger-path', str(ledger_path), '--ledger-sha256', ledger_sha256,
        '--variant', args.variant, '--seed', str(args.seed)]


def prepare(argv, *, admission_path, admission_sha256, ledger_path, ledger_sha256):
    """Check immutable metadata and fixed source bytes, without locking or ML."""
    require(all(sum(v.split('=', 1)[0] == flag for v in argv) == 1 for flag in ('--variant', '--seed')),
            'Selected variant/seed must appear once')
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--seed', type=int, choices=(42, 43), required=True)
    args = parser.parse_args(argv)
    pool = _pool_api()
    admission = pool['read'](admission_path, admission_sha256)
    ledger = pool['read'](ledger_path, ledger_sha256)
    own = Path(__file__).resolve()
    require(admission.get('schema') == 'v7.local.metrics_only_admission.v1'
            and admission.get('study') == pool['STUDY']
            and (admission.get('variant'), admission.get('seed')) == (args.variant, args.seed)
            and type(admission.get('seed')) is int and admission.get('automatic_retry') is False
            and admission.get('local_hostname') == socket.gethostname(), 'Local admission identity/host differs')
    require(admission.get('guard_source') == dict(path=str(own), sha256=pool['sha'](own))
            and admission.get('pool_adapter_source') == dict(path=str(POOL_ADAPTER), sha256=POOL_ADAPTER_SHA)
            and admission.get('runner') == dict(path=str(RUNNER), sha256=RUNNER_SHA), 'Bound guard/helper/tmux source differs')
    require(pool['sha'](RUNNER) == RUNNER_SHA
            and admission.get('ledger') == dict(path=str(pool['canonical'](ledger_path)), sha256=ledger_sha256)
            and admission.get('managed_local_plan') == dict(path=str(pool['MANAGED']), sha256=pool['MANAGED_SHA'])
            and admission.get('original_catalog') == dict(path=str(pool['ORIGINAL']), sha256=pool['ORIGINAL_SHA']),
            'Admission ledger/original launch bindings differ')
    managed = pool['bound'](admission['managed_local_plan'])
    matches = [item for item in managed['commands'] if (item['variant'], item['seed']) == (args.variant, args.seed)]
    require(len(matches) == 1, 'Selected cell is not an existing managed candidate')
    candidate = matches[0]
    require(all(admission.get(k) == candidate[k] for k in ('task_name', 'output_dir', 'log_dir', 'training_argv')),
            'Admission must retain existing task/output/log/training argv')
    runner, snapshot = pool['check_sources_and_batch'](admission)
    pool['check_data']()
    command = list(candidate['training_argv'])
    require(command[:5] == [command[0], '-I', '-B', '-u', str(snapshot / 'scripts/research/run_v7_large_batch_ablation.py')],
            'Frozen isolated training CLI differs')
    selected = runner['parse_args'](command[5:])
    runner['validate_parameters'](selected)
    require(selected.variant == args.variant and selected.seed == args.seed
            and selected.output_dir == Path(candidate['output_dir']) and selected.data_dir == pool['DATA']
            and (selected.steps, selected.eval_every, selected.micro_batch, selected.grad_accum) == (3750, 250, 32, 1)
            and selected.batch_plan == pool['PLAN'] and selected.batch_plan_sha256 == pool['PLAN_SHA']
            and selected.resume is False and selected.cpu_smoke is False and selected.save_checkpoints is False,
            'Local attempt must retain the fresh metrics-only protocol')
    runner['checked_batch_plan'](selected)
    _check_prior(pool, admission, runner)
    args.lock_path, args.lock_identity = _check_claim(pool, admission, ledger, candidate, ledger_path)
    args.device_lock_path, args.device_lock_identity = _empty_lock(pool, admission['local_device_lock'],
        Path(ledger_path).parent / 'local_gpu0.lock')
    args.training_argv, args.candidate, args.admission, args.pool = command, candidate, admission, pool
    args.admission_ref = dict(path=str(pool['canonical'](admission_path)), sha256=admission_sha256)
    args.ledger_ref = dict(path=str(pool['canonical'](ledger_path)), sha256=ledger_sha256)
    return args, command


def _process_snapshot(pool):
    observer = pool['_api'](pool['PARENT_API'], pool['PARENT_API_SHA'])
    identity = observer['coordinator_api']()['process_identity'](os.getpid())
    return dict(pid=identity['pid'], start_ticks=identity['start_ticks'], boot_id=identity['boot_id'],
        argv=observer['read_parent_cmdline'](os.getpid()))


def _check_runtime(args, expected_guard):
    pool, candidate = args.pool, args.candidate
    require(os.getcwd() == str(pool['ROOT']) and sys.executable == args.training_argv[0],
            'Actual local cwd/interpreter differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
            and all(os.environ.get(k) == v for k, v in pool['RUN_ENV'].items())
            and not any(k in os.environ for k in (*pool['DDP_ENV'], 'PYTHONPATH')),
            'Actual local environment differs from the managed protocol')
    command = candidate['command']
    require(command[:4] == ['bash', str(RUNNER), candidate['task_name'], 'env']
            and command[-len(args.training_argv):] == args.training_argv, 'Managed tmux command differs')
    envelope = command[4:-len(args.training_argv)] + expected_guard
    log = pool['canonical'](candidate['log_dir'])
    require(log.is_dir() and (log / 'started').is_file() and not (log / 'started').is_symlink(),
            'Actual tmux started sentinel missing')
    with pool['canonical'](log / 'run.log').open() as stream:
        header = [stream.readline(65537) for _ in range(3)]
    require(all(0 < len(line) <= 65536 for line in header)
            and header[0].strip().endswith(' START ' + candidate['task_name'])
            and header[1].startswith('  cmd:    ')
            and shlex.split(header[1].split('cmd:', 1)[1].strip()) == ['env', *envelope]
            and header[2].startswith('  TMPDIR:'), 'Actual tmux header/guard command differs')
    identity = _process_snapshot(pool)
    require(identity['pid'] == os.getpid() and identity['argv'] == expected_guard,
            'Actual guard process argv differs')
    return identity


def _lock_identity(descriptor, path, expected):
    opened, current = os.fstat(descriptor), path.stat()
    require(opened.st_size == 0 and (opened.st_dev, opened.st_ino) == expected
            == (current.st_dev, current.st_ino), 'Shared lock inode changed')
    return dict(st_dev=opened.st_dev, st_ino=opened.st_ino, inheritable=os.get_inheritable(descriptor))


def _execution_receipt(args, process, cell_fd, device_fd):
    payload = dict(schema='v7.local.metrics_only_guard_execution.v1',
        at_utc=datetime.now(timezone.utc).isoformat(), admission=args.admission_ref, ledger=args.ledger_ref,
        guard_source=args.admission['guard_source'], variant=args.variant, seed=args.seed,
        task_name=args.candidate['task_name'], training_argv=args.training_argv,
        process_identity=process, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        lock_identity=_lock_identity(cell_fd, args.lock_path, args.lock_identity),
        device_lock_identity=_lock_identity(device_fd, args.device_lock_path, args.device_lock_identity),
        scope='pre_exec_admission_only')
    path = args.pool['canonical'](Path(args.candidate['log_dir']) / 'guard_execution.json')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None, *, admission_path, admission_sha256, ledger_path, ledger_sha256):
    argv = list(sys.argv[1:] if argv is None else argv)
    kwargs = dict(admission_path=admission_path, admission_sha256=admission_sha256,
                  ledger_path=ledger_path, ledger_sha256=ledger_sha256)
    args, command = prepare(argv, **kwargs)
    descriptors = []
    try:
        for path, identity in ((args.device_lock_path, args.device_lock_identity), (args.lock_path, args.lock_identity)):
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            descriptors.append(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _lock_identity(descriptor, path, identity)
        # Repeat actual SHA/claim/data/source/PID checks while both locks are held.
        refreshed, command = prepare(argv, **kwargs)
        require(refreshed.lock_identity == args.lock_identity
                and refreshed.device_lock_identity == args.device_lock_identity, 'Lock identity changed after admission')
        args = refreshed
        expected_guard = guard_argv(args, **kwargs)
        process = _check_runtime(args, expected_guard)
        for descriptor in descriptors:
            os.set_inheritable(descriptor, True)
        device_fd, cell_fd = descriptors
        _execution_receipt(args, process, cell_fd, device_fd)
        os.execv(sys.executable, command)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('admission-path', 'admission-sha256', 'ledger-path', 'ledger-sha256', 'variant', 'seed'):
        parser.add_argument('--' + name, required=True)
    entry = parser.parse_args()
    main(['--variant', entry.variant, '--seed', entry.seed], admission_path=Path(entry.admission_path),
        admission_sha256=entry.admission_sha256, ledger_path=Path(entry.ledger_path), ledger_sha256=entry.ledger_sha256)
