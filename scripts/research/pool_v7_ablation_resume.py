"""Resume one explicitly bound incomplete large-batch cell; never migrate p02.

Activate conda pllm. A future reviewed dispatcher may call
main(original_pool_argv, ticket_catalog_sha256=<immutable catalog SHA>).
prepare(argv, ticket_catalog_sha256=...) is read-only. The fixed catalog is
/project_c20250601/251105016/project/dllm_test/output/
v7_large_batch_ablation_20261002/ablation_resume_tickets.json; it is not created
here. No standalone invocation can supply an unbound catalog override.
Actual execution requires an explicit ticket, terminal original job, same-cell
lock, no other active cell, exact checkpoint SHA and original frozen guards.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/v7_large_batch_ablation_20261002'
POOL = ROOT/'output/immune_research_pool_c20250601_20260930'
TICKETS = OWN/'ablation_resume_tickets.json'
LOCKS = OWN/'ablation_resume_locks'
BASE = OWN/'launcher_source_v2/scripts/research/pool_large_batch_ablation_v2_dispatch.py'
BASE_SHA = '7cf225f5e2a4db198a093ccba9baf1d5bd81e7db1b1e4768504a5228fcced4f5'
CATALOG = OWN/'jobs.json'
CATALOG_SHA = '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d'
PLAN = OWN/'batch_selection_v1.json'
PLAN_SHA = '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04'
FREEZE = OWN/'source_snapshot_v1.json'
FREEZE_SHA = '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'
DATA = ROOT/'output/v7_architecture_pilot_20261002/sample_context_plus20k'
STUDY = 'v7_context_factorial_120k_large_batch_v1'
LOGS = {'train.log', 'worker.log', 'stdout.log', 'stderr.log'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve() and
            not any(part.is_symlink() for part in (path, *path.parents)), 'Canonical nonsymlink path required')
    return path


def sha(path):
    path = canonical(path)
    require(path.is_file(), 'Required original file missing: '+str(path))
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def decode(raw):
    def invalid(value):
        raise ValueError('Nonfinite JSON metadata: '+value)
    return json.loads(raw, parse_constant=invalid)


def bound(ref, expected_path, parse=True):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'} and ref['path'] == str(expected_path)
            and isinstance(ref['sha256'], str) and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'Explicit exact file/SHA binding required')
    require(sha(expected_path) == ref['sha256'], 'Original bytes changed: '+str(expected_path))
    if not parse:
        return None
    raw = expected_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Metadata changed while reading')
    return decode(raw)


def base_api():
    require(sha(BASE) == BASE_SHA, 'Frozen large-v2 entry changed')
    value = runpy.run_path(str(BASE))
    require(sha(BASE) == BASE_SHA, 'Frozen large-v2 changed while loading')
    return value


def request(argv, ticket_catalog_sha256):
    # No checkpoint access occurs before the explicitly bound catalog exists.
    require(TICKETS.is_file(), 'No immutable recovery ticket catalog has been published')
    book = bound({'path': str(TICKETS), 'sha256': ticket_catalog_sha256}, TICKETS)
    expected = {'schema': 'v7.ablation.resume_tickets.v1', 'study': STUDY,
        'original_catalog': {'path': str(CATALOG), 'sha256': CATALOG_SHA},
        'batch_plan': {'path': str(PLAN), 'sha256': PLAN_SHA},
        'source_freeze': {'path': str(FREEZE), 'sha256': FREEZE_SHA},
        'dispatcher': {'path': str(BASE), 'sha256': BASE_SHA}}
    require(all(book.get(key) == value for key, value in expected.items()) and book.get('automatic_retry') is False,
            'Recovery catalog group/protocol differs')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    flags = ('variant', 'mode', 'steps', 'data-dir', 'profile-dir', 'output-dir', 'resume-from-checkpoint')
    require(len(argv) == 14 and set(argv[::2]) == {'--'+name for name in flags}, 'Seven unique original pool flags required')
    for name in flags:
        parser.add_argument('--'+name, required=True, type=int if name == 'steps' else str)
    args = parser.parse_args(argv)
    outer = canonical(args.output_dir)
    attempt = outer.name
    tickets = book.get('tickets')
    require(type(tickets) is dict and attempt in tickets, 'Attempt absent from immutable recovery catalog')
    origins = [item['original_job_id'] for item in tickets.values()]
    require(len(origins) == len(set(origins)), 'One explicit attempt per original cell in this catalog')
    ticket = tickets[attempt]
    attempt_pool = canonical(ticket['attempt_pool'])
    original = ticket['original_job_id']
    catalog = bound(book['original_catalog'], CATALOG)
    rows = [row for row in catalog['jobs'] if row['job_id'] == original]
    require(len(rows) == 1 and len(catalog['jobs']) == 22 and catalog['study'] == STUDY, 'Require an original 22-cell job')
    row = rows[0]
    require(re.fullmatch(r'[0-9]+_'+re.escape(row['alias'])+r'_[0-9a-f]{8}', attempt)
            and attempt != original and row['alias'].startswith('p04_v7_'), 'Different attempt bound to the original large-batch cell required')
    inner = canonical(POOL/'runs'/original)
    prior = ticket['prior_step']
    require(type(prior) is int and 0 <= prior < 3750 and prior % 250 == 0, 'Prior step must be a saved evaluation boundary')
    job = dict(variant=args.variant, mode=args.mode, steps=args.steps, data_dir=args.data_dir,
               resume_from_checkpoint=args.resume_from_checkpoint)
    require(job == ticket['job'] == dict(variant=row['alias'], mode='train', steps=3750-prior,
        data_dir=str(DATA), resume_from_checkpoint=str(inner/'latest.pt')), 'Remaining updates or resume envelope differs')
    require(outer == attempt_pool/'runs'/attempt and ticket['attempt_run'] == str(outer)
            and ticket['original_run'] == str(inner) and canonical(args.profile_dir) == attempt_pool/'profiles'/row['alias'],
            'Outer attempt and inner original directories differ')
    require(type(ticket['timeout_seconds']) is int and 0 < ticket['timeout_seconds'] <= 86400,
            'Require an explicit bounded timeout within the original pool job limit')
    return args, book, ticket, row, inner, outer


def no_live(row, attempt, job, checkpoint, attempt_pool):
    current = 0
    for pool in dict.fromkeys((POOL, attempt_pool)):
        for folder in ('pending', 'running', 'workers'):
            directory = canonical(pool/folder)
            require(directory.is_dir(), 'Pool ownership directory missing')
            for path in directory.glob('*.json'):
                value = decode(canonical(path).read_bytes())
                if folder == 'workers' and value.get('state') != 'running':
                    continue
                spec = value.get('job', {}) if folder == 'workers' else value
                job_id = value.get('job_id') if folder == 'workers' else path.stem
                same = spec.get('variant') == row['alias'] or spec.get('resume_from_checkpoint') == str(checkpoint)
                if not same:
                    continue
                require(pool == attempt_pool and job_id == attempt and spec == job,
                        'Another pending/running worker owns this cell')
                if folder == 'workers':
                    require(value.get('pid') in (None, os.getpid()), 'Current attempt belongs to another controller')
                else:
                    current += 1
    require(current <= 1, 'Duplicate pending/running attempt')


def metadata_state(ticket, inner):
    refs = ticket['evidence']
    provenance = bound(refs['provenance'], inner/'run_provenance.json')
    summary = bound(refs['summary'], inner/'summary.json')
    bound(refs['curve'], inner/'curve.jsonl', parse=False)
    points = [decode(line) for line in (inner/'curve.jsonl').read_bytes().splitlines() if line.strip()]
    require(sha(inner/'curve.jsonl') == refs['curve']['sha256'], 'Curve changed while reading')
    require(points and points[0]['step'] == 0 and all(type(point['step']) is int for point in points)
            and all(a['step'] < b['step'] for a, b in zip(points, points[1:]))
            and points[-1] == summary and summary['step'] == ticket['prior_step'], 'Curve/summary/prior-step transaction differs')
    require(summary['records'] == ticket['prior_step']*32, 'Saved record exposures differ')
    return provenance


def provenance_matches(previous, row, command, api):
    runtime = api['_runtime']
    freeze = bound({'path': str(FREEZE), 'sha256': FREEZE_SHA}, FREEZE)
    plan = bound({'path': str(PLAN), 'sha256': PLAN_SHA}, PLAN)
    runner = runpy.run_path(command[4])  # Frozen stdlib parser, not main/prepare.
    parsed = runner['parse_args'](command[5:]+['--resume'])
    runner['validate_parameters'](parsed)
    runner['check_resume_provenance'](previous, parsed)
    require(previous.get('variant') == row['variant'] and previous.get('seed') == row['seed']
            and previous.get('kind') == 'exploratory_architecture_reconstruction'
            and all(previous.get(key) is False for key in ('formal_training', 'biological_quality_claim', 'generation')),
            'Original cell/model/provenance differs')
    require([previous.get(key) for key in ('micro_batch', 'grad_accum', 'global_batch', 'scheduler_horizon',
            'evaluation_interval', 'warmup', 'learning_rate')] == [32, 1, 32, 3750, 250, 100, 1e-4], 'Resume cannot change batch or schedule')
    require(previous['study']['batch_plan'] == {'path': str(PLAN), 'sha256': PLAN_SHA,
            'profile_evidence': plan['profile_evidence']} and previous['data']['sha256'] == runtime['SAMPLE_SHA']
            and previous['data']['path'] == str(DATA/'sample_manifest.json'), 'Resume data/selected plan differs')
    sources = previous['source_sha256']
    require(type(sources) is dict and all(sources.get(name) == freeze['source_sha256'][name] for name in
        ('scripts/research/run_v7_large_batch_ablation.py', 'scripts/research/run_v7_architecture_pilot.py', 'scripts/research/context_pilot_metrics.py'))
        and all(freeze['source_sha256'].get(name) == value for name, value in sources.items()), 'Loaded source identity differs')
    require(all(sha(Path(freeze['snapshot_root'])/name) == value for name, value in sources.items()), 'Actual loaded source bytes changed')


def prepare(argv, *, ticket_catalog_sha256):
    args, book, ticket, row, inner, outer = request(argv, ticket_catalog_sha256)
    attempt_pool = canonical(ticket['attempt_pool'])
    refs = ticket['evidence']
    result = bound(refs['result'], POOL/'results'/f'{row["job_id"]}.json')
    finished = bound(refs['finished'], POOL/'finished'/f'{row["job_id"]}.json')
    require(result.get('job_id') == row['job_id'] and result.get('job') == row['job'] and finished == row['job']
            and result.get('state') == 'failed' and type(result.get('finished')) in (int, float)
            and math.isfinite(result['finished']), 'Original incomplete task is not terminal failed')
    require(inner.is_dir() and not (inner/'COMPLETE.json').exists(), 'Only an incomplete original run may resume')
    require(not (attempt_pool/'results'/f'{outer.name}.json').exists()
            and not (attempt_pool/'finished'/f'{outer.name}.json').exists(), 'Attempt is already terminal')
    require(not outer.exists() or (outer.is_dir() and all(path.name in LOGS and path.is_file()
            and not path.is_symlink() for path in outer.iterdir())), 'Attempt output cannot be overwritten')
    no_live(row, outer.name, ticket['job'], inner/'latest.pt', attempt_pool)
    previous = metadata_state(ticket, inner)
    api = base_api()
    original = ['--variant', row['alias'], '--mode', 'train', '--steps', '3750', '--data-dir', str(DATA),
                '--profile-dir', str(POOL/'profiles'/row['alias']), '--output-dir', str(inner)]
    _, command = api['prepare'](original)
    require(command[command.index('--output-dir')+1] == str(inner), 'Frozen command uses another inner directory')
    provenance_matches(previous, row, command, api)
    # Future execution only: SHA bytes after terminal/no-live/source checks, never torch.load.
    bound(refs['checkpoint'], inner/'latest.pt', parse=False)
    no_live(row, outer.name, ticket['job'], inner/'latest.pt', attempt_pool)
    require(sha(TICKETS) == ticket_catalog_sha256, 'Recovery tickets changed during admission')
    return {'args': args, 'ticket': ticket, 'row': row, 'inner': inner, 'outer': outer,
            'command': command+['--resume'], 'provenance': previous, 'ticket_catalog_sha256': ticket_catalog_sha256,
            'prior_curve_bytes': (inner/'curve.jsonl').stat().st_size}


@contextmanager
def cell_lock(original):
    require(re.fullmatch(r'[0-9]+_p04_v7_[a-z0-9_]+_[0-9a-f]{8}', original), 'Invalid original cell lock identity')
    canonical(LOCKS).mkdir(parents=True, exist_ok=True)
    path = LOCKS/f'{original}.lock'
    descriptor = os.open(path, os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield descriptor
    finally:
        os.close(descriptor)


def atomic_once(path, value):
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def supervise(command, seconds, lock_fd):
    """Same outer PG; child retains the cell flock even if cleanup is uncertain."""
    require(type(lock_fd) is int and lock_fd >= 0, 'An actual held cell-lock FD is required')
    process, started, error, normal_wait, cleanup_wait, child_group = None, time.monotonic(), None, False, False, None
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(number, frame):
        raise InterruptedError('Recovery controller received signal '+str(number))
    for number in previous:
        signal.signal(number, interrupted)
    try:
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=False, pass_fds=(lock_fd,))
        child_group = os.getpgid(process.pid)
        require(child_group == os.getpgrp(), 'Child left the outer process group')
        process.wait(timeout=seconds)
        normal_wait = True
    except BaseException as caught:
        error = repr(caught)
    finally:
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                    cleanup_wait = True
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                    cleanup_wait = True
        except BaseException as caught:
            error = str(error)+'; cleanup: '+repr(caught)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    return {'pid': None if process is None else process.pid, 'parent_pid': os.getpid(),
            'process_group_id': child_group, 'returncode': None if process is None else process.returncode,
            'wait_returned': normal_wait or cleanup_wait, 'normal_wait_returned': normal_wait,
            'child_reaped': process is not None and process.poll() is not None,
            'cell_lock_fd_passed': process is not None,
            'error': error, 'elapsed_seconds': time.monotonic()-started}


def completed(prepared, child):
    require(child.get('normal_wait_returned') is True and child.get('child_reaped') is True
            and child.get('error') is None and child.get('wait_returned') is True
            and child.get('returncode') == 0, 'No successful actual child wait')
    inner = prepared['inner']
    paths = {key: inner/name for key, name in {'completion': 'COMPLETE.json', 'summary': 'summary.json',
        'provenance': 'run_provenance.json', 'curve': 'curve.jsonl', 'checkpoint': 'latest.pt'}.items()}
    for path in paths.values():
        canonical(path)
    complete, summary, provenance = (decode(paths[key].read_bytes()) for key in ('completion', 'summary', 'provenance'))
    require(complete.get('status') == 'completed' and complete.get('global_step') == 3750
            and complete.get('kind') == 'exploratory_architecture_reconstruction'
            and complete.get('formal_training') is False and complete.get('biological_quality_claim') is False,
            'Inner full-epoch completion is absent')
    require(provenance == prepared['provenance'] and complete['summary_sha256'] == sha(paths['summary'])
            and complete['provenance_sha256'] == sha(paths['provenance']), 'Inner final provenance/summary differs')
    curve = paths['curve'].read_bytes()
    require(hashlib.sha256(curve[:prepared['prior_curve_bytes']]).hexdigest() ==
            prepared['ticket']['evidence']['curve']['sha256'], 'Original curve prefix changed during continuation')
    points = [decode(line) for line in curve.splitlines() if line.strip()]
    require(points and points[-1] == summary and summary['step'] == 3750 and summary['records'] == 120000
            and summary['record_exposures'] == 120000 and summary['epoch'] == 1 and summary['position'] == 0,
            'Inner epoch/record/curve completion differs')
    freeze = bound({'path': str(FREEZE), 'sha256': FREEZE_SHA}, FREEZE)
    require(all(sha(Path(freeze['snapshot_root'])/name) == expected for name, expected in provenance['source_sha256'].items()),
            'Actual source changed during continuation')
    return {'schema': 'v7.ablation.resume_attempt_completion.v1', 'status': 'completed',
        'study': STUDY, 'variant': prepared['row']['variant'], 'seed': prepared['row']['seed'],
        'attempt_id': prepared['outer'].name, 'original_job_id': prepared['row']['job_id'],
        'origin_pool': str(POOL), 'attempt_pool': prepared['ticket']['attempt_pool'],
        'original_run': str(inner), 'prior_step': prepared['ticket']['prior_step'], 'global_step': 3750,
        'new_optimizer_updates': 3750-prepared['ticket']['prior_step'], 'child': child,
        'ticket_catalog': {'path': str(TICKETS), 'sha256': prepared['ticket_catalog_sha256']},
        'inner_evidence': {key: {'path': str(path), 'sha256': sha(path)} for key, path in paths.items()},
        'formal_training': False, 'biological_quality_claim': False}


def main(argv=None, *, ticket_catalog_sha256=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _, _, ticket, row, _, _ = request(argv, ticket_catalog_sha256)
    require(re.fullmatch('[0-7]', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'One actual pool GPU mask required')
    with cell_lock(row['job_id']) as lock_fd:
        prepared = prepare(argv, ticket_catalog_sha256=ticket_catalog_sha256)
        outer = prepared['outer']
        outer.mkdir(parents=True, exist_ok=True)
        atomic_once(outer/'ATTEMPT.json', {'schema': 'v7.ablation.resume_attempt.v1',
            'ticket_catalog_sha256': ticket_catalog_sha256, 'ticket': ticket, 'command': prepared['command'],
            'controller_pid': os.getpid(), 'process_group_id': os.getpgrp(), 'original_run': str(prepared['inner'])})
        try:
            child = supervise(prepared['command'], ticket['timeout_seconds'], lock_fd)
            atomic_once(outer/'CHILD_EXIT.json', child)
            evidence = completed(prepared, child)
            atomic_once(outer/'COMPLETE.json', evidence)
        except BaseException as error:
            atomic_once(outer/'FAILURE.json', {'error': repr(error), 'completion_claim': False})
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
