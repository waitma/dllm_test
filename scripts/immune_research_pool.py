"""Shared-filesystem workers for independent, single-GPU research experiments.

After activating pllm:
  python scripts/immune_research_pool.py init --pool-dir /absolute/new/pool
  python scripts/immune_research_pool.py work --pool-dir /absolute/pool
  python scripts/immune_research_pool.py enqueue --pool-dir /absolute/pool \
      --variant r00_a4_control --mode train --steps 1000 --data-dir /absolute/clean/data
Workers run inside the GPU allocation; init/enqueue run on the development host.
"""
import argparse
import ast
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def variants():
    # Read the registry without importing torch or initializing CUDA on the host.
    tree = ast.parse((ROOT/'dllm/pipelines/immune_llada/research/model.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VARIANTS' for t in node.targets):
            if not isinstance(node.value, ast.Dict):
                raise ValueError('Research registry must be a literal dictionary')
            keys = [ast.literal_eval(key) for key in node.value.keys]
            if any(type(key) is not str for key in keys) or len(keys) != len(set(keys)):
                raise ValueError('Duplicate or invalid variant registry key')
            return ast.literal_eval(node.value)
    raise ValueError('Missing research registry')


def write_json(path, value):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temporary, path)


def validate(job):
    if set(job)-{'variant', 'mode', 'steps', 'data_dir', 'resume_from_checkpoint'}:
        raise ValueError('Unknown job fields; arbitrary shell commands are not accepted')
    if job.get('variant') not in variants() or job.get('mode') not in {'profile-synthetic', 'train'}:
        raise ValueError('Unknown variant or mode')
    if variants()[job['variant']].get('diagnostic_only'):
        if (job != dict(variant='d01_pilot_runtime_gpu_r20_v1', mode='profile-synthetic', steps=1)
                or type(job.get('steps')) is not int):
            raise ValueError('Diagnostic requires the exact d01 profile-synthetic/1 envelope')
        return
    limit = 5000 if job['mode'] == 'train' else 100
    if type(job.get('steps')) is not int or not 1 <= job['steps'] <= limit:
        raise ValueError('Invalid step budget')
    for key in ('data_dir', 'resume_from_checkpoint'):
        if key in job and (job['mode'] != 'train' or not isinstance(job[key], str) or not Path(job[key]).is_absolute()):
            raise ValueError(f'{key} requires a train job and an absolute path')
    if job['mode'] == 'train' and 'data_dir' not in job:
        raise ValueError('Training requires an explicit certified data directory')


def enqueue(pool, job):
    validate(job)
    if not (pool/'pool.json').is_file() or (pool/'STOP').exists() or (pool/'STOP_NOW').exists():
        raise ValueError('Pool is not initialized or is stopped')
    job_id = f'{time.time_ns()}_{job["variant"]}_{uuid.uuid4().hex[:8]}'
    write_json(pool/'pending'/f'{job_id}.json', job)
    return job_id


def claim(pool):
    for path in sorted((pool/'pending').glob('*.json')):
        destination = pool/'running'/path.name
        try:
            path.rename(destination)
            return destination
        except FileNotFoundError:
            continue
    return None


def command(pool, job_id, job):
    validate(job)
    result = [sys.executable, '-u', str(ROOT/'scripts/run_immune_research.py'),
              '--variant', job['variant'], '--mode', job['mode'], '--steps', str(job['steps']),
              '--profile-dir', str(pool/'profiles'/job['variant']),
              '--output-dir', str(pool/'runs'/job_id)]
    for key in ('data_dir', 'resume_from_checkpoint'):
        if key in job:
            result += ['--'+key.replace('_', '-'), job[key]]
    return result


def child_environment(device):
    env = dict(os.environ)
    for key in list(env):
        if key in {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                   'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'} or key.startswith(('TORCHELASTIC_', 'OMPI_', 'PMI_', 'PMIX_', 'SLURM_', 'ACCELERATE_')):
            env.pop(key)
    env['CUDA_VISIBLE_DEVICES'] = str(device)
    env['USE_TF'] = '0'
    return env


def stop_process(process):
    # Include profile-candidate grandchildren in shutdown.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def prune_checkpoints(pool, run):
    """Keep two latest checkpoints plus any best or queued-resume checkpoint."""
    checkpoints = sorted((p for p in run.glob('checkpoint-*')
                          if p.is_dir() and not p.is_symlink() and p.name[11:].isdigit()),
                         key=lambda p: int(p.name[11:]))
    protected = {p.resolve() for p in checkpoints[-2:]}
    for state in run.glob('checkpoint-*/trainer_state.json'):
        best = json.loads(state.read_text()).get('best_model_checkpoint')
        if best:
            protected.add(Path(best).resolve())
    for folder in ('pending', 'running'):
        for spec in (pool/folder).glob('*.json'):
            resume = json.loads(spec.read_text()).get('resume_from_checkpoint')
            if resume:
                protected.add(Path(resume).resolve())
    removed = []
    for checkpoint in checkpoints:
        if checkpoint.resolve() not in protected:
            shutil.rmtree(checkpoint)
            removed.append(str(checkpoint))
    return removed


def gpu_telemetry(pool, devices, stop):
    while not stop.is_set():
        record = dict(time=time.time(), hostname=socket.gethostname(), gpus=[])
        try:
            result = subprocess.run(['nvidia-smi', '--id', ','.join(devices),
                '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
            for row in csv.reader(result.stdout.splitlines()):
                record['gpus'].append(dict(index=row[0].strip(), uuid=row[1].strip(),
                    utilization_gpu=float(row[2]), memory_used_mib=float(row[3]), memory_total_mib=float(row[4])))
        except Exception as exc:
            record['error'] = str(exc)
        write_json(pool/'gpu_status.json', record)
        stop.wait(30)


class StandbyLoad:
    def __init__(self, pool, device, worker_id, deadline, enabled):
        self.pool, self.device, self.worker_id = pool, device, worker_id
        self.deadline, self.enabled = deadline, enabled
        self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()

    def stop(self):
        if self.process is not None:
            stop_process(self.process)
            self.process = None

    def maintain(self):
        if not self.enabled or (self.pool/'DISABLE_KEEPALIVE').exists():
            self.stop()
            return dict(state='idle')
        if self.process is None or self.process.poll() is not None:
            with (self.pool/'logs'/f'standby_{self.worker_id}.log').open('a') as log:
                self.process = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/gpu_standby_load.py'),
                    '--duty-cycle', '0.7', '--max-seconds', str(max(1, self.deadline-time.monotonic()))],
                    cwd=ROOT, env=child_environment(self.device), stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
        return dict(state='standby_load', pid=self.process.pid, maintenance_only=True)


def worker(pool, device, stop, deadline, idle_seconds, job_seconds, standby_load=False):
    worker_id = f'{socket.gethostname()}_{device}'.replace('/', '_')
    status_path = pool/'workers'/f'{worker_id}.json'
    failed = False
    with (pool/'workers'/f'{worker_id}.lock').open('a') as lock, StandbyLoad(
            pool, device, worker_id, deadline, standby_load) as standby:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle_since = time.monotonic()
        while (not stop.is_set() and time.monotonic() < deadline
               and not (pool/'STOP').exists() and not (pool/'STOP_NOW').exists()):
            path = claim(pool)
            if path is None:
                write_json(status_path, dict(standby.maintain(), time=time.time(), device=device))
                if idle_seconds and time.monotonic()-idle_since >= idle_seconds:
                    break
                stop.wait(min(5, idle_seconds) if idle_seconds else 5)
                continue
            standby.stop()
            job_id = path.stem
            started = time.time()
            process = None
            record = dict(job_id=job_id, worker=worker_id, device=device, started=started)
            try:
                job = json.loads(path.read_text())
                argv = command(pool, job_id, job)
                record.update(job=job, argv=argv, state='running')
                write_json(status_path, record)
                with (pool/'logs'/f'{job_id}.log').open('w') as log:
                    process = subprocess.Popen(argv, cwd=ROOT, env=child_environment(device),
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    job_deadline = min(deadline, time.monotonic()+job_seconds)
                    while process.poll() is None:
                        if stop.is_set() or (pool/'STOP_NOW').exists() or time.monotonic() >= job_deadline:
                            stop_process(process)
                            raise TimeoutError('Worker interrupted or time limit reached')
                        write_json(status_path, dict(record, heartbeat=time.time(), pid=process.pid))
                        stop.wait(2)
                record['returncode'] = process.returncode
                if process.returncode:
                    raise RuntimeError(f'Runner exit {process.returncode}; see per-job log')
                if not (pool/'runs'/job_id/'COMPLETE.json').is_file():
                    raise RuntimeError('Runner exited without completion evidence')
                record['state'] = 'succeeded'
                if job['mode'] == 'train':
                    try:
                        record['removed_checkpoints'] = prune_checkpoints(pool, pool/'runs'/job_id)
                    except Exception as exc:
                        record['cleanup_error'] = repr(exc)
            except Exception as exc:
                record.update(state='failed', error=repr(exc))
                failed = True
            finally:
                if process is not None and process.poll() is None:
                    stop_process(process)
                record['finished'] = time.time()
                write_json(pool/'results'/path.name, record)
                path.rename(pool/'finished'/path.name)
                write_json(status_path, record)
            idle_since = time.monotonic()
        write_json(status_path, dict(state='exited', time=time.time(), device=device, had_failures=failed))
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'enqueue', 'work'])
    parser.add_argument('--pool-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=sorted(variants()))
    parser.add_argument('--mode', choices=['profile-synthetic', 'train'], default='profile-synthetic')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--resume-from-checkpoint', type=Path)
    parser.add_argument('--expected-gpus', type=int, default=8)
    parser.add_argument('--idle-seconds', type=int, default=0, help='0 keeps workers available until STOP or deadline')
    parser.add_argument('--max-seconds', type=int, default=259200)
    parser.add_argument('--job-seconds', type=int, default=7200)
    parser.add_argument('--standby-load', action='store_true', help='Explicit maintenance load on otherwise idle GPUs')
    args = parser.parse_args()
    pool = args.pool_dir.resolve()
    if args.action == 'init':
        pool.mkdir(parents=True, exist_ok=False)
        for name in ('pending', 'running', 'finished', 'results', 'workers', 'logs', 'profiles', 'runs'):
            (pool/name).mkdir()
        write_json(pool/'pool.json', dict(created=time.time(), source_root=str(ROOT), schema_version=1))
        initial_variants = [name for name, options in variants().items()
                            if not options.get('architecture_v2') and not options.get('architecture_refined')
                            and not options.get('diagnostic_only')]
        for variant in initial_variants:
            enqueue(pool, dict(variant=variant, mode='profile-synthetic', steps=20))
        print(f'Initialized {pool}: {len(initial_variants)} legacy profile + synthetic jobs; no platform submission')
    elif args.action == 'enqueue':
        job = dict(variant=args.variant, mode=args.mode, steps=args.steps)
        for key in ('data_dir', 'resume_from_checkpoint'):
            if getattr(args, key):
                job[key] = str(getattr(args, key).resolve())
        print(enqueue(pool, job))
    else:
        if not (pool/'pool.json').is_file():
            raise ValueError('Initialize the pool on the development host before submission')
        if min(args.expected_gpus, args.max_seconds, args.job_seconds) <= 0 or args.idle_seconds < 0:
            raise ValueError('GPU count/time limits must be positive; idle limit may be zero')
        import torch
        count = torch.cuda.device_count()
        if count != args.expected_gpus:
            raise ValueError(f'Expected {args.expected_gpus} visible GPUs, got {count}')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        devices = [x.strip() for x in visible.split(',')] if visible else [str(x) for x in range(count)]
        if len(devices) != count or len(set(devices)) != count:
            raise ValueError('Ambiguous CUDA_VISIBLE_DEVICES mapping')
        stop = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.set())
        deadline = time.monotonic()+args.max_seconds
        telemetry = threading.Thread(target=gpu_telemetry, args=(pool, devices, stop), daemon=True)
        telemetry.start()
        try:
            with ThreadPoolExecutor(max_workers=count) as executor:
                futures = [executor.submit(worker, pool, device, stop, deadline, args.idle_seconds,
                                           args.job_seconds, args.standby_load) for device in devices]
                try:
                    codes = [future.result() for future in as_completed(futures)]
                except BaseException:
                    stop.set()
                    raise
        finally:
            stop.set()
            telemetry.join(timeout=12)
        raise SystemExit(max(codes))


if __name__ == '__main__':
    main()
