"""Remove V7 test model files while preserving results and active temp writes.

Activate conda pllm. Run this absolute script with --pool-dir /absolute/pool
--record-dir /absolute/audit --once, or --watch to clean every ten seconds until
the current pool stops. Only p01/p02/p04 V7 test run outputs are eligible.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import time

RUN_PATTERN = re.compile(r'\d+_(?:p01|p02|p04)_v7_[A-Za-z0-9_]+_[0-9a-f]{8}')
MODEL_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors'}
MODEL_NAMES = {'pytorch_model.bin', 'optimizer.bin', 'scheduler.bin'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def active_ids(pool):
    return {path.stem for path in (pool / 'running').glob('*.json')}


def remove_file(path, reason, events):
    if path.is_symlink() or not path.is_file():
        return
    try:
        before = path.stat()
        path.unlink()
    except FileNotFoundError:
        return
    events.append({'path': str(path), 'bytes': before.st_size,
                   'reason': reason, 'deleted_at_utc': utc()})


def stale(path, now, seconds):
    try:
        return not path.is_symlink() and now - path.stat().st_mtime >= seconds
    except FileNotFoundError:
        return False


def clean_once(pool, *, stale_seconds=900):
    """Never unlink a temp file belonging to a currently running job."""
    events = []
    active = active_ids(pool)
    now = time.time()
    for run in (pool / 'runs').iterdir():
        if run.is_symlink() or not run.is_dir() or not RUN_PATTERN.fullmatch(run.name):
            continue
        result = read_json(pool / 'results' / (run.name + '.json'))
        terminal = result.get('state') in ('succeeded', 'failed')
        for directory, subdirs, names in os.walk(run, followlinks=False):
            subdirs[:] = [name for name in subdirs if not (Path(directory) / name).is_symlink()]
            for name in names:
                path = Path(directory) / name
                if path.suffix in MODEL_SUFFIXES or name in MODEL_NAMES:
                    remove_file(path, 'test_checkpoint_not_retained', events)
                elif name.endswith(('.tmp', '.temp', '.part', '.partial')) and run.name not in active:
                    if terminal or stale(path, now, stale_seconds):
                        remove_file(path, 'inactive_test_temporary', events)
    # A monitor uses atomic UUID temp names. Preserve recent writes.
    monitor = pool / 'monitor'
    if monitor.is_dir() and not monitor.is_symlink():
        for path in monitor.glob('*.tmp'):
            if stale(path, now, stale_seconds):
                remove_file(path, 'stale_monitor_temporary', events)
    return events


def pool_alive(pool):
    now = time.time()
    return any(now - read_json(path).get('heartbeat', 0) < 120
               for path in (pool / 'workers').glob('*.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--pool-dir', type=Path, required=True)
    parser.add_argument('--record-dir', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--once', action='store_true')
    modes.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--max-hours', type=float, default=24)
    args = parser.parse_args()
    if (not args.pool_dir.is_absolute() or args.pool_dir.is_symlink()
            or args.pool_dir != args.pool_dir.resolve() or (args.pool_dir / 'runs').is_symlink()
            or not (args.pool_dir / 'runs').is_dir() or not args.record_dir.is_absolute()
            or args.record_dir.is_symlink() or args.interval < 1 or args.max_hours <= 0):
        parser.error('Require canonical pool/audit paths and positive bounded timing')
    args.record_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.record_dir / 'cleanup.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('A cleanup process is already running for this audit directory')
    started = time.monotonic()
    journal = args.record_dir / 'deletions.jsonl'
    previous = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()] if journal.exists() else []
    count, total_bytes, idle = len(previous), sum(item['bytes'] for item in previous), 0
    state_path = args.record_dir / 'state.json'
    while True:
        events = clean_once(args.pool_dir)
        if events:
            with journal.open('a') as handle:
                for event in events:
                    handle.write(json.dumps(event, ensure_ascii=False) + '\n')
            count += len(events)
            total_bytes += sum(event['bytes'] for event in events)
        live = pool_alive(args.pool_dir)
        idle = 0 if live else idle + 1
        active = active_ids(args.pool_dir)
        pending = list((args.pool_dir / 'pending').glob('*.json'))
        done = args.once or idle >= 3 or (not active and not pending) or time.monotonic() - started >= args.max_hours * 3600
        state = {'schema': 'v7.test_artifact_retention.v1', 'updated_at_utc': utc(),
                 'pid': os.getpid(), 'status': 'completed' if done else 'watching',
                 'pool_dir': str(args.pool_dir), 'checkpoint_retention': 'none',
                 'active_temporary_writes_preserved': True, 'deleted_file_count': count,
                 'deleted_bytes': total_bytes, 'running_count': len(active), 'pending_count': len(pending),
                 'poll_interval_seconds': args.interval, 'max_hours': args.max_hours}
        temporary = state_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)
        if done:
            print(json.dumps(state), flush=True)
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
