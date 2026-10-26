"""Run corrected AB–Ag jobs: python SCRIPT --run ABS --group NAME.

GPU workers must run in a Volc eval job. CPU: --run ABS --report.
Only tasks in the immutable plan run; existing attempts are never overwritten.
"""
import argparse
import fcntl
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _report(run):
    from downstream.benchmark.ab_antigen.abbibench_scope import summarize_run
    plan = json.loads((run / 'plan.json').read_text())
    states = {}
    for task in plan['tasks']:
        path = run / task / 'status.json'
        states[task] = json.loads(path.read_text()) if path.exists() else dict(status='pending')
    result = dict(updated_at=stamp(), model=plan['model'],
                  counts=dict(Counter(s['status'] for s in states.values())), tasks=states,
                  abbibench_benchmark16=summarize_run(run))
    write(run / 'summary.json', result)
    return result


def report(run):
    with (run / 'report.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _report(run)


def worker(run, task):
    plan = json.loads((run / 'plan.json').read_text())
    if task not in plan['gpu_tasks']:
        raise ValueError('Task not authorized in GPU plan: ' + task)
    for path, expected in plan['code_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Evaluation source changed: ' + path)
    for path, expected in plan['data_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Evaluation data changed: ' + path)
    out = run / task
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'status.json').open('x') as f:
        json.dump(dict(status='running', started_at=stamp(), task=task), f)
    try:
        # Model code must be loaded before importing any dllm module.
        from downstream.v5_native_eval import load, NativeBackbone
        model, identity = load(run / 'checkpoint')
        if identity['model_sha256'] != plan['model_sha256']:
            raise ValueError('Wrong checkpoint identity')
        source = identity['source_root']
        if source in sys.path:
            sys.path.remove(source)
        write(out / 'model_identity.json', identity)
        import torch
        torch.set_num_threads(4)
        if task == 'ag-chimera':
            from downstream.benchmark.ab_antigen.chimera_h3 import run as evaluate
            evaluate(out, model, identity, scheme='imgt', candidates=1, steps=32, temperature=1.)
        else:
            from downstream.v5_native_generation_eval import antigen
            antigen(task.removeprefix('ag-'), out, model, NativeBackbone(model, identity), identity)
        write(out / 'status.json', dict(status='success', task=task,
                                       execution='fresh_model_inference', finished_at=stamp()))
    except Exception as exc:
        write(out / 'status.json', dict(status='failed', task=task, error=repr(exc),
                                       traceback=traceback.format_exc(), finished_at=stamp()))
        raise
    finally:
        report(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--group')
    parser.add_argument('--task')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.report:
        print(json.dumps(report(args.run), ensure_ascii=False)); return
    if args.task:
        worker(args.run, args.task); return
    if not args.group:
        parser.error('Supply --group, --task or --report')
    plan = json.loads((args.run / 'plan.json').read_text())
    failed = []
    for task in plan['groups'][args.group]:
        out = args.run / task
        out.mkdir(parents=True, exist_ok=True)
        status = out / 'status.json'
        if status.exists():
            state = json.loads(status.read_text())
            if state['status'] != 'success':
                failed.append(task)
            continue
        with (out / 'run.log').open('x') as log:
            proc = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                   '--run', str(args.run), '--task', task],
                                  stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            failed.append(task)
            if not status.exists():
                write(status, dict(status='failed', task=task, returncode=proc.returncode,
                                   error='Worker failed before inference; inspect run.log', finished_at=stamp()))
        report(args.run)
    write(args.run / ('group_' + args.group + '.json'), dict(finished_at=stamp(), failed=failed))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
