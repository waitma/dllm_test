"""Evaluate pinned BERT on the existing AB/TCR non-generation protocols.

Activate pllm/protenix_abtcr; python -B SCRIPT --run ABS --check --device cpu
or --group N on an allocated GPU. Each task keeps its original splits and heads.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
from downstream.bert_full_token_eval import load, BertBackbone, sha, check


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def report(run):
    with (run / 'report.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = json.loads((run / 'plan.json').read_text())
        states = {}
        for task in plan['tasks']:
            path = run / task / 'status.json'
            states[task] = json.loads(path.read_text()) if path.exists() else dict(status='pending')
        result = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                      model=plan['model'], step=plan['step'],
                      counts=dict(Counter(s['status'] for s in states.values())), tasks=states)
        write(run / 'summary.json', result)
        return result


def verify(run):
    plan = json.loads((run / 'plan.json').read_text())
    for section in ('code_sha256', 'data_sha256'):
        for path, expected in plan[section].items():
            if sha(path) != expected:
                raise ValueError('Pinned evaluation asset changed: ' + path)
    return plan


def worker(run, task):
    """Install the BERT adapter before importing any legacy model/task modules."""
    plan = verify(run)
    if task not in plan['tasks']:
        raise ValueError('Task outside the pinned representation suite')
    model, identity = load(run / 'checkpoint')
    if identity['model_sha256'] != plan['model_sha256']:
        raise ValueError('Wrong model identity')
    import torch
    torch.set_num_threads(4)
    from downstream import v5_native_eval as native
    native.load = lambda *args, **kwargs: (model, identity)
    native.NativeBackbone = BertBackbone
    from scripts.downstream import report_v5_suite, run_v5_suite_task as runner
    report_v5_suite.update = report
    original_install, original_probes = runner.install, runner.probes

    def install(backbone):
        binding = original_install(backbone)
        binding.OURS_METHOD = 'Ours BERT full-token V7 75.6M frozen observed-mean + MLP'
        return binding

    def probes(name, checkpoint, out, backbone):
        original_probes(name, checkpoint, out, backbone)
        manifest = json.loads((out / 'run_manifest.json').read_text())
        manifest['model'] = plan['model']
        write(out / 'run_manifest.json', manifest)
        metrics = json.loads((out / 'metrics.json').read_text())
        metrics['model'] = plan['model']
        metrics['run_manifest_sha256'] = sha(out / 'run_manifest.json')
        write(out / 'metrics.json', metrics)

    runner.install, runner.probes = install, probes
    sys.argv = [runner.__file__, '--run', str(run), '--task', task]
    runner.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--group', type=int)
    parser.add_argument('--task')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.report:
        print(json.dumps(report(args.run), ensure_ascii=False)); return
    plan = verify(args.run)
    if args.check:
        result = check(args.run / 'checkpoint', args.device)
        write(args.run / f'{args.device}_adapter_checks.json', result)
        print(json.dumps(result, indent=2)); return
    if args.task:
        worker(args.run, args.task); return
    if args.group is None:
        parser.error('Supply --check, --task, --group or --report')
    # Every allocated GPU checks the real model before launching a full group.
    result = check(args.run / 'checkpoint', 'cuda')
    write(args.run / f'gpu_gate_g{args.group}.json', result)
    results = []
    for task in plan['groups'][args.group]:
        if (args.run / 'STOP').exists():
            raise RuntimeError('Stop requested')
        folder = args.run / task
        folder.mkdir(exist_ok=True)
        status = folder / 'status.json'
        if status.exists():
            state = json.loads(status.read_text())
            results.append(dict(task=task, returncode=0 if state['status'] == 'success' else 1,
                                existing_attempt=True))
            continue
        with (folder / 'run.log').open('x') as stream:
            process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                      '--run', str(args.run), '--task', task],
                                     stdout=stream, stderr=subprocess.STDOUT)
        if not status.exists():
            write(status, dict(status='failed', task=task, returncode=process.returncode,
                               error='Worker failed before task startup; inspect run.log'))
        results.append(dict(task=task, returncode=process.returncode))
        write(args.run / f'group_{args.group}.json', dict(results=results))
        report(args.run)
    if any(item['returncode'] for item in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
