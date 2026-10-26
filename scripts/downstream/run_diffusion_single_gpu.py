"""Run the prepared diffusion suite serially on one allocated GPU.

Activate pllm/protenix_abtcr; python -B THIS_FILE --run ABS [--cpu-check|--verify-only].
The shell entrypoint sets the single visible GPU and local scoring-model paths.
Each task has a separate process/log; failed tasks yield a failing job exit code.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
from downstream.bert_full_token_eval import sha
from scripts.downstream.prepare_bert_checkpoint_suite import validate_snapshot
from scripts.downstream.run_bert_representation_suite import report, verify, write


def now():
    return datetime.now(timezone.utc).isoformat()


def check_assets(run):
    plan = verify(run)
    receipt = validate_snapshot(run / 'checkpoint')
    if receipt['step'] != plan['step'] or receipt['files']['model.safetensors'] != plan['model_sha256']:
        raise ValueError('Checkpoint identity differs from plan')
    provenance = json.loads((run / 'checkpoint/run_provenance.json').read_text())
    for path, expected in provenance['sources'].items():
        if sha(path) != expected:
            raise ValueError('Checkpoint-bound training source changed: ' + path)
    for name, path in plan['scoring_models'].items():
        if os.environ.get(f'IMMUNOMATCH_{name.upper()}_PATH') != path:
            raise ValueError('Use the shell entrypoint to set the local scoring models')
    return plan


def scoring_check(run, plan):
    import pandas as pd
    import torch
    from abnumber import Chain
    from transformers import RoFormerForSequenceClassification, RoFormerTokenizer
    torch.set_num_threads(4)
    row = pd.read_csv(plan['benchmark']['path'], keep_default_na=False).iloc[0]
    heavy, light = row.h_sequence.replace('-', ''), row.l_sequence.replace('-', '')
    assert Chain(heavy, scheme='imgt').chain_type == 'H'
    assert Chain(light, scheme='imgt').chain_type in ('K', 'L')
    for path in plan['scoring_models'].values():
        tokenizer = RoFormerTokenizer.from_pretrained(path, local_files_only=True)
        model = RoFormerForSequenceClassification.from_pretrained(path, local_files_only=True).eval()
        inputs = tokenizer(' '.join(heavy), ' '.join(light), return_tensors='pt',
                           padding='max_length', truncation=True, max_length=256)
        with torch.inference_mode():
            assert torch.isfinite(model(**inputs).logits).all()
        del model
    write(run / 'scoring_dependency_check.json', dict(status='passed', checked_utc=now(),
          anarci_heavy_light=True, immunomatch_cpu_forward=['kappa', 'lambda'],
          scope='reference pair dependency check; not downstream results'))


def metrics_report(run, plan):
    from scripts.downstream.publish_diffusion_edit_results import collect
    rows, t2, completed, artifacts = collect(run, plan)
    write(run / 'metrics_summary.json', dict(step=plan['step'], model_sha256=plan['model_sha256'],
          completed=completed, updated_utc=now(), artifacts_sha256=artifacts,
          tables=[dict(table=heading, model=label, artifact=str(path), values=values)
                  for heading, label, path, values in rows],
          t2=None if t2 is None else dict(model=t2[0], artifact=str(t2[1]), aligned_points=t2[2])))


def run_tasks(run, plan, launch):
    results = []
    for number, task in enumerate(plan['tasks'], 1):
        folder = run / task
        status_path = folder / 'status.json'
        if status_path.exists():
            state = json.loads(status_path.read_text())
            result = dict(task=task, status=state['status'], existing_attempt=True,
                          returncode=0 if state['status'] == 'success' else 1)
        else:
            print(f'[{now()}] {number}/{len(plan["tasks"])} START {task}; log={folder / "run.log"}', flush=True)
            returncode = launch(['--task', task], folder / 'run.log')
            state = json.loads(status_path.read_text()) if status_path.exists() else {}
            if returncode or state.get('status') != 'success':
                state.update(status='failed', task=task, returncode=returncode,
                             driver_error='Worker failed or did not record successful completion')
                write(status_path, state)
            result = dict(task=task, status=state['status'], returncode=returncode,
                          finished_utc=now())
        results.append(result)
        write(run / 'single_gpu/results.json', results)
        report(run)
        print(f'[{now()}] {task}: {result["status"]}', flush=True)
    metrics_report(run, plan)
    return all(item['status'] == 'success' and item['returncode'] == 0 for item in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--cpu-check', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    plan = check_assets(run)
    if args.cpu_check:
        with (run / 'cpu_checks.log').open('w') as stream:
            subprocess.run([sys.executable, '-B', str(run / 'run_suite.py'), '--run', str(run),
                            '--check', '--device', 'cpu'], stdout=stream, stderr=subprocess.STDOUT, check=True)
        scoring_check(run, plan)
        gates = ('cpu_adapter_checks.json', 'cpu_generation_checks.json', 'scoring_dependency_check.json')
        for filename in gates:
            if json.loads((run / filename).read_text())['status'] != 'passed':
                raise ValueError('CPU check failed: ' + filename)
        write(run / 'preparation.json', dict(status='cpu_passed_not_submitted', checked_utc=now(),
              plan_sha256=sha(run / 'plan.json'), tasks=plan['tasks'],
              gates_sha256={filename: sha(run / filename) for filename in gates}))
        report(run)
        print('CPU_CHECKS_PASSED', flush=True)
        return 0
    preparation = json.loads((run / 'preparation.json').read_text())
    if preparation['status'] != 'cpu_passed_not_submitted' or preparation['plan_sha256'] != sha(run / 'plan.json'):
        raise ValueError('Run the prepared driver with --cpu-check first')
    for filename, expected in preparation['gates_sha256'].items():
        if sha(run / filename) != expected:
            raise ValueError('CPU check receipt changed: ' + filename)
    if args.verify_only:
        print(json.dumps(dict(status='verified', step=plan['step'], tasks=plan['tasks'],
                              gpu_started=False, run=str(run))), flush=True)
        return 0
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU is required')
    for key, value in dict(TCR_EVAL_CKPT=str(run / 'checkpoint'), TCR_EVAL_TAG=run.name,
                           TCR_EVAL_STEP=str(plan['step']), TCR_EVAL_SHA=plan['model_sha256']).items():
        os.environ[key] = value
    control = run / 'single_gpu'
    child, stopping = None, False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state = dict(status='starting', started_utc=now(), pid=os.getpid(), host=socket.gethostname(),
                 checkpoint_step=plan['step'], gpu=torch.cuda.get_device_name(0), tasks=plan['tasks'])

    def launch(arguments, log_path):
        nonlocal child
        if stopping or (run / 'STOP').exists():
            raise InterruptedError('Stop requested')
        with log_path.open('a') as stream:
            child = subprocess.Popen([sys.executable, '-B', str(run / 'run_suite.py'),
                                      '--run', str(run), *arguments], stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', child_pid=child.pid, current=arguments, log=str(log_path))
            while child.poll() is None:
                if stopping or (run / 'STOP').exists():
                    raise InterruptedError('Stop requested')
                state['updated_utc'] = now()
                write(control / 'state.json', state)
                time.sleep(3)
            result, child = child.returncode, None
        return result

    with (control / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            print(f'[{now()}] CUDA checks on {state["gpu"]}; checkpoint={plan["step"]}', flush=True)
            if launch(['--check', '--device', 'cuda'], control / 'cuda_checks.log'):
                raise RuntimeError('CUDA checks failed; inspect single_gpu/cuda_checks.log')
            passed = run_tasks(run, plan, launch)
            state['status'] = 'success' if passed else 'failed'
            return 0 if passed else 1
        except BaseException as error:
            state.update(status='stopped' if stopping else 'failed', error=repr(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            state.update(finished_utc=now(), updated_utc=now())
            write(control / 'state.json', state)
            report(run)
            print(f'[{now()}] {state["status"]}; summary={run / "summary.json"}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
