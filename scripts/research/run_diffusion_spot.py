"""Resume the frozen diffusion run on eight GPUs.

Activate pllm, then use the pinned interpreter and absolute --control path from
the training YAML. --verify-only prepares and hashes a resume bundle without GPUs.
Each reclaimed-instance retry selects the newest complete checkpoint again.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def latest_checkpoint(output, fallback, minimum_step):
    candidates = [Path(fallback)] + list(Path(output).glob('checkpoint-*'))
    complete = []
    for path in candidates:
        if not re.fullmatch(r'checkpoint-\d+', path.name):
            continue
        receipt_path = path / 'checkpoint_receipt.json'
        if not receipt_path.is_file():
            continue
        receipt = read(receipt_path)
        if receipt.get('status') != 'complete':
            continue
        step = int(path.name.split('-')[-1])
        if receipt['step'] != step:
            raise ValueError('Checkpoint step mismatch: ' + str(path))
        if step >= minimum_step:
            complete.append((step, path))
    if not complete:
        raise ValueError('No complete checkpoint at or beyond the authorized start')
    return max(complete, key=lambda item: item[0])[1]


def prepare(control, spec):
    control = Path(control).resolve()
    original = Path(spec['original_bundle']).resolve()
    if sha(original / 'plan.json') != spec['original_plan_sha256']:
        raise ValueError('Original plan changed')
    transport = Path(spec['transport']).resolve()
    if sha(transport) != spec['transport_sha256']:
        raise ValueError('Resume transport changed')
    plan = read(original / 'plan.json')
    output = Path(plan['output_dir']).resolve()
    if (control / 'STOP').exists() or (output / 'STOP').exists():
        raise ValueError('User STOP exists; refusing to restart')
    checkpoint = latest_checkpoint(output, spec['fallback_checkpoint'], spec['minimum_step']).resolve()
    receipt = read(checkpoint / 'checkpoint_receipt.json')
    if receipt['step'] >= plan['training']['max_steps']:
        raise ValueError('Target already reached; no further training authorized')
    attempt = control / 'attempts' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    launcher = attempt / 'source/scripts/research/launch_v5_direct.py'
    launcher.parent.mkdir(parents=True)
    shutil.copy2(transport, launcher)
    preemptible = spec.get('preemptible', True)
    plan.update(stage='preemptible_resume' if preemptible else 'nonpreemptible_resume',
                created_utc=now(), resume_checkpoint=str(checkpoint),
                resume_contract=dict(original_bundle=str(original),
                    original_plan_sha256=spec['original_plan_sha256'],
                    checkpoint_step=receipt['step'],
                    checkpoint_receipt_sha256=sha(checkpoint / 'checkpoint_receipt.json'),
                    authorization='用户授权续训当前BERT到diffusion模型并持续监督；Preemptible=' + str(preemptible).lower()))
    write(attempt / 'plan.json', plan)
    write(attempt / 'direct_launch.json', dict(plan_sha256=sha(attempt / 'plan.json'),
          launcher_sha256=sha(launcher), runtime_dir=str(attempt)))
    module_spec = importlib.util.spec_from_file_location('frozen_diffusion_resume_transport', launcher)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.verify(attempt, checkpoint_files=True)
    result = dict(status='verified', checked_utc=now(), bundle=str(attempt),
                  checkpoint=str(checkpoint), starting_step=receipt['step'],
                  output_dir=str(output), source_root=str(Path(plan['source_root']).resolve()),
                  target_steps=plan['training']['max_steps'],
                  checkpoint_files_verified=len(receipt['files']),
                  training_computation_retained=True, optimizer_reset=False,
                  source_files_verified=len(plan['sources']),
                  vendor_files_verified=len(plan['vendor_files']))
    write(attempt / 'verification.json', result)
    write(control / 'preparation.json', result)
    return attempt, output, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    control = args.control.resolve()
    spec = read(control / 'spec.json')
    output = Path(read(Path(spec['original_bundle']) / 'plan.json')['output_dir']).resolve()
    with (output / 'diffusion_spot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.verify_only:
            if (output / 'COMPLETE.json').exists():
                done = read(output / 'COMPLETE.json')
                if done.get('status') == 'success' and done.get('step', 0) >= 200000:
                    write(control / 'runtime.json', dict(status='already_complete', updated_utc=now()))
                    return
            state = read(output / 'status.json')
            if state.get('status') == 'training' and time.time() - state.get('time_unix', 0) < 300:
                raise RuntimeError('Fresh training heartbeat exists; refusing a duplicate writer')
        attempt, output, result = prepare(control, spec)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.verify_only:
            launcher = attempt / 'source/scripts/research/launch_v5_direct.py'
            subprocess.run([sys.executable, '-I', '-B', '-u', str(launcher),
                            '--bundle', str(attempt), '--verify-only'], check=True)
            write(control / 'data_preflight.json', read(attempt / 'data_preflight.json'))
            return
        # Interrupted atomic saves must not block the frozen trainer's mkdir.
        # Keep these partial artifacts for inspection instead of deleting them.
        for partial in output.glob('.checkpoint-*.tmp'):
            quarantine = attempt / 'interrupted_saves'
            quarantine.mkdir(exist_ok=True)
            partial.rename(quarantine / partial.name)
        launcher = attempt / 'source/scripts/research/launch_v5_direct.py'
        argv = [sys.executable, '-I', '-B', '-u', '-m', 'torch.distributed.run',
                '--standalone', '--nnodes=1', '--nproc_per_node=8', '--max_restarts=0',
                str(launcher), '--bundle', str(attempt)]
        started = time.time()
        with (attempt / 'torchrun.log').open('a') as log:
            child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace', bufsize=1,
                                     start_new_session=True)
            def forward(signum, frame):
                if child.poll() is None:
                    os.killpg(child.pid, signum)
            signal.signal(signal.SIGTERM, forward)
            signal.signal(signal.SIGINT, forward)
            write(control / 'runtime.json', dict(status='running', **{k:v for k,v in result.items() if k != 'status'},
                started_unix=started, updated_utc=now(), host=os.uname().nodename,
                supervisor_pid=os.getpid(), torchrun_pid=child.pid, log=str(attempt / 'torchrun.log')))
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = child.wait()
        completion = read(output / 'COMPLETE.json') if (output / 'COMPLETE.json').exists() else {}
        success = code == 0 and completion.get('status') == 'success' and completion.get('step') == result['target_steps']
        stopped = code == 0 and (output / 'STOPPED.json').exists()
        write(control / 'runtime.json', dict(status='success' if success else 'stopped' if stopped else 'failed',
            exit_code=code, updated_utc=now(), started_unix=started, bundle=str(attempt),
            starting_step=result['starting_step'], completion=completion, log=str(attempt / 'torchrun.log')))
        if not success and not stopped:
            raise SystemExit(code if code > 0 else 1)


if __name__ == '__main__':
    main()
