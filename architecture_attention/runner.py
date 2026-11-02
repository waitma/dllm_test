"""Local serial factorial training: bash /absolute/path/launch_arch70m_attention.sh --gpu 0."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from .common import ROOT, OUT, VARIANTS, MICRO, GLOBAL, SEED, STEPS, DATA_SHA
from .common import read, write, sha, digest, stamp, lock, prepare, snapshot, source_binding
from .common import trial, valid_profile, qualified, completed, validate_output


def trainer():
    from experiments.arch70m import train
    from .models import build
    train.build = build; train.source_binding = source_binding
    return train


def profile_one(args):
    from architecture_hierarchy import runner as memory
    # Reuse the unchanged three-update, maximum-length memory qualification.
    memory.trainer = trainer; memory.source_binding = source_binding; memory.valid_profile = valid_profile
    memory.profile_one(argparse.Namespace(output_root=args.output_root, profile_one=args.profile_one))


def publish(out, message, push):
    from .report import report
    from arch70m_seed_validation import push_results
    def git(*args): return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    with lock(ROOT / 'output/arch70m_publish.lock'):
        if git('branch', '--show-current') != 'explore/arch70m-v8-20261028': raise ValueError('Wrong publication branch.')
        if git('diff', '--cached', '--name-only'): raise ValueError('Unrelated staged changes.')
        done = report(out)
        p = ROOT / 'PROJECT_PROCESS.md'
        p.write_text(f'> Last updated: {stamp()} (UTC)。全局注意力2×2探索及原始对照{done}/5完成；{message}；'
            f'C0=A5、C1=QK归一化、C2=单位初始门控、C3=组合，seed43/1epoch/micro32/global512；'
            f'原V8固定10%保持，只读screen，不作稳定提升结论；报告 {ROOT}/docs/arch70m_results/attention_20261102/REPORT.md。\n\n' + p.read_text())
        git('add', '--', 'PROJECT_PROCESS.md', 'docs/arch70m_results/attention_20261102')
        git('diff', '--cached', '--check')
        git('-c', 'user.name=Codex', '-c', 'user.email=codex@openai.com', 'commit', '-m', 'Attention factorial: ' + message)
        result = push_results() if push else dict(status='disabled')
        write(out / 'publication.json', dict(result, updated=stamp(), commit=git('rev-parse', 'HEAD')))
        return result


def child(args, source, variant, profile=False):
    t = trial(args.output_root, variant); t.mkdir(parents=True, exist_ok=True)
    blocked = {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','MASTER_ADDR','MASTER_PORT',
               'ARCH70M_FROZEN_SOURCE','CUDA_LAUNCH_BLOCKING'}
    env = {k: v for k, v in os.environ.items() if k not in blocked and not k.startswith(('NCCL_', 'TORCHELASTIC_'))}
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONPATH=str(source), USE_TF='0',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    mode = '--profile-one' if profile else '--train-one'
    command = [sys.executable, '-B', '-m', 'architecture_attention.runner', mode, variant, '--output-root', str(args.output_root)]
    log_path = t / ('qualification.log' if profile else 'console.log')
    with log_path.open('a') as log:
        start = log.tell()
        process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(t / ('qualification_process.json' if profile else 'process.json'), dict(pid=process.pid,
              command=command, started=stamp(), source=str(source), gpu=args.gpu, log=str(log_path), log_start=start))
        print(dict(event='launch', mode=mode, variant=variant, pid=process.pid), flush=True)
        def stop(_number, _frame):
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            raise KeyboardInterrupt('Factorial exploration cancelled.')
        old = {n: signal.signal(n, stop) for n in (signal.SIGINT, signal.SIGTERM)}
        try: code = process.wait()
        finally:
            for n, handler in old.items(): signal.signal(n, handler)
    if code:
        with log_path.open('rb') as log:
            log.seek(0, 2); size = log.tell(); log.seek(max(start, size-5000)); error = log.read().decode(errors='replace')
        status = read(t / 'status.json') if (t / 'status.json').exists() else dict(step=0)
        write(t / 'status.json', dict(status, status='failed', exit_code=code, mode=mode, error=error, updated=stamp()))
        print(dict(event='failed', variant=variant, exit_code=code, error=error[-900:]), flush=True)
    return code


def execute(args):
    if not args.gpu.isdecimal(): raise ValueError('Use a numeric CUDA selector.')
    if args.dry_run:
        print(dict(variants=VARIANTS, seed=SEED, epochs=1, steps=STEPS, micro_batch=MICRO,
                   global_batch=GLOBAL, gpu=args.gpu, output_root=str(args.output_root))); return
    validate_output(args.output_root)
    with lock(ROOT / 'output' / f'arch70m_local_gpu{args.gpu}.lock', blocking=False), lock(args.output_root / 'controller.lock', blocking=False):
        try:
            prepare(args.output_root)
            from experiments.arch70m.suite import verify_data
            verify_data(args.output_root)
            source, signature = snapshot(args.output_root)
            execution = dict(started=stamp(), controller_pid=os.getpid(), variants=list(VARIANTS),
                seed=SEED, epochs=1, total_steps=STEPS, micro_batch=MICRO, global_batch=GLOBAL,
                source_sha256=signature, data_sha256=DATA_SHA, gpu=args.gpu)
            write(args.output_root / 'execution.json', dict(execution, phase='qualification'))
            for v in VARIANTS:
                p = trial(args.output_root, v) / 'profile.json'
                if p.exists(): valid_profile(read(p), v, signature)
                elif child(args, source, v, True): raise RuntimeError('All five must qualify before training: ' + v)
            write(args.output_root / 'qualification.json', dict(status='passed', updated=stamp(), micro_batch=MICRO,
                  source_sha256=signature, data_sha256=DATA_SHA,
                  profiles={v: sha(trial(args.output_root, v) / 'profile.json') for v in VARIANTS}))
            qualified(args.output_root, signature)
            print(dict(event='all_five_qualified', source_sha256=signature), flush=True)
            write(args.output_root / 'execution.json', dict(execution, phase='training', qualified=stamp()))
            failures = []
            for v in VARIANTS:
                if completed(args.output_root, v, signature): continue
                code = child(args, source, v)
                if code == 0 and not completed(args.output_root, v, signature): raise ValueError('Missing complete final receipt: ' + v)
                if code: failures.append(v)
                try: publication = publish(args.output_root, v + (' failed' if code else ' completed'), not args.no_push)
                except Exception:
                    publication = dict(status='pending', error=traceback.format_exc())
                    write(args.output_root / 'publication.json', dict(publication, updated=stamp()))
                print(dict(event='finished', variant=v, training_exit_code=code, publication_status=publication['status']), flush=True)
            write(args.output_root / 'execution.json', dict(execution, phase='finished', finished=stamp(), failed_variants=failures,
                  stable_improvement_assessed=False, note='Single-seed screening only; no stable claim or automatic confirm.'))
            if failures: raise RuntimeError('Incomplete factorial trials: ' + ','.join(failures))
        except BaseException as exc:
            error = traceback.format_exc()
            write(args.output_root / 'controller_error.json', dict(updated=stamp(), error=error))
            path = args.output_root / 'execution.json'
            state = read(path) if path.exists() else {}
            write(path, dict(state, phase='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed', updated=stamp(), error=error))
            try: publish(args.output_root, 'controller ' + ('cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed'), not args.no_push)
            except Exception: write(args.output_root / 'publication.json', dict(status='pending', error=traceback.format_exc(), updated=stamp()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0'); parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--dry-run', action='store_true'); parser.add_argument('--no-push', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--profile-one', choices=VARIANTS); mode.add_argument('--train-one', choices=VARIANTS)
    args = parser.parse_args(); args.output_root = args.output_root.resolve()
    if args.profile_one: profile_one(args)
    elif args.train_one:
        qualified(args.output_root, digest(source_binding()))
        trainer().run(argparse.Namespace(output_root=str(args.output_root), stage='screen', variant=args.train_one, seed=SEED, wandb=False))
    else: execute(args)


if __name__ == '__main__': main()
