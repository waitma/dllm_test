"""Queue one additional seed locally: bash /absolute/path/launch_arch70m_seed42.sh --gpu 0."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from .common import ROOT, OLD, OUT, SEED, VARIANTS, STEPS, MICRO, GLOBAL, PREVIOUS_SHA, DATA_SHA
from .common import read, write, sha, digest, stamp, lock, prepare, snapshot, source_binding, trial
from .common import valid_profile, qualified, completed, predecessor_ready, validate_output


def trainer():
    from experiments.arch70m import train
    from architecture_attention.models import build
    train.build = build; train.source_binding = source_binding
    return train


def profile_one(args):
    from architecture_hierarchy import runner as memory
    memory.SEED = SEED; memory.STEPS = STEPS; memory.MICRO = MICRO; memory.GLOBAL = GLOBAL
    memory.trainer = trainer; memory.source_binding = source_binding; memory.valid_profile = valid_profile
    memory.profile_one(argparse.Namespace(output_root=args.output_root, profile_one=args.profile_one))


def command(source, out, variant, profile=False):
    mode = '--profile-one' if profile else '--train-one'
    return [sys.executable, '-B', '-m', 'architecture_seed_repeat.runner', mode, variant, '--output-root', str(out)]


def child(args, source, variant, profile=False):
    t = trial(args.output_root, variant); t.mkdir(parents=True, exist_ok=True)
    blocked = {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','MASTER_ADDR','MASTER_PORT',
               'ARCH70M_FROZEN_SOURCE','CUDA_LAUNCH_BLOCKING'}
    env = {k: v for k, v in os.environ.items() if k not in blocked and not k.startswith(('NCCL_', 'TORCHELASTIC_'))}
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONPATH=str(source), USE_TF='0',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    path = t / ('qualification.log' if profile else 'console.log')
    with path.open('a') as log:
        start = log.tell(); cmd = command(source, args.output_root, variant, profile)
        process = subprocess.Popen(cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(t / ('qualification_process.json' if profile else 'process.json'), dict(pid=process.pid, command=cmd,
              started=stamp(), source=str(source), gpu=args.gpu, seed=SEED, log=str(path), log_start=start))
        print(dict(event='launch', seed=SEED, variant=variant, profile=profile, pid=process.pid), flush=True)
        def stop(_number, _frame):
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            raise KeyboardInterrupt('Seed42 queue cancelled.')
        old = {n: signal.signal(n, stop) for n in (signal.SIGINT, signal.SIGTERM)}
        try: code = process.wait()
        finally:
            for n, handler in old.items(): signal.signal(n, handler)
    if code:
        with path.open('rb') as log:
            log.seek(0, 2); size = log.tell(); log.seek(max(start, size-5000)); error = log.read().decode(errors='replace')
        state = read(t / 'status.json') if (t / 'status.json').exists() else dict(step=0)
        write(t / 'status.json', dict(state, status='failed', exit_code=code, failed_profile=profile, error=error, updated=stamp()))
    return code


def publish(out, message, push):
    from .report import report
    from arch70m_seed_validation import push_results
    def git(*args): return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    with lock(ROOT / 'output/arch70m_publish.lock'):
        if git('branch','--show-current') != 'explore/arch70m-v8-20261028': raise ValueError('Wrong branch.')
        if git('diff','--cached','--name-only'): raise ValueError('Unrelated staged files.')
        done = report(out)
        p = ROOT / 'PROJECT_PROCESS.md'
        p.write_text(f'> Last updated: {stamp()} (UTC)。新增seed42配对复核{done}/5完成；{message}；'
            f'C0/C1/C2/C3/A0、1epoch/1276步/micro32/global512，原V8固定10%保持，'
            f'与seed43双种子比较，不作稳定显著性结论；报告 {ROOT}/docs/arch70m_results/attention_seed42_20261103/REPORT.md。\n\n' + p.read_text())
        git('add','--','PROJECT_PROCESS.md','docs/arch70m_results/attention_seed42_20261103')
        git('diff','--cached','--check')
        git('-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Attention seed42: ' + message)
        result = push_results() if push else dict(status='disabled')
        write(out / 'publication.json', dict(result, updated=stamp(), commit=git('rev-parse','HEAD')))
        return result


def safe_publish(args, message):
    try: return publish(args.output_root, message, not args.no_push)
    except Exception:
        result = dict(status='pending', error=traceback.format_exc(), updated=stamp())
        write(args.output_root / 'publication.json', result); return result


def execute(args):
    if not args.gpu.isdecimal(): raise ValueError('Use one numeric local GPU selector.')
    if args.dry_run:
        print(dict(seed=SEED, reference_seed=43, variants=VARIANTS, epochs=1, steps=STEPS,
                   micro_batch=MICRO, global_batch=GLOBAL, gpu=args.gpu, waits_for=str(OLD))); return
    validate_output(args.output_root)
    with lock(args.output_root / 'controller.lock', blocking=False):
        try:
            prepare(args.output_root)
            from experiments.arch70m.suite import verify_data
            verify_data(args.output_root)
            source, signature = snapshot(args.output_root)
            execution = dict(started=stamp(), controller_pid=os.getpid(), seed=SEED, reference_seed=43,
                variants=list(VARIANTS), epochs=1, total_steps=STEPS, micro_batch=MICRO, global_batch=GLOBAL,
                source_sha256=signature, previous_source_sha256=PREVIOUS_SHA, data_sha256=DATA_SHA, gpu=args.gpu,
                predecessor_output=str(OLD))
            write(args.output_root / 'execution.json', dict(execution, phase='waiting_for_gpu'))
            print(dict(event='queued', seed=SEED, waits_for=str(OLD)), flush=True)
            # The existing seed43 controller owns this lease until its final
            # evaluation and publication finish. No GPU allocation while waiting.
            with lock(ROOT / 'output' / f'arch70m_local_gpu{args.gpu}.lock', blocking=True):
                predecessor_ready()
                write(args.output_root / 'execution.json', dict(execution, phase='qualification', acquired_gpu=stamp()))
                for v in VARIANTS:
                    p = trial(args.output_root, v) / 'profile.json'
                    if p.exists(): valid_profile(read(p), v, signature)
                    elif child(args, source, v, True): raise RuntimeError('All five seed42 models must qualify: ' + v)
                write(args.output_root / 'qualification.json', dict(status='passed', seed=SEED, source_sha256=signature,
                    data_sha256=DATA_SHA, updated=stamp(), profiles={v:sha(trial(args.output_root,v)/'profile.json') for v in VARIANTS}))
                qualified(args.output_root, signature)
                write(args.output_root / 'execution.json', dict(execution, phase='training', qualified=stamp()))
                print(dict(event='all_five_seed42_qualified'), flush=True)
                safe_publish(args, 'all five qualified; training starts')
                failures = []
                for v in VARIANTS:
                    if completed(args.output_root, v, signature): continue
                    code = child(args, source, v)
                    if code == 0 and not completed(args.output_root,v,signature): raise ValueError('Missing final seed42 receipt: ' + v)
                    if code: failures.append(v)
                    result = safe_publish(args, v + (' failed' if code else ' completed'))
                    print(dict(event='finished', variant=v, seed=SEED, training_exit_code=code, publication_status=result['status']), flush=True)
                write(args.output_root / 'execution.json', dict(execution, phase='finished', finished=stamp(),
                      failed_variants=failures, stable_improvement_assessed=False))
                if failures: raise RuntimeError('Incomplete seed42 trials: ' + ','.join(failures))
        except BaseException as exc:
            path = args.output_root / 'execution.json'; state = read(path) if path.exists() else {}
            write(path, dict(state, phase='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed', updated=stamp(), error=traceback.format_exc()))
            safe_publish(args, 'controller stopped'); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0'); parser.add_argument('--output-root',type=Path,default=OUT)
    parser.add_argument('--dry-run',action='store_true'); parser.add_argument('--no-push',action='store_true')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--profile-one',choices=VARIANTS); mode.add_argument('--train-one',choices=VARIANTS)
    args=parser.parse_args();args.output_root=args.output_root.resolve()
    if args.profile_one: profile_one(args)
    elif args.train_one:
        qualified(args.output_root,digest(source_binding()))
        trainer().run(argparse.Namespace(output_root=str(args.output_root),stage='screen',variant=args.train_one,seed=SEED,wandb=False))
    else:
        def stop(_number, _frame): raise KeyboardInterrupt('Seed42 queue cancelled while waiting.')
        signal.signal(signal.SIGTERM, stop)
        execute(args)


if __name__=='__main__':main()
