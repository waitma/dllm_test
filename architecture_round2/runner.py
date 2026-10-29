"""Local serial execution; bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_round2.sh --gpu 0."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .common import ROOT, BASE_OUT, DEFAULT_OUT, VARIANTS, SEED, DATA_SHA, BASE_SOURCE_SHA
from .common import read, write, sha, digest, stamp, lock, prepare, snapshot, source_binding


def trainer():
    """Adapt only this process; the original frozen trainer file is never edited."""
    from experiments.arch70m import train
    from .models import build
    train.build = build
    train.source_binding = source_binding
    return train


def profile_worker(args):
    import torch
    train = trainer(); t = args.output_root/'trials'/f'screen_{args.worker}_s{SEED}'
    t.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    if torch.cuda.device_count() != 1: raise RuntimeError('Round-two worker needs exactly one CUDA GPU.')
    test = torch.ones(1, device='cuda:0')+1; torch.cuda.synchronize()
    if test.item() != 2: raise RuntimeError('CUDA kernel check failed.')
    write(t/'cuda_device.json', dict(updated=stamp(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
          device=torch.cuda.get_device_name(), cuda_uuid=str(torch.cuda.get_device_properties(0).uuid)))
    path = t/'profile.json'
    if path.exists():
        profile = read(path)
        if (profile['micro_batch'] != 56 or profile.get('source_sha256') != digest(source_binding()) or
                profile.get('variant') != args.worker): raise ValueError('Round-two profile binding differs.')
    else:
        with lock(t/'trial.lock', blocking=False):
            write(t/'status.json', dict(status='profiling', step=0, total_steps=1276, pid=os.getpid(), updated=stamp()))
            settings = read(ROOT/'experiments/arch70m/base.json'); train.setup(SEED)
            data = train.IndexedPreparedDataset(args.output_root/'data/index.json')
            collate = train.VFamilyCollator(train.FixedChainConfigProxy(settings), family_spec=settings['family_prediction'])
            clean = collate([data[i] for i in range(min(512, len(data)))])
            model, optimizer = train.construct(settings, args.worker, SEED, torch.device('cuda:0'))
            scheduler = train.make_scheduler(optimizer, 100, 5)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            batch = train.profile_batch(clean, 56, settings['model']['max_length'], model.tokenizer)
            batch = train.corrupt(batch, model.tokenizer, torch.Generator().manual_seed(9100))
            start = time.monotonic()
            for _ in range(3): train.update(model, batch, optimizer, scheduler, 56, torch.device('cuda:0'))
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated(); reserved = torch.cuda.max_memory_reserved()
            free, total = torch.cuda.mem_get_info(); external = max(0, total-free-torch.cuda.memory_reserved())
            evidence = dict(micro_batch=56, status='passed', allocated_bytes=allocated, reserved_bytes=reserved,
                external_bytes=external, total_bytes=total, safe=reserved+external <= .85*total,
                seconds_per_step=(time.monotonic()-start)/3)
            write(t/'profile_attempts.json', dict(variant=args.worker, candidates=[evidence]))
            if not evidence['safe']: raise RuntimeError('Common micro56 exceeds the 85% memory gate; no formal training started.')
            write(path, dict(variant=args.worker, micro_batch=56, grad_accum=10, final_micro_batch=8,
                length=2048, candidates=[evidence], source_sha256=digest(source_binding()), data_sha256=DATA_SHA,
                device=torch.cuda.get_device_name(), note='3 real updates at maximum length; fresh exec before formal training.'))
    command = [sys.executable, '-B', '-m', 'architecture_round2.runner', '--train-one', args.worker,
               '--output-root', str(args.output_root)]
    os.execv(sys.executable, command)


def completed(out, variant, signature):
    t = out/'trials'/f'screen_{variant}_s{SEED}'
    if not (t/'status.json').exists() or read(t/'status.json')['status'] != 'completed': return False
    binding = read(t/'binding.json')
    expected = dict(variant=variant, seed=SEED, epochs=1, total_steps=1276, micro_batch=56,
                    source_sha256=signature, data_sha256=DATA_SHA)
    if any(binding.get(k) != v for k,v in expected.items()): raise ValueError('Completed round-two binding differs: '+variant)
    cp = t/'checkpoints/step_001276'; receipt = read(cp/'receipt.json')
    if receipt['status'] != 'complete' or len(receipt['files']) != 6: raise ValueError('Incomplete checkpoint: '+variant)
    if any(sha(cp/n) != h for n,h in receipt['files'].items()): raise ValueError('Checkpoint checksum differs: '+variant)
    return True


def publish(out, message, push):
    from .report import report
    def git(*args): return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    with lock(ROOT/'output/arch70m_publish.lock'):
        if git('branch', '--show-current') != 'explore/arch70m-v8-20261028': raise ValueError('Wrong publication branch.')
        if git('diff', '--cached', '--name-only'): raise ValueError('Unrelated staged changes; publication postponed.')
        result = report(out)
        p = ROOT/'PROJECT_PROCESS.md'
        p.write_text(f'> Last updated: {stamp()} (UTC)。本地第二轮架构探索 {result}/3 完成；{message}。仅seed43、每组1epoch；报告 {ROOT}/docs/arch70m_results/round2/REPORT.md。\n\n'+p.read_text())
        git('add', '--', 'PROJECT_PROCESS.md', 'docs/arch70m_results/round2'); git('diff', '--cached', '--check')
        git('-c', 'user.name=Codex', '-c', 'user.email=codex@openai.com', 'commit', '-m', 'arch70m round2: '+message)
        from arch70m_seed_validation import push_results
        publication = push_results() if push else dict(status='disabled')
        write(out/'publication.json', dict(publication, commit=git('rev-parse', 'HEAD'), updated=stamp()))
        return publication


def execute(args):
    variants = args.trials.split(',')
    if len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS): raise ValueError('Invalid round-two trials.')
    if not args.gpu.isdecimal(): raise ValueError('Use a numeric local GPU selector.')
    if args.dry_run:
        print(dict(variants=variants, gpu=args.gpu, seed=SEED, micro_batch=56, global_batch=512, epochs=1))
        return
    settings = prepare(args.output_root)
    from experiments.arch70m.suite import verify_data
    verify_data(args.output_root)
    with lock(args.output_root/'controller.lock', blocking=False):
        source, signature = snapshot(args.output_root)
        write(args.output_root/'execution.json', dict(started=stamp(), controller_pid=os.getpid(), variants=variants,
            seed=SEED, epochs=1, micro_batch=56, global_batch=512, source_sha256=signature,
            base_source_sha256=BASE_SOURCE_SHA, data_sha256=DATA_SHA, gpu=args.gpu,
            references=[str(BASE_OUT/'trials'/f'screen_{v}_s43') for v in ('A0', 'A4', 'A5')]))
        failures = []
        for variant in variants:
            if completed(args.output_root, variant, signature): continue
            t = args.output_root/'trials'/f'screen_{variant}_s{SEED}'; t.mkdir(parents=True, exist_ok=True)
            env = {k:v for k,v in os.environ.items() if k not in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE',
                'GROUP_RANK','MASTER_ADDR','MASTER_PORT') and not k.startswith(('NCCL_', 'TORCHELASTIC_'))}
            env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONPATH=str(source),
                CUBLAS_WORKSPACE_CONFIG=':4096:8', USE_TF='0', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4',
                PYTHONDONTWRITEBYTECODE='1')
            env.pop('CUDA_LAUNCH_BLOCKING', None); env.pop('ARCH70M_FROZEN_SOURCE', None)
            command = [sys.executable, '-B', '-m', 'architecture_round2.runner', '--worker', variant,
                       '--output-root', str(args.output_root)]
            print(dict(event='launch', variant=variant, seed=SEED, gpu=args.gpu, source=str(source)), flush=True)
            with (t/'console.log').open('a') as log:
                start = log.tell()
                process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                write(t/'process.json', dict(pid=process.pid, command=command, gpu=args.gpu, started=stamp(), log_start=start))
                code = process.wait()
            if code == 0 and not completed(args.output_root, variant, signature): code = 1
            if code:
                failures.append(variant)
                status = read(t/'status.json') if (t/'status.json').exists() else dict(step=0)
                with (t/'console.log').open('rb') as log:
                    log.seek(0,2); size=log.tell(); log.seek(max(start,size-5000)); error=log.read().decode(errors='replace')
                write(t/'status.json', dict(status,status='failed',error=error,training_exit_code=code,updated=stamp()))
                print(dict(event='training_failed', variant=variant, error=error[-1000:]), flush=True)
            try: publication=publish(args.output_root, variant+(' failed' if code else ' completed'), not args.no_push)
            except Exception:
                publication=dict(status='local_report_failed')
                write(t/'publication_error.json', dict(error=traceback.format_exc(),updated=stamp()))
            print(dict(event='finished',variant=variant,training_exit_code=code,publication=publication['status']),flush=True)
        if failures: raise RuntimeError('Round-two training failed: '+', '.join(failures))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='0'); p.add_argument('--trials', default=','.join(VARIANTS))
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    p.add_argument('--dry-run', action='store_true'); p.add_argument('--no-push', action='store_true')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--worker', choices=VARIANTS); mode.add_argument('--train-one',choices=VARIANTS)
    args=p.parse_args(); args.output_root=args.output_root.resolve()
    if args.worker:
        try: profile_worker(args)
        except Exception:
            path=args.output_root/'trials'/f'screen_{args.worker}_s{SEED}'/'status.json'
            write(path,dict(status='failed',step=0,error=traceback.format_exc(),updated=stamp()))
            raise
    elif args.train_one:
        trainer().run(argparse.Namespace(output_root=str(args.output_root),stage='screen',variant=args.train_one,seed=SEED,wandb=False))
    else: execute(args)


if __name__=='__main__': main()
