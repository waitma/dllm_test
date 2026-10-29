"""Run A0-A7 once each, with seed 43, on eight independent GPUs.

Run: bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_8gpu.sh --gpus 0,1,2,3,4,5,6,7
Preview: append --dry-run. This launcher reuses the frozen V8 trainer and data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from urllib.parse import urlparse

ROOT = Path(os.environ.get('ARCH70M_PROJECT_ROOT', Path(__file__).resolve().parent)).resolve()
sys.path.insert(0, os.environ.get('ARCH70M_FROZEN_SOURCE', str(ROOT)))
from experiments.arch70m.common import read, write, sha, digest, source_binding, stamp, lock, VARIANTS

SEED = 43
SOURCE_SHA = '053297983e0f2ece2aaad807032695fb5aebd1e0f221138a9bf0d202cc9b9ac4'
DATA_SHA = '4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51'
BRANCH = 'explore/arch70m-v8-20261028'
REPORT_REL = Path('docs/arch70m_results/seed43_validation')
DISTRIBUTED_ENV = {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                   'ROLE_RANK', 'ROLE_WORLD_SIZE', 'ROLE_NAME', 'MASTER_ADDR', 'MASTER_PORT'}


def gpu_ids(value):
    devices = value.split(',')
    if len(devices) != 8 or len(set(devices)) != 8 or any(not x or any(c.isspace() for c in x) for x in devices):
        raise argparse.ArgumentTypeError('Exactly eight distinct GPU IDs or UUIDs are required.')
    return devices


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gpus', type=gpu_ids, help='Eight IDs/UUIDs; defaults to ARCH_GPUS, CUDA_VISIBLE_DEVICES, then 0-7.')
    p.add_argument('--output-root', type=Path, default=ROOT/'output/arch70m_v8_20261028')
    p.add_argument('--micro-batch', type=int, default=56,
                   help='Common micro batch for all eight variants; global batch stays 512. Default: 56.')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Print all eight assignments; no GPU access or output writes.')
    mode.add_argument('--publish-only', action='store_true', help='Refresh/commit/push reports without starting GPU jobs.')
    p.add_argument('--no-push', action='store_true', help='Commit result reports locally without pushing.')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--worker', choices=VARIANTS, help=argparse.SUPPRESS)
    p.add_argument('--source', type=Path, help=argparse.SUPPRESS)
    return p


def plan(args):
    if not 1 <= args.micro_batch <= 512:
        raise ValueError('--micro-batch must be between 1 and 512.')
    return [dict(variant=v, seed=SEED, stage='screen', epochs=1, gpu=gpu,
                 micro_batch=args.micro_batch, global_batch=512,
                 trial=f'screen_{v}_s{SEED}') for v, gpu in zip(VARIANTS, args.gpus)]


def visible_uuids(devices):
    uuids = []
    for device in devices:
        result = subprocess.run(['nvidia-smi', '--id='+device, '--query-gpu=uuid', '--format=csv,noheader'],
                                text=True, capture_output=True, check=True)
        rows = [x.strip() for x in result.stdout.splitlines() if x.strip()]
        if len(rows) != 1:
            raise ValueError('Requested GPU is missing or ambiguous: '+device+'; no trials launched.')
        uuids.extend(rows)
    if len(uuids) != 8 or len(set(uuids)) != 8 or any(not x.startswith('GPU-') for x in uuids):
        raise ValueError('All eight requested IDs must resolve to different physical GPU UUIDs; no trials launched.')
    return uuids


def worker_env(source, gpu):
    env = {k:v for k,v in os.environ.items()
           if k not in DISTRIBUTED_ENV and not k.startswith(('NCCL_', 'TORCHELASTIC_'))}
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(source), ARCH70M_PROJECT_ROOT=str(ROOT),
               ARCH70M_FROZEN_SOURCE=str(source),
               CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', USE_TF='0', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    env.pop('CUDA_LAUNCH_BLOCKING', None)
    return env


def cuda_device_id(requested):
    """NVML UUIDs may differ from CUDA UUIDs on the worker's virtualized GPU stack."""
    if requested.isdecimal(): return requested
    result = subprocess.run(['nvidia-smi', '--id='+requested, '--query-gpu=index', '--format=csv,noheader'],
                            text=True, capture_output=True, check=True)
    rows = result.stdout.split()
    if len(rows) != 1 or not rows[0].isdecimal():
        raise ValueError('Cannot resolve a numeric CUDA device selector for '+requested)
    return rows[0]


def qualify_cuda():
    """Exercise the CUDA runtime and a kernel; NVML device_count alone is insufficient."""
    import torch
    torch.cuda.init()
    count = torch.cuda.device_count()
    if count != 1: raise RuntimeError(f'Expected one visible CUDA GPU, found {count}.')
    value = torch.ones(1, device='cuda:0')+1
    torch.cuda.synchronize()
    if value.item() != 2: raise RuntimeError('CUDA kernel qualification failed.')
    prop = torch.cuda.get_device_properties(0)
    return dict(stage='cuda_qualified', cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER'), device_count=count,
                device=prop.name, cuda_uuid=str(getattr(prop, 'uuid', 'unavailable')), torch=torch.__version__)


def verify_reference(out, micro):
    branch = subprocess.check_output(['git', '-C', str(ROOT), 'branch', '--show-current'], text=True).strip()
    if branch != BRANCH:
        raise ValueError('Run this experiment from the exploration branch: '+BRANCH)
    if digest(source_binding()) != SOURCE_SHA or sha(out/'data/index.json') != DATA_SHA:
        raise ValueError('Frozen model/trainer source or fixed 10% data index differs from seed42.')
    settings = read(ROOT/'experiments/arch70m/base.json')
    total = math.ceil(read(out/'data/index.json')['records']/512)
    expected = dict(settings=settings, source_sha256=SOURCE_SHA, data_sha256=DATA_SHA,
                    seed=42, epochs=1, total_steps=total)
    for v in VARIANTS:
        t = out/'trials'/f'screen_{v}_s42'
        if read(t/'status.json')['status'] != 'completed':
            raise ValueError('Original seed42 screening is incomplete: '+v)
        b = read(t/'binding.json')
        if any(b.get(k) != x for k,x in expected.items()) or b['variant'] != v:
            raise ValueError('Original seed42 binding differs: '+v)
        t = out/'trials'/f'screen_{v}_s{SEED}'
        if t.exists():
            try:
                with lock(t/'trial.lock', blocking=False): pass
            except BlockingIOError as exc:
                raise ValueError('A seed43 worker already holds its training lock: '+v) from exc
        if (t/'binding.json').exists():
            expected43 = dict(expected, seed=SEED, variant=v, micro_batch=micro)
            b = read(t/'binding.json')
            if any(b.get(k) != x for k,x in expected43.items()):
                raise ValueError('Existing seed43 trial has different settings; use its original micro batch: '+v)
    return settings, total


def completed(out, variant):
    t = out/'trials'/f'screen_{variant}_s{SEED}'
    if not (t/'status.json').exists() or read(t/'status.json')['status'] != 'completed':
        return False
    s = read(t/'status.json'); step = s['total_steps']
    cp = t/'checkpoints'/f'step_{step:06d}'
    receipt = read(cp/'receipt.json')
    if s['step'] != step or receipt['status'] != 'complete' or len(receipt['files']) != 6:
        raise ValueError('Incomplete final checkpoint: '+variant)
    if any(sha(cp/name) != checksum for name,checksum in receipt['files'].items()):
        raise ValueError('Final checkpoint hash differs: '+variant)
    metric = read(t/'metrics'/f'screen_{step:06d}.json')
    if metric['step'] != step or metric['partition'] != 'screen':
        raise ValueError('Missing final screening metrics: '+variant)
    return True


def profile_one(args):
    """Check three real updates at one common micro batch, then start a fresh frozen trainer."""
    sys.path.insert(0, str(args.source))
    import torch
    from experiments.arch70m.train import (setup, construct, profile_batch, update, make_scheduler,
                                          IndexedPreparedDataset, VFamilyCollator, FixedChainConfigProxy, corrupt)
    from experiments.arch70m.common import source_binding as worker_binding
    if digest(worker_binding()) != SOURCE_SHA or sha(args.output_root/'data/index.json') != DATA_SHA:
        raise ValueError('Worker source/data binding differs.')
    t = args.output_root/'trials'/f'screen_{args.worker}_s{SEED}'; t.mkdir(parents=True, exist_ok=True)
    try:
        evidence = qualify_cuda()
    except Exception as exc:
        write(t/'cuda_device.json', dict(stage='cuda_failed', error=str(exc),
              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), updated=stamp()))
        raise
    write(t/'cuda_device.json', dict(evidence, updated=stamp()))
    print(json.dumps(evidence), flush=True)
    profile_path = t/'profile.json'
    with lock(t/'seed43_profile.lock', blocking=False):
        if profile_path.exists():
            prior = read(profile_path)
            if (prior['variant'] != args.worker or prior['micro_batch'] != args.micro_batch or
                    prior.get('common_micro_seed_validation') != SEED):
                raise ValueError('Existing fixed-micro profile differs; do not overwrite it.')
        else:
            with lock(t/'trial.lock', blocking=False):
                settings = read(args.source/'experiments/arch70m/base.json'); setup(SEED)
                device = torch.device('cuda:0')
                dataset = IndexedPreparedDataset(args.output_root/'data/index.json')
                collate = VFamilyCollator(FixedChainConfigProxy(settings), family_spec=settings['family_prediction'])
                clean = collate([dataset[i] for i in range(min(512, len(dataset)))])
                model, optimizer = construct(settings, args.worker, SEED, device)
                scheduler = make_scheduler(optimizer, 100, 5)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                batch = profile_batch(clean, args.micro_batch, settings['model']['max_length'], model.tokenizer)
                batch = corrupt(batch, model.tokenizer, torch.Generator().manual_seed(9100))
                started = time.monotonic()
                for _ in range(3):
                    update(model, batch, optimizer, scheduler, args.micro_batch, device)
                torch.cuda.synchronize()
                allocated = torch.cuda.max_memory_allocated(); reserved = torch.cuda.max_memory_reserved()
                free, capacity = torch.cuda.mem_get_info()
                external = max(0, capacity-free-torch.cuda.memory_reserved())
                safe = reserved+external <= .85*capacity
                evidence = dict(micro_batch=args.micro_batch, status='passed', safe=safe,
                                allocated_bytes=allocated, reserved_bytes=reserved, external_bytes=external,
                                total_bytes=capacity, seconds_per_step=(time.monotonic()-started)/3)
                write(t/'profile_attempts.json', dict(variant=args.worker, candidates=[evidence]))
                if not safe:
                    raise RuntimeError('Common micro batch exceeds the 85% memory threshold. Lower --micro-batch equally for all eight variants.')
                write(profile_path, dict(variant=args.worker, micro_batch=args.micro_batch,
                    grad_accum=math.ceil(512/args.micro_batch), final_micro_batch=512%args.micro_batch or args.micro_batch,
                    length=settings['model']['max_length'], candidates=[evidence], device=torch.cuda.get_device_name(),
                    common_micro_seed_validation=SEED, source_sha256=SOURCE_SHA, data_sha256=DATA_SHA,
                    note='Three real optimizer updates at padded maximum length, common micro batch across A0-A7; exec a new process for fresh training initialization.'))
    command = [sys.executable, '-B', '-m', 'experiments.arch70m.train', '--stage', 'screen',
               '--variant', args.worker, '--seed', str(SEED), '--output-root', str(args.output_root)]
    if args.wandb: command.append('--wandb')
    os.execv(sys.executable, command)


def report(out):
    from experiments.arch70m.report import records, flatten, export_csv, plot_curves
    folder = ROOT/REPORT_REL; folder.mkdir(parents=True, exist_ok=True)
    trials = {}; summary = []; curves = []; metrics = {}
    for seed in (42, SEED):
        for variant in VARIANTS:
            name = f'screen_{variant}_s{seed}'; t = out/'trials'/name
            status = read(t/'status.json') if (t/'status.json').exists() else dict(status='pending', step=0)
            train = records(t/'train.jsonl'); evals = records(t/'eval.jsonl')
            trials[name] = dict(status=status, train=train, eval=evals)
            row = dict(trial=name, variant=variant, seed=seed, status=status['status'], step=status.get('step', 0))
            if status['status'] == 'completed':
                final = read(t/'metrics'/f"screen_{status['step']:06d}.json")
                metrics[variant, seed] = flatten(final)
                row.update(metrics[variant, seed])
            summary.append(row)
            for kind, values in (('train', train), ('eval', evals)):
                curves.extend(dict(trial=name, curve=kind, **flatten(v)) for v in values)
    delta = []
    fields = ('token.all.loss', 'generation.conditional_aar', 'generation.conditional_ce',
              'generation.conditional_validity', 'generation.conditional_length_mae',
              'generation.cdr_aar', 'representation.macro_auroc', 'representation.macro_auprc')
    for seed in (42, SEED):
        base = metrics.get(('A0', seed))
        if base:
            for variant in VARIANTS:
                m = metrics.get((variant, seed))
                if m:
                    delta.extend(dict(seed=seed, variant=variant, metric=k, baseline=base[k], value=m[k],
                                      difference_from_A0=m[k]-base[k]) for k in fields)
    write(folder/'summary.json', dict(updated=stamp(), seed=SEED, training_epochs=1, trials=summary,
          conclusion='Seed42/43 replication only; no three-seed confirmation or statistical significance claim.'))
    export_csv(folder/'summary.csv', summary); export_csv(folder/'all_curves.csv', curves)
    export_csv(folder/'paired_vs_A0.csv', delta); plot_curves(trials, folder)
    manifest = out/'seed43_validation/manifest.json'
    if manifest.exists(): write(folder/'manifest.json', read(manifest))
    done = sum(r['seed'] == SEED and r['status'] == 'completed' for r in summary)
    lines = ['# A0–A7：仅新增 seed 43', '', f'更新时间：{stamp()}；完成 {done}/8。', '',
             '固定原V8形式、653,288条10%子集；每组从头初始化1epoch/1276步，global512。',
             '八张卡各自执行一个单GPU进程，没有DDP/NCCL；统一micro batch由manifest记录。',
             '与seed42同epoch终点比较；seed42原先A5的micro96、其余micro56，须保留该数值累积差异。',
             '本轮只作第二种子复核，不能沿用原三种子显著性判定。', '',
             '| 架构 | seed43状态 | step | 整链AAR | CDR AAR | AUROC | AUPRC |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['seed'] != SEED: continue
        def fmt(key): return f'{r[key]:.4f}' if key in r else '—'
        lines.append(f"| {r['variant']} | {r['status']} | {r['step']} | {fmt('generation.conditional_aar')} | {fmt('generation.cdr_aar')} | {fmt('representation.macro_auroc')} | {fmt('representation.macro_auprc')} |")
    lines += ['', f'![完整曲线]({folder}/curves.png)', '',
              f'[原始逐步曲线]({folder}/all_curves.csv) · [同seed相对A0差值]({folder}/paired_vs_A0.csv)', '']
    (folder/'REPORT.md').write_text('\n'.join(lines))
    return done


def push_results():
    """Use a reachable runtime proxy or direct Git, without changing shared Git config."""
    remote = subprocess.check_output(['git', '-C', str(ROOT), 'remote', 'get-url', 'origin'], text=True).strip()
    configured = subprocess.run(['git', '-C', str(ROOT), 'config', '--get-urlmatch', 'http.proxy', remote],
                                text=True, capture_output=True)
    candidates = [os.environ.get(k) for k in ('https_proxy', 'HTTPS_PROXY', 'http_proxy', 'HTTP_PROXY', 'all_proxy', 'ALL_PROXY')]
    candidates.append(configured.stdout.strip() if configured.returncode == 0 else None)
    selected = ''
    for candidate in candidates:
        if not candidate: continue
        proxy = urlparse(candidate if '://' in candidate else 'http://'+candidate)
        if proxy.hostname in ('localhost', '127.0.0.1', '::1'):
            try:
                with socket.create_connection((proxy.hostname, proxy.port or 80), timeout=.5): pass
            except OSError:
                continue
        selected = candidate; break
    # A URL-specific -c setting overrides copied developer-machine proxy config.
    command = ['git', '-C', str(ROOT), '-c', 'http.proxy='+selected,
               '-c', 'http.'+remote+'.proxy='+selected, 'push', 'origin', 'HEAD:'+BRANCH]
    env = os.environ.copy()
    for key in ('https_proxy', 'HTTPS_PROXY', 'http_proxy', 'HTTP_PROXY', 'all_proxy', 'ALL_PROXY'):
        env.pop(key, None)
    try:
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return dict(status='pending', error='Git push timed out; local result commit retained.')
    error = result.stderr.strip()
    for candidate in candidates:
        if candidate:
            parsed = urlparse(candidate if '://' in candidate else 'http://'+candidate)
            if parsed.username or parsed.password: error = error.replace(candidate, '<redacted proxy>')
    return dict(status='pushed' if result.returncode == 0 else 'pending', exit_code=result.returncode,
                transport='runtime_proxy' if selected else 'direct', error=error[-3000:] if result.returncode else None)


def publish(out, message, push):
    def git(*args): return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    with lock(ROOT/'output/arch70m_publish.lock'):
        if git('branch', '--show-current') != BRANCH: raise ValueError('Result publication requires the exploration branch.')
        if git('diff', '--cached', '--name-only'): raise ValueError('Unrelated staged edits; publication stopped.')
        done = report(out)
        p = ROOT/'PROJECT_PROCESS.md'
        p.write_text(f'> Last updated: {stamp()} (UTC)。seed43八架构复核 {done}/8 完成；{message}。原始日志 {out}；完整报告 {ROOT/REPORT_REL}/REPORT.md。仅新增seed43，各1epoch；未执行旧三种子统计。\n\n'+p.read_text())
        git('add', '--', str(REPORT_REL), 'PROJECT_PROCESS.md'); git('diff', '--cached', '--check')
        git('-c', 'user.name=Codex', '-c', 'user.email=codex@openai.com', 'commit', '-m', 'arch70m: '+message)
        result = push_results() if push else dict(status='disabled')
        result.update(commit=git('rev-parse', 'HEAD'), updated=stamp())
        write(out/'seed43_validation/publication.json', result)
        if result['status'] == 'pending':
            print(json.dumps(dict(event='publication_pending', **result)), flush=True)
        return result


def execute(args):
    if args.publish_only:
        result = publish(args.output_root, 'refresh seed43 reports without GPU work', not args.no_push)
        print(json.dumps(dict(event='publication', **result)), flush=True)
        return
    jobs = plan(args)
    if args.dry_run:
        print(json.dumps(dict(seed=SEED, seeds=[SEED], epochs=1, independent_single_gpu=True,
                             output_root=str(args.output_root), assignments=jobs), indent=2))
        return
    uuids = visible_uuids(args.gpus)  # Fail before any output mutation if cards are missing/aliased.
    settings, total = verify_reference(args.output_root, args.micro_batch)
    from experiments.arch70m.suite import snapshot, verify_data
    verify_data(args.output_root)
    with lock(args.output_root/'seed43_validation/controller.lock', blocking=False):
        source, signature = snapshot(args.output_root)
        runner_sha = sha(__file__)
        runner = args.output_root/'seed43_validation'/f'launcher_{runner_sha}.py'
        if not runner.exists(): shutil.copyfile(__file__, runner)
        if sha(runner) != runner_sha: raise ValueError('Launcher snapshot changed.')
        for job, gpu in zip(jobs, uuids):
            job['gpu_uuid'] = gpu
            job['cuda_visible_device'] = cuda_device_id(job['gpu'])
        manifest = dict(created=stamp(), seeds=[SEED], baseline_seed=42, variants=list(VARIANTS), epochs=1,
                        total_steps=total, global_batch=512, micro_batch=args.micro_batch,
                        source_sha256=signature, launcher_sha256=runner_sha, data_sha256=DATA_SHA,
                        training_recipe=settings['recipe'], assignments=jobs)
        path = args.output_root/'seed43_validation/manifest.json'
        if path.exists():
            prior = read(path)
            for key in ('seeds', 'variants', 'epochs', 'total_steps', 'micro_batch', 'source_sha256', 'data_sha256'):
                if prior[key] != manifest[key]: raise ValueError('Existing seed43 manifest differs: '+key)
        write(path, manifest)

        def run_one(job):
            name = job['trial']; t = args.output_root/'trials'/name
            if completed(args.output_root, job['variant']): return name, 0, 'verified_completed'
            t.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-B', str(runner), '--worker', job['variant'], '--source', str(source),
                       '--output-root', str(args.output_root), '--micro-batch', str(args.micro_batch)]
            if args.wandb: command.append('--wandb')
            print(json.dumps(dict(job, event='launch')), flush=True)
            with (t/'console.log').open('a') as log:
                log_start = log.tell()
                p = subprocess.Popen(command, cwd=source, env=worker_env(source, job['cuda_visible_device']), stdout=log, stderr=subprocess.STDOUT)
                write(t/'process.json', dict(pid=p.pid, gpu=job['gpu'], gpu_uuid=job['gpu_uuid'],
                    cuda_visible_device=job['cuda_visible_device'], command=command, log_start=log_start, started=stamp()))
                code = p.wait()
            if code == 0 and not completed(args.output_root, job['variant']):
                code = 1
            if code:
                status = read(t/'status.json') if (t/'status.json').exists() else {}
                with (t/'console.log').open('rb') as log:
                    log.seek(0, 2); size = log.tell(); log.seek(max(log_start, size-6000))
                    tail = log.read().decode(errors='replace')
                errors = [line for line in tail.splitlines() if 'Error:' in line or 'Exception:' in line]
                error = errors[-1] if errors else '\n'.join(tail.splitlines()[-10:]) or f'Worker exited {code} without new log output.'
                write(t/'status.json', dict(status, status='failed', step=status.get('step', 0),
                    exit_code=code, training_exit_code=code, updated=stamp(), error=error, error_tail=tail))
                print(json.dumps(dict(event='training_failed', trial=name, training_exit_code=code, error=error,
                                      console_log=str(t/'console.log'))), flush=True)
            try:
                publication = publish(args.output_root, f'record {name} '+('completed' if not code else 'failed'), not args.no_push)
            except Exception:
                write(t/'seed43_publication_error.json', dict(updated=stamp(), error=traceback.format_exc()))
                publication = dict(status='local_report_failed')
            return name, code, publication['status']

        failures = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for result in as_completed([pool.submit(run_one, job) for job in jobs]):
                value = result.result()
                print(json.dumps(dict(trial=value[0], exit_code=value[1], training_exit_code=value[1],
                    status='training_failed' if value[1] else 'completed', publication_status=value[2])), flush=True)
                if value[1]: failures.append(value[0])
        try:
            publication = publish(args.output_root, 'final seed43 eight-architecture report', not args.no_push)
        except Exception:
            write(args.output_root/'seed43_validation/publication_error.json', dict(error=traceback.format_exc(), updated=stamp()))
            publication = dict(status='local_report_failed')
        print(json.dumps(dict(event='suite_finished', training_failures=failures, publication_status=publication['status'])), flush=True)
        if failures: raise RuntimeError('Training failed: '+', '.join(failures)+'; original errors printed above and saved in each console.log.')


def main():
    args = parser().parse_args(); args.output_root = args.output_root.resolve()
    if not 1 <= args.micro_batch <= 512: raise ValueError('--micro-batch must be between 1 and 512.')
    if args.worker:
        if args.source is None: raise ValueError('Worker requires a frozen --source.')
        profile_one(args)
    else:
        if args.gpus is None and not args.publish_only:
            args.gpus = gpu_ids(os.environ.get('ARCH_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES') or '0,1,2,3,4,5,6,7')
        execute(args)


if __name__ == '__main__': main()
