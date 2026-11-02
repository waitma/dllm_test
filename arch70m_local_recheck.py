"""Replay final checkpoints: bash /absolute/path/launch_arch70m_local_recheck.sh --gpu 0.

Evaluation only. The original four-GPU run, frozen sources and decision are read-only.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore')
ORIGINAL = ROOT / 'output/arch70m_confirm_4gpu_20261031'
OUT = ROOT / 'output/arch70m_local_recheck_20261102'
SOURCE_SHA = '3c9c424d816b9fda7adfba5047d9e075ac15dc036342b59461edd35031e50971'
JOBS = tuple((v, s) for s in (44, 42, 43) for v in ('A0', 'B0', 'B1'))
ENDPOINTS = ('generation.conditional_aar', 'generation.cdr_aar',
             'generation.conditional_length_mae', 'generation.conditional_validity',
             'generation.conditional_ce', 'generation.cdr_ce',
             'representation.macro_auroc', 'representation.macro_auprc')


def safe_output(output, original):
    output, original = Path(output).resolve(), Path(original).resolve()
    if output == original or original in output.parents or output in original.parents:
        raise ValueError('Replay output must be separate from the original run.')
    protected = [ROOT / 'output' / n for n in ('arch70m_v8_20261028', 'arch70m_round2_20261029',
                 'arch70m_hierarchy_m32_20261029', 'arch70m_confirm_20261031')]
    if any(output == p or p in output.parents for p in protected):
        raise ValueError('Historical experiments are read-only.')
    return output


def comparison(expected, actual, atol=1e-8, rtol=1e-6):
    """Keep list order, identity fields and discrete values strict; report all drift."""
    differences = []; exact = True; numeric = 0; max_absolute = 0.

    def visit(a, b, path):
        nonlocal exact, numeric, max_absolute
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                exact = False
                differences.append(dict(path=path, reason='keys', expected=sorted(a), actual=sorted(b)))
            for key in sorted(set(a) & set(b)):
                visit(a[key], b[key], path + '.' + str(key))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                exact = False
                differences.append(dict(path=path, reason='length', expected=len(a), actual=len(b)))
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f'{path}[{i}]')
        elif type(a) in (float, int) and type(b) in (float, int):
            numeric += 1
            finite = math.isfinite(a) and math.isfinite(b)
            delta = abs(a - b) if finite else None
            if finite: max_absolute = max(max_absolute, delta)
            if a != b: exact = False
            close = a == b if type(a) is int and type(b) is int else finite and delta <= atol + rtol * abs(a)
            if not close:
                differences.append(dict(path=path, reason='numeric', expected=a, actual=b, absolute_difference=delta))
        elif type(a) is not type(b) or a != b:
            exact = False
            differences.append(dict(path=path, reason='value', expected=a, actual=b))

    visit(expected, actual, '$')
    return dict(exact_equal=exact, strict_close=not differences, atol=atol, rtol=rtol,
                numeric_values=numeric, max_absolute_difference=max_absolute,
                difference_count=len(differences), differences=differences)


def endpoint(metrics, key):
    value = metrics
    for part in key.split('.'): value = value[part]
    return value


def worker(args):
    # The standalone replay adapter is outside the frozen set; every model/evaluator
    # import below must resolve inside the checkpoint's original source snapshot.
    source = args.source.resolve()
    sys.path[:] = [str(source)] + [p for p in sys.path if Path(p or '.').resolve() != ROOT]
    import gc
    from importlib.metadata import version
    import torch
    from safetensors.torch import load_file
    from architecture_confirm.common import read, write, sha, digest, source_binding, stamp
    from architecture_confirm.models import build
    from experiments.arch70m.evaluate import evaluate
    from experiments.arch70m.train import setup
    import experiments.arch70m.evaluate as evaluator
    if Path(evaluator.__file__).resolve().parent.parent.parent != source:
        raise ValueError('Evaluator was not imported from the immutable snapshot.')
    if digest(source_binding()) != SOURCE_SHA: raise ValueError('Frozen source changed.')
    v, s = args.worker
    s = int(s); trial = args.original / 'trials' / f'confirm_{v}_s{s}'
    checkpoint = trial / 'checkpoints/step_003828'
    target = args.output_root / 'trials' / trial.name
    target.mkdir(parents=True, exist_ok=True)
    binding = read(checkpoint / 'binding.json')
    runtime = dict(torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
        packages={n: version(n) for n in ('numpy', 'transformers', 'safetensors', 'scikit-learn')})
    if any(runtime[k] != binding[k] for k in runtime):
        raise ValueError('Replay runtime differs from checkpoint binding: ' + str(runtime))
    if sha(args.original / 'data/index.json') != binding['data_sha256']:
        raise ValueError('Evaluation data binding changed.')
    for name, value in read(checkpoint / 'receipt.json')['files'].items():
        if sha(checkpoint / name) != value: raise ValueError('Checkpoint changed: ' + name)
    setup(s); device = torch.device('cuda:0')
    tiny = torch.tensor([1., 2.], device=device).square()
    torch.cuda.synchronize()
    if tiny.tolist() != [1., 4.]: raise ValueError('Local CUDA kernel qualification failed.')
    write(target / 'runtime.json', dict(runtime, cuda_uuid=str(torch.cuda.get_device_properties(0).uuid),
          evaluator=evaluator.__file__, source_sha256=SOURCE_SHA, model_sha256=sha(checkpoint / 'model.safetensors')))
    started = time.monotonic()
    write(target / 'status.json', dict(status='loading_checkpoint', variant=v, seed=s, updated=stamp(), pid=os.getpid()))
    model = build(binding['settings'], v, s)
    model.load_state_dict(load_file(str(checkpoint / 'model.safetensors'), device='cpu'), strict=True)
    model.to(device).eval()
    if model.sdpa_backend != binding['sdpa']: raise ValueError('Attention backend changed.')
    write(target / 'status.json', dict(status='evaluating', variant=v, seed=s, updated=stamp(), pid=os.getpid()))
    print(json.dumps(dict(event='evaluation_started', variant=v, seed=s, device=runtime['device'])), flush=True)
    metrics = evaluate(model, read(args.original / 'data/index.json'), 'confirm', device, target / 'metrics', 3828, tasks=True)
    torch.cuda.synchronize()
    observations = read(target / 'metrics/confirm_003828_observations.json')
    original_metrics = read(trial / 'metrics/confirm_003828.json')
    result = dict(variant=v, seed=s, seconds=time.monotonic()-started,
        metrics=comparison(original_metrics, metrics),
        observations=comparison(read(trial / 'metrics/confirm_003828_observations.json'), observations),
        endpoints={k: dict(original=endpoint(original_metrics, k), replay=endpoint(metrics, k)) for k in ENDPOINTS},
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(target / 'comparison.json', result)
    write(target / 'status.json', dict(status='completed', updated=stamp(), variant=v, seed=s,
          metrics_strict_close=result['metrics']['strict_close'], observations_strict_close=result['observations']['strict_close']))
    print(json.dumps(dict(event='evaluation_completed', variant=v, seed=s, seconds=result['seconds'],
          metrics_strict_close=result['metrics']['strict_close'], observations_strict_close=result['observations']['strict_close'])), flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


def report(out):
    from experiments.arch70m.common import read, write, stamp
    rows = []
    for v, s in JOBS:
        p = out / 'trials' / f'confirm_{v}_s{s}' / 'comparison.json'
        if p.exists(): rows.append(read(p))
    write(out / 'summary.json', dict(updated=stamp(), completed=len(rows), total=len(JOBS), trials=rows))
    lines = ['# 本地最终检查点复核', '', f'更新时间：{stamp()} UTC；GPU重评估 {len(rows)}/9。', '',
        '原四卡结果和源代码只读；使用原V8输入、同一固定10%索引、原confirm留出、原解码/表征评估。',
        '这是同一检查点的复现检查，不增加训练种子或独立样本，不改变原稳定性门槛。',
        '未重新训练抗体辅助五折head；原五折输入/结果与checkpoint完整性在CPU阶段核验。', '',
        '| 模型 | seed | 指标严格复现 | 逐样本严格复现 | 整链AAR | CDR AAR | AUROC | AUPRC |',
        '|---|---:|---|---|---:|---:|---:|---:|']
    for r in rows:
        vals = [r['endpoints'][k]['replay'] for k in ('generation.conditional_aar','generation.cdr_aar','representation.macro_auroc','representation.macro_auprc')]
        lines.append(f"| {r['variant']} | {r['seed']} | {r['metrics']['strict_close']} | {r['observations']['strict_close']} | " + ' | '.join(f'{x:.6f}' for x in vals) + ' |')
    if (out / 'decision_comparison.json').exists():
        d = read(out / 'decision_comparison.json')
        lines += ['', f"冻结统计判定：原始={d['original_decision']}；本地重评估={d['replay_decision']}。",
                  f"完整统计文件严格一致：{d['comparison']['strict_close']}。"]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    with (out / 'endpoints.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['variant','seed','metric','original','replay','delta'])
        writer.writeheader()
        for r in rows:
            for key, x in r['endpoints'].items():
                writer.writerow(dict(variant=r['variant'], seed=r['seed'], metric=key, **x,
                                     delta=None if x['original'] is None or x['replay'] is None else x['replay']-x['original']))
    return rows


def publish(out):
    """Publish this validation receipt only; a network failure never changes its status."""
    import shutil
    from experiments.arch70m.common import read, write, stamp, lock
    from arch70m_seed_validation import push_results
    def git(*parts):
        return subprocess.check_output(['git', '-C', str(ROOT), *parts], text=True).strip()
    try:
        with lock(ROOT / 'output/arch70m_publish.lock'):
            if git('branch', '--show-current') != 'explore/arch70m-v8-20261028':
                raise ValueError('Publication requires the exploration branch.')
            if git('diff', '--cached', '--name-only'): raise ValueError('Unrelated staged changes.')
            target = ROOT / 'docs/arch70m_results/local_recheck_20261102'
            target.mkdir(parents=True, exist_ok=True)
            for name in ('REPORT.md','summary.json','endpoints.csv','integrity.json','execution.json',
                         'decision.json','decision_comparison.json','manifest.json'):
                if (out / name).exists(): shutil.copyfile(out / name, target / name)
            status = read(out / 'execution.json')
            process = ROOT / 'PROJECT_PROCESS.md'
            process.write_text(f'> Last updated: {stamp()} (UTC)。本地检查点复核 status={status["status"]}；'
                f'完成{status.get("completed", 0)}/9，原四卡输出/冻结源码只读，原V8数据不变；'
                f'报告 {target}/REPORT.md。复现原结果不增加训练种子，不改变稳定提升判据。\n\n' + process.read_text())
            git('add', '--', 'PROJECT_PROCESS.md', str(target.relative_to(ROOT)))
            git('diff', '--cached', '--check')
            git('-c','user.name=Codex','-c','user.email=codex@openai.com','commit',
                '-m', 'Record local checkpoint replay: ' + status['status'])
            result = push_results()
            write(out / 'publication.json', dict(result, commit=git('rev-parse','HEAD'), updated=stamp()))
    except Exception:
        write(out / 'publication.json', dict(status='pending', error=traceback.format_exc(), updated=stamp()))


def audit(args, source):
    from architecture_confirm.common import completed, read, write, sha, source_binding, digest, DATA_SHA, stamp
    original = args.original; out = args.output_root
    if digest(source_binding()) != SOURCE_SHA: raise ValueError('Working source binding changed.')
    provenance = read(source / 'provenance.json')
    if provenance['source_sha256'] != SOURCE_SHA: raise ValueError('Snapshot provenance changed.')
    for name, value in provenance['files'].items():
        if sha(source / name) != value: raise ValueError('Snapshot changed: ' + name)
    if sha(original / 'data/index.json') != DATA_SHA: raise ValueError('Fixed 10% data changed.')
    index = read(original / 'data/index.json'); evidence = {}
    for references in index['files'].values():
        for ref in references.values():
            p = Path(ref['path']); value = sha(p)
            if value != ref['sha256']: raise ValueError('Evaluation input changed: ' + str(p))
            if sum(bool(x.strip()) for x in p.read_text().splitlines()) != ref['records']:
                raise ValueError('Evaluation input count changed: ' + str(p))
            evidence[str(p)] = value
    for name, value in index['protected_ab_binding']['data_sha256'].items():
        if sha(name) != value: raise ValueError('Protected AB input changed: ' + name)
        evidence[name] = value
    checkpoints = []; anchors = {}
    for v, s in JOBS:
        if not completed(original, v, s, SOURCE_SHA): raise ValueError(f'Incomplete original trial {v}/{s}')
        t = original / 'trials' / f'confirm_{v}_s{s}'
        train = [json.loads(x) for x in (t / 'train.jsonl').read_text().splitlines() if x.strip()]
        if [r['step'] for r in train] != list(range(1, 3829)): raise ValueError('Missing/duplicate training steps.')
        for r in train:
            if not all(math.isfinite(r[k]) for k in ('loss', 'grad_norm')): raise ValueError('Non-finite training history.')
        receipt = read(t / 'checkpoints/step_003828/receipt.json')
        for name in ('status.json','binding.json','profile.json','train.jsonl','ab_specificity/metrics.json',
                     'metrics/confirm_003828.json','metrics/confirm_003828_observations.json','checkpoints/step_003828/receipt.json'):
            anchors[str(t / name)] = sha(t / name)
        checkpoints.append(dict(variant=v, seed=s, verified_files=len(receipt['files']), steps=len(train),
            model_sha256=receipt['files']['model.safetensors'], receipt_sha256=sha(t / 'checkpoints/step_003828/receipt.json')))
        print(json.dumps(dict(event='integrity_passed', variant=v, seed=s, checkpoint_files=len(receipt['files']))), flush=True)
    anchors[str(original / 'decision.json')] = sha(original / 'decision.json')
    anchors[str(original / 'data/index.json')] = sha(original / 'data/index.json')
    result = dict(status='passed', updated=stamp(), source_sha256=SOURCE_SHA, data_sha256=DATA_SHA,
                  checkpoints=checkpoints, input_files=evidence, original_artifact_sha256=anchors)
    write(out / 'integrity.json', result)
    return result


def execute(args):
    from architecture_confirm.common import read, write, sha, stamp, lock
    args.output_root = safe_output(args.output_root, args.original)
    if not args.gpu.isdecimal(): raise ValueError('Use one numeric local GPU selector.')
    out = args.output_root
    source = args.original / 'sources' / SOURCE_SHA
    expected = dict(original=str(args.original), source=str(source), source_sha256=SOURCE_SHA,
                    jobs=[list(x) for x in JOBS], adapter_sha256=sha(__file__), atol=1e-8, rtol=1e-6)
    with lock(ROOT / 'output' / f'arch70m_local_gpu{args.gpu}.lock', blocking=False), lock(out / 'controller.lock', blocking=False):
        if (out / 'manifest.json').exists() and read(out / 'manifest.json') != expected:
            raise ValueError('Existing replay has a different immutable binding.')
        try:
            write(out / 'manifest.json', expected)
            write(out / 'execution.json', dict(status='integrity_audit', started=stamp(), pid=os.getpid(), gpu=args.gpu))
            integrity = audit(args, source)
            (out / 'adapter.py').write_bytes(Path(__file__).read_bytes())
            report(out)
            for v, s in JOBS:
                target = out / 'trials' / f'confirm_{v}_s{s}'; target.mkdir(parents=True, exist_ok=True)
                if (target / 'status.json').exists() and read(target / 'status.json')['status'] == 'completed': continue
                blocked = {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','MASTER_ADDR','MASTER_PORT','ARCH70M_FROZEN_SOURCE','CUDA_LAUNCH_BLOCKING'}
                env = {k: x for k, x in os.environ.items() if k not in blocked and not k.startswith(('NCCL_', 'TORCHELASTIC_'))}
                env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONPATH=str(source),
                    USE_TF='0', CUBLAS_WORKSPACE_CONFIG=':4096:8', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
                command = [sys.executable, '-B', str(out / 'adapter.py'), '--worker', v, str(s),
                           '--source', str(source), '--original', str(args.original), '--output-root', str(out)]
                write(out / 'execution.json', dict(status='gpu_evaluation', updated=stamp(), pid=os.getpid(), variant=v, seed=s, completed=len(report(out))))
                with (target / 'console.log').open('a') as log:
                    process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    write(target / 'process.json', dict(pid=process.pid, command=command, started=stamp(), gpu=args.gpu))
                    print(json.dumps(dict(event='launch', variant=v, seed=s, pid=process.pid)), flush=True)
                    def stop(_number, _frame):
                        try: os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError: pass
                        raise KeyboardInterrupt('Local replay interrupted.')
                    previous = {n: signal.signal(n, stop) for n in (signal.SIGINT, signal.SIGTERM)}
                    try: code = process.wait()
                    finally:
                        for n, handler in previous.items(): signal.signal(n, handler)
                if code: raise RuntimeError(f'GPU evaluation {v}/{s} exited {code}; see {target}/console.log')
                report(out)
            write(out / 'execution.json', dict(status='recomputing_statistics', updated=stamp(), pid=os.getpid(), completed=9))
            from architecture_confirm.decision import calculate
            observations = {}; metrics = {}
            for v, s in JOBS:
                t = out / 'trials' / f'confirm_{v}_s{s}' / 'metrics'
                observations[v, s] = read(t / 'confirm_003828_observations.json')
                metrics[v, s] = read(t / 'confirm_003828.json')
            # Call calculate directly: never reuse or overwrite the original decision cache.
            decision = calculate(observations, metrics, read(source / 'architecture_confirm/protocol.json')['decision']['bootstrap_repetitions'])
            write(out / 'decision.json', decision)
            original_decision = read(args.original / 'decision.json')
            original_decision.pop('observations_sha256', None)
            write(out / 'decision_comparison.json', dict(original_decision=original_decision['decision'],
                  replay_decision=decision['decision'], comparison=comparison(original_decision, decision)))
            if any(sha(path) != value for path, value in integrity['original_artifact_sha256'].items()):
                raise ValueError('An original artifact changed during replay.')
            rows = report(out)
            strict = all(r['metrics']['strict_close'] and r['observations']['strict_close'] for r in rows)
            write(out / 'execution.json', dict(status='completed', finished=stamp(), pid=os.getpid(), completed=9,
                  strict_reproduction=strict, decision=decision['decision'], original_artifacts_unchanged=True))
            print(json.dumps(dict(event='recheck_completed', strict_reproduction=strict, decision=decision['decision'])), flush=True)
        except BaseException:
            write(out / 'execution.json', dict(status='failed', updated=stamp(), pid=os.getpid(), error=traceback.format_exc()))
            raise
        finally:
            report(out)
            publish(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0'); parser.add_argument('--original', type=Path, default=ORIGINAL)
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--worker', nargs=2, metavar=('VARIANT','SEED')); parser.add_argument('--source', type=Path)
    args = parser.parse_args(); args.original = args.original.resolve()
    args.output_root = safe_output(args.output_root, args.original)
    if args.worker:
        if args.source is None or (args.worker[0], int(args.worker[1])) not in JOBS:
            parser.error('Worker requires the original snapshot and one frozen model/seed pair.')
        worker(args)
    else: execute(args)


if __name__ == '__main__': main()
