"""Activate pllm/protenix_abtcr; python -B SCRIPT --run ABS --check|--group N.

Uses the pinned training implementation and existing downstream task protocols.
Process-local adapters never change another evaluation or the live training run.
"""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))


def report(run):
    from scripts.downstream.run_v5_latest_suite import write
    from downstream.benchmark.ab_antigen.abbibench_scope import summarize_run
    plan = json.loads((run/'plan.json').read_text())
    counts, rows = Counter(), []
    for task in plan['tasks']:
        path = run/task/'status.json'
        state = json.loads(path.read_text()) if path.exists() else {'status': 'pending'}
        counts[state['status']] += 1
        detail = state.get('error', '').replace('|', '/').replace('\n', ' ')[:300]
        rows.append(f"| {task} | {state['status']} | {detail} | [产物]({run/task}) |")
    stamp = datetime.now(timezone.utc).isoformat()
    start, end = f'<!-- {run.name}:start -->', f'<!-- {run.name}:end -->'
    block = '\n'.join([start, '## 新固定双链75.6M / checkpoint50000 下游评测', '',
        '用户指定checkpoint50000；V5数据，固定H/β187及L/α154槽位，残差0.1。独立快照使用训练绑定源码；训练继续。',
        '常规AB/TCR及AB–Ag任务；AbBiBench仅官方16 assay，CHIMERA官方IMGT H3任务。',
        'CDR使用原AirGen解码及位置/X掩码（Kong4步、SAb2步）；整链生成使用本模型每链EOS及原生解码，明确标注模型/解码差异。TCR binding沿用冻结表征+MLP五折。',
        f'更新时间：{stamp}；状态：{dict(counts)}。',
        f'[选点]({run}/selection.json)；[计划]({run}/plan.json)；[CPU检查]({run}/cpu_gate/cpu_checks.json)。', '',
        '| 任务 | 状态 | 说明 | 证据 |', '|---|---|---|---|', *rows, end])
    with (run/'report.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Shared lock with other native suite publishers.
        with (ROOT/'output/downstream_generation/v5_native_best_20261008/results_update.lock').open('w') as shared:
            fcntl.flock(shared, fcntl.LOCK_EX)
            path = ROOT/'downstream/benchmark/EVALUATION_PROCESS.md'
            text = path.read_text()
            if start in text:
                a = text.index(start); b = text.index(end, a)+len(end)
                text = text[:a]+block+text[b:]
            else:
                head, tail = text.split('\n', 1); text = head+'\n\n'+block+'\n'+tail
            temp = path.with_suffix('.fixed_chain.tmp'); temp.write_text(text); temp.replace(path)
        write(run/'summary.json', dict(updated_utc=stamp, counts=dict(counts),
                                      abbibench_benchmark16=summarize_run(run)))


def install():
    from downstream import fixed_chain_eval as bridge, v5_native_eval as native
    from scripts.downstream import run_v5_latest_suite as suite
    cache = {}
    def load(checkpoint, device='cuda'):
        key = (str(checkpoint), device)
        if key not in cache:
            cache[key] = bridge.load(checkpoint, device)
        return cache[key]
    native.load, native.NativeBackbone = load, bridge.FixedChainBackbone
    original_aligned = suite.aligned
    def aligned(run, folder):
        module = original_aligned(run, folder)
        module.inputs = bridge.airgen_inputs
        return module
    suite.aligned, suite.report = aligned, report
    from downstream import v5_native_generation_eval as generation
    generation.generate = bridge.generate
    from scripts.downstream import run_v5_suite_task as task_runner
    original_install = task_runner.install
    def install_backbone(backbone):
        binding = original_install(backbone)
        binding.OURS_METHOD = 'Ours fixed-chain 75.6M frozen observed-mean + MLP'
        return binding
    task_runner.install = install_backbone
    original_probes = task_runner.probes
    def probes(task, checkpoint, out, backbone):
        original_probes(task, checkpoint, out, backbone)
        path = out/'metrics.json'; data = json.loads(path.read_text())
        data['model'] = 'fixed_chain_75.6M'
        suite.write(path, data)
    task_runner.probes = probes
    for name in ('pairing', 'tcr_generation'):
        original = getattr(generation, name)
        def wrapped(*args, _original=original, _name=name, **kwargs):
            _original(*args, **kwargs)
            out = args[0] if _name == 'pairing' else args[1]
            path = out/'metrics.json'; data = json.loads(path.read_text())
            if _name == 'pairing':
                data['length_condition'] = 'none; per-chain learned EOS within fixed187/154 slots'
            else:
                data['native_protocol'] = 'fixed_chain native beta EOS -> existing ANARCI CDR3 extraction; no reference length'
            suite.write(path, data)
        setattr(generation, name, wrapped)
    return suite


if __name__ == '__main__':
    suite = install()
    if '--group' not in sys.argv:
        suite.main()
    else:
        import argparse
        import subprocess
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--run', type=Path, required=True)
        parser.add_argument('--group', type=int, required=True)
        args = parser.parse_args()
        plan = json.loads((args.run/'plan.json').read_text())
        results = []
        for task in plan['groups'][args.group]:
            if task == 't2b':  # Removed from benchmark scope on 2026-10-14.
                continue
            if (args.run/'STOP').exists():
                break
            folder = args.run/task; folder.mkdir(exist_ok=True)
            if (folder/'status.json').exists():
                continue
            with (folder/'run.log').open('x') as log:
                result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                         '--run', str(args.run), '--task', task],
                                        stdout=log, stderr=subprocess.STDOUT)
            if not (folder/'status.json').exists():
                suite.write(folder/'status.json', dict(status='failed',
                    error='Worker exited before task status', returncode=result.returncode))
            results.append(dict(task=task, returncode=result.returncode))
            suite.write(args.run/f'group_{args.group}.json', dict(results=results))
            report(args.run)
        if any(item['returncode'] for item in results):
            raise SystemExit(1)
