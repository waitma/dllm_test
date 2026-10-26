"""Container entrypoint for eight independent workers in a frozen launch bundle.

The existing persistent container invokes this file as launch_v5_direct.py
with --bundle /absolute/bundle. Each trial runs in a fresh child process.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def trial_specs(plan, plan_path):
    """Bind each multi-seed child to its own immutable plan and unique log name."""
    if 'trial_plans' not in plan:
        return [{'recipe': recipe, 'id': recipe, 'plan': str(plan_path)} for recipe in plan['recipes']]
    trials = plan['trial_plans']
    if len({item['id'] for item in trials}) != len(trials):
        raise ValueError('Duplicate trial identifiers')
    for item in trials:
        path = Path(item['plan'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Child plan binding changed')
        child = json.loads(path.read_text())
        if item['recipe'] not in child['recipes']:
            raise ValueError('Child recipe missing')
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    source = args.bundle / 'source'
    settings = json.loads((args.bundle / 'direct_launch.json').read_text())
    plan_path = args.bundle / 'plan.json'
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != settings['plan_sha256']:
        raise ValueError('Plan binding changed')
    plan = json.loads(plan_path.read_text())
    rank = int(os.environ['LOCAL_RANK'])
    # Communication happens only when reporting completion, never during trial training.
    import torch.distributed as dist
    dist.init_process_group('gloo', timeout=datetime.timedelta(hours=24))
    trials = trial_specs(plan, plan_path)[rank::8]
    results = []
    out = Path(plan['output'])
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for key in tuple(environment):
        if key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT') or key.startswith('TORCHELASTIC_'):
            environment.pop(key)
    for trial in trials:
        recipe = trial['recipe']
        with (out / 'logs' / (trial['id'] + '.log')).open('x') as log:
            command = [sys.executable, '-I', '-B', '-u', str(source / 'trial_entry.py'),
                       '--plan', trial['plan'], '--recipe', recipe, '--device', str(rank)]
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=source, env=environment)
        results.append({'recipe': recipe, 'trial': trial['id'], 'returncode': process.returncode})
    (out / f'worker_{rank}.json').write_text(json.dumps(results, indent=2) + '\n')
    all_results = [None] * 8
    dist.all_gather_object(all_results, results)
    dist.destroy_process_group()
    if any(item['returncode'] for group in all_results for item in group):
        raise RuntimeError('One or more trials failed; inspect individual logs. No automatic retry.')


if __name__ == '__main__':
    main()
