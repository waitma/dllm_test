"""Prepare the confirmed family's seed42 checkpoint for the existing 36-task suite.

Activate pllm/protenix_abtcr; python -m scripts.downstream.prepare_post_training_suite
    --decision /absolute/confirmation/selection.json --attempt 1
Copies verified weights, binds task code, runs CPU gates and writes eval_jobs YAMLs.
Submission is separate. Seed42 is fixed here before confirmation, not selected by score.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from downstream.fixed_chain_pairing import protocol as pairing_protocol
from scripts.research.prepare_fixed_chain_final_post_training import verify_decision
from scripts.research.run_fixed_chain_post_training import sha

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / 'output/downstream_generation/fixed_chain_75m_best112000_20261014'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    decision = args.decision.resolve()
    family = verify_decision(decision)
    selected = json.loads(decision.read_text())
    confirmation = json.loads(Path(selected['analysis_binding']['evaluation_plan']['path']).read_text())
    candidate = confirmation['models'][f'{family}_seed42']
    export = Path(candidate['checkpoint'])
    receipt = json.loads((export / 'export_verification.json').read_text())
    if not receipt['roundtrip_logits_exact'] or not receipt['all_state_tensors_exact']:
        raise ValueError('Candidate export did not pass roundtrip checks')
    for name, expected in candidate['files'].items():
        if sha(export / name) != expected:
            raise ValueError('Selected checkpoint changed')
    run = ROOT / f'output/downstream_generation/post_training_{family}_s42_r{args.attempt}_20261014'
    run.mkdir(exist_ok=False)
    shutil.copytree(export, run / 'checkpoint')
    shutil.copy2(decision, run / 'selection.json')
    shutil.copy2(TEMPLATE / 'airgen_evaluate.py', run / 'airgen_evaluate.py')
    for origin, target in [('run_post_training_suite.py', 'run_suite.py'),
                           ('run_post_training_pairing_reference.py', 'pairing_reference.py')]:
        shutil.copy2(ROOT / 'scripts/downstream' / origin, run / target)
    for folder in ['cpu_gate', *[f'gpu_gate_g{i}' for i in range(10)]]:
        gate = run / folder
        gate.mkdir()
        (gate / 'checkpoint').symlink_to(run / 'checkpoint', target_is_directory=True)
    plan = json.loads((TEMPLATE / 'plan.json').read_text())
    # Pin current shared adapters; freeze the run-specific wrappers themselves.
    code = {path: sha(path) for path in plan['code_sha256']
            if not Path(path).is_relative_to(TEMPLATE)}
    for path in [ROOT / 'downstream/post_training_eval.py', ROOT / 'downstream/fixed_chain_pairing.py',
                 ROOT / 'scripts/downstream/check_post_training_eval.py',
                 run / 'run_suite.py', run / 'airgen_evaluate.py', run / 'pairing_reference.py']:
        code[str(path)] = sha(path)
    binding = json.loads((export / 'training_binding.json').read_text())
    identity = {key: binding[key] for key in ('recipe', 'seed', 'step')}
    identity['model_sha256'] = candidate['files']['model.safetensors']
    plan.update(model=f"post_training_{family}_s42_{identity['model_sha256'][:12]}",
                step=identity['step'], identity=identity, code_sha256=code,
                checkpoint=str(run / 'checkpoint'), retain_checkpoint=True,
                selection='confirmed_family; predetermined seed42; benchmark test not used for selection',
                selection_sha256=sha(decision), created_at_utc=datetime.now(timezone.utc).isoformat())
    write(run / 'plan.json', plan)
    reference = run / 'pairing-reference'
    reference.mkdir()
    refplan = json.loads((TEMPLATE / 'pairing-reference/plan.json').read_text())
    refcode = {path: sha(path) for path in refplan['source_sha256']
               if not Path(path).is_relative_to(TEMPLATE)}
    refcode.update({str(path): sha(path) for path in
                    [run / 'pairing_reference.py', ROOT / 'downstream/post_training_eval.py',
                     ROOT / 'downstream/fixed_chain_pairing.py']})
    refplan.update(checkpoint=str(run / 'checkpoint'), identity=identity, protocol=pairing_protocol(),
                   model_sha256=identity['model_sha256'], source_sha256=refcode,
                   baseline_metrics=str(run / 'pairing/metrics.json'),
                   baseline_predictions=str(run / 'pairing/predictions.jsonl'))
    write(reference / 'plan.json', refplan)
    checks = [
        [str(ROOT / 'scripts/downstream/check_post_training_eval.py'), '--run', str(run / 'cpu_gate'), '--device', 'cpu'],
        [str(run / 'run_suite.py'), '--run', str(run), '--check'],
        [str(run / 'pairing_reference.py'), '--run', str(reference), '--device', 'cpu', '--check']]
    for index, command in enumerate(checks):
        with (run / f'prepare_check_{index}.log').open('x') as log:
            result = subprocess.run([sys.executable, '-B', *command], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError(f'CPU adapter check {index} failed; inspect log before submission')
    configs = []
    for group in range(10):
        config = yaml.safe_load((ROOT / f'eval_jobs/fixed_chain_75m_best112000_g{group}_20261014.yml').read_text())
        config['TaskName'] = f'post-training-{family.replace("_", "-")}-s42-r{args.attempt}-g{group}-20261014'
        config['Description'] = 'Confirmed family, predetermined seed42; 36-task AB/TCR suite; reference pairing uses AirGen Gumbel/margin v2'
        config['Tags'] = ['immune', 'post-training', 'downstream', family]
        config['Entrypoint'] = config['Entrypoint'].replace(str(TEMPLATE), str(run)).replace(
            'scripts/downstream/check_fixed_chain_eval.py', 'scripts/downstream/check_post_training_eval.py')
        path = ROOT / f'eval_jobs/post_training_{family}_s42_r{args.attempt}_g{group}_20261014.yml'
        with path.open('x') as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        configs.append(str(path))
    write(run / 'preparation.json', {'status': 'cpu_passed_not_submitted', 'run': str(run),
          'decision': str(decision), 'decision_sha256': sha(decision), 'identity': identity,
          'tasks': len(plan['tasks']), 'configs': configs})
    print(json.dumps({'run': str(run), 'status': 'cpu_passed_not_submitted', 'configs': configs}, indent=2))


if __name__ == '__main__':
    main()
