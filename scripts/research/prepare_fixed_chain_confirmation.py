"""Freeze completed candidates, protocol and confirmation evaluator without enqueueing.

Activate pllm/protenix_abtcr; python -m scripts.research.prepare_fixed_chain_confirmation
    --training-bundle /absolute/screen_r7_bundle --attempt 1
This builder checks metadata and exports; it never opens reserved confirmation rows.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from scripts.research.run_fixed_chain_post_training import sha

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/fixed_chain_exploration_20261014'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-bundle', type=Path, required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    training = json.loads((args.training_bundle / 'plan.json').read_text())
    protocol_path = RUN / 'selection_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['families'] != list(training['recipes']):
        raise ValueError('Confirmation family list differs from frozen training plan')
    if protocol['steps'] != training['steps']:
        raise ValueError('Confirmation step binding differs')
    if protocol['bootstrap']['interval'] != [.05 / (2*(len(protocol['families'])-1)),
                                             1-.05 / (2*(len(protocol['families'])-1))]:
        raise ValueError('Multiplicity interval does not match candidate count')
    models = {}
    for seed in protocol['seeds']:
        child_path = args.training_bundle / f'plan_seed{seed}.json'
        child = json.loads(child_path.read_text())
        for family in protocol['families']:
            trial = Path(child['output']) / family
            status = json.loads((trial / 'status.json').read_text())
            if status['status'] != 'success' or status['step'] != protocol['steps']:
                raise ValueError('Candidate not complete: ' + str(trial))
            export = trial / 'export'
            receipt = json.loads((export / 'export_verification.json').read_text())
            if (receipt['checkpoint_sha256'] != status['checkpoint_sha256']
                    or receipt['training_plan_sha256'] != sha(child_path)
                    or 'training_binding.json' not in receipt['files']):
                raise ValueError('Candidate export binding differs')
            for filename, expected in receipt['files'].items():
                if sha(export / filename) != expected:
                    raise ValueError('Export changed')
            models[f'{family}_seed{seed}'] = {'checkpoint': str(export), 'recipe': family,
                                             'seed': seed, 'files': receipt['files']}
    parent = Path(training['parent'])
    models['parent112000'] = {'checkpoint': str(parent), 'parent': True,
                             'files': {name: sha(parent / name) for name in
                             ('model.safetensors', 'immune_config.json', 'research_config.json', 'vocabulary.json')}}
    tag = f'confirmation_r{args.attempt}'
    bundle = ROOT / f'output/v5_full_binding_20261007/launch_fixed_chain_{tag}_20261014'
    source = bundle / 'source'
    bundle.mkdir(exist_ok=False)
    source.mkdir()
    for name in ('dllm', 'scripts', 'examples'):
        origin = ROOT if name == 'scripts' else args.training_bundle / 'source'
        shutil.copytree(origin / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Models are evaluated with identical implementation files to their training.
    training_source = args.training_bundle / 'source'
    for file in (training_source / 'dllm').rglob('*.py'):
        if sha(file) != sha(source / file.relative_to(training_source)):
            raise ValueError('Evaluation model implementation differs: ' + str(file))
    launcher = source / 'scripts/research/launch_v5_direct.py'
    shutil.copy2(ROOT / 'scripts/research/launch_fixed_chain_exploration.py', launcher)
    (source / 'trial_entry.py').write_text('''"""Frozen confirmation entrypoint; invoked by the persistent worker pool."""
import sys
import runpy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
runpy.run_module('scripts.research.run_fixed_chain_confirmation', run_name='__main__')
''')
    out = RUN / tag
    out.mkdir(exist_ok=False)
    # Copy the protocol so later research cannot alter this confirmation decision.
    frozen_protocol = bundle / 'selection_protocol.json'
    shutil.copy2(protocol_path, frozen_protocol)
    plan = {'schema': 'fixed_chain.confirmation.v1', 'status': 'frozen', 'models': models,
            'recipes': {name: {} for name in models}, 'output': str(out),
            'protocol': str(frozen_protocol), 'protocol_sha256': sha(frozen_protocol),
            'data_manifest': training['data_manifest'],
            'data_manifest_sha256': training['data_manifest_sha256'],
            'source_hashes': {str(p): sha(p) for p in source.rglob('*.py')}}
    path = bundle / 'plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    (out / 'evaluation_plan.json').write_text(json.dumps({
        'path': str(path), 'sha256': sha(path)}, indent=2) + '\n')
    (bundle / 'direct_launch.json').write_text(json.dumps({
        'runtime_dir': str(RUN / f'{tag}_runtime'), 'plan_sha256': sha(path),
        'purpose': 'Frozen confirmation evaluation; no optimizer updates'}, indent=2) + '\n')
    check = subprocess.run([sys.executable, '-I', '-B', str(source / 'trial_entry.py'), '--help'],
                           cwd='/tmp', capture_output=True, text=True)
    (bundle / 'isolated_import.log').write_text(check.stdout + check.stderr)
    if check.returncode:
        raise ValueError('Frozen confirmation import failed')
    tests = subprocess.run([sys.executable, '-I', '-B', '-c',
        "import sys;sys.path.insert(0,sys.argv[1]);import pytest;raise SystemExit(pytest.main(sys.argv[2:]+['-q']))",
        str(source), *[str(source / 'scripts/tests/immune_llada' / name) for name in
        ('test_fixed_chain_confirmation.py', 'test_fixed_chain_post_training.py',
         'test_fixed_chain_preference.py', 'test_fixed_chain_validation.py')]],
        cwd='/tmp', capture_output=True, text=True)
    (bundle / 'isolated_tests.log').write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise ValueError('Frozen confirmation tests failed')
    ticket = {'bundle': str(bundle), 'launcher_sha256': sha(launcher),
              'purpose': 'User-authorized held-out confirmation for fifteen models and their parent'}
    (RUN / f'prepared_{tag}_ticket.json').write_text(json.dumps(ticket, indent=2) + '\n')
    print(json.dumps({'bundle': str(bundle), 'models': len(models), 'status': 'prepared_not_enqueued',
                      'confirmation_rows_opened': False}))


if __name__ == '__main__':
    main()
