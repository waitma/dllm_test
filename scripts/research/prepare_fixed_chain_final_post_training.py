"""Freeze final full-data training after a verified confirmation decision.

Activate pllm/protenix_abtcr; python -m scripts.research.prepare_fixed_chain_final_post_training
    --decision /absolute/confirmation/selection.json --attempt 1
Prepares three native segments (50,100,10000) sharing one source/schedule binding.
It does not enqueue any segment; each continuation needs the preceding complete checkpoint.
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
OLD = ROOT / 'output/v5_full_binding_20261007/launch_fixed_chain_75m_20261012'
ADAPTERS = ('output/generic_eight_rank_trainer_integration_20261002/eight_rank_trainer_adapter_v1.py',
            'output/generic_eight_worker_group_integration_20261002/eight_rank_checkpoint_adapter_v2.py')
CONTRACT = ROOT / 'output/generic_eight_rank_integration_20261002/eight_rank_contracts_v1.py'


def verify_decision(path):
    decision = json.loads(path.read_text())
    if decision.get('status') != 'confirmation_analyzed_not_final_training_deployed':
        raise ValueError('A completed confirmation decision is required')
    binding = decision['analysis_binding']
    ref = binding['evaluation_plan']
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('Confirmation plan changed')
    plan = json.loads(Path(ref['path']).read_text())
    analyzer = Path(ref['path']).parent / 'source/scripts/research/analyze_fixed_chain_confirmation.py'
    if (plan['source_hashes'].get(str(analyzer)) != binding['analyzer_sha256']
            or sha(analyzer) != binding['analyzer_sha256']):
        raise ValueError('Decision did not use the frozen analyzer')
    if sha(plan['protocol']) != binding['protocol_sha256']:
        raise ValueError('Confirmation protocol changed')
    for file, expected in binding['inputs'].items():
        if sha(file) != expected:
            raise ValueError('Confirmation evidence changed')
    for name in plan['models']:
        status = Path(plan['output']) / name / 'status.json'
        detail = json.loads(status.read_text())
        required = [status.parent / filename for filename in ('status.json', 'metrics.json', 'generation.json')]
        if (any(str(file) not in binding['inputs'] for file in required) or detail['status'] != 'success'
                or detail['model_sha256'] != plan['models'][name]['files']['model.safetensors']):
            raise ValueError('Confirmation evidence incomplete')
    protocol = json.loads(Path(plan['protocol']).read_text())
    family = decision['selected_family']
    if family not in protocol['families'] or family not in (
            'continued', 'task_mix', 'observed_eos', 'observed_cdr', 'lora',
            'contrastive', 'binding_aux', 'observed_cdr_binding'):
        raise ValueError('Unsupported or unconfirmed final recipe')
    if family == 'continued':
        if any(item['qualifies'] for item in decision['results'].values()):
            raise ValueError('Conservative fallback conflicts with qualifying candidates')
    elif not decision['results'][family]['qualifies']:
        raise ValueError('Selected intervention did not meet confirmation criteria')
    return family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    decision_path = args.decision.resolve()
    recipe = verify_decision(decision_path)
    index = RUN / 'full_data_index/manifest.json'
    if json.loads(index.read_text())['status'] != 'ready':
        raise ValueError('Full-data row indices are not ready')
    if json.loads((RUN / 'full_data_loader_preflight.json').read_text())['status'] != 'passed':
        raise ValueError('Full-data loader preflight is missing')
    parent, pilot = RUN / 'base_checkpoint', RUN / 'data/manifest.json'
    bundles = [ROOT / f'output/v5_full_binding_20261007/launch_fixed_chain_final_post_training_r{args.attempt}_stage{i}_20261014'
               for i in (1, 2, 3)]
    output = ROOT / f'output/immune_fixed_chain_post_training_final_r{args.attempt}_20261014'
    if output.exists() or any(p.exists() for p in bundles):
        raise ValueError('Refusing to overwrite a final training preparation')
    for bundle in bundles:
        (bundle / 'source/scripts/research').mkdir(parents=True)
    source = bundles[0] / 'source'
    for name in ('dllm', 'scripts', 'examples'):
        shutil.copytree(ROOT / name, source / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ADAPTERS:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    launchers = []
    for bundle in bundles:
        launcher = bundle / 'source/scripts/research/launch_v5_direct.py'
        shutil.copy2(ROOT / 'scripts/research/launch_fixed_chain_post_training_formal.py', launcher)
        launchers.append(launcher)
    files = {str(p): sha(p) for p in source.rglob('*.py')}
    files.update({str(p): sha(p) for p in launchers})
    for p in [CONTRACT, *[ROOT / 'scripts/research' / name for name in
              ('check_fixed_chain_parent.py', 'fixed_chain_data_binding.py', 'v5_data_binding.py')]]:
        files[str(p)] = sha(p)
    manifest = bundles[0] / 'source_manifest.json'
    manifest.write_text(json.dumps({'files': files}, indent=2) + '\n')
    source_argv = json.loads((OLD / 'launch_training_args.json').read_text())
    data_binding = OLD / 'data_binding.json'
    changes = {'--source-root': source, '--source-manifest': manifest,
               '--source-manifest-sha256': sha(manifest), '--output-dir': output,
               '--data-binding': data_binding, '--data-binding-sha256': sha(data_binding),
               '--seed': 42, '--micro-batch': 64, '--grad-accum': 1, '--global-batch': 512,
               '--max-steps': 10000, '--warmup-steps': 200, '--learning-rate': 2e-5,
               '--eval-every': 500, '--save-every': 500, '--valid-per-source': 128}
    for name, value in changes.items():
        source_argv[source_argv.index(name)+1] = str(value)
    extras = {'--post-recipe': recipe, '--init-from-checkpoint': parent,
              '--init-model-sha256': sha(parent / 'model.safetensors'),
              '--post-data-index': index, '--post-data-index-sha256': sha(index),
              '--pilot-manifest': pilot, '--pilot-manifest-sha256': sha(pilot)}
    source_argv += [str(item) for pair in extras.items() for item in pair]
    tickets = []
    for stage, (bundle, launcher) in enumerate(zip(bundles, launchers), 1):
        argv = list(source_argv)
        stop = 50 if stage == 1 else 100 if stage == 2 else 0
        if '--stop-after' in argv:
            argv[argv.index('--stop-after')+1] = str(stop)
        else:
            argv += ['--stop-after', str(stop)]
        if stage > 1:
            argv += ['--resume-from-checkpoint', str(output / f'checkpoint-{50 if stage==2 else 100}')]
        argfile = bundle / 'launch_training_args.json'
        argfile.write_text(json.dumps(argv, indent=2) + '\n')
        launch = {'schema': 'fixed_chain.final_post_training.launch.v1',
                  'initialization': 'weights112000' if stage == 1 else 'native_full_state_resume',
                  'runtime_dir': str(RUN / f'final_r{args.attempt}_stage{stage}_runtime'),
                  'arguments_sha256': sha(argfile), 'launcher_sha256': sha(launcher),
                  'decision': str(decision_path), 'decision_sha256': sha(decision_path),
                  'recipe': recipe, 'global_batch': 512, 'max_steps': 10000,
                  'purpose': 'Real selected full-data training, fixed horizon across resume verification segments'}
        (bundle / 'direct_launch.json').write_text(json.dumps(launch, indent=2) + '\n')
        tickets.append({'bundle': str(bundle), 'launcher_sha256': sha(launcher),
                        'purpose': f'Confirmed {recipe} final full-data post-training; native segment {stage}'})
    test_files = ('test_fixed_chain_formal_post_training.py', 'test_fixed_chain_post_training.py',
                  'test_fixed_chain_validation.py', 'test_source_balanced_training.py')
    tests = subprocess.run([sys.executable, '-I', '-B', '-c',
        "import sys;sys.path.insert(0,sys.argv[1]);import pytest;raise SystemExit(pytest.main(sys.argv[2:]+['-q']))",
        str(source), *[str(source / 'scripts/tests/immune_llada' / name) for name in test_files]],
        cwd='/tmp', capture_output=True, text=True)
    (bundles[0] / 'isolated_tests.log').write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise ValueError('Frozen final-training tests failed')
    check = subprocess.run([sys.executable, '-I', '-B', str(launchers[0]),
                            '--bundle', str(bundles[0]), '--check-only'],
                           cwd='/tmp', capture_output=True, text=True)
    (bundles[0] / 'metadata_check.log').write_text(check.stdout + check.stderr)
    if check.returncode:
        raise ValueError('Final launch metadata validation failed')
    receipt = {'status': 'prepared_not_enqueued', 'recipe': recipe, 'output': str(output),
               'decision': str(decision_path), 'tickets': tickets,
               'stage_rule': 'Enqueue each stage only after previous complete checkpoint; all share one native schedule/source binding.'}
    (RUN / f'prepared_final_r{args.attempt}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
