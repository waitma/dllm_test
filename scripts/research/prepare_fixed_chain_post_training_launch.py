"""Freeze a screening bundle with isolated tests; does not enqueue it.

Activate pllm/protenix_abtcr; python -m scripts.research.prepare_fixed_chain_post_training_launch --attempt 3
"""
import json
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from dllm.pipelines.immune_llada.fixed_chain.post_training import RECIPES
from scripts.research.run_fixed_chain_post_training import sha

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/fixed_chain_exploration_20261014'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--recipes', nargs='+', choices=tuple(RECIPES), default=list(RECIPES))
    parser.add_argument('--semantic-validation', action='store_true')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', nargs='+', type=int, help='Independent seeds packed across eight workers')
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    args = parser.parse_args()
    if args.seeds and len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate seeds')
    tag = f'screen_r{args.attempt}'
    BUNDLE = ROOT / f'output/v5_full_binding_20261007/launch_fixed_chain_posttraining_{tag}_20261014'
    manifest = RUN / 'data/manifest.json'
    data = json.loads(manifest.read_text())
    parent = RUN / 'base_checkpoint'
    provenance = json.loads((parent / 'run_provenance.json').read_text())
    if data['parent_manifest_sha256'] != provenance['parent_metadata']['dataset_manifest.json']:
        raise ValueError('Pilot parent differs from checkpoint training data')
    source = BUNDLE / 'source'
    BUNDLE.mkdir(exist_ok=False)
    source.mkdir()
    for name in ('dllm', 'scripts', 'examples'):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    launcher = source / 'scripts/research/launch_v5_direct.py'
    shutil.copy2(ROOT / 'scripts/research/launch_fixed_chain_exploration.py', launcher)
    (source / 'trial_entry.py').write_text('''"""Frozen trial entry; invoked by the persistent container launcher."""
import sys
import runpy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
runpy.run_module('scripts.research.run_fixed_chain_post_training', run_name='__main__')
''')
    plan = dict(schema='fixed_chain.posttraining.screen.v1', parent=str(parent),
                parent_hashes={str(p): sha(p) for p in parent.iterdir() if p.is_file()},
                data_manifest=str(manifest), data_manifest_sha256=sha(manifest),
                output=str(RUN / tag), seed=args.seed, recipes={key: RECIPES[key] for key in args.recipes},
                steps=args.steps, micro_batch=8, accumulation=4, learning_rate=args.learning_rate, warmup=50,
                semantic_validation=args.semantic_validation,
                source_hashes={str(p): sha(p) for p in source.rglob('*.py')},
                selection_status='not_selected', confirmation_evaluated=False)
    if any(RECIPES[key].get('preference_weight') for key in args.recipes):
        preference_manifest = RUN / 'preference_data/manifest.json'
        if json.loads(preference_manifest.read_text())['status'] != 'ready':
            raise ValueError('Preference data preflight not complete')
        plan['preference_data'] = {'manifest': str(preference_manifest),
                                  'manifest_sha256': sha(preference_manifest)}
    (RUN / tag).mkdir(exist_ok=False)
    if args.seeds:
        trials = []
        for seed in args.seeds:
            child = {**plan, 'seed': seed, 'output': str(RUN / tag / f'seed{seed}')}
            Path(child['output']).mkdir(exist_ok=False)
            child_path = BUNDLE / f'plan_seed{seed}.json'
            child_path.write_text(json.dumps(child, indent=2) + '\n')
            for recipe in args.recipes:
                trials.append({'id': f'{recipe}_seed{seed}', 'recipe': recipe,
                               'plan': str(child_path), 'sha256': sha(child_path)})
        plan['trial_plans'] = trials
    plan_path = BUNDLE / 'plan.json'
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    (BUNDLE / 'direct_launch.json').write_text(json.dumps(dict(
        runtime_dir=str(RUN / f'{tag}_runtime'), plan_sha256=sha(plan_path),
        purpose='Independent one-GPU trials, not eight-rank formal pretraining'), indent=2) + '\n')
    receipt = dict(bundle=str(BUNDLE), launcher_sha256=sha(launcher),
                   purpose=f'User-authorized fixed-chain architecture and post-training screening; {len(plan.get("trial_plans", args.recipes))} trials')
    check = subprocess.run([sys.executable, '-I', '-B', str(source / 'trial_entry.py'), '--help'],
                           cwd='/tmp', capture_output=True, text=True)
    (BUNDLE / 'isolated_import.log').write_text(check.stdout + check.stderr)
    if check.returncode:
        raise ValueError('Frozen entrypoint import failed; see isolated_import.log')
    tests = subprocess.run([sys.executable, '-I', '-B', '-c',
                            "import sys; sys.path.insert(0,sys.argv[1]); import pytest; raise SystemExit(pytest.main(sys.argv[2:]+['-q']))",
                            str(source), str(source / 'scripts/tests/immune_llada/test_fixed_chain_post_training.py'),
                            str(source / 'scripts/tests/immune_llada/test_fixed_chain_validation.py'),
                            str(source / 'scripts/tests/immune_llada/test_fixed_chain_preference.py')],
                           cwd='/tmp', capture_output=True, text=True)
    (BUNDLE / 'isolated_tests.log').write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise ValueError('Frozen CPU tests failed; see isolated_tests.log')
    (RUN / f'prepared_ticket_r{args.attempt}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'bundle': str(BUNDLE), 'trials': len(plan.get('trial_plans', args.recipes)), 'status': 'prepared_not_enqueued'}))


if __name__ == '__main__':
    main()
