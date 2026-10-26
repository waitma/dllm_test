"""Freeze eight pilot representation diagnostics without enqueueing.

Activate pllm/protenix_abtcr; python -m scripts.research.prepare_fixed_chain_representation_probe
Uses train/screen only. Final benchmark and confirmation data remain unopened.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from scripts.research.run_fixed_chain_post_training import sha

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/fixed_chain_exploration_20261014'


def main():
    candidates = {'parent112000': RUN / 'base_checkpoint',
                  **{recipe: RUN / 'screen_r3' / recipe / 'export'
                     for recipe in ('continued', 'task_mix', 'contrastive', 'lora')},
                  **{recipe: RUN / 'screen_r5' / recipe / 'export'
                     for recipe in ('binding_aux', 'observed_cdr', 'observed_cdr_binding')}}
    models = {}
    for name, checkpoint in candidates.items():
        files = {file: sha(checkpoint / file) for file in
                 ('model.safetensors', 'immune_config.json', 'research_config.json', 'vocabulary.json')}
        if name != 'parent112000':
            receipt = json.loads((checkpoint / 'export_verification.json').read_text())
            if not receipt['roundtrip_logits_exact'] or any(receipt['files'][k] != v for k, v in files.items()):
                raise ValueError('Export was not verified')
        models[name] = {'checkpoint': str(checkpoint), 'files': files,
                        'parent': name == 'parent112000', 'step': 112000 if name == 'parent112000' else 1000}
    bundle = ROOT / 'output/v5_full_binding_20261007/launch_fixed_chain_representation_probe_r1_20261014'
    source = bundle / 'source'
    bundle.mkdir(exist_ok=False)
    source.mkdir()
    for name in ('dllm', 'scripts', 'examples'):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source / 'downstream').mkdir()
    shutil.copy2(ROOT / 'downstream/fixed_chain_eval.py', source / 'downstream/fixed_chain_eval.py')
    launcher = source / 'scripts/research/launch_v5_direct.py'
    shutil.copy2(ROOT / 'scripts/research/launch_fixed_chain_exploration.py', launcher)
    (source / 'trial_entry.py').write_text('''"""Frozen representation probe entrypoint."""
import sys
import runpy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
runpy.run_module('scripts.research.run_fixed_chain_representation_probe', run_name='__main__')
''')
    out = RUN / 'representation_probe_r1'
    out.mkdir(exist_ok=False)
    manifest = RUN / 'data/manifest.json'
    plan = {'schema': 'fixed_chain.representation_probe.v1', 'models': models,
            'recipes': {name: {} for name in models}, 'output': str(out),
            'data_manifest': str(manifest), 'data_manifest_sha256': sha(manifest),
            'source_hashes': {str(p): sha(p) for p in source.rglob('*.py')},
            'training_rows': 'up to256 per source/class, training only',
            'head': 'standardized logistic regression, C chosen on training receptor-cluster split',
            'screen': '512 labelled screening records; report per-source AUROC and shuffled-label control',
            'confirmation_opened': False, 'benchmark_test_opened': False}
    path = bundle / 'plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    (bundle / 'direct_launch.json').write_text(json.dumps({
        'runtime_dir': str(RUN / 'representation_probe_r1_runtime'), 'plan_sha256': sha(path),
        'purpose': 'Frozen representation pilot diagnostics; no foundation-model updates'}, indent=2) + '\n')
    check = subprocess.run([sys.executable, '-I', '-B', str(source / 'trial_entry.py'), '--help'],
                           cwd='/tmp', capture_output=True, text=True)
    (bundle / 'isolated_import.log').write_text(check.stdout + check.stderr)
    if check.returncode:
        raise ValueError('Frozen representation probe import failed')
    tests = subprocess.run([sys.executable, '-I', '-B', '-c',
        "import sys;sys.path.insert(0,sys.argv[1]);import pytest;raise SystemExit(pytest.main(sys.argv[2:]+['-q']))",
        str(source), str(source / 'scripts/tests/immune_llada/test_fixed_chain_representation_probe.py')],
        cwd='/tmp', capture_output=True, text=True)
    (bundle / 'isolated_tests.log').write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise ValueError('Frozen representation probe tests failed')
    (RUN / 'prepared_representation_probe_r1_ticket.json').write_text(json.dumps({
        'bundle': str(bundle), 'launcher_sha256': sha(launcher),
        'purpose': 'User-authorized architecture/post-training representation screening; eight models'}, indent=2) + '\n')
    print(json.dumps({'bundle': str(bundle), 'models': len(models), 'status': 'prepared_not_enqueued'}))


if __name__ == '__main__':
    main()
