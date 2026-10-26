"""Prepare one diffusion checkpoint for the existing 15-task single-GPU suite.

Activate pllm/protenix_abtcr; python -B THIS_FILE --checkpoint ABS --run ABS.
Copies inference weights and permanently migrates historical manifest paths.
Run the resulting driver with --cpu-check before submitting the shell entrypoint.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
STORAGE = Path('/project_c20250601/251105016')
HISTORICAL_STORAGE = Path('/vepfs-mlp2/c20250601/251105016')
REFERENCE = ROOT / 'output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020'
sys.path.insert(0, str(ROOT))
from downstream.bert_full_token_eval import sha
from scripts.downstream.prepare_bert_checkpoint_suite import snapshot
from scripts.downstream.run_bert_representation_suite import write


def migrated(value):
    """Rewrite the new artifact at preparation time; never resolve an old mount."""
    if isinstance(value, dict):
        return {migrated(k): migrated(v) for k, v in value.items()}
    if isinstance(value, list):
        return [migrated(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(HISTORICAL_STORAGE), str(STORAGE))
    return value


def checked_asset(historical_path, expected, changes):
    path = Path(migrated(historical_path))
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        historical = content.replace(str(STORAGE).encode(), str(HISTORICAL_STORAGE).encode())
        if hashlib.sha256(historical).hexdigest() != expected:
            raise ValueError('Reference asset changed beyond its storage paths: ' + str(path))
        changes[str(path)] = dict(previous_sha256=expected, sha256=actual, change='storage_paths_only')
    return path, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'plan.json').exists():
        raise FileExistsError('Already prepared; use driver.py --cpu-check or the shell entrypoint')
    original = json.loads((REFERENCE / 'plan.json').read_text())
    changes, data, code = {}, {}, {}
    for path, expected in original['data_sha256'].items():
        current, digest = checked_asset(path, expected, changes)
        data[str(current)] = digest
    for path, expected in original['code_sha256'].items():
        current, digest = checked_asset(path, expected, changes)
        if not current.is_relative_to(ROOT / 'output'):
            code[str(current)] = digest
    scoring = json.loads((REFERENCE / 'scoring_dependency_check.json').read_text())
    for assets in scoring['assets_sha256'].values():
        for path, expected in assets.items():
            current, digest = checked_asset(path, expected, changes)
            data[str(current)] = digest
    receipt = snapshot(args.checkpoint.resolve(), run)
    for filename in ('immune_config.json', 'vocabulary.json'):
        if sha(run / 'checkpoint' / filename) != sha(REFERENCE / 'checkpoint' / filename):
            raise ValueError('The reference protocols require the same native layout/vocabulary')
    research = json.loads((run / 'checkpoint/research_config.json').read_text())
    if research['schema'] != 'immune.fixed_chain.conditional_diffusion_edit.v1':
        raise ValueError('Expected a conditional diffusion/edit checkpoint')
    shutil.copy2(ROOT / 'scripts/downstream/run_diffusion_edit_suite.py', run / 'run_suite.py')
    shutil.copy2(ROOT / 'scripts/downstream/run_diffusion_single_gpu.py', run / 'driver.py')
    (run / 'airgen_evaluate.py').write_text(migrated((REFERENCE / 'airgen_evaluate.py').read_text()))
    for path in (Path(__file__).resolve(), ROOT / 'scripts/downstream/run_diffusion_single_gpu.py',
                 run / 'run_suite.py', run / 'driver.py', run / 'airgen_evaluate.py',
                 ROOT / 'scripts/downstream/eval_diffusion_best2500_single_gpu.sh',
                 ROOT / 'scripts/downstream/prepare_bert_checkpoint_suite.py',
                 ROOT / 'scripts/downstream/run_bert_representation_suite.py',
                 ROOT / 'scripts/downstream/report_v5_suite.py',
                 ROOT / 'downstream/grammar/ab_fv_input.py'):
        code[str(path)] = sha(path)
    plan = migrated(original)
    plan.update(created_utc=datetime.now(timezone.utc).isoformat(),
                model=f"diffusion_edit_v7_75.6M_step{receipt['step']}", step=receipt['step'],
                model_sha256=receipt['files']['model.safetensors'], checkpoint=str(run / 'checkpoint'),
                selection='Best frozen internal validation loss; fixed before downstream evaluation',
                execution='one allocated GPU; sequential task subprocesses; no automatic submission',
                groups=[plan['tasks']], code_sha256=code, source_sha256=code, data_sha256=data,
                comparison_runs=[str(REFERENCE)], comparison_plan_sha256=sha(REFERENCE / 'plan.json'),
                historical_source_differences=changes, runtime_source_binding='verified live evaluation files; checkpoint-bound model sources',
                scoring_models={name: str(ROOT / 'data/downstream/immunomatch' / f'immunomatch-{name}')
                                for name in ('kappa', 'lambda')})
    if len(plan['tasks']) != 15 or len(set(plan['tasks'])) != 15:
        raise ValueError('Expected the established 15-task suite')
    for task in plan['tasks']:
        (run / task).mkdir(exist_ok=True)
    (run / 'single_gpu').mkdir(exist_ok=True)
    write(run / 'plan.json', plan)
    audit = migrated(json.loads((REFERENCE / 'input_audit.json').read_text()))
    if audit['status'] != 'passed' or any(v['over_capacity'] for v in audit['checks'].values()):
        raise ValueError('Reference input audit failed')
    audit.update(checkpoint_step=receipt['step'], model_sha256=plan['model_sha256'],
                 reference=str(REFERENCE / 'input_audit.json'),
                 reference_sha256=sha(REFERENCE / 'input_audit.json'))
    write(run / 'input_audit.json', audit)
    selection = json.loads((run / 'selection.json').read_text())
    selection['selection'] = plan['selection']
    write(run / 'selection.json', selection)
    write(run / 'storage_migration.json', dict(status='passed', reference=str(REFERENCE),
          metadata_migration_only=True, current_storage=str(STORAGE), changed_assets=changes,
          data_files=len(data), code_files=len(code), plan_sha256=sha(run / 'plan.json')))
    print(json.dumps(dict(status='prepared_needs_cpu_check', run=str(run), step=receipt['step'],
                          tasks=len(plan['tasks']), plan_sha256=sha(run / 'plan.json'))), flush=True)


if __name__ == '__main__':
    main()
