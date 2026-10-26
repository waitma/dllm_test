"""Prepare a fixed BERT checkpoint for the established 13 representation tasks.

Activate pllm/protenix_abtcr and run from the repository:
python -B scripts/downstream/prepare_bert_checkpoint_suite.py \
    --checkpoint /absolute/checkpoint-107000 --run /absolute/new_evaluation_run
Preserves an inference snapshot, verifies the original protocol, runs the CPU
adapter check, and writes five single-GPU evaluation YAMLs without submitting.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from downstream.bert_full_token_eval import sha

REFERENCE = ROOT / 'output/downstream_generation/bert_full_token_75m_best48828_20261017'
INFERENCE_FILES = ('model.safetensors', 'research_config.json', 'immune_config.json',
                   'vocabulary.json', 'run_provenance.json', 'state.json')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def snapshot(source, run):
    """Copy immutable files before training retention can remove this checkpoint."""
    destination = run / 'checkpoint'
    if not destination.exists():
        temporary = run / 'checkpoint.tmp'
        temporary.mkdir(exist_ok=False)
        with ExitStack() as stack:
            # Open every file first; unlinking by retention cannot invalidate an open FD.
            streams = {name: stack.enter_context((source / name).open('rb'))
                       for name in (*INFERENCE_FILES, 'checkpoint_receipt.json')}
            for name, stream in streams.items():
                with (temporary / name).open('wb') as target:
                    shutil.copyfileobj(stream, target)
        validate_snapshot(temporary)
        temporary.rename(destination)
    receipt = validate_snapshot(destination)
    selection = run / 'selection.json'
    if selection.exists():
        previous = json.loads(selection.read_text())
        assert previous['checkpoint_source'] == str(source)
        assert previous['model_sha256'] == receipt['files']['model.safetensors']
    else:
        write(selection, dict(created_utc=datetime.now(timezone.utc).isoformat(),
              step=receipt['step'], checkpoint_source=str(source), checkpoint=str(destination),
              model_sha256=receipt['files']['model.safetensors'],
              selection='User-requested fixed training step; no downstream test selection',
              verified_receipt_files=True, all_rank_parameters_equal=True))
    return receipt


def validate_snapshot(directory):
    receipt = json.loads((directory / 'checkpoint_receipt.json').read_text())
    ranks = receipt['rank_parameter_sha256']
    assert receipt['status'] == 'complete' and len(ranks) == receipt['world_size'] == 8
    assert len(set(ranks)) == 1
    for name in INFERENCE_FILES:
        assert sha(directory / name) == receipt['files'][name], name
    assert json.loads((directory / 'state.json').read_text())['step'] == receipt['step']
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    args = parser.parse_args()
    run, reference, source = args.run.resolve(), args.reference.resolve(), args.checkpoint.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'plan.json').exists():
        raise FileExistsError('Evaluation already prepared; use its existing plan and entrypoint')
    receipt = snapshot(source, run)
    step = receipt['step']
    original = json.loads((reference / 'plan.json').read_text())
    for section in ('code_sha256', 'data_sha256'):
        for path, expected in original[section].items():
            assert sha(path) == expected, 'Reference asset changed: ' + path
    # Identical model layout and tokenization make the existing capacity audit reusable.
    for name in ('immune_config.json', 'research_config.json', 'vocabulary.json'):
        assert sha(run / 'checkpoint' / name) == sha(reference / 'checkpoint' / name), name
    provenance = json.loads((run / 'checkpoint/run_provenance.json').read_text())
    previous = json.loads((reference / 'checkpoint/run_provenance.json').read_text())
    assert provenance['sources'] == previous['sources']
    audit = json.loads((reference / 'input_audit.json').read_text())
    assert audit['status'] == 'passed'
    groups = [group for group in original['groups']
              if all(not task.startswith('ag-abbibench:') for task in group)]
    tasks = [task for group in groups for task in group]
    assert len(groups) == 5 and len(tasks) == 13
    checks = {}
    for task in tasks:
        names = ['uptcr-' + s for s in ('few-shot', 'majority', 'zero-shot')] if task == 'uptcr' else [task]
        for name in names:
            assert audit['checks'][name]['over_capacity'] == 0
            checks[name] = audit['checks'][name]
    write(run / 'input_audit.json', dict(status='passed', mode='verified_existing_data_and_identical_layout',
          reference=str(reference / 'input_audit.json'), reference_sha256=sha(reference / 'input_audit.json'),
          checkpoint_step=step, model_sha256=receipt['files']['model.safetensors'], checks=checks,
          data_sha256=original['data_sha256']))
    shutil.copy2(reference / 'run_suite.py', run / 'run_suite.py')
    code = {p: h for p, h in original['code_sha256'].items() if p != str(reference / 'run_suite.py')}
    code[str(run / 'run_suite.py')] = sha(run / 'run_suite.py')
    code[str(Path(__file__).resolve())] = sha(__file__)
    write(run / 'plan.json', dict(tasks=tasks, groups=groups, step=step,
          model=f'bert_full_token_v7_75.6M_step{step}', model_sha256=receipt['files']['model.safetensors'],
          checkpoint=str(run / 'checkpoint'), retain_checkpoint=True,
          selection='User-requested fixed checkpoint; not selected using downstream scores',
          scope='13 existing frozen-representation/binding task entries',
          excluded_tasks='AbBiBench zero-shot PLL, generation, T2B and baseline reruns',
          comparison_run=str(reference), comparison_plan_sha256=sha(reference / 'plan.json'),
          protocol_notes=original['protocol_notes'], code_sha256=code,
          data_sha256=original['data_sha256'], created_at_utc=datetime.now(timezone.utc).isoformat()))
    archive = run / 'source'
    for path in code:
        path = Path(path)
        if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'output'):
            target = archive / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    with (run / 'cpu_check.log').open('x') as log:
        result = subprocess.run([sys.executable, '-B', str(run / 'run_suite.py'), '--run', str(run),
                                 '--check', '--device', 'cpu'], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('CPU adapter check failed; inspect ' + str(run / 'cpu_check.log'))
    import yaml
    date = datetime.now(timezone.utc).strftime('%Y%m%d')
    template = yaml.safe_load((ROOT / 'eval_jobs/bert_full_token_best48828_g0_20261017.yml').read_text())
    configs = []
    for index, group in enumerate(groups):
        config = dict(template)
        config['TaskName'] = f'bert-full-token-75m-step{step}-g{index}-{date}'
        config['Description'] = f'Fixed BERT step{step} representation: ' + ','.join(group)
        config['Tags'] = ['immune', 'bert-full-token', 'representation', f'step{step}']
        entry = template['Entrypoint'].replace(str(reference), str(run)).replace(reference.name, run.name)
        entry = entry.replace('TCR_EVAL_STEP=48828', f'TCR_EVAL_STEP={step}')
        entry = entry.replace(original['model_sha256'], receipt['files']['model.safetensors'])
        entry = entry.replace('--group 0', f'--group {index}').replace('group_0.log', f'group_{index}.log')
        config['Entrypoint'] = entry
        path = ROOT / f'eval_jobs/bert_full_token_step{step}_g{index}_{date}.yml'
        with path.open('x') as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        configs.append(str(path))
    write(run / 'preparation.json', dict(status='cpu_passed_not_submitted', configs=configs,
          tasks=tasks, groups=groups, comparison_run=str(reference),
          cpu_gate=str(run / 'cpu_adapter_checks.json'), input_audit=str(run / 'input_audit.json'),
          created_at_utc=datetime.now(timezone.utc).isoformat()))
    subprocess.run([sys.executable, '-B', str(run / 'run_suite.py'), '--run', str(run), '--report'],
                   check=True, stdout=subprocess.DEVNULL)
    print(json.dumps(dict(run=str(run), step=step, tasks=len(tasks), configs=configs,
                         status='cpu_passed_not_submitted'), indent=2))


if __name__ == '__main__':
    main()
