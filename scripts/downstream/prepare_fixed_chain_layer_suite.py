"""Prepare local fixed-chain50000 block-two tests without submitting any job.

Activate pllm then protenix_abtcr, and run from the repository root:
python -B scripts/downstream/prepare_fixed_chain_layer_suite.py
Runs the real-weight CPU gate and writes a local launcher; does not launch it.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from downstream.fixed_chain_layer_eval import sha


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-preflight', action='store_true')
    args = parser.parse_args()
    baseline = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013'
    reference = ROOT/'output/downstream_generation/post_training_lora_s42_layer2_20261017'
    run = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_layer2_20261018'
    if args.resume_preflight:
        assert run.exists() and not (run/'local_execution.json').exists()
        assert not list(run.glob('*/status.json')), 'Evaluation already started'
        archive = run/'preflight_r1'
        archive.mkdir(exist_ok=False)
        for name in ('plan.json', 'prepare_check.log', 'run_suite.py'):
            shutil.copy2(run/name, archive/name)
        (run/'prepare_check.log').rename(archive/'original_prepare_check.log')
    else:
        run.mkdir(exist_ok=False)
    groups = [['gdp_a1', 'specificity', 'm396'], ['t1-cdr3b'],
              ['t1-others_cdr3b', 't1-cdr3ab', 't1-others_longab'], ['uptcr'],
              ['t2a', 't3broad', 't3deep'], ['ag-binding']]
    tasks = [task for group in groups for task in group]
    prior = json.loads((reference/'plan.json').read_text())
    old = json.loads((baseline/'plan.json').read_text())
    comparison = {}
    for task in tasks:
        assert json.loads((baseline/task/'status.json').read_text())['status'] == 'success', task
        paths = [p for pattern in ('metrics.json', 'summary.json', 'fewshot.json')
                 for p in (baseline/task).rglob(pattern)]
        location = baseline/task/'result_location.json'
        if location.exists():
            folder = Path(json.loads(location.read_text())['path'])
            paths += [p for pattern in ('metrics.json', 'summary.json', 'fewshot.json')
                      for p in folder.rglob(pattern)]
        assert paths, task
        comparison[task] = {str(p): sha(p) for p in paths}
    data = {p: sha(p) for p in prior['data_sha256']}
    assert data == prior['data_sha256'], 'Task data changed since prior layer suite'
    audit_binding = prior['input_audit_binding']
    assert sha(audit_binding['path']) == audit_binding['sha256']
    audit = json.loads(Path(audit_binding['path']).read_text())
    assert audit['status'] == 'passed'
    for task in tasks:
        keys = ['uptcr-'+s for s in ('majority', 'few-shot', 'zero-shot')] if task == 'uptcr' else [task]
        for key in keys:
            assert audit['checks'][key]['over_capacity'] == 0
    if not args.resume_preflight:
        shutil.copytree(baseline/'checkpoint', run/'checkpoint')
    shutil.copy2(ROOT/'scripts/downstream/run_fixed_chain_layer_suite.py', run/'run_suite.py')
    receipt = json.loads((run/'checkpoint/formal_checkpoint.json').read_text())
    assert receipt['step'] == 50000
    digest = receipt['files']['model.safetensors']
    assert sha(run/'checkpoint/model.safetensors') == digest
    model_identity = dict(step=50000, model_sha256=digest,
                          layer_index_one_based=2, readout='block2_pre_final_norm')
    source_paths = {Path(p) for p in old['code_sha256']
                    if Path(p).is_relative_to(ROOT) and not Path(p).is_relative_to(ROOT/'output')}
    source_paths.update(ROOT/p for p in (
        'downstream/fixed_chain_eval.py', 'downstream/fixed_chain_layer_eval.py',
        'scripts/downstream/run_fixed_chain_layer_suite.py',
        'scripts/downstream/run_local_fixed_layer_suite.py',
        'scripts/downstream/prepare_fixed_chain_layer_suite.py',
        'scripts/downstream/run_v5_suite_task.py', 'downstream/grammar/ab_probes.py',
        'downstream/grammar/tcr_features.py', 'downstream/benchmark/tcr_binding/query_protocol.py',
        'downstream/benchmark/tcr_binding/run_uptcr_ours.py'))
    source_paths.add(run/'run_suite.py')
    code = {str(p): sha(p) for p in sorted(source_paths)}
    write(run/'plan.json', dict(tasks=tasks, groups=groups, step=50000,
          model='fixed_chain_75m_step50000_block2', model_sha256=digest, identity=model_identity,
          checkpoint=str(run/'checkpoint'), readout='block2_pre_final_norm', retain_checkpoint=True,
          execution_mode='local_single_gpu', baseline_run=str(baseline),
          baseline_result_sha256=comparison, code_sha256=code, data_sha256=data,
          input_audit_binding=audit_binding,
          created_utc=datetime.now(timezone.utc).isoformat(),
          protocol_notes=['Fixed block2, pre-final-norm; frozen backbone; original pooling and probe budgets.',
                          'Same input data and task splits as the prior LoRA block2 experiment.',
                          'Compare to this fixed50000 checkpoint final-layer results, not LoRA baseline.',
                          'iter=1 is generation-only; representation does not sample.',
                          'CoV binding probe changes embedding only; PLL remains complete-model.',
                          'SPR excluded: existing probe is disabled for units/censoring; no meaningful layer comparison.']))
    with (run/'prepare_check.log').open('x') as log:
        result = subprocess.run([sys.executable, '-B', str(run/'run_suite.py'),
                                 '--run', str(run), '--check', '--device', 'cpu'],
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('CPU gate failed; see '+str(run/'prepare_check.log'))
    old_launcher = (reference/'run_local.sh').read_text()
    launcher = old_launcher[:old_launcher.index('exec python -B ')]
    launcher = launcher.replace(str(reference), str(run)).replace(reference.name, run.name)
    launcher = launcher.replace('TCR_EVAL_STEP=3000', 'TCR_EVAL_STEP=50000')
    launcher = launcher.replace(prior['model_sha256'], digest)
    driver = ROOT/'scripts/downstream/run_local_fixed_layer_suite.py'
    launcher += 'exec python -B '+shlex.quote(str(driver))+' --run '+shlex.quote(str(run))+'\n'
    (run/'run_local.sh').write_text(launcher)
    for path in source_paths:
        if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'output'):
            destination = run/'source'/path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    write(run/'preparation.json', dict(status='cpu_passed_ready_local', execution_mode='local_single_gpu',
          tasks=tasks, baseline=str(baseline), driver_sha256=sha(driver),
          launcher_sha256=sha(run/'run_local.sh'), no_remote_submission=True))
    print(json.dumps(dict(run=str(run), status='cpu_passed_ready_local', tasks=tasks), indent=2))


if __name__ == '__main__':
    main()
