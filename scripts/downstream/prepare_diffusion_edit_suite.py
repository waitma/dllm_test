"""Bind the current diffusion/edit checkpoint to established downstream protocols.

Activate pllm/protenix_abtcr; python -B THIS_FILE --run ABS --checkpoint ABS.
Creates the immutable snapshot/plan and CPU gates, without starting GPU work.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.downstream.prepare_bert_checkpoint_suite import snapshot
from downstream.bert_full_token_eval import sha
from scripts.downstream.run_bert_representation_suite import write

REPRESENTATION = ROOT / 'output/downstream_generation/bert_full_token_75m_step247000_20261019'
GENERATION = ROOT / 'output/downstream_generation/oas_only_75m_step11000_generation_20261018'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(exist_ok=True)
    if (run / 'plan.json').exists():
        raise FileExistsError('Use the existing frozen plan; do not overwrite an attempt')
    receipt = snapshot(args.checkpoint.resolve(), run)
    repr_plan = json.loads((REPRESENTATION / 'plan.json').read_text())
    gen_plan = json.loads((GENERATION / 'plan.json').read_text())
    for name in ('immune_config.json', 'vocabulary.json'):
        assert sha(run / 'checkpoint' / name) == sha(REPRESENTATION / 'checkpoint' / name), name
    research = json.loads((run / 'checkpoint/research_config.json').read_text())
    assert research['schema'] == 'immune.fixed_chain.conditional_diffusion_edit.v1'
    repr_tasks = [t for t in repr_plan['tasks'] if not t.startswith('ag-')]
    assert len(repr_tasks) == 11
    generation_tasks = ['cdr', 'pairing', 'tcr-benchmark14', 'tcr-held20']
    tasks = ['cdr', 'gdp_a1', 't2a', 't3broad', 't3deep', 'specificity', 'm396',
             'pairing', 'tcr-benchmark14', 'tcr-held20',
             't1-cdr3b', 't1-others_cdr3b', 't1-cdr3ab', 't1-others_longab', 'uptcr']
    assert set(tasks) == set(repr_tasks + generation_tasks)
    for task in tasks:
        (run / task).mkdir(exist_ok=True)
    (run / 'local').mkdir(exist_ok=True)
    shutil.copy2(GENERATION / 'airgen_evaluate.py', run / 'airgen_evaluate.py')
    shutil.copy2(ROOT / 'scripts/downstream/run_diffusion_edit_suite.py', run / 'run_suite.py')
    shutil.copy2(ROOT / 'scripts/downstream/run_local_diffusion_edit.py', run / 'local/driver.py')
    spec = importlib.util.spec_from_file_location('_cdr_input_audit', run / 'airgen_evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = dict(repr_plan['data_sha256'])
    cdr_counts = {}
    for dataset in module.MODES:
        rows, files = module.records(dataset)
        data.update(files)
        cdr_counts[dataset] = len(rows)
    assert cdr_counts == {'kong': 9381, 'sab23': 360}
    benchmark = gen_plan['benchmark']
    data[benchmark['path']] = benchmark['sha256']
    tcr_data = ROOT / 'downstream/benchmark/data/tcr_generation_bench'
    for name in ('eval_conditional.json', 'train_binders.csv', 'bioseq_unseen_pmhc.json'):
        data[str(tcr_data / name)] = sha(tcr_data / name)
    for path, expected in data.items():
        assert sha(path) == expected, 'Reference data changed: ' + path
    old_code = {**repr_plan['code_sha256'], **gen_plan['source_sha256']}
    # Bind the present dependencies in a new experiment, retaining any historical
    # difference as evidence. Never change a prior run's hashes or results.
    missing = [p for p in old_code if not Path(p).exists()]
    assert not missing, missing
    differences = {p: dict(reference=h, current=sha(p))
                   for p, h in old_code.items() if sha(p) != h}
    code = {p: sha(p) for p in old_code}
    for path in (
        Path(__file__), run / 'run_suite.py', run / 'airgen_evaluate.py',
        run / 'local/driver.py', ROOT / 'downstream/diffusion_edit_eval.py',
        ROOT / 'scripts/downstream/run_diffusion_edit_suite.py',
        ROOT / 'scripts/downstream/run_local_diffusion_edit.py',
        ROOT / 'scripts/downstream/publish_diffusion_edit_results.py',
        ROOT / 'scripts/downstream/run_oas_generation_eval.py',
        ROOT / 'downstream/v7_native.py',
        ROOT / 'downstream/v7_tcr_scoring.py',
        ROOT / 'downstream/benchmark/tcr_generation_bench/scoring.py',
    ):
        code[str(path)] = sha(path)
    protocol = {**gen_plan['protocol'], 'geometry': 'native V7 fixed167/135',
                'model_output_filter': 'native 25 residues plus EOS; grammar excluded'}
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        model=f"diffusion_edit_v7_75.6M_step{receipt['step']}",
        step=receipt['step'], model_sha256=receipt['files']['model.safetensors'],
        checkpoint=str(run / 'checkpoint'), tasks=tasks, generation_tasks=generation_tasks,
        representation_tasks=repr_tasks, groups=[tasks],
        selection='Latest complete at request inspection; fixed before any downstream scores',
        execution='local shared GPU, sequential; user local preference retained',
        comparison_runs=[str(REPRESENTATION), str(GENERATION)],
        code_sha256=code, data_sha256=data, source_sha256=code,
        historical_source_differences=differences,
        benchmark=benchmark, protocol=protocol,
        cdr={**gen_plan['cdr'], 'region_predictions': cdr_counts,
             'model_output_filter': 'AirGen known-span 20AA intersection with native residue/EOS vocabulary',
             'layout': 'native167/135'},
        tcr=dict(data=str(tcr_data / 'eval_conditional.json'),
                 model='full beta native167, EOS learned; alpha absent',
                 conditions='peptide + provided MHC + desired binding relation',
                 stochastic_candidates=100, independent_greedy=1, steps=32,
                 sampler='native multinomial + sampled-probability confidence',
                 temperature=1., cfg_scale=0., seed_assignment='42 + candidate_index, per target',
                 reference_length=False, reference_residues=False,
                 extraction='existing ANARCI IMGT104:118 with legacy regex fallback',
                 scoring='existing sequence metrics with explicit null/empty denominators',
                 pretraining_overlap='not yet established; historical unseen is not current unseen'),
        excluded=dict(
            cov_unibind='Author fold IDs unavailable; do not restart superseded diagnostic splits',
            t2b='Removed by user', abbibench='Zero-shot PLL outside this representation run',
            baselines='Use existing published/local comparisons; no rerun',
            chimera='Separate antigen-conditioned design protocol, outside this core AB/TCR suite'),
        gpu_index=0, minimum_free_mib=16384)
    write(run / 'plan.json', plan)
    audit = json.loads((REPRESENTATION / 'input_audit.json').read_text())
    assert audit['status'] == 'passed'
    checks = {k: v for k, v in audit['checks'].items() if not k.startswith('ag-')}
    assert all(v['over_capacity'] == 0 for v in checks.values())
    write(run / 'input_audit.json', dict(
        status='passed', reference=str(REPRESENTATION / 'input_audit.json'),
        reference_sha256=sha(REPRESENTATION / 'input_audit.json'),
        identical_config_and_vocabulary=True, checks=checks,
        cdr_region_predictions=cdr_counts))
    archive = run / 'source'
    for path in code:
        path = Path(path)
        if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'output'):
            target = archive / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    with (run / 'cpu_checks.log').open('x') as stream:
        process = subprocess.run([sys.executable, '-B', str(run / 'run_suite.py'),
                                  '--run', str(run), '--check', '--device', 'cpu'],
                                 stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError('CPU check failed; inspect cpu_checks.log before any GPU launch')
    write(run / 'preparation.json', dict(status='cpu_passed', tasks=len(tasks),
                                         plan_sha256=sha(run / 'plan.json')))
    print(json.dumps(dict(run=str(run), status='cpu_passed', tasks=len(tasks))))


if __name__ == '__main__':
    main()
