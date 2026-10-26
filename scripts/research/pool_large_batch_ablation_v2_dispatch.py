"""Capacity-v2 integration for the unchanged frozen 22-condition training entry.

Activate conda pllm. The future pool calls main(original_argv) before ML imports.
For a read-only check use prepare(original_argv). This module never publishes
registry/queue entries or creates plans. It reuses the SHA-bound private v1
catalog/source/runner implementation with only capacity protocol and fallback
replaced. Repository: /project_c20250601/251105016/project/dllm_test.
"""
import hashlib
from pathlib import Path
import runpy

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/v7_large_batch_ablation_20261002'
BASE = OWN/'launcher_source_v1/pool_large_batch_ablation_dispatch.py'
BASE_SHA = 'a99a2b5ba6f247e8e3ac903240fd604f59b05bd922f5d7b2be36534821f1ed67'
PREVIOUS = OWN/'capacity_launcher_source_v2/scripts/research/pool_large_batch_capacity_v2_dispatch.py'
PREVIOUS_SHA = '73cda09463f9cba4df22ec2f2bf42170f9bdea51d81d838c0ec27fff7caf9827'
CAPACITY_JOB = '1790992416352344471_p05_v7_capacity_v2_a4854542'
ROOT_REVIEW = OWN/'capacity_v2_root_terminal_review.json'
ROOT_REVIEW_SHA = 'b6ae90637728f22bef378e71d0d2c943e50ee74e4ad52165b1f0a66d95970c5b'

if BASE.is_symlink() or BASE != BASE.resolve() or hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError('Private v1 dispatcher source changed')
_base = runpy.run_path(str(BASE))
if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError('Private v1 changed while loading')
_runtime = _base['prepare'].__globals__
for _name in ('POOL', 'CONFIG', 'PROFILE_VARIANTS', 'require', 'read', 'number', 'sha'):
    globals()[_name] = _runtime[_name]


def check_capacity(plan, snapshot, sources, canonical):
    run = POOL / 'runs' / CAPACITY_JOB
    required_paths = {'ledger': OWN / 'capacity_job_v2.json', 'summary': run / 'summary.json',
                      'completion': run / 'COMPLETE.json', 'result': POOL / 'results' / (CAPACITY_JOB + '.json')}
    refs = plan['profile_evidence']
    mapping = {item['path']: item['sha256'] for item in refs}
    require(len(mapping) == len(refs), 'Duplicate profile evidence reference')
    require(set(map(str, required_paths.values())) <= set(mapping), 'Selection lacks the unique capacity job evidence quartet')
    bound = {name: read(path, canonical, mapping[str(path)]) for name, path in required_paths.items()}
    ledger, summary, complete, result = (bound[name] for name in ('ledger', 'summary', 'completion', 'result'))
    validate_v2_intent_and_review(ledger, mapping, canonical)
    envelope = dict(variant='p05_v7_capacity_v2', mode='profile-synthetic', steps=36)
    require(ledger.get('schema') == 'v7.capacity.enqueue_intent.v2' and ledger.get('job_id') == CAPACITY_JOB
            and ledger.get('job') == envelope and ledger.get('pool_dir') == str(POOL)
            and ledger.get('run_dir') == str(run), 'Capacity ledger identity/envelope differs')
    # The ledger is the original enqueue snapshot; final status comes from the pool result.
    expected_argv = ['-u', str(ROOT / 'scripts/run_immune_research.py'), '--variant', 'p05_v7_capacity_v2',
                     '--mode', 'profile-synthetic', '--steps', '36', '--profile-dir',
                     str(POOL / 'profiles/p05_v7_capacity_v2'), '--output-dir', str(run)]
    argv = result.get('argv')
    require(result.get('job_id') == CAPACITY_JOB and result.get('job') == envelope
            and result.get('state') == 'succeeded' and type(result.get('returncode')) is int and result['returncode'] == 0
            and isinstance(argv, list) and len(argv) == len(expected_argv) + 1
            and isinstance(argv[0], str) and Path(argv[0]).is_absolute() and argv[1:] == expected_argv,
            'Capacity pool success/command envelope is not established')
    require(complete.get('kind') == 'synthetic_capacity_profile_only' and complete.get('status') == 'completed'
            and complete.get('summary_sha256') == mapping[str(required_paths['summary'])]
            and all(complete.get(flag) is False for flag in ('training_completed', 'actual_corpus_test', 'checkpoint_created')),
            'Capacity completion flags or summary SHA differ')
    require(summary.get('kind') == 'synthetic_whole_model_capacity_sweep' and summary.get('status') == 'completed'
            and summary.get('biological_quality_claim') is False and summary.get('planned_optimizer_updates') == 36
            and summary.get('optimizer_update_hard_cap') == 100 and summary.get('global_batch') == 32
            and summary.get('candidate_seconds') == 240 and summary.get('max_seconds') == 1800
            and summary.get('source_root') == str(snapshot)
            and summary.get('profile_script_sha256') == sources['scripts/research/profile_v7_large_batch.py']
            and all(summary.get('config', {}).get(key) == value for key, value in CONFIG.items()),
            'Capacity summary protocol/source differs')
    measurements = summary.get('measurements')
    require(isinstance(measurements, list) and len(measurements) == 9, 'Require all nine measured capacity cells')
    cells = set()
    for item in measurements:
        variant, micro = item.get('variant'), item.get('micro_batch')
        require(variant in PROFILE_VARIANTS and type(micro) is int and micro in (4, 16, 32)
                and (variant, micro) not in cells, 'Duplicate or unexpected capacity cell')
        cells.add((variant, micro))
        original = run / f'{variant}_micro{micro}' / 'measurement.json'
        require(str(original) in mapping and read(original, canonical, mapping[str(original)]) == item,
                'Original measurement differs from the bound summary cell')
        require(item.get('kind') == 'synthetic_whole_model_capacity' and item.get('status') in ('passed', 'oom')
                and item.get('global_batch') == 32 and item.get('grad_accum') == 32 // micro
                and item.get('shape') == [micro, 2048] and item.get('attended_positions_per_record') == 2048
                and all(item.get('config', {}).get(key) == value for key, value in CONFIG.items())
                and isinstance(item.get('gpu'), str) and 'A100' in item['gpu']
                and number(item.get('total_gib'), 75) and item['total_gib'] <= 85
                and item.get('precision') == 'fp32_parameters_adam_bf16_autocast'
                and item.get('activation_checkpointing') == 'whole_layer' and item.get('seed') == 42
                and item.get('warmup_steps') == 2 and item.get('measure_steps') == 2
                and item.get('optimizer') == 'AdamW' and item.get('learning_rate') == 1e-4
                and item.get('max_grad_norm') == 1.0 and item.get('biological_quality_claim') is False
                and item.get('actual_corpus_test') is False, 'Capacity cell model/hardware/protocol differs')
        measured_sources = item.get('source_sha256')
        require(isinstance(measured_sources, dict) and measured_sources, 'Missing actual capacity source identity')
        for name, expected in measured_sources.items():
            path = canonical(Path(name))
            require(path.is_relative_to(snapshot) and sources.get(str(path.relative_to(snapshot))) == expected,
                    'Measured source is outside or differs from the frozen model')
        require(all(str(snapshot / name) in measured_sources for name in
                    ('scripts/research/profile_v7_large_batch.py', 'dllm/pipelines/immune_llada/research_refined/factorial.py')),
                'Missing measured profile/model source')
        require(number(item.get('peak_allocated_gib')) and number(item.get('peak_reserved_gib')),
                'Capacity peak memory is unavailable')
        if item['status'] == 'passed':
            require(item.get('completed_optimizer_steps') == 4 and isinstance(item.get('measured_losses'), list)
                    and len(item['measured_losses']) == 2 and all(number(value) for value in item['measured_losses'])
                    and number(item.get('optimizer_state_tensor_bytes'), 1)
                    and number(item.get('measured_seconds'), 1e-12)
                    and number(item.get('input_tokens_per_second'), 1e-12), 'Passed capacity cell lacks four real optimizer updates under capacity v2')
        else:
            require(item.get('expected_capacity_failure') is True and isinstance(item.get('error'), str) and item['error'],
                    'OOM must be the explicit expected capacity failure')
        if micro == plan['micro_batch']:
            require(item['status'] == 'passed' and item['peak_allocated_gib'] <= .85 * item['total_gib']
                    and item['peak_reserved_gib'] <= .85 * item['total_gib'], 'Selected micro lacks all-three safe capacity evidence')


def validate_v2_intent_and_review(ledger, mapping, canonical):
    cold = _runtime['previous']()
    require(mapping.get(str(cold['LEDGER'])) == cold['LEDGER_SHA'], 'Capacity v2 intent SHA differs')
    require(ledger['protocol'] == cold['PROTOCOL'] and ledger['automatic_retry'] is False
            and type(ledger['max_attempts']) is int and ledger['max_attempts'] == 1
            and ledger['source_freeze'] == {'path': str(cold['FREEZE']), 'sha256': cold['FREEZE_SHA']},
            'Capacity v2 intent protocol differs')
    cold['prior_failure'](ledger['previous_failure'])
    require(mapping.get(str(ROOT_REVIEW)) == ROOT_REVIEW_SHA, 'Missing original v2 terminal review')
    review = read(ROOT_REVIEW, canonical, ROOT_REVIEW_SHA)
    require(review['schema'] == 'v7.capacity.root_terminal_review.v2' and review['status'] == 'passed'
            and review['job_id'] == CAPACITY_JOB and review['protocol'] == cold['PROTOCOL']
            and review['actual_loaded_sources_checked'] == 40, 'Terminal review scope differs')
    for key in ('intent', 'summary', 'completion', 'result'):
        reference = review['refs'][key]
        require(mapping.get(reference['path']) == reference['sha256'], 'Plan/review original reference differs')
    require(len(review['measurements']) == 9, 'Terminal review lacks nine original measurements')
    for item in review['measurements']:
        reference = item['measurement']
        require(mapping.get(reference['path']) == reference['sha256'], 'Plan/review measurement SHA differs')


# These assignments affect only this private runpy namespace, never frozen files.
_runtime.update(PREVIOUS=PREVIOUS, PREVIOUS_SHA=PREVIOUS_SHA, CAPACITY_JOB=CAPACITY_JOB,
                check_capacity=check_capacity)
prepare = _runtime['prepare']
main = _runtime['main']

if __name__ == '__main__':
    raise SystemExit(main())
