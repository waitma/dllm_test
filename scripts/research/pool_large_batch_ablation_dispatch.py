"""Validate and exec one frozen large-batch job before any ML imports.

Activate conda pllm. The future shared pool entry calls main(original_argv).
For a read-only launch check, call prepare(original_argv); it never creates
plans, jobs, outputs or processes. This file is source-only until separately
published. Unknown aliases retain the SHA-bound capacity dispatch chain.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
OWN = ROOT / 'output/v7_large_batch_ablation_20261002'
POOL = ROOT / 'output/immune_research_pool_c20250601_20260930'
DATA = ROOT / 'output/v7_architecture_pilot_20261002/sample_context_plus20k'
REGISTRY = ROOT / 'dllm/pipelines/immune_llada/research/model.py'
FREEZE = OWN / 'source_snapshot_v1.json'
FREEZE_SHA = '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'
PLAN = OWN / 'batch_selection_v1.json'
CATALOG = OWN / 'jobs.json'
PREVIOUS = ROOT / 'scripts/research/pool_large_batch_dispatch.py'
PREVIOUS_SHA = '76d07d623f0fe068dcffa55a28c60a38695fe0a256d4a4bfd7751bded50aad5c'
CAPACITY_JOB = '1790981433321562451_p04_v7_capacity_febb28e3'
STUDY = 'v7_context_factorial_120k_large_batch_v1'
SAMPLE_SHA = '1efa27ea19d10bc94143c3977c80dba8c3a29bad921c093ca7f24447635b9478'
METHODS = ('f000', 'f001', 'f010', 'f011', 'f100', 'f101', 'f110', 'f111',
           'cv_segment_global', 'cv_late_fusion', 'fixed_gate_control')
ALIASES = {f'p04_v7_{variant}_seed{seed}': (variant, seed) for variant in METHODS for seed in (42, 43)}
PROFILE_VARIANTS = ('f000', 'f111', 'cv_segment_global')
CONFIG = dict(ablation='A4', d_model=768, n_layers=8, n_heads=12, mlp_hidden_size=3072,
              max_length=2048, full_capacity=256, fragment_capacity=64)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def previous():
    require(sha(PREVIOUS) == PREVIOUS_SHA and not PREVIOUS.is_symlink(), 'Capacity dispatcher source changed')
    return runpy.run_path(str(PREVIOUS))


def read(path, canonical, expected=None):
    path = canonical(path)
    require(path.is_file(), 'Required metadata is missing: ' + str(path))
    raw = path.read_bytes()
    if expected is not None:
        require(digest(expected) and hashlib.sha256(raw).hexdigest() == expected, 'Metadata SHA differs: ' + str(path))
    return json.loads(raw)


def registry(canonical):
    tree = ast.parse(canonical(REGISTRY).read_text(), filename=str(REGISTRY))
    assignments = [node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'VARIANTS' for target in node.targets)]
    require(len(assignments) == 1 and isinstance(assignments[0], ast.Dict), 'Require one literal VARIANTS registry')
    node = assignments[0]
    keys = [ast.literal_eval(key) for key in node.keys]
    require(len(keys) == len(set(keys)), 'Duplicate literal registry key')
    entries = {key: ast.literal_eval(value) for key, value in zip(keys, node.values) if key in ALIASES}
    require(set(entries) == set(ALIASES), 'All 22 literal launcher entries must be published together')
    return entries


def check_catalog(entries, canonical):
    first = entries[next(iter(ALIASES))]
    catalog_sha, plan_sha = first.get('jobs_catalog_sha256'), first.get('batch_plan_sha256')
    require(digest(catalog_sha) and digest(plan_sha), 'Registry lacks catalog/plan identity')
    for alias, (variant, seed) in ALIASES.items():
        item = entries[alias]
        expected = dict(launcher_only=True, model_variant=variant, seed=seed, study=STUDY,
            jobs_catalog_sha256=catalog_sha, batch_plan_sha256=plan_sha, source_freeze_sha256=FREEZE_SHA)
        require(all(item.get(key) == value for key, value in expected.items()), 'Cross-group registry identity: ' + alias)
        require(item.get('launcher_only') is True and type(item.get('seed')) is int, 'Literal launcher/seed types differ')
    catalog = read(CATALOG, canonical, catalog_sha)
    require(catalog.get('study') == STUDY
            and catalog.get('batch_plan') == {'path': str(PLAN), 'sha256': plan_sha}
            and catalog.get('source_freeze') == {'path': str(FREEZE), 'sha256': FREEZE_SHA}, 'Catalog group bindings differ')
    rows = catalog.get('jobs')
    require(isinstance(rows, list) and len(rows) == 22, 'Complete 22-job ledger must precede queue publication')
    by_id, seen = {}, set()
    for row in rows:
        alias, job_id = row['alias'], row['job_id']
        require(alias in ALIASES and alias not in seen and isinstance(job_id, str) and job_id not in by_id,
                'Unknown/duplicate catalog alias or job ID')
        require(re.fullmatch(r'[0-9]+_' + re.escape(alias) + r'_[0-9a-f]{8}', job_id), 'Catalog job ID is not bound to alias')
        variant, seed = ALIASES[alias]
        require(row['variant'] == variant and type(row['seed']) is int and row['seed'] == seed,
                'Catalog architecture/seed differs')
        require(row['job'] == dict(variant=alias, mode='train', steps=3750, data_dir=str(DATA)), 'Catalog job envelope differs')
        canonical(POOL / 'runs' / job_id)
        by_id[job_id] = row
        seen.add(alias)
    require(seen == set(ALIASES), 'Catalog omits a design cell')
    return by_id, plan_sha


def check_freeze(canonical):
    frozen = read(FREEZE, canonical, FREEZE_SHA)
    snapshot = canonical(Path(frozen['snapshot_root']))
    require(snapshot == OWN / 'source_snapshot_v1' and frozen.get('study') == STUDY
            and frozen.get('horizon') == 3750 and frozen.get('global_batch') == 32
            and frozen.get('evaluation_micro_batch') == 4 and frozen.get('sample_manifest_sha256') == SAMPLE_SHA,
            'Frozen study/source/schedule differs')
    sources, evidence = frozen['source_sha256'], frozen['evidence_sha256']
    require(len(sources) == 568 and len(evidence) == 2, 'Require the exact 568-source/two-CPU-evidence freeze')
    for name, expected in sources.items():
        path = canonical(snapshot / name)
        require(path.is_relative_to(snapshot) and digest(expected) and sha(path) == expected, 'Frozen source differs: ' + name)
    for name, expected in evidence.items():
        require(digest(expected) and sha(canonical(Path(name))) == expected, 'CPU evidence differs: ' + name)
    return snapshot, sources


def check_capacity(plan, snapshot, sources, canonical):
    run = POOL / 'runs' / CAPACITY_JOB
    required_paths = {'ledger': OWN / 'capacity_job.json', 'summary': run / 'summary.json',
                      'completion': run / 'COMPLETE.json', 'result': POOL / 'results' / (CAPACITY_JOB + '.json')}
    refs = plan['profile_evidence']
    mapping = {item['path']: item['sha256'] for item in refs}
    require(len(mapping) == len(refs), 'Duplicate profile evidence reference')
    require(set(map(str, required_paths.values())) <= set(mapping), 'Selection lacks the unique capacity job evidence quartet')
    bound = {name: read(path, canonical, mapping[str(path)]) for name, path in required_paths.items()}
    ledger, summary, complete, result = (bound[name] for name in ('ledger', 'summary', 'completion', 'result'))
    envelope = dict(variant='p04_v7_capacity', mode='profile-synthetic', steps=72)
    require(ledger.get('schema') == 'v7.capacity.enqueue_intent.v1' and ledger.get('job_id') == CAPACITY_JOB
            and ledger.get('job') == envelope and ledger.get('pool_dir') == str(POOL)
            and ledger.get('run_dir') == str(run), 'Capacity ledger identity/envelope differs')
    # The ledger is the original enqueue snapshot; final status comes from the pool result.
    expected_argv = ['-u', str(ROOT / 'scripts/run_immune_research.py'), '--variant', 'p04_v7_capacity',
                     '--mode', 'profile-synthetic', '--steps', '72', '--profile-dir',
                     str(POOL / 'profiles/p04_v7_capacity'), '--output-dir', str(run)]
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
            and summary.get('biological_quality_claim') is False and summary.get('planned_optimizer_updates') == 72
            and summary.get('optimizer_update_hard_cap') == 100 and summary.get('global_batch') == 32
            and summary.get('candidate_seconds') == 180 and summary.get('max_seconds') == 1800
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
        require(item.get('kind') == 'synthetic_whole_model_capacity' and item.get('status') in ('passed', 'oom')
                and item.get('global_batch') == 32 and item.get('grad_accum') == 32 // micro
                and item.get('shape') == [micro, 2048] and item.get('attended_positions_per_record') == 2048
                and all(item.get('config', {}).get(key) == value for key, value in CONFIG.items())
                and isinstance(item.get('gpu'), str) and 'A100' in item['gpu']
                and number(item.get('total_gib'), 75) and item['total_gib'] <= 85
                and item.get('precision') == 'fp32_parameters_adam_bf16_autocast'
                and item.get('activation_checkpointing') == 'whole_layer' and item.get('seed') == 42
                and item.get('warmup_steps') == 2 and item.get('measure_steps') == 6
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
            require(item.get('completed_optimizer_steps') == 8 and isinstance(item.get('measured_losses'), list)
                    and len(item['measured_losses']) == 6 and all(number(value) for value in item['measured_losses'])
                    and number(item.get('optimizer_state_tensor_bytes'), 1)
                    and number(item.get('measured_seconds'), 1e-12)
                    and number(item.get('input_tokens_per_second'), 1e-12), 'Passed capacity cell lacks eight real optimizer updates')
        else:
            require(item.get('expected_capacity_failure') is True and isinstance(item.get('error'), str) and item['error'],
                    'OOM must be the explicit expected capacity failure')
        if micro == plan['micro_batch']:
            require(item['status'] == 'passed' and item['peak_allocated_gib'] <= .85 * item['total_gib']
                    and item['peak_reserved_gib'] <= .85 * item['total_gib'], 'Selected micro lacks all-three safe capacity evidence')


def prepare(argv):
    flags = ('--variant', '--mode', '--steps', '--data-dir', '--profile-dir', '--output-dir')
    require(all(sum(value.split('=', 1)[0] == flag for value in argv) == 1 for flag in flags),
            'Require each original pool argument exactly once')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--variant', choices=ALIASES, required=True)
    parser.add_argument('--mode', choices=('train',), required=True)
    parser.add_argument('--steps', type=int, choices=(3750,), required=True)
    for name in ('data-dir', 'profile-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    canonical = previous()['canonical']
    require(canonical(args.data_dir) == DATA and canonical(args.profile_dir) == POOL / 'profiles' / args.variant
            and canonical(args.output_dir).parent == POOL / 'runs', 'Require the original data/profile/pool roots')
    jobs, plan_sha = check_catalog(registry(canonical), canonical)
    require(args.output_dir.name in jobs and jobs[args.output_dir.name]['alias'] == args.variant,
            'Current pool job is absent or cross-bound in the complete catalog')
    snapshot, sources = check_freeze(canonical)
    runner_path = snapshot / 'scripts/research/run_v7_large_batch_ablation.py'
    runner = runpy.run_path(str(runner_path))  # This frozen entry imports only stdlib until main().
    plan = read(PLAN, canonical, plan_sha)
    selected = SimpleNamespace(batch_plan=PLAN, batch_plan_sha256=plan_sha,
                               micro_batch=plan.get('micro_batch'), grad_accum=plan.get('grad_accum'))
    runner['checked_batch_plan'](selected)
    for item in plan['profile_evidence']:
        canonical(Path(item['path']))
    check_capacity(plan, snapshot, sources, canonical)
    require(sha(canonical(DATA / 'sample_manifest.json')) == SAMPLE_SHA, 'Registered subset metadata changed')
    variant, seed = ALIASES[args.variant]
    command = [sys.executable, '-I', '-B', '-u', str(runner_path), '--variant', variant, '--seed', str(seed),
        '--steps', '3750', '--eval-every', '250', '--data-dir', str(DATA), '--output-dir', str(args.output_dir),
        '--micro-batch', str(plan['micro_batch']), '--grad-accum', str(plan['grad_accum']),
        '--batch-plan', str(PLAN), '--batch-plan-sha256', plan_sha]
    runner['validate_parameters'](runner['parse_args'](command[5:]))
    return args, command


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == '--variant']
    selected += [value.split('=', 1)[1] for value in argv if value.startswith('--variant=')]
    require(not any(value in ALIASES for value in selected) or len(selected) == 1, 'Ambiguous new-study alias selection')
    if len(selected) != 1 or selected[0] not in ALIASES:
        return previous()['main'](argv)
    require(re.fullmatch('[0-7]', os.environ.get('CUDA_VISIBLE_DEVICES', '')),
            'Require exactly one physical GPU mask assigned by the existing pool')
    _, command = prepare(argv)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    raise SystemExit(main())
