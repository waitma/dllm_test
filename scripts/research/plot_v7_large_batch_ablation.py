"""Plot one explicitly selected large-batch study using metadata only.

Activate conda pllm and run:
  python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_large_batch_ablation.py --jobs-file /absolute/jobs.json --pool-dir /absolute/pool --batch-plan /absolute/selection.json --batch-plan-sha256 HEX64 --output-prefix /absolute/figures/large_batch

The jobs ledger must bind this study and batch_plan={path,sha256}; each job
binds its explicit alias, architecture and seed. No models, sequences, binary
checkpoints, queue mutations or GPU operations are used. Profile references
are checked for byte identity, never interpreted as proof of GPU success.
"""
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re


BASE_PATH = Path(__file__).with_name('plot_v7_context_ablation.py').resolve()
spec = importlib.util.spec_from_file_location('context_ablation_plot_helpers', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
require = base.require
STUDY = 'v7_context_factorial_120k_large_batch_v1'
PLAN_SCHEMA = 'v7.large_batch_selection.v1'
PROTOCOL_COLUMNS = ('study', 'micro_batch', 'grad_accum', 'global_batch',
                    'evaluation_micro_batch', 'batch_plan_sha256')


def reference(path, expected_sha, evidence, parse=True):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink()
            and path.is_file(), f'Expected canonical regular metadata file: {path}')
    require(isinstance(expected_sha, str) and re.fullmatch('[0-9a-f]{64}', expected_sha),
            'Expected a lowercase SHA256')
    raw = path.read_bytes()
    require(base.sha(raw) == expected_sha, f'Metadata SHA differs: {path}')
    evidence[str(path)] = {'sha256': expected_sha, 'bytes': len(raw)}
    return json.loads(raw) if parse else None


def selected_plan(path, expected_sha, evidence):
    plan = reference(path, expected_sha, evidence)
    expected = {'schema': PLAN_SCHEMA, 'study': STUDY, 'status': 'selected',
                'global_batch': 32, 'evaluation_micro_batch': 4,
                'sample_manifest_sha256': base.SAMPLE_SHA,
                'variants': list(base.VARIANTS), 'seeds': list(base.SEEDS)}
    require(all(plan.get(key) == value for key, value in expected.items()),
            'Selection does not bind this full large-batch design')
    require((plan.get('micro_batch'), plan.get('grad_accum')) in ((16, 2), (32, 1)),
            'Require one selected micro16/GA2 or micro32/GA1 protocol')
    refs = plan.get('profile_evidence')
    require(isinstance(refs, list) and refs, 'Selection lacks actual profile evidence references')
    seen = set()
    for item in refs:
        require(isinstance(item, dict) and set(item) == {'path', 'sha256'}, 'Invalid profile reference')
        require(item['path'] not in seen, 'Duplicate profile reference')
        seen.add(item['path'])
        reference(item['path'], item['sha256'], evidence, parse=False)
    return plan, {'path': str(path), 'sha256': expected_sha, 'profile_evidence': refs}


def identity(provenance, plan, plan_ref):
    require(provenance.get('kind') == 'exploratory_architecture_reconstruction'
            and all(provenance.get(key) is False for key in
                    ('formal_training', 'biological_quality_claim', 'generation')),
            'Require actual reconstruction metadata; artificial CPU runs are excluded')
    result = {key: provenance[key] for key in base.COMPARABLE}
    sources = result['source_sha256']
    require(isinstance(sources, dict) and sources and
            all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in sources.values()),
            'Missing source identity')
    require(all(name in sources for name in ('scripts/research/run_v7_large_batch_ablation.py',
                'scripts/research/run_v7_architecture_pilot.py', 'scripts/research/context_pilot_metrics.py')),
            'Missing executed entry/core/evaluator source identities')
    require([result['config'][key] for key in
             ('d_model', 'n_layers', 'n_heads', 'mlp_hidden_size', 'max_length')]
            == [768, 8, 12, 3072, 2048], 'Expected unchanged full-size model')
    require([result[key] for key in ('micro_batch', 'grad_accum', 'global_batch', 'scheduler_horizon',
                                    'warmup', 'evaluation_interval', 'learning_rate')]
            == [plan['micro_batch'], plan['grad_accum'], 32, 3750, 100, 250, 1e-4],
            'Selected training batch or schedule differs')
    require(result['loss'] == 'existing_micro_mean' and
            result['mask_protocol'] == 'independent_seed_per_global_microstep', 'Training protocol differs')
    expected = dict(name=STUDY, factor_order=['G', 'S', 'L'],
        relation_objective='existing_binding_positives_only', context_sequence_loss=False,
        negative_pair_construction=False, expected_train_records=120000, complete_epoch_updates=3750,
        evaluation='source_and_context_strata_with_relation_token_counts_v1', evaluation_micro_batch=4,
        loss='existing_micro_mean', training_batch_protocol=f'micro{plan["micro_batch"]}_ga{plan["grad_accum"]}_global32',
        mask_stream='seed*1000003+(step-1)*grad_accum+physical_micro_index',
        p02_mask_or_gradient_equivalence_claim=False, profile_status='selected', batch_plan=plan_ref)
    study = result['study']
    require(all(study.get(key) == value for key, value in expected.items()), 'Different study/evaluation/plan binding')
    require(isinstance(study.get('input_preflight_sha256'), str) and
            re.fullmatch('[0-9a-f]{64}', study['input_preflight_sha256']), 'Missing input-preflight binding')
    data = provenance['data']
    require(data['sha256'] == base.SAMPLE_SHA, 'Not the registered context subset')
    files = data['manifest']['files']
    require(files['train']['records'] == 120000 and files['valid']['records'] == 1024,
            'Expected 120000 train / 1024 validation records')
    result.update(data_sha256=data['sha256'], data_manifest=data['manifest'])
    return result


def collect(jobs_file, pool, plan_path, plan_sha):
    evidence, warnings, runs, seen, ids, aliases, initial = {}, [], [], set(), set(), set(), {}
    plan, plan_ref = selected_plan(plan_path, plan_sha, evidence)
    jobs = base.read_json(jobs_file, evidence)
    require(jobs.get('study') == STUDY and jobs.get('batch_plan') ==
            {'path': str(plan_path), 'sha256': plan_sha}, 'Jobs ledger study/selection differs')
    require(isinstance(jobs.get('jobs'), list), 'Missing explicit jobs ledger')
    common_identity = None
    for entry in jobs['jobs']:
        variant, seed, job_id, alias = (entry[key] for key in ('variant', 'seed', 'job_id', 'alias'))
        require(variant in base.VARIANTS and type(seed) is int and seed in base.SEEDS, 'Unexpected design cell')
        require((variant, seed) not in seen and job_id not in ids and alias not in aliases,
                'Duplicate cell/job/alias; choose one attempt explicitly')
        require(isinstance(job_id, str) and Path(job_id).name == job_id and job_id not in ('.', '..'), 'Invalid job ID')
        require(isinstance(alias, str) and re.fullmatch(r'[A-Za-z0-9_]+', alias)
                and not alias.startswith('p02_') and alias.endswith(f'_v7_{variant}_seed{seed}'),
                'Alias does not bind this architecture/seed or belongs to old p02')
        job = entry['job']
        require(job['variant'] == alias and job['mode'] == 'train' and job['steps'] == 3750
                and Path(job['data_dir']).is_absolute(), 'Job specification differs')
        seen.add((variant, seed)); ids.add(job_id); aliases.add(alias)
        directory = pool / 'runs' / job_id
        require(not directory.is_symlink(), 'Run metadata directory cannot be a symlink')
        meta = {name: base.read_json(directory / filename, evidence, optional=True) for name, filename in
                [('provenance', 'run_provenance.json'), ('progress', 'progress.json'),
                 ('completion', 'COMPLETE.json'), ('summary', 'summary.json')]}
        result = base.read_json(pool / 'results' / f'{job_id}.json', evidence, optional=True)
        status = 'not_observed'
        for queue in ('pending', 'running'):
            queued = base.read_json(pool / queue / f'{job_id}.json', evidence, optional=True)
            if queued is not None:
                require(queued == job, 'Ledger/actual queue job differ')
                status = queue
        if result is not None:
            require(result.get('job_id') == job_id, 'Pool result belongs to a different job')
            status = result.get('state', 'result_without_state')
        elif meta['completion'] is not None:
            status = 'complete_pool_unconfirmed'
        run = dict(job_id=job_id, alias=alias, variant=variant, seed=seed, status=status,
            directory=str(directory), **meta, pool_result=result,
            points=base.read_curve(directory / 'curve.jsonl', evidence, warnings), terminal_verified=False)
        provenance, complete = meta['provenance'], meta['completion']
        if provenance is not None:
            require(provenance['variant'] == variant and provenance['seed'] == seed, 'Job/provenance identity mismatch')
            current = identity(provenance, plan, plan_ref)
            require(common_identity is None or current == common_identity, 'Data/source/config/budget differ between runs')
            common_identity = current
            require(Path(provenance['data']['path']) == Path(job['data_dir']) / 'sample_manifest.json', 'Data path differs')
            common_sha = provenance['common_initialization_sha256']
            require(seed not in initial or initial[seed] == common_sha, 'Same-seed shared initialization differs')
            initial[seed] = common_sha
            base.validate_points(run)
        else:
            require(not run['points'] and complete is None, 'Curve/completion without provenance')
        if complete is not None:
            require(complete.get('status') == 'completed' and complete.get('global_step') == 3750
                    and complete.get('kind') == provenance['kind']
                    and complete.get('formal_training') is False and complete.get('biological_quality_claim') is False,
                    'Completion is not a full large-batch cycle')
            require(run['points'] and run['points'][-1] == meta['summary'] and meta['summary']['step'] == 3750,
                    'Completed summary differs from the actual last curve point')
            for name, filename in [('summary', 'summary.json'), ('provenance', 'run_provenance.json')]:
                require(complete[f'{name}_sha256'] == evidence[str(directory / filename)]['sha256'],
                        'Completion artifact identity differs')
            run['terminal_verified'] = bool(result and result.get('state') == 'succeeded' and result.get('returncode') == 0)
        run['latest_evaluation_step'] = run['points'][-1]['step'] if run['points'] else None
        runs.append(run)
    eligible = [run for run in runs if any(point['step'] > 0 for point in run['points'])]
    shared = set.intersection(*(set(point['step'] for point in run['points']) for run in eligible)) if len(eligible) >= 2 else set()
    step = max((value for value in shared if value > 0), default=None)
    endpoints, budgets = {}, {}
    if step is not None:
        for run in eligible:
            point = next(point for point in run['points'] if point['step'] == step)
            exposure = tuple(point[key] for key in ('records', 'input_tokens', 'eligible_tokens', 'selected_aa_terminal_tokens'))
            require(run['seed'] not in budgets or exposure == budgets[run['seed']], 'Paired-seed exposure/mask budget differs')
            budgets[run['seed']] = exposure
            endpoints[base.cell(run['variant'], run['seed'])] = point
    all_cells = [base.cell(variant, seed) for variant in base.VARIANTS for seed in base.SEEDS]
    full_scope = len(endpoints) == len(all_cells)
    final = full_scope and step == 3750 and all(run['terminal_verified'] for run in runs)
    protocol = {key: plan[key] for key in PROTOCOL_COLUMNS if key != 'batch_plan_sha256'}
    protocol['batch_plan_sha256'] = plan_sha
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), **protocol,
        status='final' if final else 'partial' if step is not None else 'awaiting_shared_evaluation',
        final=final, partial=not final, endpoint_step=step, endpoint_scope='all_22' if full_scope else 'partial',
        all_22_common_step=step if full_scope else None, endpoint_cells=sorted(endpoints),
        missing_registered_cells=[name for name in all_cells if name not in {base.cell(*item) for item in seen}],
        missing_nonzero_evaluation_cells=[name for name in all_cells if name not in
            {base.cell(run['variant'], run['seed']) for run in eligible}],
        missing_endpoint_cells=[name for name in all_cells if name not in endpoints],
        jobs_file=str(jobs_file), pool_dir=str(pool), runs=runs, endpoints=endpoints,
        batch_plan=plan_ref, selected_batch_plan=plan, profile_evidence_interpretation='byte_identity_only_not_GPU_pass_certification',
        common_initialization_by_seed=initial, input_snapshots=evidence, warnings=warnings,
        source_sha256={str(Path(__file__).resolve()): base.sha(Path(__file__).read_bytes()),
                       str(BASE_PATH): base.sha(BASE_PATH.read_bytes())},
        figures=[], confidence_intervals=False, biological_quality_claim=False, formal_training=False,
        figure_title=f'Large-batch context ablation; micro{plan["micro_batch"]}/GA{plan["grad_accum"]}, eval micro4',
        scope='Observed reconstruction only; existing binding positives do not measure binary discrimination',
        interpretation='One dot per seed; no CI or significance claim. Old p02 is excluded; missing cells remain absent.')
    report.update(base.derive(endpoints, step))
    return report


def export(prefix, report):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    base.write_csv(prefix, report)
    # Preserve the existing CSV metrics while attaching the cohort protocol to every row.
    for name in ('', '_paired', '_interactions', '_removal'):
        path = prefix.with_name(prefix.name + name).with_suffix('.csv')
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        base.csv_table(path, [{**{key: report[key] for key in PROTOCOL_COLUMNS}, **row} for row in rows],
                       [*PROTOCOL_COLUMNS, *columns])
    base.csv_table(prefix.with_name(prefix.name + '_runs').with_suffix('.csv'),
        [{**{key: report[key] for key in PROTOCOL_COLUMNS}, **{key: run[key] for key in
            ('job_id', 'alias', 'variant', 'seed', 'status', 'latest_evaluation_step', 'terminal_verified')}}
         for run in report['runs']], [*PROTOCOL_COLUMNS, 'job_id', 'alias', 'variant', 'seed', 'status',
                                    'latest_evaluation_step', 'terminal_verified'])
    if report['endpoint_step'] is not None:
        base.draw(report, prefix)
    else:
        report['message'] = 'No shared positive completed evaluation from at least two matched runs; no numerical figures written.'
        report['preexisting_figures_not_updated'] = [str(path) for path in prefix.parent.glob(prefix.name + '_*')
                                                    if path.suffix in ('.svg', '.pdf', '.png')]
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('jobs-file', 'pool-dir', 'batch-plan', 'output-prefix'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batch-plan-sha256', required=True)
    args = parser.parse_args(argv)
    require(all(path.is_absolute() for path in (args.jobs_file, args.pool_dir, args.batch_plan, args.output_prefix)),
            'Use absolute paths')
    report = collect(args.jobs_file, args.pool_dir, args.batch_plan, args.batch_plan_sha256)
    export(args.output_prefix, report)
    print(json.dumps({key: report[key] for key in
        ('status', 'final', 'endpoint_step', 'all_22_common_step', 'missing_endpoint_cells', 'figures')}))


if __name__ == '__main__':
    main()
