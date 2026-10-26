"""Export actual p02 metadata and curves without importing models or reading sequences.

Activate pllm and run:
  python -B /project_c20250601/251105016/project/dllm_test/scripts/research/plot_v7_context_ablation.py --jobs-file /absolute/jobs.json --pool-dir /absolute/pool --output-prefix /absolute/figures/context_ablation

Each rendered page is saved as SVG/PDF/PNG. CSV/JSON always retain missing cells
and raw denominators. Empty or initial-only inputs produce no numerical figures.
Only all 22 completed, source-matched runs at step 3750 constitute a final result.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path


FACTORIAL = tuple('f' + ''.join(bits) for bits in itertools.product('01', repeat=3))
CONTROLS = ('cv_segment_global', 'cv_late_fusion', 'fixed_gate_control')
VARIANTS = FACTORIAL + CONTROLS
SEEDS = (42, 43)
MASKS = ('mask25', 'mask50', 'mask75', 'mask100')
COUNT_FOR = {'loss': 'selected_count', 'aa_loss': 'aa_count', 'aa_accuracy': 'aa_count',
             'terminal_loss': 'terminal_count', 'terminal_accuracy': 'terminal_count',
             'relation_loss': 'relation_count', 'relation_accuracy': 'relation_count'}
SAMPLE_SHA = '1efa27ea19d10bc94143c3977c80dba8c3a29bad921c093ca7f24447635b9478'
STUDY = 'v7_context_factorial_120k_v1'
COMPARABLE = ('config', 'source_sha256', 'micro_batch', 'grad_accum', 'global_batch',
              'scheduler_horizon', 'warmup', 'learning_rate', 'loss', 'mask_protocol',
              'evaluation_interval', 'torch_version', 'study')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, evidence, optional=False):
    if optional and not path.exists():
        return None
    require(path.is_file() and not path.is_symlink(), f'Expected regular metadata file: {path}')
    raw = path.read_bytes()
    evidence[str(path)] = {'sha256': sha(raw), 'bytes': len(raw)}
    return json.loads(raw)


def read_curve(path, evidence, warnings):
    if not path.exists():
        return []
    require(path.is_file() and not path.is_symlink(), f'Expected regular curve: {path}')
    raw = path.read_bytes()
    evidence[str(path)] = {'sha256': sha(raw), 'bytes': len(raw)}
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        warnings.append(f'{path}: ignored incomplete final line ({len(lines[-1])} bytes)')
        lines.pop()
    points = [json.loads(line) for line in lines]
    steps = [point['step'] for point in points]
    require(all(type(step) is int and 0 <= step <= 3750 for step in steps)
            and steps == sorted(set(steps)), f'Nonunique or unordered curve steps: {path}')
    return points


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def near(first, second):
    return math.isclose(first, second, rel_tol=1e-6, abs_tol=1e-6)


def identity(provenance):
    require(provenance.get('kind') == 'exploratory_architecture_reconstruction'
            and all(provenance.get(key) is False for key in
                    ('formal_training', 'biological_quality_claim', 'generation')),
            'Only actual p02 reconstruction metadata is eligible; CPU checks are excluded')
    result = {key: provenance[key] for key in COMPARABLE}
    require(isinstance(result['source_sha256'], dict) and result['source_sha256'], 'Missing source identity')
    config, study = result['config'], result['study']
    require([config[key] for key in ('d_model', 'n_layers', 'n_heads', 'mlp_hidden_size', 'max_length')]
            == [768, 8, 12, 3072, 2048], 'Expected unchanged full-size model')
    require([result[key] for key in ('micro_batch', 'grad_accum', 'global_batch', 'scheduler_horizon',
                                    'warmup', 'evaluation_interval', 'learning_rate')]
            == [4, 8, 32, 3750, 100, 250, 1e-4], 'Expected p02 global32 / 3750 schedule')
    require(study['name'] == STUDY and study['factor_order'] == ['G', 'S', 'L']
            and study['expected_train_records'] == 120000 and study['complete_epoch_updates'] == 3750
            and study['relation_objective'] == 'existing_binding_positives_only'
            and study['context_sequence_loss'] is False and study['negative_pair_construction'] is False,
            'Different study or objective')
    data = provenance['data']
    require(data['sha256'] == SAMPLE_SHA, 'Not the registered p02 data manifest')
    files = data['manifest']['files']
    require(files['train']['records'] == 120000 and files['valid']['records'] == 1024,
            'Expected 120000 train / 1024 validation records')
    result.update(data_sha256=data['sha256'], data_manifest=data['manifest'])
    return result


def validate_aggregate(result):
    require(set(result['mask_levels']) == set(MASKS), 'Missing evaluation mask level')
    for mask in MASKS:
        level = result['mask_levels'][mask]
        for name in set(COUNT_FOR.values()):
            require(type(level[name]) is int and level[name] >= 0, 'Invalid recorded denominator')
        require(level['selected_count'] == sum(level[name] for name in
                ('aa_count', 'terminal_count', 'relation_count')), 'Selected token categories do not partition')
        for metric, denominator in COUNT_FOR.items():
            value, total, count = level[metric], level[metric + '_sum'], level[denominator]
            require(number(total) and total >= 0, 'Invalid metric numerator')
            require((value is None and total == 0) if count == 0 else
                    (number(value) and near(value, total / count)), 'Metric/count/numerator disagree')
            if metric.endswith('accuracy') and value is not None:
                require(0 <= value <= 1, 'Accuracy is outside [0,1]')
    for metric in COUNT_FOR:
        values = [result['mask_levels'][mask][metric] for mask in MASKS
                  if result['mask_levels'][mask][metric] is not None]
        mean = result['mask_mean'][metric]
        require(mean is None if not values else number(mean) and near(mean, sum(values) / len(values)),
                'Four-mask mean differs from its actual nonempty mask levels')


def validate_points(run):
    expected = run['provenance']['data']['manifest']['files']['valid']
    previous_tokens, previous_seconds = -1, -1.
    for point in run['points']:
        step, validation = point['step'], point['validation']
        require(point['records'] == step * 32 and point['subset_train_records'] == 120000,
                f'Exposure/update mismatch: {run["job_id"]}')
        require(point['epoch'] == (1 if step == 3750 else 0)
                and point['position'] == point['records'] % 120000, 'Epoch cursor differs from full single pass')
        require(point['input_tokens'] >= previous_tokens and point['train_seconds'] >= previous_seconds
                and number(point['train_seconds']), 'Nonmonotonic recorded cost')
        previous_tokens, previous_seconds = point['input_tokens'], point['train_seconds']
        require(validation['records'] == expected['records'] and
                {name: item['records'] for name, item in validation['sources'].items()} == expected['source_counts'],
                'Validation source counts differ from the common subset')
        context = validation['context_strata']
        require(set(context) <= {'with_context', 'without_context'} and
                sum(item['records'] for item in context.values()) == validation['records'],
                'Context strata do not partition validation records')
        pooled = validation['token_pooled']
        for item in [pooled, *validation['sources'].values(), *context.values()]:
            validate_aggregate(item)
        for partition in (validation['sources'], context):
            for mask in MASKS:
                for count in set(COUNT_FOR.values()):
                    require(sum(item['mask_levels'][mask][count] for item in partition.values())
                            == pooled['mask_levels'][mask][count], 'Stratum token counts do not pool')
                for metric in COUNT_FOR:
                    require(near(sum(item['mask_levels'][mask][metric + '_sum'] for item in partition.values()),
                                 pooled['mask_levels'][mask][metric + '_sum']), 'Stratum numerators do not pool')
        require(number(pooled['mask_mean']['aa_loss']), 'Overall reconstruction loss unavailable')


def cell(variant, seed):
    return f'{variant}/s{seed}'


def factorial_effects(values):
    """Full 2^3 contrast: mean positive-sign cells minus negative-sign cells.

    No imputation, fitting, significance estimate, or biological interpretation.
    Main, pair and triple contrasts use the same 1/4 coefficient convention.
    """
    require(set(values) == set(FACTORIAL) and all(number(value) for value in values.values()),
            'A factorial contrast requires all eight finite observed cells')
    effects = {}
    for order in (1, 2, 3):
        for factors in itertools.combinations(range(3), order):
            name = ''.join('GSL'[index] for index in factors)
            effects[name] = sum(math.prod(1 if variant[index + 1] == '1' else -1 for index in factors)
                                * value for variant, value in values.items()) / 4.
    return effects


def collect(jobs_file, pool):
    evidence, warnings, runs, seen, job_ids, initial = {}, [], [], set(), set(), {}
    jobs = read_json(jobs_file, evidence)
    reference = None
    for entry in jobs['jobs']:
        variant, seed, job_id, alias = entry['variant'], entry['seed'], entry['job_id'], entry['alias']
        require(variant in VARIANTS and type(seed) is int and seed in SEEDS, 'Unexpected design cell')
        require((variant, seed) not in seen and job_id not in job_ids, 'Duplicate cell/attempt requires explicit selection')
        require(isinstance(job_id, str) and Path(job_id).name == job_id and job_id not in ('.', '..'), 'Invalid job ID')
        job = entry['job']
        require(isinstance(alias, str) and alias.startswith('p02_') and job['variant'] == alias
                and job['mode'] == 'train' and job['steps'] == 3750
                and Path(job['data_dir']).is_absolute(), 'Expected explicit p02 training job binding')
        seen.add((variant, seed)); job_ids.add(job_id)
        directory = pool / 'runs' / job_id
        require(not directory.is_symlink(), 'Run metadata directory cannot be a symlink')
        provenance = read_json(directory / 'run_provenance.json', evidence, optional=True)
        progress = read_json(directory / 'progress.json', evidence, optional=True)
        complete = read_json(directory / 'COMPLETE.json', evidence, optional=True)
        summary = read_json(directory / 'summary.json', evidence, optional=True)
        result = read_json(pool / 'results' / (job_id + '.json'), evidence, optional=True)
        status = ('running_metadata_present' if (pool / 'running' / (job_id + '.json')).exists() else
                  'pending' if (pool / 'pending' / (job_id + '.json')).exists() else 'not_observed')
        if result:
            status = result.get('state', 'result_without_state')
        elif complete:
            status = 'complete_pool_unconfirmed'
        run = dict(job_id=job_id, alias=alias, variant=variant, seed=seed, status=status,
                   directory=str(directory), provenance=provenance, progress=progress,
                   completion=complete, pool_result=result, points=read_curve(directory / 'curve.jsonl', evidence, warnings),
                   terminal_verified=False)
        if provenance:
            require(provenance['variant'] == variant and provenance['seed'] == seed, 'Job/provenance identity mismatch')
            current = identity(provenance)
            require(reference is None or current == reference, 'Data/source/model/schedule identities differ between runs')
            reference = current
            require(Path(provenance['data']['path']) == Path(job['data_dir']) / 'sample_manifest.json',
                    'Job/provenance data path mismatch')
            common_sha = provenance['common_initialization_sha256']
            require(seed not in initial or initial[seed] == common_sha, 'Same-seed common initialization differs')
            initial[seed] = common_sha
            validate_points(run)
        else:
            require(not run['points'] and complete is None, 'Curve/completion without provenance')
        if complete:
            require(complete['status'] == 'completed' and complete['global_step'] == 3750
                    and complete['kind'] == provenance['kind'] and complete['formal_training'] is False
                    and complete['biological_quality_claim'] is False, 'Completion is not a full p02 cycle')
            require(run['points'] and run['points'][-1] == summary and summary['step'] == 3750,
                    'Completed summary does not match last actual curve point')
            require(complete['summary_sha256'] == evidence[str(directory / 'summary.json')]['sha256']
                    and complete['provenance_sha256'] == evidence[str(directory / 'run_provenance.json')]['sha256'],
                    'Completion source artifacts changed')
            run['terminal_verified'] = bool(result and result.get('state') == 'succeeded' and result.get('returncode') == 0)
        run['latest_evaluation_step'] = run['points'][-1]['step'] if run['points'] else None
        runs.append(run)
    eligible = [run for run in runs if any(point['step'] > 0 for point in run['points'])]
    shared = set.intersection(*(set(point['step'] for point in run['points']) for run in eligible)) if len(eligible) >= 2 else set()
    step = max((value for value in shared if value > 0), default=None)
    endpoints = {}
    if step is not None:
        budgets = {}
        for run in eligible:
            point = next(point for point in run['points'] if point['step'] == step)
            exposure = tuple(point[key] for key in ('records', 'input_tokens', 'eligible_tokens', 'selected_aa_terminal_tokens'))
            require(run['seed'] not in budgets or exposure == budgets[run['seed']], 'Paired seed exposure/mask budget differs')
            budgets[run['seed']] = exposure
            endpoints[cell(run['variant'], run['seed'])] = point
    final = len(runs) == 22 and all(run['terminal_verified'] for run in runs) and step == 3750
    all_cells = [cell(variant, seed) for variant in VARIANTS for seed in SEEDS]
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), study=STUDY,
        status='final' if final else 'partial' if step is not None else 'awaiting_shared_evaluation',
        final=final, partial=not final, endpoint_step=step, endpoint_cells=sorted(endpoints),
        missing_registered_cells=[name for name in all_cells if name not in {cell(*item) for item in seen}],
        missing_nonzero_evaluation_cells=[name for name in all_cells if name not in
            {cell(run['variant'], run['seed']) for run in eligible}],
        missing_endpoint_cells=[name for name in all_cells if name not in endpoints],
        jobs_file=str(jobs_file), pool_dir=str(pool), runs=runs, endpoints=endpoints,
        common_initialization_by_seed=initial, input_snapshots=evidence, warnings=warnings,
        source_sha256=sha(Path(__file__).read_bytes()), figures=[], confidence_intervals=False,
        biological_quality_claim=False, formal_training=False,
        scope='Observed reconstruction metrics only; positive relation tokens do not measure discrimination',
        interpretation='Each dot is one seed. Missing values stay absent. Cross terms are full 2^3 signed contrasts, not significance tests.')
    report.update(derive(endpoints, step))
    return report


def derive(endpoints, step):
    pairs, interactions, removals, incomplete = [], [], [], []
    for seed in SEEDS:
        baseline = endpoints.get(cell('f000', seed))
        for variant in VARIANTS[1:]:
            point = endpoints.get(cell(variant, seed))
            if baseline is not None and point is not None:
                for metric in COUNT_FOR:
                    current = point['validation']['token_pooled']['mask_mean'][metric]
                    control = baseline['validation']['token_pooled']['mask_mean'][metric]
                    if current is not None and control is not None:
                        pairs.append(dict(variant=variant, seed=seed, step=step, metric=metric,
                                          difference=current-control, baseline=control, observed=current))
        absent = [variant for variant in FACTORIAL if cell(variant, seed) not in endpoints]
        if absent:
            incomplete.append({'seed': seed, 'missing': absent})
        else:
            values = {variant: endpoints[cell(variant, seed)]['validation']['token_pooled']['mask_mean']['aa_loss']
                      for variant in FACTORIAL}
            interactions.extend(dict(seed=seed, step=step, metric='aa_loss', contrast=name, value=value)
                                for name, value in factorial_effects(values).items())
        full = endpoints.get(cell('f111', seed))
        for factor, variant in zip('GSL', ('f011', 'f101', 'f110')):
            removed = endpoints.get(cell(variant, seed))
            if full is not None and removed is not None:
                a = removed['validation']['token_pooled']['mask_mean']['aa_loss']
                b = full['validation']['token_pooled']['mask_mean']['aa_loss']
                removals.append(dict(seed=seed, step=step, removed=factor, variant=variant,
                                     metric='aa_loss', difference=a-b, observed=a, baseline=b))
    return dict(paired_deltas=pairs, factorial_contrasts=interactions,
                factorial_missing_by_seed=incomplete, component_removal_deltas=removals)


def csv_table(path, rows, columns):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(prefix, report):
    rows = []
    for run in report['runs']:
        for point in run['points']:
            validation = point['validation']
            strata = [('pooled', 'ALL', validation['token_pooled'], validation['records'])]
            strata += [(kind, name, item, item['records']) for kind, key in
                       [('source', 'sources'), ('context', 'context_strata')]
                       for name, item in validation[key].items()]
            for kind, name, result, records in strata:
                for mask in (*MASKS, 'mask_mean'):
                    values = result['mask_mean'] if mask == 'mask_mean' else result['mask_levels'][mask]
                    row = {key: run[key] for key in ('job_id', 'variant', 'seed', 'status')}
                    row.update({key: point[key] for key in ('step', 'records', 'input_tokens', 'train_seconds')})
                    row.update(stratum_kind=kind, stratum=name, validation_records=records, mask=mask)
                    row.update({key: values.get(key) for key in (*COUNT_FOR, *set(COUNT_FOR.values()),
                               *(metric + '_sum' for metric in COUNT_FOR))})
                    rows.append(row)
    columns = ['job_id', 'variant', 'seed', 'status', 'step', 'records', 'input_tokens', 'train_seconds',
               'stratum_kind', 'stratum', 'validation_records', 'mask', *COUNT_FOR,
               *sorted(set(COUNT_FOR.values())), *(metric + '_sum' for metric in COUNT_FOR)]
    csv_table(prefix.with_suffix('.csv'), rows, columns)
    csv_table(prefix.with_name(prefix.name + '_paired.csv'), report['paired_deltas'],
              ['variant', 'seed', 'step', 'metric', 'difference', 'baseline', 'observed'])
    csv_table(prefix.with_name(prefix.name + '_interactions.csv'), report['factorial_contrasts'],
              ['seed', 'step', 'metric', 'contrast', 'value'])
    csv_table(prefix.with_name(prefix.name + '_removal.csv'), report['component_removal_deltas'],
              ['seed', 'step', 'removed', 'variant', 'metric', 'difference', 'observed', 'baseline'])


def draw(report, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    endpoints, step = report['endpoints'], report['endpoint_step']
    require(step is not None and len(endpoints) >= 2, 'No shared observed endpoint subset to draw')
    colors = {variant: plt.get_cmap('tab20')(index) for index, variant in enumerate(VARIANTS)}
    aa = lambda point: point['validation']['token_pooled']['mask_mean']['aa_loss']
    suffix = f'{"FINAL" if report["final"] else "PARTIAL"}: {len(endpoints)}/22 cells at shared step {step}'
    def save(figure, name):
        figure.suptitle(report.get('figure_title', 'p02 context ablation') + ' — ' + suffix, fontsize=13)
        figure.text(.5, .005, 'Observed reconstruction only; two seeds, no CI or significance claim. Missing cells are blank. '
                    'Record and per-mask token denominators remain in CSV/JSON.', ha='center', fontsize=8)
        figure.tight_layout(rect=(0, .03, 1, .95))
        for extension in ('svg', 'pdf', 'png'):
            path = prefix.with_name(prefix.name + '_' + name).with_suffix('.' + extension)
            figure.savefig(path, dpi=180, metadata={'Creator': 'plot_v7_context_ablation.py'})
            report['figures'].append(str(path))
        plt.close(figure)
    def missing(axis, text):
        axis.text(.5, .5, text, ha='center', va='center', transform=axis.transAxes, fontsize=9)
    def dots(axis, records, names, field, title):
        for item in records:
            index = names.index(item[field])
            axis.scatter(index + (-.08 if item['seed'] == 42 else .08),
                         item.get('difference', item.get('value')), marker='o' if item['seed'] == 42 else 's')
        axis.axhline(0, color='#64748b', linewidth=.8)
        axis.set_xticks(range(len(names)), names, rotation=30, ha='right')
        axis.set_title(title); axis.grid(alpha=.15)
        if not records:
            missing(axis, 'Required observed matched cells unavailable')
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    curves, paired, cost, memory = axes.flat
    for run in report['runs']:
        if not run['points']:
            continue
        style = dict(color=colors[run['variant']], linestyle='-' if run['seed'] == 42 else '--',
                     marker='o' if run['seed'] == 42 else 's', markersize=2, linewidth=1,
                     label=f'{run["variant"]} s{run["seed"]} [{run["status"]}]')
        curves.plot([point['step'] for point in run['points']], [aa(point) for point in run['points']], **style)
        cost.plot([point['train_seconds'] for point in run['points']], [aa(point) for point in run['points']], **style)
        endpoint = endpoints.get(cell(run['variant'], run['seed']))
        if endpoint and number(endpoint.get('peak_allocated_gib')):
            memory.scatter(endpoint['peak_allocated_gib'], aa(endpoint), color=colors[run['variant']],
                           marker='o' if run['seed'] == 42 else 's')
            memory.annotate(cell(run['variant'], run['seed']), (endpoint['peak_allocated_gib'], aa(endpoint)), fontsize=6)
    curves.set(title='Recorded learning curves (no interpolation)', xlabel='Actual optimizer updates', ylabel='Four-mask mean AA CE')
    curves.legend(fontsize=6, ncol=2)
    cost.set(title='Recorded training cost (evaluation/checkpoint time excluded)', xlabel='Training-loop seconds', ylabel='Four-mask mean AA CE')
    memory.set(title=f'Peak allocated GPU memory at shared step {step}', xlabel='GiB', ylabel='Four-mask mean AA CE')
    dots(paired, [item for item in report['paired_deltas'] if item['variant'] in FACTORIAL and item['metric'] == 'aa_loss'],
         list(FACTORIAL[1:]), 'variant', 'Same-seed difference from f000 (candidate − baseline)')
    save(figure, 'overview')
    figure, axes = plt.subplots(1, 3, figsize=(19, 6))
    matrix = np.array([[aa(endpoints[cell(variant, seed)]) if cell(variant, seed) in endpoints else np.nan
                       for variant in FACTORIAL] for seed in SEEDS])
    axes[0].set_xticks(range(8), FACTORIAL, rotation=45); axes[0].set_yticks(range(2), ['s42', 's43'])
    axes[0].set_title('Complete fGSL design; unobserved cells blank')
    if np.isfinite(matrix).any():
        shown = axes[0].imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap='viridis')
        for row, col in zip(*np.where(np.isfinite(matrix))):
            axes[0].text(col, row, f'{matrix[row,col]:.3f}', ha='center', va='center', color='white', fontsize=8)
        figure.colorbar(shown, ax=axes[0], label='Four-mask mean AA CE', shrink=.6)
    else:
        missing(axes[0], 'No factorial cells at this observed endpoint')
    dots(axes[1], report['factorial_contrasts'], ['G', 'S', 'L', 'GS', 'GL', 'SL', 'GSL'], 'contrast',
         'Full-eight-cell signed factorial effects (1/4 convention)')
    dots(axes[2], report['component_removal_deltas'], list('GSL'), 'removed', 'Remove one component: removed − f111')
    save(figure, 'factorial')
    labels = [cell(variant, seed) for variant in VARIANTS for seed in SEEDS if cell(variant, seed) in endpoints]
    sources = sorted(next(iter(endpoints.values()))['validation']['sources'])
    figure, axes = plt.subplots(2, 1, figsize=(max(12, len(labels)*.65), max(9, len(sources)*.4+5)))
    for axis, key, strata in ((axes[0], 'sources', sources),
                             (axes[1], 'context_strata', ['with_context', 'without_context'])):
        matrix, names = [], []
        for stratum in strata:
            row, counts = [], set()
            for label in labels:
                result = endpoints[label]['validation'][key].get(stratum)
                row.append(result['mask_mean']['aa_loss'] if result is not None else np.nan)
                if result is not None:
                    counts.add(result['records'])
            matrix.append(row); names.append(f'{stratum} (n={"/".join(map(str, sorted(counts))) or "missing"})')
        data = np.array(matrix, dtype=float)
        image = axis.imshow(np.ma.masked_invalid(data), aspect='auto', cmap='viridis')
        axis.set_xticks(range(len(labels)), labels, rotation=65, ha='right', fontsize=8)
        axis.set_yticks(range(len(names)), names, fontsize=8)
        axis.set_title(f'{key}: matched step {step}; record denominators shown; per-mask token counts in CSV')
        figure.colorbar(image, ax=axis, label='Four-mask mean AA CE', shrink=.8)
    save(figure, 'strata')
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, metric in zip(axes, ('aa_loss', 'relation_loss', 'relation_accuracy')):
        dots(axis, [item for item in report['paired_deltas'] if item['variant'] in CONTROLS and item['metric'] == metric],
             list(CONTROLS), 'variant', f'{metric}: separate controls minus same-seed f000')
    save(figure, 'separate_controls')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--jobs-file', required=True, type=Path)
    parser.add_argument('--pool-dir', required=True, type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    require(all(path.is_absolute() for path in (args.jobs_file, args.pool_dir, args.output_prefix)), 'Use absolute paths')
    report = collect(args.jobs_file, args.pool_dir)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix, report)
    if report['endpoint_step'] is not None:
        draw(report, args.output_prefix)
    else:
        report['message'] = 'Awaiting at least two source-matched runs with a shared positive completed evaluation; no numerical figures written.'
        report['preexisting_figures_not_updated'] = [str(path) for path in args.output_prefix.parent.glob(args.output_prefix.name + '_*')
                                                    if path.suffix in ('.svg', '.pdf', '.png')]
    args.output_prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'final', 'endpoint_step', 'missing_endpoint_cells', 'figures')}))


if __name__ == '__main__':
    main()
