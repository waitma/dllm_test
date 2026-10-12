"""Export recorded V7 pilot ablations without importing models or reading sequences.

Activate conda pllm, then run this script with --jobs-file /absolute/jobs.json
--pool-dir /absolute/pool --output-prefix /absolute/figures/v7_pilot.
Outputs SVG/PDF/PNG plus long-form CSV and a JSON evidence snapshot. Missing runs
and seeds remain missing. Only shared, completed evaluation steps are compared;
all recorded curves remain visible. No confidence intervals are inferred.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap

VARIANTS = {'r20': 'r20_refined_control', 'r21': 'r21_refined_head_gate',
            'r24': 'r24_refined_fixed_scale', 'r26': 'r26_refined_local_global'}
LABELS = {'r20': 'R20 baseline', 'r21': 'R21 head gate',
          'r24': 'R24 residual scale', 'r26': 'R26 local/global'}
COLORS = {'r20': '#475569', 'r21': '#0072B2', 'r24': '#D55E00', 'r26': '#009E73'}
MASKS = ('mask25', 'mask50', 'mask75', 'mask100')
METRICS = ('loss', 'aa_loss', 'terminal_loss', 'aa_accuracy', 'terminal_accuracy')
COMPARABLE = ('config', 'source_sha256', 'micro_batch', 'grad_accum', 'global_batch',
              'scheduler_horizon', 'warmup', 'learning_rate', 'loss', 'mask_protocol',
              'evaluation_interval', 'torch_version')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, evidence, optional=False):
    if optional and not path.exists():
        return None
    raw = path.read_bytes()
    evidence[str(path)] = {'sha256': digest(raw), 'bytes': len(raw)}
    return json.loads(raw)


def read_curve(path, evidence, warnings):
    if not path.exists():
        return []
    raw = path.read_bytes()
    evidence[str(path)] = {'sha256': digest(raw), 'bytes': len(raw)}
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        warnings.append(f'{path}: ignored incomplete final line ({len(lines[-1])} bytes)')
        lines.pop()
    points = [json.loads(line) for line in lines]
    steps = [point['step'] for point in points]
    if any(type(step) is not int or step < 0 for step in steps) or steps != sorted(set(steps)):
        raise ValueError(f'Nonunique or unordered evaluation steps: {path}')
    return points


def comparable_identity(provenance):
    if (provenance.get('kind') != 'exploratory_architecture_reconstruction'
            or provenance.get('formal_training') is not False
            or provenance.get('biological_quality_claim') is not False
            or provenance.get('generation') is not False):
        raise ValueError('Only the real subset reconstruction pilot is eligible, never CPU smoke')
    identity = {key: provenance[key] for key in COMPARABLE}
    if not identity['source_sha256'] or not isinstance(identity['source_sha256'], dict):
        raise ValueError('Source identity is missing')
    data = provenance['data']
    identity['data_sha256'] = data['sha256']
    identity['data_manifest'] = data['manifest']
    config = identity['config']
    if ([config[key] for key in ('d_model', 'n_layers', 'n_heads', 'mlp_hidden_size', 'max_length')]
            != [768, 8, 12, 3072, 2048]
            or [identity[key] for key in ('micro_batch', 'grad_accum', 'global_batch', 'scheduler_horizon')]
            != [4, 8, 32, 3125]):
        raise ValueError('Not the fixed full-size, global32, horizon3125 pilot')
    return identity


def validate_points(run):
    provenance, points = run['provenance'], run['points']
    expected = provenance['data']['manifest']['files']['valid']
    previous_tokens, previous_seconds = -1, -1.
    for point in points:
        step = point['step']
        if step > provenance['scheduler_horizon'] or point['records'] != step * provenance['global_batch']:
            raise ValueError(f'Exposure/update mismatch in {run["job_id"]} at {step}')
        if point['input_tokens'] < previous_tokens or point['train_seconds'] < previous_seconds:
            raise ValueError(f'Nonmonotonic token/time budget in {run["job_id"]}')
        previous_tokens, previous_seconds = point['input_tokens'], point['train_seconds']
        validation = point['validation']
        if (validation['records'] != expected['records']
                or {key: value['records'] for key, value in validation['sources'].items()} != expected['source_counts']):
            raise ValueError(f'Validation source/count mismatch in {run["job_id"]}')
        for result in [validation['token_pooled'], *validation['sources'].values()]:
            if set(result['mask_levels']) != set(MASKS):
                raise ValueError('Missing validation mask level')
            for mask in MASKS:
                for metric in METRICS:
                    value = result['mask_levels'][mask][metric]
                    if value is not None and (not isinstance(value, (float, int)) or not math.isfinite(value)):
                        raise ValueError('Nonfinite recorded validation metric')
        if not isinstance(validation['token_pooled']['mask_mean']['aa_loss'], (float, int)):
            raise ValueError('Overall AA cross-entropy is unavailable')


def collect(jobs_file, pool):
    evidence, warnings, runs, identities, seeds = {}, [], [], [], {}
    jobs = read_json(jobs_file, evidence)
    seen = set()
    for entry in jobs['jobs']:
        job_id = entry['job_id']
        if not isinstance(job_id, str) or Path(job_id).name != job_id or job_id in ('.', '..'):
            raise ValueError('Invalid pool job identity')
        match = re.search(r'(?:^|_)(r20|r21|r24|r26)_seed(42|43)(?:_|$)', entry.get('alias', job_id))
        if not match:
            raise ValueError(f'Unexpected candidate/seed alias: {job_id}')
        variant, seed = match.group(1), int(match.group(2))
        if (variant, seed) in seen:
            raise ValueError(f'Duplicate candidate/seed attempts need explicit selection: {variant}/{seed}')
        seen.add((variant, seed))
        directory = pool / 'runs' / job_id
        provenance = read_json(directory / 'run_provenance.json', evidence, optional=True)
        progress = read_json(directory / 'progress.json', evidence, optional=True)
        complete = read_json(directory / 'COMPLETE.json', evidence, optional=True)
        result = read_json(pool / 'results' / (job_id + '.json'), evidence, optional=True)
        status = ('running' if (pool / 'running' / (job_id + '.json')).exists() else
                  'pending' if (pool / 'pending' / (job_id + '.json')).exists() else 'pool_state_unknown')
        if result:
            status = result.get('state', 'pool_result_without_state')
        elif complete:
            status = 'training_complete_pool_unconfirmed'
        points = read_curve(directory / 'curve.jsonl', evidence, warnings)
        run = dict(job_id=job_id, variant=variant, seed=seed, status=status, directory=str(directory),
                   provenance=provenance, progress=progress, completion=complete, pool_result=result,
                   points=points, latest_evaluation_step=points[-1]['step'] if points else None,
                   progress_step=progress.get('step') if progress else None)
        if provenance:
            if provenance['variant'] != VARIANTS[variant] or provenance['seed'] != seed:
                raise ValueError(f'Alias/provenance mismatch for {job_id}')
            identity = comparable_identity(provenance)
            if identities and identity != identities[0]:
                changed = [key for key in identity if identity[key] != identities[0][key]]
                raise ValueError(f'Incomparable run {job_id}: {changed}')
            identities.append(identity)
            initialization = provenance['common_initialization_sha256']
            if seed in seeds and initialization != seeds[seed]:
                raise ValueError(f'Common initialization differs for seed {seed}')
            seeds[seed] = initialization
            validate_points(run)
        elif points:
            raise ValueError(f'Curve without provenance: {job_id}')
        runs.append(run)
    available = [run for run in runs if run['points']]
    common = set.intersection(*(set(point['step'] for point in run['points']) for run in available)) if available else set()
    # A partial launch is not an eight-run ablation. Wait for all fixed cells.
    common_step = max((step for step in common if step > 0), default=None) if len(available) == 8 else None
    if common_step:
        by_seed = {}
        for run in available:
            point = next(point for point in run['points'] if point['step'] == common_step)
            budget = tuple(point[key] for key in ('records', 'input_tokens', 'eligible_tokens', 'selected_aa_terminal_tokens'))
            if run['seed'] in by_seed and budget != by_seed[run['seed']]:
                raise ValueError(f'Paired seed {run["seed"]} token/mask exposure differs at common step')
            by_seed[run['seed']] = budget
    missing = [f'{variant}/s{seed}' for variant in VARIANTS for seed in (42, 43)
               if not any(run['variant'] == variant and run['seed'] == seed and run['points'] for run in runs)]
    return dict(created_at=datetime.now(timezone.utc).isoformat(), kind='exploratory_pilot_ablation',
        formal_training=False, biological_quality_claim=False, confidence_intervals=False,
        jobs_file=str(jobs_file), pool_dir=str(pool), runs=runs, comparable=bool(available),
        common_evaluation_step=common_step,
        endpoint_scope='all_eight_matched_runs' if common_step else 'partial_or_initial_curves_only_no_endpoint',
        design='Four existing variants times two training seeds; not a complete three-factor factorial design',
        missing_curves=missing, warnings=warnings, input_snapshots=evidence,
        script_sha256=digest(Path(__file__).read_bytes()), figures=[])


def write_csv(path, report):
    columns = ['job_id', 'variant', 'seed', 'status', 'step', 'records', 'input_tokens',
               'train_seconds', 'source', 'validation_records', 'mask', *METRICS,
               'selected_count', 'aa_count', 'terminal_count']
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for run in report['runs']:
            for point in run['points']:
                validation = point['validation']
                for source, result in [('ALL_TOKEN_POOLED', validation['token_pooled']), *validation['sources'].items()]:
                    for mask in (*MASKS, 'mask_mean'):
                        values = result['mask_mean'] if mask == 'mask_mean' else result['mask_levels'][mask]
                        row = {key: run[key] for key in ('job_id', 'variant', 'seed', 'status')}
                        row.update({key: point[key] for key in ('step', 'records', 'input_tokens', 'train_seconds')})
                        row.update(source=source, validation_records=result.get('records', validation['records']), mask=mask)
                        row.update({key: values.get(key) for key in (*METRICS, 'selected_count', 'aa_count', 'terminal_count')})
                        writer.writerow(row)


def draw(report, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    runs = sorted((run for run in report['runs'] if run['points']), key=lambda run: (run['variant'], run['seed']))
    common = report['common_evaluation_step']
    endpoints = {(run['variant'], run['seed']): next(point for point in run['points'] if point['step'] == common)
                 for run in runs} if common else {}
    source_counts = runs[0]['provenance']['data']['manifest']['files']['valid']['source_counts']
    figure, axes = plt.subplots(2, 2, figsize=(16, max(11, min(16, 8 + len(source_counts) * .16))))
    curve, paired, heat, cost = axes.flat
    aa_ce = lambda point: point['validation']['token_pooled']['mask_mean']['aa_loss']
    for run in runs:
        points, variant, seed = run['points'], run['variant'], run['seed']
        label = f'{LABELS[variant]} / s{seed} ({run["status"]}; step {points[-1]["step"]})'
        style = dict(color=COLORS[variant], marker='o' if seed == 42 else 's', markersize=3,
                     linewidth=1.1, linestyle='-' if seed == 42 else '--', label=label)
        curve.plot([point['input_tokens'] / 1e6 for point in points], [aa_ce(point) for point in points], **style)
        cost.plot([point['train_seconds'] for point in points], [aa_ce(point) for point in points], **style)
    curve.set(title='A  Recorded learning curves (all available evaluations)',
              xlabel='Cumulative attended input tokens (millions)', ylabel='Four-mask mean AA cross-entropy')
    cost.set(title='D  Training cost / reconstruction trade-off', xlabel='Single-GPU training-loop seconds',
             ylabel='Four-mask mean AA cross-entropy')
    cost.text(.01, .01, 'Evaluation and checkpoint time excluded; no interpolation.', transform=cost.transAxes, fontsize=8)
    paired.axhline(0, color='#64748b', linewidth=.9)
    pairs = []
    for index, variant in enumerate(('r21', 'r24', 'r26')):
        for seed in (42, 43):
            if (variant, seed) in endpoints and ('r20', seed) in endpoints:
                delta = aa_ce(endpoints[variant, seed]) - aa_ce(endpoints['r20', seed])
                paired.scatter(index + (-.09 if seed == 42 else .09), delta, color=COLORS[variant],
                               marker='o' if seed == 42 else 's', s=45)
                pairs.append(dict(variant=variant, seed=seed, step=common, aa_ce_delta=delta))
    paired.set_xticks(range(3), [LABELS[key] for key in ('r21', 'r24', 'r26')])
    paired.set(title=f'B  Paired endpoint difference at common step {common or "not yet available"}',
               ylabel='Candidate minus same-seed R20 AA CE (lower is better)')
    if not pairs:
        paired.text(.5, .5, 'No nonzero common evaluation with a matched baseline seed', ha='center',
                    va='center', transform=paired.transAxes, fontsize=9, wrap=True)
    else:
        paired.text(.02, .02, 'Each dot is one paired seed; no CI or significance claim.', transform=paired.transAxes, fontsize=8)
    report['paired_endpoint_deltas'] = pairs
    if endpoints:
        sources = sorted(source_counts)
        matrix = [[endpoints[run['variant'], run['seed']]['validation']['sources'][source]['mask_mean']['aa_loss']
                   for run in runs] for source in sources]
        matrix = np.array([[np.nan if value is None else value for value in row] for row in matrix])
        shown = heat.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap='viridis')
        heat.set_xticks(range(len(runs)), [f'{run["variant"].upper()}\ns{run["seed"]}' for run in runs])
        heat.set_yticks(range(len(sources)), [f'{source} (n={source_counts[source]})' for source in sources], fontsize=8)
        figure.colorbar(shown, ax=heat, fraction=.035, pad=.02, label='Four-mask mean AA CE')
        heat.set_title(f'C  Source-specific reconstruction, step {common} (record counts)')
    else:
        heat.text(.5, .5, 'Awaiting a shared nonzero evaluation step\nSource counts are recorded in JSON/CSV',
                  ha='center', va='center', transform=heat.transAxes)
        heat.set_title('C  Source-specific reconstruction: endpoint unavailable')
    for axis in (curve, paired, cost):
        axis.grid(alpha=.18)
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('V7 exploratory architecture pilot — recorded evidence, not a final architecture winner', fontsize=15)
    handles, labels = curve.get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .045), ncol=2, fontsize=8, frameon=False)
    missing = ', '.join(report['missing_curves']) or 'none'
    counts = runs[0]['provenance']['data']['manifest']['files']
    note = (f'Train subset n={counts["train"]["records"]:,}; validation n={counts["valid"]["records"]:,}. '
            f'Available curves {len(runs)}/8; missing curves: {missing}. Common endpoint: {common or "none"}. '
            'Two seeds; not a complete three-factor factorial. No downstream biological performance claim.')
    figure.text(.5, .01, '\n'.join(textwrap.wrap(note, width=170)), ha='center', fontsize=8, color='#475569')
    figure.tight_layout(rect=(0, .16, 1, .95))
    for extension in ('svg', 'pdf', 'png'):
        path = prefix.with_suffix('.' + extension)
        figure.savefig(path, dpi=180, metadata={'Creator': 'plot_v7_architecture_pilot.py'})
        report['figures'].append(str(path))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs-file', required=True, type=Path)
    parser.add_argument('--pool-dir', required=True, type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.jobs_file, args.pool_dir, args.output_prefix)):
        parser.error('All paths must be absolute')
    report = collect(args.jobs_file, args.pool_dir)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix('.csv'), report)
    if report['common_evaluation_step'] is not None:
        draw(report, args.output_prefix)
        report['status'] = 'recorded_evidence_available'
    else:
        report['status'] = 'awaiting_shared_evaluation'
        report['message'] = '暂无八项共同非零步评价；未生成图像。 No shared positive evaluation across all eight runs; no figures generated.'
        report['preexisting_figures_not_updated'] = [str(args.output_prefix.with_suffix('.' + ext))
            for ext in ('svg', 'pdf', 'png') if args.output_prefix.with_suffix('.' + ext).exists()]
    args.output_prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'common_evaluation_step', 'missing_curves', 'figures')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
