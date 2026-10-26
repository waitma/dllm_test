"""Apply a predeclared confirmation rule with paired cluster/seed bootstrap.

Activate pllm; python -m scripts.research.analyze_fixed_chain_confirmation
    --run /absolute/confirmation --protocol /absolute/selection_protocol.json
Reads completed evaluation artifacts only; never opens training or benchmark rows.
"""
import argparse
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import numpy as np

VIEWS = ('observed_mask100', 'observed_first_chain', 'observed_second_chain', 'cdr')


def domains(values):
    return np.array([values[0], (values[1] + values[2]) / 2, values[3]])


def primary_change(candidate, control):
    def value(metrics):
        return domains([metrics[view]['macro']['aa_loss'] for view in VIEWS])
    return float(np.mean(value(candidate) / value(control) - 1))


def bootstrap_primary(candidates, controls, replicates=2000, seed=20261014, interval=(.025, .975)):
    """Crossed resampling of validation clusters and paired training seeds."""
    frame = controls[0]['observed_mask100']['records']
    clusters = {source: sorted({row['cluster'] for row in records}) for source, records in frame.items()}
    indexes = {source: {key: i for i, key in enumerate(keys)} for source, keys in clusters.items()}

    def pack(metrics, reference):
        result = []
        for view in VIEWS:
            by_source = {}
            if set(metrics[view]['records']) != set(reference[view]['records']):
                raise ValueError('Paired validation source populations differ')
            for source, rows in metrics[view]['records'].items():
                other = reference[view]['records'][source]
                if ([(r['row'], r['cluster'], r['totals'][3]) for r in rows]
                        != [(r['row'], r['cluster'], r['totals'][3]) for r in other]):
                    raise ValueError('Paired validation records or AA counts differ')
                values = np.zeros((len(clusters[source]), 2), dtype=np.float64)
                for row in rows:
                    values[indexes[source][row['cluster']]] += [row['totals'][2], row['totals'][3]]
                by_source[source] = values
            result.append(by_source)
        return result

    control = [pack(m, controls[0]) for m in controls]
    candidate = [pack(m, controls[i]) for i, m in enumerate(candidates)]

    def score(packed, weights):
        views = []
        for by_source in packed:
            means = []
            for source, values in by_source.items():
                total, count = weights[source] @ values
                if count:
                    means.append(total / count)
            if not means:
                raise ValueError('Bootstrap sample has no AA targets in a primary domain')
            views.append(np.mean(means))
        return domains(views)

    rng, values = np.random.default_rng(seed), []
    for _ in range(replicates):
        weights = {source: rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)))
                   for source, keys in clusters.items()}
        seeds = rng.integers(len(controls), size=len(controls))
        values.append(np.mean([np.mean(score(candidate[i], weights) / score(control[i], weights) - 1)
                               for i in seeds]))
    return np.quantile(values, interval).tolist()


def generation_summary(report):
    values = defaultdict(lambda: [0, 0, 0.0])
    for record in report['records']:
        for chain in record['chains']:
            item = values[record['source']]
            item[1] += 1
            if chain['length_error'] is not None:
                item[0] += 1
                item[2] += chain['length_error']
    lengths = [v[2] / v[0] for v in values.values() if v[0]]
    if not values:
        raise ValueError('No generation records')
    return {'validity': float(np.mean([v[0] / v[1] for v in values.values()])),
            'valid_only_length_mae': float(np.mean(lengths)) if lengths else None,
            'valid_length_sources': len(lengths),
            'valid': sum(v[0] for v in values.values()), 'chains': sum(v[1] for v in values.values())}


def source_change(candidate, control, source):
    ratios = []
    for group in [('observed_mask100',), ('observed_first_chain', 'observed_second_chain'), ('cdr',)]:
        left, right = [], []
        for view in group:
            a = candidate[view]['sources'].get(source, {}).get('aa_loss')
            b = control[view]['sources'].get(source, {}).get('aa_loss')
            if a is not None and b is not None:
                left.append(a)
                right.append(b)
        if left:
            ratios.append(np.mean(left) / np.mean(right) - 1)
    return float(np.mean(ratios)) if ratios else None


def analyze(run, protocol):
    models, generations = {}, {}
    for family in protocol['families']:
        models[family], generations[family] = [], []
        for seed in protocol['seeds']:
            path = run / f'{family}_seed{seed}'
            if json.loads((path / 'status.json').read_text())['status'] != 'success':
                raise ValueError('Confirmation incomplete: ' + str(path))
            models[family].append(json.loads((path / 'metrics.json').read_text()))
            generations[family].append(generation_summary(json.loads((path / 'generation.json').read_text())))
    baseline, rules, results = models['continued'], protocol['advancement'], {}
    for family, values in models.items():
        if family == 'continued':
            continue
        changes = [primary_change(a, b) for a, b in zip(values, baseline)]
        interval = bootstrap_primary(values, baseline, protocol['bootstrap']['replicates'],
                                     protocol['bootstrap']['seed'], protocol['bootstrap']['interval'])
        source_changes = {source: float(np.mean([source_change(a, b, source)
                          for a, b in zip(values, baseline)]))
                          for source in baseline[0]['observed_mask100']['sources']}
        partial_changes = {view: float(np.mean([a[view]['macro']['aa_loss'] / b[view]['macro']['aa_loss'] - 1
                           for a, b in zip(values, baseline)]))
                           for view in ('observed_mask50', 'observed_mask75')}
        validity_change = float(np.mean([a['validity'] - b['validity']
                                for a, b in zip(generations[family], generations['continued'])]))
        lengths = [(a['valid_only_length_mae'], b['valid_only_length_mae'])
                   for a, b in zip(generations[family], generations['continued'])]
        length_change = (float(np.mean([a / b - 1 if b else (0. if not a else 1.) for a, b in lengths]))
                         if all(a is not None and b is not None for a, b in lengths) else None)
        checks = {'primary_mean': np.mean(changes) <= rules['mean_primary_relative_change_max'],
                  'primary_ci': interval[1] < rules['primary_ci_upper_max'],
                  'seed_reproduction': sum(x < 0 for x in changes) >= rules['improving_seeds_min'],
                  'source_guard': max(source_changes.values()) <= rules['per_source_primary_relative_regression_max'],
                  'partial_mask_guard': max(partial_changes.values()) <= rules['observed_partial_mask50_75_relative_regression_max'],
                  'generation_validity': validity_change >= -rules['generation_source_macro_validity_drop_max'],
                  'generation_length': length_change is not None and length_change <= rules['generation_valid_only_length_mae_relative_increase_max']}
        results[family] = {'mean_primary_change': float(np.mean(changes)), 'seed_changes': changes,
                          'primary_interval': interval, 'interval_quantiles': protocol['bootstrap']['interval'],
                          'source_changes': source_changes,
                          'partial_mask_changes': partial_changes, 'generation_validity_change': validity_change,
                          'generation_valid_only_length_change': length_change,
                          'checks': {k: bool(v) for k, v in checks.items()}, 'qualifies': bool(all(checks.values()))}
    # Close ties follow the protocol: adapter efficiency, then simpler training loss.
    eligible = [name for name, value in results.items() if value['qualifies']]
    if eligible:
        best = min(results[name]['mean_primary_change'] for name in eligible)
        close = [name for name in eligible if results[name]['mean_primary_change'] <= best + .003]
        preference = ['lora', 'task_mix', 'observed_eos', 'observed_cdr']
        selected = min(close, key=lambda name: preference.index(name) if name in preference else len(preference))
    else:
        selected = 'continued'
    return {'results': results, 'selected_family': selected, 'generation': generations,
            'status': 'confirmation_analyzed_not_final_training_deployed',
            'caveat': 'Exact receptor clusters, not homology clusters; three seeds limit uncertainty estimates.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    binding = json.loads((args.run / 'evaluation_plan.json').read_text())
    if sha(binding['path']) != binding['sha256']:
        raise ValueError('Frozen confirmation plan changed')
    plan = json.loads(Path(binding['path']).read_text())
    if (sha(args.protocol) != plan['protocol_sha256']
            or plan['source_hashes'].get(str(Path(__file__).resolve())) != sha(__file__)):
        raise ValueError('Run the frozen analyzer with its bound protocol')
    result = analyze(args.run, json.loads(args.protocol.read_text()))
    result['analysis_binding'] = {'evaluation_plan': binding, 'protocol_sha256': sha(args.protocol),
                                 'analyzer_sha256': sha(__file__),
                                 'inputs': {str(p): sha(p) for directory in sorted(args.run.iterdir())
                                            if directory.is_dir() for p in directory.iterdir()
                                            if p.name in ('metrics.json', 'generation.json', 'status.json')}}
    (args.run / 'selection.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
