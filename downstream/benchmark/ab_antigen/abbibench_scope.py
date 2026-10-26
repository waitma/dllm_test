"""Pin AbBiBench's leaderboard assay scope without deleting source data.

List suite tasks: python -m downstream.benchmark.ab_antigen.abbibench_scope
Aggregate an existing suite: ... --run RUN --out SUMMARY.json
Extra assays require --include-extra-assays; the headline remains the fixed 16.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

SOURCE = 'https://github.com/MSBMI-SAFE/AbBiBench/blob/7d9c73e23c3c535fbb73ef073cf2fb9633e1fda8/README.md'
PROTOCOL = 'abbibench_leaderboard16_v1'
ASSAY_MAP = {
    '1mhp': '1mhp_LC_benchmarking_data',
    '1mlc': '1mlc_benchmarking_data',
    '1n8z': '1n8z_benchmarking_data',
    '2fjg': '2fjg_benchmarking_data',
    '3gbn_h1': '3gbn_h1_benchmarking_data',
    '3gbn_h9': '3gbn_h9_benchmarking_data',
    '4fqi_h1': '4fqi_h1_benchmarking_data',
    '4fqi_h3': '4fqi_h3_benchmarking_data',
    '4d5_her2': '4d5_her2_benchmarking_data_trimmed',
    '5a12_ang2': '5a12_ang2_benchmarking_data',
    '5a12_vegf': '5a12_vegf_benchmarking_data',
    'aayl50': 'aayl50_LC_benchmarking_data',
    'aayl49': 'aayl49_benchmarking_data',
    'aayl49_ML': 'aayl49_ML_benchmarking_data',
    'aayl51': 'aayl51_benchmarking_data',
    'aayl52': 'aayl52_LC_benchmarking_data',
}
ASSAYS = tuple(ASSAY_MAP.values())
EXTRA_ASSAYS = ('g6_LC_benchmarking_data',)


def validate_assay(assay, *, include_extra_assays=False):
    if not assay or assay in ASSAYS:
        return
    if assay in EXTRA_ASSAYS:
        if include_extra_assays:
            return
        raise ValueError(f'{assay} is outside the official 16-assay leaderboard; '
                         'use --include-extra-assays for a supplementary result')
    raise ValueError(f'Unknown AbBiBench assay: {assay}')


def select_rows(rows, *, assay=None, include_extra_assays=False):
    """Select an evaluation view, preserving every selected row and its order."""
    validate_assay(assay, include_extra_assays=include_extra_assays)
    unknown = {r.get('assay') for r in rows} - set(ASSAYS) - set(EXTRA_ASSAYS)
    if unknown:
        raise ValueError(f'Unknown AbBiBench assays: {sorted(map(str, unknown))}')
    allowed = set(ASSAYS) | (set(EXTRA_ASSAYS) if include_extra_assays else set())
    return [r for r in rows if r['assay'] in allowed and (not assay or r['assay'] == assay)]


def select_tasks(tasks, *, include_extra_assays=False):
    selected = []
    for task in tasks:
        name, _, assay = task.partition(':')
        if name == 'ag-abbibench':
            validate_assay(assay, include_extra_assays=True)
            if assay in EXTRA_ASSAYS and not include_extra_assays:
                continue
        selected.append(task)
    return selected


def summarize_scores(scores):
    """Do not publish a full-16 mean when assays are missing or undefined."""
    unknown = set(scores) - set(ASSAYS) - set(EXTRA_ASSAYS)
    if unknown:
        raise ValueError(f'Unknown AbBiBench assays: {sorted(unknown)}')
    valid = {k: v for k, v in scores.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
    selected = {k: valid[k] for k in ASSAYS if k in valid}
    missing = [k for k in ASSAYS if k not in selected]
    partial_mean = mean(selected.values()) if selected else None
    return dict(protocol=PROTOCOL, source=SOURCE, expected_assays=len(ASSAYS),
                completed_assays=len(selected), status='partial' if missing else 'complete',
                mean_spearman=partial_mean if not missing else None,
                partial_mean_spearman=partial_mean, missing_assays=missing,
                per_assay=selected,
                supplementary_assays={k: scores[k] for k in EXTRA_ASSAYS if k in scores})


def summarize_run(run):
    scores, sources = {}, {}
    for assay in ASSAYS + EXTRA_ASSAYS:
        folder = Path(run) / ('ag-abbibench:' + assay)
        path = folder / 'metrics.json'
        status = folder / 'status.json'
        if not path.exists():
            continue
        if status.exists() and json.loads(status.read_text()).get('status') != 'success':
            continue
        data = json.loads(path.read_text())
        if data.get('status') != 'complete':
            continue
        groups = data['per_assay']['groups'] + data['per_assay'].get('excluded', [])
        if {g['group'] for g in groups} != {assay}:
            raise ValueError(f'Assay/result mismatch: {path}')
        scores[assay] = data['per_assay']['macro']['spearman']['mean']
        sources[assay] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return {**summarize_scores(scores), 'run': str(Path(run).resolve()), 'sources': sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--include-extra-assays', action='store_true')
    args = parser.parse_args()
    result = summarize_run(args.run) if args.run else dict(
        protocol=PROTOCOL, source=SOURCE, tasks=select_tasks(
            ['ag-abbibench:' + a for a in ASSAYS + EXTRA_ASSAYS],
            include_extra_assays=args.include_extra_assays))
    content = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists() and args.out.read_text() != content:
            raise FileExistsError('Use a new output for changed results')
        args.out.write_text(content)
    else:
        print(content, end='')


def summarize_groups(per_assay):
    return summarize_scores({g['group']: g.get('spearman')
                             for g in per_assay['groups'] + per_assay.get('excluded', [])})


if __name__ == '__main__':
    main()
