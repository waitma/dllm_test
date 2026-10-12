"""Audit source exposure using counts and integer indices only, never sequence rows.

After activating pllm:
USE_TF=0 python scripts/architecture_prototypes/source_exposure.py --manifest /absolute/dataset_manifest.json --output /outside/prepared/exposure.json
This writes a small planning report, not a sampler, training job, or certificate.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
from itertools import accumulate, islice
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def counts_from_manifest(manifest):
    counts = Counter()
    split = manifest['splits']['train']
    for shard in split['shards']:
        source, size = shard['source'], shard['records']
        if not isinstance(source, str) or not source or type(size) is not int or size < 0:
            raise ValueError('Invalid source/count metadata')
        counts[source] += size
    if not counts or min(counts.values()) <= 0 or type(split['records']) is not int or sum(counts.values()) != split['records']:
        raise ValueError('Empty sources or inconsistent training counts')
    return dict(sorted(counts.items()))


def allocation(counts, budget, alpha):
    """Integer largest-remainder plan, not an implemented weighted sampler."""
    if type(budget) is not int or budget < 1 or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Invalid presentation budget or alpha')
    if not counts or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError('Positive integer counts required')
    weights = {source: size**alpha for source, size in sorted(counts.items())}
    total = sum(weights.values())
    expected = {source: budget*weight/total for source, weight in weights.items()}
    assigned = {source: math.floor(value) for source, value in expected.items()}
    remainder = budget-sum(assigned.values())
    for source in sorted(expected, key=lambda s: (-(expected[s]-assigned[s]), s))[:remainder]:
        assigned[source] += 1
    return dict(alpha=alpha, probability={s: w/total for s, w in weights.items()},
                expected_presentations=expected, apportioned_presentations=assigned)


def describe(counts, presentations, *, cyclic_within_source):
    result = {}
    for source, n in counts.items():
        draws = presentations[source]
        result[source] = dict(records=n, presentations=draws,
            presentations_per_record=draws/n, presentation_fraction=draws/sum(presentations.values()))
        if cyclic_within_source:
            result[source].update(unique_records=min(n, draws), full_source_passes=draws//n,
                                  repeated_presentations=max(0, draws-n))
    return result


def audit(counts, step_budgets=(1000, 5000), *, global_batch=256, seed=42):
    """Exact current natural/equal-source prefixes; tempered plans stay proposals."""
    from scripts.run_immune_research import EpochRandomSampler
    names, sizes = list(counts), list(counts.values())
    if names != sorted(names):
        raise ValueError('Source order must match the sorted runner order')
    if type(global_batch) is not int or global_batch < len(names) or not step_budgets:
        raise ValueError('Invalid batch/budgets')
    if any(type(step) is not int or step < 1 for step in step_budgets) or len(set(step_budgets)) != len(step_budgets):
        raise ValueError('Distinct positive step budgets required')
    total = sum(sizes)
    maximum = max(step_budgets)*global_batch
    if maximum > total:
        raise ValueError('This exact natural-prefix audit is restricted to the first epoch')
    # EpochRandomSampler uses only len(dataset): range carries no data rows.
    sampler = EpochRandomSampler(range(total), seed=seed)
    ends = list(accumulate(sizes))
    observed, prefixes = Counter(), {}
    requested = {step*global_batch for step in step_budgets}
    for position, index in enumerate(islice(iter(sampler), maximum), 1):
        observed[names[bisect_right(ends, index)]] += 1
        if position in requested:
            prefixes[position] = {source: observed[source] for source in names}
    reports = {}
    for step in sorted(step_budgets):
        budget = step*global_batch
        uniform = {source: (budget+len(names)-1-i)//len(names) for i, source in enumerate(names)}
        proposed = []
        for alpha in (.5, .75):
            plan = allocation(counts, budget, alpha)
            plan.update(status='proposal_only_sampler_not_implemented',
                        sources=describe(counts, plan['apportioned_presentations'], cyclic_within_source=False))
            proposed.append(plan)
        reports[str(step)] = dict(presentation_budget=budget,
            natural=dict(kind='exact_existing_EpochRandomSampler_seeded_prefix',
                         sources=describe(counts, prefixes[budget], cyclic_within_source=True)),
            equal_source=dict(kind='exact_existing_FullCoverageSourceSampler_round_robin_prefix',
                              sources=describe(counts, uniform, cyclic_within_source=True)),
            tempered_proposals=proposed)
    return dict(schema='immune.source_exposure_counts.v1', data_scope='manifest counts and integer indices only',
        training_started=False, training_certification=False, seed=seed, global_batch=global_batch,
        source_order=names, training_rows=total, stages=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if args.expected_manifest_sha256 and digest != args.expected_manifest_sha256:
        raise ValueError('Manifest digest changed')
    if args.output.resolve().is_relative_to(manifest_path.parent):
        raise ValueError('Write the planning report outside the prepared directory')
    if args.output.exists():
        raise FileExistsError(args.output)
    counts = counts_from_manifest(json.loads(raw))
    report = audit(counts)
    report.update(manifest=str(manifest_path), manifest_sha256=digest,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), ROOT/'scripts/run_immune_research.py',
            ROOT/'dllm/pipelines/immune_llada/data/balanced_sampling.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'report':str(args.output.resolve()), 'rows':report['training_rows'], 'training_started':False}))


if __name__ == '__main__':
    main()
