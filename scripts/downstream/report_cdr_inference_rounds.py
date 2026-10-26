"""Activate pllm; python -B FILE --out ABS [--finalize].

Without --finalize, audit the R1 oracle intervention and write its paired
diagnostic. Finalization requires all three exact platform jobs to succeed,
complete independent audits, unchanged weights, and frozen confirmation
policies. It writes a reviewable report, not a new production default.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read_predictions(folder):
    cases = defaultdict(dict)
    for line in (folder/'predictions.jsonl').open():
        r = json.loads(line)
        cases[r['policy']][r['id']] = r
    return cases


def paired_interval(current, baseline):
    """Whole sequence-pair resampling, with shared weights across folds."""
    assert set(current) == set(baseline)
    rows = list(current.values())
    groups = {g: i for i, g in enumerate(sorted({r['group'] for r in rows}))}
    folds = {f: i for i, f in enumerate(sorted({r['fold'] for r in rows}))}
    sums = np.zeros((len(groups), len(folds)))
    counts = np.zeros_like(sums)
    for r in rows:
        b = baseline[r['id']]
        assert (r['reference_ids'], r['group'], r['fold']) == (b['reference_ids'], b['group'], b['fold'])
        i, j = groups[r['group']], folds[r['fold']]
        sums[i, j] += (r['correct']-b['correct'])/r['length']*100
        counts[i, j] += 1
    rng = np.random.default_rng(20261015)
    w = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=5000).astype(float)
    denom = w @ counts
    valid = (denom > 0).all(1)
    replicates = ((w[valid] @ sums)/denom[valid]).mean(1)
    return dict(delta_pp=float((sums.sum(0)/counts.sum(0)).mean()),
                paired_HL_group_bootstrap95ci_pp=np.percentile(replicates, [2.5, 97.5]).tolist(),
                valid_replicates=len(replicates), n=len(rows), sequence_pair_groups=len(groups))


def oracle_report(out):
    folder = out/'round1'
    audit = json.loads((folder/'independent_audit.json').read_text())
    assert audit['status'] == 'passed' and audit['transitions_audited']
    assert audit['predictions_sha256'] == sha(folder/'predictions.jsonl')
    cases = read_predictions(folder)
    comparisons = {}
    for steps in (4, 8):
        for mode in ('cdrh1', 'cdrh2', 'cdrh3'):
            take = lambda name: {k: r for k, r in cases[name].items() if r['mode'] == mode}
            oracle, baseline = take(f'oracle{steps}'), take(f'original{steps}')
            result = paired_interval(oracle, baseline)
            result.update(before_aar=audit['scores'][f'original{steps}']['kong:'+mode],
                          oracle_aar=audit['scores'][f'oracle{steps}']['kong:'+mode])
            comparisons[f'oracle{steps}:{mode}'] = result
    report = dict(status='passed', comparisons=comparisons,
        interpretation='Same frozen commit positions; replace only previously committed input tokens with gold. '
                       'Stored output remains model proposals. Tests effects of erroneous generated context; '
                       'cannot distinguish all training/calibration causes and is not a deployable method.',
        limitations='Exploratory subset; unadjusted descriptive intervals, no independent validation.',
        independent_audit_sha256=sha(folder/'independent_audit.json'))
    write(out/'oracle_diagnostic.json', report)
    return report


def finalize(out, oracle):
    plan = json.loads((out/'plan.json').read_text())
    selection = json.loads((out/'selection.json').read_text())
    rows = json.loads((out/'rows.json').read_text())
    expected = {s: {r['id'] for r in rows if r['split'] == s} for s in ('explore', 'confirm')}
    gate = json.loads((out/'gate_cpu.json').read_text())
    assert gate['status'] == 'passed' and gate['plan_sha256'] == sha(out/'plan.json')
    for checks in ('source_sha256', 'external_source_sha256', 'data_sha256'):
        for path, value in plan[checks].items(): assert sha(path) == value, path
    assert sha(out/'rows.json') == plan['rows_sha256']
    summaries, audits, all_cases, integrity, requirements = {}, {}, {}, {}, {}
    for n in (1, 2, 3):
        folder = out/f'round{n}'
        platform = json.loads((out/f'round{n}_platform.json').read_text())
        submission = json.loads((out/f'round{n}_submission.json').read_text())
        assert platform['JobId'] == submission['task_id']
        assert platform['Status'] == 'Success' and platform['End'], ('not complete', n, platform)
        assert json.loads((folder/'status.json').read_text())['status'] == 'success'
        assert json.loads((folder/'audit.json').read_text())['status'] == 'passed'
        audit = json.loads((folder/'independent_audit.json').read_text())
        dynamics = json.loads((folder/'trace_dynamics.json').read_text())
        assert dynamics['status'] == 'passed'
        assert dynamics['independent_audit_sha256'] == sha(folder/'independent_audit.json')
        identity = json.loads((folder/'identity.json').read_text())
        summary = json.loads((folder/'summary.json').read_text())
        assert audit['status'] == 'passed' and summary['status'] == 'passed'
        for key in ('recovered_references_from_source', 'disjoint_sequence_pair_groups', 'transitions_audited'):
            assert audit[key]
        assert audit['predictions_sha256'] == sha(folder/'predictions.jsonl')
        assert audit['traces_sha256'] == sha(folder/'traces.jsonl')
        configs = plan['policies'] if n == 1 else selection['policies']
        assert identity['configs'] == configs
        if n > 1: assert identity['selection_sha256'] == sha(out/'selection.json')
        records = read_predictions(folder)
        assert set(records) == {c['name'] for c in configs}
        for values in records.values():
            assert set(values) == expected['explore' if n == 1 else 'confirm']
        assert audit['predictions'] == len(configs)*len(next(iter(records.values())))
        assert identity['model']['step'] == (112000 if n == 3 else 50000)
        actual_weight_sha = sha(Path(identity['model']['checkpoint'])/'model.safetensors')
        assert actual_weight_sha == identity['model']['model_sha256']
        integrity[str(identity['model']['step'])] = actual_weight_sha
        requirements[f'round{n}'] = dict(platform=platform, independent_audit_sha256=sha(folder/'independent_audit.json'),
            summary_sha256=sha(folder/'summary.json'), predictions=audit['predictions'],
            trace_dynamics_sha256=sha(folder/'trace_dynamics.json'),
            data_coverage_verified=True, frozen_configs_verified=True, unchanged_weight_verified=True)
        summaries[n], audits[n], all_cases[n] = summary, audit, records
    assert integrity['50000'] != integrity['112000']
    assert selection['round1_summary_sha256'] == sha(out/'round1/summary.json')
    history_path = out/'round2/historical_sab23_cpu_equivalence.json'
    history = json.loads(history_path.read_text())
    assert set(history) == {'original1', 'original4', 'original8', 'logprob4'}
    assert all(x['n'] == x['exact_prediction_matches'] == 60 and not x['mismatches'] for x in history.values())
    assert (out/'unit_tests.log').read_text().strip().endswith('OK')
    confirmations, stochastic_comparisons, repair_comparisons = {}, {}, {}
    for n in (2, 3):
        checks = {}
        for p in selection['policies']:
            if p['family'] not in ('schedule', 'adaptive', 'repair'): continue
            name = p['name']
            checks[name] = {key: audits[n]['comparisons'][name+':'+key]
                           for key in audits[n]['scores'][name]}
        confirmations[str(n)] = checks
        records = all_cases[n]
        repair_name = next(p['name'] for p in selection['policies'] if p['family'] == 'repair')
        repair_comparisons[str(n)] = {}
        for key in audits[n]['scores']['original1']:
            take = lambda name: {i: r for i, r in records[name].items() if r['dataset']+':'+r['mode'] == key}
            repair_comparisons[str(n)][key] = paired_interval(take(repair_name), take('matched_random_repair'))
        etas = sorted({p['eta'] for p in selection['policies'] if p['kind'] == 'remdm' and p['eta'] > 0})
        assert len(etas) == 1
        eta = etas[0]
        def averaged(setting):
            names = [p['name'] for p in selection['policies'] if p['kind'] == 'remdm' and p['eta'] == setting]
            assert len(names) == 3
            return {key: {**row, 'correct': sum(records[name][key]['correct'] for name in names)/3}
                    for key, row in records[names[0]].items()}
        remdm, mdlm = averaged(eta), averaged(0.)
        stochastic_comparisons[str(n)] = {}
        for key in audits[n]['scores']['original1']:
            take = lambda rows: {i: r for i, r in rows.items() if r['dataset']+':'+r['mode'] == key}
            result = paired_interval(take(remdm), take(mdlm))
            result['eta'] = eta
            result['seed_deltas_pp'] = {str(seed): audits[n]['scores'][f'remdm8_eta{eta}_s{seed}'][key]
                 -audits[n]['scores'][f'remdm8_eta0.0_s{seed}'][key] for seed in (42, 43, 44)}
            stochastic_comparisons[str(n)][key] = result
    report = dict(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
        requirements=requirements, checkpoint_sha256=integrity,
        historical_sab23_cpu_equivalence_sha256=sha(history_path),
        gate_cpu_sha256=sha(out/'gate_cpu.json'), unit_tests_log_sha256=sha(out/'unit_tests.log'),
        report_source_sha256=sha(Path(__file__)),
        scope='Three rounds of inference-only diagnostic and frozen-policy confirmation completed; '
              'not a claim that AAR improvement was achieved.',
        confirmations=confirmations, stochastic_remdm_vs_mdlm=stochastic_comparisons,
        greedy_repair_vs_random=repair_comparisons, oracle=oracle,
        production_default_changed=False, training_performed=False,
        limitations=['Benchmark-internal split, not pretraining-decontaminated independent validation',
                     'Unadjusted exploratory confidence intervals',
                     'Fixed batch32 common random-number schedules; three seeds, no batch-invariance claim',
                     'Timing includes diagnostic tracing; not a deployment latency benchmark'])
    write(out/'completion_audit.json', report)
    # Compact, machine-readable tables for review and publishing.
    tables = {}
    for n in (1, 2, 3):
        table = {}
        for name, scores in audits[n]['scores'].items():
            cost = summaries[n]['costs'][name]
            table[name] = dict(scores=scores, mean_actual_forwards=cost['sample_forwards']/summaries[n]['n'],
                               seconds=cost['seconds'])
        tables[str(n)] = table
    write(out/'review_tables.json', tables)
    pending_path = out/'completion_pending.json'
    if pending_path.exists():
        prior = json.loads(pending_path.read_text())
        if prior.get('status') == 'pending':
            write(out/'completion_gate_rejection.json', prior)
        write(pending_path, dict(status='resolved', completion_audit_sha256=sha(out/'completion_audit.json')))
    print(json.dumps(dict(status='completed', rounds=3,
        predictions=sum(r['predictions'] for r in requirements.values()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    oracle = oracle_report(args.out)
    if args.finalize:
        finalize(args.out, oracle)
    else:
        print(json.dumps(oracle['comparisons'], indent=2))


if __name__ == '__main__': main()
