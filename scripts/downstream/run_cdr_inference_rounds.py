"""Activate pllm/protenix_abtcr; python -B FILE prepare|gate|run|analyze --out ABS.

Prepare freezes code/data/policies before any scores are examined. GPU runs go
through eval_jobs YAML. Round 1 explores a hash-selected Kong subset; round 2
uses remaining sequence-pair groups plus SAb23; round 3 checks checkpoint112000.
All are exploratory benchmark diagnostics, not a new independent validation set.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
EVAL = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/airgen_evaluate.py'
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def group(row):
    return hashlib.sha256((row['gold_h']+'|'+row['gold_l']).encode()).hexdigest()


def row_id(row):
    return ':'.join(str(row[k]) for k in ('dataset', 'mode', 'fold', 'row', 'key'))


def policies():
    result = []
    def add(name, family, **kw):
        result.append(dict(name=name, family=family, **kw))
    for n in (1, 2, 4, 8):
        add(f'original{n}', 'baseline', kind='scheduled', steps=n)
    for n in (4, 8):
        add(f'logprob{n}', 'schedule', kind='scheduled', steps=n, confidence='logprob')
        for sched in ('root', 'cosine'):
            add(f'{sched}{n}', 'schedule', kind='scheduled', steps=n, schedule=sched)
        add(f'oracle{n}', 'diagnostic', kind='oracle', steps=n)
        for frac in (.25, .5):
            add(f'repair{n}_{frac}', 'repair', kind='repair', steps=n, fraction=frac)
        add(f'random_repair{n}', 'random_control', kind='random_repair', steps=n, fraction=.25, seed=42)
    for val in (.5, .8):
        add(f'threshold_{val}', 'adaptive', kind='threshold', steps=8, value=val)
    for val in (.5, 2.):
        add(f'entropy_{val}', 'adaptive', kind='entropy', steps=8, value=val)
    for eta in (0., .1, .3):
        for seed in (42, 43, 44):
            add(f'remdm8_eta{eta}_s{seed}', 'stochastic', kind='remdm', steps=8, eta=eta, seed=seed)
    return result


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    bundle = out/'source'; bundle.mkdir()
    sources = [Path(__file__), Path(__file__).with_name('cdr_inference_sampling.py'),
               ROOT/'downstream/fixed_chain_eval.py', EVAL]
    frozen = {}
    for p in sources:
        dest = bundle/p.name
        shutil.copy2(p, dest)
        frozen[str(dest)] = sha(dest)
    evaluator = module(bundle/EVAL.name, 'frozen_cdr_eval')
    rows, files = [], {}
    for dataset in ('kong', 'sab23'):
        data, hashes = evaluator.records(dataset)
        files.update(hashes)
        for row in data:
            row['dataset'] = dataset
            row['group'] = group(row)
            row['id'] = row_id(row)
        rows += data
    # Entire sequence-pair duplicates remain together across CDRs and folds.
    sab_groups = {r['group'] for r in rows if r['dataset'] == 'sab23'}
    eligible = {r['group'] for r in rows if r['dataset'] == 'kong'} - sab_groups
    dev = set(sorted(eligible)[:128])
    assert len(dev) == 128
    for row in rows:
        row['split'] = 'explore' if row['dataset'] == 'kong' and row['group'] in dev else 'confirm'
    assert not ({r['group'] for r in rows if r['split'] == 'explore'} &
                {r['group'] for r in rows if r['split'] == 'confirm'})
    write(out/'rows.json', rows)
    original_sources = [evaluator.AIR/'src/byprot/models/lm/dplm_multichain.py',
                        evaluator.AIR/'src/byprot/models/lm/model_utils.py']
    original_sources += [ROOT/'output/downstream_generation'/d/'checkpoint/formal_checkpoint.json' for d in
                        ('fixed_chain_75m_step50000_20261013', 'fixed_chain_75m_best112000_20261014')]
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Training-free internal benchmark diagnostic; no independent validation or headline selection',
        checkpoints={k: str(ROOT/'output/downstream_generation'/v/'checkpoint') for k, v in
                     [('50000', 'fixed_chain_75m_step50000_20261013'),
                      ('112000', 'fixed_chain_75m_best112000_20261014')]},
        policies=policies(), batch_size=32, precision='fp32',
        rows_sha256=sha(out/'rows.json'), source_sha256=frozen,
        external_source_sha256={str(p): sha(p) for p in original_sources}, data_sha256=files,
        split_counts={s: sum(r['split'] == s for r in rows) for s in ('explore', 'confirm')},
        selection='Round1 equal-H1/H2/H3 fold-macro AAR; best config in schedule/adaptive/repair; '
                  'stochastic eta selected by mean across seeds; deterministic ties prefer fewer forwards, then name. '
                  'Always keep original1/4/8, matched MDLM eta0 seeds and random repair control. '
                  'Round2 and Round3 reuse selected policies without tuning.')
    write(out/'plan.json', plan)
    print(json.dumps({k: plan[k] for k in ('split_counts', 'selection')}, ensure_ascii=False), flush=True)


def verify(out):
    plan = json.loads((out/'plan.json').read_text())
    assert sha(out/'rows.json') == plan['rows_sha256']
    for mapping in ('source_sha256', 'external_source_sha256', 'data_sha256'):
        for path, digest in plan[mapping].items():
            assert sha(path) == digest, path
    return plan, json.loads((out/'rows.json').read_text())


def gate(out, device):
    import torch
    torch.set_num_threads(4)
    plan, rows = verify(out)
    sampler = module(out/'source/cdr_inference_sampling.py', 'frozen_sampler')
    adapter = module(out/'source/fixed_chain_eval.py', 'frozen_adapter')
    evaluator = module(out/'source/airgen_evaluate.py', 'frozen_evaluator')
    model, identity = adapter.load(plan['checkpoints']['50000'], device)
    chosen = [next(r for r in rows if r['dataset'] == ds and r['mode'] == mode)
              for ds in ('kong', 'sab23') for mode in evaluator.MODES[ds]]
    chosen.append(next(r for r in rows if not r['light']))
    features, batch, gold, target, maps = adapter.airgen_inputs(model, chosen)
    altered = [{**r, 'gold_h': 'G'*len(r['gold_h']), 'gold_l': 'G'*len(r['gold_l'])} for r in chosen]
    f2, b2, _, t2, _ = adapter.airgen_inputs(model, altered)
    assert all(torch.equal(v, f2[k]) for k, v in features.items())
    assert torch.equal(batch['input_ids'], b2['input_ids']) and torch.equal(target, t2)
    features = {k: v.to(device) for k, v in features.items()}
    batch = {k: v.to(device) for k, v in batch.items()}
    target, gold = target.to(device), gold.to(device)
    checks = []
    with torch.inference_mode():
        for steps in (1, 2, 4, 8):
            bridge = evaluator.decoder(model, features, maps)
            got, trace = sampler.sample(bridge, batch, target, dict(kind='scheduled', steps=steps))
            expected, _ = bridge.generate(batch, max_iter=steps, temperature=1.,
                partial_masks=~target & batch['input_ids'].ne(bridge.bos_id),
                sampling_strategy='argmax', cfg_scale=0.)
            assert torch.equal(got, expected), ('original equivalence', steps)
            checks.append(f'original_equivalence_{steps}')
            if steps == 4:
                oracle, _ = sampler.sample(bridge, batch, target, dict(kind='oracle', steps=4), gold=gold, replay=trace)
                assert torch.equal(oracle[trace[0]['commit'].to(device)], got[trace[0]['commit'].to(device)])
        for p in plan['policies']:
            if p['kind'] in ('oracle', 'scheduled'):
                continue
            got, trace = sampler.sample(bridge, batch, target, {**p, 'steps': 2}, seed=p.get('seed', 42))
            assert torch.equal(got[~target], batch['input_ids'][~target])
            assert not got[target].eq(bridge.mask_id).any()
            checks.append(p['name'])
    write(out/f'gate_{device}.json', dict(status='passed', identity=identity, n=len(chosen),
        checks=checks, hidden_reference_independent=True, visible_context_preserved=True,
        all_nine_modes_and_empty_light=True, plan_sha256=sha(out/'plan.json')))
    print('GATE PASSED', device, len(checks), flush=True)


def metrics(predictions):
    groups = defaultdict(list)
    for r in predictions:
        groups[(r['dataset'], r['mode'], r['fold'])].append(r['correct']/r['length'])
    modes = defaultdict(list)
    for (dataset, mode, fold), vals in groups.items():
        modes[(dataset, mode)].append(sum(vals)/len(vals)*100)
    return {ds+':'+mode: dict(fold_macro_aar=sum(v)/len(v), n=sum(
        r['dataset'] == ds and r['mode'] == mode for r in predictions)) for (ds, mode), v in modes.items()}


def run(out, round_number, device):
    import torch
    torch.set_num_threads(4)
    plan, all_rows = verify(out)
    gate_result = json.loads((out/'gate_cpu.json').read_text())
    assert gate_result['status'] == 'passed' and gate_result['plan_sha256'] == sha(out/'plan.json')
    if round_number == 1:
        configs = plan['policies']
    else:
        selection = json.loads((out/'selection.json').read_text())
        assert selection['round1_summary_sha256'] == sha(out/'round1/summary.json')
        configs = selection['policies']
    rows = [r for r in all_rows if r['split'] == ('explore' if round_number == 1 else 'confirm')]
    dest = out/f'round{round_number}'
    dest.mkdir(exist_ok=False)
    write(dest/'status.json', dict(status='running', started_utc=datetime.now(timezone.utc).isoformat()))
    sampler = module(out/'source/cdr_inference_sampling.py', 'frozen_sampler')
    adapter = module(out/'source/fixed_chain_eval.py', 'frozen_adapter')
    evaluator = module(out/'source/airgen_evaluate.py', 'frozen_evaluator')
    checkpoint = plan['checkpoints']['112000' if round_number == 3 else '50000']
    try:
        model, identity = adapter.load(checkpoint, device)
        write(dest/'identity.json', dict(model=identity, configs=configs, n=len(rows),
            device=device, precision=plan['precision'], batch_size=plan['batch_size'],
            plan_sha256=sha(out/'plan.json'), selection_sha256=sha(out/'selection.json') if round_number > 1 else None))
        accum = {p['name']: [] for p in configs}
        costs = {p['name']: dict(seconds=0., batch_forwards=0, sample_forwards=0) for p in configs}
        with (dest/'predictions.jsonl').open('x') as predstream, (dest/'traces.jsonl').open('x') as tracestream:
            for start in range(0, len(rows), plan['batch_size']):
                chunk = rows[start:start+plan['batch_size']]
                features, batch, gold, target, maps = adapter.airgen_inputs(model, chunk)
                features = {k: v.to(device) for k, v in features.items()}
                batch = {k: v.to(device) for k, v in batch.items()}
                target, gold = target.to(device), gold.to(device)
                cache = {}
                with torch.inference_mode():
                    if start == 0:
                        # Warm up before timed decoding; no result is scored.
                        evaluator.decoder(model, features, maps).net(**batch)
                    for p in configs:
                        name = p['name']; bridge = evaluator.decoder(model, features, maps)
                        kwargs = dict(gold=gold, replay=cache[f"original{p['steps']}"]) if p['kind'] == 'oracle' else {}
                        if device == 'cuda': torch.cuda.synchronize()
                        begin = time.monotonic()
                        pred, traces = sampler.sample(bridge, batch, target, p, p.get('seed', 42), **kwargs)
                        if device == 'cuda': torch.cuda.synchronize()
                        costs[name]['seconds'] += time.monotonic()-begin
                        costs[name]['batch_forwards'] += len(traces)
                        costs[name]['sample_forwards'] += len(traces)*len(chunk)
                        if name in ('original4', 'original8'): cache[name] = traces
                        if start == 0 and p['family'] == 'baseline':
                            expected, _ = bridge.generate(batch, max_iter=p['steps'], temperature=1.,
                                partial_masks=~target & batch['input_ids'].ne(bridge.bos_id),
                                sampling_strategy='argmax', cfg_scale=0.)
                            assert torch.equal(pred, expected), ('GPU original equivalence', name)
                        for i, row in enumerate(chunk):
                            positions = target[i].nonzero().flatten().cpu()
                            guess, ref = pred[i, target[i]].tolist(), gold[i, target[i]].tolist()
                            record = {k: row[k] for k in ('id', 'group', 'dataset', 'mode', 'fold', 'row', 'key')}
                            record.update(policy=name, prediction_ids=guess, reference_ids=ref,
                                correct=sum(a == b for a, b in zip(guess, ref)), length=len(ref),
                                sample_forwards=len(traces), oracle=p['kind'] == 'oracle')
                            predstream.write(json.dumps(record)+'\n'); accum[name].append(record)
                            steps = []
                            for tr in traces:
                                steps.append({k: tr[k][i, positions].tolist() for k in
                                    ('commit', 'remasked', 'before', 'after', 'proposal', 'pending', 'confidence')})
                            tracestream.write(json.dumps(dict(id=row['id'], policy=name, steps=steps))+'\n')
                predstream.flush(); tracestream.flush()
                write(dest/'progress.json', dict(completed=start+len(chunk), total=len(rows), costs=costs))
                print('round', round_number, start+len(chunk), '/', len(rows), flush=True)
        summary = dict(status='passed', n=len(rows), checkpoint=identity, costs=costs,
                       cases={name: metrics(values) for name, values in accum.items()})
        write(dest/'summary.json', summary)
        write(dest/'status.json', dict(status='success', finished_utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as exc:
        write(dest/'status.json', dict(status='failed', error=repr(exc)))
        raise


def analyze(out, round_number):
    """Independent token-level metric recomputation and grouped paired bootstrap."""
    import numpy as np
    plan, rows = verify(out)
    folder = out/f'round{round_number}'
    summary = json.loads((folder/'summary.json').read_text())
    assert summary['status'] == 'passed'
    cases = defaultdict(dict)
    for line in (folder/'predictions.jsonl').read_text().splitlines():
        r = json.loads(line)
        assert r['id'] not in cases[r['policy']]
        assert r['length'] == len(r['prediction_ids']) == len(r['reference_ids'])
        assert r['correct'] == sum(a == b for a, b in zip(r['prediction_ids'], r['reference_ids']))
        cases[r['policy']][r['id']] = r
    expected = {r['id'] for r in rows if r['split'] == ('explore' if round_number == 1 else 'confirm')}
    for name, predictions in cases.items():
        assert set(predictions) == expected
        assert metrics(list(predictions.values())) == summary['cases'][name]
    comparisons = {}
    for name, predictions in cases.items():
        for dataset in ('kong', 'sab23'):
            for mode in ('cdrh1', 'cdrh2', 'cdrh3', 'cdrl1', 'cdrl2', 'cdrl3'):
                matched = [r for r in predictions.values() if r['dataset'] == dataset and r['mode'] == mode]
                if not matched: continue
                by_fold = defaultdict(lambda: defaultdict(list))
                for r in matched:
                    base = cases['original1'][r['id']]
                    by_fold[r['fold']][r['group']].append((r['correct']-base['correct'])/r['length']*100)
                rng = np.random.default_rng(20261015)
                boots, means = [], []
                for fold_groups in by_fold.values():
                    arrays = list(fold_groups.values())
                    sums, counts = np.array([sum(x) for x in arrays]), np.array([len(x) for x in arrays])
                    indices = rng.integers(0, len(arrays), size=(2000, len(arrays)))
                    boots.append(sums[indices].sum(1)/counts[indices].sum(1))
                    means.append(sums.sum()/counts.sum())
                ci = np.percentile(np.mean(boots, axis=0), [2.5, 97.5]).tolist()
                comparisons[f'{name}:{dataset}:{mode}'] = dict(delta_vs_original1_pp=float(np.mean(means)),
                    paired_group_bootstrap95ci_pp=ci, n=len(matched))
    # Trace audit catches implementations that score the last proposal rather than committed output.
    trace_count = 0
    for line in (folder/'traces.jsonl').open():
        tr = json.loads(line); r = cases[tr['policy']][tr['id']]
        assert tr['steps'][-1]['after'] == r['prediction_ids']
        assert len(tr['steps']) == r['sample_forwards']
        for prev, cur in zip(tr['steps'], tr['steps'][1:]): assert prev['after'] == cur['before']
        trace_count += 1
    assert trace_count == sum(len(x) for x in cases.values())
    write(folder/'audit.json', dict(status='passed', predictions=trace_count,
        metrics_recomputed=True, traces_reconciled=True, comparisons=comparisons,
        predictions_sha256=sha(folder/'predictions.jsonl'), traces_sha256=sha(folder/'traces.jsonl'),
        limitations='Descriptive unadjusted intervals; benchmark-internal partition, not unbiased validation.'))
    if round_number == 1:
        def score(p):
            return sum(x['fold_macro_aar'] for x in summary['cases'][p['name']].values())/3
        selected = [p for p in plan['policies'] if p['name'] in ('original1', 'original4', 'original8')]
        for family in ('schedule', 'adaptive', 'repair'):
            candidates = [p for p in plan['policies'] if p['family'] == family]
            best = sorted(candidates, key=lambda p: (-score(p), summary['costs'][p['name']]['sample_forwards'], p['name']))[0]
            selected.append(best)
        repair = next(p for p in selected if p['family'] == 'repair')
        selected.append(dict(name='matched_random_repair', family='random_control',
                             kind='random_repair', steps=repair['steps'], fraction=repair['fraction'], seed=42))
        etas = (.1, .3)
        best_eta = max(etas, key=lambda eta: sum(score(p) for p in plan['policies']
                         if p['kind'] == 'remdm' and p['eta'] == eta)/3)
        selected += [p for p in plan['policies'] if p['kind'] == 'remdm' and p['eta'] in (0., best_eta)]
        selection = dict(policies=selected, round1_summary_sha256=sha(folder/'summary.json'),
            round1_audit_sha256=sha(folder/'audit.json'), selection_rule=plan['selection'],
            frozen_utc=datetime.now(timezone.utc).isoformat())
        if (out/'selection.json').exists():
            existing = json.loads((out/'selection.json').read_text())
            assert existing['policies'] == selected and existing['round1_summary_sha256'] == selection['round1_summary_sha256']
        else: write(out/'selection.json', selection)
        print('SELECTED', [p['name'] for p in selected], flush=True)
    print('AUDIT PASSED', round_number, trace_count, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'gate', 'run', 'analyze'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--round', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.out)
    elif args.action == 'gate': gate(args.out, args.device)
    elif args.action == 'run': run(args.out, args.round, args.device)
    else: analyze(args.out, args.round)


if __name__ == '__main__':
    main()
