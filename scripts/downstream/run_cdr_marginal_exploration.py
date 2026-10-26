"""Activate pllm/protenix_abtcr; python -B FILE prepare|gate|run --out ABS.

Fresh, grouped modern-SAbDab development pool and three new probability
aggregation hypotheses. Submit GPU runs via eval_jobs YAML. No checkpoint is
modified. Confirmation rows are never loaded into an exploration evaluation.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
OLD = ROOT/'output/downstream_generation/cdr_inference_rounds_20261015'
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def families():
    values = []
    for reveal in (.25, .5):
        for temperature in (0., .5, 1.):
            values.append(dict(name=f'mc_r{reveal}_t{temperature}', kind='mc_context',
                reveal=reveal, temperature=temperature, draws=[8, 32], seed=42))
    values.append(dict(name='loo_greedy_context', kind='leave_one_out'))
    for rule in ('confidence', 'entropy'):
        for anchors in (1, 3):
            values.append(dict(name=f'branch_{rule}_a{anchors}', kind='branch_context',
                              anchor_rule=rule, anchors=anchors, topk=8))
    return values


def prepare(out):
    out.mkdir(parents=True, exist_ok=False); source = out/'source'; source.mkdir()
    old_rows = json.loads((OLD/'rows.json').read_text())
    old_heavy = {r['gold_h'] for r in old_rows}
    old_pdb = {str(r['key'])[:4].lower() for r in old_rows}
    source_hashes = {}
    for p in [Path(__file__), Path(__file__).with_name('cdr_marginal_inference.py'),
              OLD/'source/fixed_chain_eval.py', OLD/'source/airgen_evaluate.py']:
        dest = source/p.name; shutil.copy2(p, dest); source_hashes[str(dest)] = sha(dest)
    data_hashes = {str(OLD/'rows.json'): sha(OLD/'rows.json')}
    by_mode, exclusions = {}, {}
    for mode in ('cdrh1', 'cdrh2', 'cdrh3'):
        records = {}; excluded = Counter()
        for path in sorted((ROOT/'data/downstream/cdr_infilling/sabdab'/mode).glob('fold_*/test.json')):
            data_hashes[str(path)] = sha(path)
            for index, line in enumerate(path.read_text().splitlines()):
                d = json.loads(line); h = d['heavy_chain_seq'].replace('J', 'L'); l = d['light_chain_seq'].replace('J', 'L')
                pdb = d['pdb_id'][:4].lower(); start, end = d[mode+'_pos']
                assert h[start:end+1] == d[mode+'_seq'].replace('J', 'L'), 'Coordinate convention mismatch'
                if pdb in old_pdb: excluded['old_pdb'] += 1; continue
                if h in old_heavy: excluded['old_full_heavy'] += 1; continue
                if len(h) > 148 or len(l) > 126: excluded['grid_overflow'] += 1; continue
                identity = hashlib.sha256((pdb+'|'+h+'|'+l).encode()).hexdigest()
                assert identity not in records
                records[identity] = dict(dataset='modern', mode=mode, fold=0, row=index,
                    key=d['pdb_id'], heavy=h, light=l, gold_h=h, gold_l=l,
                    mask_h=list(range(start, end+1)), mask_l=[], source=str(path),
                    source_fold=int(path.parent.name.split('_')[-1]), id=f'modern:{mode}:{identity}',
                    identity=identity, pdb=pdb)
        by_mode[mode] = records; exclusions[mode] = dict(excluded)
    assert all(set(v) == set(by_mode['cdrh3']) for v in by_mode.values())
    # Connected components group same PDB, full heavy or exact CDR-H3 together.
    ids = sorted(by_mode['cdrh3']); parent = {i: i for i in ids}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i in ids:
        r = by_mode['cdrh3'][i]; h3 = ''.join(r['gold_h'][j] for j in r['mask_h'])
        for tag in ('pdb:'+r['pdb'], 'heavy:'+r['gold_h'], 'h3:'+h3):
            if tag in seen: parent[find(i)] = find(seen[tag])
            else: seen[tag] = i
    members = defaultdict(list)
    for i in ids: members[find(i)].append(i)
    group_ids = {i: hashlib.sha256('|'.join(sorted(group)).encode()).hexdigest()
                 for group in members.values() for i in group}
    unique_groups = sorted(set(group_ids.values())); assert len(unique_groups) >= 512
    explore = set(unique_groups[:256])
    rows = []
    for mode, records in by_mode.items():
        for i, r in sorted(records.items()):
            r['group'] = group_ids[i]
            r['split'] = 'explore' if r['group'] in explore else 'confirm'
            rows.append(r)
    for kind in ('pdb', 'gold_h', 'group'):
        assert not ({r[kind] for r in rows if r['split'] == 'explore'} &
                    {r[kind] for r in rows if r['split'] == 'confirm'})
    write(out/'rows.json', rows)
    old_plan = json.loads((OLD/'plan.json').read_text())
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        families=families(), prior_weights=[0., .5, .8], checkpoints=old_plan['checkpoints'],
        primary_checkpoint='112000', batch_size=24, precision='fp32',
        source_sha256=source_hashes, data_sha256=data_hashes,
        external_source_sha256=old_plan['external_source_sha256'], rows_sha256=sha(out/'rows.json'),
        exclusions=exclusions, components=len(unique_groups),
        split_counts={s: Counter(r['mode'] for r in rows if r['split'] == s) for s in ('explore', 'confirm')},
        scope='Fresh modern-SAbDab subset excludes previous benchmark PDBs and exact full-heavy sequences; '
              'not a claim of pretraining decontamination or unseen by every historical model evaluation.',
        selection='Explore H3 only. Candidate requires positive AAR difference versus single-pass on explore; '
                  'freeze best mean sequence AAR (ties fewer logical forwards then name) before confirm. '
                  'Do not inspect confirm scores if no explore candidate improves. Confirm all three H CDRs, '
                  'then frozen cross-checkpoint and established benchmark checks. No success declaration based on exploration alone.',
        success='A positive primary H3 effect must survive reserved-group confirmation with paired group uncertainty '
                'and frozen cross-checkpoint/benchmark checking. A negative sweep is not completion of this exploration goal.')
    write(out/'plan.json', plan)
    print(json.dumps({k: plan[k] for k in ('exclusions', 'components', 'split_counts')}, ensure_ascii=False))


def verify(out):
    plan = json.loads((out/'plan.json').read_text())
    for field in ('source_sha256', 'data_sha256', 'external_source_sha256'):
        for path, value in plan[field].items(): assert sha(path) == value, path
    assert sha(out/'rows.json') == plan['rows_sha256']
    return plan, json.loads((out/'rows.json').read_text())


def setup(out, checkpoint, device):
    adapter = module(out/'source/fixed_chain_eval.py', 'marginal_adapter')
    evaluator = module(out/'source/airgen_evaluate.py', 'marginal_evaluator')
    sampler = module(out/'source/cdr_marginal_inference.py', 'marginal_sampler')
    model, identity = adapter.load(checkpoint, device)
    return adapter, evaluator, sampler, model, identity


def prediction_fn(bridge, batch, target):
    import math, torch
    counter = dict(calls=0)
    def predict(tokens):
        assert torch.equal(tokens[~target], batch['input_ids'][~target])
        logits = bridge.net(input_ids=tokens, chain_ids=batch['chain_ids'])['logits'].float()
        logits[..., bridge.special_token_list] = -torch.inf
        logits[..., bridge.eos_id] += 1e-3*math.log(1e-3)
        counter['calls'] += 1
        return logits.softmax(-1)
    return predict, counter


def gate(out):
    import torch
    torch.set_num_threads(4)
    plan, rows = verify(out)
    adapter, evaluator, sampler, model, identity = setup(out, plan['checkpoints']['112000'], 'cpu')
    chosen = [next(r for r in rows if r['mode'] == m and r['split'] == 'explore') for m in ('cdrh1', 'cdrh2', 'cdrh3')]
    # Include a missing-light variant without changing the target reference.
    chosen.append({**chosen[-1], 'light': '', 'gold_l': ''})
    f, b, gold, target, maps = adapter.airgen_inputs(model, chosen)
    altered = [{**r, 'gold_h': 'G'*len(r['gold_h']), 'gold_l': 'G'*len(r['gold_l'])} for r in chosen]
    f2, b2, _, t2, _ = adapter.airgen_inputs(model, altered)
    assert all(torch.equal(v, f2[k]) for k, v in f.items())
    assert torch.equal(b['input_ids'], b2['input_ids']) and torch.equal(target, t2)
    bridge = evaluator.decoder(model, f, maps); predict, counter = prediction_fn(bridge, b, target)
    results = []
    with torch.inference_mode():
        p0 = sampler.base_distribution(predict, b, target, bridge.mask_id)
        base, _ = sampler.mix_prediction(b['input_ids'], target, p0, p0, 1.)
        expected, _ = bridge.generate(b, max_iter=1, temperature=1.,
            partial_masks=~target & b['input_ids'].ne(bridge.bos_id), sampling_strategy='argmax', cfg_scale=0.)
        assert torch.equal(base, expected)
        for config in [dict(name='mc_gate', kind='mc_context', reveal=.5, temperature=1., draws=[4]),
                       dict(name='loo_gate', kind='leave_one_out'),
                       dict(name='branch_gate', kind='branch_context', anchor_rule='entropy', anchors=1, topk=2)]:
            start = counter['calls']
            for name, post, count, nfe, meta in sampler.aggregate(predict, b, target, bridge.mask_id, p0, config):
                assert counter['calls']-start == nfe-1
                for weight in (0., .5, .8, 1.):
                    pred, mixture = sampler.mix_prediction(b['input_ids'], target, p0, post, weight)
                    assert not pred[target].eq(bridge.mask_id).any()
                    if weight == 1.: assert torch.equal(pred, base)
                results.append(name)
    write(out/'gate_cpu.json', dict(status='passed', identity=identity,
        original_single_pass_equivalence=True, no_reference_in_features=True,
        self_visible_outputs_excluded=True, context_preserved=True, families=results,
        actual_forwards_counted=True, plan_sha256=sha(out/'plan.json')))
    print('CPU GATE PASSED', results, flush=True)


def run(out, checkpoint, device):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    plan, all_rows = verify(out)
    assert json.loads((out/'gate_cpu.json').read_text())['plan_sha256'] == sha(out/'plan.json')
    rows = [r for r in all_rows if r['split'] == 'explore' and r['mode'] == 'cdrh3']
    dest = out/f'explore_{checkpoint}'; dest.mkdir(exist_ok=False)
    write(dest/'status.json', dict(status='running', started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        adapter, evaluator, sampler, model, identity = setup(out, plan['checkpoints'][checkpoint], device)
        write(dest/'identity.json', dict(model=identity, plan_sha256=sha(out/'plan.json'), n=len(rows)))
        scores = defaultdict(list); logical_cost = defaultdict(int); executed = 0; gate_forwards = 0
        with (dest/'predictions.jsonl').open('x') as predictions, (dest/'query_audit.jsonl').open('x') as queries:
            for start in range(0, len(rows), plan['batch_size']):
                chunk = rows[start:start+plan['batch_size']]
                f, b, gold, target, maps = adapter.airgen_inputs(model, chunk)
                f = {k: v.to(device) for k, v in f.items()}; b = {k: v.to(device) for k, v in b.items()}
                gold, target = gold.to(device), target.to(device)
                bridge = evaluator.decoder(model, f, maps); predict, counter = prediction_fn(bridge, b, target)
                packed = {}; positions = [t.nonzero().flatten() for t in target]
                width = max(map(len, positions))
                def pack(tensor):
                    result = np.zeros((len(chunk), width, tensor.shape[-1]), dtype=np.float32)
                    for i, pos in enumerate(positions): result[i, :len(pos)] = tensor[i, pos].cpu().numpy()
                    return result
                def save_case(name, pred, nfe):
                    for i, row in enumerate(chunk):
                        ref, guess = gold[i, target[i]].tolist(), pred[i, target[i]].tolist()
                        correct = sum(x == y for x, y in zip(guess, ref))
                        r = {k: row[k] for k in ('id', 'group', 'dataset', 'mode', 'fold', 'key')}
                        r.update(policy=name, reference_ids=ref, prediction_ids=guess, length=len(ref), correct=correct, logical_forwards=nfe)
                        scores[name].append(correct/len(ref)*100); logical_cost[name] += nfe
                        predictions.write(json.dumps(r)+'\n')
                def observe(tag, inputs, scored, probs):
                    # Store actual query contexts and scoring masks; no self-visible score enters aggregation.
                    for i, row in enumerate(chunk):
                        pos = positions[i]
                        queries.write(json.dumps(dict(id=row['id'], family=current_family, query=tag,
                            inputs=inputs[i, pos].tolist(), scored=scored[i, pos].tolist()))+'\n')
                with torch.inference_mode():
                    p0 = sampler.base_distribution(predict, b, target, bridge.mask_id)
                    packed['prior'] = pack(p0)
                    pred, _ = sampler.mix_prediction(b['input_ids'], target, p0, p0, 1.)
                    save_case('single_pass', pred, 1)
                    expected, _ = bridge.generate(b, max_iter=1, temperature=1.,
                        partial_masks=~target & b['input_ids'].ne(bridge.bos_id), sampling_strategy='argmax', cfg_scale=0.)
                    assert torch.equal(pred, expected)
                    gate_forwards += len(chunk)
                    for family in plan['families']:
                        current_family = family['name']
                        begin = time.monotonic()
                        for name, posterior, counts, nfe, meta in sampler.aggregate(predict, b, target, bridge.mask_id, p0, family, observe):
                            packed[name] = pack(posterior)
                            for weight in plan['prior_weights']:
                                pred, _ = sampler.mix_prediction(b['input_ids'], target, p0, posterior, weight)
                                save_case(name+f'_prior{weight}', pred, nfe)
                executed += counter['calls']*len(chunk)
                packed['lengths'] = np.array([len(p) for p in positions], dtype=np.int32)
                packed['ids'] = np.array([r['id'] for r in chunk])
                np.savez_compressed(dest/f'posterior_batch_{start:05d}.npz', **packed)
                predictions.flush(); queries.flush()
                write(dest/'progress.json', dict(completed=start+len(chunk), total=len(rows), actual_sample_forwards=executed))
                print(checkpoint, start+len(chunk), '/', len(rows), flush=True)
        summary = dict(status='passed', n=len(rows), model=identity, actual_sample_forwards=executed,
            additional_equivalence_sample_forwards=gate_forwards,
            scores={k: float(np.mean(v)) for k, v in scores.items()},
            logical_mean_forwards={k: logical_cost[k]/len(rows) for k in scores},
            shared_computation='Prior and MC prefixes reused across configurations; logical per-configuration costs are not physical executed calls.')
        write(dest/'summary.json', summary); write(dest/'status.json', dict(status='success'))
        print('EXPLORATION COMPLETE', checkpoint, flush=True)
    except BaseException as exc:
        write(dest/'status.json', dict(status='failed', error=repr(exc))); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'gate', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('50000', '112000'), default='112000')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.out)
    elif args.action == 'gate': gate(args.out)
    else: run(args.out, args.checkpoint, args.device)


if __name__ == '__main__': main()
