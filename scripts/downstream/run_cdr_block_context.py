"""Activate pllm/protenix_abtcr; python -B FILE gate|run --out ABS.

GPU runs use eval_jobs YAML. Frozen grouped soft contexts never consume gold.
Development selects on old development rows; confirmation uses new OAS only.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def context(out):
    plan = json.loads((out/'plan.json').read_text())
    for field in ('source_sha256', 'external_source_sha256'):
        for path, digest in plan[field].items():
            assert sha(path) == digest, path
    assert sha(out/'data_plan.json') == plan['data_plan_sha256']
    assert sha(out/'rows.json') == plan['rows_sha256']
    assert sha(out/'data_independent_audit.json') == plan['data_audit_sha256']
    parent = Path(plan['parent'])
    assert sha(parent/'rows.json') == plan['development_rows_sha256']
    old = module(parent/'source/run_cdr_marginal_exploration.py', 'block_parent')
    _, rows = old.verify(parent)
    dev = [r for r in rows if r['split'] == 'explore' and r['mode'] == 'cdrh3']
    assert len(dev) == 318
    sampler = module(out/'source/cdr_block_soft.py', 'block_sampler')
    return plan, old, sampler, dev


def prepare(adapter, evaluator, old, model, chunk, device):
    f, b, gold, target, maps = adapter.airgen_inputs(model, chunk)
    f = {k: v.to(device) for k, v in f.items()}
    b = {k: v.to(device) for k, v in b.items()}
    gold, target = gold.to(device), target.to(device)
    bridge = evaluator.decoder(model, f, maps)
    original_net = bridge.net; physical = dict(calls=0)
    def counted(*args, **kwargs):
        physical['calls'] += 1
        return original_net(*args, **kwargs)
    bridge.net = counted
    predict, _ = old.prediction_fn(bridge, b, target)
    prior = predict(b['input_ids'])
    return f, b, gold, target, maps, bridge, predict, prior, physical


def gate(args):
    import torch
    torch.set_num_threads(4)
    plan, old, sampler, dev = context(args.out)
    adapter, evaluator, _, model, identity = old.setup(Path(plan['parent']), plan['checkpoints'][args.checkpoint], args.device)
    legacy = module(Path(plan['legacy_soft_source']), 'block_legacy_soft')
    chunk = dev[:2]
    with torch.inference_mode():
        f, b, gold, target, maps, bridge, predict, prior, physical = prepare(adapter, evaluator, old, model, chunk, args.device)
        changed = [{**r, 'gold_h': 'G'*len(r['gold_h']), 'gold_l': 'G'*len(r['gold_l'])} for r in chunk]
        f2, b2, _, t2, _ = adapter.airgen_inputs(model, changed)
        assert all(torch.equal(v.cpu(), f2[k]) for k, v in f.items())
        assert torch.equal(b['input_ids'].cpu(), b2['input_ids']) and torch.equal(target.cpu(), t2)
        original, _ = bridge.generate(b, max_iter=1, temperature=1., partial_masks=~target & b['input_ids'].ne(bridge.bos_id), sampling_strategy='argmax', cfg_scale=0.)
        assert torch.equal(original[target], prior.argmax(-1)[target])
        for groups, strength in ((1, .5), (3, 0.)):
            p, _ = sampler.posterior(model, predict, b, target, f, maps, prior, groups, 'interleaved', .5, strength)
            assert torch.equal(p, prior)
        maxlen = int(target.sum(-1).max())
        p, _ = sampler.posterior(model, predict, b, target, f, maps, prior, maxlen, 'interleaved', .5, .25)
        q = legacy.selected_posteriors(model, None, predict, b, target, f, maps, prior, dict(kind='soft_context', temperature=.5, strength=.25))['candidate'][0]
        assert torch.equal(p[target], q[target])
        # Check that hooks were removed, including after all scored queries.
        assert torch.equal(predict(b['input_ids']), prior)
        assert not any(p.requires_grad for p in model.parameters()) and not model.training
    record = dict(status='passed', checkpoint=args.checkpoint, device=args.device, n=len(chunk), identity=identity,
        no_reference_in_features=True, original_single_pass_argmax_exact=True, zero_strength_exact=True,
        single_group_exact=True, singleton_partition_equals_legacy_loo_exact=True, hook_cleanup_exact=True,
        physical_sample_forwards=physical['calls']*len(chunk), plan_sha256=sha(args.out/'plan.json'))
    write(args.out/f'gate_{args.device}_{args.checkpoint}.json', record)
    print(json.dumps(record), flush=True)


def run(args):
    import torch
    torch.set_num_threads(4)
    plan, old, sampler, dev = context(args.out)
    gate_path = args.out/f'gate_{args.device}_{args.checkpoint}.json'
    assert json.loads(gate_path.read_text())['plan_sha256'] == sha(args.out/'plan.json')
    families = plan['families']; weights = plan['prior_weights']; rows = dev
    if args.phase == 'confirm':
        selection = json.loads((args.out/'selection.json').read_text())
        assert selection['eligible'] and selection['plan_sha256'] == sha(args.out/'plan.json')
        assert selection['development_audit_sha256'] == sha(args.out/'explore_112000/independent_audit.json')
        config = selection['config']
        families = [f for f in families if f['name'] == config['name']]
        assert len(families) == 1
        weights = [config['prior_weight']]
        rows = [r for r in json.loads((args.out/'rows.json').read_text()) if r['split'] == 'confirm']
        assert len(rows) == 6282
        audit = json.loads((args.out/'data_independent_audit.json').read_text())
        assert audit['status'] == 'passed_scoped_checks'
    else:
        assert args.checkpoint == '112000'
    dest = args.out/(f'explore_{args.checkpoint}' if args.phase == 'explore' else f'confirmation_{args.checkpoint}')
    dest.mkdir(exist_ok=False)
    write(dest/'status.json', dict(status='running', started_utc=datetime.now(timezone.utc).isoformat()))
    adapter, evaluator, _, model, identity = old.setup(Path(plan['parent']), plan['checkpoints'][args.checkpoint], args.device)
    write(dest/'identity.json', dict(model=identity, plan_sha256=sha(args.out/'plan.json'), phase=args.phase))
    scores = defaultdict(lambda: defaultdict(list)); forwards = 0
    with (dest/'predictions.jsonl').open('x') as stream, (dest/'query_audit.jsonl').open('x') as queries, torch.inference_mode():
        for start in range(0, len(rows), plan['batch_size']):
            chunk = rows[start:start+plan['batch_size']]
            f, b, gold, target, maps, bridge, predict, prior, physical = prepare(adapter, evaluator, old, model, chunk, args.device)
            positions = [x.nonzero().flatten() for x in target]
            lengths = np.array([len(p) for p in positions], dtype=np.int64)
            def pack(probs):
                packed = np.zeros((len(chunk), int(lengths.max()), probs.shape[-1]), dtype=np.float32)
                for i, pos in enumerate(positions): packed[i, :len(pos)] = probs[i, pos].cpu().numpy()
                return packed
            position_array = np.full((len(chunk), int(lengths.max())), -1, dtype=np.int64)
            for i, pos in enumerate(positions): position_array[i, :len(pos)] = pos.cpu().numpy()
            arrays = dict(ids=np.array([r['id'] for r in chunk]), lengths=lengths, positions=position_array, prior=pack(prior))
            predictions = {'single_pass': (prior.argmax(-1), 1)}
            for family in families:
                def observer(index, scored, feedback):
                    for i, row in enumerate(chunk):
                        queries.write(json.dumps(dict(id=row['id'], family=family['name'], query=index,
                            scored=scored[i, target[i]].cpu().tolist(), feedback=feedback[i, target[i]].cpu().tolist()))+'\n')
                post, nfe = sampler.posterior(model, predict, b, target, f, maps, prior,
                    **{k: v for k, v in family.items() if k != 'name'}, observer=observer)
                arrays[family['name']] = pack(post)
                for weight in weights:
                    predictions[f"{family['name']}_prior{weight}"] = ((weight*prior+(1-weight)*post).argmax(-1), nfe)
            if args.phase == 'confirm':
                for steps in (4, 8):
                    before = physical['calls']
                    pred, _ = bridge.generate(b, max_iter=steps, temperature=1., partial_masks=~target & b['input_ids'].ne(bridge.bos_id), sampling_strategy='argmax', cfg_scale=0.)
                    assert physical['calls']-before == steps
                    predictions[f'original{steps}'] = (pred, steps)
            expected_calls = 1+sum(f['groups'] for f in families)+(12 if args.phase == 'confirm' else 0)
            assert physical['calls'] == expected_calls
            forwards += physical['calls']*len(chunk)
            np.savez_compressed(dest/f'batch_{start:05d}.npz', **arrays)
            for i, row in enumerate(chunk):
                ref = gold[i, target[i]].tolist()
                for name, (pred, nfe) in predictions.items():
                    guess = pred[i, target[i]].tolist(); correct = sum(a == b for a, b in zip(guess, ref))
                    scores[name][row['dataset']+':'+row['mode']].append(100*correct/len(ref))
                    stream.write(json.dumps(dict(id=row['id'], group=row['group'], dataset=row['dataset'], mode=row['mode'], fold=row['fold'],
                        policy=name, reference_ids=ref, prediction_ids=guess, correct=correct, length=len(ref), logical_nfe=nfe))+'\n')
            stream.flush(); queries.flush()
            print(args.phase, args.checkpoint, start+len(chunk), '/', len(rows), flush=True)
    write(dest/'summary.json', dict(status='passed', n=len(rows), physical_sample_forwards=forwards,
        mean_sequence_aar={n: {k: float(np.mean(s)) for k, s in task.items()} for n, task in scores.items()},
        gate_cost_separate=True, plan_sha256=sha(args.out/'plan.json')))
    write(dest/'status.json', dict(status='passed', finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('gate', 'run')); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phase', choices=('explore', 'confirm'), default='explore')
    p.add_argument('--checkpoint', choices=('112000', '50000'), default='112000')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = p.parse_args(); globals()[args.command](args)
