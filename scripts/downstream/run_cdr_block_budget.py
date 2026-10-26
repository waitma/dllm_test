"""Activate pllm/protenix_abtcr; python -B FILE gate|run --out ABS --checkpoint 112000.

Compare frozen grouped-soft inference at four and eight logical forwards.
GPU execution uses eval_jobs YAML; every rerun prior and four-forward posterior
must exactly reproduce the preceding ablation on the same rows and batches.
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
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def context(out):
    plan = read(out / 'plan.json')
    for field in ('source_sha256', 'external_source_sha256', 'parent_evidence_sha256'):
        for path, digest in plan[field].items():
            assert sha(path) == digest, path
    assert sha(out / 'rows.json') == plan['rows_sha256'] == plan['parent_rows_sha256']
    assert sha(out / 'data_plan.json') == plan['data_plan_sha256']
    parent = Path(plan['parent'])
    assert sha(parent / 'plan.json') == plan['parent_plan_sha256']
    previous = module(parent / 'source/run_cdr_block_ablation.py', 'budget_ablation_parent')
    old_plan, block, old, reference, sampler, dev = previous.context(parent)
    assert old_plan['checkpoints'] == plan['checkpoints']
    assert old_plan['batch_size'] == plan['batch_size'] == 24
    return plan, block, old, reference, sampler, dev


def gate(args):
    import torch
    torch.set_num_threads(4)
    plan, block, old, reference, sampler, dev = context(args.out)
    adapter, evaluator, _, model, identity = old.setup(
        Path(plan['marginal_parent']), plan['checkpoints'][args.checkpoint], args.device)
    with torch.inference_mode():
        f, batch, _, target, maps, bridge, predict, prior, physical = block.prepare(
            adapter, evaluator, old, model, dev[:2], args.device)
        changed = [{**row, 'gold_h': 'G' * len(row['gold_h']), 'gold_l': 'G' * len(row['gold_l'])} for row in dev[:2]]
        f2, b2, _, t2, _ = adapter.airgen_inputs(model, changed)
        assert all(torch.equal(value.cpu(), f2[key]) for key, value in f.items())
        assert torch.equal(batch['input_ids'].cpu(), b2['input_ids']) and torch.equal(target.cpu(), t2)
        native, _ = bridge.generate(batch, max_iter=1, temperature=1.,
            partial_masks=~target & batch['input_ids'].ne(bridge.bos_id), sampling_strategy='argmax', cfg_scale=0.)
        assert torch.equal(native[target], prior.argmax(-1)[target])
        common = dict(rule='interleaved', temperature=.5, strength=.5)
        expected, _ = reference.posterior(model, predict, batch, target, f, maps, prior, groups=3, **common)
        actual, nfe = sampler.posterior(model, predict, batch, target, f, maps, prior, groups=3, **common)
        assert nfe == 4 and torch.equal(actual, expected)
        traces = []
        before = physical['calls']
        def observer(index, scored, feedback, weights, probs):
            traces.append((index, scored, feedback, weights, probs))
        posterior, nfe = sampler.posterior(model, predict, batch, target, f, maps, prior,
            groups=7, observer=observer, **common)
        assert nfe == 8 and physical['calls'] - before == 7 and len(traces) == 7
        counts = torch.zeros_like(target, dtype=torch.int)
        w = prior.square(); w /= w.sum(-1, keepdim=True)
        ordinal = target.long().cumsum(-1) - 1
        for index, scored, feedback, weights, probs in traces:
            assert torch.equal(scored, target & ordinal.remainder(7).eq(index))
            assert torch.equal(feedback, target & ~scored)
            assert torch.equal(weights[feedback], w[feedback]) and weights[~feedback].eq(0).all()
            assert torch.equal(posterior[scored], probs[scored])
            counts += scored
        assert counts[target].eq(1).all() and counts[~target].eq(0).all()
        zero, nfe = sampler.posterior(model, predict, batch, target, f, maps, prior,
            groups=7, **{**common, 'strength': 0.})
        assert nfe == 8 and torch.equal(zero, prior)
        assert torch.equal(predict(batch['input_ids']), prior)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert {p.dtype for p in model.parameters() if p.is_floating_point()} == {torch.float32}
    record = dict(status='passed', checkpoint=args.checkpoint, device=args.device, identity=identity, n=2,
        no_reference_in_features=True, original_single_pass_exact=True, grouped4_matches_parent_exact=True,
        grouped8_partition_and_weights_audited=True, grouped8_zero_strength_exact=True,
        hook_cleanup_exact=True, fp32_frozen_model=True,
        physical_sample_forwards=physical['calls'] * 2, plan_sha256=sha(args.out / 'plan.json'))
    write(args.out / f'gate_{args.device}_{args.checkpoint}.json', record)
    print(json.dumps(record), flush=True)


def run(args):
    import torch
    torch.set_num_threads(4)
    plan, block, old, _, sampler, _ = context(args.out)
    gate_record = read(args.out / f'gate_{args.device}_{args.checkpoint}.json')
    assert gate_record['status'] == 'passed' and gate_record['plan_sha256'] == sha(args.out / 'plan.json')
    rows = read(args.out / 'rows.json')
    assert len(rows) == 5254 and rows == read(Path(plan['parent']) / 'rows.json')
    destination = args.out / f'evaluation_{args.checkpoint}'
    destination.mkdir(exist_ok=False)
    write(destination / 'status.json', dict(status='running', started_utc=datetime.now(timezone.utc).isoformat()))
    adapter, evaluator, _, model, identity = old.setup(
        Path(plan['marginal_parent']), plan['checkpoints'][args.checkpoint], args.device)
    write(destination / 'identity.json', dict(model=identity, phase='budget', plan_sha256=sha(args.out / 'plan.json')))
    scores = defaultdict(lambda: defaultdict(list))
    forwards = 0
    with (destination / 'predictions.jsonl').open('x') as stream, (destination / 'query_audit.jsonl').open('x') as queries, torch.inference_mode():
        for start in range(0, len(rows), plan['batch_size']):
            chunk = rows[start:start + plan['batch_size']]
            f, batch, gold, target, maps, _, predict, prior, physical = block.prepare(
                adapter, evaluator, old, model, chunk, args.device)
            positions = [mask.nonzero().flatten() for mask in target]
            lengths = np.array([len(pos) for pos in positions], dtype=np.int64)
            def pack(probs):
                packed = np.zeros((len(chunk), int(lengths.max()), probs.shape[-1]), dtype=np.float32)
                for i, pos in enumerate(positions):
                    packed[i, :len(pos)] = probs[i, pos].cpu().numpy()
                return packed
            position_array = np.full((len(chunk), int(lengths.max())), -1, dtype=np.int64)
            for i, pos in enumerate(positions):
                position_array[i, :len(pos)] = pos.cpu().numpy()
            arrays = dict(ids=np.array([row['id'] for row in chunk]), lengths=lengths,
                          positions=position_array, prior=pack(prior))
            predictions = {'single_pass': (prior.argmax(-1), 1)}
            for family in plan['families']:
                weights_trace, probs_trace = [], []
                def observer(index, scored, feedback, weights, probs):
                    assert index == len(weights_trace)
                    weights_trace.append(pack(weights)); probs_trace.append(pack(probs))
                    for i, row in enumerate(chunk):
                        queries.write(json.dumps(dict(id=row['id'], family=family['name'], query=index,
                            scored=scored[i, target[i]].cpu().tolist(), feedback=feedback[i, target[i]].cpu().tolist())) + '\n')
                config = {key: value for key, value in family.items() if key not in ('name', 'prior_weight')}
                post, nfe = sampler.posterior(model, predict, batch, target, f, maps, prior, **config, observer=observer)
                assert nfe == family['groups'] + 1 and len(weights_trace) == family['groups']
                arrays[family['name']] = pack(post)
                arrays[family['name'] + '__weights'] = np.stack(weights_trace)
                arrays[family['name'] + '__queries'] = np.stack(probs_trace)
                weight = family['prior_weight']
                predictions[family['name']] = ((weight * prior + (1 - weight) * post).argmax(-1), nfe)
            assert physical['calls'] == 11
            # Batch layout and all saved four-forward probabilities must exactly
            # match the completed parent run before accepting this extension.
            with np.load(Path(plan['parent']) / f'evaluation_{args.checkpoint}/batch_{start:05d}.npz', allow_pickle=False) as previous:
                for key in ('ids', 'lengths', 'positions', 'prior'):
                    assert np.array_equal(arrays[key], previous[key]), (start, key)
                for suffix in ('', '__weights', '__queries'):
                    assert np.array_equal(arrays['grouped4' + suffix], previous['soft_frozen' + suffix]), (start, suffix)
            forwards += physical['calls'] * len(chunk)
            np.savez_compressed(destination / f'batch_{start:05d}.npz', **arrays)
            for i, row in enumerate(chunk):
                reference = gold[i, target[i]].tolist()
                for name, (prediction, nfe) in predictions.items():
                    guess = prediction[i, target[i]].tolist()
                    correct = sum(a == b for a, b in zip(guess, reference))
                    scores[name][row['dataset'] + ':' + row['mode']].append(100 * correct / len(reference))
                    stream.write(json.dumps(dict(id=row['id'], group=row['group'], dataset=row['dataset'], mode=row['mode'],
                        fold=row['fold'], policy=name, reference_ids=reference, prediction_ids=guess,
                        correct=correct, length=len(reference), logical_nfe=nfe)) + '\n')
            stream.flush(); queries.flush()
            print('BUDGET', args.checkpoint, start + len(chunk), '/', len(rows), flush=True)
    write(destination / 'summary.json', dict(status='passed', n=len(rows), physical_sample_forwards=forwards,
        mean_sequence_aar={name: {key: float(np.mean(values)) for key, values in modes.items()} for name, modes in scores.items()},
        prior_and_grouped4_exact_to_parent=True, gate_cost_separate=True, plan_sha256=sha(args.out / 'plan.json')))
    write(destination / 'status.json', dict(status='passed', finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('gate', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('112000', '50000'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    globals()[args.command](args)
