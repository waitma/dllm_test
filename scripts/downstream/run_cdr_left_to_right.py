"""Activate pllm/protenix_abtcr; python -B FILE gate|run --out ABS --checkpoint 112000.

Frozen, greedy, left-to-right target decoding. Submit GPU work via eval_jobs;
reuse the initial all-MASK forward for the first target, without extra priors.
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
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
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
    previous = module(parent / 'source/run_cdr_block_budget.py', 'ltr_budget_parent')
    parent_plan, block, old, _, _, dev = previous.context(parent)
    assert parent_plan['checkpoints'] == plan['checkpoints']
    assert parent_plan['batch_size'] == plan['batch_size'] == 24
    sampler = module(out / 'source/cdr_left_to_right.py', 'ltr_sampler')
    return plan, block, old, sampler, dev


def gate(args):
    import torch
    torch.set_num_threads(4)
    plan, block, old, sampler, dev = context(args.out)
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
        traces, native_calls = [], []
        expected_native = [None]
        def hook(layer, inputs):
            assert expected_native[0] is not None
            assert torch.equal(inputs[0], expected_native[0]), 'Bridge must pass the actual sequential tokens to the model'
            native_calls.append(1)
        def checked_predict(tokens):
            expected = f['input_ids'].clone()
            for i, mapping in enumerate(maps):
                virtual, actual = zip(*mapping)
                expected[i, list(actual)] = tokens[i, list(virtual)]
            expected_native[0] = expected
            return predict(tokens)
        def observe(step, inputs, scored, accepted, probs):
            traces.append((step, inputs, scored, accepted, probs))
        handle = model.backbone.transformer.wte.register_forward_pre_hook(hook)
        before = physical['calls']
        try:
            prediction, logical, padded = sampler.generate(checked_predict, batch, target,
                bridge.mask_id, prior, observer=observe)
        finally:
            handle.remove()
        lengths = target.sum(-1)
        assert torch.equal(logical, lengths) and padded == int(lengths.max())
        assert physical['calls'] - before == padded - 1 == len(native_calls)
        assert len(traces) == padded
        tokens = batch['input_ids'].clone()
        ordinal = target.long().cumsum(-1) - 1
        for step, inputs, scored, accepted, probs in traces:
            assert torch.equal(inputs, tokens)
            expected_scored = target & ordinal.eq(step)
            assert torch.equal(scored, expected_scored)
            assert tokens[target & ordinal.ge(step)].eq(bridge.mask_id).all()
            if step == 0:
                assert torch.equal(probs, prior)
            active = scored.any(-1)
            assert torch.equal(accepted[active], probs.argmax(-1)[scored])
            assert accepted[~active].eq(-1).all()
            tokens[scored] = accepted[active]
        assert torch.equal(prediction, tokens)
        assert torch.equal(prediction[~target], batch['input_ids'][~target])
        assert batch['input_ids'][target].eq(bridge.mask_id).all()
        assert torch.equal(predict(batch['input_ids']), prior)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert {p.dtype for p in model.parameters() if p.is_floating_point()} == {torch.float32}
    record = dict(status='passed', checkpoint=args.checkpoint, device=args.device, identity=identity, n=2,
        no_reference_in_features=True, original_single_pass_exact=True,
        strict_ltr_inputs_and_argmax=True, first_query_reuses_prior=True,
        observed_context_unchanged=True, native_bridge_inputs_verified=True, fp32_frozen_model=True,
        physical_sample_forwards=physical['calls'] * 2, plan_sha256=sha(args.out / 'plan.json'))
    write(args.out / f'gate_{args.device}_{args.checkpoint}.json', record)
    print(json.dumps(record), flush=True)


def run(args):
    import torch
    torch.set_num_threads(4)
    plan, block, old, sampler, _ = context(args.out)
    gate_record = read(args.out / f'gate_{args.device}_{args.checkpoint}.json')
    assert gate_record['status'] == 'passed' and gate_record['plan_sha256'] == sha(args.out / 'plan.json')
    rows = read(args.out / 'rows.json')
    assert len(rows) == 5254 and rows == read(Path(plan['parent']) / 'rows.json')
    destination = args.out / f'evaluation_{args.checkpoint}'
    destination.mkdir(exist_ok=False)
    write(destination / 'status.json', dict(status='running', started_utc=datetime.now(timezone.utc).isoformat()))
    adapter, evaluator, _, model, identity = old.setup(
        Path(plan['marginal_parent']), plan['checkpoints'][args.checkpoint], args.device)
    write(destination / 'identity.json', dict(model=identity, phase='left_to_right', plan_sha256=sha(args.out / 'plan.json')))
    scores = defaultdict(list)
    logical_total = physical_total = batch_forwards = 0
    with (destination / 'predictions.jsonl').open('x') as stream, torch.inference_mode():
        for start in range(0, len(rows), plan['batch_size']):
            chunk = rows[start:start + plan['batch_size']]
            _, batch, gold, target, _, bridge, predict, prior, physical = block.prepare(
                adapter, evaluator, old, model, chunk, args.device)
            positions = [mask.nonzero().flatten() for mask in target]
            lengths = np.array([len(pos) for pos in positions], dtype=np.int64)
            width = int(lengths.max())
            position_array = np.full((len(chunk), width), -1, dtype=np.int64)
            prior_array = np.zeros((len(chunk), width, prior.shape[-1]), dtype=np.float32)
            for i, pos in enumerate(positions):
                position_array[i, :len(pos)] = pos.cpu().numpy()
                prior_array[i, :len(pos)] = prior[i, pos].cpu().numpy()
            arrays = dict(ids=np.array([row['id'] for row in chunk]), lengths=lengths,
                          positions=position_array, prior=prior_array)
            with np.load(Path(plan['parent']) / f'evaluation_{args.checkpoint}/batch_{start:05d}.npz', allow_pickle=False) as previous:
                for key in ('ids', 'lengths', 'positions', 'prior'):
                    assert np.array_equal(arrays[key], previous[key]), (start, key)
            input_trace, probability_trace, position_trace = [], [], []
            def observer(step, inputs, scored, accepted, probs):
                assert step == len(input_trace)
                input_trace.append(inputs.cpu().numpy())
                selected = np.full(len(chunk), -1, dtype=np.int64)
                probability = np.zeros((len(chunk), probs.shape[-1]), dtype=np.float32)
                for i in range(len(chunk)):
                    pos = scored[i].nonzero().flatten()
                    assert len(pos) == int(step < lengths[i])
                    if len(pos):
                        selected[i] = int(pos[0])
                        probability[i] = probs[i, pos[0]].cpu().numpy()
                        assert int(accepted[i]) == int(probs[i, pos[0]].argmax())
                position_trace.append(selected)
                probability_trace.append(probability)
            prediction, logical, padded = sampler.generate(predict, batch, target, bridge.mask_id, prior, observer=observer)
            assert padded == width == physical['calls'] == len(input_trace)
            assert np.array_equal(logical.cpu().numpy(), lengths)
            arrays.update(query_inputs=np.stack(input_trace), selected_probs=np.stack(probability_trace),
                          selected_positions=np.stack(position_trace))
            np.savez_compressed(destination / f'batch_{start:05d}.npz', **arrays)
            logical_total += int(lengths.sum())
            physical_total += physical['calls'] * len(chunk)
            batch_forwards += physical['calls']
            for i, row in enumerate(chunk):
                reference = gold[i, target[i]].tolist()
                guess = prediction[i, target[i]].tolist()
                correct = sum(a == b for a, b in zip(guess, reference))
                scores[row['dataset'] + ':' + row['mode']].append(100 * correct / len(reference))
                stream.write(json.dumps(dict(id=row['id'], group=row['group'], dataset=row['dataset'], mode=row['mode'],
                    fold=row['fold'], policy='left_to_right', reference_ids=reference, prediction_ids=guess,
                    correct=correct, length=len(reference), logical_nfe=len(reference))) + '\n')
            stream.flush()
            print('LTR', args.checkpoint, start + len(chunk), '/', len(rows), flush=True)
    cost = read(args.out / 'data_plan.json')['cost_per_checkpoint']
    assert (logical_total, physical_total, batch_forwards) == (
        cost['logical_sample_forwards'], cost['physical_sample_forwards'], cost['physical_batch_forwards'])
    write(destination / 'summary.json', dict(status='passed', n=len(rows),
        logical_sample_forwards=logical_total, physical_sample_forwards=physical_total,
        physical_batch_forwards=batch_forwards,
        mean_sequence_aar={'left_to_right': {key: float(np.mean(values)) for key, values in scores.items()}},
        prior_exact_to_parent=True, gate_cost_separate=True, plan_sha256=sha(args.out / 'plan.json')))
    write(destination / 'status.json', dict(status='passed', finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('gate', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('112000', '50000'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    globals()[args.command](args)
