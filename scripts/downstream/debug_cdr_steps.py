"""Activate pllm/protenix_abtcr; python -B SCRIPT --checkpoint ABS --out NEW_ABS.

CPU-only, fixed-chain SAb23H2 H3 diagnostic. Compare the original AirGen
argmax decoder with normalized log-probability confidence, keeping proposals,
linear schedule, tie handling, EOS penalty and inputs unchanged. No retraining
or production default changes. References are used only for scoring and traces.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
EVALUATOR = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/airgen_evaluate.py'


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def instrument(bridge, gold, target, confidence):
    import torch
    net, forward, commit = bridge.net, bridge.forward_decoder, bridge._decoding
    cache, trace = {}, []

    def capture(**kwargs):
        result = net(**kwargs)
        cache['logits'] = result['logits'].clone()
        return result

    def propose(**kwargs):
        previous = kwargs['prev_decoder_out']
        result = forward(**kwargs)
        logits = cache.pop('logits').to(result['output_scores'].dtype)
        logits[..., bridge.special_token_list] = -math.inf
        logits[..., bridge.eos_id] += 1e-3*math.log(previous['step']/previous['max_step']+1e-3)
        score, token = logits.max(-1)
        pending = previous['output_masks']
        assert torch.equal(token[pending], result['output_tokens'][pending])
        assert torch.equal(score[pending], result['output_scores'][pending])
        logp = logits.float().log_softmax(-1).gather(-1, token[..., None]).squeeze(-1)
        cache['probability'] = logp.exp()
        if 'initial' not in cache:
            cache['initial'] = token.clone()
        if confidence == 'logprob':
            result['output_scores'][pending] = logp[pending].to(result['output_scores'].dtype)
        return result

    def accept(**kwargs):
        before = kwargs['output_tokens'].clone()
        pending = kwargs['xt_neq_x0'].clone()
        proposal = kwargs['cur_tokens'].clone()
        masks, tokens, scores = commit(**kwargs)
        assert torch.equal(tokens[~pending], before[~pending])
        assert torch.equal(masks, tokens.eq(bridge.mask_id))
        newly = pending & ~masks
        initial = cache['initial']
        records = []
        for i in range(len(tokens)):
            positions = target[i].nonzero().flatten()
            records.append(dict(
                pending_before=int(pending[i].sum()), pending_after=int(masks[i].sum()),
                committed=int(newly[i].sum()),
                committed_correct=int((newly[i] & proposal[i].eq(gold[i])).sum()),
                committed_initial_correct=int((newly[i] & initial[i].eq(gold[i])).sum()),
                initial_right_to_wrong=int((newly[i] & initial[i].eq(gold[i]) & proposal[i].ne(gold[i])).sum()),
                initial_wrong_to_right=int((newly[i] & initial[i].ne(gold[i]) & proposal[i].eq(gold[i])).sum()),
                positions=positions.tolist(), pending=pending[i, positions].tolist(),
                proposal_ids=proposal[i, positions].tolist(),
                proposal_probabilities=cache['probability'][i, positions].tolist(),
                confidence_scores=kwargs['cur_scores'][i, positions].tolist(),
                committed_positions=newly[i].nonzero().flatten().tolist(),
                accepted_ids=tokens[i, positions].tolist()))
        trace.append(dict(step=kwargs['t'], rows=records))
        return masks, tokens, scores

    bridge.net, bridge.forward_decoder, bridge._decoding = capture, propose, accept
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(4)
    from downstream.fixed_chain_eval import airgen_inputs, load
    model, identity = load(args.checkpoint, 'cpu')
    spec = importlib.util.spec_from_file_location('cdr_debug_evaluator', EVALUATOR)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    rows, files = evaluator.records('sab23')
    rows = [row for row in rows if row['mode'] == 'cdrh3'][:args.limit]
    assert rows
    sources = [Path(__file__), EVALUATOR, ROOT/'downstream/fixed_chain_eval.py',
               evaluator.AIR/'src/byprot/models/lm/dplm_multichain.py',
               evaluator.AIR/'src/byprot/models/lm/model_utils.py']
    write(args.out/'plan.json', dict(identity=identity, dataset='sab23', mode='cdrh3',
          n=len(rows), device='cpu', precision='fp32', batch_size=args.batch_size,
          steps=[1, 2, 4, 8], confidences=['raw_logit', 'logprob'],
          sampling='argmax', temperature=1, cfg=0, seed=42,
          source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
          data_sha256=files, created_utc=datetime.now(timezone.utc).isoformat()))
    cases, outputs = {}, {}
    # A per-position offset preserves categorical probabilities but changes raw-logit ranking.
    toy = torch.tensor([[2., 0.], [1., 0.]])
    shifted = toy + torch.tensor([[0.], [10.]])
    assert torch.equal(toy.argmax(-1), shifted.argmax(-1))
    assert torch.allclose(toy.softmax(-1), shifted.softmax(-1))
    assert toy.max(-1).values.argmax() != shifted.max(-1).values.argmax()
    checks = dict(raw_logit_not_offset_invariant=True, proposal_tokens_preserved=True,
                  committed_context_preserved=True, hidden_reference_independent=True,
                  original_decoder_equivalence=True)
    for steps in (1, 2, 4, 8):
        for confidence in ('raw_logit', 'logprob'):
            name = f'{confidence}_iter{steps}'
            predictions, all_traces = [], []
            start = time.monotonic()
            for offset in range(0, len(rows), args.batch_size):
                chunk = rows[offset:offset+args.batch_size]
                features, batch, gold, target, maps = airgen_inputs(model, chunk)
                if steps == 1 and confidence == 'raw_logit':
                    altered = [{**r, 'gold_h': 'G'*len(r['gold_h']), 'gold_l': 'G'*len(r['gold_l'])} for r in chunk]
                    f2, b2, _, m2, _ = airgen_inputs(model, altered)
                    assert all(torch.equal(v, f2[k]) for k, v in features.items())
                    assert torch.equal(batch['input_ids'], b2['input_ids']) and torch.equal(target, m2)
                bridge = evaluator.decoder(model, features, maps)
                trace = instrument(bridge, gold, target, confidence)
                kwargs = dict(max_iter=steps, temperature=1., sampling_strategy='argmax', cfg_scale=0.,
                              partial_masks=~target & batch['input_ids'].ne(bridge.bos_id))
                with torch.inference_mode():
                    prediction, _ = bridge.generate(batch, **kwargs)
                    if offset == 0 and confidence == 'raw_logit':
                        original = evaluator.decoder(model, features, maps)
                        expected, _ = original.generate(batch, **kwargs)
                        assert torch.equal(prediction, expected)
                assert torch.equal(prediction[~target], batch['input_ids'][~target])
                assert not prediction[target].eq(bridge.mask_id).any()
                for i, row in enumerate(chunk):
                    guess, ref = prediction[i, target[i]], gold[i, target[i]]
                    predictions.append(dict(key=row['key'], row=row['row'], prediction_ids=guess.tolist(),
                                            reference_ids=ref.tolist(), correct=int(guess.eq(ref).sum()),
                                            length=len(ref), aar=float(guess.eq(ref).float().mean())))
                    all_traces.append(dict(key=row['key'], steps=[dict(step=t['step'], **t['rows'][i]) for t in trace]))
                print(name, offset+len(chunk), '/', len(rows), flush=True)
            outputs[name] = predictions
            mean = sum(p['aar'] for p in predictions)/len(predictions)
            totals = {k: sum(s[k] for row in all_traces for s in row['steps']) for k in
                      ('committed', 'committed_correct', 'committed_initial_correct', 'initial_right_to_wrong', 'initial_wrong_to_right')}
            assert totals['committed'] == sum(p['length'] for p in predictions)
            assert totals['committed_correct'] == sum(p['correct'] for p in predictions)
            assert totals['committed_correct']-totals['committed_initial_correct'] == totals['initial_wrong_to_right']-totals['initial_right_to_wrong']
            cases[name] = dict(mean_sequence_aar_percent=mean*100,
                airgen_aar_percent=float(__import__('numpy').round(mean, 4)*100),
                transitions=totals,
                no_op_sample_steps=sum(s['committed']==0 for row in all_traces for s in row['steps']),
                seconds=time.monotonic()-start)
            write(args.out/f'{name}_predictions.json', predictions)
            write(args.out/f'{name}_traces.json', all_traces)
            write(args.out/'summary.json', dict(status='running', checks=checks, cases=cases))
    assert outputs['raw_logit_iter1'] == outputs['logprob_iter1']
    checks['one_step_raw_equals_logprob'] = True
    write(args.out/'summary.json', dict(status='passed', checks=checks, cases=cases,
          scope='One checkpoint, all 60 SAb23H2 H3 if limit=60; exploratory test-set diagnostic, not validation or a production fix.'))
    print(json.dumps(cases, indent=2), flush=True)


if __name__ == '__main__':
    main()
