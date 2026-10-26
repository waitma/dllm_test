"""Activate pllm/protenix_abtcr; python -B FILE --out ABS --device cuda.

One-pass 50000 probe on the identical development cohort; combine with frozen
112000 probabilities already collected, comparing against BOTH individuals.
No confirm scoring and no parameter updates. GPU execution uses eval_jobs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(); out = args.out
    spec = importlib.util.spec_from_file_location('frozen_marginal_runner', out/'source/run_cdr_marginal_exploration.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    plan, all_rows = runner.verify(out)
    parent = out/'explore_112000'
    prior_audit = json.loads((parent/'independent_audit.json').read_text()); assert prior_audit['status'] == 'passed'
    old = {}
    for path in sorted(parent.glob('posterior_batch_*.npz')):
        assert runner.sha(path) == prior_audit['posterior_sha256'][str(path)]
        with np.load(path, allow_pickle=False) as arrays:
            for i, (key, length) in enumerate(zip(arrays['ids'].tolist(), arrays['lengths'].tolist())):
                old[key] = arrays['prior'][i, :length].copy()
    rows = [r for r in all_rows if r['split'] == 'explore' and r['mode'] == 'cdrh3']
    assert set(old) == {r['id'] for r in rows}
    dest = out/'checkpoint_ensemble'; dest.mkdir(exist_ok=False)
    runner.write(dest/'status.json', dict(status='running'))
    try:
        import torch
        torch.set_num_threads(4)
        adapter, evaluator, sampler, model, identity = runner.setup(out, plan['checkpoints']['50000'], args.device)
        assert json.loads((Path(plan['checkpoints']['50000'])/'vocabulary.json').read_text()) == json.loads((Path(plan['checkpoints']['112000'])/'vocabulary.json').read_text())
        runner.write(dest/'identity.json', dict(model=identity, reused_parent=str(parent),
            reused_parent_audit_sha256=runner.sha(parent/'independent_audit.json'),
            source_sha256=runner.sha(Path(__file__)), n=len(rows)))
        scores = defaultdict(list); outputs = defaultdict(dict)
        with (dest/'predictions.jsonl').open('x') as stream:
            for start in range(0, len(rows), plan['batch_size']):
                chunk = rows[start:start+plan['batch_size']]
                f, b, gold, target, maps = adapter.airgen_inputs(model, chunk)
                f = {k: v.to(args.device) for k, v in f.items()}; b = {k: v.to(args.device) for k, v in b.items()}
                target, gold = target.to(args.device), gold.to(args.device)
                bridge = evaluator.decoder(model, f, maps); predict, counter = runner.prediction_fn(bridge, b, target)
                with torch.inference_mode():
                    prior = sampler.base_distribution(predict, b, target, bridge.mask_id)
                    original, _ = bridge.generate(b, max_iter=1, temperature=1.,
                        partial_masks=~target & b['input_ids'].ne(bridge.bos_id), sampling_strategy='argmax', cfg_scale=0.)
                    assert torch.equal(prior.argmax(-1)[target], original[target])
                arrays = {}
                for i, r in enumerate(chunk):
                    p50 = prior[i, target[i]].cpu().numpy(); p112 = old[r['id']]
                    assert p50.shape == p112.shape
                    arrays[r['id']] = p50
                    distributions = {'single50000': p50, 'single112000': p112}
                    for weight in (.1, .25, .5, .75, .9):
                        distributions[f'arithmetic_w112_{weight}'] = weight*p112+(1-weight)*p50
                        z = weight*np.log(p112.clip(1e-30))+(1-weight)*np.log(p50.clip(1e-30))
                        z = np.exp(z-z.max(-1, keepdims=True)); z /= z.sum(-1, keepdims=True)
                        distributions[f'geometric_w112_{weight}'] = z
                    ref = gold[i, target[i]].tolist()
                    for name, p in distributions.items():
                        assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(-1), 1., atol=1e-5)
                        guess = p.argmax(-1).tolist(); correct = sum(x == y for x, y in zip(guess, ref))
                        record = dict(id=r['id'], group=r['group'], policy=name,
                            reference_ids=ref, prediction_ids=guess, correct=correct, length=len(ref))
                        outputs[name][r['id']] = record; scores[name].append(correct/len(ref)*100)
                        stream.write(json.dumps(record)+'\n')
                np.savez_compressed(dest/f'prior50000_batch_{start:05d}.npz', **arrays)
                stream.flush(); print('ENSEMBLE PROBE', start+len(chunk), '/', len(rows), flush=True)
        means = {name: float(np.mean(values)) for name, values in scores.items()}
        strong = max(('single50000', 'single112000'), key=lambda name: means[name])
        candidates = [name for name in means if not name.startswith('single')]
        winner = min(candidates, key=lambda name: (-means[name], name))
        runner.write(dest/'summary.json', dict(status='passed', n=len(rows), scores=means,
            stronger_individual=strong, strongest_candidate=winner, delta_vs_stronger_pp=means[winner]-means[strong],
            per_method_forward_equivalents=2, individual_forward_equivalents=1,
            actual_new_sampler_sample_forwards=len(rows), actual_new_equivalence_sample_forwards=len(rows),
            source_sha256=runner.sha(Path(__file__)),
            limitation='Exploration only; two-checkpoint inference ensemble, not a single-checkpoint sampler fix.'))
        runner.write(dest/'status.json', dict(status='success'))
        print(json.dumps(dict(scores=means, stronger_individual=strong, winner=winner,
                              delta_vs_stronger=means[winner]-means[strong])), flush=True)
    except BaseException as exc:
        runner.write(dest/'status.json', dict(status='failed', error=repr(exc))); raise


if __name__ == '__main__': main()
