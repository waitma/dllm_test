"""CPU-only probability checks with artificial integer arrays; no dataset reads.

Activate pllm, then run:
python scripts/architecture_prototypes/noise_probability_diagnostics.py
Only three pure masking functions are AST-loaded from the existing sampler.
No model, vocabulary, biological sequence or benchmark is imported or executed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc
from scipy.stats import binom
import torch


EPSILON = .001


def probability_mass(n, eps=EPSILON, force_nonempty=True):
    k = np.arange(n + 1)
    result = (1 - betainc(k + 1, n - k + 1, eps)) / ((n + 1) * (1 - eps))
    if force_nonempty:
        result[1] += result[0]
        result[0] = 0
    assert abs(result.sum() - 1) < 1e-12
    return result


def full_probability(n, eps=EPSILON):
    return (1 - eps ** (n + 1)) / ((n + 1) * (1 - eps))


def load_masking_functions(path):
    names = {'_diffusion_eligible_mask', '_guarantee_at_least_one_corruption',
             'sample_bioseq_diffusion_noise'}
    parsed = ast.parse(path.read_text())
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {'torch': torch, 'Any': Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['sample_bioseq_diffusion_noise']


def actual_sampler_check(sample, content_length, targets):
    """All integers are arbitrary symbol IDs; the last symbol of each target is END."""
    n = targets * (content_length + 1)
    b = 2048
    ids = torch.full((b, n + 3), 7, dtype=torch.long, device='cpu')
    eligible = torch.zeros_like(ids, dtype=torch.bool)
    eligible[:, 2:2 + n] = True
    attention = torch.ones_like(ids, dtype=torch.bool)
    attention[:, -1] = False
    residue = eligible.clone()
    for target in range(targets):
        end = 2 + (target + 1) * (content_length + 1) - 1
        residue[:, end] = False
        ids[:, end] = 8
    batch = dict(input_ids=ids, attention_mask=attention,
                 diffusion_eligible_mask=eligible, diffusion_loss_mask=eligible,
                 residue_mask=residue)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7351 + n)
        noised, labels, selected, times = sample(batch, mask_token_id=99)
        assert selected.any(1).all()
        assert not (selected & ~eligible).any()
        assert torch.equal(noised[selected], torch.full_like(noised[selected], 99))
        assert torch.equal(labels[selected], ids[selected])
        assert (labels[~selected] == -100).all()
        assert (times >= EPSILON).all() and (times <= 1).all()
        _, _, all_selected, _ = sample(batch, mask_token_id=99, token_t=torch.ones_like(ids))
        assert torch.equal(all_selected, eligible)
    return {'rows': b, 'selected_min': int(selected.sum(1).min()),
            'selected_max': int(selected.sum(1).max()), 'device': 'cpu',
            'context_padding_and_labels_checked': True, 'explicit_full_mask_checked': True}


def diagnostics(content_length, targets, trials, rng, sample):
    n = targets * (content_length + 1)
    pmf = probability_mass(n)
    times = rng.uniform(EPSILON, 1, size=trials)
    raw_k = rng.binomial(n, times)
    k = np.maximum(raw_k, 1)
    thresholds = {'all': n, 'at_most_one_visible': n - 1,
                  'at_least_95_percent': math.ceil(.95 * n),
                  'at_least_90_percent': math.ceil(.9 * n)}
    events = {}
    for name, minimum in thresholds.items():
        analytic = float(pmf[minimum:].sum())
        integral = quad(lambda t: binom.sf(minimum - 1, n, t)/(1 - EPSILON),
                        EPSILON, 1, epsabs=1e-12)[0]
        assert abs(analytic - integral) < 1e-10
        measured = float((k >= minimum).mean())
        se = math.sqrt(analytic * (1 - analytic) / trials)
        assert abs(measured - analytic) <= 6 * se + 1 / trials
        events[name] = dict(minimum_masked=minimum, analytic_probability=analytic,
                            monte_carlo_probability=measured, binomial_standard_error=se,
                            independent_quadrature=integral)
    # Given a masked count, exchangeability makes the selected subset uniform.
    selected_end = rng.hypergeometric(targets, n - targets, k)
    coefficient = selected_end / k
    end_share = targets / n
    coefficient_se = float(coefficient.std(ddof=1) / math.sqrt(trials))
    assert abs(float(coefficient.mean()) - end_share) <= 6 * coefficient_se
    zero_before_force = (1 - EPSILON) ** n / (n + 1)
    expected_k = n * (1 + EPSILON) / 2 + zero_before_force
    assert abs(float(np.dot(np.arange(n + 1), pmf)) - expected_k) < 1e-10
    result = dict(content_length_per_target=content_length, target_count=targets, eligible_count=n,
                  events=events, probability_zero_before_force=zero_before_force,
                  expected_masked_count=expected_k,
                  terminal_coefficient_share=end_share,
                  mc_mean_per_record_terminal_coefficient=float(coefficient.mean()),
                  mc_pooled_terminal_coefficient=float(selected_end.sum() / k.sum()),
                  terminal_coefficient_standard_error=coefficient_se,
                  actual_sampler=actual_sampler_check(sample, content_length, targets))
    if targets == 2:
        one = full_probability(content_length + 1)
        both = full_probability(n)
        result['shared_t_two_target'] = dict(one_specified_target_full=one,
            both_targets_full=both, at_least_one_target_full=2 * one - both,
            exactly_one_target_full=2 * (one - both), incorrect_independent_product=one ** 2)
    return result


def normalization_counterexample():
    """The same four scalar per-record mean losses, counts and gradient graph."""
    counts = torch.tensor([1., 9., 1., 9.], dtype=torch.float64)
    groupings = {'mixed_pairs': [[0, 1], [2, 3]],
                 'same_count_pairs': [[0, 2], [1, 3]],
                 'unequal_last_batch': [[0], [1, 2, 3]]}
    rows = {}
    for name, groups in groupings.items():
        records = torch.tensor([1., 3., 1., 3.], dtype=torch.float64, requires_grad=True)
        current = torch.stack([(records[group]*counts[group]).sum()/counts[group].sum()
                               for group in groups]).mean()
        # For equal microbatch sizes this equals averaging microbatch means;
        # unequal sizes require weighting by actual records/global record count.
        per_record = sum(records[group].sum()/len(records) for group in groups)
        current_gradient = torch.autograd.grad(current, records, retain_graph=True)[0]
        record_gradient = torch.autograd.grad(per_record, records)[0]
        torch.testing.assert_close(record_gradient, torch.full_like(records, .25))
        assert float(per_record.detach()) == 2.
        rows[name] = dict(groups=groups, current_loss=float(current.detach()),
                          current_gradient_weights=current_gradient.tolist(),
                          record_weighted_loss=float(per_record.detach()),
                          record_weighted_gradient_weights=record_gradient.tolist())
    assert abs(rows['mixed_pairs']['current_loss']-2.8) < 1e-12
    assert rows['same_count_pairs']['current_loss'] == 2.
    return dict(per_record_mean_ce=[1,3,1,3], masked_counts=[1,9,1,9],
                global_token_mean=2.8, groups=rows,
                interpretation='Current equal-microbatch loss means differ; explicit per-record normalization is additive.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=200000)
    args = parser.parse_args()
    if not 10000 <= args.trials <= 1000000:
        raise ValueError('Keep CPU diagnostic bounded to 10k–1M trials per case')
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2]
    source = root/'dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    sample = load_masking_functions(source)
    rng = np.random.default_rng(91357)
    rows = [diagnostics(length, count, args.trials, rng, sample)
            for length in (24, 127, 255) for count in (1, 2)]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report = dict(status='passed_cpu_artificial_inputs_only', created_at=datetime.now(timezone.utc).isoformat(),
                  epsilon=EPSILON, seed=91357, monte_carlo_trials_per_case=args.trials,
                  actual_sampler_rows_per_case=2048, rows=rows,
                  normalization_counterexample=normalization_counterexample(),
                  eligible_length_convention='N = target_count * (content_length + one END)',
                  count_only_convention={str(n): full_probability(n) for n in (24, 127, 255, 48, 254, 510)},
                  sampler_path=str(source), sampler_sha256=digest,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitations=['No trained model or loss values measured.', 'Terminal share is CE coefficient mass, not actual loss/gradient share.',
                               'Analytic probabilities use real-valued uniform t; float32 discretization is negligible here.',
                               'The forced singleton affects K=0/1 only; all listed high-mask thresholds exceed 1.',
                               'No GPU, project imports, dataset, biological sequence or benchmark used.'])
    out = root/'output/immune_noise_math_20260930'
    out.mkdir(parents=True, exist_ok=True)
    (out/'probabilities.json').write_text(json.dumps(report, indent=2)+'\n')
    for row in rows:
        print(json.dumps({k: row[k] for k in ('content_length_per_target','target_count','eligible_count','terminal_coefficient_share')} |
                         {k: value['analytic_probability'] for k,value in row['events'].items()}))
    print(report['status'])


if __name__ == '__main__':
    main()
