"""Activate pllm; python -B SCRIPT ABS_DEBUG_RUN to independently audit artifacts.

Recompute per-sequence AAR and paired bootstrap intervals from saved tokens.
Verify source/data identities, commit trajectories, and one-step equivalence.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    run = Path(sys.argv[1])
    plan = json.loads((run/'plan.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    assert summary['status'] == 'passed'
    for path, expected in {**plan['source_sha256'], **plan['data_sha256']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    all_scores, all_predictions, reports = {}, {}, {}
    for name, case in summary['cases'].items():
        predictions = json.loads((run/f'{name}_predictions.json').read_text())
        traces = json.loads((run/f'{name}_traces.json').read_text())
        assert len(predictions) == len(traces) == plan['n']
        scores, total_steps, totals = [], 0, dict(committed=0, committed_correct=0,
            committed_initial_correct=0, initial_right_to_wrong=0, initial_wrong_to_right=0)
        for pred, trace in zip(predictions, traces):
            assert pred['key'] == trace['key']
            gold, guess = pred['reference_ids'], pred['prediction_ids']
            assert len(gold) == len(guess) == pred['length']
            correct = sum(a == b for a, b in zip(gold, guess))
            assert correct == pred['correct']
            assert abs(correct/len(gold)-pred['aar']) < 1e-6
            scores.append(correct/len(gold)*100)
            positions = trace['steps'][0]['positions']
            initial = trace['steps'][0]['proposal_ids']
            committed, previous = set(), None
            for step in trace['steps']:
                newly = set(step['committed_positions'])
                assert not newly & committed and newly <= set(positions)
                assert step['pending_before'] == len(gold)-len(committed)
                assert step['committed'] == len(newly)
                assert step['pending_after'] == step['pending_before']-len(newly)
                indexes = [i for i, p in enumerate(positions) if p in newly]
                if previous is not None:
                    assert all(step['accepted_ids'][i] == previous[i] for i, p in enumerate(positions) if p in committed)
                observed = dict(committed=len(newly),
                    committed_correct=sum(step['proposal_ids'][i] == gold[i] for i in indexes),
                    committed_initial_correct=sum(initial[i] == gold[i] for i in indexes),
                    initial_right_to_wrong=sum(initial[i] == gold[i] and step['proposal_ids'][i] != gold[i] for i in indexes),
                    initial_wrong_to_right=sum(initial[i] != gold[i] and step['proposal_ids'][i] == gold[i] for i in indexes))
                for key, value in observed.items():
                    assert step[key] == value
                    totals[key] += value
                committed |= newly
                previous = step['accepted_ids']
                total_steps += not newly
            assert committed == set(positions) and previous == guess
        assert totals == case['transitions']
        assert total_steps == case['no_op_sample_steps']
        assert abs(np.mean(scores)-case['mean_sequence_aar_percent']) < 1e-5
        all_scores[name], all_predictions[name] = np.array(scores), predictions
        reports[name] = dict(aar_percent=float(np.mean(scores)), no_op_sample_steps=total_steps,
                             transitions=totals)
    assert all_predictions['raw_logit_iter1'] == all_predictions['logprob_iter1']
    for predictions in all_predictions.values():
        for current, baseline in zip(predictions, all_predictions['raw_logit_iter1']):
            assert (current['key'], current['row'], current['reference_ids']) == (baseline['key'], baseline['row'], baseline['reference_ids'])
    generator = np.random.default_rng(42)
    samples = generator.integers(0, plan['n'], size=(10000, plan['n']))
    comparisons = {}
    for steps in plan['steps']:
        for current, base in [(f'raw_logit_iter{steps}', 'raw_logit_iter1'),
                              (f'logprob_iter{steps}', f'raw_logit_iter{steps}')]:
            differences = all_scores[current]-all_scores[base]
            low, high = np.quantile(differences[samples].mean(1), [.025, .975])
            comparisons[current+' minus '+base] = dict(delta_pp=float(differences.mean()),
                paired_sequence_bootstrap_95ci_pp=[float(low), float(high)])
    report = dict(status='passed', n_sequences=plan['n'], cases=reports, comparisons=comparisons,
                  limitations='Exploratory one-checkpoint H3 test-set diagnostic. CPU fp32/batch4; no production-default selection. Bootstrap intervals are unadjusted descriptive intervals.')
    (run/'independent_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
