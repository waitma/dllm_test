"""Activate pllm; python -B FILE --out ABS --round 1|2|3.

Analyze completed, independently audited traces for repeated masked inputs and
repair coverage. This uses references only after generation for diagnosis.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    folder = args.out/f'round{args.round}'
    audit = json.loads((folder/'independent_audit.json').read_text())
    assert audit['status'] == 'passed'
    identity = json.loads((folder/'identity.json').read_text())
    configs = {p['name']: p for p in identity['configs']}
    vocab = json.loads((Path(identity['model']['checkpoint'])/'vocabulary.json').read_text())
    mask = vocab.index('<mask>')
    refs = {}
    for line in (folder/'predictions.jsonl').open():
        r = json.loads(line)
        if r['policy'] == 'original1': refs[r['id']] = r
    result = defaultdict(lambda: defaultdict(int))
    with (folder/'traces.jsonl').open('rb') as stream:
        digest = hashlib.sha256()
        for line in stream:
            digest.update(line)
            r = json.loads(line); p = r['policy']; ref = refs[r['id']]
            if configs[p]['kind'] == 'oracle': continue  # Actual input contains gold intervention.
            key = p+':'+ref['dataset']+':'+ref['mode']; stats = result[key]
            stats['sequences'] += 1; stats['forward_samples'] += len(r['steps'])
            inputs, remasked = set(), set()
            gold = ref['reference_ids']; initial = r['steps'][0]['after']
            for step, tr in enumerate(r['steps']):
                # Greedy repair masks before forwarding; ReMDM masks after forwarding.
                x = tuple(mask if m else t for t, m in zip(tr['before'], tr['remasked'])) \
                    if configs[p]['kind'] in ('repair', 'random_repair') else tuple(tr['before'])
                stats['repeated_masked_input_forwards'] += x in inputs
                stats['unchanged_output_forwards'] += tr['before'] == tr['after']
                inputs.add(x)
                if step: remasked.update(i for i, m in enumerate(tr['remasked']) if m)
            stats['unique_masked_inputs'] += len(inputs)
            if configs[p]['kind'] in ('repair', 'random_repair'):
                wrong = {i for i, (a, b) in enumerate(zip(initial, gold)) if a != b}
                final = r['steps'][-1]['after']
                stats['initial_wrong_positions'] += len(wrong)
                stats['initial_wrong_ever_revisited'] += len(wrong & remasked)
                stats['initial_correct_revisited'] += len(remasked - wrong)
                stats['unique_positions_revisited'] += len(remasked)
                stats['target_positions'] += len(gold)
                stats['final_wrong_never_revisited'] += sum(a != b and i not in remasked
                     for i, (a, b) in enumerate(zip(final, gold)))
        assert digest.hexdigest() == audit['traces_sha256']
    report = dict(status='passed', round=args.round, by_policy_and_mode=result,
        interpretation='Repeated masked inputs give identical model input tensors because visible context, '
                       'positions and features are fixed and this checkpoint has no timestep argument. '
                       'The external EOS penalty can still vary by step; repeated input does not by itself '
                       'authorize dropping complete sampler steps without equivalence checks.',
        independent_audit_sha256=hashlib.sha256((folder/'independent_audit.json').read_bytes()).hexdigest())
    (folder/'trace_dynamics.json').write_text(json.dumps(report, indent=2)+'\n')
    for key, stats in result.items():
        if key.startswith('repair8_0.25:') or key.startswith('original8:'):
            print(key, dict(stats))


if __name__ == '__main__': main()
