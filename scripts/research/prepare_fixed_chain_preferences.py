"""Build weak-label preference pairs from the frozen training subset only.

Activate pllm; python -m scripts.research.prepare_fixed_chain_preferences
No confirmation or benchmark files are opened; outputs never overwrite source rows.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/fixed_chain_exploration_20261014'
ROLES = {'antibody_heavy', 'antibody_light', 'tcr_beta', 'tcr_alpha'}


def signature(row):
    """Match context, chain presence, completion type and coarse observed lengths."""
    context = sorted((role, seq) for role, seq in zip(row['chain_roles'], row['chains']) if role not in ROLES)
    receptors = []
    observed = []
    for index, (role, seq) in enumerate(zip(row['chain_roles'], row['chains'])):
        if role not in ROLES:
            continue
        synthetic = row.get('chain_metadata', {}).get(str(index), {}).get('synthetic_residue_mask', [0] * len(seq))
        residues = ''.join(aa for aa, flag in zip(seq, synthetic) if not flag)
        receptors.append((role, bool(residues), any(synthetic), len(residues) // 16))
        observed.append((role, residues))
    key = json.dumps([context, sorted(receptors)], separators=(',', ':'))
    identity = json.dumps([context, sorted(observed)], separators=(',', ':'))
    return key, identity, bool(context)


def binding_condition(row):
    result = copy.deepcopy(row)
    result.setdefault('labels', {})['relation'] = 'binding'
    result.setdefault('metadata', {})['relation_supervised'] = True
    return result


def main():
    output = RUN / 'preference_data'
    output.mkdir(exist_ok=False)
    manifest = json.loads((RUN / 'data/manifest.json').read_text())
    train = Path(manifest['files']['train']['path'])
    raw = train.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['files']['train']['sha256']:
        raise ValueError('Frozen training rows changed')
    rows = [json.loads(line) for line in raw.splitlines()]
    groups, labels = defaultdict(lambda: defaultdict(list)), defaultdict(set)
    cursor = 0
    for key, count in manifest['source_counts'].items():
        if not key.startswith('train/'):
            continue
        source = key.split('/', 1)[1]
        for row in rows[cursor:cursor + count]:
            label = (row.get('labels') or {}).get('relation')
            if label not in ('binding', 'nonbinding'):
                continue
            group, identity, conditioned = signature(row)
            if not conditioned:
                continue
            labels[identity].add(label)
            groups[(source, group)][label].append((identity, row))
        cursor += count
    assert cursor == len(rows)
    conflicts = {key for key, values in labels.items() if len(values) > 1}
    rng, pairs, counts = random.Random(20261014), [], Counter()
    for (source, key), data in sorted(groups.items()):
        positive = {i: row for i, row in data['binding'] if i not in conflicts}
        negative = {i: row for i, row in data['nonbinding'] if i not in conflicts}
        if not positive or not negative:
            continue
        left, right = list(positive.items()), list(negative.items())
        rng.shuffle(left)
        rng.shuffle(right)
        counts[source + '/groups'] += 1
        for i in range(min(32, max(len(left), len(right)))):
            chosen_id, chosen = left[i % len(left)]
            rejected_id, rejected = right[i % len(right)]
            assert chosen_id != rejected_id
            pairs.append({'source': source, 'group_sha256': hashlib.sha256(key.encode()).hexdigest(),
                          'chosen': binding_condition(chosen), 'rejected': binding_condition(rejected),
                          'label_provenance': 'original dataset binding versus nonbinding; not certified measured affinity'})
            counts[source + '/pairs'] += 1
    if len(pairs) < 128:
        raise ValueError('Insufficient matched pairs')
    path = output / 'train_pairs.jsonl'
    path.write_text(''.join(json.dumps(p, separators=(',', ':')) + '\n' for p in pairs))
    report = {'schema': 'fixed_chain.weak_preference.v1', 'status': 'prepared',
              'source_train_sha256': manifest['files']['train']['sha256'], 'pairs': len(pairs),
              'counts': dict(counts), 'conflicting_observed_identities_removed': len(conflicts),
              'file': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
              'scope': 'training-only weak dataset preferences; same condition/roles/completion type/length bins',
              'benchmark_or_confirmation_read': False, 'generation_feature_equivalence': 'pending_preflight'}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
