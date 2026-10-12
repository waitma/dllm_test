"""Audit exact artificial heldout input collisions on CPU, without any model.

Activate pllm, then run USE_TF=0 python scripts/audit_immune_expanded_identifiability.py
  --reference-summary /absolute/run/summary.json --output /absolute/audit.json
The actual fixture, per-stratum collator and four masks are used. No GPU, model,
checkpoint, queue action or biological sequence corpus is accessed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def calculate(entries):
    groups = defaultdict(list)
    for entry in entries:
        groups[entry['input_sha256']].append(entry)
    result = {}
    joint_end_min = joint_end_max = 0
    for category in ('selected', 'aa', 'terminal'):
        correct = total = 0
        entropy = 0.
        for members in groups.values():
            positions = sorted({position for entry in members for position in entry[category]})
            for position in positions:
                counts = Counter(entry['labels'][position] for entry in members if position in entry[category])
                count = sum(counts.values())
                maximum = max(counts.values())
                total += count
                correct += maximum
                entropy += sum(-value*math.log(value/count) for value in counts.values())
                if category == 'selected':
                    end_counts = Counter(entry['labels'][position] for entry in members if position in entry['terminal'])
                    choices = [label for label, value in counts.items() if value == maximum]
                    joint_end_min += min(end_counts[label] for label in choices)
                    joint_end_max += max(end_counts[label] for label in choices)
        result[category] = dict(maximum_correct=correct, count=total,
            accuracy_upper_bound=correct/total if total else None,
            empirical_token_ce_lower_bound=entropy/total if total else None)
    terminal_count = result['terminal']['count']
    result['joint_optimal_terminal_correct_range'] = [joint_end_min, joint_end_max]
    result['joint_optimal_terminal_accuracy_range'] = ([joint_end_min/terminal_count, joint_end_max/terminal_count]
                                                       if terminal_count else [None, None])
    result['input_groups'] = [[entry['fixture_id'] for entry in members] for members in groups.values()]
    result['input_group_count'] = len(groups)
    result['records'] = len(entries)
    return result


def first_terminal_bound(entries):
    """Exact first-END identifiability; predicts one position per input/chain."""
    groups = defaultdict(list)
    for entry in entries:
        groups[entry['input_sha256']].append(entry)
    maximum = count = 0
    evidence = []
    for input_hash, members in sorted(groups.items()):
        chain_ids = set(members[0]['chain_terminal_positions'])
        if any(set(entry['chain_terminal_positions']) != chain_ids for entry in members):
            raise ValueError('Identical full-mask inputs must have identical target chain slots')
        positions = {}
        for chain in sorted(chain_ids):
            distribution = Counter(entry['chain_terminal_positions'][chain] for entry in members)
            if any(not isinstance(position, int) or position < 0 for position in distribution):
                raise ValueError('Terminal positions must be nonnegative chain-local slot indices')
            maximum += max(distribution.values())
            count += sum(distribution.values())
            positions[chain] = dict(sorted(distribution.items()))
        evidence.append(dict(input_sha256=input_hash,
            fixture_ids=[entry['fixture_id'] for entry in members],
            terminal_position_counts=positions))
    return dict(maximum_correct=maximum, count=count,
        accuracy_upper_bound=maximum/count if count else None,
        position_convention='zero-based ordinal among attended target slots within a chain; true residue length',
        input_groups=evidence)


def audit(reference):
    from scripts import run_immune_refined as runner
    from dllm.pipelines.immune_llada.v3.model import ImmuneModel
    fixtures = runner.build_fixtures(runner.ImmuneConfig(ablation='A4'))
    assert reference['fixture_manifest'] == fixtures.manifest, 'Fixture protocol differs from recorded run'
    assert reference['source_sha256'] == runner.fingerprints(), 'Source differs from recorded run'
    collate = runner.base.collator_for(runner.ImmuneConfig(ablation='A4'))
    strata = defaultdict(list)
    for row in fixtures.valid_records:
        strata[row.metadata['diagnostic_group']+'/'+row.metadata['length_bin']].append(row)
    entries = {mask: [] for mask, _ in runner.previous.MASK_LEVELS}
    for name, rows in sorted(strata.items()):
        # Production evaluation uses at most four records in each same-stratum batch.
        for start in range(0, len(rows), 4):
            selected_rows = rows[start:start+4]
            batch = collate(selected_rows)
            for mask, ratio in runner.previous.MASK_LEVELS:
                selected = (runner.base.fixed_validation_mask(batch, ratio).bool()
                            & batch['eligible_mask'].bool() & batch['attention_mask'].bool())
                features = ImmuneModel.feature_args(batch)
                features['input_ids'] = batch['input_ids'].masked_fill(selected, collate.tokenizer.mask_token_id)
                for index, row in enumerate(selected_rows):
                    encoded = json.dumps({key: value[index].tolist() for key, value in features.items()}, sort_keys=True).encode()
                    aa = selected[index] & batch['residue_mask'][index].bool()
                    terminal = selected[index] & batch['terminal_mask'][index].bool()
                    assert int(selected[index].sum()) == int(aa.sum()+terminal.sum())
                    target_slots = batch['target_slot_mask'][index].bool() & batch['attention_mask'][index].bool()
                    chain_terminal_positions = {}
                    for chain in batch['chain_ids'][index][target_slots].unique().tolist():
                        positions = (target_slots & (batch['chain_ids'][index] == chain)).nonzero().flatten()
                        ends = batch['terminal_mask'][index][positions].nonzero().flatten().tolist()
                        assert len(ends) == 1, 'Expected exactly one true terminal per target chain'
                        chain_terminal_positions[int(chain)] = ends[0]
                    entries[mask].append(dict(input_sha256=hashlib.sha256(encoded).hexdigest(),
                        fixture_id=row.metadata['fixture_id'], stratum=name,
                        subset='unconditional' if name.startswith('unconditional_') else 'conditional',
                        labels=batch['labels'][index].tolist(),
                        selected=set(selected[index].nonzero().flatten().tolist()),
                        aa=set(aa.nonzero().flatten().tolist()), terminal=set(terminal.nonzero().flatten().tolist()),
                        chain_terminal_positions=chain_terminal_positions))
    groups = {}
    for subset in ('all', 'conditional', 'unconditional'):
        levels = {mask: calculate([entry for entry in values if subset == 'all' or entry['subset'] == subset])
                  for mask, values in entries.items()}
        groups[subset] = dict(levels=levels,
            mask_mean_aa_upper_bound=sum(level['aa']['accuracy_upper_bound'] for level in levels.values())/len(levels),
            fullmask_first_terminal=first_terminal_bound([entry for entry in entries['mask100']
                if subset == 'all' or entry['subset'] == subset]))
    for mask, values in groups['all']['levels'].items():
        for category, expected in [('selected', 'selected_count'), ('aa', 'aa_count'), ('terminal', 'terminal_count')]:
            assert values[category]['count'] == reference['initial']['mask_levels'][mask][expected], (mask, expected)
    return dict(time=time.time(), protocol='exact_production_heldout_per_stratum_batches_four',
        biological_quality_claim=False, feature_names=list(features), groups=groups,
        fixture_manifest_sha256=fixtures.manifest['manifest_sha256'], source_sha256=runner.fingerprints(),
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Input-deterministic argmax upper bounds on this finite artificial heldout set only. Bounds do not establish learnability or generalization. AA-only, terminal-only and joint optima differ; joint-optimal terminal range explicitly accounts for modal-label ties.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    reference = json.loads(args.reference_summary.read_text())
    report = audit(reference)
    report['reference_summary'] = str(args.reference_summary.resolve())
    report['reference_summary_sha256'] = hashlib.sha256(args.reference_summary.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps({subset: dict(fullmask_group_count=value['levels']['mask100']['input_group_count'],
        fullmask_aa_upper_bound=value['levels']['mask100']['aa']['accuracy_upper_bound'],
        fullmask_joint_upper_bound=value['levels']['mask100']['selected']['accuracy_upper_bound'],
        joint_optimal_terminal_range=value['levels']['mask100']['joint_optimal_terminal_accuracy_range'],
        mask_mean_aa_upper_bound=value['mask_mean_aa_upper_bound']) for subset, value in report['groups'].items()}))


if __name__ == '__main__':
    main()
