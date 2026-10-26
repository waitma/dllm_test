"""Activate pllm; python -B FILE --out ABS to freeze the four/eight-forward follow-up.

Reuse the completed ablation rows without consuming unused reserve components.
Freeze exactly one additional budget, never selecting a policy from its scores.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
PARENT = ROOT / 'output/downstream_generation/cdr_block_ablation_20261016'
PARENT_DELIVERY_SHA = '0c0531a339a1fdfa1560c0997d5c16811e83b1225cfefe31982653a803fd891a'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    assert not (out / 'plan.json').exists()
    assert sha(PARENT / 'ablation_delivery.json') == PARENT_DELIVERY_SHA
    delivery = read(PARENT / 'ablation_delivery.json')
    assert delivery['status'] == 'complete' and not delivery['winner_selected']
    evidence = {**delivery['evidence_sha256'], str(PARENT / 'ablation_delivery.json'): PARENT_DELIVERY_SHA}
    for path, digest in evidence.items():
        assert sha(path) == digest, path
    previous = read(PARENT / 'plan.json')
    assert sha(PARENT / 'plan.json') == '60b4ea16fe684c03098b5bd8c283035dfd3835a4cc915dfbfb0a7197698cb22e'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'rows.json').exists()
    shutil.copy2(PARENT / 'rows.json', out / 'rows.json')
    rows = read(out / 'rows.json')
    counts = dict(Counter(row['mode'] for row in rows))
    assert counts == {'cdrh1': 526, 'cdrh2': 526, 'cdrh3': 4202}
    group_stats = {}
    for mode in sorted(counts):
        subset = [row for row in rows if row['mode'] == mode]
        lengths = [len(row['mask_h']) for row in subset]
        group_stats[mode] = dict(n=len(subset), components=len({row['group'] for row in subset}),
            length_below_seven=sum(length < 7 for length in lengths),
            length_at_most_seven=sum(length <= 7 for length in lengths),
            effective_group_count=dict(sorted(Counter(min(length, 7) for length in lengths).items())),
            empty_group_sample_forwards=sum(max(7 - length, 0) for length in lengths))
    data = dict(parent=str(PARENT), rows_sha256=sha(out / 'rows.json'),
        rows_identical_to_parent=True, counts_by_mode=counts, group_stats=group_stats,
        sample_use='Same previously scored ablation rows and order; this fixed follow-up is not independent unseen confirmation. No unused reserve rows consumed.',
        parent_data_audit_sha256=sha(PARENT / 'data_independent_audit.json'))
    write(out / 'data_plan.json', data)
    sources = {}
    (out / 'source').mkdir(exist_ok=False)
    for relative in ('downstream/run_cdr_block_budget.py', 'downstream/audit_cdr_block_budget.py',
                     'downstream/freeze_cdr_block_budget.py', 'tests/test_cdr_block_budget.py'):
        original = ROOT / 'scripts' / relative
        destination = out / 'source' / original.name
        shutil.copy2(original, destination)
        sources[str(destination)] = sha(destination)
    external = {**previous['source_sha256'], **previous['external_source_sha256']}
    common = dict(rule='interleaved', temperature=.5, strength=.5, context='soft', feedback='frozen', prior_weight=.5)
    plan = dict(schema='cdr.block.budget.protocol.v1', created_utc=datetime.now(timezone.utc).isoformat(),
        parent=str(PARENT), block_parent=previous['parent'], marginal_parent=previous['marginal_parent'],
        checkpoints=previous['checkpoints'], batch_size=24, precision='fp32', sampling='argmax',
        families=[dict(name='grouped4', groups=3, **common), dict(name='grouped8', groups=7, **common)],
        policies=['single_pass', 'grouped4', 'grouped8'], reused_controls=['original4', 'original8'],
        counts_by_mode=counts, source_sha256=sources, external_source_sha256=external,
        parent_evidence_sha256=evidence, rows_sha256=sha(out / 'rows.json'),
        parent_rows_sha256=sha(PARENT / 'rows.json'), parent_plan_sha256=sha(PARENT / 'plan.json'),
        parent_audit_sha256={cp: sha(PARENT / f'evaluation_{cp}/independent_audit.json') for cp in previous['checkpoints']},
        data_plan_sha256=sha(out / 'data_plan.json'),
        bootstrap=dict(repeats=50000, seed=20261016, unit='connected_component', family_size=2),
        primary_contrast='grouped8_vs_grouped4', primary_mode='cdrh3',
        inferential_rule='Two prespecified follow-up H3 contrasts, one per checkpoint: grouped8 minus grouped4. Report paired component 95% and Bonferroni 97.5% percentile intervals (1.25/98.75 percentiles). Both adjusted lower bounds must exceed zero for cross-checkpoint improvement; crossing zero is inconclusive, not equivalence.',
        secondary_rule='H1/H2 and grouped8 versus single_pass/original4/original8 are descriptive 95% intervals. No winner selection or tuning. Local family correction does not control all historical adaptive exploration.',
        budget_definition='One initial all-target MASK prediction plus three or seven independent group queries from the same frozen p0. Output is 0.5*p0 + 0.5*group posterior, followed by argmax. Empty group queries still count toward logical NFE.',
        cost='Run prior plus 3 plus 7 queries = 11 sample-forwards per row. Original4/8 controls are reused from the completed parent run, not executed again. Gates separate.',
        sample_use=data['sample_use'],
        limitations=['Changing modulo partitions from three to seven changes both masking geometry and soft-context coverage; these partitions are not nested.',
            'This is a grouped-query budget comparison, not repeated iterative refinement or a test of monotonic quality at all step counts.',
            'H3 has 16 lengths below seven and H2 has one; 23 empty-group sample-forwards per checkpoint remain charged.',
            'Eight better than four does not establish superiority to single pass.',
            'OAS supplement and inherited exclusion boundaries do not establish universal pretraining decontamination.'])
    write(out / 'plan.json', plan)
    print(json.dumps(dict(status='frozen', plan_sha256=sha(out / 'plan.json'), rows=len(rows), data=group_stats)))


if __name__ == '__main__':
    main()
