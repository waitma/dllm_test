"""Activate pllm; python -B scripts/downstream/prepare_cdr_block_ablation.py --out ABS.

Freeze a score-independent ablation cohort from the previous OAS reserve only.
Use all H3 rows in the first 4096 reserve components and H1/H2 rows in the first
512 of those components. Then independently verify selection, original source
coordinates, and separation from previous confirmation rows. No model is run.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
PARENT = ROOT / 'output/downstream_generation/cdr_block_context_20261015'
PRIMARY_GROUPS = 4096
SECONDARY_GROUPS = 512


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def normalized(seq):
    return seq.replace('J', 'L')


def independent_audit(out, parent):
    """Reopen saved rows; reconstruct expected IDs separately from row generation."""
    plan, rows = read(out / 'data_plan.json'), read(out / 'rows.json')
    previous = read(parent / 'rows.json')
    old_plan, old_audit = read(parent / 'data_plan.json'), read(parent / 'data_independent_audit.json')
    assert sha(out / 'rows.json') == plan['rows_sha256']
    assert sha(out / 'prepare_source.py') == plan['source_sha256']
    assert old_audit['status'] == 'passed_scoped_checks'
    assert old_audit['rows_sha256'] == old_plan['rows_sha256'] == sha(parent / 'rows.json')
    assert old_audit['data_plan_sha256'] == sha(parent / 'data_plan.json')
    for path, digest in plan['parent_artifact_sha256'].items():
        assert sha(path) == digest, path
    binding = old_audit['training_hash_binding']
    assert binding['all_scan_hashes_match_original_checkpoint_training_audit']
    assert sha(binding['audit_path']) == binding['audit_sha256']
    full = read(binding['audit_path'])
    full_hashes = {entry['path']: entry['sha256'] for entry in full['shard_results']}
    assert all(full_hashes[path] == digest for path, digest in old_plan['train_shard_sha256'].items())

    all_by_id = {row['id']: row for row in previous}
    reserve_by_group = defaultdict(list)
    for row in previous:
        if row['split'] == 'reserve':
            reserve_by_group[row['group']].append(row)
    ordered_groups = sorted(reserve_by_group)
    expected_ids = set()
    for rank, group in enumerate(ordered_groups):
        for row in reserve_by_group[group]:
            if (row['mode'] == 'cdrh3' and rank < PRIMARY_GROUPS) or (
                    row['mode'] in ('cdrh1', 'cdrh2') and rank < SECONDARY_GROUPS):
                expected_ids.add(row['id'])
    assert len(rows) == len({row['id'] for row in rows})
    assert {row['id'] for row in rows} == expected_ids
    assert dict(Counter(row['mode'] for row in rows)) == plan['counts_by_mode']
    for row in rows:
        source_row = all_by_id[row['id']]
        assert source_row['split'] == 'reserve'
        assert row['split'] == 'ablation' and row['parent_split'] == 'reserve'
        restored = {key: value for key, value in row.items() if key != 'parent_split'}
        restored['split'] = 'reserve'
        assert restored == source_row

    source_records = {}
    for path in sorted({row['source'] for row in rows}):
        assert sha(path) == old_plan['source_data_sha256'][path]
        source_records[path] = [json.loads(line) for line in Path(path).open()]
    selected_h3 = {}
    for row in rows:
        raw = source_records[row['source']][row['row']]
        assert raw['chain_roles'] == ['antibody_heavy', 'antibody_light']
        heavy, light = map(normalized, raw['chains'])
        assert heavy == row['heavy'] == row['gold_h']
        assert light == row['light'] == row['gold_l']
        regions = raw['regions']['0']
        assert normalized(''.join(regions[k] for k in ('FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4'))) == heavy
        labels = raw['chain_metadata']['0']['h_region_labels']
        assert len(labels) == len(heavy)
        assert row['mask_h'] == [i for i, label in enumerate(labels) if label == row['mode'][-1]]
        assert ''.join(heavy[i] for i in row['mask_h']) == normalized(regions['CDR' + row['mode'][-1]])
        assert not row['mask_l'] and row['h_cluster'] == raw['chain_metadata']['0']['H_cluster_id']
        assert row['key'] == hashlib.sha256((heavy + '|' + light).encode()).hexdigest()
        selected_h3[row['key']] = normalized(regions['CDR3'])

    def key_sets(values):
        h3_rows = [row for row in values if row['mode'] == 'cdrh3']
        return dict(group={row['group'] for row in h3_rows}, heavy={row['gold_h'] for row in h3_rows},
                    h3={''.join(row['gold_h'][i] for i in row['mask_h']) for row in h3_rows},
                    h_cluster={row['h_cluster'] for row in h3_rows})
    selected_sets = key_sets(rows)
    old_confirm_sets = key_sets([row for row in previous if row['split'] == 'confirm'])
    unused_sets = key_sets([row for row in previous if row['split'] == 'reserve'
                            and row['group'] not in selected_sets['group']])
    cross_old = {key: len(selected_sets[key] & old_confirm_sets[key]) for key in selected_sets}
    cross_unused = {key: len(selected_sets[key] & unused_sets[key]) for key in selected_sets}
    assert not any(cross_old.values()) and not any(cross_unused.values())
    assert set(selected_h3.values()) == selected_sets['h3']
    pair_modes = defaultdict(set)
    for row in rows:
        pair_modes[row['key']].add(row['mode'])
    all_cdr_pairs = sum(modes == {'cdrh1', 'cdrh2', 'cdrh3'} for modes in pair_modes.values())
    h3_only_pairs = sum(modes == {'cdrh3'} for modes in pair_modes.values())
    assert all_cdr_pairs + h3_only_pairs == len(pair_modes)
    assert all_cdr_pairs == plan['allocation']['all_three_cdr']['pairs']
    assert h3_only_pairs == plan['allocation']['h3_only']['pairs']
    assert len(unused_sets['group']) == plan['allocation']['wholly_unallocated_reserve']['groups']
    unused_pairs = {row['key'] for row in previous if row['split'] == 'reserve'
                    and row['group'] in unused_sets['group']}
    assert len(unused_pairs) == plan['allocation']['wholly_unallocated_reserve']['pairs']
    return dict(status='passed', created_utc=datetime.now(timezone.utc).isoformat(),
        rows_sha256=sha(out / 'rows.json'), data_plan_sha256=sha(out / 'data_plan.json'),
        preparation_source_sha256=sha(out / 'prepare_source.py'),
        rows=len(rows), counts_by_mode=dict(Counter(row['mode'] for row in rows)),
        selected_pairs=len(pair_modes), selected_groups=len(selected_sets['group']),
        all_three_cdr_pairs=all_cdr_pairs, h3_only_pairs=h3_only_pairs,
        wholly_unallocated_reserve_pairs=len(unused_pairs),
        wholly_unallocated_reserve_groups=len(unused_sets['group']),
        score_independent_group_order_reconstructed=True,
        every_row_parent_split_reserve=True, unchanged_original_fields_except_split=True,
        every_source_coordinate_and_sequence_recovered=True,
        overlap_with_previous_confirm=cross_old, overlap_with_wholly_unallocated_reserve=cross_unused,
        inherited_training_hash_binding=binding,
        train_records_rescanned=False, scores_read=False,
        limitations=plan['limitations'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ('data_plan.json', 'rows.json', 'data_independent_audit.json', 'prepare_source.py'):
        assert not (out / name).exists(), 'Refusing to overwrite ' + str(out / name)
    old_plan = read(PARENT / 'data_plan.json')
    old_audit = read(PARENT / 'data_independent_audit.json')
    assert old_audit['status'] == 'passed_scoped_checks'
    assert sha(PARENT / 'rows.json') == old_plan['rows_sha256'] == old_audit['rows_sha256']
    assert sha(PARENT / 'data_plan.json') == old_audit['data_plan_sha256']
    prior_rows = read(PARENT / 'rows.json')
    reserve = [row for row in prior_rows if row['split'] == 'reserve']
    group_order = sorted({row['group'] for row in reserve})
    assert len(group_order) > PRIMARY_GROUPS > SECONDARY_GROUPS
    primary, secondary = set(group_order[:PRIMARY_GROUPS]), set(group_order[:SECONDARY_GROUPS])
    selected = [dict(row, split='ablation', parent_split='reserve') for row in reserve
                if row['group'] in primary and (row['mode'] == 'cdrh3' or row['group'] in secondary)]
    pairs = lambda groups: len({row['key'] for row in reserve if row['group'] in groups})
    unallocated = set(group_order) - primary
    shutil.copyfile(Path(__file__), out / 'prepare_source.py')
    write(out / 'rows.json', selected)
    limitations = [
        'No scores are read and no parameters are selected by this data preparation; all cohort allocation is fixed by previously frozen component hashes.',
        'This allocation is frozen before evaluation; it does not assert that scoring has already occurred.',
        'Pairs allocated to H3 alone are not a wholly untouched reserve after their H3 evaluation, even when their H1/H2 outputs remain unscored.',
        'Training and prior checkpoint-selection exclusions inherit the previous preparation and scoped independent audit. Training records are not rescanned in this derivation.',
        'Exact heavy/H3 and supplied cluster exclusions do not establish universal homology or wider pretraining decontamination.',
        'This is an OAS supplementary ablation cohort, not the official SAbDab benchmark.'
    ]
    info = dict(schema='cdr.block.ablation.data.v1', created_utc=datetime.now(timezone.utc).isoformat(),
        parent=str(PARENT), parent_artifact_sha256={str(PARENT / name): sha(PARENT / name)
            for name in ('rows.json', 'data_plan.json', 'data_independent_audit.json')},
        checkpoints=old_plan['checkpoints'], source_sha256=sha(out / 'prepare_source.py'),
        rows_sha256=sha(out / 'rows.json'), split='ablation', parent_split='reserve',
        counts_by_mode=dict(Counter(row['mode'] for row in selected)),
        selection_rule='Lexicographically sort existing reserve group SHA256 strings; first 4096 complete components for H3; first 512 of those complete components for H1/H2. No score-dependent selection.',
        selected_groups=group_order[:PRIMARY_GROUPS], secondary_groups=group_order[:SECONDARY_GROUPS],
        allocation={
            'all_three_cdr': dict(groups=SECONDARY_GROUPS, pairs=pairs(secondary), modes=['cdrh1', 'cdrh2', 'cdrh3']),
            'h3_only': dict(groups=PRIMARY_GROUPS-SECONDARY_GROUPS, pairs=pairs(primary-secondary),
                           modes=['cdrh3'], h1_h2_state='Not selected for scoring; pairs are allocated to H3 and must not be called wholly untouched.'),
            'wholly_unallocated_reserve': dict(groups=len(unallocated), pairs=pairs(unallocated),
                                             modes=['cdrh1', 'cdrh2', 'cdrh3']),
            'previous_confirm_excluded': dict(groups=old_audit['confirm_groups'], pairs=old_plan['counts']['confirm:cdrh3'])},
        source_data_sha256={path: old_plan['source_data_sha256'][path] for path in sorted({row['source'] for row in selected})},
        inherited_train_shard_sha256=old_plan['train_shard_sha256'],
        original_parent_unchanged=True, scores_read=False, limitations=limitations)
    write(out / 'data_plan.json', info)
    report = independent_audit(out, PARENT)
    write(out / 'data_independent_audit.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'counts_by_mode', 'selected_pairs',
        'selected_groups', 'h3_only_pairs', 'wholly_unallocated_reserve_pairs',
        'wholly_unallocated_reserve_groups', 'rows_sha256', 'data_plan_sha256')}, indent=2))


if __name__ == '__main__':
    main()
