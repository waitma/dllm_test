"""CPU-only exporter integrity tests using invented temporary JSON records.

Activate pllm, then:
  USE_TF=0 python -m pytest scripts/tests/immune_llada/test_expanded_clean_export.py -q
No real prepared corpora or biological annotation evidence is modified.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.data import export_immune_expanded_clean as export


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, value, blanks=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n' if blanks else '') + ''.join(json.dumps(row) + ('\n\n' if blanks else '\n') for row in value))


def make_row(source, split, family, number):
    return dict(schema_version='immune_llada.semantic.v1', chains=['AAAA', 'CAAAF', 'CQQQW', 'PEPT'],
        chain_roles=['mhc', 'tcr_alpha', 'tcr_beta', 'peptide'], task_type='tcr_pmhc', source='invented/' + source,
        split=split, weight=1., labels={'relation': 'paired'}, identifiers={'id': f'{source}/{split}/{number}'},
        metadata={'v7_family_group': family, 'other_provenance': {'preserve': [number]}},
        chain_metadata={'0': {'sequence_scope': 'full'}, '1': {'sequence_scope': 'full'},
                        '2': {'sequence_scope': 'full'}, '3': {'sequence_scope': 'full'}},
        regions={'1': {'cdr3': 'CAAAF'}}, extension_field={'not_in_standard_record': number})


@pytest.fixture
def fixture(tmp_path):
    parent, mhc, auditdir = tmp_path/'parent', tmp_path/'mhc', tmp_path/'audit'
    report_path = auditdir/'parent'/'report.json'
    plan_path, hits_path = auditdir/'cleanup_plan.json', auditdir/'parent'/'row_hits.jsonl'
    cohorts = ['invented/support', 'invented/test']
    registry = dict(scope_version=export.SCOPE, inputs=[], script_sha256=export.sha(export.ROOT/'scripts/data/audit_tcr_exposure.py'))
    for i, cohort in enumerate(cohorts):
        path = tmp_path/f'benchmark{i}.txt'
        path.write_text('invented generic rows only')
        registry['inputs'].append(dict(path=str(path), sha256=export.sha(path), cohort=cohort))
    write(auditdir/'banks.json', {'mock_only': True})
    registry['banks_sha256'] = export.sha(auditdir/'banks.json')
    write(auditdir/'registry.json', registry)
    reference, full, original, updated = (tmp_path/name for name in ('reference.txt', 'full.jsonl', 'input.csv', 'updated.csv'))
    for path in (reference, full, original, updated):
        path.write_text('invented provenance bytes')
    source_report = dict(policy='sequence_first_mhc_v1',
        reference=dict(path=str(reference), sha256=export.sha(reference)), full_source=dict(path=str(full), sha256=export.sha(full)),
        files={'mock': dict(input=str(original), input_sha256=export.sha(original), output=str(updated), output_sha256=export.sha(updated))})
    source_report_path = tmp_path/'source_report.json'
    write(source_report_path, source_report)
    spec = []
    hits = []
    for split in ('train', 'valid'):
        for source in ('trait', 'tcr_native'):
            relative = f'{split}/{source}-00000.jsonl'
            families = ['a_shared', source+'_a', source+'_b', source+'_removed'] if split == 'train' else [source+'_valid', source+'_valid_removed']
            old_rows = [make_row(source, split, family, i) for i, family in enumerate(families)]
            new_rows = deepcopy(old_rows)
            for row in new_rows:
                row['chains'][0] = 'QQQQ'
                row['chain_metadata']['0'].update(mhc_policy='sequence_first_mhc_v1', mhc_representation='provided')
                row['metadata']['mhc_update'] = {'invented': True}
            write_rows(parent/relative, old_rows, blanks=True)
            write_rows(mhc/relative, new_rows, blanks=True)
            spec.append(dict(path=relative, source=source, split=split, records=len(old_rows)))
            reasons = {'edit80_cdr3a': 1}
            if source == 'trait':
                reasons['exact_cdr3b'] = 2
            hits.append(dict(shard=relative, row=len(old_rows)-1, reasons=reasons))
    write_rows(hits_path, hits)
    f = SimpleNamespace(parent=parent, mhc=mhc, plan_path=plan_path, hits_path=hits_path,
        report_path=report_path, registry_path=auditdir/'registry.json', source_report_path=source_report_path,
        spec=spec, cohorts=cohorts, tmp=tmp_path)

    def rebind():
        blocks = {s: dict(records=0, shards=[]) for s in ('train', 'valid')}
        for item in f.spec:
            shard = dict(item, records=len(rows(parent/item['path'])), sha256=export.sha(parent/item['path']))
            blocks[item['split']]['shards'].append(shard)
            blocks[item['split']]['records'] += shard['records']
        pm = dict(schema_version='immune_llada.semantic.v1', format='jsonl', dataset_version='mock_parent',
            budget={'unchanged': 7}, sources=['trait', 'tcr_native'], splits=blocks)
        write(parent/'dataset_manifest.json', pm)
        mm = deepcopy(pm)
        mm['dataset_version'] = 'mock_mhc'
        for block in mm['splits'].values():
            for shard in block['shards']:
                shard['parent_sha256'] = shard['sha256']
                shard['sha256'] = export.sha(mhc/shard['path'])
        source = export.read_json(source_report_path)
        mm['mhc_update'] = dict(policy='sequence_first_mhc_v1', parent=str(parent),
            parent_manifest_sha256=export.sha(parent/'dataset_manifest.json'), source_report=str(source_report_path),
            source_report_sha256=export.sha(source_report_path), reference_sha256=source['reference']['sha256'])
        write(mhc/'dataset_manifest.json', mm)
        totals = {}
        current_hits = rows(hits_path)
        for item in f.spec:
            key = item['split'] + '/' + item['source']
            totals[key+'/rows'] = len(rows(parent/item['path']))
            totals[key+'/flagged_rows'] = sum(hit.get('shard') == item['path'] for hit in current_hits)
        audit = dict(scope_version=export.SCOPE, status='overlap_found',
            dataset_manifest_sha256=export.sha(parent/'dataset_manifest.json'), registry_sha256=export.sha(f.registry_path),
            row_hits_sha256=export.sha(hits_path), totals=totals, cohorts=cohorts,
            shards=[dict(path=str((parent/item['path']).resolve()), rows=len(rows(parent/item['path'])),
                         sha256=export.sha(parent/item['path'])) for item in f.spec])
        write(report_path, audit)
        plan = dict(status='not_applied', row_indexing=export.INDEXING, source_dataset=str(parent),
            source_manifest_sha256=export.sha(parent/'dataset_manifest.json'), audit_report=str(report_path),
            audit_report_sha256=export.sha(report_path), row_hits=str(hits_path), row_hits_sha256=export.sha(hits_path),
            cohort_bit_order=cohorts,
            rule='union of exact core/FV/pair, edit80 core/peptide and exact epitope-in-antigen reasons; preserve official benchmark rows')
        write(plan_path, plan)
    f.rebind = rebind
    rebind()
    f.load = lambda: export.load_evidence(mhc, plan_path, mhc_manifest_sha256=export.sha(mhc/'dataset_manifest.json'),
        cleanup_plan_sha256=export.sha(plan_path))
    return f


def test_plan_does_not_read_large_bound_inputs(fixture, monkeypatch):
    real_sha = export.sha
    large = {fixture.hits_path.resolve(), (fixture.tmp/'full.jsonl').resolve()}
    large.update((base/item['path']).resolve() for base in (fixture.parent, fixture.mhc) for item in fixture.spec)
    def guarded_sha(path):
        assert Path(path).resolve() not in large
        return real_sha(path)
    monkeypatch.setattr(export, 'sha', guarded_sha)
    report = export.plan_report(fixture.load(), min_valid_rows=4)
    assert report['status'] == 'estimated_not_row_verified'
    assert report['training_ready'] is False
    assert report['total_retained_rows'] == 8
    assert report['validation']['scarce_sources'] == {'trait': 1, 'tcr_native': 1}
    assert report['validation']['family_feasibility'] == 'requires_verified_row_scan'
    assert str(fixture.hits_path.resolve()) in report['pending_binding_paths']


def test_verified_dryrun_family_moves_are_whole_and_not_applied(fixture):
    before = {str(p): export.sha(p) for base in (fixture.parent, fixture.mhc) for p in base.rglob('*') if p.is_file()}
    report = export.verify_or_export(fixture.load(), min_valid_rows=2)
    assert report['status'] == 'verified_selection_audit_pending'
    assert report['pending_binding_paths'] == []
    validation = report['validation']
    assert validation['family_split_overlap'] == 0
    assert validation['split_reassignment_applied'] is False
    assert validation['feasibility'] == 'constructive_whole_family_example_found'
    assert validation['proposed_whole_family_moves'] == [dict(family_group='a_shared', source_rows={'tcr_native': 1, 'trait': 1}, total_rows=2)]
    assert {path: export.sha(path) for path in before} == before
    assert not list(fixture.tmp.glob('immune-expanded-families-*'))


def test_export_preserves_mhc_all_metadata_rows_splits_and_loader(fixture):
    from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
    output = fixture.tmp/'new_candidate'
    before = {str(p): export.sha(p) for base in (fixture.parent, fixture.mhc) for p in base.rglob('*') if p.is_file()}
    report = export.verify_or_export(fixture.load(), output_dir=output, min_valid_rows=2)
    manifest = export.read_json(output/'dataset_manifest.json')
    assert report['training_ready'] is manifest['training_ready'] is False
    assert manifest['decontamination']['status'] == 'audit_pending'
    assert manifest['decontamination']['pending']
    assert manifest['budget'] == {'unchanged': 7}
    assert manifest['mhc_update'] == fixture.load().manifest['mhc_update']
    assert not (output/'audit_certificate.json').exists()
    assert not (output/'downstream_scope_certificate.json').exists()
    mapping = rows(output/'row_mapping.jsonl')
    assert len(mapping) == 12
    for item in fixture.spec:
        selected = rows(fixture.mhc/item['path'])[:-1]
        assert rows(output/item['path']) == selected
        assert export.sha(output/item['path']) != export.sha(fixture.mhc/item['path'])
        records = [record for record in mapping if record['old']['shard'] == item['path']]
        assert [record['old']['row'] for record in records] == list(range(item['records']))
        assert [record['new']['row'] for record in records[:-1]] == list(range(item['records']-1))
        assert records[-1]['new'] is None and records[-1]['reasons']
        assert records[0]['family_group'] == selected[0]['metadata']['v7_family_group']
    assert len(PreparedImmuneDataset(output, split='train')) == 6
    assert len(PreparedImmuneDataset(output, split='valid', source='trait')) == 1
    restored = PreparedImmuneDataset(output, split='train')[0]
    assert restored.chains[0].sequence == 'QQQQ'
    assert restored.metadata['mhc_update'] == {'invented': True}
    assert {path: export.sha(path) for path in before} == before
    assert report['counts']['train/trait']['exact_reason_rows'] == 1
    assert report['counts']['train/tcr_native']['near_only_rows'] == 1
    assert report['row_mapping_sha256'] == export.sha(output/'row_mapping.jsonl')
    assert not list(fixture.tmp.glob('.*.staging-*'))


@pytest.mark.parametrize('target', ['mhc_manifest', 'plan', 'parent_manifest', 'audit', 'registry', 'source_report'])
def test_small_binding_tamper_rejected(fixture, target):
    pins = dict(mhc_manifest_sha256=export.sha(fixture.mhc/'dataset_manifest.json'), cleanup_plan_sha256=export.sha(fixture.plan_path))
    path = {'mhc_manifest': fixture.mhc/'dataset_manifest.json', 'plan': fixture.plan_path,
        'parent_manifest': fixture.parent/'dataset_manifest.json', 'audit': fixture.report_path,
        'registry': fixture.registry_path, 'source_report': fixture.source_report_path}[target]
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        export.load_evidence(fixture.mhc, fixture.plan_path, **pins)


@pytest.mark.parametrize('target', ['mhc_shard', 'parent_shard', 'hits', 'benchmark', 'reference', 'full', 'mhc_input', 'mhc_output', 'banks'])
def test_full_bound_content_tamper_rejected(fixture, target):
    evidence = fixture.load()
    path = {'mhc_shard': fixture.mhc/fixture.spec[0]['path'], 'parent_shard': fixture.parent/fixture.spec[0]['path'],
        'hits': fixture.hits_path, 'benchmark': fixture.tmp/'benchmark0.txt', 'reference': fixture.tmp/'reference.txt',
        'full': fixture.tmp/'full.jsonl', 'mhc_input': fixture.tmp/'input.csv', 'mhc_output': fixture.tmp/'updated.csv',
        'banks': fixture.registry_path.parent/'banks.json'}[target]
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        export.verify_or_export(evidence, output_dir=fixture.tmp/'rejected')
    assert not (fixture.tmp/'rejected').exists()
    assert not list(fixture.tmp.glob('.*.staging-*'))


@pytest.mark.parametrize('mutation', ['order', 'target', 'chain_metadata', 'regions', 'labels', 'metadata', 'extension', 'task'])
def test_non_mhc_change_or_row_order_rejected_even_when_hashes_match(fixture, mutation):
    path = fixture.mhc/fixture.spec[0]['path']
    changed = rows(path)
    if mutation == 'order':
        changed[0], changed[1] = changed[1], changed[0]
    elif mutation == 'target':
        changed[0]['chains'][1] = 'CVVVF'
    elif mutation == 'chain_metadata':
        changed[0]['chain_metadata']['1']['other'] = 'changed'
    elif mutation == 'regions':
        changed[0]['regions']['1']['cdr3'] = 'AAA'
    elif mutation == 'labels':
        changed[0]['labels']['relation'] = 'unpaired'
    elif mutation == 'metadata':
        changed[0]['metadata']['other_provenance'] = 'changed'
    elif mutation == 'extension':
        changed[0]['extension_field'] = 'changed'
    elif mutation == 'task':
        changed[0]['task_type'] = 'unconditional'
    write_rows(path, changed)
    fixture.rebind()
    with pytest.raises(ValueError, match='Non-MHC|task_type'):
        export.verify_or_export(fixture.load())


@pytest.mark.parametrize('mutation', ['duplicate', 'unsorted', 'unknown_shard', 'negative', 'too_large', 'bad_mask', 'bool_mask', 'unknown_reason', 'empty_reasons'])
def test_bad_hit_stream_rejected_and_staging_cleaned(fixture, mutation):
    hits = rows(fixture.hits_path)
    if mutation == 'duplicate':
        hits.insert(1, deepcopy(hits[0]))
    elif mutation == 'unsorted':
        hits[0], hits[1] = hits[1], hits[0]
    elif mutation == 'unknown_shard':
        hits[0]['shard'] = 'train/unknown.jsonl'
    elif mutation == 'negative':
        hits[0]['row'] = -1
    elif mutation == 'too_large':
        hits[0]['row'] = 999
    elif mutation == 'bad_mask':
        hits[0]['reasons'] = {'edit80_cdr3a': 4}
    elif mutation == 'bool_mask':
        hits[0]['reasons'] = {'edit80_cdr3a': True}
    elif mutation == 'unknown_reason':
        hits[0]['reasons'] = {'unknown_reason': 1}
    else:
        hits[0]['reasons'] = {}
    write_rows(fixture.hits_path, hits)
    fixture.rebind()
    with pytest.raises(ValueError):
        export.verify_or_export(fixture.load(), output_dir=fixture.tmp/'rejected')
    assert not (fixture.tmp/'rejected').exists()
    assert not list(fixture.tmp.glob('.*.staging-*'))


def test_family_cross_split_rejected_even_if_crossing_row_removed(fixture):
    relative = fixture.spec[-1]['path']
    for base in (fixture.parent, fixture.mhc):
        changed = rows(base/relative)
        changed[-1]['metadata']['v7_family_group'] = 'a_shared'
        write_rows(base/relative, changed)
    fixture.rebind()
    with pytest.raises(ValueError, match='family crosses train/valid'):
        export.verify_or_export(fixture.load())


def test_missing_family_rejected(fixture):
    relative = fixture.spec[0]['path']
    for base in (fixture.parent, fixture.mhc):
        changed = rows(base/relative)
        del changed[0]['metadata']['v7_family_group']
        write_rows(base/relative, changed)
    fixture.rebind()
    with pytest.raises(ValueError, match='Missing v7_family_group'):
        export.verify_or_export(fixture.load())


def test_insufficient_whole_family_report_does_not_move_rows_or_claim_impossible(fixture):
    report = export.verify_or_export(fixture.load(), min_valid_rows=100)
    assert report['validation']['feasibility'] == 'no_constructive_solution_not_a_proof_of_infeasibility'
    assert report['validation']['split_reassignment_applied'] is False
    assert all(value['train'] >= 1 for value in report['validation']['hypothetical_counts'].values())
    assert report['counts']['valid/trait']['retained_rows'] == 1


@pytest.mark.parametrize('destination', ['existing', 'original', 'within_original', 'ancestor', 'dangling_symlink'])
def test_output_cannot_overwrite_or_touch_originals(fixture, destination):
    path = {'existing': fixture.tmp/'exists', 'original': fixture.mhc, 'within_original': fixture.mhc/'new',
            'ancestor': fixture.tmp, 'dangling_symlink': fixture.tmp/'dangling'}[destination]
    if destination == 'existing':
        path.mkdir()
    elif destination == 'dangling_symlink':
        path.symlink_to(fixture.tmp/'missing')
    with pytest.raises(ValueError, match='already exists|separate'):
        export.verify_or_export(fixture.load(), output_dir=path)


def test_mhc_row_count_change_rejected_with_consistent_hashes(fixture):
    path = fixture.mhc/fixture.spec[0]['path']
    write_rows(path, rows(path)[:-1])
    fixture.rebind()
    with pytest.raises(ValueError, match='row membership'):
        export.verify_or_export(fixture.load())


def test_symlink_unchanged_shard_compatible(fixture):
    relative = fixture.spec[0]['path']
    (fixture.mhc/relative).unlink()
    (fixture.mhc/relative).symlink_to(fixture.parent/relative)
    fixture.rebind()
    report = export.verify_or_export(fixture.load())
    assert report['pending_binding_paths'] == []


def test_all_removed_shard_omitted_but_ledger_complete(fixture):
    hit_rows = rows(fixture.hits_path)
    relative = fixture.spec[0]['path']
    hit_rows = [dict(shard=relative, row=i, reasons={'edit80_peptide': 1}) for i in range(4)] + hit_rows[1:]
    write_rows(fixture.hits_path, hit_rows)
    fixture.rebind()
    output = fixture.tmp/'candidate'
    result = export.verify_or_export(fixture.load(), output_dir=output)
    assert result['counts']['train/trait']['retained_rows'] == 0
    assert not (output/relative).exists()
    manifest = export.read_json(output/'dataset_manifest.json')
    assert not any(shard['path'] == relative for shard in manifest['splits']['train']['shards'])
    assert len(rows(output/'row_mapping.jsonl')) == 12


def test_negative_minimum_rejected(fixture):
    with pytest.raises(ValueError, match='positive'):
        export.plan_report(fixture.load(), 0)


def test_mhc_chain_expansion_preserves_indexed_non_mhc_fields(fixture):
    relative = fixture.spec[0]['path']
    changed = rows(fixture.mhc/relative)
    for row in changed:
        row['chains'].insert(0, 'GGGG')
        row['chain_roles'].insert(0, 'mhc')
        for field in ('regions', 'chain_metadata'):
            row[field] = {str(int(i)+1): value for i, value in row[field].items()}
        row['chain_metadata']['0'] = {'sequence_scope': 'full', 'mhc_policy': 'sequence_first_mhc_v1'}
    write_rows(fixture.mhc/relative, changed)
    fixture.rebind()
    report = export.verify_or_export(fixture.load())
    assert report['status'] == 'verified_selection_audit_pending'


def test_bound_audit_counts_cannot_understate_hits(fixture):
    audit = export.read_json(fixture.report_path)
    audit['totals']['train/trait/flagged_rows'] = 0
    write(fixture.report_path, audit)
    plan = export.read_json(fixture.plan_path)
    plan['audit_report_sha256'] = export.sha(fixture.report_path)
    write(fixture.plan_path, plan)
    with pytest.raises(ValueError, match='removal counts differ'):
        export.verify_or_export(fixture.load())


def test_manifest_shard_reordering_rejected(fixture):
    manifest = export.read_json(fixture.mhc/'dataset_manifest.json')
    manifest['splits']['train']['shards'].reverse()
    write(fixture.mhc/'dataset_manifest.json', manifest)
    with pytest.raises(ValueError, match='membership/order/counts'):
        fixture.load()


def test_cohort_bit_order_rejected(fixture):
    plan = export.read_json(fixture.plan_path)
    plan['cohort_bit_order'].reverse()
    write(fixture.plan_path, plan)
    with pytest.raises(ValueError, match='bit order'):
        fixture.load()


def test_strict_union_policy_cannot_silently_become_exact_only(fixture):
    plan = export.read_json(fixture.plan_path)
    plan['rule'] = 'exact_only'
    write(fixture.plan_path, plan)
    with pytest.raises(ValueError, match='strict-union'):
        fixture.load()


def test_progress_reports_verified_shards(fixture):
    events = []
    export.verify_or_export(fixture.load(), progress=events.append)
    assert len(events) == 4
    assert events[-1]['shards_finished'] == events[-1]['shards_total'] == 4
    assert sum(event['scanned_rows'] for event in events) == 12


@pytest.mark.parametrize('path', ['../outside.jsonl', '/tmp/out.jsonl', 'train/../../out.jsonl', 'train/deeper/file.jsonl', 'train/x.txt'])
def test_unsafe_shard_paths_rejected(path):
    with pytest.raises(ValueError, match='path'):
        export.relative_path(path)
