"""Test assay label and fixed-framework admission gates.

Activate pllm, then run USE_TF=0 python -B -m pytest
/project_c20250601/251105016/project/dllm_test/scripts/tests/test_antibody_negative_expansion.py -q
"""
import pytest
import csv
import json

from scripts.data.expand_v8_antibody_negatives import strict_relation, framework_template, antigen_reference


def assay(bound='0', total='2', ratio='0', binder='false', kd=''):
    return {'Binding Replicates': bound, 'Replicates': total,
            'Ratio Binding Replicates': ratio, 'Binder': binder, 'KD (nM)': kd}


def test_measured_zero_binding_requires_observed_replicates():
    assert strict_relation(assay()) == ('nonbinding', 'zero_binding_replicates')
    for total in ('0', '1', 'nan', 'inf', '2.5', ''):
        assert strict_relation(assay(total=total))[0] is None


def test_mixed_replicates_and_missing_kd_do_not_become_strict_negatives():
    assert strict_relation(assay('1', '2', '.5')) == (None, 'discordant_replicates')
    assert strict_relation(assay(kd='120'))[0] is None
    assert strict_relation(assay(bound='1', ratio='0'))[0] is None
    assert strict_relation(assay(binder='unknown'))[0] is None


def test_positive_is_source_measurement_without_arbitrary_affinity_cutoff():
    assert strict_relation(assay('2', '2', '1', 'true', '2000'))[0] == 'binding'
    assert strict_relation(assay('2', '2', '1', 'true', ''))[0] is None


def test_framework_requires_unique_mapping_and_fixed_light_chain():
    row = {'heavy': 'ACDEFHIKLMN', 'light': 'ACDEFG', 'HCDR1': 'CD', 'HCDR2': 'FH', 'HCDR3': 'LM'}
    template, light = framework_template([row])
    assert template.format(**row) == row['heavy'] and light == row['light']
    with pytest.raises(ValueError, match='unique fixed'):
        framework_template([row, dict(row, light='ACDEFA')])
    with pytest.raises(ValueError, match='Ambiguous'):
        framework_template([dict(row, heavy='ACDCD')])


def test_reference_target_mismatch_is_rejected(tmp_path):
    path = tmp_path / 'source.fasta'
    path.write_text('>6PPG_1|Chain A|Interleukin-17F|Homo sapiens\nACDEFGH\n')
    with pytest.raises(ValueError, match='Ambiguous reference antigen'):
        antigen_reference(path, 'interleukin-17a')


def test_independent_audit_accepts_float_counts_and_rejects_false_sequence(tmp_path, monkeypatch):
    from scripts.data import expand_v8_antibody_negatives as builder
    monkeypatch.setattr(builder, 'RAW', tmp_path)
    source, reference = tmp_path / 'source.csv', tmp_path / 'reference.csv'
    raw = dict(assay(total='4.0'), HCDR1='CD', HCDR2='FH', HCDR3='LM')
    ref = dict(raw, heavy='ACDEFHIKLMN', light='ACDEFG')
    for path, item in ((source, raw), (reference, ref)):
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(item))
            writer.writeheader()
            writer.writerow(item)
    (tmp_path / 'reference.fasta').write_text('>antigen\nACDEF\n')
    row = dict(chain_roles=['antigen', 'antibody_heavy', 'antibody_light'],
               chains=['ACDEF', ref['heavy'], ref['light']], labels=dict(relation='nonbinding'),
               metadata=dict(measurement=dict(source=str(source), source_row=2),
                 framework_reference=str(reference), antigen_reference_id='reference',
                 antigen_reference_entity='antigen'))
    assert builder.audit_exported_source_rows([row])['records'] == 1
    row['chains'][1] += 'A'
    with pytest.raises(ValueError, match='source sequence'):
        builder.audit_exported_source_rows([row])


def test_mixed_antibody_shards_do_not_turn_negative_smoke_rows_into_positives(tmp_path):
    from scripts.research.prepare_bert_unbinding_v8 import smoke_indices
    path = tmp_path / 'abexp_mixed.jsonl'
    path.write_text('\n'.join(json.dumps(dict(labels=dict(relation=label)))
                              for label in ['nonbinding', 'binding'] * 5) + '\n')
    index = dict(shards=[dict(split='train', path=str(path), records=10),
        dict(split='train', path='/data/base.jsonl', records=100),
        dict(split='train', path='/data/unbind_x.jsonl', records=100)])
    selected = smoke_indices(index, 32)
    assert len(selected) == len(set(selected)) == 32
    for rank in range(8):
        values = selected[rank::8]
        assert sum((i < 10 and i % 2 == 0) or i >= 110 for i in values) == 2
