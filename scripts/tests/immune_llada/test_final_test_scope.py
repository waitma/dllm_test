"""Run in pllm: python -B -m pytest -p no:cacheprovider scripts/tests/immune_llada/test_final_test_scope.py."""
import json
import pytest
from scripts.data import finalize_immune_test_scope as finish
from scripts.data import tcr_scope_gate as gate
from dllm.pipelines.immune_llada.data.sources import _recognition_identifiers


def sample():
    return dict(chain_roles=['tcr_beta','tcr_alpha','peptide'],chains=['BBB','AAA','PPP'],
                metadata={'source':'tcr_pmhc_fulllength'}, identifiers={'cdr3b_core':'B'})


def test_adapter_preserves_both_existing_cores():
    assert _recognition_identifiers({'cdr3a':'AAA','cdr3b':'BBB'},strip_anchors=False)['cdr3a_core']=='AAA'
    assert _recognition_identifiers({'cdr3a':'CAA AF'.replace(' ',''),'cdr3b':'CBBBF'},strip_anchors=True)['cdr3a_core']=='AAA'


def test_backfill_exact_provenance_and_ambiguity():
    row=sample();lookup={('AAA','BBB','PPP'):[{'cdr3a':'A','record_id':'r1'}]}
    restored,ids=finish.repair(row,lookup)
    assert restored['identifiers']['cdr3a_core']=='A' and ids==['r1']
    assert 'cdr3a_core' not in row['identifiers']
    lookup[('AAA','BBB','PPP')].append({'cdr3a':'AA','record_id':'r2'})
    with pytest.raises(AssertionError):finish.repair(row,lookup)


def test_same_pair_and_t4_conditional_keys():
    r=sample();r['identifiers']['cdr3a_core']='A'
    assert finish.conflicts(r,{'A|B'},set())==['exact_test_cdr3_pair']
    assert finish.conflicts(r,{'A|C','C|B'},set())==[]
    assert finish.conflicts(r,set(),{'B|PPP'})==['exact_t4_beta_epitope_test_key']
    del r['identifiers']['cdr3a_core']
    with pytest.raises(AssertionError):finish.conflicts(r,set(),set())


def test_test_scope_gate_rejects_incomplete_and_changed_evidence(tmp_path,monkeypatch):
    test=tmp_path/'test.csv';test.write_text('heldout')
    monkeypatch.setattr(gate,'required_test_paths',lambda:[test])
    m={'training_ready':True,'decontamination':{'status':'passed','pending':[]},
       'scope_finalization':{'scope_version':gate.TEST_SCOPE,'user_excluded_audit_items':gate.WAIVERS}}
    manifest=tmp_path/'dataset_manifest.json';manifest.write_text(json.dumps(m))
    scan={'status':'passed','scope_version':gate.TEST_SCOPE,'dataset_manifest_sha256':gate.sha(manifest),
          'unresolved_observed_pair_fields':0,'forbidden_overlap_rows':0,'family_split_overlap':0,
          't4_missing_from_historical_blocklist':0}
    sp=tmp_path/'final_verification.json';sp.write_text(json.dumps(scan))
    proof={'status':'passed','scope_version':gate.TEST_SCOPE,'pending':[],
           'dataset_manifest_sha256':gate.sha(manifest),'forbidden_overlap_rows':0,
           'user_excluded_audit_items':gate.WAIVERS,'benchmark_inputs':{str(test):gate.sha(test)},
           'evidence_files':{str(sp):gate.sha(sp)}}
    cp=tmp_path/'downstream_scope_certificate.json';cp.write_text(json.dumps(proof))
    assert gate.require_current_scope(tmp_path)==proof
    scan['unresolved_observed_pair_fields']=1;sp.write_text(json.dumps(scan))
    with pytest.raises(ValueError,match='incomplete'):gate.require_current_scope(tmp_path)
    scan['unresolved_observed_pair_fields']=0;sp.write_text(json.dumps(scan))
    test.write_text('changed')
    with pytest.raises(ValueError,match='inputs changed'):gate.require_current_scope(tmp_path)
