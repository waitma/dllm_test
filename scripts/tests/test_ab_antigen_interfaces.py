"""Run: python -m pytest scripts/tests/test_ab_antigen_interfaces.py -q."""
from dataclasses import asdict
import json
import math
import pytest
from downstream.benchmark.ab_antigen.interfaces import (
    evaluate,fingerprint,generation_inputs,load_dataset,PredictionJournal,read_jsonl,run_adapter,
)


def rows(task='binding'):
    return [dict(id=str(i),dataset=task,antigen='AC',heavy='DE',light='FG',
        antibody_key='ab',antigen_key=fingerprint('AC'),lineage='lineage',
        source_doi='source',assay='experiment',label=float(i%2),endpoint='log10_kd' if task=='spr' else 'binding',
        pkd=None,pkd_status='unit_and_censoring_unverified') for i in range(4)]


def generation_rows():
    return [dict(id=str(i),dataset='chimera',antigen=ag,antigen_key=fingerprint(ag),
        heavy='DE',light='FG',split='test',numbering={'reference':'must not leak'}) for i,ag in enumerate(['AC','AD'])]


def test_scores_join_by_id_not_row_order():
    r=rows();p=[dict(id=x['id'],score=x['label']) for x in reversed(r)]
    m=evaluate('binding',r,p)
    assert m['status']=='complete'
    assert m['fixed_antibody']['macro']['roc_auc']['mean']==1


def test_missing_results_require_explicit_partial_flag():
    r=rows();p=[dict(id='0',score=.2)]
    with pytest.raises(ValueError,match='missing predictions'):evaluate('binding',r,p)
    m=evaluate('binding',r,p,allow_partial=True)
    assert m['status']=='partial' and m['scored']==1 and m['expected']==4


@pytest.mark.parametrize('predictions',[[dict(id='unknown',score=1)],
    [dict(id='0',score=1),dict(id='0',score=2)]])
def test_unknown_or_duplicate_predictions_rejected(predictions):
    with pytest.raises(ValueError):evaluate('binding',rows(),predictions,allow_partial=True)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),'0.5'])
def test_nonfinite_or_string_scores_rejected(value):
    with pytest.raises(ValueError):evaluate('binding',rows(),[dict(id='0',score=value)],allow_partial=True)


def test_regression_requires_explicit_verified_pkd():
    r=rows('spr');p=[dict(id=x['id'],score=x['label'],pkd_prediction=8.) for x in r]
    m=evaluate('spr',r,p,mode='frozen_probe')
    assert m['pkd_regression']['n']==0 and m['pkd_regression']['mae'] is None
    for x in r:x.update(pkd=7.,pkd_status='verified')
    m=evaluate('spr',r,p,mode='frozen_probe')
    assert m['pkd_regression']['mae']==1 and m['pkd_regression']['rmse']==1
    with pytest.raises(ValueError):evaluate('spr',r,p)


def test_spr_sources_are_not_pooled():
    r=rows('spr');r[0]['source_doi']='different'
    m=evaluate('spr',r,[dict(id=x['id'],score=x['label']) for x in r])
    assert len(m['fixed_antibody']['excluded'])==1
    assert m['fixed_antibody']['groups'][0]['n']==3


def test_model_adapter_receives_no_labels_and_resume_skips_calls(tmp_path):
    calls=[]
    class Scorer:
        def score_pair(self,request):
            assert set(asdict(request))=={'id','antigen','heavy','light'}
            calls.append(request.id);return .5
    first=run_adapter('binding',rows(),Scorer(),tmp_path,model_identity={'weights_sha':'pinned'})
    second=run_adapter('binding',rows(),Scorer(),tmp_path,model_identity={'weights_sha':'pinned'})
    assert first==second and len(calls)==4
    with pytest.raises(ValueError,match='provenance'):
        run_adapter('binding',rows(),Scorer(),tmp_path,model_identity={'weights_sha':'changed'})
    changed=rows();changed[0]['label']=1
    with pytest.raises(ValueError,match='provenance'):
        run_adapter('binding',changed,Scorer(),tmp_path,model_identity={'weights_sha':'pinned'})


def test_wt_subtraction_does_not_reuse_another_assays_wt(tmp_path):
    r=rows('abbibench')[:2]
    for i,x in enumerate(r):x.update(assay=str(i),wt_heavy='A'*(i+1),wt_light='C')
    class Scorer:
        def score_pair(self,request):return float(len(request.heavy))
    p=run_adapter('abbibench',r,Scorer(),tmp_path,model_identity='length_control_test_only')
    assert [x['score'] for x in p]==[1,0]


def test_generation_requests_reference_free_and_ablation_seed_matched():
    r=generation_rows();a=generation_inputs(r,candidates=2)
    r[0].update(heavy='W'*150,light='Y'*130,numbering={'different':'reference'})
    assert a==generation_inputs(r,candidates=2)
    assert len(a)==8
    for item in a:
        assert 'heavy' not in asdict(item) and 'light' not in asdict(item)
        partner=next(x for x in a if x.antigen_key==item.antigen_key and x.candidate==item.candidate and x.condition!=item.condition)
        assert item.seed==partner.seed and item.antigen!=partner.antigen


def test_failed_generation_attempts_are_retained(tmp_path):
    class Generator:
        def generate(self,request):
            return dict(heavy='',light='',heavy_eos=False,light_eos=False,token_valid=False,failure_reason='missing_eos')
    r=generation_rows();p=run_adapter('chimera',r,Generator(),tmp_path,model_identity='failure_fixture',candidates=2,legacy_full_chain=True)
    m=evaluate('chimera',r,p,candidates=2,legacy_full_chain=True)
    assert m['scored']==8 and m['conditions']['correct']['token_valid_rate']==0
    p[0]['seed']+=1
    with pytest.raises(ValueError,match='mismatch'):evaluate('chimera',r,p,candidates=2,legacy_full_chain=True)


def test_chimera_does_not_silently_use_full_chain_protocol(tmp_path):
    with pytest.raises(ValueError,match='chimera_h3'):
        run_adapter('chimera',generation_rows(),None,tmp_path,model_identity='fixture')
    with pytest.raises(ValueError,match='chimera_h3'):
        evaluate('chimera',generation_rows(),[])


def test_truncated_resume_journal_rejected(tmp_path):
    journal=PredictionJournal(tmp_path,{'config':'a'})
    journal.predictions_path.write_text('{"id":')
    with pytest.raises(ValueError,match='incomplete'):PredictionJournal(tmp_path,{'config':'a'})


def test_wrong_dataset_and_duplicate_input_ids_rejected(tmp_path):
    p=tmp_path/'data.jsonl';r=rows();p.write_text('\n'.join(json.dumps(x) for x in r)+'\n')
    assert len(load_dataset(p,'binding'))==4
    with pytest.raises(ValueError):load_dataset(p,'spr')
    p.write_text(json.dumps(r[0])+'\n'+json.dumps(r[0])+'\n')
    with pytest.raises(ValueError,match='Duplicate'):load_dataset(p,'binding')


def test_cli_export_and_score(tmp_path,monkeypatch):
    from downstream.benchmark.ab_antigen.interfaces import main
    data=tmp_path/'data.jsonl'
    data.write_text(''.join(json.dumps(r)+'\n' for r in rows()))
    export=tmp_path/'export'
    monkeypatch.setattr('sys.argv',['interfaces','export','--task','binding','--data',str(data),'--out-dir',str(export)])
    main()
    assert all(set(r)=={'id','antigen','heavy','light'} for r in read_jsonl(export/'inputs.jsonl'))
    predictions=tmp_path/'predictions.jsonl'
    predictions.write_text(''.join(json.dumps(dict(id=r['id'],score=r['label']))+'\n' for r in reversed(rows())))
    output=tmp_path/'score'
    monkeypatch.setattr('sys.argv',['interfaces','evaluate','--task','binding','--data',str(data),'--predictions',str(predictions),'--out-dir',str(output)])
    main()
    assert json.loads((output/'interface_metrics.json').read_text())['scored']==4


def test_generation_rejects_probe_track():
    with pytest.raises(ValueError,match='zero-shot only'):
        evaluate('chimera',generation_rows(),[],mode='frozen_probe',allow_partial=True)
