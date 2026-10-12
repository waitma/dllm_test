"""Run python -m pytest -q scripts/tests/immune_llada/test_v5_eval_repairs.py."""
import json
import sys
from types import SimpleNamespace
import pytest
import torch

from downstream.v5_single_chain_generation import SingleChainModel
from downstream.benchmark.ab_antigen.sequence_metrics import number_worker
from downstream.benchmark.tcr_generation_bench import scoring
from downstream.benchmark.tcr_generation_bench.result_io import prepare_metrics_for_json


def test_single_chain_excludes_only_separator_and_does_not_mutate_logits():
    logits=torch.randn(1,5,50)
    original=logits.clone()
    model=SimpleNamespace(tokenizer=SimpleNamespace(chain_separator_id=lambda:29),
                          denoise=lambda **kw:(logits,'hidden'))
    constrained,hidden=SingleChainModel(model).denoise(input_ids=None)
    assert hidden=='hidden'
    torch.testing.assert_close(logits,original)
    keep=[i for i in range(50) if i!=29]
    torch.testing.assert_close(constrained[...,keep],original[...,keep])
    assert (constrained.softmax(-1)[...,29]==0).all()


def test_anarci_bad_sequence_isolated_without_losing_good_sequences(tmp_path,monkeypatch):
    def fake(items,**kw):
        if any(s=='BAD' for _,s in items):
            raise TypeError("unsupported operand type(s) for +: 'NoneType' and 'int'")
        return ([ [([((1,' '),s)],0,1)] for _,s in items],[[{'chain_type':'H'}] for _ in items],None)
    monkeypatch.setitem(sys.modules,'anarci',SimpleNamespace(anarci=fake))
    source=tmp_path/'in.json';dest=tmp_path/'out.json'
    source.write_text(json.dumps({'a':'A','bad':'BAD','c':'C'}))
    number_worker(source,dest)
    output=json.loads(dest.read_text())
    assert output['a']['fv']=='A' and output['c']['fv']=='C' and output['bad'] is None
    assert set(json.loads(dest.with_suffix('.errors.json').read_text()))=={'bad'}


def test_empty_generation_is_explicit_null_not_fabricated_score(monkeypatch):
    monkeypatch.setattr(scoring,'load_conditioning_by_epitope',lambda:{})
    entries={'a':dict(epitope='a',category='benchmark14',ref_binders=['CASSF'])}
    raw=scoring.score_conditional({'a':['']*100},entries,k=100)
    result=prepare_metrics_for_json(raw,{'a'},candidate_counts={'a':0})
    assert result['per_pmhc']['a']['precision'] is None
    assert result['summary']['overall']['n_scored_by_metric']['precision']==0
    json.dumps(result,allow_nan=False)
    with pytest.raises(ValueError):
        prepare_metrics_for_json(raw,{'a'},candidate_counts={'a':100})
    raw['per_pmhc']['a']['precision']=float('inf')
    with pytest.raises(ValueError):
        prepare_metrics_for_json(raw,{'a'},candidate_counts={'a':0})
