"""Protocol regression checks. Run: python -m pytest scripts/tests/test_ab_antigen_sequence.py -q."""
import numpy as np
from downstream.benchmark.ab_antigen.sequence_data import check_sequences
from downstream.benchmark.ab_antigen.sequence_eval import group_metrics,make_generation_record
from downstream.benchmark.ab_antigen.sequence_metrics import recovery
from downstream.benchmark.ab_antigen.sequence_audit import split_groups


def test_generation_has_only_antigen_and_fixed_placeholder_canvases():
    r=make_generation_record('ACDEFG')
    assert [c.sequence for c in r.chains]==['ACDEFG','A'*166,'A'*134]
    assert [c.role for c in r.chains]==['antigen','antibody_heavy','antibody_light']
    assert not r.labels


def test_reject_overflow_not_truncation():
    assert check_sequences('A'*1800,'A'*120,'C'*110)=='context_overflow'
    assert check_sequences('ACDE','A'*167,'C'*110)=='receptor_canvas_overflow'
    assert check_sequences('ACDE','A'*120,'C'*110) is None


def test_noncanonical_and_missing_are_not_silently_repaired():
    assert check_sequences('AX','A','C')=='noncanonical_residue'
    assert check_sequences('AC','A','')=='missing_chain'


def test_group_macro_not_pooled_metric():
    rows=[dict(antibody_key='a',label=y,score=s) for y,s in [(0,0),(1,1)]]
    rows += [dict(antibody_key='b',label=y,score=s) for y,s in [(0,11),(1,10)]]
    result=group_metrics(rows,classification=True)
    assert result['macro']['roc_auc']['mean']==.5
    assert len(result['groups'])==2


def test_single_class_group_excluded_not_zero():
    r=group_metrics([dict(antibody_key='a',label=1,score=1)],classification=True)
    assert r['macro']['roc_auc']['mean'] is None
    assert len(r['excluded'])==1


def test_stronger_affinity_direction():
    r=group_metrics([dict(antibody_key='a',label=-x,score=-x) for x in [1,2,3]])
    assert np.isclose(r['macro']['spearman']['mean'],1)


def test_missing_reference_position_counts_as_mismatch():
    assert recovery([[105,' ','A']],[[105,' ','A'],[106,' ','C']],[0,1])==(1,2)


def test_imgt_insertions_not_collapsed():
    ref=[[111,' ','A'],[111,'A','C']]
    assert recovery([[111,' ','A'],[111,'A','D']],ref,[0,1])==(1,2)


def test_insufficient_homology_groups_cannot_make_three_splits():
    assert set(split_groups(['a','b']).values())=={'train'}
    assert split_groups(list('abcdef'))==split_groups(list('fedcba'))
    assert set(split_groups(list('abcdef')).values())=={'train','val','test'}


def test_pll_masks_only_one_antibody_residue_and_pools_each_chain():
    import torch
    from types import SimpleNamespace
    from downstream.benchmark.ab_antigen.sequence_eval import score_pair
    calls=[]
    class FakeModel:
        config=SimpleNamespace(mask_token_id=9)
        def last_hidden_state(self,**batch):
            assert batch['input_ids'].tolist()==[[1,2,3,4,5]]
            return torch.arange(5,dtype=torch.float32).reshape(1,5,1)
        def _denoise(self,**batch):
            ids=batch['input_ids']
            assert ids[:,:2].eq(torch.tensor([1,2])).all()
            assert ids.eq(9).sum(1).eq(1).all()
            calls.extend(torch.where(ids.eq(9))[1].tolist())
            return SimpleNamespace(logits=torch.zeros(len(ids),5,10))
    def collator(records):
        return dict(input_ids=torch.tensor([[1,2,3,4,5]]),attention_mask=torch.ones(1,5,dtype=torch.long),
            residue_mask=torch.ones(1,5,dtype=torch.bool),position_ids_chain=torch.tensor([[0,0,1,1,2]]))
    score,features=score_pair(SimpleNamespace(model=FakeModel(),collator=collator),'AC','DE','F',features=True,device='cpu')
    assert calls==[2,3,4]
    assert np.isclose(score,-np.log(10))
    assert features==[.5,2.5,4.]
