"""Activate pllm; python -B -m pytest -q scripts/tests/test_cdr_block_soft.py."""
from types import SimpleNamespace
import pytest
import torch
from scripts.downstream.cdr_block_soft import partitions,posterior


def fixture():
    torch.manual_seed(18);embed=torch.nn.Embedding(7,5)
    with torch.no_grad():embed.weight.mul_(.2)
    model=SimpleNamespace(backbone=SimpleNamespace(transformer=SimpleNamespace(wte=embed)),
                          tokenizer=SimpleNamespace(mask_token_id=0))
    ids=torch.tensor([[2,0,0,0,4,0],[3,0,5,0,0,0]]);target=ids.eq(0)
    # The real model has a different native layout from the virtual AirGen grid.
    # Include unmapped native positions and different permutations in each row.
    maps=[list(enumerate([4,1,6,2,7,0])),list(enumerate([6,4,2,0,5,7]))]
    native_ids=torch.full((len(ids),8),6,dtype=torch.long)
    for i,mapping in enumerate(maps):
        for virtual,native in mapping:native_ids[i,native]=ids[i,virtual]
    features={'input_ids':native_ids};batch={'input_ids':ids};seen=[]
    def predict(tokens):
        assert torch.equal(tokens,ids)
        x=embed(native_ids);seen.append(x.detach().clone())
        logits=(x+x.mean(1,keepdim=True))@embed.weight.T
        logits[...,0]=-torch.inf  # Match the production predictor's MASK ban.
        probs=logits.softmax(-1)
        return torch.stack([probs[i,[n for _,n in mapping]] for i,mapping in enumerate(maps)])
    prior=predict(ids)
    return model,predict,batch,target,features,maps,prior,seen


def test_partition_coverage():
    target=torch.tensor([[False,True,True,False,True],[True,False,False,False,False]])
    for rule in ('interleaved','contiguous'):
        for k in (1,2,3,8):assert torch.equal(torch.stack(partitions(target,k,rule)).sum(0),target.long())


def test_zero_strength_and_single_group_are_baseline():
    for k,strength in [(1,.9),(2,0.),(3,0.)]:
        args=fixture();p,nfe=posterior(*args[:7],groups=k,rule='interleaved',temperature=.5,strength=strength)
        assert nfe==k+1 and torch.equal(p,args[6])


def test_scored_embeddings_stay_mask_and_visible_context_stays_exact():
    for rule in ('interleaved','contiguous'):
        args=fixture();before=args[-1][0];events=[]
        posterior(*args[:7],groups=3,rule=rule,temperature=1.,strength=.5,observer=lambda i,s,f:events.append((s.clone(),f.clone())))
        assert len(args[-1])==4 and len(events)==3
        for actual,(scored,feedback) in zip(args[-1][1:],events):
            native_scored=torch.zeros(actual.shape[:2],dtype=torch.bool)
            native_feedback=torch.zeros_like(native_scored)
            expected=before.clone()
            for i,mapping in enumerate(args[5]):
                for virtual,native in mapping:
                    native_scored[i,native]=scored[i,virtual]
                    native_feedback[i,native]=feedback[i,virtual]
                    if feedback[i,virtual]:
                        soft=args[6][i,virtual]@args[0].backbone.transformer.wte.weight
                        expected[i,native]=.5*before[i,native]+.5*soft
            assert torch.equal(actual[native_scored],before[native_scored])
            assert torch.equal(actual[~native_feedback],before[~native_feedback])
            assert not (scored & feedback).any() and torch.equal(scored | feedback,args[3])
            assert not torch.equal(actual[native_feedback],before[native_feedback])
            assert torch.allclose(actual,expected,atol=1e-7,rtol=1e-6)


@pytest.mark.parametrize('failure',['predict','observer'])
def test_hook_is_removed_after_failure(failure):
    args=fixture();embed=args[0].backbone.transformer.wte
    def fail(*unused):raise RuntimeError('intentional failure')
    call_args=list(args[:7])
    if failure=='predict':call_args[1]=fail
    with pytest.raises(RuntimeError,match='intentional failure'):
        posterior(*call_args,groups=2,rule='interleaved',temperature=.5,strength=.5,
                  observer=fail if failure=='observer' else None)
    assert not embed._forward_hooks
    assert torch.equal(args[1](args[2]['input_ids']),args[6])


@pytest.mark.parametrize('invalid',['visible_virtual_target','visible_native_target','aliased_mapping'])
def test_rejects_visible_scored_targets_and_ambiguous_native_maps(invalid):
    args=fixture()
    if invalid=='visible_virtual_target':args[2]['input_ids'][0,1]=2
    elif invalid=='visible_native_target':args[4]['input_ids'][0,1]=2
    else:args[5][0][2]=(2,1)
    with pytest.raises(AssertionError):
        posterior(*args[:7],groups=2,rule='interleaved',temperature=.5,strength=.5)
    assert not args[0].backbone.transformer.wte._forward_hooks
