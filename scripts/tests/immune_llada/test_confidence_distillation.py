"""Activate pllm; USE_TF=0 python -m pytest THIS_ABSOLUTE_FILE -q.

Verify confidence attenuates whole interfaces and preserves global gradient
normalization across uneven microbatches/ranks; teacher weights are detached.
"""
import pytest
import torch

from dllm.pipelines.immune_llada.fixed_chain.structure_losses import chain_aware_kd, normalized_objective


def fixture_tensors():
    generator = torch.Generator().manual_seed(93)
    student = torch.randn(6, 5, generator=generator, requires_grad=True)
    teacher = torch.randn(6, 5, generator=generator, requires_grad=True)
    pairs = torch.tensor([[0,0,1],[0,0,2],[0,1,3],[1,0,1],[1,0,2],[1,1,3]])
    chains = torch.tensor([[0,0,1,1],[0,0,1,1]])
    noise = torch.tensor([[0,1,2,0],[1,0,0,1]])
    return student, teacher, pairs, chains, noise


@pytest.mark.parametrize('mask_weights',[None,(1.,2.,3.)])
@pytest.mark.parametrize('confidence',[0.,.1,.5,1.])
def test_uniform_low_confidence_reduces_loss_without_cancelling(mask_weights, confidence):
    args=fixture_tensors()
    reference=chain_aware_kd(*args,mask_weights=mask_weights)
    actual=chain_aware_kd(*args,mask_weights=mask_weights,
        pair_confidence=torch.full((6,),confidence))
    for name in ('intra','inter'):
        torch.testing.assert_close(actual[name+'_sum'],reference[name+'_sum']*confidence)
        assert actual[name+'_count']==reference[name+'_count']


def test_low_quality_interface_does_not_change_intra_and_detaches_teacher():
    s,t,p,c,n=fixture_tensors()
    w=torch.tensor([1.,.1,.1,1.,0.,0.],requires_grad=True)
    reference=chain_aware_kd(s,t,p,c,n)
    actual=chain_aware_kd(s,t,p,c,n,pair_confidence=w)
    torch.testing.assert_close(actual['intra_sum'],reference['intra_sum'])
    assert actual['inter_count']==2
    (actual['intra_sum']+actual['inter_sum']).backward()
    assert t.grad is None and w.grad is None
    assert torch.count_nonzero(s.grad[4:])==0
    assert torch.count_nonzero(s.grad[:4])>0


def test_uneven_rank_partition_matches_single_global_gradient():
    s,t,p,c,n=fixture_tensors()
    confidence=torch.tensor([.8,.1,0.,.4,.5,1.])
    full=chain_aware_kd(s,t,p,c,n,pair_confidence=confidence)
    objective=normalized_objective(s.sum()*0,1,full,intra_weight=.05,inter_weight=.01)
    expected=torch.autograd.grad(objective,s)[0]
    distributed=torch.zeros_like(s)
    # Two nonempty ranks plus one empty rank; DDP averages their scaled gradients.
    for selected in (torch.arange(3),torch.arange(3,6),torch.tensor([],dtype=torch.long)):
        local=s.detach()[selected].clone().requires_grad_(True)
        kd=chain_aware_kd(local,t[selected],p[selected],c,n,pair_confidence=confidence[selected])
        obj=normalized_objective(local.sum()*0,1,kd,.05,.01,denominators=(1,2,2),world_size=3)
        distributed[selected]=torch.autograd.grad(obj,local)[0]/3
    torch.testing.assert_close(distributed,expected)


@pytest.mark.parametrize('bad',[torch.ones(6)*-1,torch.ones(6)*1.01,
    torch.ones(6)*float('nan'),torch.ones(6,1)])
def test_invalid_confidence_rejected(bad):
    with pytest.raises(ValueError,match='confidence'):
        chain_aware_kd(*fixture_tensors(),pair_confidence=bad)
