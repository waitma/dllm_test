"""Activate pllm; pytest -q scripts/tests/test_cdr_block_soft.py.

Partition target residues into disjoint groups. Score each group while every
member retains its MASK embedding; apply soft feedback only to other targets.
"""
import math

import torch


def partitions(target, groups, rule):
    if groups<1 or rule not in ('interleaved','contiguous'):raise ValueError((groups,rule))
    ordinal=target.long().cumsum(-1)-1
    labels=ordinal.remainder(groups) if rule=='interleaved' else (ordinal*groups//target.sum(-1,keepdim=True).clamp_min(1))
    return [target & labels.eq(i) for i in range(groups)]


def posterior(model,predict,batch,target,features,maps,prior,groups,rule,temperature,strength,observer=None):
    if not math.isfinite(temperature) or temperature<=0 or not 0<=strength<=1:raise ValueError((temperature,strength))
    tokens=batch['input_ids'];native_ids=features['input_ids'];mask_id=model.tokenizer.mask_token_id
    assert target.dtype==torch.bool and target.shape==tokens.shape
    assert tokens[target].eq(mask_id).all(), 'Every scored target must begin masked'
    assert prior.shape[:2]==tokens.shape and len(maps)==len(tokens)==len(native_ids)
    assert torch.isfinite(prior).all() and (prior>=0).all()
    assert torch.allclose(prior.sum(-1),torch.ones_like(prior[...,0]),atol=1e-5)
    native=[]
    for i,mapping in enumerate(maps):
        coordinates=dict(mapping)
        assert len(coordinates)==len(mapping)==len(set(coordinates.values())), 'Coordinate map must be injective'
        assert all(0<=v<tokens.shape[1] and 0<=n<native_ids.shape[1] for v,n in mapping)
        assert all(v in coordinates for v in target[i].nonzero().flatten().tolist())
        assert all(native_ids[i,coordinates[v]]==mask_id for v in target[i].nonzero().flatten().tolist())
        native.append(coordinates)
    embed=model.backbone.transformer.wte;original=embed(native_ids)
    weights=prior.pow(1/temperature);normalizer=weights.sum(-1,keepdim=True)
    assert (normalizer>0).all(), 'Temperature caused probability underflow'
    weights/=normalizer
    expected=weights@embed.weight
    active=torch.zeros_like(features['input_ids'],dtype=torch.bool);replacement=original.clone();calls=[0]
    masks=partitions(target,groups,rule);counts=torch.zeros_like(target,dtype=torch.int);result=prior.clone()
    def hook(layer,inputs,output):
        calls[0]+=1;assert torch.equal(inputs[0],features['input_ids'])
        revised=torch.where(active[...,None],replacement,output)
        assert torch.equal(revised[~active],output[~active])
        return revised
    handle=embed.register_forward_hook(hook)
    try:
        for index,scored in enumerate(masks):
            active.zero_();replacement.copy_(original);feedback=target & ~scored
            for i in range(len(target)):
                for v in feedback[i].nonzero().flatten().tolist():
                    n=native[i][v];active[i,n]=True
                    replacement[i,n]=(1-strength)*original[i,n]+strength*expected[i,v]
                for v in scored[i].nonzero().flatten().tolist():assert not active[i,native[i][v]]
            before=calls[0];probs=predict(batch['input_ids']);assert calls[0]==before+1
            assert probs.shape==prior.shape
            assert torch.isfinite(probs).all() and (probs>=0).all()
            assert torch.allclose(probs.sum(-1),torch.ones_like(probs[...,0]),atol=1e-5)
            result[scored]=probs[scored];counts+=scored
            if observer is not None:observer(index,scored,feedback)
    finally:handle.remove()
    assert counts[target].eq(1).all() and counts[~target].eq(0).all()
    return result,groups+1
