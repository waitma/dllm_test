"""Decode each fixed receptor region to its own EOS; test with test_fixed_chain.py."""
import math
import torch
from ..data.records import CHAIN_ROLE_TO_ID
from .layout import ROLES

def parse_chain(tokenizer, tokens):
    ids=[int(t) for t in tokens]
    if tokenizer.eos_token_id not in ids:return {'status':'capacity_exhausted','sequence':None}
    end=ids.index(tokenizer.eos_token_id);content=ids[:end]
    if not content:return {'status':'empty_chain','sequence':None}
    if tokenizer.mask_token_id in content:return {'status':'incomplete','sequence':None}
    aa=set(tokenizer.encode_residues('LAGVSERTIDPKQNFYMHWCXBUZO'))
    if any(t not in aa for t in content):return {'status':'invalid_token','sequence':None}
    return {'status':'ok','sequence':''.join(tokenizer.token(t) for t in content),'length':end}

@torch.no_grad()
def generate(model,batch,*,steps=32,temperature=0.,seed=42,expected_chains=None):
    if type(steps) is not int or steps<1 or not math.isfinite(temperature) or temperature<0:
        raise ValueError('Invalid decoding parameters')
    device=next(model.parameters()).device;features={k:v.to(device) for k,v in model.feature_args(batch).items()}
    ids=features['input_ids'].clone();canvas=features['target_slot_mask']&features['attention_mask']
    if not canvas.any(1).all() or not ids[canvas].eq(model.tokenizer.mask_token_id).all():
        raise ValueError('Targets must be fully masked generation-mode fixed regions')
    roles={CHAIN_ROLE_TO_ID[r]:r for r in ROLES};regions=[]
    for b in range(ids.shape[0]):
        row=[]
        for uid in features['chain_ids'][b].unique(sorted=True).tolist():
            positions=((features['chain_ids'][b]==uid)&features['attention_mask'][b]).nonzero().flatten()
            if uid<0 or not len(positions):continue
            role=int(features['role_ids'][b,positions[0]])
            if role in roles:row.append((positions,roles[role],bool(canvas[b,positions].any())))
        count=expected_chains[b] if isinstance(expected_chains,(tuple,list)) else expected_chains
        if count is not None and count!=len(row):raise ValueError('Declared chain count mismatch')
        regions.append(row)
    rng=torch.Generator(device=device).manual_seed(seed);was_training=model.training;model.eval();used=0
    try:
        for step in range(steps):
            logits,_=model.denoise(**{**features,'input_ids':ids});used=step+1
            for b,row in enumerate(regions):
                for positions,role,target in row:
                    if not target:continue
                    if parse_chain(model.tokenizer,ids[b,positions])['status']=='ok':continue
                    pending=positions[ids[b,positions].eq(model.tokenizer.mask_token_id)]
                    if not len(pending):continue
                    scores=logits[b,pending].float()
                    allowed=model.allowed_target
                    if not torch.isfinite(scores[:,allowed]).all():raise ValueError('Nonfinite decoder logits')
                    probs=(scores.masked_fill(~allowed, float("-inf"))/(temperature or 1.)).softmax(-1)
                    prediction=probs.argmax(-1) if temperature==0 else torch.multinomial(probs,1,generator=rng).squeeze(1)
                    confidence=probs.gather(1,prediction[:,None]).squeeze(1)
                    remaining=len(positions)*(steps-step-1)//steps
                    count=max(0,len(pending)-remaining);chosen=confidence.topk(count).indices
                    ids[b,pending[chosen]]=prediction[chosen]
            if all(parse_chain(model.tokenizer,ids[b,p])['status']=='ok' for b,row in enumerate(regions) for p,_,t in row if t):break
    finally:model.train(was_training)
    results=[]
    for b,row in enumerate(regions):
        chains=[dict(role=role,target=target,**parse_chain(model.tokenizer,ids[b,p])) for p,role,target in row]
        results.append(dict(status='ok' if all(c['status']=='ok' for c in chains) else 'invalid',chains=chains,
                            raw_input_ids=ids[b].tolist(),steps=used))
    return results
