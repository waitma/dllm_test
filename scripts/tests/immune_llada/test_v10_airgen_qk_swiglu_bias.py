"""Activate pllm; USE_TF=0 python -B -m pytest THIS_ABSOLUTE_FILE -q.

Check the new MINT variant against independently materialized dense attention,
including independent layer-bias gradients, checkpoint replay, unchanged
V10 data/noise, and the actual CUDA efficient-attention backend.
"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from torch.nn import functional as F

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord, CHAIN_ROLE_TO_ID
from dllm.pipelines.immune_llada.fixed_chain.v10_airgen import V10Model, V10Collator as LegacyV10Collator
from dllm.pipelines.immune_llada.fixed_chain.v10_airgen_qk_swiglu_bias import (
    ARCHITECTURE, SCHEMA, OPTIONS, AirGenQKSwiGLUBiasBlock, V10Collator, V10QKSwiGLUBiasModel,
    OBJECTIVE, RECIPE, corrupt, parameter_count, swiglu_hidden_size)
from dllm.pipelines.immune_llada.fixed_chain.v9_bert import OBJECTIVE as V9_OBJECTIVE, corrupt as legacy_corrupt
from scripts.research.bert_mlm_objectives import components, for_model
from scripts.tests.immune_llada.test_v10_airgen import SPEC, config as old_config, records, copies


def config():
    return replace(old_config(), mlp_hidden_size=swiglu_hidden_size(old_config().d_model))


def dense(block, hidden, chains, valid, positions, bias, type_indices=None, roles=None):
    """Select intra/inter scores BEFORE one softmax; no doubled-key features."""
    x = block.attn_norm(hidden)
    qkv = {}
    for branch in ['intra', 'inter']:
        for letter in 'qkv':
            name = branch+'_'+letter
            value = getattr(block, name)(x)
            if letter in 'qk':
                gain = getattr(block, name+'_norm').weight
                value = F.layer_norm(value, (hidden.shape[-1],), gain)
            value = value.reshape(*hidden.shape[:2], block.n_heads, block.head_dim).transpose(1, 2)
            if branch == 'intra' and letter in 'qk':
                value = block.rope(value, positions)
            qkv[name] = value
    same = chains[:, None, :, None] == chains[:, None, None, :]
    intra = qkv['intra_q'] @ qkv['intra_k'].transpose(-1, -2) / block.head_dim**.5
    inter = qkv['inter_q'] @ qkv['inter_k'].transpose(-1, -2) / block.head_dim**.5
    if roles is None:
        inter = inter + (block.pair_score_bias.sum()+block.recognition_score_bias.sum())*0
    else:
        left,right=roles[:,None,:,None],roles[:,None,None,:]
        pair=((left.eq(1)&right.eq(2))|(left.eq(2)&right.eq(1))
              |(left.eq(4)&right.eq(5))|(left.eq(5)&right.eq(4)))
        recognition=torch.zeros_like(pair)
        for receptor in (1,2,3,4,5):
            for context in (6,7,8):
                recognition |= (left.eq(receptor)&right.eq(context))|(left.eq(context)&right.eq(receptor))
        inter = inter + block.pair_score_bias[None,:,None,None]*pair
        inter = inter + block.recognition_score_bias[None,:,None,None]*recognition
    scores = torch.where(same, intra, inter + bias[None, :, None, None])
    weights = scores.masked_fill(~valid[:, None, None, :], float('-inf')).softmax(-1)
    out = (weights.masked_fill(~same, 0) @ qkv['intra_v']
           + weights.masked_fill(same, 0) @ qkv['inter_v'])
    out = out.transpose(1, 2).reshape_as(hidden)
    hidden = hidden + block.out_proj(out)
    if type_indices is None:
        hidden = hidden + block.type_feature_gate * 0
    else:
        # Independent, explicit per-position reference for dynamic feedback.
        rows = []
        for row, indices in zip(hidden, type_indices.tolist()):
            rows.append(torch.stack([
                value + block.type_feature_gate * (F.layer_norm(row[source], (row.shape[-1],))
                                                   if source >= 0 else torch.zeros_like(value))
                for value, source in zip(row, indices)]))
        hidden = torch.stack(rows)
    gate, value = F.linear(block.ff_norm(hidden), block.fc1.weight).chunk(2, -1)
    return hidden + F.linear(F.silu(gate) * value, block.fc2.weight)


@pytest.mark.parametrize('chain_ids', [[0]*9, [0]*4+[1]*5, [-1,-1,0,0,1,1,2,2,2]])
@pytest.mark.parametrize('deltas', [[0.,0.,0.,0.], [.7,-1.2,.1,.4]])
def test_dense_joint_softmax_outputs_and_all_gradients(chain_ids, deltas):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    block = AirGenQKSwiGLUBiasBlock(config()).double()
    with torch.no_grad():
        block.inter_score_bias.copy_(torch.tensor(deltas,dtype=torch.float64))
    reference = deepcopy(block)
    x = torch.randn(2,9,32,dtype=torch.float64,requires_grad=True)
    y = x.detach().clone().requires_grad_()
    chains = torch.tensor([chain_ids]*2)
    valid = torch.ones_like(chains,dtype=torch.bool)
    valid[0,-1] = False
    positions = torch.tensor([[0,0,1,2,1,2,1,2,3]]*2)
    actual = block(x,copies(chains,valid),positions)
    expected = dense(reference,y,chains,valid,positions,reference.inter_score_bias)
    torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-11)
    weights = torch.randn_like(actual)
    (actual*weights).sum().backward()
    (expected*weights).sum().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=3e-12,rtol=3e-11)
    for (name,left),(other_name,right) in zip(block.named_parameters(),reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(left.grad,right.grad,atol=5e-12,rtol=5e-10,msg=name)


def test_bias_has_expected_probability_and_gradient():
    block = AirGenQKSwiGLUBiasBlock(config()).double()
    q = torch.zeros(1,4,4,8,dtype=torch.float64)
    vi, vc = torch.zeros_like(q), torch.ones_like(q)
    bias = torch.tensor([-.8,0.,.4,1.],dtype=torch.float64,requires_grad=True)
    chains = torch.tensor([[0,1,1,1]])
    valid = torch.ones_like(chains,dtype=torch.bool)
    out = block.attention(q,q,q,q,vi,vc,copies(chains,valid),bias)
    expected = 3*bias.exp()/(1+3*bias.exp())
    torch.testing.assert_close(out[0,:,0,0],expected)
    grad = torch.autograd.grad(out[0,:,0,0].sum(),bias)[0]
    torch.testing.assert_close(grad,expected*(1-expected))
    # With only one chain, no inter edge can consume the bias.
    single = copies(torch.zeros_like(chains),valid)
    out = block.attention(q,q,q,q,vi,vc,single,bias)
    assert out.eq(0).all()
    assert torch.autograd.grad(out.sum(),bias)[0].eq(0).all()


@pytest.mark.parametrize('roles',[[1,2,7,0],[4,5,6,0],[5,4,8,0]])
def test_pair_recognition_probabilities_and_analytic_gradient(roles):
    block=AirGenQKSwiGLUBiasBlock(config()).double()
    with torch.no_grad():
        block.pair_score_bias.fill_(torch.log(torch.tensor(2.)).item())
        block.recognition_score_bias.fill_(torch.log(torch.tensor(4.)).item())
    q=torch.zeros(1,4,4,8,dtype=torch.float64)
    vi,vc=torch.zeros_like(q),torch.zeros_like(q)
    vi[:,:,0,3]=1;vc[:,:,1,0]=1;vc[:,:,2,1]=1;vc[:,:,3,2]=1
    chains=torch.tensor([[0,1,2,-1]])
    out=block.attention(q,q,q,q,vi,vc,copies(chains,torch.ones_like(chains,dtype=torch.bool)),
                        block.inter_score_bias,torch.tensor([roles]))
    # Intra:Pair:Recognition:Grammar = 1:2:4:1 in one joint softmax.
    expected=torch.tensor([.25,.5,.125,.125],dtype=torch.float64).expand(4,4)
    torch.testing.assert_close(out[0,:,0,:4],expected,atol=1e-8,rtol=1e-7)
    p=out[0,:,0,0]
    grad=torch.autograd.grad(p.sum(),block.pair_score_bias)[0]
    torch.testing.assert_close(grad,p*(1-p))


def test_typed_relations_match_dense_all_gradients_and_distinguish_antigen_chains():
    torch.manual_seed(71)
    block=AirGenQKSwiGLUBiasBlock(config()).double()
    with torch.no_grad():
        block.inter_score_bias.copy_(torch.tensor([.1,-.2,.3,-.4]))
        block.pair_score_bias.copy_(torch.tensor([.7,-.5,.3,1.1]))
        block.recognition_score_bias.copy_(torch.tensor([-.6,.8,-.1,.9]))
        block.type_feature_gate.fill_(.2)
    other=deepcopy(block)
    # Ag chains 2 and 3 share a role but remain inter, with only generic bias.
    chains=torch.tensor([[0,0,1,1,2,2,3,3,-1,-1,4]])
    roles=torch.tensor([[1,1,2,2,7,7,7,7,0,0,11]])
    valid=torch.ones_like(chains,dtype=torch.bool);valid[:,-1]=False
    positions=torch.tensor([[1,2,1,2,1,2,1,2,0,0,1]])
    indices=torch.tensor([[8,8,8,8,-1,-1,-1,-1,-1,-1,-1]])
    x=torch.randn(1,11,32,dtype=torch.float64,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    actual=block(x,copies(chains,valid),positions,indices,roles)
    expected=dense(other,y,chains,valid,positions,other.inter_score_bias,indices,roles)
    torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-11)
    weights=torch.randn_like(actual)
    (actual*weights).sum().backward();(expected*weights).sum().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=3e-12,rtol=3e-11)
    for (name,left),(other_name,right) in zip(block.named_parameters(),other.named_parameters()):
        assert name==other_name
        torch.testing.assert_close(left.grad,right.grad,atol=6e-12,rtol=5e-10,msg=name)
    assert block.pair_score_bias.grad.abs().sum()>0 and block.recognition_score_bias.grad.abs().sum()>0


def test_relation_bias_types_do_not_depend_on_binding_truth():
    torch.manual_seed(73)
    model=V10QKSwiGLUBiasModel(config())
    for block in model.backbone.transformer.blocks:
        with torch.no_grad():
            block.pair_score_bias.fill_(.7);block.recognition_score_bias.fill_(-.4)
    collator=V10Collator(config())
    positive,negative=[corrupt(collator([r]),model.tokenizer,torch.Generator().manual_seed(79)) for r in records()]
    assert all(torch.equal(model.feature_args(positive)[k],model.feature_args(negative)[k]) for k in model.feature_args(positive))
    torch.testing.assert_close(model.denoise(**model.feature_args(positive))[0],
                               model.denoise(**model.feature_args(negative))[0],atol=0,rtol=0)


def test_padding_rope_and_esmc_ffn_contract():
    torch.manual_seed(19)
    cfg = config()
    block = AirGenQKSwiGLUBiasBlock(cfg).eval()
    assert swiglu_hidden_size(960)==2560 and swiglu_hidden_size(1152)==3072
    assert swiglu_hidden_size(1280)==3584
    assert block.fc1.out_features==2*cfg.mlp_hidden_size
    assert block.fc1.bias is None and block.fc2.bias is None
    for branch in ['intra','inter']:
        for letter in 'qk':
            norm = getattr(block,branch+'_'+letter+'_norm')
            assert norm.normalized_shape==(cfg.d_model,) and norm.bias is None
    chains = torch.tensor([[0,0,0,1,1,1,-1,-1]])
    valid = torch.tensor([[1,1,1,1,1,1,0,0]],dtype=torch.bool)
    pos = torch.tensor([[1,2,3,1,2,3,0,0]])
    x = torch.randn(1,8,32)
    with torch.no_grad():block.inter_score_bias.copy_(torch.tensor([.5,-.5,.1,1.]))
    expected = block(x,copies(chains,valid),pos)
    changed = x.clone(); changed[:,6:] = 1000
    shifted = pos.clone(); shifted[:,:3] += 50; shifted[:,3:6] += 900
    torch.testing.assert_close(expected[:,:6],block(changed,copies(chains,valid),pos)[:,:6])
    torch.testing.assert_close(expected[:,:6],block(x,copies(chains,valid),shifted)[:,:6],atol=2e-6,rtol=2e-5)


def test_data_corruption_and_objectives_remain_v10():
    cfg = config()
    before = LegacyV10Collator(old_config(),family_spec=SPEC)(records())
    after = V10Collator(cfg)(records())
    assert before['scope_ids'].ne(0).any()
    assert before.keys()-{'scope_ids','family_labels'}==after.keys()-{'type_token_indices','cdr_region_ids'}
    assert all(torch.equal(before[k],after[k]) for k in before if k not in ('scope_ids','family_labels'))
    type_indices=after['type_token_indices'].clone()
    tokenizer = V10Collator(cfg).tokenizer
    before = legacy_corrupt(before,tokenizer,torch.Generator().manual_seed(42))
    after = corrupt(after,tokenizer,torch.Generator().manual_seed(42),mode='mlm')
    assert before.keys()-{'scope_ids','family_labels'}==after.keys()-{'type_token_indices','cdr_region_ids','corruption_mode','span_mask'}
    assert all(torch.equal(before[k],after[k]) for k in before if k not in ('scope_ids','family_labels'))
    assert torch.equal(after['type_token_indices'],type_indices)
    assert components(OBJECTIVE,None,ARCHITECTURE)[3]==RECIPE
    assert components(OBJECTIVE,None,ARCHITECTURE)[1] is V10QKSwiGLUBiasModel
    with pytest.raises(ValueError):components('receptor_aa',SPEC,ARCHITECTURE)
    with pytest.raises(ValueError):components(OBJECTIVE,SPEC,'airgen_unknown')
    with pytest.raises(ValueError):components(OBJECTIVE,SPEC,ARCHITECTURE)
    with pytest.raises(ValueError):components(V9_OBJECTIVE,SPEC,ARCHITECTURE)


def test_type_mapping_covers_ab_tcr_vhh_mask_eos_and_padding_before_noise():
    cfg=config()
    examples=records()+[
        BioSeqRecord([BioSeqChain('ACDEF','peptide'),BioSeqChain('ACDEFGHIK','tcr_beta'),
                      BioSeqChain('LMNPQR','tcr_alpha')],'tcr_epitope','test'),
        BioSeqRecord([BioSeqChain('LMNPQR','antigen'),BioSeqChain('ACDEFGHIK','nanobody_vhh')],
                     'nanobody_antigen','test'),
        BioSeqRecord([BioSeqChain('ACDEFGHIK','nanobody_vhh')],'antibody','test'),
    ]
    collator=V10Collator(cfg)
    batch=collator(examples)
    type_for_role={1:'<ab>',2:'<ab>',3:'<nb>',4:'<tcr>',5:'<tcr>'}
    indices=batch['type_token_indices']
    for b in range(len(examples)):
        for position in range(indices.shape[1]):
            role=int(batch['role_ids'][b,position])
            source=int(indices[b,position])
            if role in type_for_role and batch['attention_mask'][b,position]:
                assert source>=0 and source<position
                assert batch['input_ids'][b,source]==collator.tokenizer.special_id(type_for_role[role])
                assert batch['chain_ids'][b,source]==-1 and batch['role_ids'][b,source]==0
            else:
                assert source==-1
        for uid in batch['chain_ids'][b][indices[b]>=0].unique():
            assert indices[b][batch['chain_ids'][b]==uid].unique().numel()==1
    assert (indices.ge(0)&batch['input_ids'].eq(collator.tokenizer.mask_token_id)).any()
    assert (indices.ge(0)&batch['terminal_mask']).any()
    assert (~batch['attention_mask']).any()
    # Corruption preserves structural routing, even when the marker itself is MASK.
    noisy=corrupt(batch,collator.tokenizer,torch.Generator().manual_seed(41))
    noisy['input_ids'][batch['chain_ids'].eq(-1)]=collator.tokenizer.mask_token_id
    assert torch.equal(indices,noisy['type_token_indices'])
    model=V10QKSwiGLUBiasModel(cfg)
    model.denoise(**model.feature_args(noisy))
    # Generation uses all capacity slots, without a reference residue/length
    # mask deciding which slots receive the feature.
    collator.mode='generation'
    row=collator.row(BioSeqRecord([BioSeqChain('ACD','antibody_heavy')],'antibody','test'))
    longer=collator.row(BioSeqRecord([BioSeqChain('ACDEFGHIK','antibody_heavy')],'antibody','test'))
    assert row==longer
    slots=[i for i,role in enumerate(row['role_ids']) if role==CHAIN_ROLE_TO_ID['antibody_heavy']]
    assert len(slots)==cfg.first_chain_capacity
    assert all(row['input_ids'][i]==collator.tokenizer.mask_token_id and row['type_token_indices'][i]>=0 for i in slots)


def test_dynamic_feedback_matches_dense_post_attention_reference_and_gradients():
    torch.manual_seed(51)
    block=AirGenQKSwiGLUBiasBlock(config()).double()
    with torch.no_grad():block.type_feature_gate.fill_(.35)
    other=deepcopy(block)
    chains=torch.tensor([[-1,0,0,0,-1,1,1,1,-1]])
    valid=torch.tensor([[1,1,1,1,1,1,1,1,0]],dtype=torch.bool)
    positions=torch.tensor([[0,1,2,3,0,1,2,3,0]])
    # Two independent type markers in one record, plus an unmapped padded token.
    indices=torch.tensor([[-1,0,0,0,-1,4,4,4,-1]])
    x=torch.randn(1,9,32,dtype=torch.float64,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    actual=block(x,copies(chains,valid),positions,indices)
    expected=dense(other,y,chains,valid,positions,other.inter_score_bias,indices)
    torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-11)
    weights=torch.randn_like(actual)
    (actual*weights).sum().backward();(expected*weights).sum().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=3e-12,rtol=3e-11)
    for (name,left),(other_name,right) in zip(block.named_parameters(),other.named_parameters()):
        assert name==other_name
        torch.testing.assert_close(left.grad,right.grad,atol=5e-12,rtol=5e-10,msg=name)
    assert block.type_feature_gate.grad.abs()>0
    # Isolate the broadcast itself: other objects and grammar/padding receive
    # exactly zero update; source features retain a differentiable path.
    fresh=x.detach().clone().requires_grad_()
    result=block.add_type_features(fresh,indices)
    torch.testing.assert_close(result[indices.lt(0)],fresh[indices.lt(0)],atol=0,rtol=0)
    changed=fresh.detach().clone();changed[:,0,0]+=2
    changed=block.add_type_features(changed,indices)
    assert not torch.allclose(result[:,1:4],changed[:,1:4])
    torch.testing.assert_close(result[:,5:8],changed[:,5:8],atol=0,rtol=0)
    assert torch.autograd.grad((result[:,1:4]*weights[:,1:4]).sum(),fresh)[0][:,0].abs().sum().item()>0


def test_zero_gates_preserve_baseline_and_are_independent_across_layers():
    torch.manual_seed(57)
    model=V10QKSwiGLUBiasModel(replace(config(),n_layers=3))
    batch=V10Collator(model.config)(records())
    gates=[b.type_feature_gate for b in model.backbone.transformer.blocks]
    assert len({g.data_ptr() for g in gates})==3 and all(g.eq(0) for g in gates)
    embedded=model.backbone.transformer.wte(batch['input_ids'])+model.role_embeddings(batch['role_ids'])
    args=[embedded,batch['attention_mask'],batch['chain_ids'],batch['position_ids']]
    actual=model.denoise(**model.feature_args(batch))[0]
    baseline=model._denoise_embedded(*args)[0]
    torch.testing.assert_close(actual,baseline,atol=0,rtol=0)
    with torch.no_grad():gates[0].fill_(.4)
    assert gates[1].eq(0) and gates[2].eq(0)
    changed=model.denoise(**model.feature_args(batch))[0]
    assert not torch.allclose(actual[...,model.allowed_target],changed[...,model.allowed_target])


def test_synthetic_profile_shifts_type_sources_when_inserting_context(monkeypatch):
    from scripts.research import profile_bert_mlm as profile
    monkeypatch.setattr(profile,'BertMLMCollator',V10Collator)
    cfg=config()
    batch,tokenizer=profile.synthetic_batch(cfg,2,512)
    indices=batch['type_token_indices']
    receiver=indices.ge(0)
    sources=indices.clamp_min(0)
    assert batch['input_ids'].shape==(2,512)
    assert batch['input_ids'].gather(1,sources)[receiver].eq(tokenizer.special_id('<ab>')).all()
    assert batch['chain_ids'].gather(1,sources)[receiver].eq(-1).all()
    assert indices[~receiver].eq(-1).all()
    model=V10QKSwiGLUBiasModel(cfg)
    model.denoise(**model.feature_args(batch))


def test_family_is_absent_and_negative_supervises_only_relation(monkeypatch):
    torch.manual_seed(63)
    model=V10QKSwiGLUBiasModel(config())
    collator=V10Collator(config())
    batch=collator(records())
    assert not any('family' in name for name in batch)
    assert not any('family' in name for name in model.state_dict())
    assert not hasattr(model,'family_head') and not hasattr(model,'family_spec')
    assert not hasattr(model,'family_loss_weight')
    # Gene/species annotations no longer construct any training targets.
    changed=records()
    changed[0]=replace(changed[0],metadata=dict(changed[0].metadata,species='mouse'),
                       chains=[replace(c,metadata=dict(c.metadata,v_gene='IGHV4-1*01')) for c in changed[0].chains])
    other=collator(changed)
    assert batch.keys()==other.keys() and all(torch.equal(batch[k],other[k]) for k in batch)
    negative=corrupt(collator(records()[1:]),model.tokenizer,torch.Generator().manual_seed(67))
    assert torch.equal(negative['labels'].ne(-100),negative['explicit_relation_mask'])
    result=model(**negative)
    assert result.keys()=={'loss_sum','logits'}
    expected=F.cross_entropy(result['logits'][negative['labels'].ne(-100)].float(),
                             negative['labels'][negative['labels'].ne(-100)],reduction='sum')
    torch.testing.assert_close(result['loss_sum'],expected)
    result['loss_sum'].backward()
    # The final layer broadcasts only to chain slots. Relation-only negatives
    # have no direct chain supervision, so this layer's gate gradient is zero.
    assert model.backbone.transformer.blocks[-1].type_feature_gate.grad.eq(0)
    from scripts.research import bert_mlm_eval
    monkeypatch.setattr(bert_mlm_eval,'read_samples',lambda _: [dict(record=r,source_group='test') for r in records()])
    metrics=bert_mlm_eval.token_validation(model,{'files':{'token_valid':{}}},torch.device('cpu'),batch_size=2)
    assert not any('family' in key for group in metrics.values() for key in group)


@pytest.mark.parametrize('invalid',['shape','dtype','range','missing','context','source'])
def test_malformed_type_routing_is_rejected(invalid):
    model=V10QKSwiGLUBiasModel(config())
    batch=V10Collator(config())(records())
    indices=batch['type_token_indices']
    receiver=indices[0].ge(0).nonzero()[0,0]
    if invalid=='shape':batch['type_token_indices']=indices[:,:-1]
    elif invalid=='dtype':batch['type_token_indices']=indices.float()
    elif invalid=='range':indices[0,receiver]=indices.shape[1]
    elif invalid=='missing':indices[0,receiver]=-1
    elif invalid=='context':indices[0,0]=0
    elif invalid=='source':indices[0,receiver]=receiver
    with pytest.raises(ValueError,match='type_token_indices|Type feature sources'):
        model.denoise(**model.feature_args(batch))


def test_each_layer_bias_is_independent():
    torch.manual_seed(37)
    model=V10QKSwiGLUBiasModel(replace(config(),n_layers=3))
    blocks=list(model.backbone.transformer.blocks)
    biases=[block.inter_score_bias for block in blocks]
    for name in ('pair_score_bias','recognition_score_bias'):
        params=[getattr(block,name) for block in blocks]
        assert len({p.data_ptr() for p in params})==3 and all(p.eq(0).all() for p in params)
    assert len({bias.data_ptr() for bias in biases})==3
    assert all(bias.eq(0).all() for bias in biases)
    chains=torch.tensor([[0,0,1,1]])
    allowed=copies(chains,torch.ones_like(chains,dtype=torch.bool))
    pos=torch.tensor([[1,2,1,2]])
    x=torch.randn(1,4,32)
    before=[block(x,allowed,pos).detach() for block in blocks]
    with torch.no_grad():biases[0].fill_(.8)
    assert biases[1].eq(0).all() and biases[2].eq(0).all()
    after=[block(x,allowed,pos) for block in blocks]
    assert not torch.allclose(after[0],before[0])
    torch.testing.assert_close(after[1],before[1],atol=0,rtol=0)
    torch.testing.assert_close(after[2],before[2],atol=0,rtol=0)
    after[0].square().sum().backward()
    assert biases[0].grad.abs().sum()>0
    assert biases[1].grad is None and biases[2].grad is None


def test_scope_is_absent_and_zero_scope_control_matches():
    import inspect
    torch.manual_seed(29)
    model=V10QKSwiGLUBiasModel(config())
    batch=V10Collator(config())(records())
    assert 'scope_ids' not in batch and 'scope_ids' not in model.feature_args(batch)
    assert 'scope_ids' not in inspect.signature(model.denoise).parameters
    assert not hasattr(model,'scope_embeddings')
    assert not any('scope' in name for name in model.state_dict())
    # Reuse the previous scoped adapter with identical backbone/role weights.
    class ScopedControl(V10QKSwiGLUBiasModel):
        uses_scope_embeddings=True
        denoise=V10Model.denoise
        feature_args=staticmethod(V10Model.feature_args)
    control=ScopedControl(config())
    missing=control.load_state_dict(model.state_dict(),strict=False)
    assert missing.missing_keys==['scope_embeddings.weight'] and not missing.unexpected_keys
    with torch.no_grad():control.scope_embeddings.weight.zero_()
    scoped_batch={**batch,'scope_ids':torch.zeros_like(batch['input_ids'])}
    left=model.denoise(**model.feature_args(batch),output_hidden_states=True)
    right=control.denoise(**control.feature_args(scoped_batch),output_hidden_states=True)
    for a,b in zip(left,right):torch.testing.assert_close(a,b,atol=0,rtol=0)
    # Legacy metadata is never forwarded to the encoder, even if a caller retains it.
    poisoned={**batch,'scope_ids':torch.full_like(batch['input_ids'],999)}
    assert model.feature_args(poisoned).keys()==model.feature_args(batch).keys()


def test_checkpoint_layer_biases_and_optimizer_replay(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = config()
    model = V10QKSwiGLUBiasModel(cfg)
    assert parameter_count(cfg)==sum(p.numel() for p in model.parameters())
    assert [k for k in model.state_dict() if 'score_bias' in k]==[
        f'backbone.transformer.blocks.{layer}.{name}' for layer in range(cfg.n_layers)
        for name in ('inter_score_bias','pair_score_bias','recognition_score_bias')]
    assert all(block.inter_score_bias.eq(0).all() for block in model.backbone.transformer.blocks)
    model.backbone.set_activation_checkpointing('whole_layer')
    batch = corrupt(V10Collator(cfg)(records()),model.tokenizer,torch.Generator().manual_seed(12))
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
    def step(m,opt):
        opt.zero_grad(set_to_none=True)
        result = m(**batch)
        loss = result['loss_sum']/batch['labels'].ne(-100).sum()
        loss.backward()
        assert torch.isfinite(loss)
        for b in m.backbone.transformer.blocks:
            assert b.inter_score_bias.grad.abs().sum()>0
            assert b.pair_score_bias.grad.abs().sum()>0 and b.recognition_score_bias.grad.abs().sum()>0
            assert b.type_feature_gate.grad.abs()>0
            for name in ['intra_q_norm','intra_k_norm','inter_q_norm','inter_k_norm']:
                assert getattr(b,name).weight.grad.abs().sum()>0
            for name in ['fc1','fc2','intra_v','inter_v']:
                assert getattr(b,name).weight.grad.abs().sum()>0
        opt.step()
        return loss.detach()
    step(model,optimizer)
    model.save_pretrained(tmp_path/'new')
    saved = deepcopy(optimizer.state_dict())
    restored = V10QKSwiGLUBiasModel.from_pretrained(tmp_path/'new')
    assert for_model(restored)[1] is V10QKSwiGLUBiasModel
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    other = torch.optim.AdamW(restored.parameters(),lr=1e-4)
    other.load_state_dict(saved)
    torch.testing.assert_close(step(model,optimizer),step(restored,other),atol=0,rtol=0)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],atol=0,rtol=0)
    V10Model(old_config(),family_spec=SPEC).save_pretrained(tmp_path/'old')
    with pytest.raises(ValueError,match='contract changed'):V10Model.from_pretrained(tmp_path/'new')
    with pytest.raises(ValueError,match='contract changed'):V10QKSwiGLUBiasModel.from_pretrained(tmp_path/'old')
    settings = json.loads((tmp_path/'new/research_config.json').read_text())
    assert settings['schema']==SCHEMA and settings['options']==OPTIONS
    no_span_settings=deepcopy(settings)
    no_span_settings['schema']='immune.fixed_chain.v10_airgen_qk_swiglu_bias.v6'
    (tmp_path/'research_config.json').write_text(json.dumps(no_span_settings))
    with pytest.raises(ValueError):V10QKSwiGLUBiasModel.from_pretrained(tmp_path)
    assert 'family_spec' not in settings and settings['variant']==OBJECTIVE
    untyped_settings=deepcopy(settings)
    untyped_settings['schema']='immune.fixed_chain.v10_airgen_qk_swiglu_bias.v5'
    (tmp_path/'new/research_config.json').write_text(json.dumps(untyped_settings))
    with pytest.raises(ValueError,match='contract changed'):V10QKSwiGLUBiasModel.from_pretrained(tmp_path/'new')
    family_settings=deepcopy(settings)
    family_settings['schema']='immune.fixed_chain.v10_airgen_qk_swiglu_bias.v4'
    family_settings['family_spec']=SPEC
    (tmp_path/'new/research_config.json').write_text(json.dumps(family_settings))
    with pytest.raises(ValueError,match='contract changed'):V10QKSwiGLUBiasModel.from_pretrained(tmp_path/'new')
    shared_settings=deepcopy(settings)
    shared_settings['schema']='immune.fixed_chain.v10_airgen_qk_swiglu_bias.v1'
    shared_settings['options']['score_bias']='inter_relative_to_intra; per_head; shared_across_layers; zero_initialized'
    (tmp_path/'new/research_config.json').write_text(json.dumps(shared_settings))
    with pytest.raises(ValueError,match='contract changed'):V10QKSwiGLUBiasModel.from_pretrained(tmp_path/'new')
    scoped_settings=deepcopy(settings)
    scoped_settings['schema']='immune.fixed_chain.v10_airgen_qk_swiglu_bias.v2'
    (tmp_path/'new/research_config.json').write_text(json.dumps(scoped_settings))
    with pytest.raises(ValueError,match='contract changed'):V10QKSwiGLUBiasModel.from_pretrained(tmp_path/'new')
    no_feedback_settings=deepcopy(settings)
    no_feedback_settings['schema']='immune.fixed_chain.v10_airgen_qk_swiglu_bias.v3'
    no_feedback_settings['options'].pop('type_feature_feedback')
    (tmp_path/'new/research_config.json').write_text(json.dumps(no_feedback_settings))
    with pytest.raises(ValueError,match='contract changed'):V10QKSwiGLUBiasModel.from_pretrained(tmp_path/'new')
    settings['options']['ffn_linear_bias']=True
    (tmp_path/'new/research_config.json').write_text(json.dumps(settings))
    with pytest.raises(ValueError,match='contract changed'):V10QKSwiGLUBiasModel.from_pretrained(tmp_path/'new')


def test_new_training_plan_preserves_data_and_checks_exact_parameter_budget(tmp_path,monkeypatch):
    from scripts.research.prepare_bert_v10_qk_swiglu_bias import REFERENCE, build_plan
    from scripts.research.run_bert_mlm import data_model_for_plan, expected_parameter_count
    reference=json.loads(REFERENCE.read_text())
    plan=build_plan(reference,tmp_path,tmp_path/'bundle/source')
    for key in ['data_model','training','budget','data_dir','data_index',
                'data_index_sha256','smoke_indices','initialization']:
        assert plan[key]==reference[key],key
    assert plan['model']['mlp_hidden_size']==3584
    assert data_model_for_plan(plan)==reference['data_model']
    assert expected_parameter_count(plan)==77476073
    assert plan['objective']==OBJECTIVE and plan['recipe']==RECIPE
    assert not any(k in plan for k in ('family_prediction','family_spec_path','family_spec_sha256','smoke_family_records'))
    changed=deepcopy(plan);changed['family_prediction']=SPEC
    with pytest.raises(ValueError,match='family prediction'):data_model_for_plan(changed)
    assert 'scope_ids' not in plan['input_features']
    assert 'type_token_indices' in plan['input_features']
    assert plan['wandb']['run_id']!=reference['wandb']['run_id']
    changed=deepcopy(plan);changed['parameter_budget']['parameters']-=1
    with pytest.raises(ValueError,match='parameter budget'):expected_parameter_count(changed)
    changed=deepcopy(plan);changed['model']['n_layers']=4
    with pytest.raises(ValueError,match='only the explicit'):data_model_for_plan(changed)
    # Exercise the data gate's explicit compatibility with the original VHH
    # release while retaining the new objective/recipe architecture checks.
    from scripts.research import bert_mlm_v8_data
    from scripts.research.bert_mlm_data import sha
    from scripts.research.run_bert_mlm import data_gate
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    index=tmp_path/'index.json'
    index.write_text(json.dumps(dict(binding_kind='immune.bert_mlm.v8_binding.v1',binding='unused',
        binding_sha256='unused',manifest=str(manifest),manifest_sha256=sha(manifest),shards=[],files={},records={'train':0,'valid':0})))
    def bound_release(path,digest,cfg):
        assert cfg==reference['data_model']
        return dict(required_objective='binding_mlm_nonbinding_relation_only_vhh',shards=[])
    monkeypatch.setattr(bert_mlm_v8_data,'verify_binding',bound_release)
    bound=deepcopy(plan);bound.update(data_index=str(index),data_index_sha256=sha(index))
    assert data_gate(bound)['status']=='passed'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('autocast', [False,True])
def test_cuda_efficient_matches_math_with_bias_gradients(autocast):
    from contextlib import nullcontext
    torch.manual_seed(33)
    cfg = replace(config(),d_model=128,n_heads=2,mlp_hidden_size=swiglu_hidden_size(128))
    block = AirGenQKSwiGLUBiasBlock(cfg).cuda()
    with torch.no_grad():
        block.inter_score_bias.copy_(torch.tensor([.7,-.8],device='cuda'))
        block.pair_score_bias.copy_(torch.tensor([.2,-.3],device='cuda'))
        block.recognition_score_bias.copy_(torch.tensor([-.4,.6],device='cuda'))
        block.type_feature_gate.fill_(.2)
    chains = torch.tensor([[0]*4+[1]*3+[2]*6+[-1]*2],device='cuda')
    roles = torch.tensor([[1]*4+[2]*3+[7]*6+[0]*2],device='cuda')
    valid = torch.ones_like(chains,dtype=torch.bool);valid[:,-1]=False
    allowed = copies(chains,valid)
    pos = torch.arange(15,device='cuda')[None]
    type_indices=torch.tensor([[-1]+[0]*6+[13]*6+[-1]*2],device='cuda')
    x = torch.randn(1,15,128,device='cuda',requires_grad=True)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        def evaluate(backend):
            block.sdpa_backend=backend
            with torch.autocast('cuda',dtype=torch.bfloat16) if autocast else nullcontext():
                out=block(x,allowed,pos,type_indices,roles)
                loss=out.square().sum()
            return out,torch.autograd.grad(loss,[x,*block.parameters()])
        expected,reference=evaluate('math')
        actual,gradients=evaluate('efficient')
        atol,rtol = (3e-2,5e-2) if autocast else (3e-5,5e-5)
        torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
        for left,right in zip(gradients,reference):
            assert torch.isfinite(left).all()
            if autocast:
                # BF16 kernels round intermediates differently. Relative L2
                # checks each complete gradient tensor without amplifying
                # cancellation around individual near-zero matrix entries.
                relative_l2=(left.float()-right.float()).norm()/right.float().norm().clamp_min(1e-12)
                assert relative_l2 < .02
            else:
                torch.testing.assert_close(left,right,atol=atol,rtol=rtol)
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous
