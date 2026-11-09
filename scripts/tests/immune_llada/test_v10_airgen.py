"""Activate pllm; run USE_TF=0 python -m pytest THIS_FILE -q.

Verify against the existing official MINT layer, unchanged V9 data/noise,
all-chain gradients, checkpoint replay and real fused CUDA attention.
"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from dllm.pipelines.bioseq.ophiuchus.mint.modules import TransformerLayer
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.v10_airgen import (
    ARCHITECTURE, AirGenBlock, V10Collator, V10Model, corrupt)
from dllm.pipelines.immune_llada.fixed_chain.v9_bert import V9Collator, RECIPE, OBJECTIVE
from dllm.pipelines.immune_llada.fixed_chain.v_family import SCHEMA as FAMILY_SCHEMA
from scripts.research.bert_mlm_objectives import components, for_model

SPEC = dict(schema=FAMILY_SCHEMA, classes=['human:IGHV1', 'unknown'], loss_weight=.1)


def config():
    return FixedChainConfig(d_model=32, n_layers=2, n_heads=4, mlp_hidden_size=128,
                            first_chain_capacity=65, second_chain_capacity=15, full_capacity=82)


def records():
    return [BioSeqRecord([
        BioSeqChain('LMNPQRST', 'antigen'),
        BioSeqChain('ACDEFGHIKLMNPQRSTVWY' * 3, 'antibody_heavy',
                     metadata={'v_call':'IGHV1-2*01','sequence_scope':'full','h_anarci_type':'H'}),
        BioSeqChain('VWYACD', 'antibody_light')], 'antibody_antigen', 'test',
        metadata={'species':'human','relation_supervised':True}, labels={'relation':label})
        for label in ('binding', 'nonbinding')]


def copies(chains, valid):
    same = chains[:, :, None].eq(chains[:, None, :])
    return torch.cat((same & valid[:, None, :], ~same & valid[:, None, :]), -1)[:, None]


@pytest.mark.parametrize('chains', [[0]*9, [0]*4+[1]*5, [-1, -1, 0, 0, 1, 1, 2, 2, 2]])
def test_matches_official_airgen_layer_outputs_and_input_gradients(chains):
    torch.set_num_threads(2)
    torch.manual_seed(9)
    cfg = config()
    actual = AirGenBlock(cfg)
    reference = TransformerLayer(cfg.d_model, cfg.mlp_hidden_size, cfg.n_heads,
        add_bias_kv=False, use_esm1b_layer_norm=True, use_rotary_embeddings=True, use_multimer=True)
    for branch, prefix in [(reference.self_attn, 'intra'), (reference.multimer_attn, 'inter')]:
        for letter in 'qkv':
            getattr(branch, letter+'_proj').load_state_dict(getattr(actual, prefix+'_'+letter).state_dict())
    for dst, src in [(reference.self_attn.out_proj, actual.out_proj),
                     (reference.self_attn_layer_norm, actual.attn_norm),
                     (reference.final_layer_norm, actual.ff_norm),
                     (reference.fc1, actual.fc1), (reference.fc2, actual.fc2)]:
        dst.load_state_dict(src.state_dict())
    ids = torch.tensor([chains]*2)
    valid = torch.ones_like(ids, dtype=torch.bool)
    positions = torch.arange(len(chains))[None].expand_as(ids)
    x = torch.randn(2, len(chains), cfg.d_model, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    out = actual(x, copies(ids, valid), positions)
    expected, _ = reference(y.transpose(0, 1), self_attn_mask=ids[:, :, None].ne(ids[:, None, :]))
    expected = expected.transpose(0, 1)
    torch.testing.assert_close(out, expected, atol=2e-6, rtol=2e-5)
    weights = torch.randn_like(out)
    (out * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=3e-6, rtol=3e-5)
    for dst, src in [(reference.self_attn.q_proj, actual.intra_q),
                     (reference.multimer_attn.q_proj, actual.inter_q),
                     (reference.multimer_attn.v_proj, actual.inter_v)]:
        torch.testing.assert_close(src.weight.grad, dst.weight.grad, atol=8e-6, rtol=8e-5)


def test_padding_invariance_and_intra_only_position_rotation():
    torch.manual_seed(19)
    block = AirGenBlock(config()).eval()
    ids = torch.tensor([[0,0,0,1,1,1,-1,-1]])
    valid = torch.tensor([[1,1,1,1,1,1,0,0]], dtype=torch.bool)
    positions = torch.tensor([[1,2,3,1,2,3,0,0]])
    x = torch.randn(1,8,32)
    out = block(x, copies(ids,valid), positions)
    changed = x.clone(); changed[:,6:] = 1000
    shifted = positions.clone(); shifted[:,:3] += 50; shifted[:,3:6] += 900
    torch.testing.assert_close(out[:,:6], block(changed, copies(ids,valid), positions)[:,:6])
    torch.testing.assert_close(out[:,:6], block(x, copies(ids,valid), shifted)[:,:6], atol=2e-6, rtol=1e-5)


def test_v9_inputs_labels_and_noise_are_byte_identical():
    cfg = config()
    a = V9Collator(cfg, family_spec=SPEC)(records())
    b = V10Collator(cfg, family_spec=SPEC)(records())
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k],b[k]) for k in a)
    left = corrupt(a, V9Collator(cfg,family_spec=SPEC).tokenizer, torch.Generator().manual_seed(42))
    right = corrupt(b, V10Collator(cfg,family_spec=SPEC).tokenizer, torch.Generator().manual_seed(42))
    assert all(torch.equal(left[k],right[k]) for k in left)
    relation = right['explicit_relation_mask']
    assert right['labels'][1][~relation[1]].eq(-100).all()
    positive = corrupt(V10Collator(cfg,family_spec=SPEC)(records()[:1]),
                       V10Collator(cfg,family_spec=SPEC).tokenizer,torch.Generator().manual_seed(42))
    negative = corrupt(V10Collator(cfg,family_spec=SPEC)(records()[1:]),
                       V10Collator(cfg,family_spec=SPEC).tokenizer,torch.Generator().manual_seed(42))
    assert torch.equal(positive['input_ids'], negative['input_ids'])
    assert components(OBJECTIVE,SPEC,ARCHITECTURE)[3] == RECIPE
    with pytest.raises(ValueError): components('receptor_aa',SPEC,ARCHITECTURE)


def test_gradients_checkpoint_reload_and_adam_replay(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = V10Model(config(),family_spec=SPEC)
    model.backbone.set_activation_checkpointing('whole_layer')
    collator = V10Collator(config(),family_spec=SPEC)
    batch = corrupt(collator(records()), model.tokenizer, torch.Generator().manual_seed(12))
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
    def step(m, opt):
        opt.zero_grad(set_to_none=True)
        result=m(**batch)
        loss=result['loss_sum']/batch['labels'].ne(-100).sum()+.1*result['family_loss_sum']/batch['family_labels'].ne(-100).sum()
        loss.backward()
        assert torch.isfinite(loss)
        for block in m.backbone.transformer.blocks:
            for name in ['intra_q','inter_q','inter_v']:
                assert getattr(block,name).weight.grad.abs().sum()>0
        assert m.family_head.weight.grad.abs().sum()>0
        opt.step()
        return loss.detach()
    step(model,optimizer)
    model.save_pretrained(tmp_path)
    saved_optimizer=deepcopy(optimizer.state_dict())
    restored=V10Model.from_pretrained(tmp_path)
    assert for_model(restored)[1].func is V10Model
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    other=torch.optim.AdamW(restored.parameters(),lr=1e-4)
    other.load_state_dict(saved_optimizer)
    torch.testing.assert_close(step(model,optimizer),step(restored,other),atol=0,rtol=0)
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,restored.state_dict()[k],atol=0,rtol=0)
    settings=json.loads((tmp_path/'research_config.json').read_text())
    settings['options']['residual_scale']=.1
    (tmp_path/'research_config.json').write_text(json.dumps(settings))
    with pytest.raises(ValueError,match='contract changed'): V10Model.from_pretrained(tmp_path)


def test_frozen_v9_training_and_data_are_preserved_in_v10_plan():
    from pathlib import Path
    from scripts.research.prepare_bert_v10 import build_plan, REFERENCE
    from scripts.research.run_bert_mlm import data_model_for_plan, expected_parameter_count
    reference=json.loads((REFERENCE/'bundle/plan.json').read_text())
    plan=build_plan(reference,Path('/tmp/v10_plan_test'),Path('/tmp/v10_plan_test/bundle/source'))
    for key in ['training','budget','objective','recipe','data_index','data_index_sha256',
                'data_dir','family_prediction','smoke_indices','initialization']:
        assert plan[key]==reference[key],key
    assert data_model_for_plan(plan)==reference['model']
    assert expected_parameter_count(plan)==75755753
    assert plan['parameter_budget']['relative_difference']<.001
    changed=deepcopy(plan);changed['model']['second_chain_capacity']-=1
    with pytest.raises(ValueError,match='only the explicit'):data_model_for_plan(changed)
    changed=deepcopy(plan);changed['parameter_budget']['parameters']=80000000
    with pytest.raises(ValueError,match='parameter budget'):data_model_for_plan(changed)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_efficient_attention_forward_backward_matches_math():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _check_cuda_attention()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _check_cuda_attention():
    torch.manual_seed(33)
    cfg=replace(config(),d_model=128,n_heads=2,mlp_hidden_size=512)
    model=AirGenBlock(cfg).cuda()
    ids=torch.tensor([[0]*7+[1]*6],device='cuda')
    allowed=copies(ids,torch.ones_like(ids,dtype=torch.bool))
    pos=torch.arange(13,device='cuda')[None]
    x=torch.randn(1,13,128,device='cuda',requires_grad=True)
    model.sdpa_backend='math'
    expected=model(x,allowed,pos)
    reference=torch.autograd.grad(expected.square().sum(),x)[0]
    model.sdpa_backend='efficient'
    result=model(x,allowed,pos)
    grad=torch.autograd.grad(result.square().sum(),x)[0]
    torch.testing.assert_close(result,expected,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(grad,reference,atol=1e-5,rtol=2e-5)
