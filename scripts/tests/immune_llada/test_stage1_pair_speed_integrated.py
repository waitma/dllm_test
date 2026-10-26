"""Activate pllm; CUDA_VISIBLE_DEVICES='' USE_TF=0 python -B -m pytest THIS_FILE -q.

Check the integrated execution path against the measured candidate, original
algebra, nonzero full-model feedback, checkpoint replay and uneven two-rank DDP.
"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
import torch.multiprocessing as mp

from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import PairConfig, ResiduePairEncoder, StructureMLMModel
from scripts.research.stage1_pair_speed_candidates import install_cast, install_symmetric
from scripts.research.prepare_stage1_ab500k_scratch import execution_pair_config
from scripts.tests.immune_llada import test_stage1_ab500k_training as helpers


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fast_model(arm="pair_kd"):
    assert arm=="pair_kd"
    torch.manual_seed(18)
    return StructureMLMModel(helpers.model_config(), PairConfig(width=8, query_chunk=8,
        bias_implementation="symmetric", autocast_pair_embeddings=True,
        symmetric_pair_chunk=37, checkpoint_pair_chunks=False))


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_integrated_matches_previously_measured_candidate(bf16, checkpointed):
    torch.manual_seed(175)
    baseline=ResiduePairEncoder(16,PairConfig(width=8,checkpoint_pair_chunks=checkpointed)).train()
    integrated=ResiduePairEncoder(16,replace(baseline.config,bias_implementation="symmetric",
        autocast_pair_embeddings=True)).train()
    integrated.load_state_dict(baseline.state_dict())
    # The measured prototype expects a model-like object with a pair_encoder.
    class Holder: pass
    holder=Holder();holder.pair_encoder=baseline
    install_cast(holder);install_symmetric(holder)
    out1=torch.nn.Linear(8,4,bias=False);out2=deepcopy(out1)
    h1=torch.randn(2,97,16,requires_grad=True);h2=h1.detach().clone().requires_grad_()
    positions=torch.randint(0,80,(2,97));chains=torch.randint(-1,3,(2,97))
    roles=torch.randint(0,baseline.num_roles,(2,97));valid=torch.rand(2,97)>.2
    with torch.autocast("cpu",dtype=torch.bfloat16,enabled=bf16):
        a=baseline.attention_bias(h1,positions,chains,roles,valid,out1)
        b=integrated.attention_bias(h2,positions,chains,roles,valid,out2)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    upstream=torch.randn_like(a)
    (a*upstream).sum().backward();(b*upstream).sum().backward()
    torch.testing.assert_close(h1.grad,h2.grad,atol=0,rtol=0)
    for (name,p),(_,q) in zip(baseline.named_parameters(),integrated.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=name)
    torch.testing.assert_close(out1.weight.grad,out2.weight.grad,atol=0,rtol=0)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_full_model_nonzero_feedback_preserves_outputs_gradients_and_hard_masks(checkpointed):
    fast=fast_model()
    fast.pair_config=fast.pair_encoder.config=replace(fast.pair_config,checkpoint_pair_chunks=checkpointed)
    base=StructureMLMModel(fast.config,replace(fast.pair_config,bias_implementation="dense",
        autocast_pair_embeddings=False,checkpoint_pair_chunks=True))
    for layer in fast.pair_bias_outputs:torch.nn.init.normal_(layer.weight,std=.03)
    base.load_state_dict(fast.state_dict())
    batch,pairs,_=helpers.prepare_batch(fast,helpers.rows(fast.config,3),seed=10042)
    a=base(pair_indices=pairs,**batch);b=fast(pair_indices=pairs,**batch)
    torch.testing.assert_close(a['logits'],b['logits'],atol=3e-6,rtol=3e-6)
    torch.testing.assert_close(a['pair_logits'],b['pair_logits'],atol=3e-6,rtol=3e-6)
    for out in (a,b):(out['loss_sum']+out['pair_logits'].square().mean()).backward()
    for (name,p),(_,q) in zip(base.named_parameters(),fast.named_parameters()):
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=3e-4,msg=name)
    captures=[]
    handle=fast.backbone.transformer.blocks[0].register_forward_hook(lambda m,args,out:captures.append(out.detach().clone()))
    fast.eval();fast(**batch)
    changed={**batch,'input_ids':batch['input_ids'].clone()}
    light=batch['chain_ids'].eq(1)&batch['residue_mask']
    changed['input_ids'][light]=fast.tokenizer.encode_residues('W')[0]
    fast(**changed);handle.remove()
    heavy=batch['chain_ids'].eq(0)&batch['residue_mask']
    torch.testing.assert_close(captures[0][heavy],captures[1][heavy],atol=0,rtol=0)


def test_saved_model_keeps_execution_flags_and_loads_old_config_defaults(tmp_path):
    fast=fast_model().eval();fast.save_pretrained(tmp_path/'fast')
    restored=StructureMLMModel.from_pretrained(tmp_path/'fast').eval()
    assert restored.pair_config==fast.pair_config
    batch,pairs,_=helpers.prepare_batch(fast,helpers.rows(fast.config,2),seed=10042)
    with torch.no_grad():
        torch.testing.assert_close(fast(pair_indices=pairs,**batch)['logits'],
            restored(pair_indices=pairs,**batch)['logits'],atol=0,rtol=0)
    old=helpers.make_model('pair_kd');old.save_pretrained(tmp_path/'old')
    path=tmp_path/'old/research_config.json';config=json.loads(path.read_text())
    for key in ('bias_implementation','autocast_pair_embeddings','symmetric_pair_chunk'):config['pair'].pop(key)
    path.write_text(json.dumps(config))
    restored=StructureMLMModel.from_pretrained(tmp_path/'old')
    assert restored.pair_config.bias_implementation=='dense'
    assert restored.pair_config.checkpoint_pair_chunks and not restored.pair_config.autocast_pair_embeddings


def test_fast_checkpoint_replays_model_adam_scheduler_and_noise(tmp_path,monkeypatch):
    monkeypatch.setattr(helpers,'make_model',fast_model)
    helpers.test_real_checkpoint_restores_next_update_and_scheduler(tmp_path,'pair_kd')


def fast_ddp_worker(rank,temporary):
    helpers.make_model=fast_model
    helpers.ddp_worker(rank,temporary,'pair_kd')


def test_fast_two_rank_uneven_tail_and_gradients(tmp_path):
    mp.spawn(fast_ddp_worker,args=(str(tmp_path),),nprocs=2,join=True)
    actual=torch.load(tmp_path/'distributed.pt',weights_only=False)
    model=fast_model();spec=helpers.settings()
    optimizer=helpers.new_optimizer(model,spec,'pair_kd');scheduler=helpers.scheduler_for(optimizer,spec)
    expected=helpers.optimizer_update(model,optimizer,scheduler,
        helpers.build_micros(model,helpers.rows(model.config,5),5,901,'pair_kd'),spec,'pair_kd',0)
    assert actual['metrics']['records']==expected['records']==5
    assert actual['metrics']['selected_tokens']==expected['selected_tokens']
    assert actual['metrics']['intra_kl']==pytest.approx(expected['intra_kl'],rel=3e-6)
    assert all(actual['pair_gradients'].values())
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,actual['model'][name],atol=3e-6,rtol=3e-5,msg=name)
    for key,values in optimizer.state_dict()['state'].items():
        for name,value in values.items():
            torch.testing.assert_close(value,actual['optimizer']['state'][key][name],atol=3e-7,rtol=2e-4)


def test_execution_choices_preserve_supervision_and_do_not_mutate_base():
    pair=dict(width=64,feedback='all',checkpoint_pair_chunks=True)
    original=deepcopy(pair)
    fast=execution_pair_config(pair,'symmetric_no_recompute')
    assert pair==original and fast['width']==64 and fast['feedback']=='all'
    assert not fast['checkpoint_pair_chunks'] and fast['autocast_pair_embeddings']
    assert execution_pair_config(pair,'symmetric')['checkpoint_pair_chunks']
    assert execution_pair_config(pair,'baseline')==pair
    with pytest.raises(ValueError):PairConfig(bias_implementation='unsupported')
    with pytest.raises(ValueError):PairConfig(symmetric_pair_chunk=0)
