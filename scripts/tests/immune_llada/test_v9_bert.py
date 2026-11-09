"""Activate pllm; run USE_TF=0 python -m pytest on this file."""
from dataclasses import replace
import json

import pytest
import torch

from dllm.pipelines.immune_llada.data.records import BioSeqChain,BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.v9_bert import V9Collator,V9Model,corrupt,OBJECTIVE
from dllm.pipelines.immune_llada.fixed_chain.v_family import SCHEMA as FAMILY_SCHEMA
from scripts.research.bert_mlm_objectives import components,for_model

SPEC=dict(schema=FAMILY_SCHEMA,classes=['human:IGHV1','unknown'],loss_weight=.1)


def config():
    return FixedChainConfig(d_model=48,n_layers=1,n_heads=4,mlp_hidden_size=96)


def records(label):
    return [BioSeqRecord([BioSeqChain('LMNPQRST','antigen'),
        BioSeqChain('ACDEFGHIKLMNPQRSTVWY'*3,'antibody_heavy',
        metadata={'v_call':'IGHV1-2*01','sequence_scope':'full','h_anarci_type':'H'}),
        BioSeqChain('VWYACD','antibody_light')],
        'antibody_antigen','test',metadata={'species':'human','relation_supervised':True},labels={'relation':label})]*4


@pytest.mark.parametrize('seed',[0,42,531])
def test_label_flip_has_identical_encoder_inputs_and_distinct_targets(seed):
    collator=V9Collator(config(),family_spec=SPEC)
    positive=corrupt(collator(records('binding')),collator.tokenizer,torch.Generator().manual_seed(seed))
    negative=corrupt(collator(records('nonbinding')),collator.tokenizer,torch.Generator().manual_seed(seed))
    assert positive['explicit_relation_mask'].any() and negative['explicit_relation_mask'].any()
    for k,value in V9Model.feature_args(positive).items():
        assert torch.equal(value,V9Model.feature_args(negative)[k]),k
    assert negative['labels'][~negative['explicit_relation_mask']].eq(-100).all()
    assert ((positive['labels']!=-100)&~positive['explicit_relation_mask']).any()
    assert torch.equal(positive['family_labels'],negative['family_labels'])
    assert (negative['input_ids'][negative['explicit_relation_mask']]==collator.tokenizer.mask_token_id).all()


def test_real_gradient_save_load_and_objective_dispatch(tmp_path):
    torch.set_num_threads(2)
    collator=V9Collator(config(),family_spec=SPEC)
    model=V9Model(config(),family_spec=SPEC)
    batch=corrupt(collator(records('binding')+records('nonbinding')),collator.tokenizer,torch.Generator().manual_seed(42))
    output=model(**batch)
    loss=output['loss_sum']/batch['labels'].ne(-100).sum()+.1*output['family_loss_sum']/batch['family_labels'].ne(-100).sum()
    loss.backward()
    assert model.family_head.weight.grad.abs().sum()>0
    assert torch.isfinite(loss)
    model.save_pretrained(tmp_path)
    restored=V9Model.from_pretrained(tmp_path)
    assert all(torch.equal(x,restored.state_dict()[k]) for k,x in model.state_dict().items())
    assert for_model(restored)[2] is corrupt
    assert components(OBJECTIVE,SPEC)[2] is corrupt
    settings=json.loads((tmp_path/'research_config.json').read_text());settings['recipe']['input_corruption']='wrong'
    (tmp_path/'research_config.json').write_text(json.dumps(settings))
    with pytest.raises(ValueError,match='contract changed'):V9Model.from_pretrained(tmp_path)
