"""Activate pllm/protenix_abtcr; python -B -m pytest scripts/tests/immune_llada/test_fixed_chain.py -q."""
import json
from dataclasses import replace
import pytest
import torch
from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig,FixedChainModel,FixedChainCollator
from dllm.pipelines.immune_llada.fixed_chain.generation import parse_chain
from dllm.pipelines.immune_llada.v3.layout import GenerationRequest
from dllm.pipelines.immune_llada.data.records import BioSeqChain,BioSeqRecord,CHAIN_ROLE_TO_ID

def config():
    return FixedChainConfig(d_model=24,n_heads=3,n_layers=2,mlp_hidden_size=48,max_length=256,
                            first_chain_capacity=40,second_chain_capacity=36,full_capacity=78)

def record(beta='CASSF',alpha='CAVF',metadata=None):
    return BioSeqRecord([BioSeqChain('ACD','peptide'),BioSeqChain(beta,'tcr_beta',metadata=metadata or {}),
                        BioSeqChain(alpha,'tcr_alpha')],'tcr_epitope','test')

def model():
    torch.set_num_threads(1)
    return FixedChainModel(config())

def test_generation_features_do_not_depend_on_hidden_length_or_residues():
    m=model();train=FixedChainCollator(config());gen=FixedChainCollator(config(),mode='generation')
    a=record();b=record('C'+'A'*80+'F','C'+'G'*70+'F',{'synthetic_residue_mask':[1]})
    expected=m.feature_args(gen([a]));hidden=m.feature_args(gen([b]))
    assert all(torch.equal(expected[k],hidden[k]) for k in expected)
    for r in [a,record('CASSSSSF','CAVVVF'),record('XXXCASSF','CAVF',{'synthetic_residue_mask':[1,1,1,0,0,0,0,0]})]:
        batch=train([r]);noised,_,_=m.corrupt(batch,torch.ones_like(batch['input_ids'],dtype=torch.float))
        got=m.feature_args({**batch,'input_ids':noised})
        assert all(torch.equal(got[k],expected[k]) for k in got)

def test_chain_rope_attention_and_fixed_delimiters():
    m=model();b=FixedChainCollator(config())([record()]);t=m.tokenizer
    first=(b['role_ids'][0]==CHAIN_ROLE_TO_ID['tcr_beta']).nonzero().flatten()
    second=(b['role_ids'][0]==CHAIN_ROLE_TO_ID['tcr_alpha']).nonzero().flatten()
    assert len(first)==40 and len(second)==36
    assert b['position_ids'][0,first[0]]==b['position_ids'][0,second[0]]==1
    local=m.layer_allowed(0,b['chain_ids'],b['position_ids'],b['attention_mask'])
    assert local[0,first[0],first].all()  # includes distances greater than 32
    assert local[0,second[0],second].all()
    assert m.options == {'residual_scale': .1, 'segment_global': True}
    assert m.mask_key(0) == ('segment',) and m.mask_key(1) == ('global',)
    assert not local[0,first[0],second[0]]
    assert m.layer_allowed(1,b['chain_ids'],b['position_ids'],b['attention_mask'])[0,first[0],second[0]]
    dot=(b['input_ids'][0]==t.chain_separator_id()).nonzero().flatten()
    assert len(dot)==1 and int(dot[0])==int(first[-1])+1
    assert not b['eligible_mask'][0,dot].any() and not b['target_slot_mask'][0,dot].any()
    assert int(b['terminal_mask'].sum())==2
    assert b['input_ids'][b['terminal_mask']].eq(t.eos_token_id).all()
    for block in m.backbone.transformer.blocks:
        assert torch.allclose(block.attention_scale,torch.full((24,),.1))
        assert not block.attention_scale.requires_grad
        assert torch.allclose(block.ffn_scale,torch.full((24,),.1))

def test_scaffold_slots_and_no_supervision_for_unknown_chain_length():
    c=FixedChainCollator(config());t=c.tokenizer
    b=c([record('XXCAF','XXXX',{'synthetic_residue_mask':[1,1,0,0,0]})])
    pos=(b['role_ids'][0]==CHAIN_ROLE_TO_ID['tcr_beta']).nonzero().flatten()
    assert b['input_ids'][0,pos[:2]].eq(t.mask_token_id).all()
    assert b['labels'][0,pos[:2]].eq(-100).all()
    missing=BioSeqRecord([BioSeqChain('XXX','antibody_heavy',metadata={'synthetic_residue_mask':[1,1,1]})],'antibody','test')
    b=c([missing]);assert not b['eligible_mask'].any() and not b['terminal_mask'].any()

@pytest.mark.parametrize('role',['antibody_heavy','antibody_light','tcr_beta','tcr_alpha'])
def test_single_chain(role):
    r=BioSeqRecord([BioSeqChain('CAAF',role)],'tcr' if role.startswith('tcr') else 'antibody','test')
    c=FixedChainCollator(config());b=c([r]);n=40 if role in ['antibody_heavy','tcr_beta'] else 36
    assert int(b['target_slot_mask'].sum())==n
    assert not b['input_ids'].eq(c.tokenizer.chain_separator_id()).any()

def test_known_chain_and_reverse_completion():
    for role in ['tcr_alpha','tcr_beta']:
        c=FixedChainCollator(config(),mode='generation',target_roles=[role]);b=c([record()])
        assert b['input_ids'][b['target_slot_mask']].eq(c.tokenizer.mask_token_id).all()
        assert b['role_ids'][b['target_slot_mask']].eq(CHAIN_ROLE_TO_ID[role]).all()
    request=GenerationRequest((BioSeqChain('ACD','peptide'),),(('tcr_beta','full'),('tcr_alpha','full')))
    c=FixedChainCollator(config(),mode='generation')
    assert all(torch.equal(c([request])[k],c([record()])[k]) for k in ['input_ids','chain_ids','position_ids','role_ids','target_slot_mask'])

def test_overflow_is_explicit_and_unknown_length_is_not_checked():
    with pytest.raises(ValueError,match='chain_capacity_overflow'):
        FixedChainCollator(config())([record('A'*40)])
    FixedChainCollator(config(),mode='generation')([record('A'*1000)])
    with pytest.raises(ValueError,match='Capacity'):
        replace(config(),full_capacity=302)

def test_backward_padding_and_roundtrip(tmp_path):
    m=model();c=FixedChainCollator(config());b=c([record()])
    out=m(**b,token_t=torch.ones_like(b['input_ids'],dtype=torch.float));assert torch.isfinite(out['loss'])
    out['loss'].backward();assert m.backbone.transformer.wte.weight.grad.abs().sum()>0
    m.eval();r=record();long=replace(r,chains=[BioSeqChain('A'*20,'peptide'),*r.chains[1:]])
    with torch.no_grad():
        a,_=m.denoise(**m.feature_args(b));bb=c([r,long]);both,_=m.denoise(**m.feature_args(bb))
    torch.testing.assert_close(a[0,:,m.allowed_target],both[0,:a.shape[1],m.allowed_target],atol=1e-5,rtol=1e-5)
    m.save_pretrained(tmp_path);loaded=FixedChainModel.from_pretrained(tmp_path)
    with torch.no_grad():v,_=loaded.denoise(**loaded.feature_args(b))
    torch.testing.assert_close(a,v)
    # Retain test metrics, not disposable checkpoint files.
    for path in tmp_path.iterdir():
        if path.suffix == ".safetensors": path.unlink()
    from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
    with pytest.raises(ValueError):FactorialModel.from_pretrained(tmp_path)

def test_chain_end_parser_and_actual_generation_api():
    m=model();t=m.tokenizer;c=FixedChainCollator(config(),mode='generation');b=c([record()]);before=b['input_ids'].clone()
    assert parse_chain(t,[t.encode_residues('C')[0],t.eos_token_id,t.mask_token_id])['sequence']=='C'
    assert parse_chain(t,[t.mask_token_id,t.eos_token_id])['status']=='incomplete'
    assert parse_chain(t,[t.eos_token_id])['status']=='empty_chain'
    def logits(**features):
        ids=features['input_ids'];scores=torch.full((*ids.shape,t.vocab_size),-20.)
        for i in range(ids.shape[0]):
            for j in range(ids.shape[1]):
                token=t.encode_residues('C')[0] if features['position_ids'][i,j]==1 else t.eos_token_id
                scores[i,j,token]=20.
        return scores,None
    m.denoise=logits
    out=m.generate(b,steps=2,expected_chains=2)
    assert out[0]['status']=='ok' and [x['sequence'] for x in out[0]['chains']]==['C','C']
    assert torch.equal(b['input_ids'],before)
    assert m.training


def test_rejects_object_collator():
    from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
    m=model()
    b=ImmuneCollator('V5',full_capacity=78,max_length=256)([record()])
    with pytest.raises(ValueError,match='FixedChainCollator'):
        m.denoise(**m.feature_args(b))
