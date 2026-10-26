"""Activate pllm; python -B -m pytest -q scripts/tests/immune_llada/test_fixed_chain_validation.py."""
import torch
import pytest
from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig,FixedChainModel,FixedChainCollator
from dllm.pipelines.immune_llada.data.records import BioSeqChain,BioSeqRecord
from scripts.research.fixed_chain_validation import selected_token_metrics,summarize,chain_endpoint_totals


def test_missing_partner_is_not_a_scored_length_and_fixed_delimiters_are_not_targets():
    c=FixedChainConfig(d_model=24,n_heads=3,n_layers=2,mlp_hidden_size=48)
    model=FixedChainModel(c).eval()
    r=BioSeqRecord([BioSeqChain('CASSF','tcr_beta'),BioSeqChain('X'*100,'tcr_alpha',metadata={'synthetic_residue_mask':[1]*100})],'tcr','fixture')
    b=FixedChainCollator(c)([r])
    scores=summarize(selected_token_metrics(model,b,1.)['sample_totals'])
    assert scores['chain_count']==scores['terminal_count']==1
    assert scores['split_count']==0 and scores['aa_count']==5
    assert scores['selected_count']==6 and scores['relation_count']==0
    scores=summarize(selected_token_metrics(model,b,.5)['sample_totals'])
    assert scores['chain_count']==0
    broken={**b,'terminal_mask':torch.zeros_like(b['terminal_mask'])}
    with pytest.raises(ValueError,match='true END'):
        chain_endpoint_totals(broken,torch.zeros_like(b['terminal_mask']),full_mask=True)
