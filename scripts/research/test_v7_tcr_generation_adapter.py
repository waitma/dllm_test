"""Check the new TCR adapter with artificial pMHC and a tiny frozen native model.

Activate conda pllm; use protenix_abtcr Python with CUDA_VISIBLE_DEVICES='':
python -I -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_tcr_generation_adapter.py
No corpus, production weights, annotation, GPU, or checkpoint files are used.
"""
from pathlib import Path
import inspect
import json
import sys

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
SOURCE = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/source_snapshot_v1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))
from downstream.v7_tcr_generation import beta_input, beta_decode, beta_result
from downstream.v7_native import input_fingerprint
from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
from dllm.pipelines.immune_llada.v3.layout import SCOPES
import torch


def main():
    torch.set_num_threads(1); torch.manual_seed(42)
    model = FactorialModel(ImmuneConfig(d_model=24, n_layers=2, n_heads=3,
        mlp_hidden_size=48, max_length=128, full_capacity=24, fragment_capacity=12), 'f011').eval()
    assert list(inspect.signature(beta_input).parameters) == ['model','epitope','mhc_pseudo']
    assert Path(sys.modules[type(model).__module__].__file__).resolve().is_relative_to(SOURCE)
    inputs, target = beta_input(model, 'ACDEFGHIK', 'A'*34)
    assert int(target.sum()) == 12 and torch.all(inputs['input_ids'][target] == model.tokenizer.mask_token_id)
    alpha = inputs['role_ids'] == CHAIN_ROLE_TO_ID['tcr_alpha']
    assert not target[alpha].any() and torch.all(inputs['scope_ids'][alpha] == SCOPES['missing'])
    assert inputs['input_ids'][0,1] == model.tokenizer.special_id('<binding>')
    initial = input_fingerprint(inputs); calls=[]; original=model.denoise
    def capture(**actual):
        calls.append({key:value.clone() for key,value in actual.items()})
        return original(**actual)
    model.denoise=capture
    try:
        output,audit=beta_decode(model,inputs,target,seed=41)
    finally:
        model.denoise=original
    assert len(calls)==audit['forward_calls'] and input_fingerprint(inputs)==initial
    assert torch.equal(output[~target],inputs['input_ids'][~target])
    for actual,trace in zip(calls,audit['trace']):
        assert set(actual)==set(inputs) and 'labels' not in actual
        assert torch.equal(actual['input_ids'][~target],inputs['input_ids'][~target])
        assert int((target & actual['input_ids'].eq(model.tokenizer.mask_token_id)).sum())==trace['pending_before']
    # Output parser must retain invalid/empty/no-terminal candidates without adding anchors.
    cases=[('CASSF',True,True),('CASSW',True,True),('AASSF',True,False),('',True,False),('CASSF',False,False)]
    for sequence,terminate,valid in cases:
        ids=model.tokenizer.encode_residues(sequence)
        if terminate:ids+=[model.terminal_id]
        ids+=[model.tokenizer.encode_residues('A')[0]]*(12-len(ids))
        fake=inputs['input_ids'].clone();fake[target]=torch.tensor(ids)
        row=beta_result(model,fake,target)
        assert row['raw_beta_token_ids']==ids and row['syntactically_valid']==valid and row['repaired'] is False
        assert row['sequence']==(sequence if terminate else sequence+'A'*(12-len(sequence)))
    rejected=0
    for ep,mhc in [('', 'A'*34),('ACD','A'*33),('ACX','A'*34),('ACD','X'*34)]:
        try:beta_input(model,ep,mhc)
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid condition accepted')
    print(json.dumps(dict(status='passed',native_tiny_f011=True,actual_forward_calls=len(calls),
        pMHC_fixed_every_step=True,hidden_entire_junction=True,missing_alpha_not_generated=True,
        reference_or_length_input_api=False,raw_invalid_candidates_retained=len(cases),
        invalid_conditions_rejected=rejected,production_model_loaded=False,annotation_called=False,
        real_data_read=False,gpu_used=False,checkpoints_created=0)),flush=True)


if __name__ == '__main__':
    main()
