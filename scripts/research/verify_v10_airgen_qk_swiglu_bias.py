"""Qualify the actual new V10 architecture without starting formal training.

Activate pllm; from the project root run:
USE_TF=0 python -B -m scripts.research.verify_v10_airgen_qk_swiglu_bias --output ABS_NEW_DIRECTORY --device cuda
Uses real frozen-V10 smoke records, forced MLM equivalence and whole-CDR tasks,
then length-2048 stress updates. No W&B, submission or checkpoint promotion.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.v10_airgen import V10Collator as LegacyV10Collator
from dllm.pipelines.immune_llada.fixed_chain.v9_bert import corrupt as legacy_corrupt
from dllm.pipelines.immune_llada.fixed_chain.v10_airgen_qk_swiglu_bias import (
    ARCHITECTURE, OPTIONS, V10Collator, V10QKSwiGLUBiasModel, components, corrupt, parameter_count, swiglu_hidden_size)
from scripts.research.bert_mlm_data import IndexedPreparedDataset, sha, write_json
from scripts.research.prepare_bert_v10_qk_swiglu_bias import REFERENCE, ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if not args.output.is_absolute() or not args.reference.is_absolute():
        raise ValueError('Use absolute paths')
    if args.output.exists():
        raise FileExistsError(args.output)
    reference=json.loads(args.reference.read_text())
    if sha(args.reference)!=json.loads((args.reference.parent/'direct_launch.json').read_text())['plan_sha256']:
        raise ValueError('Reference plan changed')
    old_cfg=FixedChainConfig(**reference['model'])
    cfg=replace(old_cfg,mlp_hidden_size=swiglu_hidden_size(old_cfg.d_model))
    spec=reference['family_prediction']
    previous=LegacyV10Collator(old_cfg,family_spec=spec)
    collator=V10Collator(cfg)
    dataset=IndexedPreparedDataset(reference['data_index'])
    # Selection is inherited from frozen smoke indices, not model outcomes.
    selected=[];groups=set()
    for index in reference['smoke_indices']:
        row=dataset[index]
        key=(row.source,row.task_type,row.labels.get('relation'))
        if key not in groups:
            selected.append((index,row));groups.add(key)
        if len(selected)==8:
            break
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device=torch.device(args.device)
    model=V10QKSwiGLUBiasModel(cfg).to(device).train()
    model.backbone.set_activation_checkpointing('whole_layer')
    if device.type=='cuda':
        for block in model.backbone.transformer.blocks:
            block.sdpa_backend='efficient'
    actual=sum(p.numel() for p in model.parameters())
    assert actual==parameter_count(cfg)==77476073
    assert not hasattr(model,'scope_embeddings')
    assert not any('scope' in name for name in model.state_dict())
    assert not hasattr(model,'family_head') and not hasattr(model,'family_spec')
    assert not any('family' in name for name in model.state_dict())
    assert all(block.type_feature_gate.eq(0) for block in model.backbone.transformer.blocks)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    checks=[]
    def update(raw,seed,mode=None):
        noisy=corrupt(raw,model.tokenizer,torch.Generator().manual_seed(seed),mode=mode)
        batch={k:v.to(device) for k,v in noisy.items()}
        optimizer.zero_grad(set_to_none=True)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext():
            result=model(**batch)
            assert 'family_labels' not in batch and not any('family' in k for k in result)
            loss=result['loss_sum']/batch['labels'].ne(-100).sum().clamp_min(1)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        bias_gradients=[block.inter_score_bias.grad.detach().norm().item()
                        for block in model.backbone.transformer.blocks]
        type_gate_gradients=[block.type_feature_gate.grad.detach().abs().item()
                             for block in model.backbone.transformer.blocks]
        pair_gradients=[block.pair_score_bias.grad.detach().norm().item()
                        for block in model.backbone.transformer.blocks]
        recognition_gradients=[block.recognition_score_bias.grad.detach().norm().item()
                               for block in model.backbone.transformer.blocks]
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if device.type=='cuda':torch.cuda.synchronize(device)
        return dict(length=batch['input_ids'].shape[1],loss=loss.detach().item(),gradient_norm=norm.item(),
                    corruption_mode=int(noisy['corruption_mode'][0,0]),
                    corrupted_span_residues=int(noisy['span_mask'].sum()),
                    span_target_tokens=int((noisy['span_mask'] & noisy['labels'].ne(-100)).sum()),
                    total_target_tokens=int(noisy['labels'].ne(-100).sum()),
                    per_layer_bias_gradient_norms=bias_gradients,
                    per_layer_type_gate_absolute_gradients=type_gate_gradients,
                    per_layer_pair_bias_gradient_norms=pair_gradients,
                    per_layer_recognition_bias_gradient_norms=recognition_gradients,
                    type_feature_receiver_slots=raw['type_token_indices'].ge(0).sum().item(),
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None)
    for number,(index,row) in enumerate(selected):
        before,after=previous([row]),collator([row])
        assert before.keys()-{'scope_ids','family_labels'}==after.keys()-{'type_token_indices','cdr_region_ids'}
        assert all(torch.equal(before[k],after[k]) for k in before if k not in ('scope_ids','family_labels'))
        left=legacy_corrupt(before,model.tokenizer,torch.Generator().manual_seed(42000+number))
        right=corrupt(after,model.tokenizer,torch.Generator().manual_seed(42000+number),mode='mlm')
        assert left.keys()-{'scope_ids','family_labels'}==right.keys()-{'type_token_indices','cdr_region_ids','corruption_mode','span_mask'}
        assert all(torch.equal(left[k],right[k]) for k in left if k not in ('scope_ids','family_labels'))
        assert torch.equal(after['type_token_indices'],right['type_token_indices'])
        check=update(after,42000+number)
        check.update(kind='real_record',index=index,source=row.source,task=row.task_type,relation=row.labels.get('relation'),
                     legacy_scope_values=before['scope_ids'].unique().tolist())
        checks.append(check)
        print(json.dumps(check),flush=True)
    # Audit only the frozen diagnostic subset; these are not dataset-wide rates.
    coverage={}; span_examples={}
    for index in reference['smoke_indices']:
        row=dataset[index]
        raw=collator([row])
        group=coverage.setdefault(row.source,dict(records=0,span_eligible_records=0,cdr_residues=0))
        group['records']+=1
        group['span_eligible_records']+=int(raw['cdr_region_ids'].gt(0).any())
        group['cdr_residues']+=int(raw['cdr_region_ids'].gt(0).sum())
        if raw['cdr_region_ids'].gt(0).any():
            family='tcr' if any(c.role.startswith('tcr_') for c in row.chains) else 'antibody'
            span_examples.setdefault(family,(index,row,raw))
    assert set(span_examples)=={'antibody','tcr'}
    for family,(index,row,raw) in span_examples.items():
        for mode in ('span_mask','span_mutation'):
            check=update(raw,73000,mode=mode)
            assert check['corrupted_span_residues']>0 and check['span_target_tokens']>0
            check.update(kind='real_region_task',family=family,source=row.source,index=index,mode=mode)
            checks.append(check)
            print(json.dumps(check),flush=True)
    from scripts.research import profile_bert_mlm as profile
    profile.BertMLMCollator,_,profile.corrupt,_=components()
    raw,_=profile.synthetic_batch(cfg,1,cfg.max_length)
    check=update(raw,43)
    assert check['length']==2048 and all(value>0 for value in check['per_layer_bias_gradient_norms'])
    assert all(value>0 for value in check['per_layer_type_gate_absolute_gradients'])
    assert all(value>0 for value in check['per_layer_pair_bias_gradient_norms'])
    assert all(value>0 for value in check['per_layer_recognition_bias_gradient_norms'])
    assert all(getattr(block,name).weight.grad.abs().sum()>0
               for block in model.backbone.transformer.blocks
               for name in ['intra_q_norm','intra_k_norm','inter_q_norm','inter_k_norm'])
    check['kind']='synthetic_length_stress'
    checks.append(check)
    files=['dllm/pipelines/immune_llada/fixed_chain/v10_airgen.py',
           'dllm/pipelines/immune_llada/fixed_chain/v10_airgen_qk_swiglu_bias.py',
           'dllm/pipelines/immune_llada/fixed_chain/region_mlm.py',
           'scripts/research/bert_mlm_objectives.py','scripts/research/run_bert_mlm.py',
           'scripts/research/bert_mlm_eval.py',
           'scripts/research/profile_bert_mlm.py',
           'scripts/research/launch_bert_unbinding_v8.py',
           'scripts/research/prepare_bert_v10_qk_swiglu_bias.py',
           'scripts/research/verify_v10_airgen_qk_swiglu_bias.py']
    report=dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),architecture=ARCHITECTURE,
        objective=model.objective,recipe=model.recipe,
        architecture_options=OPTIONS,parameters=actual,device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        dtype='bfloat16 autocast' if device.type=='cuda' else 'float32',
        attention_backend='efficient' if device.type=='cuda' else 'math',
        reference_plan=str(args.reference),reference_plan_sha256=sha(args.reference),
        data_index=reference['data_index'],data_index_sha256=reference['data_index_sha256'],
        real_collation_exact_except_removed_scope_family_and_added_training_metadata=True,
        forced_mlm_corruption_exact=True,span_masking_implemented=True,
        diagnostic_subset_span_coverage=coverage,
        family_head_labels_loss_metrics_present=False,
        removed_input_feature='scope_ids',added_input_feature='type_token_indices',checks=checks,
        sources={str(ROOT/p):sha(ROOT/p) for p in files},
        submitted=False,formal_training_started=False,
        limitation='Single-GPU transient qualification updates on random weights, not formal training or downstream performance.')
    args.output.mkdir(parents=True)
    write_json(args.output/'checks.json',report)
    print(json.dumps(dict(status='passed',parameters=actual,real_records=len(selected),stress=check)),flush=True)


if __name__=='__main__':
    main()
