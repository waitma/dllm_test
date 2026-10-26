"""Activate pllm/protenix_abtcr; python -B SCRIPT --model fixed50000|v7.

Check causal heavy-chain dependence before sampling on the first 16 benchmark
rows. Swap only the heavy chain, keep reference length and native layout. This
tests information flow, not binding quality or population-level performance.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['fixed50000','v7'],required=True)
    args=p.parse_args()
    import torch
    import pandas as pd
    torch.set_num_threads(2)
    gen=ROOT/'output/downstream_generation'
    if args.model=='v7':
        from downstream.v7_native import load_native_checkpoint
        model,identity=load_native_checkpoint(ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/run/checkpoint-50082',
                                             'cpu',allow_snapshot_relocations=True)
        from downstream.v7_pairing import pairing_input
        adapter=pairing_input
    else:
        from downstream.fixed_chain_eval import load
        model,identity=load(gen/'fixed_chain_75m_step50000_20261013/checkpoint','cpu')
        from scripts.downstream.run_fixed_chain_pairing_reference import reference_input
        adapter=reference_input
    from downstream.v7_native import input_fingerprint
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    frame=pd.read_csv(ROOT/'data/oas_previous_clean/splits/holdout_select_500_eval_OAS.csv')
    allowed=model.allowed_target.nonzero().flatten()
    index={int(token):i for i,token in enumerate(allowed)}
    rows=[]
    with torch.inference_mode():
        for i in range(16):
            heavy=frame.iloc[i].h_sequence.replace('-','')
            swapped=frame.iloc[(i+250)%500].h_sequence.replace('-','')
            reference=frame.iloc[i].l_sequence.replace('-','')
            assert heavy!=swapped
            features,mask=adapter(model,heavy,len(reference))[:2]
            other,other_mask=adapter(model,swapped,len(reference))[:2]
            assert torch.equal(mask,other_mask) and torch.equal(features['input_ids'][mask],other['input_ids'][mask])
            assert features['input_ids'][mask].eq(model.tokenizer.mask_token_id).all()
            before=input_fingerprint(features)
            logits,_=model.denoise(**features)
            assert input_fingerprint(features)==before
            alternative,_=model.denoise(**other)
            first=logits[mask][:,allowed].double();second=alternative[mask][:,allowed].double()
            assert torch.isfinite(first).all() and torch.isfinite(second).all()
            logp=first.log_softmax(-1);logq=second.log_softmax(-1)
            labels=torch.tensor([index[token] for token in model.tokenizer.encode_residues(reference)])
            row=dict(row_index=i,swapped_heavy_row=(i+250)%500,length=len(reference),
                max_abs_logit_change=float((first-second).abs().max()),
                mean_total_variation=float((logp.exp()-logq.exp()).abs().sum(-1).mean()/2),
                mean_kl_original_to_swapped=float((logp.exp()*(logp-logq)).sum(-1).mean()),
                argmax_change_fraction=float(first.argmax(-1).ne(second.argmax(-1)).double().mean()),
                true_light_nll_original=float(-logp.gather(1,labels[:,None]).mean()),
                true_light_nll_swapped=float(-logq.gather(1,labels[:,None]).mean()),
                original_input_sha256=before,swapped_input_sha256=input_fingerprint(other))
            rows.append(row)
    report=dict(status='complete',model=args.model,precision='cpu_float32',checkpoint=identity,
        rows_checked=len(rows),rows_with_changed_logits=sum(r['max_abs_logit_change']>1e-6 for r in rows),
        mean_total_variation=sum(r['mean_total_variation'] for r in rows)/len(rows),
        mean_argmax_change_fraction=sum(r['argmax_change_fraction'] for r in rows)/len(rows),
        scope='First 16 real input pairs, one forward per condition, no sampling or training; dependency alone does not establish useful conditioning.',rows=rows)
    out=gen/'pairing_sampler_debug_20261014'/f'{args.model}_conditioning_audit.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','checkpoint']},indent=2))


if __name__=='__main__':
    main()
