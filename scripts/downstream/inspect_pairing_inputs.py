"""Activate pllm/protenix_abtcr; python -B SCRIPT --model fixed50000|v7.

Audit all 500 actual adapter inputs using bound checkpoint implementations on
CPU. Save an input/output example and structural differences without sampling.
"""
import argparse
from collections import Counter
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
        original=gen/'v7_f011_step50082_pairing_local_gpu_20261006/holdout500_n8.csv'
    else:
        from downstream.fixed_chain_eval import load
        model,identity=load(gen/'fixed_chain_75m_step50000_20261013/checkpoint','cpu')
        from scripts.downstream.run_fixed_chain_pairing_reference import reference_input
        adapter=reference_input
        original=gen/'fixed_chain_75m_step50000_pairing_reference_20261013/generated.csv'
    from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
    from downstream.v7_native import input_fingerprint
    frame=pd.read_csv(ROOT/'data/oas_previous_clean/splits/holdout_select_500_eval_OAS.csv')
    outputs=pd.read_csv(original)
    fingerprints=[];example=None
    for i,row in frame.iterrows():
        heavy=row.h_sequence.replace('-','');length=len(row.l_sequence.replace('-',''))
        features,target=adapter(model,heavy,length)[:2]
        assert set(features)=={'input_ids','attention_mask','role_ids','scope_ids','chain_ids','position_ids','target_slot_mask'}
        ids=features['input_ids'];pos=features['position_ids'];role=features['role_ids'];attention=features['attention_mask']
        h=(role==CHAIN_ROLE_TO_ID['antibody_heavy']) & (pos>=1) & (pos<=len(heavy))
        light=(role==CHAIN_ROLE_TO_ID['antibody_light'])
        assert ids[h].tolist()==model.tokenizer.encode_residues(heavy) and attention[h].all()
        assert torch.equal(target,light & (pos>=1) & (pos<=length) & attention)
        assert int(target.sum())==length and ids[target].eq(model.tokenizer.mask_token_id).all()
        terminal=light & (pos==length+1)
        assert terminal.sum()==1 and ids[terminal].item()==model.terminal_id and attention[terminal].all()
        assert not target[h].any() and not target[terminal].any()
        fingerprints.append(input_fingerprint(features))
        if i==0:
            chains={}
            for chain_role in ['antibody_heavy','antibody_light']:
                m=role==CHAIN_ROLE_TO_ID[chain_role]
                end=len(heavy) if chain_role=='antibody_heavy' else length
                tail=m & (pos>end+1)
                chains[chain_role]=dict(slots=int(m.sum()),attended=int(attention[m].sum()),
                    tail_slots=int(tail.sum()),tail_attended=int(attention[tail].sum()),
                    tail_tokens=dict(Counter(model.tokenizer.token(x) for x in ids[tail].tolist())),
                    target_slot_flags=int(features['target_slot_mask'][m].sum()))
            example=dict(heavy_sequence=heavy,reference_length_condition=length,
                true_light_residues_passed=False,input_shape=list(ids.shape),chains=chains,
                leading_tokens=[model.tokenizer.token(x) for x in ids[0,:12].tolist()],
                terminal_token=model.tokenizer.token(model.terminal_id),
                first_historical_generated_light=outputs.iloc[0].gen_l_sequence)
    report=dict(status='passed',records=len(frame),model=args.model,identity=identity,
        checks=['heavy_residues_exact','all_light_targets_masked','known_terminal_visible',
                'target_positions_exact','no_labels_or_reference_residues_in_model_features'],
        input_fingerprints=fingerprints,example=example)
    out=gen/'pairing_sampler_debug_20261014'/f'{args.model}_input_audit.json'
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(example,indent=2))


if __name__=='__main__':
    main()
