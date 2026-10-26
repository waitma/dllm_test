"""Extract frozen sequence features for the experimental structure task.

After activating pllm:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/extract_stage1_structure_features.py --output ABS_STRUCTURE_DIRECTORY --arm mlm
Repeat with --arm pair_kd. Uses CPU FP32, all selected structures, and the full
checkpoint-specific forward. Geometry and labels are never passed to the model.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from evaluate_stage1_partner_retrieval import model_and_collator
from evaluate_stage1_frozen_gdp import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['mlm','pair_kd'],required=True)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args();out=args.output;assert out.is_absolute()
    plan=json.loads((out/'geometry_plan.json').read_text())
    for name,digest in plan['files'].items():assert sha(out/name)==digest
    assert sha(out/'inventory_plan.json')==plan['inventory_plan_sha256']
    target=out/(args.arm+'_features.npz')
    if target.exists():raise FileExistsError(target)
    records=json.loads((out/'structures.json').read_text())
    model,collator=model_and_collator(plan,args.arm,'cpu',args.threads)
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain
    start=time.monotonic();next_log=0;result=[]
    with torch.inference_mode():
        for offset in range(0,len(records),args.batch_size):
            subset=records[offset:offset+args.batch_size]
            seqs=[BioSeqRecord([BioSeqChain(r['h'],'antibody_heavy'),BioSeqChain(r['l'],'antibody_light')],
                               task_type='antibody',source='stage1_experimental_structure_eval') for r in subset]
            batch=collator(seqs)
            mask=batch['residue_mask'].bool()&batch['attention_mask'].bool()
            assert mask.sum(1).tolist()==[len(r['h'])+len(r['l']) for r in subset]
            assert not (mask&batch['terminal_mask'].bool()).any()
            assert (batch['input_ids'][batch['relation_mask']]==model.tokenizer.mask_token_id).all()
            features=model.feature_args(batch)
            _,hidden=model.denoise(**features,output_hidden_states=True)
            mean=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
            assert torch.isfinite(mean).all()
            if offset==0:
                _,single=model.denoise(**{k:v[:1] for k,v in features.items()},output_hidden_states=True)
                single=(single*mask[:1].unsqueeze(-1)).sum(1)/mask[:1].sum(1,keepdim=True)
                torch.testing.assert_close(single,mean[:1],atol=2e-5,rtol=2e-4)
                error=float((single-mean[:1]).abs().max())
            result.append(mean.numpy())
            if time.monotonic()>=next_log or offset+args.batch_size>=len(records):
                print(json.dumps(dict(phase='features',arm=args.arm,done=min(offset+args.batch_size,len(records)),
                                     total=len(records),seconds=time.monotonic()-start)),flush=True)
                next_log=time.monotonic()+60
    values=np.concatenate(result)
    assert values.shape==(len(records),model.config.d_model)
    assert sha(Path(plan['arms'][args.arm]['checkpoint'])/'model.safetensors')==plan['arms'][args.arm]['model_sha256']
    np.savez_compressed(target,features=values,row_ids=np.array([r['instance'] for r in records],dtype=str))
    write(out/(args.arm+'_features_receipt.json'),dict(status='complete',shape=list(values.shape),
        features_sha256=sha(target),plan_sha256=sha(out/'geometry_plan.json'),script_sha256=sha(__file__),
        model_sha256=plan['arms'][args.arm]['model_sha256'],single_batch_max_abs=error,
        device='cpu',dtype='float32',trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        counts_verified=True,relation_hidden=True,full_pair_forward=True,seconds=time.monotonic()-start))


if __name__=='__main__':main()
