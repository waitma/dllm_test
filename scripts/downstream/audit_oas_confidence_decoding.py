"""Test four-step confidence mechanisms on the fixed128 OAS screen antibodies.

Activate pllm/protenix_abtcr; from repository root run:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/downstream/audit_oas_confidence_decoding.py --out ABS_NEW_DIRECTORY
Reuse the pinned AirGen sampler unchanged. Compare parent/raw, CE1/raw,
CE1/log-probability and CE1 with oracle parent log-partition offsets. The oracle
uses a second model and is diagnostic only, never a final-model performance claim.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    root=ROOT/'output/diagnostics'
    source=root/'final_layer_preservation_pilot_20261018/screen.jsonl'
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows)==128
    prior=root/'final_layer_cdr_span_preservation_v2_20261018'
    assert json.loads((prior/'verification.json').read_text())['status']=='passed'
    assert sha(source)==json.loads((prior/'plan.json').read_text())['screen_sha256']
    cases={key:[] for key in ['CDR1','CDR2','CDR3']}
    for i,row in enumerate(rows):
        hi=row['chain_roles'].index('antibody_heavy');li=row['chain_roles'].index('antibody_light')
        heavy,light=row['chains'][hi],row['chains'][li]
        assert len(heavy)<=148 and len(light)<=126
        reg=row['regions'][str(hi)];offset=0
        for key in ['FR1','CDR1','FR2','CDR2','FR3','CDR3','FR4']:
            if key.startswith('CDR'):
                cases[key].append(dict(heavy=heavy,light=light,gold_h=heavy,gold_l=light,
                    mask_h=list(range(offset,offset+len(reg[key]))),mask_l=[],row=i,key=str(i)))
            offset+=len(reg[key])
        assert offset==len(heavy)
    parent_root=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013'
    parent_script=parent_root/'airgen_evaluate.py'
    spec=importlib.util.spec_from_file_location('oas_confidence_airgen',parent_script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from downstream.fixed_chain_eval import load,airgen_inputs
    from downstream.final_layer_preservation_eval import load as load_student
    parent,parent_identity=load(parent_root/'checkpoint')
    student,student_identity=load_student(root/'final_layer_joint_convergence_pilot_20261018/token_all_direction')
    variants=['parent_raw','student_raw','student_log_probability','student_teacher_partition_oracle']
    status=dict(status='running',stage='prepare',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',status)
    plan=dict(created_utc=status['started_utc'],variants=variants,parent_identity=parent_identity,
        student_identity=student_identity,source=str(source),source_sha256=sha(source),steps=4,batch_size=16,
        sampler='pinned AirGen methods; confidence-deterministic-linear, threshold ties unchanged',
        intervention='only a shared offset to the logits of each position; all allowed token argmax predictions verified unchanged for the same context',
        oracle='parent evaluated on the current student context; replace student log partition with parent log partition; requires teacher at inference',
        scope='OAS128 screen diagnostic, all three heavy CDRs; not Kong score or deployable oracle method')
    write(out/'plan.json',plan)
    controls={name:np.load(prior/f'{key}_predictions.npz') for name,key in [('parent_raw','parent'),('student_raw','ce1_5000')]}
    results={name:{} for name in variants};all_predictions=[];preflight=[];offset_calls=0
    for region,items in cases.items():
        for name in variants:
            model=parent if name=='parent_raw' else student
            scores=[]
            for start in range(0,len(items),16):
                batch_rows=items[start:start+16]
                features,virtual,gold,target,maps=airgen_inputs(model,batch_rows)
                features={k:v.cuda() if torch.is_tensor(v) else v for k,v in features.items()}
                virtual={k:v.cuda() if torch.is_tensor(v) else v for k,v in virtual.items()}
                gold,target=gold.cuda(),target.cuda()
                bridge=module.decoder(model,features,maps)
                allowed=[i for i in range(model.tokenizer.vocab_size) if i not in bridge.special_token_list]
                if name in controls and start==0:
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        original=bridge.net(**virtual)['logits'][target].float().cpu().numpy()
                    cache=controls[name];prefix=region+'__span'
                    expected=cache[prefix+'_logits'][cache[prefix+'_sequence_index']<16]
                    np.testing.assert_array_equal(original,expected)
                    preflight.append(dict(model=name,region=region,prior_single_forward_logits_bitwise=True))
                if name not in controls:
                    original_net=bridge.net
                    parent_net=module.decoder(parent,features,maps).net
                    def corrected_net(input_ids,chain_ids):
                        nonlocal offset_calls
                        output=original_net(input_ids,chain_ids)
                        logits=output['logits'].float()
                        z=torch.logsumexp(logits[...,allowed],dim=-1,keepdim=True)
                        if name=='student_log_probability':
                            offset=-z
                        else:
                            parent_logits=parent_net(input_ids,chain_ids)['logits'].float()
                            offset=torch.logsumexp(parent_logits[...,allowed],dim=-1,keepdim=True)-z
                        corrected=logits+offset
                        assert torch.equal(logits[...,allowed].argmax(-1),corrected[...,allowed].argmax(-1))
                        assert torch.isfinite(offset).all()
                        offset_calls+=1
                        return {**output,'logits':corrected}
                    bridge.net=corrected_net
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction,_=bridge.generate(virtual,max_iter=4,temperature=1.,
                        partial_masks=~target & virtual['input_ids'].ne(bridge.bos_id),sampling_strategy='argmax',cfg_scale=0.)
                assert torch.equal(prediction[~target],virtual['input_ids'][~target])
                for j,row in enumerate(batch_rows):
                    reference=gold[j,target[j]];pred=prediction[j,target[j]]
                    aar=float((reference==pred).float().mean());scores.append(aar)
                    all_predictions.append(dict(variant=name,region=region,row=row['row'],
                        reference_ids=reference.cpu().tolist(),prediction_ids=pred.cpu().tolist(),aar=aar))
                status.update(stage='sampling',variant=name,region=region,completed=start+len(batch_rows))
                write(out/'status.json',status)
            results[name][region]=dict(n=len(scores),mean_sequence_aar=float(np.mean(scores)))
            write(out/'results.json',results);print(name,region,results[name][region],flush=True)
    with (out/'predictions.jsonl').open('w') as stream:
        for row in all_predictions:stream.write(json.dumps(row)+'\n')
    for name in variants:
        for region in cases:
            selected=[x for x in all_predictions if x['variant']==name and x['region']==region]
            assert [x['row'] for x in selected]==list(range(128))
            rescored=[sum(a==b for a,b in zip(x['reference_ids'],x['prediction_ids']))/len(x['reference_ids']) for x in selected]
            assert abs(np.mean(rescored)-results[name][region]['mean_sequence_aar'])<1e-7
    assert sha(source)==plan['source_sha256']
    write(out/'verification.json',dict(status='passed',single_forward_controls=preflight,
        offset_forward_calls=offset_calls,allowed_argmax_unchanged_on_every_corrected_forward=True,
        context_unchanged=True,predictions_rescored=True,source_sha256={str(p):sha(p) for p in [Path(__file__),source,parent_script,ROOT/'downstream/fixed_chain_eval.py']}))
    status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',status)


if __name__=='__main__':main()
