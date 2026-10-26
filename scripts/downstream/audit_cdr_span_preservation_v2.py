"""Compare contiguous CDR and count-matched random masks on held-out OAS.

Activate pllm/protenix_abtcr; from repository root run:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/downstream/audit_cdr_span_preservation_v2.py --out ABS_NEW_DIRECTORY
Use the existing128 OAS screen records, native attention and frozen checkpoints.
This is a one-forward diagnostic, not Kong multi-step AAR or a training run.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    root=ROOT/'output/diagnostics'
    source=root/'final_layer_preservation_pilot_20261018/screen.jsonl'
    manifest=json.loads((source.parent/'data.json').read_text())
    assert sha(source)==manifest['files']['screen']['sha256']
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows)==128
    order=['FR1','CDR1','FR2','CDR2','FR3','CDR3','FR4']
    spans=[]
    for row in rows:
        i=row['chain_roles'].index('antibody_heavy');regions=row['regions'][str(i)]
        assert ''.join(regions[k] for k in order)==row['chains'][i]
        current={};offset=0
        for key in order:
            if key.startswith('CDR'):
                assert regions[key]
                current[key]=list(range(offset,offset+len(regions[key])))
            offset+=len(regions[key])
        spans.append(current)
    checkpoints=dict(parent=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint',
        token1700=root/'final_layer_token_direction_pilot_20261018/token_all_direction',
        ce0_5000=root/'final_layer_convergence_pilot_20261018/token_all_direction',
        ce1_5000=root/'final_layer_joint_convergence_pilot_20261018/token_all_direction')
    status=dict(status='running',stage='prepare',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',status)
    from downstream.fixed_chain_eval import load as load_parent,FixedChainBackbone
    from downstream.final_layer_preservation_eval import load as load_adapted
    model,identity=load_parent(checkpoints['parent'])
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord,CHAIN_ROLE_TO_ID
    records=[BioSeqRecord.from_prepared_dict(row) for row in rows]
    backbone=FixedChainBackbone(model,identity)
    batches=[];views={}
    for start in range(0,len(rows),16):
        batch=backbone.collator(records[start:start+16])
        selections={}
        for region in ['CDR1','CDR2','CDR3']:
            for kind in ['span','random_matched_count']:
                selected=torch.zeros_like(batch['input_ids'],dtype=torch.bool)
                for j,row in enumerate(rows[start:start+16]):
                    hi=row['chain_roles'].index('antibody_heavy')
                    heavy=(batch['role_ids'][j]==CHAIN_ROLE_TO_ID['antibody_heavy']) & batch['residue_mask'][j]
                    positions=heavy.nonzero().flatten()
                    assert len(positions)==len(row['chains'][hi])
                    assert batch['input_ids'][j,positions].tolist()==model.tokenizer.encode_residues(row['chains'][hi])
                    idx=spans[start+j][region]
                    if kind=='random_matched_count':
                        rng=np.random.default_rng(20261018+1000*(start+j)+int(region[-1]))
                        idx=rng.choice(len(positions),len(idx),replace=False).tolist()
                    selected[j,positions[idx]]=True
                name=region+'/'+kind
                assert not (selected & ~batch['residue_mask']).any()
                selections[name]=selected
                views.setdefault(name,[]).extend([m.nonzero().flatten().tolist() for m in selected])
        batches.append((batch,selections))
    write(out/'selected_native_positions.json',views)
    plan=dict(created_utc=status['started_utc'],screen=str(source),screen_sha256=sha(source),
              models={k:str(v) for k,v in checkpoints.items()},views=list(views),batch_size=16,
              precision='BF16 autocast',attention='native in all conditions',steps=1,
              kl_definition='float64 KL on verified common finite native vocabulary support; excluded classes have exactly zero probability',
              comparison='same heavy-chain masked count; contiguous annotated CDR versus random heavy residues',
              limitations='Targets differ between span and random masks; not a position-matched causal proof. Held-out OAS diagnostic only, no training or benchmark replacement.')
    write(out/'plan.json',plan)
    allowed=model.tokenizer.encode_residues('ACDEFGHIKLMNPQRSTVWY')+[model.tokenizer.special_id('<protd>'),model.tokenizer.chain_separator_id()]
    parent_head={k:v.detach().cpu().clone() for k,v in model.backbone.transformer.ff_out.state_dict().items()}
    parent_outputs={};results={};identities={}
    for name,path in checkpoints.items():
        if name!='parent':
            del model,backbone
            torch.cuda.empty_cache()
            assert json.loads((path.parent/'status.json').read_text())['status']=='success'
            model,identity=load_adapted(path)
            for key,value in model.backbone.transformer.ff_out.state_dict().items():
                torch.testing.assert_close(value.cpu(),parent_head[key],rtol=0,atol=0)
        backbone=FixedChainBackbone(model,identity);identities[name]=identity
        collected={key:dict(logits=[],gold=[],sequence_index=[]) for key in views}
        with torch.no_grad():
            for batch_index,(cpu_batch,selections) in enumerate(batches):
                batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in cpu_batch.items()}
                for view,selected_cpu in selections.items():
                    selected=selected_cpu.cuda();gold=batch['input_ids'][selected]
                    noised=batch['input_ids'].clone();noised[selected]=model.tokenizer.mask_token_id
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits,_=model.denoise(**{**model.feature_args(batch),'input_ids':noised})
                    collected[view]['logits'].append(logits[selected].float().cpu())
                    collected[view]['gold'].append(gold.cpu())
                    ids=torch.arange(batch_index*16,batch_index*16+len(selected_cpu))[:,None].expand_as(selected_cpu)
                    collected[view]['sequence_index'].append(ids[selected_cpu])
                status.update(stage='inference',model=name,completed=min((batch_index+1)*16,len(rows)))
                write(out/'status.json',status)
        results[name]={};arrays={}
        for view,parts in collected.items():
            logits=torch.cat(parts['logits']);gold=torch.cat(parts['gold']);ids=torch.cat(parts['sequence_index'])
            filtered=logits.clone();keep=torch.zeros(logits.shape[1],dtype=torch.bool);keep[allowed]=True;filtered[:,~keep]=-float('inf')
            pred=filtered.argmax(-1)
            seq_acc=[float((pred[ids==i]==gold[ids==i]).float().mean()) for i in range(len(rows))]
            if name=='parent':parent_outputs[view]=dict(logits=logits,gold=gold,ids=ids,pred=pred)
            reference=parent_outputs[view]
            assert torch.equal(gold,reference['gold']) and torch.equal(ids,reference['ids'])
            support=torch.isfinite(reference['logits'])
            assert torch.equal(support,torch.isfinite(logits))
            assert torch.equal(support,support[0:1].expand_as(support))
            assert torch.isneginf(logits[~support]).all()
            # Native denoise hard-masks24 vocabulary entries to negative inf.
            # KL on that common zero-probability support otherwise creates0*inf.
            active=support[0]
            log_p=F.log_softmax(logits[:,active].double(),dim=-1)
            target_p=F.softmax(reference['logits'][:,active].double(),dim=-1)
            kl=F.kl_div(log_p,target_p,reduction='batchmean')
            assert torch.isfinite(kl) and float(kl)>-1e-10
            kl=kl.clamp_min(0.)
            results[name][view]=dict(n_sequences=len(rows),n_masked=len(gold),ce_all_vocab=float(F.cross_entropy(logits,gold)),
                mean_sequence_aar=float(np.mean(seq_acc)),token_accuracy=float((pred==gold).float().mean()),
                parent_argmax_agreement=float((pred==reference['pred']).float().mean()),kl_parent_to_student=float(kl))
            prefix=view.replace('/','__')
            arrays[prefix+'_logits']=logits.numpy();arrays[prefix+'_gold']=gold.numpy();arrays[prefix+'_sequence_index']=ids.numpy()
        np.savez_compressed(out/f'{name}_predictions.npz',**arrays)
        write(out/'results.json',results)
        print(name,results[name],flush=True)
    write(out/'model_identities.json',identities)
    assert sha(source)==plan['screen_sha256']
    write(out/'verification.json',dict(status='passed',all128_screen_regions_reconstructed=True,
          native_heavy_positions_match_original_sequence=True,all_selected_positions_observed=True,
          all_models_same_inputs_and_masks=True,original_vocabulary_head_bitwise_unchanged=True,common_finite_logit_support_verified=True,
          source_sha256={str(p):sha(p) for p in [Path(__file__),source,ROOT/'downstream/fixed_chain_eval.py',ROOT/'downstream/final_layer_preservation_eval.py']}))
    status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',status)


if __name__=='__main__':main()
