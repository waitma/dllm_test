"""Match a fixed rotation/scale of teacher features in the actual final layer.

Activate pllm/protenix_abtcr; from repository root run:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/research/train_final_layer_coordinate_pilot.py --out ABS_NEW_DIRECTORY
This follows the relational pilot with the same data, updates, optimizer and CE.
The pooled CKA term is replaced by fixed-coordinate MSE. No inference head.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha,prepare_data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'output/diagnostics/final_layer_preservation_pilot_20261018'
    assert json.loads((prior/'status.json').read_text())['status']=='success'
    groups=prepare_data(out)
    for split in ('train','screen'):assert sha(out/f'{split}.jsonl')==sha(prior/f'{split}.jsonl')
    schedule=json.loads((prior/'schedule.json').read_text());assert len(schedule)==300
    checkpoint=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    files=[Path(__file__),ROOT/'scripts/research/train_final_layer_preservation_pilot.py',
           ROOT/'downstream/fixed_chain_eval.py',ROOT/'downstream/grammar/ab_probes.py']
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(checkpoint),
          parent_experiment=str(prior),data_identical=True,schedule_sha256=sha(prior/'schedule.json'),
          steps=300,batch_size=16,seed=42,learning_rate=2e-5,warmup=20,optimizer='same AdamW and grad clip1 as relational pilot',
          trainable='blocks3-8 and final RMSNorm; everything else frozen',
          calibration='4096 training sequences only: fit orthogonal Procrustes R between centered parent block2 and final means',
          target='normalized student (s-mu_s)/rms_s matches ((teacher-mu_t)/rms_t) @ R; fixed train means/scalars/R',
          objective='native CE + pooled coordinate MSE +0.1*token relation + same pooled variance floor',
          evaluation='native actual final RMSNorm hidden only, fixed step300; no projection/rotation at inference',
          source_sha256={str(p):sha(p) for p in files})
    write(out/'plan.json',plan)
    status=dict(status='running',stage='calibration',pid=os.getpid(),host=socket.gethostname(),started_utc=plan['created_utc'])
    write(out/'status.json',status)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from downstream.fixed_chain_eval import load,FixedChainBackbone
    from downstream.fixed_chain_layer_eval import block_hidden
    torch.set_num_threads(4)
    model,identity=load(checkpoint,device='cuda')
    from dllm.pipelines.immune_llada.fixed_chain.layout import FixedChainCollator
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain
    reference={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
    prefixes=tuple(f'backbone.transformer.blocks.{i}.' for i in range(2,8))+('backbone.transformer.ln_f.',)
    train_names={n for n,_ in model.named_parameters() if n.startswith(prefixes)}
    write(out/'identity.json',{**identity,'trainable_names':sorted(train_names)})
    backbone=FixedChainBackbone(model,identity)
    collator=FixedChainCollator.from_config(model.config,mode='train')
    def device(batch):return {k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
    def pool(h,m):return h.float().masked_fill(~m[...,None],0).sum(1)/m.sum(1,keepdim=True)
    def clean(rows,grad=False):
        batch=device(backbone.collator([BioSeqRecord.from_prepared_dict(r) for r in rows]))
        teacher=[]
        handle=model.backbone.transformer.blocks[1].register_forward_hook(
            lambda module,inputs,result:teacher.append(block_hidden(result).detach()))
        try:
            with torch.set_grad_enabled(grad),torch.autocast('cuda',dtype=torch.bfloat16):
                _,hidden=model.denoise(**model.feature_args(batch),output_hidden_states=True)
        finally:handle.remove()
        assert len(teacher)==1 and not teacher[0].requires_grad
        mask=batch['residue_mask'] & batch['attention_mask']
        return hidden,teacher[0],mask
    def relation(s,t):
        def gram(x):
            x=x.float()-x.float().mean(-2,keepdim=True)
            return x@x.transpose(-1,-2)
        ks,kt=gram(s),gram(t.detach())
        return (1-(ks*kt).sum((-1,-2))/(ks.square().sum((-1,-2))*kt.square().sum((-1,-2))).clamp_min(1e-20).sqrt()).mean()
    students,teachers=[],[]
    for start in range(0,len(groups['train']),16):
        h,t,m=clean(groups['train'][start:start+16])
        students.append(pool(h,m).cpu());teachers.append(pool(t,m).cpu())
        if start%512==0:print('calibration',start,len(groups['train']),flush=True)
    s,t=torch.cat(students).double(),torch.cat(teachers).double()
    mu_s,mu_t=s.mean(0),t.mean(0)
    rms_s,rms_t=(s-mu_s).square().mean().sqrt(),(t-mu_t).square().mean().sqrt()
    ts,ss=(t-mu_t)/rms_t,(s-mu_s)/rms_s
    u,_,vh=torch.linalg.svd(ts.T@ss,full_matrices=False)
    rotation=u@vh
    error=float((rotation.T@rotation-torch.eye(rotation.shape[0])).abs().max())
    assert error<1e-10
    np.savez_compressed(out/'calibration.npz',student=s.numpy(),teacher=t.numpy(),mu_s=mu_s.numpy(),mu_t=mu_t.numpy(),
                        rms_s=rms_s.numpy(),rms_t=rms_t.numpy(),rotation=rotation.numpy())
    write(out/'calibration.json',dict(n=len(s),orthogonality_max_error=error,
          initial_normalized_mse=float((ss-ts@rotation).square().mean()),rms_s=float(rms_s),rms_t=float(rms_t),
          artifact_sha256=sha(out/'calibration.npz'),labels_used=False))
    mu_s,mu_t,rms_s,rms_t,rotation=[x.float().to('cuda') for x in (mu_s,mu_t,rms_s,rms_t,rotation)]
    del students,teachers,s,t,ts,ss,u,vh,h,m
    def alignment(h,t,m):
        sp,tp=pool(h,m),pool(t,m)
        loss=(((sp-mu_s)/rms_s)-((tp-mu_t)/rms_t)@rotation).square().mean()
        st,tt=[],[]
        for i in range(len(h)):
            available=m[i].nonzero().flatten()
            idx=available[torch.linspace(0,len(available)-1,32,device='cuda').long()]
            st.append(h[i,idx]);tt.append(t[i,idx])
        token=relation(torch.stack(st),torch.stack(tt))
        sv=(sp-sp.mean(0)).square().mean();tv=(tp-tp.mean(0)).square().mean().detach()
        floor=F.relu(.25*tv-sv)/(tv+1e-12)
        return loss+.1*token+floor,dict(coordinate=float(loss.detach()),token=float(token.detach()),variance_penalty=float(floor.detach()))
    def validate(step):
        model.eval();student,teacher=[],[]
        with torch.no_grad():
            for start in range(0,len(groups['screen']),16):
                h,t,m=clean(groups['screen'][start:start+16])
                student.append(pool(h,m));teacher.append(pool(t,m))
        sp,tp=torch.cat(student),torch.cat(teacher)
        metrics=dict(step=step,coordinate_mse=float((((sp-mu_s)/rms_s)-((tp-mu_t)/rms_t)@rotation).square().mean()),
                     heldout_pool_cka=float(1-relation(sp,tp)))
        for ratio in (.5,1.):
            total,ce_count,correct,aa_count=0.,0,0,0
            with torch.random.fork_rng(devices=[0]),torch.no_grad():
                torch.manual_seed(87913)
                for start in range(0,len(groups['screen']),16):
                    batch=device(collator(groups['screen'][start:start+16]))
                    noised,targets,selected=model.corrupt(batch,torch.full_like(batch['input_ids'],ratio,dtype=torch.float))
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits,_=model.denoise(**{**model.feature_args(batch),'input_ids':noised})
                    total+=float(F.cross_entropy(logits[selected].float(),targets[selected],reduction='sum'));ce_count+=int(selected.sum())
                    aa=selected & batch['residue_mask'];correct+=int((logits.argmax(-1)[aa]==targets[aa]).sum());aa_count+=int(aa.sum())
            metrics[f'ce_{ratio}']=total/ce_count;metrics[f'aa_accuracy_{ratio}']=correct/aa_count
        model.train();return metrics
    folder=out/'joint_coordinate';folder.mkdir()
    history=[validate(0)];write(folder/'validation.json',history)
    old=json.loads((prior/'baseline_validation.json').read_text())
    for name in ('heldout_pool_cka','ce_0.5','ce_1.0','aa_accuracy_0.5','aa_accuracy_1.0'):
        assert abs(history[0][name]-old[name])<1e-6,(name,history[0][name],old[name])
    for n,p in model.named_parameters():p.requires_grad_(n in train_names)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,weight_decay=.01)
    status.update(stage='train',step=0);write(out/'status.json',status)
    started=time.monotonic()
    for step,indices in enumerate(schedule,1):
        rows=[groups['train'][i] for i in indices];torch.manual_seed(91000+step)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=2e-5*min(step/20,1.)
        batch=device(collator(rows))
        with torch.autocast('cuda',dtype=torch.bfloat16):ce=model(**batch)['loss']
        ce.backward()
        h,t,m=clean(rows,True);rep,detail=alignment(h,t,m);rep.backward()
        grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        assert torch.isfinite(ce) and torch.isfinite(rep) and torch.isfinite(grad)
        optimizer.step()
        if step==1 or step%10==0:
            row=dict(step=step,ce=float(ce.detach()),representation_loss=float(rep.detach()),gradient_norm=float(grad),
                     seconds=time.monotonic()-started,**detail)
            with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            status.update(step=step,last_train=row);write(out/'status.json',status);print(row,flush=True)
        if step%100==0:history.append(validate(step));write(folder/'validation.json',history)
    changed=[]
    for n,p in model.named_parameters():
        equal=torch.equal(p.detach().cpu(),reference[n])
        if n not in train_names:assert equal,n
        elif not equal:changed.append(n)
    assert changed
    save_file({n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if n in train_names},
              str(folder/'trained_parameters.safetensors'),metadata={'parent_sha256':identity['model_sha256'],'arm':'joint_coordinate'})
    write(folder/'checkpoint.json',dict(parent_checkpoint=str(checkpoint),parent_identity=identity,
          parameter_file=str(folder/'trained_parameters.safetensors'),sha256=sha(folder/'trained_parameters.safetensors'),
          trainable_names=sorted(train_names),changed_names=changed,steps=300,
          frozen_parameters_bitwise_equal=True,source_sha256=plan['source_sha256']))
    model.eval().requires_grad_(False)
    status.update(stage='gdp');write(out/'status.json',status)
    from downstream.grammar.ab_probes import prepare_task,run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL
    from scipy.stats import spearmanr
    from types import SimpleNamespace
    df,pairs,_,_=prepare_task('gdp_a1')
    records=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],task_type='antibody',source='native_v5_probe') for h,l in pairs]
    x=backbone.vectors(records);np.save(folder/'final_features.npy',x)
    gdpfolder=folder/'gdp_a1';gdpfolder.mkdir()
    metrics=run_gdp(df,torch.from_numpy(x.astype(np.float64)),SimpleNamespace(out_dir=gdpfolder))
    write(gdpfolder/'metrics.json',metrics)
    checks=[]
    for col in ALL_COLS:
        saved=np.load(gdpfolder/col/'legacy_oof.npz');valid=df[col].notna().to_numpy()
        np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid));np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL])
        np.testing.assert_array_equal(saved['y_true'],df.loc[valid,col])
        scores=[float(spearmanr(saved['y_transformed'][saved['fold']==i],saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
        assert abs(np.mean(scores)-metrics['properties'][col]['legacy_reference']['cv_mean_spearman'])<1e-12
        checks.append(dict(property=col,fold_scores=scores))
    del optimizer,model,backbone,h,t,m,batch,reference
    torch.cuda.empty_cache()
    from downstream.final_layer_preservation_eval import load as reload
    restored,restored_identity=reload(folder)
    recovered=FixedChainBackbone(restored,restored_identity).vectors(records)
    np.testing.assert_array_equal(recovered,x)
    for p,digest in plan['source_sha256'].items():assert sha(p)==digest,p
    write(folder/'verification.json',dict(status='passed',oof_checks=checks,reloaded_features_bitwise_equal=True,
          frozen_parameters_bitwise_equal=True,orthogonal_target=True,original_baseline_validation_reproduced=True))
    write(out/'results.json',dict(validation=history[-1],gdp=metrics))
    status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',status)


if __name__=='__main__':main()
