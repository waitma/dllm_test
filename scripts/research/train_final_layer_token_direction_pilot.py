"""Compare sparse and dense token-direction preservation at fixed training budget.

Activate pllm then protenix_abtcr; run from the repository root:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/research/train_final_layer_token_direction_pilot.py --out ABS_NEW_DIRECTORY
Both arms start at spectral1000, restore identical AdamW moments, and run the
same 700 updates at LR2e-4. Replace .1*token Gram relation with .1*centered token
cosine error using the existing OAS-only teacher-to-student rotation. One arm
uses the original32 positions; the other uses all genuine residues. Keep pooled
spectral supervision, CE0, no-mask student, native teacher, and frozen ranges.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'output/diagnostics/final_layer_coordinate_pilot_20261018'
    data_root=ROOT/'output/diagnostics/final_layer_preservation_pilot_20261018'
    spectral_parent=ROOT/'output/diagnostics/final_layer_spectral_pilot_20261018'
    initial=spectral_parent/'spectral_continue'
    assert json.loads((spectral_parent/'status.json').read_text())['status']=='success'
    side=json.loads((initial/'checkpoint.json').read_text())
    groups={name:[json.loads(line) for line in (data_root/f'{name}.jsonl').read_text().splitlines()] for name in ('train','screen')}
    data_manifest=json.loads((data_root/'data.json').read_text())
    for name in groups:assert sha(data_root/f'{name}.jsonl')==data_manifest['files'][name]['sha256']
    rng=random.Random(42);schedule=[]
    while len(schedule)<1700:
        order=list(range(len(groups['train'])));rng.shuffle(order)
        schedule.extend([order[i:i+16] for i in range(0,len(order),16)])
    assert schedule[:300]==json.loads((data_root/'schedule.json').read_text())
    assert schedule[300:1000]==json.loads((spectral_parent/'schedule.json').read_text())
    schedule=schedule[1000:1700];write(out/'schedule.json',schedule)
    sources=[Path(__file__),ROOT/'scripts/research/train_final_layer_preservation_pilot.py',
             ROOT/'downstream/final_layer_preservation_eval.py',ROOT/'downstream/fixed_chain_eval.py',ROOT/'downstream/grammar/ab_probes.py']
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),initial_checkpoint=str(initial),
          initial_parameter_sha256=side['sha256'],parent_checkpoint=side['parent_checkpoint'],
          arms=['token32_direction','token_all_direction'],initial_steps=1000,additional_steps=700,total_steps=1700,
          data_manifest=str(data_root/'data.json'),data_sha256=sha(data_root/'data.json'),schedule_sha256=sha(out/'schedule.json'),
          seed=42,batch_size=16,lr=2e-4,warmup_after_restart=0,
          optimizer='restore identical spectral1000 AdamW moments; clip1, constant LR increased tenfold; same weight_decay=.01',
          initial_optimizer_sha256=sha(initial/'optimizer.pt'),
          trainable='same blocks3-8 and final RMSNorm; early teacher and vocabulary head remain frozen',
          loss='spectral pooled coordinate error +0.1 centered token directional cosine error + variance floor; native CE monitored without gradients',
          intervention='replace token Gram constraint with fixed-basis token direction alignment; compare original32 positions versus all real residues; same loss coefficient but not matched initial gradient magnitude',
          direction='center each sequence over all real residues, then compare student token directions with frozen teacher token directions rotated by the existing OAS global rotation; no token-norm target',
          hypothesis='stronger token constraint may aid unseen-sequence pooling generalization; identical pooled means alone imply identical downstream input, so the old token symmetry is not by itself a causal proof',
          control=str(ROOT/'output/diagnostics/final_layer_optimization_pilot_20261018/representation_lr10'),
          ce_weight=0.0,
          spectral='OAS train PCA eigenvalues lambda; w_i=1/(lambda_i+lambda_64); normalized by mean(lambda_i*w_i)',
          calibration=str(prior/'calibration.npz'),calibration_sha256=sha(prior/'calibration.npz'),
          selection='fixed1700; both native/no-mask readouts and all five GDP properties; no intermediate selection',
          source_sha256={str(p):sha(p) for p in sources})
    write(out/'plan.json',plan)
    status=dict(status='running',stage='prepare',pid=os.getpid(),host=socket.gethostname(),started_utc=plan['created_utc'])
    write(out/'status.json',status)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from downstream.final_layer_preservation_eval import load
    from downstream.fixed_chain_eval import FixedChainBackbone
    from downstream.fixed_chain_layer_eval import block_hidden
    torch.set_num_threads(4)
    model,identity=load(initial)
    from dllm.pipelines.immune_llada.fixed_chain.layout import FixedChainCollator
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain
    reference={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
    train_names=set(side['trainable_names'])
    backbone=FixedChainBackbone(model,identity)
    collator=FixedChainCollator.from_config(model.config,mode='train')
    cal=np.load(prior/'calibration.npz')
    normalized=(cal['teacher']-cal['mu_t'])/cal['rms_t']
    _,singular,vh=np.linalg.svd(normalized,full_matrices=False)
    eigen=singular**2/len(normalized);floor=float(eigen[63]);weights=1/(eigen+floor)
    denominator=float(np.mean(eigen*weights))
    assert floor>0 and denominator>0 and np.isfinite(weights).all()
    np.savez_compressed(out/'spectrum.npz',eigenvalues=eigen,axes=vh,weights=weights)
    write(out/'spectrum.json',dict(floor_at_PC64=floor,normalizer=denominator,
          max_to_min_weight=float(weights.max()/weights.min()),fit_rows=len(normalized),
          training_target_weighted_variance=float(np.mean(((normalized@vh.T)**2)*weights)/denominator),
          artifact_sha256=sha(out/'spectrum.npz')))
    assert abs(float(np.mean(((normalized@vh.T)**2)*weights)/denominator)-1)<1e-8
    mu_s,mu_t,rms_s,rms_t,rotation=[torch.as_tensor(cal[k],device='cuda',dtype=torch.float32) for k in ['mu_s','mu_t','rms_s','rms_t','rotation']]
    axes=torch.tensor(vh.T,device='cuda',dtype=torch.float32);weight=torch.tensor(weights,device='cuda',dtype=torch.float32)
    def device(batch):return {k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
    def pool(h,m):return h.float().masked_fill(~m[...,None],0).sum(1)/m.sum(1,keepdim=True)
    def relation(s,t):
        def gram(x):
            x=x.float()-x.float().mean(-2,keepdim=True);return x@x.transpose(-1,-2)
        ks,kt=gram(s),gram(t.detach())
        return (1-(ks*kt).sum((-1,-2))/(ks.square().sum((-1,-2))*kt.square().sum((-1,-2))).clamp_min(1e-20).sqrt()).mean()
    def clean(rows,grad=False,nomask=False):
        batch=device(backbone.collator([BioSeqRecord.from_prepared_dict(r) for r in rows]))
        teacher=[]
        hook=model.backbone.transformer.blocks[1].register_forward_hook(lambda mod,inp,result:teacher.append(block_hidden(result).detach()))
        try:
            # The no-mask arm still obtains its teacher under original native attention.
            with torch.set_grad_enabled(grad and not nomask),torch.autocast('cuda',dtype=torch.bfloat16):
                _,h=model.denoise(**model.feature_args(batch),output_hidden_states=True)
        finally:hook.remove()
        mask=batch['residue_mask'] & batch['attention_mask']
        if nomask:
            modified={**batch,'attention_mask':batch['attention_mask'] & batch['input_ids'].ne(model.tokenizer.mask_token_id)}
            assert not (mask & ~modified['attention_mask']).any()
            with torch.set_grad_enabled(grad),torch.autocast('cuda',dtype=torch.bfloat16):
                _,h=model.denoise(**model.feature_args(modified),output_hidden_states=True)
        assert len(teacher)==1 and not teacher[0].requires_grad
        return h,teacher[0],mask
    class NoMaskBackbone(FixedChainBackbone):
        def _final_hidden(self,batch):
            modified={**batch,'attention_mask':batch['attention_mask'] & batch['input_ids'].ne(self.model.tokenizer.mask_token_id)}
            assert not (batch['residue_mask'] & ~modified['attention_mask']).any()
            return super()._final_hidden(modified)
    def errors(s,t):
        original=(s-mu_s)/rms_s-((t-mu_t)/rms_t)@rotation
        teacher_basis=original@rotation.T@axes
        return original.square().mean(),(teacher_basis.square()*weight).mean()/denominator
    def token_direction(h,t,m,dense):
        hs=h.float()-pool(h,m)[:,None,:]
        tt=(t.float()-pool(t,m)[:,None,:])@rotation
        if dense:
            error=1-F.cosine_similarity(hs,tt,dim=-1,eps=1e-8)
            return (error.masked_fill(~m,0).sum(1)/m.sum(1)).mean()
        losses=[]
        for i in range(len(h)):
            available=m[i].nonzero().flatten()
            idx=available[torch.linspace(0,len(available)-1,32,device='cuda').long()]
            losses.append((1-F.cosine_similarity(hs[i,idx],tt[i,idx],dim=-1,eps=1e-8)).mean())
        return torch.stack(losses).mean()
    def objective(h,t,m,spectral,arm):
        s,u=pool(h,m),pool(t,m)
        uniform,weighted=errors(s,u)
        st,tt=[],[]
        for i in range(len(h)):
            available=m[i].nonzero().flatten();idx=available[torch.linspace(0,len(available)-1,32,device='cuda').long()]
            st.append(h[i,idx]);tt.append(t[i,idx])
        token=relation(torch.stack(st),torch.stack(tt))
        sv=(s-s.mean(0)).square().mean();tv=(u-u.mean(0)).square().mean().detach()
        floor_loss=F.relu(.25*tv-sv)/(tv+1e-12)
        directional=token_direction(h,t,m,arm=='token_all_direction')
        base=(weighted if spectral else uniform)+floor_loss
        legacy=(weighted if spectral else uniform)+.1*token+floor_loss
        return base+.1*directional,dict(uniform_mse=float(uniform.detach()),spectral_mse=float(weighted.detach()),token=float(token.detach()),directional=float(directional.detach()),legacy_representation_loss=float(legacy.detach()))
    def validate(step,nomask=False):
        model.eval();student,teacher=[],[];token32_errors=[];token_all_errors=[]
        with torch.no_grad():
            for start in range(0,len(groups['screen']),16):
                h,t,m=clean(groups['screen'][start:start+16],nomask=nomask);student.append(pool(h,m));teacher.append(pool(t,m))
                token32_errors.append(float(token_direction(h,t,m,False)))
                token_all_errors.append(float(token_direction(h,t,m,True)))
        s,t=torch.cat(student),torch.cat(teacher);uniform,weighted=errors(s,t)
        metrics=dict(step=step,coordinate_mse=float(uniform),spectral_mse=float(weighted),heldout_pool_cka=float(1-relation(s,t)),token32_direction_error=float(np.mean(token32_errors)),token_all_direction_error=float(np.mean(token_all_errors)))
        for ratio in (.5,1.):
            total,count,correct,aa_count=0.,0,0,0
            with torch.random.fork_rng(devices=[0]),torch.no_grad():
                torch.manual_seed(87913)
                for start in range(0,len(groups['screen']),16):
                    batch=device(collator(groups['screen'][start:start+16]))
                    noised,targets,selected=model.corrupt(batch,torch.full_like(batch['input_ids'],ratio,dtype=torch.float))
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model.denoise(**{**model.feature_args(batch),'input_ids':noised})
                    total+=float(F.cross_entropy(logits[selected].float(),targets[selected],reduction='sum'));count+=int(selected.sum())
                    aa=selected & batch['residue_mask'];correct+=int((logits.argmax(-1)[aa]==targets[aa]).sum());aa_count+=int(aa.sum())
            metrics[f'ce_{ratio}']=total/count;metrics[f'aa_accuracy_{ratio}']=correct/aa_count
        model.train();return metrics
    initial_validation=validate(1000)
    old=json.loads((initial/'validation.json').read_text())[-1]
    for k in old:assert abs(old[k]-initial_validation[k])<2e-6,(k,old[k],initial_validation[k])
    write(out/'initial_validation.json',initial_validation)
    control=Path(plan['control'])
    control_start=json.loads((control/'train.jsonl').read_text().splitlines()[0])
    assert schedule==json.loads((control.parent/'schedule.json').read_text())
    plan['control_artifact_sha256']={str(control/n):sha(control/n) for n in ('train.jsonl','validation.json','checkpoint.json')}
    write(out/'plan.json',plan)
    with torch.no_grad():
        h0,t0,m0=clean([groups['train'][i] for i in schedule[0]],nomask=True)
        aligned=t0.float()@rotation
        perfect={str(dense):float(token_direction(aligned,t0,m0,dense)) for dense in (False,True)}
        reversed_direction={str(dense):float(token_direction(-aligned,t0,m0,dense)) for dense in (False,True)}
        assert all(abs(v)<2e-6 for v in perfect.values()),perfect
        assert all(abs(v-2)<2e-6 for v in reversed_direction.values()),reversed_direction
        original_relation=float(relation(h0[0,m0[0]][None],t0[0,m0[0]][None]))
        # A deterministic coordinate permutation changes directions but preserves
        # within-sequence Gram matrices; it does not establish a pooling failure.
        permuted=h0[0,m0[0]].flip(-1)[None]
        permuted_relation=float(relation(permuted,t0[0,m0[0]][None]))
        assert abs(original_relation-permuted_relation)<2e-6
        write(out/'loss_contract_verification.json',dict(status='passed',perfect_alignment_error=perfect,opposite_alignment_error=reversed_direction,relation_before_permutation=original_relation,relation_after_permutation=permuted_relation,interpretation='tests objective geometry only, not a cause of GDP gaps'))
        del h0,t0,m0,aligned,permuted
    results={}
    for arm in plan['arms']:
        folder=out/arm;folder.mkdir();model.load_state_dict(reference,strict=True)
        nomask=True
        for n,p in model.named_parameters():p.requires_grad_(n in train_names)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=.01)
        restored_optimizer=torch.load(initial/'optimizer.pt',map_location='cpu',weights_only=False)
        assert restored_optimizer['global_step']==1000
        assert restored_optimizer['ordered_parameter_names']==[n for n,p in model.named_parameters() if p.requires_grad]
        optimizer.load_state_dict(restored_optimizer['optimizer']);del restored_optimizer
        assert all(group['lr']==2e-5 for group in optimizer.param_groups)
        for group in optimizer.param_groups:group['lr']=plan['lr']
        assert all(group['lr']==2e-4 for group in optimizer.param_groups)
        optimizer_steps=sorted({int(state['step'].item()) for state in optimizer.state.values()})
        assert optimizer_steps==[700],optimizer_steps
        write(folder/'resume_verification.json',dict(global_step=1000,optimizer_steps=optimizer_steps,
              optimizer_sha256=sha(initial/'optimizer.pt'),ordered_parameter_names_verified=True))
        model.train();history=[validate(1000,nomask)];started=time.monotonic()
        for k,v in json.loads((control/'validation.json').read_text())[0].items():
            assert abs(history[0][k]-v)<2e-6,(k,history[0][k],v)
        for local_step,indices in enumerate(schedule,1):
            step=local_step+1000;torch.manual_seed(91000+step)
            rows=[groups['train'][i] for i in indices];optimizer.zero_grad(set_to_none=True)
            batch=device(collator(rows))
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):ce=model(**batch)['loss']
            h,t,m=clean(rows,True,nomask);rep,detail=objective(h,t,m,True,arm)
            if local_step==1:
                assert float(ce)==control_start['ce']
                assert detail['legacy_representation_loss']==control_start['representation_loss']
                for k in ('uniform_mse','spectral_mse','token'):assert detail[k]==control_start[k],k
                write(folder/'first_batch_control_verification.json',dict(status='passed',ce=float(ce),representation_loss=float(rep.detach()),details=detail,ce_requires_grad=ce.requires_grad))
            assert not ce.requires_grad
            rep.backward()
            grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(ce) and torch.isfinite(rep) and torch.isfinite(grad)
            optimizer.step()
            if local_step==1 or step%20==0:
                row=dict(step=step,ce=float(ce.detach()),representation_loss=float(rep.detach()),gradient_norm=float(grad),seconds=time.monotonic()-started,**detail)
                with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                status.update(stage='train',arm=arm,step=step,last_train=row);write(out/'status.json',status);print(arm,row,flush=True)
            if step%100==0:history.append(validate(step,nomask));write(folder/'validation.json',history)
        changed=[]
        for n,p in model.named_parameters():
            equal=torch.equal(p.detach().cpu(),reference[n])
            if n not in train_names:assert equal,n
            elif not equal:changed.append(n)
        assert changed
        save_file({n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if n in train_names},str(folder/'trained_parameters.safetensors'))
        torch.save(dict(optimizer=optimizer.state_dict(),global_step=1700,ordered_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad]),folder/'optimizer.pt')
        write(folder/'checkpoint.json',dict(parent_checkpoint=side['parent_checkpoint'],parent_identity=side['parent_identity'],
              parameter_file=str(folder/'trained_parameters.safetensors'),sha256=sha(folder/'trained_parameters.safetensors'),
              trainable_names=sorted(train_names),changed_names=changed,steps=1700,frozen_parameters_bitwise_equal=True,
              source_sha256=plan['source_sha256'],initial_adaptation_sha256=side['sha256'],optimizer_sha256=sha(folder/'optimizer.pt')))
        model.eval().requires_grad_(False);status.update(stage='gdp');write(out/'status.json',status)
        from downstream.grammar.ab_probes import prepare_task,run_gdp
        from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL
        from scipy.stats import spearmanr
        from types import SimpleNamespace
        df,pairs,_,_=prepare_task('gdp_a1')
        records=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],task_type='antibody',source='native_v5_probe') for h,l in pairs]
        checks=[];arm_results=dict(validation=history[-1])
        restored,restored_identity=load(folder)
        for readout in ('native','no_mask_keys'):
            cls=FixedChainBackbone if readout=='native' else NoMaskBackbone
            x=cls(model,identity).vectors(records)
            feature_name='final_features.npy' if readout=='native' else 'final_no_mask_features.npy'
            np.save(folder/feature_name,x)
            gdpfolder=folder/('gdp_a1' if readout=='native' else 'gdp_no_mask_keys');gdpfolder.mkdir()
            metrics=run_gdp(df,torch.from_numpy(x.astype(np.float64)),SimpleNamespace(out_dir=gdpfolder));write(gdpfolder/'metrics.json',metrics)
            for col in ALL_COLS:
                saved=np.load(gdpfolder/col/'legacy_oof.npz');valid=df[col].notna().to_numpy()
                np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid));np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL]);np.testing.assert_array_equal(saved['y_true'],df.loc[valid,col])
                scores=[float(spearmanr(saved['y_transformed'][saved['fold']==i],saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
                assert abs(np.mean(scores)-metrics['properties'][col]['legacy_reference']['cv_mean_spearman'])<1e-12
                checks.append(dict(readout=readout,property=col,fold_scores=scores))
            np.testing.assert_array_equal(cls(restored,restored_identity).vectors(records),x)
            arm_results[readout]=metrics
        del restored,optimizer
        torch.cuda.empty_cache()
        write(folder/'verification.json',dict(status='passed',oof_checks=checks,reloaded_features_bitwise_equal=True,frozen_parameters_bitwise_equal=True))
        results[arm]=arm_results;write(out/'results.json',results)
    for p,digest in plan['source_sha256'].items():assert sha(p)==digest,p
    assert sha(prior/'calibration.npz')==plan['calibration_sha256']
    assert sha(initial/'optimizer.pt')==plan['initial_optimizer_sha256']
    for p,digest in plan['control_artifact_sha256'].items():assert sha(p)==digest,p
    status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',status)


if __name__=='__main__':main()
