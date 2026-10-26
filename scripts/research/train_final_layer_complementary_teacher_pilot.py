"""Distill matched early512 versus early256/global256 targets into actual final hidden.

Activate pllm then protenix_abtcr; from the repository root run:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/research/train_final_layer_complementary_teacher_pilot.py --out ABS_NEW_DIRECTORY
Both arms restore dense-token1700 parameters and AdamW, use the same3300
batches to fixed5000, native CE1 and dense original-block2 token directions.
Only the OAS-fitted pooled teacher target and its target-specific PC64 metric
change between arms. No teacher concatenation or projection at inference.
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
    initial=ROOT/'output/diagnostics/final_layer_token_direction_pilot_20261018/token_all_direction'
    assert json.loads((spectral_parent/'status.json').read_text())['status']=='success'
    assert json.loads((initial.parent/'status.json').read_text())['status']=='success'
    paired=ROOT/'output/diagnostics/final_layer_joint_convergence_pilot_20261018/token_all_direction'
    targets_root=ROOT/'output/diagnostics/global_teacher_targets_20261018'
    assert json.loads((targets_root/'status.json').read_text())['status']=='success'
    assert json.loads((paired.parent/'status.json').read_text())['status']=='success'
    side=json.loads((initial/'checkpoint.json').read_text())
    groups={name:[json.loads(line) for line in (data_root/f'{name}.jsonl').read_text().splitlines()] for name in ('train','screen')}
    data_manifest=json.loads((data_root/'data.json').read_text())
    for name in groups:assert sha(data_root/f'{name}.jsonl')==data_manifest['files'][name]['sha256']
    rng=random.Random(42);schedule=[]
    while len(schedule)<5000:
        order=list(range(len(groups['train'])));rng.shuffle(order)
        schedule.extend([order[i:i+16] for i in range(0,len(order),16)])
    assert schedule[:300]==json.loads((data_root/'schedule.json').read_text())
    assert schedule[300:1000]==json.loads((spectral_parent/'schedule.json').read_text())
    assert schedule[1000:1700]==json.loads((initial.parent/'schedule.json').read_text())
    schedule=schedule[1700:5000];assert len(schedule)==3300
    assert schedule==json.loads((paired.parent/'schedule.json').read_text())
    write(out/'schedule.json',schedule)
    sources=[Path(__file__),ROOT/'scripts/research/train_final_layer_preservation_pilot.py',
             ROOT/'downstream/final_layer_preservation_eval.py',ROOT/'downstream/fixed_chain_eval.py',ROOT/'downstream/grammar/ab_probes.py']
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),initial_checkpoint=str(initial),
          initial_parameter_sha256=side['sha256'],parent_checkpoint=side['parent_checkpoint'],
          arms=['early512','early256_global256'],initial_steps=1700,additional_steps=3300,total_steps=5000,
          data_manifest=str(data_root/'data.json'),data_sha256=sha(data_root/'data.json'),schedule_sha256=sha(out/'schedule.json'),
          seed=42,batch_size=16,lr=2e-4,warmup_after_restart=0,
          optimizer='restore dense-token1700 AdamW moments and LR2e-4; clip1, constant LR, same weight_decay=.01',
          initial_optimizer_sha256=sha(initial/'optimizer.pt'),
          trainable='same blocks3-8 and final RMSNorm; early teacher and vocabulary head remain frozen',
          loss='native CE1 + target-specific PC64 pooled spectral MSE +0.1 dense original-block2 token directions; no variance floor (fixed teacher targets)',
          intervention='matched512-dimensional early-only versus early/global teacher; same1700 start, AdamW, LR2e-4, 3300 batches, CE1 and dense token loss; not a matched-gradient-norm comparison',
          direction='center each sequence over all real residues, then compare student token directions with frozen teacher token directions rotated by the existing OAS global rotation; no token-norm target',
          hypothesis='complementary OAS-fitted teacher improves pooled property signal beyond early-only target; evaluate actual trained final hidden, not target features',
          control=str(initial),
          ce_weight=1.0,
          full_early_ce1_reference=str(paired),target_artifact=str(targets_root),
          spectral='each arm target normalized by parent RMS; OAS4096 PCA in768 output coordinates; w_i=1/(lambda_i+lambda_64), normalized by mean(lambda_i*w_i); includes nullspace error',
          calibration=str(prior/'calibration.npz'),calibration_sha256=sha(prior/'calibration.npz'),
          selection='fixed5000; screen-only monitoring every100; no intermediate GDP evaluation or selection; all five GDP properties and both readouts at endpoint',
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
    mu_s,mu_t,rms_s,rms_t,rotation=[torch.as_tensor(cal[k],device='cuda',dtype=torch.float32) for k in ['mu_s','mu_t','rms_s','rms_t','rotation']]
    # Frozen parent screen features use the exact original topology for block2
    # and all-global topology for the complementary branch. Neither uses labels.
    from downstream.fixed_chain_eval import load as load_parent
    teacher_model,teacher_identity=load_parent(side['parent_checkpoint'])
    assert teacher_identity==side['parent_identity']
    teacher_backbone=FixedChainBackbone(teacher_model,teacher_identity,batch_size=16)
    def parent_features(rows):
        records=[BioSeqRecord.from_prepared_dict(r) for r in rows]
        early=[]
        for start in range(0,len(records),16):
            batch=teacher_backbone.collator(records[start:start+16])
            batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
            captured=[]
            hook=teacher_model.backbone.transformer.blocks[1].register_forward_hook(lambda mod,inp,result:captured.append(block_hidden(result).detach()))
            try:teacher_backbone._final_hidden(batch)
            finally:hook.remove()
            mask=batch['residue_mask'] & batch['attention_mask']
            value=captured[0].float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)
            early.append(value.cpu().numpy())
        old=dict(teacher_model.options)
        try:
            teacher_model.options['segment_global']=False
            global_features=teacher_backbone.vectors(records)
        finally:
            teacher_model.options.clear();teacher_model.options.update(old)
        return np.concatenate(early).astype(np.float64),global_features.astype(np.float64)
    first_early,first_global=parent_features(groups['train'][:16])
    np.testing.assert_array_equal(first_early,cal['teacher'][:16])
    all_global=np.load(targets_root/'all_global_oas.npy')
    np.testing.assert_array_equal(first_global,all_global[:16])
    screen_early,screen_global=parent_features(groups['screen'])
    np.savez_compressed(out/'screen_teacher_features.npz',early=screen_early,all_global=screen_global)
    del teacher_model,teacher_backbone
    torch.cuda.empty_cache()
    early_path=ROOT/'output/diagnostics/multilayer_teacher_targets_20261018/oas_fitted_targets.npz'
    global_path=targets_root/'global_pca.npz'
    early_axes=np.load(early_path);global_axes=np.load(global_path)
    target_data={}
    pinned_targets=[early_path,global_path,targets_root/'all_global_oas.npy',targets_root/'verification.json']
    for name in plan['arms']:
        folder=out/name;folder.mkdir()
        target_cal=np.load(targets_root/name/'calibration.npz')
        parts=[('early',512)] if name=='early512' else [('early',256),('global',256)]
        def transform(early,global_):
            branch={'early':(early-early_axes['early_mean'])@early_axes['early_axes'].T,
                    'global':(global_-global_axes['mean'])@global_axes['axes'].T}
            projected=np.concatenate([branch[n][:,:rank]/target_cal[n+'_scale'] for n,rank in parts],axis=1)
            return target_cal['mean']+projected@target_cal['rotation']*target_cal['scale']
        target_train=np.load(targets_root/name/'oas_targets.npy')
        np.testing.assert_allclose(transform(cal['teacher'],all_global),target_train,atol=1e-9,rtol=1e-9)
        target_screen=transform(screen_early,screen_global)
        np.save(folder/'screen_targets.npy',target_screen)
        normalized=(target_train-cal['mu_s'])/cal['rms_s']
        np.testing.assert_allclose(normalized.mean(0),0,atol=1e-10)
        _,singular,vh=np.linalg.svd(normalized,full_matrices=False)
        eigen=singular**2/len(normalized);floor=float(eigen[63]);weights=1/(eigen+floor)
        denominator=float(np.mean(eigen*weights))
        assert floor>0 and denominator>0 and np.isfinite(weights).all()
        assert abs(float(np.mean((normalized@vh.T)**2*weights)/denominator)-1)<1e-8
        assert np.linalg.matrix_rank(normalized)==512
        np.savez_compressed(folder/'spectrum.npz',axes=vh,eigenvalues=eigen,weights=weights)
        write(folder/'target_verification.json',dict(status='passed',oas_target_reconstructed=True,rank=512,
              teacher_input_control_bitwise=True,mean_target_energy=1.,floor_at_PC64=floor,normalizer=denominator,
              screen_rows=len(target_screen),oas_rows=len(target_train),labels_used=False))
        target_data[name]=dict(train=torch.tensor(target_train,device='cuda',dtype=torch.float32),
              screen=torch.tensor(target_screen,device='cuda',dtype=torch.float32),
              axes=torch.tensor(vh.T,device='cuda',dtype=torch.float32),
              weights=torch.tensor(weights,device='cuda',dtype=torch.float32),denominator=denominator)
        pinned_targets += [targets_root/name/'calibration.npz',targets_root/name/'oas_targets.npy']
    plan['target_sha256']={str(p):sha(p) for p in pinned_targets}
    write(out/'plan.json',plan)
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
    def errors(s,target):
        error=(s-target.detach())/rms_s
        projected=error@active_target['axes']
        return error.square().mean(),(projected.square()*active_target['weights']).mean()/active_target['denominator']
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
    def objective(h,t,m,target):
        uniform,weighted=errors(pool(h,m),target)
        directional=token_direction(h,t,m,True)
        loss=weighted+.1*directional
        return loss,dict(uniform_mse=float(uniform.detach()),spectral_mse=float(weighted.detach()),directional=float(directional.detach()))
    def validate(step,nomask=False):
        model.eval();student,teacher=[],[];token32_errors=[];token_all_errors=[]
        with torch.no_grad():
            for start in range(0,len(groups['screen']),16):
                h,t,m=clean(groups['screen'][start:start+16],nomask=nomask);student.append(pool(h,m));teacher.append(pool(t,m))
                token32_errors.append(float(token_direction(h,t,m,False)))
                token_all_errors.append(float(token_direction(h,t,m,True)))
        s,t=torch.cat(student),torch.cat(teacher);uniform,weighted=errors(s,active_target['screen'])
        metrics=dict(step=step,coordinate_mse=float(uniform),spectral_mse=float(weighted),heldout_pool_cka=float(1-relation(s,active_target['screen'])),original_block2_cka=float(1-relation(s,t)),token32_direction_error=float(np.mean(token32_errors)),token_all_direction_error=float(np.mean(token_all_errors)))
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
    old=json.loads((initial/'validation.json').read_text())[-1]
    paired_start=json.loads((paired/'first_continued_batch.json').read_text())
    plan['initial_artifact_sha256']={str(initial/n):sha(initial/n) for n in ('validation.json','checkpoint.json')}
    write(out/'plan.json',plan)
    results={}
    for arm in plan['arms']:
        folder=out/arm;model.load_state_dict(reference,strict=True)
        active_target=target_data[arm]
        nomask=True
        for n,p in model.named_parameters():p.requires_grad_(n in train_names)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=.01)
        restored_optimizer=torch.load(initial/'optimizer.pt',map_location='cpu',weights_only=False)
        assert restored_optimizer['global_step']==1700
        assert restored_optimizer['ordered_parameter_names']==[n for n,p in model.named_parameters() if p.requires_grad]
        optimizer.load_state_dict(restored_optimizer['optimizer']);del restored_optimizer
        assert all(group['lr']==2e-4 for group in optimizer.param_groups)
        optimizer_steps=sorted({int(state['step'].item()) for state in optimizer.state.values()})
        assert optimizer_steps==[1400],optimizer_steps
        write(folder/'resume_verification.json',dict(global_step=1700,optimizer_steps=optimizer_steps,
              optimizer_sha256=sha(initial/'optimizer.pt'),ordered_parameter_names_verified=True))
        model.train();history=[validate(1700,nomask)];started=time.monotonic()
        for k in ('ce_0.5','ce_1.0','aa_accuracy_0.5','aa_accuracy_1.0','token_all_direction_error'):
            assert abs(history[0][k]-old[k])<2e-6,(k,history[0][k],old[k])
        write(folder/'initial_validation.json',history[0])
        for local_step,indices in enumerate(schedule,1):
            step=local_step+1700;torch.manual_seed(91000+step)
            rows=[groups['train'][i] for i in indices];optimizer.zero_grad(set_to_none=True)
            batch=device(collator(rows))
            with torch.autocast('cuda',dtype=torch.bfloat16):ce=model(**batch)['loss']
            h,t,m=clean(rows,True,nomask);rep,detail=objective(h,t,m,active_target['train'][indices])
            if local_step==1:
                assert float(ce.detach())==paired_start['ce']
                write(folder/'first_continued_batch.json',dict(step=step,ce=float(ce.detach()),representation_loss=float(rep.detach()),details=detail,ce_requires_grad=ce.requires_grad,initial_ce_matches_full_early_control=True))
            assert ce.requires_grad
            loss=rep+ce
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(ce) and torch.isfinite(rep) and torch.isfinite(grad)
            optimizer.step()
            if local_step==1 or step%20==0:
                row=dict(step=step,ce=float(ce.detach()),representation_loss=float(rep.detach()),total_loss=float(loss.detach()),gradient_norm=float(grad),seconds=time.monotonic()-started,**detail)
                with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                status.update(stage='train',arm=arm,step=step,last_train=row);write(out/'status.json',status);print(arm,row,flush=True)
            if step%100==0:history.append(validate(step,nomask));write(folder/'validation.json',history)
            if step%500==0 and step<5000:
                bundle=dict(global_step=step,trained_parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in train_names},
                    optimizer=optimizer.state_dict(),ordered_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad],
                    source_sha256=plan['source_sha256'],schedule_sha256=plan['schedule_sha256'],parent_checkpoint=side['parent_checkpoint'],
                    initial_checkpoint=str(initial),initial_parameter_sha256=side['sha256'])
                temporary=folder/'recovery_latest.tmp.pt';destination=folder/'recovery_latest.pt'
                torch.save(bundle,temporary);os.replace(temporary,destination);del bundle
                write(folder/'recovery_latest.json',dict(global_step=step,sha256=sha(destination),bundle=str(destination),purpose='recovery only; no downstream selection'))

        changed=[]
        for n,p in model.named_parameters():
            equal=torch.equal(p.detach().cpu(),reference[n])
            if n not in train_names:assert equal,n
            elif not equal:changed.append(n)
        assert changed
        save_file({n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if n in train_names},str(folder/'trained_parameters.safetensors'))
        torch.save(dict(optimizer=optimizer.state_dict(),global_step=5000,ordered_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad]),folder/'optimizer.pt')
        write(folder/'checkpoint.json',dict(parent_checkpoint=side['parent_checkpoint'],parent_identity=side['parent_identity'],
              parameter_file=str(folder/'trained_parameters.safetensors'),sha256=sha(folder/'trained_parameters.safetensors'),
              trainable_names=sorted(train_names),changed_names=changed,steps=5000,frozen_parameters_bitwise_equal=True,
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
    for p,digest in plan['initial_artifact_sha256'].items():assert sha(p)==digest,p
    for p,digest in plan['target_sha256'].items():assert sha(p)==digest,p
    status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',status)


if __name__=='__main__':main()
