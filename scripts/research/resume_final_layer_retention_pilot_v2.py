"""Recover the interrupted fixed6000 retention intervention from its5500 bundle.

Activate pllm/protenix_abtcr; from repository root run:
python -B scripts/research/resume_final_layer_retention_pilot_v2.py --out ABS_EXISTING_RUN
Reuses the original loss/evaluation implementation, archives interrupted logs,
restores AdamW5200 and fixed schedules, and records numerical drift at the overlapping5520 update. Identical-seed
repeat trials established non-bitwise execution. Exact initial state plus
explicit numerical health checks are required; the trajectory is not bitwise. Does not
rerun the completed control. Original source and plan remain immutable.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    original_plan=json.loads((out/'plan.json').read_text())
    previous_status=json.loads((out/'status.json').read_text())
    old_process=Path(f"/proc/{previous_status['pid']}/cmdline")
    assert not old_process.exists() or b'train_final_layer_retention_pilot.py' not in old_process.read_bytes(), 'Original training still live'
    assert previous_status['status']=='failed' and previous_status['stage']=='recovery_overlap_check'
    repeat_dir=ROOT/'output/diagnostics/retention_recovery_repeatability_20261019'
    repeat=json.loads((repeat_dir/'results.json').read_text())
    assert repeat['status']=='passed' and not repeat['two_trials_logs_bitwise_equal']
    assert repeat['starting_parameters_and_optimizer_verified']
    for path,digest in repeat['source_sha256'].items():assert sha(path)==digest
    for group in ['source_sha256','control_artifact_sha256','replay_source_sha256']:
        for path,digest in original_plan[group].items():assert sha(Path(path))==digest,path
    recovery_dir=out/'tcr_cdr_replay'
    recovery_meta=json.loads((recovery_dir/'recovery_latest.json').read_text())
    assert recovery_meta['global_step']==5500 and sha(Path(recovery_meta['bundle']))==recovery_meta['sha256']
    assert json.loads((out/'ab_control/verification.json').read_text())['status']=='passed'
    archive=out/'recovery_audit_v2_20261019';archive.mkdir(exist_ok=False)
    archive_names=['status.json','plan.json','results.json','tcr_cdr_replay/train.jsonl',
        'tcr_cdr_replay/validation.json','tcr_cdr_replay/replay_validation.jsonl',
        'tcr_cdr_replay/recovery_latest.json']
    for name in archive_names:
        target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(out/name,target)
    control_hashes={str(p):sha(p) for p in (out/'ab_control').rglob('*') if p.is_file()}
    recovery_record=dict(status='running',original_status=previous_status,recovery_bundle=recovery_meta,
        source=str(Path(__file__).resolve()),source_sha256=sha(Path(__file__)),
        archived_sha256={str(archive/n):sha(archive/n) for n in archive_names},control_sha256=control_hashes,
        restore_global_step=5500,expected_optimizer_step=5200,remaining_updates=500,
        repeatability_evidence=str(repeat_dir/'results.json'),repeatability_sha256=sha(repeat_dir/'results.json'),
        criterion='exact initial parameters and AdamW; original5520 CE/representation/total atol1e-4+rtol1%, all other floating metrics atol1e-4+rtol5%; numerical health checks, not bitwise trajectory claim; fixed6000')
    write(archive/'recovery.json',recovery_record)
    prior=ROOT/'output/diagnostics/final_layer_coordinate_pilot_20261018'
    data_root=ROOT/'output/diagnostics/final_layer_preservation_pilot_20261018'
    spectral_parent=ROOT/'output/diagnostics/final_layer_spectral_pilot_20261018'
    initial=ROOT/'output/diagnostics/final_layer_joint_convergence_pilot_20261018/token_all_direction'
    assert json.loads((initial.parent/'status.json').read_text())['status']=='success'
    side=json.loads((initial/'checkpoint.json').read_text())
    assert side['steps']==5000
    groups={name:[json.loads(line) for line in (data_root/f'{name}.jsonl').read_text().splitlines()] for name in ('train','screen')}
    data_manifest=json.loads((data_root/'data.json').read_text())
    for name in groups:assert sha(data_root/f'{name}.jsonl')==data_manifest['files'][name]['sha256']
    rng=random.Random(42);schedule=[]
    while len(schedule)<6000:
        order=list(range(len(groups['train'])));rng.shuffle(order)
        schedule.extend([order[i:i+16] for i in range(0,len(order),16)])
    assert schedule[1700:5000]==json.loads((initial.parent/'schedule.json').read_text())
    schedule=schedule[5000:6000];assert len(schedule)==1000
    rng=random.Random(942);tcr_schedule=[]
    while len(tcr_schedule)<1000:
        order=list(range(4096));rng.shuffle(order)
        tcr_schedule.extend([order[i:i+16] for i in range(0,len(order),16)])
    tcr_schedule=tcr_schedule[:1000]
    assert schedule==json.loads((out/'schedule.json').read_text())
    assert tcr_schedule==json.loads((out/'tcr_schedule.json').read_text())
    plan=original_plan
    status=dict(status='running',stage='recovery_prepare',pid=os.getpid(),host=socket.gethostname(),started_utc=plan['created_utc'],resumed_utc=datetime.now(timezone.utc).isoformat(),recovered_from_step=5500)
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
    saved_spectrum=np.load(out/'spectrum.npz')
    for key,value in [('eigenvalues',eigen),('axes',vh),('weights',weights)]:np.testing.assert_array_equal(saved_spectrum[key],value)
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
    initial_validation=validate(5000,nomask=True)
    old=json.loads((initial/'validation.json').read_text())[-1]
    for k in old:assert abs(old[k]-initial_validation[k])<2e-6,(k,old[k],initial_validation[k])
    for key,value in json.loads((out/'initial_validation.json').read_text()).items():assert abs(initial_validation[key]-value)<2e-6
    control=Path(plan['control'])
    for path,digest in plan['control_artifact_sha256'].items():assert sha(path)==digest
    from downstream.final_retention_objectives import RetentionReplay
    replay=RetentionReplay(ROOT,model,backbone,groups,sha)
    normalization=replay.calibrate();saved_normalization=json.loads((out/'replay_normalization.json').read_text())
    assert normalization==saved_normalization
    assert replay.hashes==plan['replay_source_sha256']
    replay.weights=saved_normalization['weights']
    for n,p in model.named_parameters():p.requires_grad_(n in train_names)
    results=json.loads((out/'results.json').read_text());assert set(results)=={'ab_control'}
    first_reference=json.loads((recovery_dir/'first_continued_batch.json').read_text())
    for arm in ['tcr_cdr_replay']:
        folder=out/arm;model.load_state_dict(reference,strict=True)
        nomask=True
        for n,p in model.named_parameters():p.requires_grad_(n in train_names)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=.01)
        recovered=torch.load(recovery_meta['bundle'],map_location='cpu',weights_only=False)
        assert recovered['global_step']==5500
        assert recovered['source_sha256']==plan['source_sha256']
        assert recovered['schedule_sha256']==plan['schedule_sha256']==sha(out/'schedule.json')
        assert recovered['initial_parameter_sha256']==side['sha256']
        assert set(recovered['trained_parameters'])==train_names
        assert recovered['ordered_parameter_names']==[n for n,p in model.named_parameters() if p.requires_grad]
        loaded=model.load_state_dict(recovered['trained_parameters'],strict=False)
        assert not loaded.unexpected_keys and set(loaded.missing_keys)==set(model.state_dict())-train_names
        for n,p in model.named_parameters():
            expected=recovered['trained_parameters'][n] if n in train_names else reference[n]
            assert torch.equal(p.detach().cpu(),expected),n
        optimizer.load_state_dict(recovered['optimizer'])
        assert {int(v['step'].item()) for v in optimizer.state.values()}=={5200}
        assert all(group['lr']==2e-4 for group in optimizer.param_groups)
        del recovered
        history=json.loads((folder/'validation.json').read_text())
        assert [v['step'] for v in history]==list(range(5000,5501,100))
        observed=validate(5500,True)
        for key,value in history[-1].items():assert abs(observed[key]-value)<2e-6,(key,observed[key],value)
        prior_replay=[json.loads(line) for line in (folder/'replay_validation.jsonl').read_text().splitlines()]
        assert [v['step'] for v in prior_replay]==list(range(5100,5501,100))
        actual_replay=replay.screen()
        for key,value in actual_replay.items():assert abs(value-prior_replay[-1][key])<2e-6,(key,value,prior_replay[-1][key])
        prior_train=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
        original_train=[json.loads(line) for line in (out/'recovery_audit_20261019/tcr_cdr_replay/train.jsonl').read_text().splitlines()]
        overlap={v['step']:v for v in original_train if v['step']>5500}
        assert set(overlap)=={5520}
        retained=[v for v in prior_train if v['step']<=5500]
        assert [v['step'] for v in retained]==[5001]+list(range(5020,5501,20))
        (folder/'train.jsonl').write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in retained))
        recovery_record.update(recovered_parameters_bitwise_equal=True,restored_optimizer_steps=[5200],
            restored_validation=observed,restored_replay=actual_replay)
        write(archive/'recovery.json',recovery_record)
        elapsed_before=retained[-1]['seconds'];model.train();started=time.monotonic()
        for local_step,indices in enumerate(schedule[500:],501):
            step=local_step+5000;torch.manual_seed(91000+step)
            rows=[groups['train'][i] for i in indices];optimizer.zero_grad(set_to_none=True)
            batch=device(collator(rows))
            with torch.autocast('cuda',dtype=torch.bfloat16):ce=model(**batch)['loss']
            h,t,m=clean(rows,True,nomask);rep,detail=objective(h,t,m,True,'token_all_direction')
            raw=replay.raw('train',indices,tcr_schedule[local_step-1],(local_step-1)%3,arm=='tcr_cdr_replay')
            retention=replay.weighted(raw)
            if local_step==1:
                first=dict(ce=float(ce.detach()),representation_loss=float(rep.detach()),
                    replay_raw={k:float(v.detach()) for k,v in raw.items()})
                if first_reference is None:first_reference=first
                else:assert first==first_reference,(first,first_reference)
                write(folder/'first_continued_batch.json',first)
                write(folder/'paired_first_batch_verification.json',dict(status='passed',all_objective_forwards_equal=True))
            assert ce.requires_grad and rep.requires_grad
            loss=rep+ce+(retention if arm=='tcr_cdr_replay' else retention.detach()*0.)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(ce) and torch.isfinite(rep) and torch.isfinite(grad)
            optimizer.step()
            if local_step==1 or step%20==0:
                row=dict(step=step,ce=float(ce.detach()),representation_loss=float(rep.detach()),total_loss=float(loss.detach()),gradient_norm=float(grad),replay_weighted=float(retention.detach()),replay_raw={k:float(v.detach()) for k,v in raw.items()},seconds=elapsed_before+time.monotonic()-started,**detail)
                if step in overlap:
                    expected=overlap[step]
                    differences={}
                    for key,value in row.items():
                        if key in ['seconds','step']:continue
                        values=value if isinstance(value,dict) else {key:value}
                        references=expected[key] if isinstance(value,dict) else {key:expected[key]}
                        for metric,actual in values.items():
                            reference_value=references[metric]
                            rtol=.01 if metric in ['ce','representation_loss','total_loss'] else .05
                            tolerance=1e-4+rtol*abs(reference_value)
                            difference=abs(actual-reference_value)
                            differences[metric]=dict(actual=actual,original=reference_value,abs_difference=difference,tolerance=tolerance)
                            assert difference<=tolerance,(step,metric,actual,reference_value,tolerance)
                    recovery_record['overlapping5520_bitwise_equal']=all(v['abs_difference']==0 for v in differences.values())
                    recovery_record['overlapping5520_numerical_checks_passed']=True
                    recovery_record['overlap_differences']=differences
                    write(archive/'recovery.json',recovery_record)
                with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                status.update(stage='train',arm=arm,step=step,last_train=row);write(out/'status.json',status);print(arm,row,flush=True)
            if step%100==0:
                history.append(validate(step,nomask));write(folder/'validation.json',history)
                observed=replay.screen();observed['step']=step
                with (folder/'replay_validation.jsonl').open('a') as f:f.write(json.dumps(observed,allow_nan=False)+'\n')
            if step%500==0 and step<6000:
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
        torch.save(dict(optimizer=optimizer.state_dict(),global_step=6000,ordered_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad]),folder/'optimizer.pt')
        write(folder/'checkpoint.json',dict(parent_checkpoint=side['parent_checkpoint'],parent_identity=side['parent_identity'],
              parameter_file=str(folder/'trained_parameters.safetensors'),sha256=sha(folder/'trained_parameters.safetensors'),
              trainable_names=sorted(train_names),changed_names=changed,steps=6000,frozen_parameters_bitwise_equal=True,
              source_sha256={**plan['source_sha256'],str(Path(__file__).resolve()):sha(Path(__file__))},initial_adaptation_sha256=side['sha256'],optimizer_sha256=sha(folder/'optimizer.pt')))
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
    for p,digest in replay.hashes.items():assert sha(p)==digest,p
    assert recovery_record['overlapping5520_numerical_checks_passed']
    for path,digest in control_hashes.items():assert sha(path)==digest,path
    recovery_record.update(status='success',control_unchanged=True,finished_utc=datetime.now(timezone.utc).isoformat())
    write(archive/'recovery.json',recovery_record)
    status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',status)


if __name__=='__main__':main()
