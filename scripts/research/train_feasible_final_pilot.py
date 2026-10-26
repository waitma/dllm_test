"""Compare pooled versus feasible-token distillation from original fixed50000.

Activate pllm then protenix_abtcr. From repository root run:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B scripts/research/train_feasible_final_pilot.py --out ABS_NEW_DIR [--check-only]
Both arms use fixed1000 updates, identical original weights/schedules, frozen
embedding/blocks1-2/head/final RMSNorm. Only feasible token loss is toggled.
GDP is evaluated at the fixed endpoint; no intermediate downstream selection.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', stage='prepare', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', state)
    import numpy as np
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(4)
    d = ROOT/'output/diagnostics'
    target_root = d/'feasible_final_targets_20261019'
    assert json.loads((target_root/'status.json').read_text())['status']=='success'
    target_verification = json.loads((target_root/'verification.json').read_text())
    assert target_verification['status']=='passed' and not target_verification['gdp_read']
    for mapping in [target_verification['source_sha256'], *target_verification['artifact_sha256'].values()]:
        for path,digest in mapping.items(): assert sha(Path(path))==digest,path
    data = d/'final_layer_preservation_pilot_20261018'
    groups = {s:[json.loads(line) for line in (data/f'{s}.jsonl').read_text().splitlines()] for s in ['train','screen']}
    target = {s:np.load(target_root/s/'token_targets.npy', mmap_mode='r') for s in groups}
    coordinates = {s:dict(np.load(target_root/s/'coordinates.npz')) for s in groups}
    pooled = {s:dict(np.load(target_root/s/'pooled_features.npz')) for s in groups}
    cal_path = d/'final_layer_coordinate_pilot_20261018/calibration.npz'
    cal = np.load(cal_path); rms = float(cal['rms_s'])
    mean = pooled['train']['desired'].mean(0)
    normalized = (pooled['train']['desired']-mean)/rms
    _, singular, vh = np.linalg.svd(normalized, full_matrices=False)
    eigen = singular**2/len(normalized); floor = float(eigen[63])
    weights = 1/(eigen+floor); denominator = float(np.mean(eigen*weights))
    assert floor>0 and denominator>0
    np.savez(out/'spectrum.npz',mean=mean,axes=vh,eigenvalues=eigen,weights=weights,denominator=denominator,rms=rms)
    axes = torch.tensor(vh.T, device='cuda', dtype=torch.float32)
    metric_weights = torch.tensor(weights, device='cuda', dtype=torch.float32)
    from downstream.fixed_chain_eval import load, FixedChainBackbone
    from downstream.final_retention_objectives import RetentionReplay
    checkpoint = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    model, identity = load(checkpoint)
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
    from dllm.pipelines.immune_llada.fixed_chain.layout import FixedChainCollator
    reference = {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
    update_prefix = tuple(f'backbone.transformer.blocks.{i}.' for i in range(2,8))
    updated_names = {n for n,p in model.named_parameters() if n.startswith(update_prefix)}
    # The established evaluation loader expects this full delta key set. ln_f
    # is serialized unchanged for compatibility, and excluded from the optimizer.
    delta_names = {n for n,p in model.named_parameters() if n.startswith(update_prefix+('backbone.transformer.ln_f.',))}
    backbone = FixedChainBackbone(model,identity,batch_size=16)
    collator = FixedChainCollator.from_config(model.config, mode='train')
    replay = RetentionReplay(ROOT,model,backbone,groups,sha)
    replay_normalization_path = d/'final_layer_retention_pilot_20261018/replay_normalization.json'
    replay.weights = json.loads(replay_normalization_path.read_text())['weights']
    assert all(np.isfinite(x) and x>0 for x in replay.weights.values())
    rng=random.Random(42);schedule=[];tschedule=[]
    while len(schedule)<1000:
        order=list(range(len(groups['train'])));rng.shuffle(order)
        schedule.extend([order[i:i+16] for i in range(0,len(order),16)])
    trng=random.Random(4201)
    while len(tschedule)<1000:
        order=list(range(len(replay.tcr_rows['train'])));trng.shuffle(order)
        tschedule.extend([order[i:i+16] for i in range(0,len(order),16)])
    schedule=schedule[:1000];tschedule=tschedule[:1000]
    write(out/'schedule.json',dict(ab=schedule,tcr=tschedule))
    sources = {str(p):sha(p) for p in [Path(__file__), ROOT/'scripts/research/train_final_layer_preservation_pilot.py',
        ROOT/'downstream/final_retention_objectives.py',ROOT/'downstream/fixed_chain_eval.py',
        ROOT/'downstream/final_layer_preservation_eval.py',ROOT/'downstream/grammar/ab_probes.py',
        target_root/'verification.json',target_root/'plan.json',cal_path,replay_normalization_path,
        data/'train.jsonl',data/'screen.jsonl']}
    sources.update(replay.hashes)
    plan=dict(created_utc=state['started_utc'],parent_identity=identity,parent_checkpoint=str(checkpoint),
        arms=['pool_only','feasible_token'],steps=1000,batch_size=16,seed=42,learning_rate=1e-4,warmup=20,
        optimizer='fresh AdamW betas(.9,.999),eps1e-8,weight_decay.01,clip1; identical initialization per arm',
        optimized_parameter_names=sorted(updated_names),serialized_delta_names=sorted(delta_names),
        frozen='embedding, blocks1-2, vocabulary head and final RMSNorm',
        objective='native CE1 + PC64 spectral pooled target MSE + fixed TCR/CDR replay; feasible_token adds0.1 normalized feasible-token MSE',
        token_scale='mean MSE to feasible float32 token targets on first128 OAS training rows at original weights; equal sequence weighting',
        replay_weights=replay.weights,replay_coefficient_origin=str(replay_normalization_path),
        replay_weight_limitation='Reuses previous fixed CE1-start coefficients in BOTH arms; not recalibrated at this original start and not gradient-norm matched',
        replay_scope='both attention TCR pooled MSE, rotating full-CDR26class KL and log-partition; no sampled-intermediate-state coverage',
        selection='fixed1000 only, screen every100; no GDP checkpoint selection',
        comparison='only feasible token backward toggled; all other losses and schedules identical between these two arms, not a single-factor comparison against old pilots',
        target_root=str(target_root),target_verification_sha256=sha(target_root/'verification.json'),
        schedule_sha256=sha(out/'schedule.json'),source_sha256=sources)
    write(out/'plan.json',plan)
    def device(batch): return {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
    def pool(h,mask): return h.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)
    def clean(split, indices, grad):
        batch=device(backbone.collator([BioSeqRecord.from_prepared_dict(groups[split][i]) for i in indices]))
        mask=batch['residue_mask']&batch['attention_mask']
        expected=[]
        coords=coordinates[split]
        for j,i in enumerate(indices):
            lo,hi=map(int,coords['offsets'][i:i+2]);pos=mask[j].nonzero().flatten()
            np.testing.assert_array_equal(pos.cpu().numpy(),coords['positions'][lo:hi])
            np.testing.assert_array_equal(batch['input_ids'][j,pos].cpu().numpy(),coords['token_ids'][lo:hi])
            expected.append(torch.tensor(np.array(target[split][lo:hi]),device='cuda'))
        with torch.set_grad_enabled(grad),torch.autocast('cuda',dtype=torch.bfloat16):
            _,h=model.denoise(**model.feature_args(batch),output_hidden_states=True)
        expected=torch.cat(expected)
        losses=(h[mask].float()-expected).square().mean(-1)
        token=torch.stack([v.mean() for v in losses.split(mask.sum(1).tolist())]).mean()
        x=pool(h,mask)
        desired=torch.tensor(pooled[split]['desired'][indices],device='cuda',dtype=torch.float32)
        error=(x-desired)/rms
        spectral=(((error@axes).square()*metric_weights).mean()/denominator)
        return spectral,token,x
    token_scale=float(np.mean([float(clean('train',list(range(i,i+16)),False)[1]) for i in range(0,128,16)]))
    assert np.isfinite(token_scale) and token_scale>1e-10
    plan['token_mse_normalizer']=token_scale;write(out/'plan.json',plan)
    parameters=[]
    for n,p in model.named_parameters():
        p.requires_grad_(n in updated_names)
        if p.requires_grad:parameters.append(p)
    model.train();torch.manual_seed(91001)
    test_indices=schedule[0]
    batch=device(collator([groups['train'][i] for i in test_indices]))
    with torch.autocast('cuda',dtype=torch.bfloat16):ce=model(**batch)['loss']
    ce.backward()
    rep,tok,_=clean('train',test_indices,True)
    (rep+.1*tok/token_scale).backward()
    raw=replay.raw('train',test_indices,tschedule[0],0,True)
    replay.weighted(raw).backward()
    grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    assert grads and set(grads)<=updated_names and all(np.isfinite(x) for x in grads.values())
    for n,p in model.named_parameters():
        assert torch.equal(p.detach().cpu(),reference[n]),n
        if n not in updated_names:assert p.grad is None,n
    write(out/'preflight.json',dict(status='passed',no_weights_updated=True,gradient_names=sorted(grads),
        frozen_gradients_absent=True,ce=float(ce.detach()),spectral=float(rep.detach()),
        token_mse=float(tok.detach()),normalized_token_mse=float(tok.detach())/token_scale,
        replay={k:float(v.detach()) for k,v in raw.items()},max_gradient_norm=max(grads.values())))
    model.zero_grad(set_to_none=True)
    del ce,rep,tok,raw,batch
    if args.check_only:
        state.update(status='success',stage='preflight_complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',state)
        return
    class NoMaskBackbone(FixedChainBackbone):
        def _final_hidden(self,batch):
            batch={**batch,'attention_mask':batch['attention_mask']&batch['input_ids'].ne(self.model.tokenizer.mask_token_id)}
            return super()._final_hidden(batch)
    def screen(step):
        model.eval();values=[]
        for i in range(0,len(groups['screen']),16):
            a,b,_=clean('screen',list(range(i,min(i+16,len(groups['screen'])))),False)
            values.append((float(a),float(b)/token_scale))
        result=dict(step=step,spectral=float(np.mean([x[0] for x in values])),normalized_token_mse=float(np.mean([x[1] for x in values])),replay=replay.screen())
        model.train();return result
    baseline=screen(0);write(out/'baseline_validation.json',baseline)
    results={}
    for arm in plan['arms']:
        folder=out/arm;folder.mkdir()
        model.load_state_dict(reference,strict=True)
        for n,p in model.named_parameters():p.requires_grad_(n in updated_names)
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
        history=[baseline];model.train()
        for step,(indices,tindices) in enumerate(zip(schedule,tschedule),1):
            torch.manual_seed(91000+step);optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=1e-4*min(step/20,1.)
            batch=device(collator([groups['train'][i] for i in indices]))
            with torch.autocast('cuda',dtype=torch.bfloat16):ce=model(**batch)['loss']
            ce.backward()
            rep,tok,_=clean('train',indices,True)
            loss=rep+(.1*tok/token_scale if arm=='feasible_token' else 0.)
            loss.backward()
            raw=replay.raw('train',indices,tindices,(step-1)%3,True)
            replay_loss=replay.weighted(raw);replay_loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
            assert torch.isfinite(grad) and torch.isfinite(loss) and torch.isfinite(ce) and torch.isfinite(replay_loss)
            optimizer.step()
            if step==1 or step%20==0:
                record=dict(step=step,ce=float(ce.detach()),spectral=float(rep.detach()),normalized_token_mse=float(tok.detach())/token_scale,
                    replay={k:float(v.detach()) for k,v in raw.items()},gradient_norm=float(grad))
                with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                state.update(arm=arm,stage='train',step=step);write(out/'status.json',state)
                print(arm,record,flush=True)
            if step%100==0:
                history.append(screen(step));write(folder/'validation.json',history)
                recovery=dict(step=step,arm=arm,parameters={n:p.detach().cpu() for n,p in model.named_parameters() if n in delta_names},
                    optimizer=optimizer.state_dict(),optimized_parameter_names=sorted(updated_names),source_sha256=sources,
                    schedule_sha256=plan['schedule_sha256'],plan_sha256=sha(out/'plan.json'))
                tmp=folder/'recovery_latest.tmp.pt';torch.save(recovery,tmp);os.replace(tmp,folder/'recovery_latest.pt')
                write(folder/'recovery_latest.json',dict(step=step,sha256=sha(folder/'recovery_latest.pt')))
        changed=[]
        for n,p in model.named_parameters():
            equal=torch.equal(p.detach().cpu(),reference[n])
            if n not in updated_names:assert equal,n
            elif not equal:changed.append(n)
        assert changed
        parameter_file=folder/'trained_parameters.safetensors'
        save_file({n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if n in delta_names},str(parameter_file))
        torch.save(dict(optimizer=optimizer.state_dict(),global_step=1000,ordered_parameter_names=[n for n,p in model.named_parameters() if n in updated_names]),folder/'optimizer.pt')
        write(folder/'checkpoint.json',dict(parent_checkpoint=str(checkpoint),parent_identity=identity,
            parameter_file=str(parameter_file),sha256=sha(parameter_file),trainable_names=sorted(delta_names),
            optimizer_parameter_names=sorted(updated_names),frozen_serialized_names=sorted(delta_names-updated_names),
            changed_names=changed,steps=1000,frozen_parameters_bitwise_equal=True,source_sha256=sources,
            optimizer_sha256=sha(folder/'optimizer.pt')))
        state.update(stage='gdp',arm=arm);write(out/'status.json',state)
        model.eval().requires_grad_(False)
        from downstream.final_layer_preservation_eval import load as load_adapted
        from downstream.grammar.ab_probes import prepare_task,run_gdp
        from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL
        from scipy.stats import spearmanr
        df,pairs,_,_=prepare_task('gdp_a1')
        records=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],task_type='antibody',source='native_v5_probe') for h,l in pairs]
        restored,restored_identity=load_adapted(folder)
        arm_results={};checks=[]
        for mode,cls in [('native',FixedChainBackbone),('no_mask_keys',NoMaskBackbone)]:
            x=cls(model,identity,batch_size=8).vectors(records)
            np.testing.assert_array_equal(cls(restored,restored_identity,batch_size=8).vectors(records),x)
            np.save(folder/('final_features.npy' if mode=='native' else 'final_no_mask_features.npy'),x)
            dest=folder/('gdp_a1' if mode=='native' else 'gdp_no_mask_keys');dest.mkdir()
            metric=run_gdp(df,torch.from_numpy(x.astype(np.float64)),SimpleNamespace(out_dir=dest));write(dest/'metrics.json',metric)
            arm_results[mode]=metric
            for prop in ALL_COLS:
                saved=np.load(dest/prop/'legacy_oof.npz');valid=df[prop].notna().to_numpy()
                np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid));np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL]);np.testing.assert_array_equal(saved['y_true'],df.loc[valid,prop])
                scores=[float(spearmanr(saved['y_transformed'][saved['fold']==k],saved['prediction'][saved['fold']==k]).statistic) for k in range(5)]
                assert abs(np.mean(scores)-metric['properties'][prop]['legacy_reference']['cv_mean_spearman'])<1e-12
                checks.append(dict(mode=mode,property=prop,fold_scores=scores))
        write(folder/'verification.json',dict(status='passed',oof_checks=checks,reloaded_features_bitwise_equal=True,
            frozen_parameters_bitwise_equal=True,final_norm_bitwise_original=True))
        results[arm]=arm_results;write(out/'results.json',results)
        del restored,optimizer;torch.cuda.empty_cache()
    for path,digest in sources.items():assert sha(Path(path))==digest,path
    for mapping in target_verification['artifact_sha256'].values():
        for path,digest in mapping.items():assert sha(Path(path))==digest,path
    state.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',state)


if __name__=='__main__':
    main()
