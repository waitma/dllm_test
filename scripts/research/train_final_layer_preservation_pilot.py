"""Bounded local three-arm pilot preserving layer-two relations in final hidden.

Activate pllm then protenix_abtcr; from repository root run:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/research/train_final_layer_preservation_pilot.py --out ABS_NEW_DIRECTORY
Fixed 300 updates per arm. Only existing pretraining OAS train rows are trained.
GDPa1 is opened after training each arm, never for a loss or checkpoint selection.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def key(row):
    return tuple(sorted(zip(row['chain_roles'],row['chains'])))


def prepare_data(out):
    base=ROOT/'output/fixed_chain_exploration_20261014/data'
    manifest=json.loads((base/'manifest.json').read_text())
    groups={}
    for split,count in [('train',4096),('screen',128)]:
        path=base/f'{split}.jsonl'
        assert sha(path)==manifest['files'][split]['sha256']
        candidates={}
        with path.open() as stream:
            for line in stream:
                row=json.loads(line)
                if (row['source']!='oas_paired' or set(row['chain_roles'])!={'antibody_heavy','antibody_light'}
                        or len(row['chains'])!=2):continue
                seqs=dict(zip(row['chain_roles'],row['chains']))
                if len(seqs['antibody_heavy'])>=187 or len(seqs['antibody_light'])>=154:continue
                if any(any(v.get('synthetic_residue_mask',[])) for v in row.get('chain_metadata',{}).values()):continue
                candidates.setdefault(key(row),row)
        rows=list(candidates.values())
        random.Random(20261018).shuffle(rows)
        assert len(rows)>=count,(split,len(rows))
        groups[split]=rows[:count]
        target=out/f'{split}.jsonl'
        target.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in groups[split]))
    assert not set(map(key,groups['train'])) & set(map(key,groups['screen']))
    write(out/'data.json',dict(parent_manifest=str(base/'manifest.json'),parent_sha256=sha(base/'manifest.json'),
          selection='seed20261018 shuffled unique observed OAS H/L pairs within fixed capacities',
          files={s:dict(path=str(out/f'{s}.jsonl'),sha256=sha(out/f'{s}.jsonl'),records=len(rows)) for s,rows in groups.items()},
          labels='No GDPa1 labels or benchmark rows read by this preparation'))
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    groups=prepare_data(out)
    checkpoint=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    source=Path(__file__).resolve()
    arms={'denoise':(1.,0.),'joint':(1.,1.),'representation':(0.,1.)}
    seed,steps,batch_size=42,300,16
    rng=random.Random(seed)
    schedule=[]
    order=[]
    while len(schedule)<steps:
        order=list(range(len(groups['train'])));rng.shuffle(order)
        schedule.extend([order[i:i+batch_size] for i in range(0,len(order),batch_size)])
    schedule=schedule[:steps]
    write(out/'schedule.json',schedule)
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(checkpoint),
          steps=steps,batch_size=batch_size,seed=seed,learning_rate=2e-5,warmup=20,
          optimizer='AdamW betas(0.9,0.999), eps1e-8, weight_decay0.01; constant LR after warmup; grad clip1',
          arms=arms,clean_representation_loss='1-CKA(pooled,teacher) +0.1*mean token relation loss + pool variance floor',
          variance_floor='student global centered pooled variance >=0.25*teacher variance; one-sided penalty',
          teacher='original embedding and blocks1-2 frozen, zero dropout; shared computation, stop-gradient',
          trainable='blocks3-8 and final RMSNorm; embedding, blocks1-2 and vocabulary head frozen',
          input='unchanged native segment/global topology and all attended MASK tails',
          evaluation='fixed final300 only, all arms; native final RMSNorm hidden and observed-residue global pooling',
          source_sha256={str(source):sha(source),str(ROOT/'downstream/fixed_chain_eval.py'):sha(ROOT/'downstream/fixed_chain_eval.py'),
                         str(ROOT/'downstream/grammar/ab_probes.py'):sha(ROOT/'downstream/grammar/ab_probes.py')},
          schedule_sha256=sha(out/'schedule.json'),data_sha256=sha(out/'data.json'),
          execution='local user-authorized bounded pilot; no Volc submission; matched updates, joint uses two forwards')
    write(out/'plan.json',plan)
    status=dict(status='running',stage='load',pid=os.getpid(),host=socket.gethostname(),started_utc=plan['created_utc'])
    write(out/'status.json',status)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from downstream.fixed_chain_eval import load,FixedChainBackbone
    torch.set_num_threads(4)
    model,identity=load(checkpoint,device='cuda')
    from dllm.pipelines.immune_llada.fixed_chain.layout import FixedChainCollator
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain
    from downstream.fixed_chain_layer_eval import block_hidden
    assert all(getattr(model.backbone.config,k)==0. for k in ('attention_dropout','residual_dropout','embedding_dropout'))
    reference={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    allowed_prefixes=tuple(f'backbone.transformer.blocks.{i}.' for i in range(2,8))+('backbone.transformer.ln_f.',)
    train_names={k for k,p in model.named_parameters() if k.startswith(allowed_prefixes)}
    frozen_names=set(dict(model.named_parameters()))-train_names
    write(out/'identity.json',{**identity,'trainable_names':sorted(train_names),
          'trainable_count':sum(p.numel() for n,p in model.named_parameters() if n in train_names)})
    collator=FixedChainCollator.from_config(model.config,mode='train')
    representation=FixedChainBackbone(model,identity)
    # Only sequence records are needed by the representation path; original train
    # records retain native target supervision through the train collator.
    def records(rows):return [BioSeqRecord.from_prepared_dict(r) for r in rows]
    def device(batch):return {k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
    def pool(h,mask):return h.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)
    def gram(x):
        x=x.float()-x.float().mean(-2,keepdim=True)
        return x@x.transpose(-1,-2)
    def relation(s,t):
        ks,kt=gram(s),gram(t.detach())
        numerator=(ks*kt).sum(dim=(-1,-2))
        denominator=(ks.square().sum(dim=(-1,-2))*kt.square().sum(dim=(-1,-2))).clamp_min(1e-20).sqrt()
        return (1-numerator/denominator).mean()
    def clean_forward(rows,grad):
        batch=device(representation.collator(records(rows)))
        captured=[]
        hook=model.backbone.transformer.blocks[1].register_forward_hook(
            lambda module,inputs,result:captured.append(block_hidden(result).detach()))
        try:
            with torch.set_grad_enabled(grad),torch.autocast('cuda',dtype=torch.bfloat16):
                _,hidden=model.denoise(**model.feature_args(batch),output_hidden_states=True)
        finally:hook.remove()
        assert len(captured)==1 and not captured[0].requires_grad
        mask=batch['residue_mask'] & batch['attention_mask']
        teacher=captured[0]
        student_pool,teacher_pool=pool(hidden,mask),pool(teacher,mask)
        sp,tp=[],[]
        for i in range(len(rows)):
            available=mask[i].nonzero().flatten()
            indices=available[torch.linspace(0,len(available)-1,32,device='cuda').long()]
            sp.append(hidden[i,indices]);tp.append(teacher[i,indices])
        pooled_loss=relation(student_pool,teacher_pool)
        token_loss=relation(torch.stack(sp),torch.stack(tp))
        sv=(student_pool-student_pool.mean(0)).square().mean()
        tv=(teacher_pool-teacher_pool.mean(0)).square().mean().detach()
        variance_penalty=F.relu(.25*tv-sv)/(tv+1e-12)
        loss=pooled_loss+.1*token_loss+variance_penalty
        return loss,dict(pooled=float(pooled_loss.detach()),token=float(token_loss.detach()),
                         variance_penalty=float(variance_penalty.detach()),variance_ratio=float((sv/tv).detach())),student_pool,teacher_pool
    def validate(step):
        model.eval()
        all_s,all_t=[],[]
        token_losses=[]
        with torch.no_grad():
            for start in range(0,len(groups['screen']),batch_size):
                rows=groups['screen'][start:start+batch_size]
                _,detail,s,t=clean_forward(rows,False)
                all_s.append(s.cpu());all_t.append(t.cpu());token_losses.append(detail['token'])
        s,t=torch.cat(all_s),torch.cat(all_t)
        centered=s-s.mean(0)
        energy=torch.linalg.svdvals(centered.double()).square();p=energy/energy.sum()
        metrics=dict(step=step,heldout_pool_cka=float(1-relation(s,t)),
                     heldout_token_relation_loss=float(np.mean(token_losses)),
                     effective_rank=float(torch.exp(-(p*p.clamp_min(1e-30).log()).sum())),
                     feature_centered_variance=float(centered.square().mean()))
        # Same validation masks at every check without perturbing training RNG.
        for ratio in (.5,1.):
            total,correct,count=0.,0,0
            with torch.random.fork_rng(devices=[0]),torch.no_grad():
                torch.manual_seed(87913)
                for start in range(0,len(groups['screen']),batch_size):
                    batch=device(collator(groups['screen'][start:start+batch_size]))
                    token_t=torch.full_like(batch['input_ids'],ratio,dtype=torch.float)
                    noised,targets,selected=model.corrupt(batch,token_t)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits,_=model.denoise(**{**model.feature_args(batch),'input_ids':noised})
                    total+=float(F.cross_entropy(logits[selected].float(),targets[selected],reduction='sum'))
                    aa=selected & batch['residue_mask']
                    correct+=int((logits.argmax(-1)[aa]==targets[aa]).sum());count+=int(aa.sum())
                    # CE uses all selected including observed EOS; keep its own denominator.
                    metrics.setdefault(f'ce_count_{ratio}',0);metrics[f'ce_count_{ratio}']+=int(selected.sum())
            metrics[f'ce_{ratio}']=total/metrics[f'ce_count_{ratio}']
            metrics[f'aa_accuracy_{ratio}']=correct/count
        model.train()
        return metrics
    baseline=validate(0);write(out/'baseline_validation.json',baseline)
    # Necessary loss/teacher contracts before any update.
    assert abs(float(relation(torch.randn(8,32),torch.randn(8,32))))<=1.001
    check_x=torch.randn(16,32)
    assert abs(float(relation(check_x,check_x)))<1e-6
    results={}
    for arm,(ce_weight,repr_weight) in arms.items():
        folder=out/arm;folder.mkdir()
        model.load_state_dict(reference,strict=True)
        for n,p in model.named_parameters():p.requires_grad_(n in train_names)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,weight_decay=.01)
        model.train();torch.manual_seed(seed)
        history=[baseline]
        started=time.monotonic()
        status.update(arm=arm,stage='train',step=0);write(out/'status.json',status)
        for step,indices in enumerate(schedule,1):
            rows=[groups['train'][i] for i in indices]
            torch.manual_seed(91000+step)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=2e-5*min(step/20,1.)
            ce=torch.zeros((),device='cuda');rep=torch.zeros((),device='cuda');detail={}
            if ce_weight:
                batch=device(collator(rows))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    ce=model(**batch)['loss']
                (ce_weight*ce).backward()
            if repr_weight:
                rep,detail,_,_=clean_forward(rows,True)
                (repr_weight*rep).backward()
            grad_norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(ce) and torch.isfinite(rep) and torch.isfinite(grad_norm)
            optimizer.step()
            if step==1 or step%10==0:
                row=dict(step=step,ce=float(ce.detach()),representation_loss=float(rep.detach()),
                         gradient_norm=float(grad_norm),seconds=time.monotonic()-started,**detail)
                with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                status.update(step=step,last_train=row);write(out/'status.json',status)
                print(arm,row,flush=True)
            if step%100==0:
                history.append(validate(step));write(folder/'validation.json',history)
        for n,p in model.named_parameters():
            if n in frozen_names:assert torch.equal(p.detach().cpu(),reference[n]),n
        changed=[n for n,p in model.named_parameters() if n in train_names and not torch.equal(p.detach().cpu(),reference[n])]
        assert changed
        state={n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if n in train_names}
        save_file(state,str(folder/'trained_parameters.safetensors'),metadata={'parent_sha256':identity['model_sha256'],'arm':arm})
        write(folder/'checkpoint.json',dict(parent_checkpoint=str(checkpoint),parent_identity=identity,
              parameter_file=str(folder/'trained_parameters.safetensors'),sha256=sha(folder/'trained_parameters.safetensors'),
              trainable_names=sorted(train_names),changed_names=changed,steps=steps,
              frozen_parameters_bitwise_equal=True,source_sha256=plan['source_sha256']))
        model.eval().requires_grad_(False)
        status.update(stage='gdp_features');write(out/'status.json',status)
        # Only fixed final checkpoints are now evaluated, without adapting any weights.
        from downstream.grammar.ab_probes import prepare_task,run_gdp
        from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL
        from scipy.stats import spearmanr
        from types import SimpleNamespace
        df,pairs,_,_=prepare_task('gdp_a1')
        rec=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],
                          task_type='antibody',source='native_v5_probe') for h,l in pairs]
        features=representation.vectors(rec)
        np.save(folder/'final_features.npy',features)
        gdpfolder=folder/'gdp_a1';gdpfolder.mkdir()
        metrics=run_gdp(df,torch.from_numpy(features.astype(np.float64)),SimpleNamespace(out_dir=gdpfolder))
        write(gdpfolder/'metrics.json',metrics)
        checks=[]
        for col in ALL_COLS:
            saved=np.load(gdpfolder/col/'legacy_oof.npz');valid=df[col].notna().to_numpy()
            np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid))
            np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL].to_numpy())
            np.testing.assert_array_equal(saved['y_true'],df.loc[valid,col].to_numpy())
            scores=[float(spearmanr(saved['y_transformed'][saved['fold']==i],saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
            assert abs(np.mean(scores)-metrics['properties'][col]['legacy_reference']['cv_mean_spearman'])<1e-12
            checks.append(dict(property=col,fold_scores=scores))
        write(folder/'verification.json',dict(status='passed',frozen_parameters_bitwise_equal=True,oof_checks=checks))
        results[arm]=dict(validation=history[-1],gdp=metrics,checkpoint_sha256=sha(folder/'trained_parameters.safetensors'))
        write(out/'results.json',results)
        del optimizer
        torch.cuda.empty_cache()
    for p,digest in plan['source_sha256'].items():assert sha(p)==digest,p
    for s,item in json.loads((out/'data.json').read_text())['files'].items():assert sha(item['path'])==item['sha256']
    status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',status)


if __name__=='__main__':main()
