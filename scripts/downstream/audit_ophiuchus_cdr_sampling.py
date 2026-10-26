"""Local official-weight sampling audit: activate pllm, then protenix_abtcr; python -B FILE --run ABS.

Uses the existing checkpoint-compatible Ophiuchus backbone and executes AirGen's
original decoder/collator methods extracted from source. Saves every prediction;
never changes existing evaluations, weights, training, or shared scoring code.
"""
import argparse, ast, hashlib, json, math, random, sys, time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/project_c20250601/251105016/project/dllm_test')
AIR=ROOT.parent/'airgen/AirGen-Dev'
sys.path.insert(0,str(ROOT))
from dllm.pipelines.bioseq import MultiChainOphiuchusAbModel, load_ophiuchus_checkpoint
from downstream.ophiuchus_eval.cdr_sabdab import SAbDabDataset
from downstream.ophiuchus_eval.cdr_sab23h2 import SAb23H2Dataset
PAPER={'kong':{'cdrh1':75.50,'cdrh2':70.18,'cdrh3':43.55},'sab23':{'cdrh1':74.8,'cdrh2':68.6,'cdrh3':36.8,'cdrl1':81.1,'cdrl2':80.1,'cdrl3':73.7}}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def official_decoder(model):
    source=AIR/'src/byprot/models/lm/dplm_multichain.py';util=AIR/'src/byprot/models/lm/model_utils.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MultiChainDiffusionProteinLanguageModel')
    names={'generate','forward_decoder','forward_encoder','initialize_output_tokens','get_non_special_sym_mask','_decoding'}
    fnames={'topk_masking','sample_from_categorical','stochastic_sample_from_categorical'}
    nodes=[n for n in ast.parse(util.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in fnames]
    nodes += [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    env=dict(torch=torch,np=np,math=math,tqdm=lambda x,**k:x)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    obj=type('OriginalAirGenDecoder',(),{n:env[n] for n in names})()
    obj.net=model.net
    for n in ['mask_id','pad_id','bos_id','eos_id','special_token_list']:setattr(obj,n,getattr(model,n))
    return obj

def collator(model,dataset):
    name='zeroshot_cdr.py' if dataset=='kong' else 'zeroshot_SAb23H2.py'
    path=AIR/'downstream/infill'/name
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MyCollateFn')
    provider=type('LocalTokenizerProvider',(),{'from_pretrained':staticmethod(lambda *a,**k:model.tokenizer)})
    env=dict(torch=torch,deepcopy=deepcopy,random=random,EsmTokenizer=provider)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env['MyCollateFn']()

def summarize(run,results):
    write(run/'summary.json',dict(status='running',results=results))
    lines=['# Ophiuchus-Ab official-weight local sampling audit','','Official decoder methods; local checkpoint-compatible backbone. CFG=0, temperature=1, seed=42. This is a test-set protocol sensitivity analysis, not recovery of unpublished paper settings.','','| Dataset | Sampling | Iter | Precision | H1 | H2 | H3 | L1 | L2 | L3 | MAE to paper (pp) |','|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in results:
        vals=[f"{x['modes'][m]['aar_percent']:.4f}" if m in x['modes'] else '—' for m in ['cdrh1','cdrh2','cdrh3','cdrl1','cdrl2','cdrl3']]
        lines.append('| '+' | '.join([x['dataset'],x['strategy'],str(x['iteration']),x['precision'],*vals,f"{x['mae_pp']:.4f}"])+' |')
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;run.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    ck=ROOT/'model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt'
    sources=[Path(__file__),ROOT/'downstream/common.py',ROOT/'downstream/ophiuchus_eval/cdr_sabdab.py',ROOT/'downstream/ophiuchus_eval/cdr_sab23h2.py',AIR/'src/byprot/models/lm/dplm_multichain.py',AIR/'src/byprot/models/lm/model_utils.py',AIR/'downstream/infill/zeroshot_cdr.py',AIR/'downstream/infill/zeroshot_SAb23H2.py']
    sources+=list((ROOT/'dllm/pipelines/bioseq/ophiuchus').rglob('*.py'))
    matrix=[(d,s,n,'bf16' if d=='kong' else 'fp32') for d in ['sab23','kong'] for s,n in [('argmax',1),('argmax',2),('argmax',4),('vanilla',1),('gumbel_argmax',1)]]+[('sab23','argmax',2,'bf16')]
    write(run/'plan.json',dict(matrix=matrix,paper=PAPER,seed=42,batch_size=64,temperature=1,cfg_scale=0,checkpoint=str(ck),checkpoint_sha256=sha(ck),code_sha256={str(f):sha(f) for f in sources},scope='all records, no per-CDR selection; stochastic methods single-seed diagnostic'))
    print('Loading official weights',flush=True)
    model=MultiChainOphiuchusAbModel();missing,unexpected=load_ophiuchus_checkpoint(model,ck,device='cpu',strict=True)
    assert not missing and not unexpected
    model=model.eval().requires_grad_(False).cuda();decoder=official_decoder(model)
    write(run/'load.json',dict(missing=missing,unexpected=unexpected,parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name()))
    datasets={};files={}
    for d in PAPER:
        for mode in PAPER[d]:
            for fold in range(10 if d=='kong' else 1):
                base=ROOT/'data/downstream/cdr_infilling'
                if d=='kong':
                    path=base/'sabdab_kong'/mode/f'fold_{fold}/test.json';files[str(path)]=sha(path);ds=SAbDabDataset(str(base/'sabdab_kong'),mode,fold)
                else:
                    base=base/'sab23h2/SAb-23-H2-Ab';ds=SAb23H2Dataset(str(base),mode)
                    for path in [base/'prot_ids.txt',*Path(ds.design_path).glob('*.fasta'),*Path(ds.native_path).glob('*.fasta')]:files[str(path)]=sha(path)
                datasets[d,mode,fold]=ds
    write(run/'data_hashes.json',files)
    # Exact original collator and original decoder vs historical local decoder.
    gate={}
    with torch.inference_mode():
        for d in PAPER:
            ds=datasets[d,'cdrh3',0];items=[ds[i][:4] for i in range(2)];tokens,chains,labels=collator(model,d)(items);tokens=tokens.cuda();chains=chains.cuda();partial=tokens.ne(model.mask_id)&tokens.ne(model.bos_id)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                kw=dict(batch={'input_ids':tokens,'chain_ids':chains},max_iter=2,temperature=1.,sampling_strategy='argmax',partial_masks=partial,cfg_scale=0.)
                out=decoder.generate(**kw)[0];old=model.generate(**kw)[0]
            assert out.equal(old),'Official vs local decoder drift'
            assert out[partial].equal(tokens[partial]);gate[d]={'decoder_equal_iter2':True,'unchanged_context':True}
    write(run/'gate.json',gate)
    results=[]
    with torch.inference_mode():
        for d,strategy,it,precision in matrix:
            key=f'{d}_{strategy}_iter{it}_{precision}';folder=run/key;folder.mkdir();t0=time.monotonic();torch.manual_seed(42);np.random.seed(42);random.seed(42);modes={}
            write(run/'state.json',dict(status='running',current=key,completed=len(results)));print('START',key,flush=True)
            with (folder/'predictions.jsonl').open('w') as out:
                for mode in PAPER[d]:
                    folds=[];n=0;batchcollate=collator(model,d)
                    for fold in range(10 if d=='kong' else 1):
                        ds=datasets[d,mode,fold];values=[]
                        for start in range(0,len(ds),64):
                            items=[ds[i][:4] for i in range(start,min(start+64,len(ds)))];tokens,chains,labels=batchcollate(items);tokens=tokens.cuda();chains=chains.cuda();labels=labels.cuda();mask=tokens.eq(model.mask_id);partial=tokens.ne(model.mask_id)&tokens.ne(model.bos_id)
                            ctx=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
                            with ctx:pred=decoder.generate(batch={'input_ids':tokens,'chain_ids':chains},max_iter=it,temperature=1.,sampling_strategy=strategy,partial_masks=partial,cfg_scale=0.)[0]
                            assert pred[partial].equal(tokens[partial]);assert not pred[mask].eq(model.mask_id).any();correct=((pred==labels)&mask).sum(1);length=mask.sum(1);assert (length>0).all();aar=(correct/length).cpu().tolist();values+=aar
                            for j,v in enumerate(aar):out.write(json.dumps(dict(mode=mode,fold=fold,row=start+j,n_correct=int(correct[j]),n_masked=int(length[j]),aar=v,prediction_ids=pred[j,mask[j]].tolist(),reference_ids=labels[j,mask[j]].tolist()))+'\n')
                        folds.append(float(np.mean(values)*100));n+=len(values)
                    score=float(np.mean(folds)) if d=='kong' else float(np.round(folds[0]/100,4)*100)
                    modes[mode]=dict(aar_percent=score,n=n,fold_means_percent=folds,error_pp=score-PAPER[d][mode]);out.flush();print(key,mode,score,flush=True)
            result=dict(dataset=d,strategy=strategy,iteration=it,precision=precision,modes=modes,seconds=time.monotonic()-t0,mae_pp=float(np.mean([abs(x['error_pp']) for x in modes.values()])))
            write(folder/'metrics.json',result);results.append(result);summarize(run,results)
    write(run/'state.json',dict(status='complete',completed=len(results)));write(run/'summary.json',dict(status='complete',results=results));print('COMPLETE',flush=True)
if __name__=='__main__':main()
