"""Frozen AB confirmation: audited Fv inputs, official five folds, fixed 100-epoch heads."""
import hashlib
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset
from transformers.optimization import get_linear_schedule_with_warmup
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator
from .common import read,write,sha,digest
from .evaluate import amp

DATA=Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/data/downstream')

def clean(x):
    if not isinstance(x,str): raise ValueError('Nonstring antibody')
    x=''.join(x.split()).upper().replace('-','').replace('J','L')
    if not x or set(x)-set('ACDEFGHIKLMNPQRSTVWYBXZOU'): raise ValueError('Invalid antibody')
    return x

def inputs():
    """Match the existing ab_probes loader and ab_fv_input audited boundaries."""
    base=DATA/'specificity'; path=base/'hd-0_flu-1_cov-2.csv'; df=pd.read_csv(path)
    if df.name.duplicated().any() or set(df.label)!={0,1,2}: raise ValueError('Specificity IDs/classes invalid')
    pairs=list(zip(map(clean,df.h_sequence),map(clean,df.l_sequence)))
    names={str(n):i for i,n in enumerate(df.name)}; folds=[]; seen=np.zeros(len(df),int); paths=[path]
    for k in range(5):
        split=[]
        for side in ('train','test'):
            p=base/'TTE'/f'hd-0_flu-1_cov-2_{side}{k}.csv'; paths.append(p); frame=pd.read_csv(p)
            if frame.name.duplicated().any(): raise ValueError('Duplicate fold IDs')
            ids=np.array([names[str(n)] for n in frame.name]); ps=list(zip(map(clean,frame.h_sequence),map(clean,frame.l_sequence)))
            if ps!=[pairs[i] for i in ids] or not np.array_equal(frame.label,df.label.iloc[ids]): raise ValueError('Fold mismatch')
            split.append(ids)
        tr,te=split
        if set(tr)&set(te) or set(tr)|set(te)!=set(range(len(df))): raise ValueError('Fold is not an exact partition')
        if {pairs[i] for i in tr}&{pairs[i] for i in te}: raise ValueError('Full sequence pair leakage')
        seen[te]+=1; folds.append((tr,te))
    if not (seen==1).all(): raise ValueError('Not exactly one OOF test per record')
    path=DATA/'ab_fixed_v2_fv_manifest.json'; manifest=read(path); paths.append(path); adapted=[]; changed=0
    for pair in pairs:
        output=[]
        for seq,role,cap in zip(pair,('h','l'),(166,134)):
            value=seq
            if len(seq)>cap:
                rule=manifest['overflow'].get(hashlib.sha256(seq.encode()).hexdigest())
                if rule is None or rule['role']!=role or rule['sequence']!=seq: raise ValueError('Unverified overflow chain')
                value=seq[rule['start']:rule['end']]
            if not value or len(value)>cap: raise ValueError('Audited Fv exceeds native capacity')
            output.append(value)
        changed+=tuple(output)!=pair; adapted.append(tuple(output))
    for tr,te in folds:
        if {adapted[i] for i in tr}&{adapted[i] for i in te}: raise ValueError('Fv sequence pair leakage')
    return df,adapted,folds,dict(policy='fixed-v2-fv',boundary_method='ANARCI_IMGT',changed_pairs=changed,
                                data_sha256={str(p):sha(p) for p in paths})

class Head(torch.nn.Module):
    def __init__(self,width):
        super().__init__(); self.dense=torch.nn.Linear(width,width); self.out_proj=torch.nn.Linear(width,3)
    def forward(self,x): return self.out_proj(torch.tanh(self.dense(x)))

def evaluate_ab(model,device,output,expected_audit):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    df,pairs,folds,audit=inputs()
    if audit!=expected_audit: raise ValueError('Protected AB data changed since pretraining freeze')
    write(output/'input_audit.json',audit)
    features=[]; model.eval(); collator=VFamilyCollator(model.config,family_spec=model.family_spec,hide_relation=True)
    with torch.no_grad():
        for start in range(0,len(pairs),16):
            records=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],'antibody','ab_specificity') for h,l in pairs[start:start+16]]
            batch={k:v.to(device) for k,v in collator(records).items()}; mask=batch['residue_mask']&batch['attention_mask']
            if mask.sum(1).tolist()!=[sum(map(len,p)) for p in pairs[start:start+16]]: raise ValueError('Lost AB residues')
            with amp(device): _,hidden=model.denoise(**model.feature_args(batch),output_hidden_states=True)
            features.append((hidden.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)).cpu())
    x=torch.cat(features); y=torch.tensor(df.label.to_numpy(),dtype=torch.long)
    np.save(output/'features.npy',x.numpy()); scores=[]; predictions=[]
    for k,(tr,te) in enumerate(folds):
        seed=42+k; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        head=Head(x.shape[1]).to(device); opt=torch.optim.AdamW(head.parameters(),lr=5e-5)
        loader=DataLoader(TensorDataset(x[tr],y[tr]),batch_size=8,shuffle=True)
        scheduler=get_linear_schedule_with_warmup(opt,int(.1*100*len(loader)),100*len(loader)); history=[]
        for epoch in range(100):
            head.train(); total=0.
            for xb,yb in loader:
                opt.zero_grad(set_to_none=True); loss=torch.nn.functional.cross_entropy(head(xb.to(device)),yb.to(device))
                loss.backward(); opt.step(); scheduler.step(); total+=float(loss.detach())*len(xb)
            history.append(dict(epoch=epoch+1,train_loss=total/len(tr)))
        head.eval(); file=output/f'head_fold_{k}.pt'; torch.save(head.state_dict(),file)
        restored=Head(x.shape[1]).to(device).eval(); restored.load_state_dict(torch.load(file,weights_only=True,map_location=device))
        with torch.no_grad():
            logits=restored(x[te].to(device)).cpu(); original=head(x[te].to(device)).cpu()
        torch.testing.assert_close(logits,original,atol=0,rtol=0); pred=logits.argmax(-1).numpy()
        score=dict(fold=k,accuracy=float(accuracy_score(y[te],pred)),f1_macro=float(f1_score(y[te],pred,average='macro')),
                   mcc=float(matthews_corrcoef(y[te],pred))); scores.append(score)
        write(output/f'fold_{k}.json',dict(metrics=score,history=history,train_ids=df.name.iloc[tr].tolist(),test_ids=df.name.iloc[te].tolist(),head_sha256=sha(file)))
        predictions.extend(dict(id=str(df.name.iloc[i]),fold=k,label=int(y[i]),prediction=int(p),logits=l) for i,p,l in zip(te,pred,logits.tolist()))
    result=dict(protocol='native_ab_fixed_last_head_5fold_v1',input_audit=audit,epochs=100,head_batch=8,lr=5e-5,selection='fixed_last_no_test_selection',
                pooling='clean receptor observed-residue global mean; masked relation',folds=scores,
                mean={k:float(np.mean([s[k] for s in scores])) for k in ('accuracy','f1_macro','mcc')},
                sample_sd={k:float(np.std([s[k] for s in scores],ddof=1)) for k in ('accuracy','f1_macro','mcc')},
                note='Fv protocol; do not compare as unchanged full-chain benchmark. Secondary confirmation only; never candidate selection.')
    write(output/'predictions.json',predictions); write(output/'metrics.json',result); return result
