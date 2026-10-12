"""AirGen-aligned CDR evaluation: activate pllm; python -B FILE --audit or --dataset kong|sab23.

Reuse AirGen's actual generation methods via AST extraction, not a substitute
sampler. Its 150/128 token grids map residue-for-residue to the trained native
V5 object canvas. Native vocabulary/architecture remain model differences.
"""
import argparse,ast,hashlib,json,math,sys,time
from pathlib import Path
from contextlib import nullcontext
ROOT=Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
AIR=ROOT.parent/'airgen/AirGen-Dev'
OUT=Path(__file__).resolve().parent
CKPT=OUT/'checkpoint'
sys.path.insert(0,str(ROOT))
MODES={'kong':['cdrh1','cdrh2','cdrh3'],'sab23':['cdrh3','cdrh2','cdrh1','cdrl3','cdrl2','cdrl1']}
STEPS={'kong':4,'sab23':2}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def records(dataset):
    base=ROOT/'data/downstream/cdr_infilling';files={};rows=[]
    if dataset=='kong':
        for mode in MODES[dataset]:
            for fold in range(10):
                p=base/'sabdab_kong'/mode/f'fold_{fold}/test.json';files[str(p)]=sha(p)
                for i,line in enumerate(p.read_text().splitlines()):
                    d=json.loads(line);h=d['heavy_chain_seq'].replace('J','L');l=d['light_chain_seq'].replace('J','L');start,end=d[mode+'_pos']
                    assert 0<=start<=end<len(h)
                    # Position metadata is authoritative, never substring search.
                    positions=list(range(start,end+1))
                    rows.append(dict(mode=mode,fold=fold,row=i,key=d.get('pdb',str(i)),heavy=h,light=l,mask_h=positions,mask_l=[],gold_h=h,gold_l=l,source=str(p)))
    else:
        raw=base/'sab23h2/SAb-23-H2-Ab';p=raw/'prot_ids.txt';files[str(p)]=sha(p);ids=p.read_text().splitlines()
        assert len(ids)==60
        official=AIR/'downstream/infill/IgGM_Test_set/SAb-23-H2-Ab/prot_ids.txt'
        assert ids==official.read_text().splitlines();files[str(official)]=sha(official)
        for mode in MODES[dataset]:
            for i,key in enumerate(ids):
                design_mode=mode[3]+'_cdr'+mode[4]
                d=raw/f'fasta.files.design/{design_mode}/{key}.fasta';n=raw/f'fasta.files.native/{key}.fasta'
                files[str(d)]=sha(d);files[str(n)]=sha(n)
                design=d.read_text().splitlines();native=n.read_text().splitlines()
                h,l=[design[k].strip().replace('J','L') for k in (1,3)]
                gh,gl=[native[k].strip().replace('J','L') for k in (1,3)]
                assert len(h)==len(gh) and len(l)==len(gl)
                rows.append(dict(mode=mode,fold=0,row=i,key=key,heavy=h,light=l,mask_h=[j for j,x in enumerate(h) if x=='X'],mask_l=[j for j,x in enumerate(l) if x=='X'],gold_h=gh,gold_l=gl,source=str(d)))
    for row in rows:
        assert len(row['heavy'])+2<=150 and len(row['light'])+2<=128,('AirGen fixed-grid overflow',row['key'])
        assert row['mask_h'] or row['mask_l']
    return rows,files

def methods():
    import torch,numpy as np
    source=AIR/'src/byprot/models/lm/dplm_multichain.py';util=AIR/'src/byprot/models/lm/model_utils.py'
    tree=ast.parse(source.read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiChainDiffusionProteinLanguageModel')
    names={'generate','forward_decoder','initialize_output_tokens','get_non_special_sym_mask','_decoding'}
    nodes=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in names]
    top=next(n for n in ast.parse(util.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='topk_masking')
    env=dict(torch=torch,np=np,math=math,tqdm=lambda x,**kw:x)
    exec(compile(ast.Module(body=[top,*nodes],type_ignores=[]),str(source),'exec'),env)
    return {name:env[name] for name in names}

def inputs(model,rows):
    import torch
    from dllm.pipelines.immune_llada.data.records import BioSeqChain,BioSeqRecord
    from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
    tok=model.tokenizer;end=tok.special_id('<protd>');bos=tok.special_id('<prots>');mask=tok.mask_token_id
    virtual=torch.full((len(rows),278),end,dtype=torch.long);gold=virtual.clone();target=torch.zeros_like(virtual,dtype=torch.bool);records=[];maps=[]
    for i,row in enumerate(rows):
        chains=[];mapping=[];offset=0
        for role,seq,label,positions,base in [('antibody_heavy',row['heavy'],row['gold_h'],row['mask_h'],0),('antibody_light',row['light'],row['gold_l'],row['mask_l'],150)]:
            virtual[i,base]=bos;gold[i,base]=bos
            virtual[i,base+1:base+1+len(seq)]=torch.tensor(tok.encode_residues(seq),dtype=torch.long)
            gold[i,base+1:base+1+len(seq)]=torch.tensor(tok.encode_residues(label),dtype=torch.long)
            for j in positions:virtual[i,base+1+j]=mask;target[i,base+1+j]=True
            if seq:
                # Hide target letters before constructing any native features.
                placeholder=''.join('A' if j in positions else c for j,c in enumerate(seq))
                chains.append(BioSeqChain(placeholder,role))
                mapping += [(base+1+j,offset+j) for j in range(len(seq))]
                offset+=len(seq)+1
        records.append(BioSeqRecord(chains,task_type='antibody',source='airgen_cdr_aligned'));maps.append(mapping)
    batch=ImmuneCollator.from_config(model.config,mode='representation')(records);features=model.feature_args(batch)
    for i,mapping in enumerate(maps):
        slots=features['target_slot_mask'][i].nonzero().flatten()
        maps[i]=[(v,int(slots[n])) for v,n in mapping]
        for v,n in maps[i]:features['input_ids'][i,n]=virtual[i,v]
    chain_ids=torch.zeros_like(virtual);chain_ids[:,150:]=1
    return features,dict(input_ids=virtual,chain_ids=chain_ids),gold,target,maps

def decoder(model,features,maps):
    import torch
    cls=type('AirGenNativeBridge',(),methods());bridge=cls();tok=model.tokenizer
    bridge.mask_id=tok.mask_token_id;bridge.pad_id=tok.pad_token_id;bridge.bos_id=tok.special_id('<prots>');bridge.eos_id=tok.special_id('<protd>')
    # Native grammar has no '-' token. This unrepresentable ESM token is documented.
    allowed=set(tok.encode_residues('ACDEFGHIKLMNPQRSTVWY'))|{bridge.eos_id,tok.chain_separator_id()}
    bridge.special_token_list=[n for n in range(tok.vocab_size) if n not in allowed]
    bridge.forward_encoder=lambda batch:None
    def net(input_ids,chain_ids):
        x={k:v.clone() for k,v in features.items()}
        for i,mapping in enumerate(maps):
            vi,ni=zip(*mapping);x['input_ids'][i,list(ni)]=input_ids[i,list(vi)]
        logits,_=model.denoise(**x)
        result=logits.new_zeros((*input_ids.shape,logits.shape[-1]))
        for i,mapping in enumerate(maps):
            vi,ni=zip(*mapping);result[i,list(vi)]=logits[i,list(ni)]
        return {'logits':result,'last_hidden_state':None}
    bridge.net=net
    return bridge

def report():
    import fcntl
    start='<!-- airgen-aligned-cdr:start -->';end='<!-- airgen-aligned-cdr:end -->'
    lines=[start,'## 当前模型 CDR infilling：AirGen 协议对齐重评','',
    '固定best45000。Kong H1/H2/H3：官方位置字段、10折、argmax/4步；SAb23H2六CDR：原始design FASTA的X掩码、argmax/2步（AirGen示例命令）。batch64、temperature1、CFG0；直接复用AirGen解码方法，按原始logits置信度线性填充及其阈值并列规则。',
    '这是任务/解码协议对齐，不是模型实现相同：保留训练过的V5对象302布局和50词表；AirGen的ESM词表中gap `-` 在本模型中不存在。旧iter1/首次子串匹配结果仅作历史对照，不作为本次对齐结果。Kong仍为既有本地重建划分，不宣称已复现未公开的论文运行参数。','',
    '| 数据集 | 状态 | 各CDR AAR (%) | 证据 |','|---|---|---|---|']
    for d in MODES:
        p=OUT/d/'status.json';state=json.loads(p.read_text()) if p.exists() else {'status':'pending'};metrics=OUT/d/'metrics.json';scores='待结果'
        if metrics.exists():scores='；'.join(k+' '+str(v['airgen_aar_percent']) for k,v in json.loads(metrics.read_text())['modes'].items())
        lines.append(f"| {d} | {state['status']} | {scores} | [目录]({OUT/d}) |")
    lines.append(end);block='\n'.join(lines);target=ROOT/'downstream/benchmark/RESULTS.md'
    # Share the original suite writer's lock.
    with (ROOT/'output/downstream_generation/v5_native_best_20261008/results_update.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);s=target.read_text()
        if start in s:a=s.index(start);b=s.index(end,a)+len(end);s=s[:a]+block+s[b:]
        else:head,tail=s.split('\n',1);s=head+'\n\n'+block+'\n'+tail
        temp=target.with_suffix('.aligned.tmp');temp.write_text(s);temp.replace(target)

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--dataset',choices=MODES);a=p.parse_args()
    if a.audit:
        audit={}
        sources=[AIR/'downstream/infill/zeroshot_cdr.py',AIR/'downstream/infill/zeroshot_SAb23H2.py',AIR/'run/zero_shot_test.sh',AIR/'src/byprot/models/lm/dplm_multichain.py',AIR/'src/byprot/models/lm/model_utils.py',Path(__file__)]
        for d in MODES:
            rows,files=records(d);audit[d]={'records':len(rows),'missing_light':sum(not x['light'] for x in rows),'files':files,'steps':STEPS[d]}
        write(OUT/'protocol_audit.json',{'status':'passed','datasets':audit,'code_sha256':{str(p):sha(p) for p in sources},'differences':['native V5 object302 layout','native vocabulary has no gap -','SAb raw folder aliases h_cdr1 etc map to AirGen cdrh1 etc; original files unchanged'],'scope':'current checkpoint all 9 CDR targets'});report();print({k:v['records'] for k,v in audit.items()});return
    audit=json.loads((OUT/'protocol_audit.json').read_text())
    for path,h in audit['code_sha256'].items():assert sha(path)==h,path
    from downstream.v5_native_eval import load
    import torch,numpy as np
    model,identity=load(CKPT,'cpu' if a.check else 'cuda')
    if a.check:
        results={}
        for d in MODES:
            rows,_=records(d);chosen=[next(x for x in rows if x['mode']==mode) for mode in MODES[d]]
            if d=='kong':chosen.append(next(x for x in rows if not x['light']))
            f,b,gold,mask,maps=inputs(model,chosen);bridge=decoder(model,f,maps)
            with torch.inference_mode():out,_=bridge.generate(b,max_iter=STEPS[d],temperature=1.,partial_masks=~mask & b['input_ids'].ne(bridge.bos_id),sampling_strategy='argmax',cfg_scale=0.)
            assert not out[mask].eq(model.tokenizer.mask_token_id).any()
            assert out[~mask].equal(b['input_ids'][~mask])
            altered=[{**x,'gold_h':'G'*len(x['gold_h']),'gold_l':'G'*len(x['gold_l'])} for x in chosen]
            f2,b2,_,m2,_=inputs(model,altered);assert all(f[k].equal(f2[k]) for k in f) and b['input_ids'].equal(b2['input_ids']) and mask.equal(m2)
            results[d]={'forward':'passed','unchanged_context':True,'gold_independent_inputs':True,'samples':len(chosen)}
        write(OUT/'cpu_checks.json',{'status':'passed','checks':results});print(results);return
    d=a.dataset;assert d;folder=OUT/d;folder.mkdir(exist_ok=False);write(folder/'status.json',{'status':'running'});report()
    try:
        rows,files=records(d);assert files==audit['datasets'][d]['files']
        write(folder/'identity.json',{'model':identity,'audit_sha256':sha(OUT/'protocol_audit.json'),'batch_size':64,'steps':STEPS[d],'precision':'bf16_autocast' if d=='kong' else 'fp32'})
        accum={};device=next(model.parameters()).device
        with (folder/'predictions.jsonl').open('x') as stream:
            for mode in MODES[d]:
                per_fold=[];all_aar=[]
                for fold in range(10 if d=='kong' else 1):
                    subset=[x for x in rows if x['mode']==mode and x['fold']==fold];values=[]
                    for i in range(0,len(subset),64):
                        chunk=subset[i:i+64];f,b,gold,mask,maps=inputs(model,chunk)
                        f={k:v.to(device) for k,v in f.items()};b={k:v.to(device) for k,v in b.items()};mask=mask.to(device);gold=gold.to(device);bridge=decoder(model,f,maps)
                        precision=torch.autocast('cuda',dtype=torch.bfloat16) if d=='kong' else nullcontext()
                        with torch.inference_mode(),precision:pred,_=bridge.generate(b,max_iter=STEPS[d],temperature=1.,partial_masks=~mask & b['input_ids'].ne(bridge.bos_id),sampling_strategy='argmax',cfg_scale=0.)
                        assert pred[~mask].equal(b['input_ids'][~mask])
                        correct=((pred==gold)&mask).sum(1);aar=(correct/mask.sum(1)).cpu().tolist();values+=aar
                        for j,row in enumerate(chunk):stream.write(json.dumps(dict(mode=mode,fold=fold,row=row['row'],key=row['key'],masked_positions=mask[j].nonzero().flatten().tolist(),reference_ids=gold[j,mask[j]].tolist(),prediction_ids=pred[j,mask[j]].tolist(),aar=aar[j]))+'\n')
                        stream.flush()
                    per_fold.append(float(np.mean(values)*100));all_aar+=values
                score=float(np.mean(per_fold)) if d=='kong' else float(np.round(np.mean(all_aar),4)*100)
                sd=float(np.std(per_fold)) if d=='kong' else float(np.round(np.std(all_aar),4)*100)
                accum[mode]={'airgen_aar_percent':score,'airgen_sd':sd,'mean_sequence_aar_percent':float(np.mean(all_aar)*100),'fold_means':per_fold,'n_sequences':len(all_aar)}
                print(d,mode,accum[mode],flush=True)
        write(folder/'metrics.json',{'dataset':d,'modes':accum,'protocol_audit_sha256':sha(OUT/'protocol_audit.json')});write(folder/'status.json',{'status':'success'})
    except BaseException as e:write(folder/'status.json',{'status':'failed','error':repr(e)});raise
    finally:report()

if __name__=='__main__':main()
