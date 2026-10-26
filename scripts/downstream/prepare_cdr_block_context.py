"""Activate pllm; python -B FILE --out ABS.

Freeze a new OAS confirmation cohort before inference. Exclude checkpoint
selection examples, previous benchmark heavy/H3 sequences, and exact training
heavy/H3 sequences or shared supplied H-cluster labels. No scores are read.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/project_c20250601/251105016/project/dllm_test')
PARENT=ROOT/'output/downstream_generation/cdr_marginal_exploration_20261015'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def keys(r):
    heavy=set();h3=set();clusters=set()
    for i,(role,seq) in enumerate(zip(r['chain_roles'],r['chains'])):
        if role!='antibody_heavy':continue
        heavy.add(seq.replace('J','L'))
        reg=r.get('regions',{}).get(str(i),{})
        if reg.get('CDR3'):h3.add(reg['CDR3'].replace('J','L'))
        meta=r.get('chain_metadata',{}).get(str(i),{})
        if meta.get('H_cluster_id'):clusters.add(meta['H_cluster_id'])
    identifiers=r.get('identifiers',{})
    if heavy and identifiers.get('heavy_h3_core'):h3.add(identifiers['heavy_h3_core'].replace('J','L'))
    return heavy,h3,clusters


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out
    out.mkdir(exist_ok=False,parents=True);shutil.copyfile(Path(__file__),out/'prepare_source.py')
    parent=json.loads((PARENT/'plan.json').read_text());provenance=[]
    for cp in parent['checkpoints'].values():provenance.append(json.loads((Path(cp)/'run_provenance.json').read_text()))
    data=Path(provenance[0]['data_dir']);manifest=data/'dataset_manifest.json'
    assert all(v['data_dir']==str(data) and v['parent_metadata']['dataset_manifest.json']==sha(manifest) for v in provenance)
    manifest_data=json.loads(manifest.read_text());hashes={str(manifest):sha(manifest)}
    banned=[set(),set(),set()]
    previous=[]
    for path in [PARENT/'rows.json',PARENT.parent/'cdr_inference_rounds_20261015/rows.json']:
        hashes[str(path)]=sha(path);previous+=json.loads(path.read_text())
    for r in previous:
        banned[0].add(r['gold_h'])
        if r['mode']=='cdrh3':banned[1].add(''.join(r['gold_h'][j] for j in r['mask_h']))
    # The training evaluator's indices are per-source validation indices.
    for source in provenance[0]['validation_indices']:
        selected=set().union(*(set(v['validation_indices'][source]) for v in provenance))
        files=sorted((data/'valid').glob(source+'-*.jsonl'));offset=0
        for path in files:
            hashes[str(path)]=sha(path)
            for i,line in enumerate(path.open()):
                if offset+i in selected:
                    for dest,values in zip(banned,keys(json.loads(line))):dest.update(values)
            offset+=i+1
        assert max(selected)<offset
    source=data/'valid/oas-00000.jsonl';records=[];excluded=Counter();order=['FR1','CDR1','FR2','CDR2','FR3','CDR3','FR4']
    for index,line in enumerate(source.open()):
        r=json.loads(line);assert r['chain_roles']==['antibody_heavy','antibody_light']
        h,l=[s.replace('J','L') for s in r['chains']];reg=r['regions']['0'];parts=[reg[k].replace('J','L') for k in order]
        if ''.join(parts)!=h:excluded['invalid_region_concatenation']+=1;continue
        if not set(h+l)<=set('ACDEFGHIKLMNPQRSTVWY'):excluded['noncanonical_residue']+=1;continue
        if len(h)>148 or len(l)>126:excluded['original_grid_overflow']+=1;continue
        hkey,h3key,ckey=keys(r)
        if any(a&b for a,b in zip((hkey,h3key,ckey),banned)):excluded['previous_experiments_or_checkpoint_selection']+=1;continue
        masks={};start=0
        for region,seq in zip(order,parts):
            if region.startswith('CDR'):masks['cdrh'+region[-1]]=list(range(start,start+len(seq)))
            start+=len(seq)
        labels=r['chain_metadata']['0']['h_region_labels'];assert len(labels)==len(h)
        for m,positions in masks.items():assert positions==[j for j,c in enumerate(labels) if c==m[-1]]
        assert all(masks.values()) and len(ckey)==1
        records.append(dict(heavy=h,light=l,h3=reg['CDR3'].replace('J','L'),cluster=next(iter(ckey)),
            masks=masks,source=str(source),row=index,identity=hashlib.sha256((h+'|'+l).encode()).hexdigest()))
    lookup=[defaultdict(set) for _ in range(3)]
    for i,r in enumerate(records):
        for table,key in zip(lookup,(r['heavy'],r['h3'],r['cluster'])):table[key].add(i)
    reject=set();train_counts=Counter();train_hashes={}
    # Hash all current bound train shards; parse only rows with heavy-chain role.
    # Identifiers/regions are checked for H3; missing annotations use a conservative substring scan.
    prefixes=defaultdict(list)
    for h3 in lookup[1]:prefixes[h3[:4]].append(h3)
    for shard in manifest_data['splits']['train']['shards']:
        path=data/shard['path'];digest=hashlib.sha256();count=0
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line);count+=1
                if b'"antibody_heavy"' not in line:continue
                r=json.loads(line);h,h3,c=keys(r);train_counts['antibody_rows']+=1
                if h and not h3:
                    train_counts['missing_h3_annotation']+=1
                    for seq in h:
                        for j in range(len(seq)-3):
                            h3.update(s for s in prefixes.get(seq[j:j+4],[]) if seq.startswith(s,j))
                for table,values in zip(lookup,(h,h3,c)):
                    for value in values:reject.update(table.get(value,()))
        assert count==shard['records'];train_counts['all_rows']+=count;train_hashes[str(path)]=digest.hexdigest()
        print('TRAIN AUDIT',path.name,count,'excluded candidate rows',len(reject),flush=True)
    excluded['training_exact_heavy_h3_or_cluster']=len(reject)
    keep=[r for i,r in enumerate(records) if i not in reject]
    parent_index=list(range(len(keep)))
    def find(i):
        while parent_index[i]!=i:parent_index[i]=parent_index[parent_index[i]];i=parent_index[i]
        return i
    seen={}
    for i,r in enumerate(keep):
        for tag in ('H:'+r['heavy'],'H3:'+r['h3'],'cluster:'+r['cluster']):
            if tag in seen:parent_index[find(i)]=find(seen[tag])
            else:seen[tag]=i
    components=defaultdict(list)
    for i,r in enumerate(keep):components[find(i)].append(r)
    grouped={hashlib.sha256('|'.join(sorted({r['identity'] for r in group})).encode()).hexdigest():group for group in components.values()}
    assert len(grouped)>=4096, 'Need enough fresh components for confirm and future reserve'
    confirm=set(sorted(grouped)[:2048]);rows=[];unique=set()
    for group,members in sorted(grouped.items()):
        for r in members:
            if r['identity'] in unique:excluded['duplicate_exact_pair']+=1;continue
            unique.add(r['identity'])
            for mode,mask in r['masks'].items():
                rows.append(dict(dataset='oas_new',mode=mode,fold=0,id=f"oas_new:{mode}:{r['identity']}",
                    key=r['identity'],group=group,split='confirm' if group in confirm else 'reserve',
                    heavy=r['heavy'],light=r['light'],gold_h=r['heavy'],gold_l=r['light'],mask_h=mask,mask_l=[],
                    source=r['source'],row=r['row'],h_cluster=r['cluster']))
    write(out/'rows.json',rows)
    info=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(out/'prepare_source.py'),
        parent=str(PARENT),checkpoints=parent['checkpoints'],source_data_sha256=hashes,train_shard_sha256=train_hashes,
        rows_sha256=sha(out/'rows.json'),exclusions=dict(excluded),train_scan=dict(train_counts),groups=len(grouped),
        counts=dict(Counter(r['split']+':'+r['mode'] for r in rows)),
        scope='OAS original validation supplement with existing IMGT region annotations, not the SAbDab official benchmark. '
              'Excludes recorded checkpoint-selection examples, previous heavy/H3, and current bound train exact heavy/H3 or supplied H-cluster labels. '
              'No claim of universal homology/pretraining decontamination; no scores read. Remaining components stay reserved.')
    write(out/'data_plan.json',info);print(json.dumps({k:info[k] for k in ('exclusions','train_scan','groups','counts')}),flush=True)


if __name__=='__main__':main()
