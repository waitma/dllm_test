"""Activate pllm; python -B -m scripts.research.audit_fixed_chain --output ABS --workers 4.

Read-only full V5 per-chain capacity audit, with shard hashes/counts and renderer
parity samples. No filtering, cropping, training or checkpoint creation.
"""
import argparse,hashlib,json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
try:
    import orjson
    loads=orjson.loads
except ImportError:loads=json.loads
ROOT=Path('/project_c20250601/251105016/project/dllm_test')
DATA=ROOT/'data/prepared/immune_v5_receptor_completion'
CAP={'antibody_heavy':187,'antibody_light':154,'tcr_beta':187,'tcr_alpha':154}

def scan(job):
    from dllm.pipelines.immune_llada.v3.v5_contract import geometry
    path=DATA/job['path'];before=path.stat();digest=hashlib.sha256();count=0;reasons=Counter();maxima={};parity=[];max_total=0
    with path.open('rb',buffering=4*1024*1024) as f:
        for line in f:
            digest.update(line)
            if not line.strip():continue
            row=loads(line);count+=1;g=geometry(row,capacity=343)
            if g.get('excluded'):reasons['excluded_vhh']+=1;continue
            reasons.update(g['reasons'])
            roles=[role for role in row['chain_roles'] if role in CAP]
            for role,seq in zip(row['chain_roles'],row['chains']):
                if role in CAP:
                    maxima[role]=max(maxima.get(role,0),len(seq))
                    if len(seq)+1>CAP[role]:reasons['chain_capacity_overflow:'+role]+=1
            if 'total_length' in g:
                total=g['total_length']-343+sum(CAP[x] for x in roles)+len(roles)
                max_total=max(max_total,total)
                if total>2048:reasons['sequence_budget_overflow']+=1
                if len(parity)<2:parity.append(dict(record=row,total=total))
    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert count==job['records'],path
    return dict(path=str(path),sha256=digest.hexdigest(),records=count,reasons=dict(reasons),max_residues=maxima,max_total=max_total,parity=parity)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    manifest=json.loads((DATA/'dataset_manifest.json').read_text());jobs=[j for split in manifest['splits'].values() for j in split['shards']];results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(scan,j) for j in jobs]
        for f in as_completed(futures):
            results.append(f.result());print('shards',len(results),'/',len(jobs),flush=True)
    from dllm.pipelines.immune_llada.fixed_chain import FixedChainCollator
    c=FixedChainCollator();parity_count=0
    for result in results:
        for sample in result.pop('parity'):
            b=c([sample['record']]);assert int(b['attention_mask'].sum())==sample['total'];parity_count+=1
    problems=Counter();maxima={}
    for x in results:
        problems.update(x['reasons'])
        for role,n in x['max_residues'].items():maxima[role]=max(maxima.get(role,0),n)
    report=dict(chain_capacities=CAP, layout_contract='immune.fixed_chain.f011.v1', status='passed' if not problems else 'failed',data_dir=str(DATA),records=sum(x['records'] for x in results),shards=len(results),reasons=dict(problems),max_residues=maxima,max_total=max(x['max_total'] for x in results),renderer_parity=parity_count,shard_results=results)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='shard_results'})
    if problems:raise SystemExit(1)
if __name__=='__main__':main()
