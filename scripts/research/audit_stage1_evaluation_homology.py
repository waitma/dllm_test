"""Audit near-homology for native-partner and experimental-structure tests.

Activate pllm and run this absolute file with --output ABS_NEW_DIRECTORY.
Reads fixed evaluation manifests and the actual shared 500k training requests.
Runs existing MMseqs2 on CPU; temporary training databases are deleted on exit.
Best observed local identity is not an exhaustive global-neighbor guarantee.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import pandas as pd
from evaluate_stage1_frozen_gdp import ROOT,sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;assert out.is_absolute()
    if out.exists():raise FileExistsError(out)
    base=ROOT/'output/stage1_downstream_9770_20261025'
    sources={'native_partner':base/'native_partner/records.json',
             'experimental_structure':base/'experimental_structure/structures.json'}
    queries=[]
    for scope,path in sources.items():
        for r in json.loads(path.read_text()):
            queries.append(dict(qid='q'+str(len(queries)),scope=scope,row=r['row'],h=r['h'],l=r['l']))
    plan=json.loads((base/'native_partner/plan.json').read_text())
    subset_path=ROOT/'output/stage1_ab500k_pilot_20261022/data/subset.json'
    assert sha(subset_path)==plan['arms']['mlm']['provenance']['subset_manifest_sha256']
    subset=json.loads(subset_path.read_text())
    binary='/project_c20250601/251105016/conda/envs/flow/bin/mmseqs'
    out.mkdir(parents=True)
    receipt=dict(status='running',script_sha256=sha(__file__),sources={k:dict(path=str(v),sha256=sha(v)) for k,v in sources.items()},
        version=subprocess.check_output([binary,'version'],text=True).strip(),binary_sha256=sha(binary),
        method='Separate H/L MMseqs2 searches, sensitivity7.5, query coverage>=90%, min identity50%, max-seqs100',commands=[])
    write(out/'receipt.json',receipt);start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='stage1_eval_homology_') as tmp:
        temp=Path(tmp);seen={'h':set(),'l':set()};counts=Counter();digest=hashlib.sha256()
        files={role:(temp/(role+'.fasta')).open('w') for role in seen}
        with Path(subset['requests']).open('rb',buffering=8<<20) as f:
            for raw in f:
                digest.update(raw);r=json.loads(raw);counts[r['split']]+=1
                if r['split']!='train':continue
                c={x['role']:x['sequence'] for x in r['request']['chains']}
                for role,full in [('h','antibody_heavy'),('l','antibody_light')]:
                    seq=c[full]
                    if seq not in seen[role]:
                        seen[role].add(seq);files[role].write(f'>{role}_{len(seen[role])}\n{seq}\n')
        for f in files.values():f.close()
        assert counts['train']==500000 and counts['valid']==4096 and digest.hexdigest()==subset['requests_sha256']
        receipt.update(request_sha256=digest.hexdigest(),unique_training_chains={k:len(v) for k,v in seen.items()});del seen
        print(json.dumps(dict(phase='training_ready',queries=len(queries),unique=receipt['unique_training_chains'])),flush=True)
        summaries=[]
        columns=['query','target','fident','qcov','tcov','alnlen','qlen','tlen','evalue','bits']
        for role in ['h','l']:
            query=out/(role+'_queries.fasta');query.write_text(''.join(f">{r['qid']}\n{r[role]}\n" for r in queries))
            target=temp/(role+'.fasta');result=out/(role+'_alignments.tsv')
            command=[binary,'easy-search',str(query),str(target),str(result),str(temp/(role+'_work')),
                '--threads','8','-s','7.5','--min-seq-id','0.5','-c','0.9','--cov-mode','2',
                '--alignment-mode','3','--max-seqs','100','--format-output',','.join(columns)]
            receipt['commands'].append(command);receipt[role+'_target_sha256']=sha(target)
            with (out/(role+'_mmseqs.log')).open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            hits=pd.read_csv(result,sep='\t',names=columns)
            assert (hits.qcov>=.899).all() and (hits.fident>=.499).all()
            best=hits.sort_values(['query','fident','qcov','bits'],ascending=[True,False,False,False]).drop_duplicates('query')
            mapping=pd.DataFrame([{k:r[k] for k in ['qid','scope','row']} for r in queries]).rename(columns={'qid':'query'})
            best=mapping.merge(best,on='query',how='left',validate='one_to_one');best.to_csv(out/(role+'_best_observed.csv'),index=False)
            receipt[role+'_alignments_sha256']=sha(result)
            for scope,frame in best.groupby('scope'):
                summary=dict(scope=scope,role=role,queries=len(frame),queries_with_hits=int(frame.fident.notna().sum()),
                    observed_identity_ge_90=int((frame.fident>=.9).sum()),observed_identity_ge_95=int((frame.fident>=.95).sum()),
                    median_observed_identity=float(frame.fident.median()))
                summaries.append(summary);print(json.dumps(dict(phase='search_complete',**summary)),flush=True)
        for source in receipt['sources'].values():assert sha(source['path'])==source['sha256']
        receipt.update(status='complete',summaries=summaries,seconds=time.monotonic()-start,
            limitation='Best identity among heuristic local alignments covering >=90% query. No exhaustive nearest-neighbor guarantee; '
            'does not establish complete pair-level independence or absence from teacher pretraining.')
        write(out/'receipt.json',receipt)


if __name__=='__main__':main()
