"""Activate pllm; run python -B -m scripts.research.audit_v5_full --help.

Read every V5 train/valid JSONL record and hash every shard. Writes reports only.
Use bounded CPU workers; no GPU, checkpoints, dataset mutations or filtering.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time

from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from dllm.pipelines.immune_llada.v3.v5_contract import geometry
try:
    import orjson
    loads=orjson.loads
except ImportError:
    loads=json.loads


def stat_identity(path):
    s=path.stat()
    return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino)


def scan(job):
    path=Path(job['path']);before=stat_identity(path)
    digest=hashlib.sha256();counts=Counter();reasons=Counter();hist=defaultdict(Counter);examples={};parity=[]
    with path.open('rb',buffering=4*1024*1024) as handle:
        for line_number,line in enumerate(handle,1):
            digest.update(line)
            if not line.strip():
                counts['blank_lines']+=1;continue
            counts['records']+=1
            try:
                row=loads(line)
                BioSeqRecord.from_prepared_dict(row)
                if row.get('schema_version','immune_llada.semantic.v1')!='immune_llada.semantic.v1':
                    raise ValueError('unsupported schema')
                if row.get('split') not in (None,job['split']):raise ValueError('split mismatch')
                result=geometry(row,job['capacity'],job['max_length'])
                if result.get('excluded'):
                    counts['excluded_vhh']+=1;continue
                counts['in_scope']+=1
                for key in ('synthetic_positions','observed_residues','missing_chains'):
                    counts[key]+=result.get(key,0)
                if 'object_length' in result:
                    hist[result['category']][result['object_length']]+=1
                    hist['total_input'][result['total_length']]+=1
                if result['reasons']:
                    counts['invalid_records']+=1
                    for reason in result['reasons']:
                        reasons[reason]+=1
                        examples.setdefault(reason,[])
                        if len(examples[reason])<5:
                            examples[reason].append(dict(line=line_number,object_length=result.get('object_length'),
                                total_length=result.get('total_length'),roles=row['chain_roles'],lengths=list(map(len,row['chains']))))
                else:
                    counts['valid_records']+=1
                    # Concrete renderer parity controls per shard; full pass uses
                    # equivalent arithmetic plus schema/provenance validation.
                    if len(parity)<3:parity.append(row)
            except Exception as error:
                counts['invalid_records']+=1;reasons['invalid_record']+=1
                if len(examples.setdefault('invalid_record',[]))<5:
                    examples['invalid_record'].append(dict(line=line_number,error=repr(error)))
    after=stat_identity(path)
    if after!=before:raise RuntimeError('Shard changed during audit: '+str(path))
    if counts['records']!=job['records']:raise RuntimeError('Manifest count mismatch: '+str(path))
    from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
    c=ImmuneCollator('V5',full_capacity=job['capacity'],max_length=job['max_length'])
    for row in parity:
        batch=c([row]);g=geometry(row,job['capacity'],job['max_length'])
        assert int(batch['attention_mask'].sum())==g['total_length']
        assert int(batch['eligible_mask'][batch['target_slot_mask']].sum())==g['observed_residues']+len(g['receptor_roles'])
        assert int(batch['residue_mask'][batch['target_slot_mask']].sum())==g['observed_residues']
    return dict(path=str(path),relative_path=job['relative_path'],split=job['split'],source=job['source'],
                sha256=digest.hexdigest(),stat=after,counts=dict(counts),reasons=dict(reasons),
                histograms={k:dict(v) for k,v in hist.items()},examples=examples,renderer_parity_records=len(parity))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--capacity',type=int,default=302);p.add_argument('--max-length',type=int,default=2048)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    root=Path(__file__).resolve().parents[2];data=args.data_dir.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'full_audit.json').exists():raise ValueError('Report already exists; use a new output directory')
    raw=(data/'dataset_manifest.json').read_bytes();manifest=loads(raw)
    assert manifest['schema_version']=='immune_llada.semantic.v1' and manifest['format']=='jsonl'
    jobs=[]
    for split in ('train','valid'):
        for shard in manifest['splits'][split]['shards']:
            path=(data/shard['path']).resolve();assert path.is_relative_to(data)
            jobs.append(dict(shard,path=str(path),relative_path=shard['path'],split=split,
                             capacity=args.capacity,max_length=args.max_length))
    start=time.time();reports=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(scan,j) for j in jobs]
        for future in as_completed(futures):
            result=future.result();reports.append(result)
            path=out/(result['relative_path'].replace('/','__')+'.audit.json')
            path.write_text(json.dumps(result,sort_keys=True)+'\n')
            print(json.dumps(dict(done=len(reports),total=len(jobs),shard=result['relative_path'],
                counts=result['counts'],reasons=result['reasons'],elapsed=round(time.time()-start,1))),flush=True)
    assert (data/'dataset_manifest.json').read_bytes()==raw
    counts=Counter();reasons=Counter();hist=defaultdict(Counter);splits={}
    for report in reports:
        counts.update(report['counts']);reasons.update(report['reasons'])
        for category,values in report['histograms'].items():hist[category].update(values)
    for split in ('train','valid'):
        rows=sum(x['counts']['records'] for x in reports if x['split']==split)
        assert rows==manifest['splits'][split]['records'];splits[split]=rows
    def summary(values):
        values=sorted((int(k),v) for k,v in values.items());n=sum(v for _,v in values)
        result=dict(records=n,min=values[0][0],max=values[-1][0])
        for q in (.5,.95,.99,.999):
            cumulative=0
            for length,count in values:
                cumulative+=count
                if cumulative>=n*q:result[str(q)]=length;break
        return result
    sources=[root/'scripts/research/audit_v5_full.py',root/'dllm/pipelines/immune_llada/v3/v5_contract.py',
             root/'dllm/pipelines/immune_llada/v3/v5_canvas.py',root/'dllm/pipelines/immune_llada/v3/layout.py',
             root/'dllm/pipelines/immune_llada/data/grammar.py',root/'dllm/pipelines/immune_llada/data/records.py']
    result=dict(schema='immune.v5.full_layout_audit.v1',status='passed' if not reasons and not counts['excluded_vhh'] else 'needs_resolution',
        data_dir=str(data),manifest_sha256=hashlib.sha256(raw).hexdigest(),capacity=args.capacity,max_length=args.max_length,
        splits=splits,counts=dict(counts),reasons=dict(reasons),lengths={k:summary(v) for k,v in hist.items()},
        histograms={k:dict(v) for k,v in hist.items()},shards=sorted(reports,key=lambda x:x['relative_path']),
        elapsed_seconds=time.time()-start,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        scope='All serialized records checked; renderer parity controls per shard; no benchmark decontamination rerun')
    (out/'full_audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('status','counts','reasons','lengths')},indent=2),flush=True)


if __name__=='__main__':main()
