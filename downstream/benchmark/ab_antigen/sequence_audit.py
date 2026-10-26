"""Freeze homology-grouped probe splits and audit the actual pretraining corpus.

Run: python -m downstream.benchmark.ab_antigen.sequence_audit [--training]
MMseqs clustering is a reproducible operational homology definition, not proof of
novel antigen families. Unknown original-source membership remains unknown.
"""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from downstream.benchmark.ab_antigen.sequence_data import ROOT, DATA, digest, write_json
from downstream.benchmark.ab_antigen.sequence_eval import read_rows,sha
MMSEQS='/project_c20250601/251105016/conda/envs/flow/bin/mmseqs'


def clusters(sequences, name, threshold):
    folder=DATA/'audit'/name;folder.mkdir(parents=True,exist_ok=True)
    seqs=sorted(set(sequences));fasta=folder/'sequences.fasta'
    text=''.join(f'>{digest(s)}\n{s}\n' for s in seqs)
    if fasta.exists() and fasta.read_text()!=text:raise ValueError('Cluster input changed; new protocol version required')
    fasta.write_text(text)
    output=folder/'clusters_cluster.tsv'
    if not output.exists():
        with (folder/'mmseqs.log').open('w') as log:
            subprocess.run([MMSEQS,'easy-cluster',str(fasta),str(folder/'clusters'),str(folder/'tmp'),
                '--min-seq-id',str(threshold),'-c','0.8','--cov-mode','0','--cluster-mode','1',
                '--threads','8'],stdout=log,stderr=subprocess.STDOUT,check=True)
    result={}
    for line in output.read_text().splitlines():
        rep,member=line.split('\t');result[member]=rep
    if set(result)!=set(digest(s) for s in seqs):raise ValueError('Incomplete cluster mapping')
    return result


def split_groups(keys):
    keys=sorted(set(keys));rng=np.random.default_rng(42);rng.shuffle(keys)
    if len(keys)<3:return {k:'train' for k in keys}
    nval=max(1,round(len(keys)*.1));ntest=nval
    return {k:('test' if i<ntest else 'val' if i<ntest+nval else 'train') for i,k in enumerate(keys)}


def make_splits():
    rows=read_rows(DATA/'binding.jsonl')+read_rows(DATA/'spr.jsonl')
    ag=clusters([r['antigen'] for r in rows],'cov_antigen',.7)
    h=clusters([r['heavy'] for r in rows],'cov_heavy',.9)
    l=clusters([r['light'] for r in rows],'cov_light',.9)
    audit={}
    for task in ['binding','spr']:
        selected=[r for r in rows if r['dataset']==task];definitions={};audit[task]={}
        for axis in ['antigen','antibody']:
            group={r['id']:(ag[digest(r['antigen'])] if axis=='antigen' else h[digest(r['heavy'])]+'_'+l[digest(r['light'])]) for r in selected}
            assignment=split_groups(group.values())
            definitions[axis]={key:assignment[value] for key,value in group.items()}
            audit[task][axis]=dict(groups=len(assignment),rows=dict(Counter(definitions[axis].values())),
                status='ready' if len(set(definitions[axis].values()))==3 else 'insufficient_homology_groups',
                groups_sha256=digest(group))
        write_json(DATA/f'{task}_splits.json',definitions)
    write_json(DATA/'split_audit.json',audit)
    print(json.dumps(audit),flush=True)


def training_audit(resume=False):
    folder=DATA/'audit/training';folder.mkdir(parents=True,exist_ok=True)
    # Include all accepted sequences, including staging AbBiBench rows.
    queries={role:{} for role in ['antigen','heavy','light']};pairs=set()
    for name in ['binding','spr','abbibench','chimera']:
        p=DATA/f'{name}.jsonl'
        if not p.exists():raise FileNotFoundError(p)
        for row in read_rows(p):
            for role in queries:queries[role][row[role]]=digest(row[role])
            pairs.add(digest([row['heavy'],row['light']]))
    if not resume:
        exact={role:Counter() for role in queries};pair_counts=Counter();sources=Counter();files=[]
        training=ROOT/'data/prepared/immune_v6_binding_only/train'
        handles={r:(folder/f'train_{r}.fasta').open('w') for r in queries}
        seen={r:set() for r in queries}
        for path in sorted(training.glob('*.jsonl')):
            # Manifest sources with exclusively TCR roles cannot contain antibody chains.
            if path.name.startswith(('ots-','tcr_','trait-')):continue
            file_hash=hashlib.sha256();count=0
            with path.open('rb') as f:
                for raw in f:
                    file_hash.update(raw);row=json.loads(raw);count+=1
                    roles=dict(zip(row['chain_roles'],row['chains']))
                    if not any(r.startswith('antibody_') for r in roles):continue
                    sources[row['source']]+=1
                    extracted={'antigen':roles.get('antigen'),'heavy':roles.get('antibody_heavy'),'light':roles.get('antibody_light')}
                    for role,seq in extracted.items():
                        if not seq:continue
                        if seq in queries[role]:exact[role][queries[role][seq]]+=1
                        key=digest(seq)
                        if key not in seen[role]:
                            handles[role].write(f'>{key}\n{seq}\n');seen[role].add(key)
                    if extracted['heavy'] and extracted['light']:
                        key=digest([extracted['heavy'],extracted['light']])
                        if key in pairs:pair_counts[key]+=1
            files.append(dict(file=str(path),sha256=file_hash.hexdigest(),rows=count))
            print('audited',path.name,count,flush=True)
        for h in handles.values():h.close()
        write_json(folder/'exact.json',dict(exact_matches=exact,pair_matches=pair_counts,files=files,
            training_sources=sources,upstream_unprepared_sources='not_audited',h3='pending_numbering_not_claimed_clean',
            excluded_role_only_sources=['ots','tcr_*','trait']))
    for role in queries:
        fasta=folder/f'query_{role}.fasta';fasta.write_text(''.join(f'>{key}\n{seq}\n' for seq,key in queries[role].items()))
        with (folder/f'{role}_search.log').open('w') as log:
            subprocess.run([MMSEQS,'easy-search',str(folder/f'train_{role}.fasta'),str(fasta),
                str(folder/f'{role}_homology.tsv'),str(folder/f'tmp_reversed_{role}'),
                '--min-seq-id',str(.7 if role=='antigen' else .9),'-c','0.8','--cov-mode','0',
                '--max-seqs','100','--threads','2','--split-memory-limit','4G','--split','16','--format-output','target,query,fident,tcov,qcov'],
                stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(folder/'status.json',dict(status='exact_and_mmseqs_search_complete',
        h3='pending_numbering',interpretation='No clean-family claim; reversed MMseqs search is heuristic, max100 benchmark hits per training sequence',
        thresholds={'antigen_identity':.7,'antibody_identity':.9,'bidirectional_coverage':.8}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    make_splits()
    if args.training:training_audit(args.resume)
