"""Reference-only generation metrics and isolated ANARCI numbering worker.

Run: python -m downstream.benchmark.ab_antigen.sequence_metrics --run-dir DIR
Worker: biophi/bin/python sequence_metrics.py --number-input in.json --number-output out.json
Reference contact labels and native sequences never enter generation.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import subprocess


def number_worker(input_path,output_path):
    from anarci import anarci
    seqs=json.loads(Path(input_path).read_text())
    output={}
    errors={}
    def number_batch(items):
        try:
            numbering,details,_=anarci(items,scheme='imgt',output=False,ncpu=4)
            return list(zip(items,numbering,details))
        except TypeError as error:
            # ANARCI check_for_j can encounter a missing alignment index.
            # Isolate that sequence; unrelated/environment failures still raise.
            if "NoneType" not in str(error) or "int" not in str(error):
                raise
            if len(items)>1:
                middle=len(items)//2
                return number_batch(items[:middle])+number_batch(items[middle:])
            errors[items[0][0]]=repr(error)
            return [(items[0],None,None)]
    for start in range(0,len(seqs),100):
        items=list(seqs.items())[start:start+100]
        for (key,seq),domains,info in number_batch(items):
            if not domains or len(domains)!=1:
                output[key]=None;continue
            numbers,begin,end=domains[0]
            entries=[[int(pos),ins,aa] for (pos,ins),aa in numbers if aa!='-']
            fv=''.join(e[2] for e in entries)
            actual_start=seq.find(fv)
            if actual_start<0 or seq.find(fv,actual_start+1)>=0:
                output[key]=None;continue
            output[key]=dict(numbering=entries,chain_type=info[0]['chain_type'],
                start=actual_start,end=actual_start+len(fv),fv=fv)
    Path(output_path).write_text(json.dumps(output)+'\n')
    Path(output_path).with_suffix('.errors.json').write_text(json.dumps(errors)+'\n')


def numbering(sequences,out_dir):
    from downstream.benchmark.ab_antigen.sequence_data import digest
    inputs={digest(s):s for s in sequences if s}
    source=out_dir/'numbering_input.json';dest=out_dir/'numbering.json'
    if source.exists() and json.loads(source.read_text())!=inputs:
        raise ValueError('Numbering cache input changed; separate output directory required')
    source.write_text(json.dumps(inputs)+'\n')
    if not dest.exists():
        env=os.environ.copy();prefix='/vepfs-mlp2/c20250601/251105016/conda/envs/biophi'
        env['PATH']=prefix+'/bin:'+env['PATH']
        subprocess.run([prefix+'/bin/python',str(Path(__file__).resolve()),'--number-input',str(source),'--number-output',str(dest)],env=env,check=True)
    return json.loads(dest.read_text())


def recovery(generated,reference,indices):
    lookup={(int(p),ins):aa for p,ins,aa in generated}
    selected=[reference[i] for i in indices]
    return sum(lookup.get((int(p),ins))==aa for p,ins,aa in selected),len(selected)


def main(run_dir, reference_path=None):
    import numpy as np
    from Bio.Align import PairwiseAligner
    from downstream.benchmark.ab_antigen.sequence_data import DATA,digest,write_json
    from downstream.benchmark.ab_antigen.sequence_eval import read_rows
    rows=read_rows(run_dir/'predictions.jsonl');refs=read_rows(reference_path or DATA/'chimera.jsonl')
    by_target=defaultdict(list)
    for r in refs:
        if r['split']=='test':by_target[r['antigen_key']].append(r)
    numbered=numbering([r[c] for r in rows for c in ['heavy','light'] if r['token_valid']],run_dir)
    result=[]
    for r in rows:
        nums={c:numbered.get(digest(r[c])) for c in ['heavy','light']}
        valid=bool(r['token_valid'] and nums['heavy'] and nums['light'] and
            nums['heavy']['chain_type']=='H' and nums['light']['chain_type'] in ['K','L'] and
            all(nums[c]['start']==0 and nums[c]['end']==len(r[c]) for c in nums))
        row={**r,'paired_valid':valid,'cdr_aar':0.,'h3_aar':0.,'caar':0. if any(ref['contact_indices']['heavy'] or ref['contact_indices']['light'] for ref in by_target[r['antigen_key']]) else None}
        row['motif_counts']={name:sum(len(re.findall(pattern,r[c])) for c in ['heavy','light'])
            for name,pattern in [('n_glycosylation','N[^P][ST]'),('deamidation_NG_NS','N[GS]'),('isomerization_DG','DG'),('cleavage_DP','DP')]}
        if valid:
            row['h3']=''.join(aa for p,_,aa in nums['heavy']['numbering'] if 105<=p<=117)
            row['reference_insertions_min']=min(sum(len({(p,i) for p,i,_ in nums[c]['numbering']}-{(p,i) for p,i,_ in ref['numbering'][c]}) for c in ['heavy','light']) for ref in by_target[r['antigen_key']])
            scores=[]
            for ref in by_target[r['antigen_key']]:
                totals={key:[0,0] for key in ['cdr_aar','h3_aar','caar']}
                for c in ['heavy','light']:
                    reference=ref['numbering'][c]
                    if ''.join(e[2] for e in reference)!=ref[c]:raise ValueError('Reference numbering mismatch')
                    cdr=[i for i,(p,_,_) in enumerate(reference) if 27<=p<=38 or 56<=p<=65 or 105<=p<=117]
                    h3=[i for i,(p,_,_) in enumerate(reference) if c=='heavy' and 105<=p<=117]
                    for key,indices in [('cdr_aar',cdr),('h3_aar',h3),('caar',ref['contact_indices'][c])]:
                        correct,n=recovery(nums[c]['numbering'],reference,indices)
                        totals[key][0]+=correct;totals[key][1]+=n
                scores.append({key:(a/b if b else None) for key,(a,b) in totals.items()})
            for key in totals:
                vals=[s[key] for s in scores if s[key] is not None]
                row[key]=max(vals) if vals else None
            row['cdr_sequence']='|'.join(''.join(aa for p,_,aa in nums[c]['numbering'] if 27<=p<=38 or 56<=p<=65 or 105<=p<=117) for c in ['heavy','light'])
        result.append(row)
    grouped=defaultdict(list)
    for r in result:grouped[(r['antigen_key'],r['condition'])].append(r)
    target_metrics=[]
    aligner=PairwiseAligner(mode='global',match_score=0,mismatch_score=-1,open_gap_score=-1,extend_gap_score=-1)
    for (target,condition),items in grouped.items():
        valid=[r for r in items if r['paired_valid']]
        diversity=[-aligner.score(a['cdr_sequence'],b['cdr_sequence'])/max(len(a['cdr_sequence']),len(b['cdr_sequence'])) for i,a in enumerate(valid) for b in valid[i+1:]]
        target_metrics.append(dict(antigen_key=target,condition=condition,attempts=len(items),
            paired_valid_rate=len(valid)/len(items),unique_pair_rate=len({(r['heavy'],r['light']) for r in valid})/len(items),
            cdr_diversity=float(np.mean(diversity)) if diversity else None,
            **{key:float(np.mean([r[key] for r in items if r[key] is not None])) if any(r[key] is not None for r in items) else None for key in ['cdr_aar','h3_aar','caar']}))
    summary={}
    for condition in ['correct','replaced']:
        summary[condition]={}
        for metric in ['paired_valid_rate','unique_pair_rate','cdr_diversity','cdr_aar','h3_aar','caar']:
            values=np.array([r[metric] for r in target_metrics if r['condition']==condition and r[metric] is not None])
            rng=np.random.default_rng(42)
            summary[condition][metric]=dict(mean=float(values.mean()) if len(values) else None,
                ci95=np.quantile([rng.choice(values,len(values),replace=True).mean() for _ in range(1000)],[.025,.975]).tolist() if len(values)>1 else None)
    try:
        novelty=generated_novelty(result,numbered,run_dir)
    except (subprocess.CalledProcessError,OSError) as exc:
        novelty={'status':'blocked_search_failure','error':str(exc),'log_directory':str(run_dir)}
    write_json(run_dir/'sequence_metrics_rows.json',result)
    write_json(run_dir/'sequence_metrics.json',dict(protocol='chimera_sequence_only_v1',per_target=target_metrics,macro=summary,
        novelty_status='see_novelty_json_heuristic_nearest_hits',novelty=novelty,scope='sequence_quality_not_binding_success',
        recovery='best_reference_per_candidate_then_mean_all_attempts',caar='reference_CA_CA_8A_contacts',
        motif_interpretation='sequence_developability_motifs_not_experimental_failures'))




def generated_novelty(rows,numbered,run_dir):
    """Exact novelty plus nearest retrieved >=80%-coverage MMseqs hit (heuristic)."""
    from downstream.benchmark.ab_antigen.sequence_data import DATA,digest,write_json
    from downstream.benchmark.ab_antigen.sequence_audit import MMSEQS
    bank=DATA/'audit/training';report={}
    for chain in ['heavy','light','h3']:
        queries={digest(r[chain]):r[chain] for r in rows if r['paired_valid']}
        database=bank/f'train_{chain}.fasta'
        if not database.exists():report[chain]={'status':'training_bank_missing'};continue
        exact=set()
        with database.open() as f:
            for line in f:
                if line.startswith('>') and line[1:].strip() in queries:exact.add(line[1:].strip())
        q=run_dir/f'novelty_{chain}.fasta';q.write_text(''.join(f'>{k}\n{s}\n' for k,s in queries.items()))
        hits=run_dir/f'novelty_{chain}.tsv'
        if queries:
            with (run_dir/f'novelty_{chain}.log').open('w') as log:
                subprocess.run([MMSEQS,'easy-search',str(database),str(q),str(hits),str(run_dir/f'novelty_tmp_{chain}'),
                    '--min-seq-id','0.0','-c','0.8','--cov-mode','0','--max-seqs','100','--threads','2','--split-memory-limit','4G','--split','16',
                    '--format-output','target,query,fident,tcov,qcov'],stdout=log,stderr=subprocess.STDOUT,check=True)
        nearest={}
        if hits.exists():
            for line in hits.read_text().splitlines():
                key,target,identity,qcov,tcov=line.split('\t')
                nearest[key]=max(nearest.get(key,0.),float(identity))
        report[chain]=dict(unique_valid_sequences=len(queries),exact_training_matches=sorted(exact),nearest_retrieved_identity=nearest,
            missing_hit='not_proof_of_no_homology',coverage_threshold=.8)
    pair_bank=bank/'train_pair_hashes.txt'
    query_pairs={digest([r['heavy'],r['light']]) for r in rows if r['paired_valid']}
    exact_pairs=set()
    if pair_bank.exists():
        with pair_bank.open() as f:
            for line in f:
                if line.strip() in query_pairs:exact_pairs.add(line.strip())
    report['paired_exact']=dict(unique_generated_pairs=len(query_pairs),matched=sorted(exact_pairs),bank_available=pair_bank.exists())
    write_json(run_dir/'novelty.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path);p.add_argument('--number-input');p.add_argument('--number-output');args=p.parse_args()
    if args.number_input:number_worker(args.number_input,args.number_output)
    elif args.run_dir:main(args.run_dir)
    else:p.error('Supply --run-dir or worker inputs')
