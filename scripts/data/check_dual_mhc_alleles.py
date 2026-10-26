"""Activate pllm; python -B scripts/data/check_dual_mhc_alleles.py.

Recheck allele names in audited context proposals, normalizing optional HLA-* notation.
Preserves the previous full collator audit; writes a separate authoritative allele report.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.data.rebuild_v7_from_v5 import sha, write
OUT = ROOT / 'output/dual_mhc_exploration_20261015'


def main():
    fasta = ROOT / 'data/tcr_pmhc_fulllength/imgt_hla/hla_prot.fasta'
    reference = defaultdict(set)
    name, parts = None, []
    def add():
        fields = name.split(':')
        for n in range(1,len(fields)+1):
            reference[':'.join(fields[:n]).replace('*','')].add(''.join(parts))
    for line in fasta.read_text().splitlines():
        if line.startswith('>'):
            if name:add()
            name,parts=line.split()[1],[]
        else:parts.append(line.strip())
    if name:add()
    counts,alleles=Counter(),{}
    source=OUT/'local_context_candidates.jsonl'
    dest=OUT/'local_context_candidates_verified.jsonl'
    with source.open() as f,dest.open('w') as out:
        for line in f:
            item=json.loads(line)
            sequence=item['record']['chains'][0]
            states=[]
            for name in item['allele_names']:
                possible=reference.get(name.replace('HLA-','').replace('*',''),set())
                status='exact_unique_sequence' if possible=={sequence} else 'ambiguous_allele_sequences' if len(possible)>1 else 'unresolved_or_mismatch'
                alleles[name]=dict(status=status,unique_reference_sequences=len(possible),stored_sequence_in_reference=sequence in possible)
                states.append(status)
            approved=bool(states) and all(s=='exact_unique_sequence' for s in states)
            item['allele_status']='exact_unique_sequence' if approved else '+'.join(sorted(set(states)))
            item['ready_for_context_replacement']=approved
            counts[item['split']+'/'+item['allele_status']]+=1
            out.write(json.dumps(item)+'\n')
    report=dict(status='complete',counts=dict(counts),alleles=alleles,
                normalization='remove optional HLA- prefix and * only; retain all allele fields',
                supersedes='allele classifications in local_audit.json; full 39048-row layout checks remain valid',
                files={str(p):sha(p) for p in (fasta,source,dest,OUT/'local_audit.json',Path(__file__))})
    write(OUT/'local_allele_audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
