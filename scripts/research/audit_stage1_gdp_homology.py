"""Search GDP chains against the actual Stage1 training sequences.

After activating pllm: python -B /project_c20250601/251105016/project/dllm_test/scripts/research/audit_stage1_gdp_homology.py --output ABS_GDP_DIRECTORY
MMseqs2 runs on CPU. Temporary databases stay in /tmp; retained results are
small alignments and a provenance receipt. This is a heuristic search audit,
not proof of absence of all homologs or of teacher training exposure.
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
from evaluate_stage1_frozen_gdp import ROOT, load_plan, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output
    assert out.is_absolute()
    plan = load_plan(out)
    dest = out/'homology'
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(dest)
    dest.mkdir(exist_ok=True)
    subset_path = ROOT/'output/stage1_ab500k_pilot_20261022/data/subset.json'
    assert sha(subset_path) == plan['arms']['mlm']['provenance']['subset_manifest_sha256']
    subset = json.loads(subset_path.read_text())
    df = pd.read_csv(out/'data_snapshot.csv')
    binary = '/project_c20250601/251105016/conda/envs/flow/bin/mmseqs'
    version = subprocess.check_output([binary, 'version'], text=True).strip()
    try:
        import orjson
        loads = orjson.loads
    except ImportError:
        loads = json.loads
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='stage1_gdp_homology_') as temp:
        work = Path(temp); seen = {'heavy': set(), 'light': set()}; counts = Counter()
        handles = {role: (work/(role+'.fasta')).open('w') for role in seen}
        digest = hashlib.sha256()
        with Path(subset['requests']).open('rb', buffering=8<<20) as source:
            for raw in source:
                digest.update(raw); row = loads(raw); counts[row['split']] += 1
                if row['split'] != 'train': continue
                for chain in row['request']['chains']:
                    role = chain['role'].removeprefix('antibody_'); sequence = chain['sequence']
                    if sequence not in seen[role]:
                        seen[role].add(sequence)
                        handles[role].write(f'>{role}_{len(seen[role])}\n{sequence}\n')
        for f in handles.values(): f.close()
        assert digest.hexdigest() == subset['requests_sha256']
        assert counts['train'] == 500000 and counts['valid'] == 4096
        unique_counts = {k: len(v) for k, v in seen.items()}; del seen
        receipt = dict(status='running', version=version, binary=binary, binary_sha256=sha(binary),
                       requests_sha256=digest.hexdigest(), unique_training_chains=unique_counts,
                       script_sha256=sha(__file__), commands=[], method='MMseqs2 easy-search; sensitivity 7.5; '
                       'query coverage >= 0.9; alignment identity >= 0.5; max-seqs 100; separate H/L searches')
        write(dest/'receipt.json', receipt)
        print(json.dumps({'phase':'training_fasta_ready','unique':unique_counts}), flush=True)
        summaries = {}
        columns = ['query','target','fident','qcov','tcov','alnlen','qlen','tlen','evalue','bits']
        for role, col in [('heavy','vh_protein_sequence'), ('light','vl_protein_sequence')]:
            query = dest/(role+'_queries.fasta')
            query.write_text(''.join(f'>{i}\n{s}\n' for i, s in enumerate(df[col])))
            target = work/(role+'.fasta'); result = dest/(role+'_alignments.tsv')
            command = [binary,'easy-search',str(query),str(target),str(result),str(work/(role+'_tmp')),
                '--threads','8','-s','7.5','--min-seq-id','0.5','-c','0.9','--cov-mode','2',
                '--alignment-mode','3','--max-seqs','100','--format-output',','.join(columns)]
            receipt['commands'].append(command)
            receipt[role+'_target_fasta_sha256'] = sha(target)
            with (dest/(role+'_mmseqs.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            matches = pd.read_csv(result, sep='\t', names=columns)
            assert (matches.qcov >= .899).all() and (matches.fident >= .499).all()
            best = matches.sort_values(['query','fident','qcov','bits'], ascending=[True,False,False,False]).drop_duplicates('query')
            best = pd.DataFrame({'query':range(len(df)), 'id':df.antibody_id}).merge(best, on='query', how='left')
            best.to_csv(dest/(role+'_best_observed.csv'), index=False)
            summaries[role] = dict(queries=len(df), queries_with_hits=int(best.fident.notna().sum()),
                observed_identity_ge_90=int((best.fident >= .9).sum()),
                observed_identity_ge_95=int((best.fident >= .95).sum()),
                median_observed_identity=float(best.fident.median()), alignments_sha256=sha(result))
            print(json.dumps({'phase':'search_complete','role':role,**summaries[role]}), flush=True)
        receipt.update(status='complete', summaries=summaries, seconds=time.monotonic()-start,
            limitation='Best identity among returned heuristic alignments, not exhaustive global nearest identity. '
            'Identity is measured over a local alignment covering at least 90% of the query. '
            'This does not establish pair-level structural independence or absence from teacher pretraining.')
        write(dest/'receipt.json', receipt)


if __name__ == '__main__':
    main()
