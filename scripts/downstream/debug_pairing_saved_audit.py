"""Activate pllm/protenix_abtcr; python -B SCRIPT for saved-candidate audits.

Check the exact 500x8 row mapping, reproduce diversity from saved sequences,
and bound early-stop effects without rerunning the generative model.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
GEN = ROOT/'output/downstream_generation'
sys.path.insert(0, str(ROOT/'downstream/comp_chain/eval_scripts'))


def main():
    import numpy as np
    import pandas as pd
    from generation_eval import compute_diversity
    benchmark = pd.read_csv(ROOT/'data/oas_previous_clean/splits/holdout_select_500_eval_OAS.csv')
    variants = [('v7', 'v7_f011_step50082_pairing_local_gpu_20261006/holdout500_n8.csv',
                 'v7_f011_step50082_pairing_scores_local_gpu_20261006/metrics.json'),
                ('fixed50000', 'fixed_chain_75m_step50000_pairing_reference_20261013/generated.csv',
                 'fixed_chain_75m_step50000_pairing_reference_20261013/metrics.json'),
                ('fixed112000', 'fixed_chain_75m_best112000_20261014/pairing-reference/generated.csv',
                 'fixed_chain_75m_best112000_20261014/pairing-reference/metrics.json')]
    report = {'status':'passed', 'runs':{}}
    frames = {}
    for name, csv, metric in variants:
        path = GEN/csv
        frame = pd.read_csv(path, keep_default_na=False)
        rowcol = '_input_row_idx' if name=='v7' else 'row_index'
        assert frame[rowcol].tolist() == np.repeat(np.arange(500),8).tolist()
        assert frame.variant_idx.tolist() == list(range(8))*500
        for column, refcol in [('h_sequence','h_sequence'),('raw_l_sequence','l_sequence')]:
            expected = benchmark[refcol].str.replace('-','',regex=False).repeat(8).tolist()
            assert frame[column].tolist() == expected, (name,column)
        diversities = [compute_diversity(group.gen_l_sequence.tolist())[0]
                       for _,group in frame.groupby('h_sequence',sort=False)]
        actual = float(np.mean(diversities))
        metrics = json.loads((GEN/metric).read_text())
        assert abs(actual-metrics['diversity_mean']) < 1e-10
        item = dict(count=len(frame), unique_heavies=frame.h_sequence.nunique(),
                    unique_generated=frame.gen_l_sequence.nunique(), diversity=actual,
                    im=metrics['gen_immunomatch_mean'],
                    generated_length_mean=float(frame.gen_l_sequence.str.len().mean()),
                    reference_length_mean=float(frame.raw_l_sequence.str.len().mean()),
                    csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if name!='v7':
            records=[json.loads(s) for s in (path.parent/'predictions.jsonl').read_text().splitlines()]
            item['terminated_before_step16']=sum(r['raw']['steps']<16 for r in records)
            item['early_eos_count']=sum(r['raw']['early_eos'] for r in records)
            assert item['terminated_before_step16']==0
        report['runs'][name]=item;frames[name]=frame
    partial=GEN/'pairing_sampler_debug_20261014/fixed50000_legacy/predictions.jsonl'
    rows=[json.loads(s) for s in partial.read_text().splitlines()]
    # Only completed heavy groups from the interrupted job; not a headline score.
    subset=pd.DataFrame(rows);counts=subset.groupby('row_index').size()
    keep=counts[counts==8].index.tolist();subset=subset[subset.row_index.isin(keep)]
    report['partial_legacy_diagnostic']={'heavies':len(keep),'representative_full_benchmark':False}
    for name,frame in [('fixed50000_legacy',subset), *frames.items()]:
        if name!='fixed50000_legacy':frame=frame.iloc[np.concatenate([np.arange(i*8,i*8+8) for i in keep])]
        div=float(np.mean([compute_diversity(g.gen_l_sequence.tolist())[0] for _,g in frame.groupby('h_sequence')]))
        report['partial_legacy_diagnostic'][name]={'diversity':div,'unique_generated':frame.gen_l_sequence.nunique()}
    out=GEN/'pairing_sampler_debug_20261014/saved_candidate_audit.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
