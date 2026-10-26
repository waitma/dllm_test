"""Independently rebuild AB test pairs and verify exact-overlap audit witnesses.

Activate pllm then protenix_abtcr, and run from repository root:
python -B scripts/downstream/verify_ab_pretraining_exposure.py --audit ABS_AUDIT_DIRECTORY
Writes verification and row masks; preserves all original evaluation samples.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def norm(s):
    return str(s or '').strip().upper().replace('-', '').replace('J','L')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    a = p.parse_args()
    audit = json.loads((a.audit/'exposure.json').read_text())
    plan = json.loads((a.audit/'plan.json').read_text())
    from downstream.grammar.ab_probes import prepare_task
    from downstream.grammar.ab_fv_input import adapt_pairs
    from downstream.ophiuchus_eval.developability import ALL_COLS
    cohorts, raw_indices = {}, {}
    for task in ('specificity','gdp_a1'):
        df,pairs,paths,_ = prepare_task(task)
        selected = list(range(len(pairs))) if task=='specificity' else [i for i,v in enumerate(df[list(ALL_COLS)].notna().any(axis=1)) if v]
        raw = [tuple(map(norm,pairs[i])) for i in selected]
        adapted,_ = adapt_pairs(task,raw)
        name='ab_repr/'+task
        cohorts[name]=list(zip(raw,adapted))
        raw_indices[name]=selected
        for path in paths:
            assert sha(path)==plan['evaluation_inputs_checked'][str(path)],path
    base=ROOT/'data/downstream/cdr_infilling'
    for mode in ('cdrh1','cdrh2','cdrh3'):
        name='ab_cdr/kong/'+mode
        pairs=[]
        for fold in range(10):
            path=base/'sabdab_kong'/mode/f'fold_{fold}/test.json'
            assert sha(path)==plan['evaluation_inputs_checked'][str(path)]
            for line in path.read_text().splitlines():
                row=json.loads(line)
                pair=(norm(row['heavy_chain_seq']),norm(row['light_chain_seq']))
                pairs.append((pair,pair))
        cohorts[name]=pairs
        raw_indices[name]=list(range(len(pairs)))
    raw=base/'sab23h2/SAb-23-H2-Ab'
    pairs=[]
    for name in (raw/'prot_ids.txt').read_text().splitlines():
        path=raw/f'fasta.files.native/{name}.fasta'
        assert sha(path)==plan['evaluation_inputs_checked'][str(path)]
        lines=path.read_text().splitlines()
        pair=(norm(lines[1]),norm(lines[3]))
        pairs.append((pair,pair))
    cohorts['ab_cdr/sab23h2']=pairs
    raw_indices['ab_cdr/sab23h2']=list(range(len(pairs)))
    bank_path=ROOT/'output/v7_from_v5_scaffold_20261015/test_scope_v2/banks.json'
    banks=json.loads(bank_path.read_text())
    for name, pairs in cohorts.items():
        assert len(pairs)==plan['cohorts'][name]['rows']
        for role, values in [('heavy',{h for pair in pairs for h,l in pair if h}),
                             ('light',{l for pair in pairs for h,l in pair if l}),
                             ('pair',{h+'|'+l for pair in pairs for h,l in pair if h and l})]:
            assert values==set(banks[name][role]),(name,role)
    hit_keys={(x['split'],x['cohort']):{r:set(v) for r,v in x['keys'].items()} for x in audit['matched_keys']}
    summaries=[]
    row_masks=[]
    for split in ('train','valid','lora_pilot_train'):
        for name,pairs in cohorts.items():
            keys=hit_keys.get((split,name),{})
            masks={r:[] for r in ('heavy','light','pair','either_chain')}
            for raw_idx,variants in zip(raw_indices[name],pairs):
                flags={r:False for r in ('heavy','light','pair')}
                for h,l in set(variants):
                    flags['heavy'] |= h in keys.get('heavy',set())
                    flags['light'] |= l in keys.get('light',set())
                    flags['pair'] |= h+'|'+l in keys.get('pair',set())
                flags['either_chain']=flags['heavy'] or flags['light']
                for r,v in flags.items():
                    if v:masks[r].append(raw_idx)
            summaries.append(dict(split=split,cohort=name,n_test_rows=len(pairs),
                                  matched_test_rows={r:len(v) for r,v in masks.items()},
                                  unique_matching_keys={r:len(v) for r,v in keys.items()}))
            row_masks.append(dict(split=split,cohort=name,row_indices=masks))
    # Re-read exact full-sequence witnesses using a separate role->sequence implementation.
    wanted=defaultdict(lambda:defaultdict(list))
    for group in audit['witnesses']:
        if group['role']=='h3':continue
        for row in group['rows']:
            wanted[row['path']][row['line']].append((group,row))
    checked=0
    for path,lines in wanted.items():
        with open(path,'rb') as f:
            for i,raw in enumerate(f,1):
                if i>max(lines):break
                if i not in lines:continue
                row=json.loads(raw)
                h=[norm(s) for r,s in zip(row['chain_roles'],row['chains']) if r=='antibody_heavy']
                l=[norm(s) for r,s in zip(row['chain_roles'],row['chains']) if r=='antibody_light']
                actual={'heavy':set(h),'light':set(l),'pair':{hh+'|'+ll for hh in h for ll in l}}
                for group,w in lines[i]:
                    assert hashlib.sha256(raw).hexdigest()==w['raw_sha256']
                    assert set(w['values']) <= actual[group['role']] & set(banks[group['cohort']][group['role']])
                    checked+=1
    # The parent evaluates only a fixed 256 records/source, not the entire valid split.
    prov=json.loads((Path(plan['lineage']['parent'])/'run_provenance.json').read_text())
    counts=defaultdict(Counter)
    for name in cohorts:
        pairbank=set(banks[name]['pair'])
        for entry in audit['files']:
            if entry['split']!='valid':continue
            selected=set(prov['validation_indices'][entry['source']])
            # Every source has one validation shard in this frozen manifest.
            assert sum(x['split']=='valid' and x['source']==entry['source'] for x in audit['files'])==1
            with open(entry['path']) as f:
                for i,line in enumerate(f):
                    if i not in selected:continue
                    row=json.loads(line)
                    h=[norm(s) for r,s in zip(row['chain_roles'],row['chains']) if r=='antibody_heavy']
                    l=[norm(s) for r,s in zip(row['chain_roles'],row['chains']) if r=='antibody_light']
                    if {hh+'|'+ll for hh in h for ll in l} & pairbank:counts[name][entry['source']]+=1
    result=dict(status='passed',test_banks_rebuilt_from_current_inputs=True,exact_witnesses_verified=checked,
                cohorts=summaries,parent_actual_validation_subset_pair_hits={k:dict(v) for k,v in counts.items()},
                exposure_sha256=sha(a.audit/'exposure.json'),script_sha256=sha(__file__),
                distinction='Validation-corpus membership alone is not optimizer exposure or actual validation-subset membership.')
    (a.audit/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.audit/'test_row_exposure.json').write_text(json.dumps(row_masks,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
