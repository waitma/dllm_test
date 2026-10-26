"""Retry Kong after empty-light fix: python -B SCRIPT --run ABS [--check-only].

This process-local adapter leaves all running groups and their hashed sources
unchanged. A missing light chain is absent, never an empty BioSeqChain.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0,str(ROOT))


def cdr_input(model,heavy,light,role,start,end):
    from dllm.pipelines.immune_llada.data.records import BioSeqChain,BioSeqRecord
    from downstream.v5_layout import cdr_inputs
    target=heavy if role=='antibody_heavy' else light
    if role not in ('antibody_heavy','antibody_light') or not 0<=start<end<=len(target):
        raise ValueError('Invalid observed CDR target')
    placeholder=target[:start]+'A'*(end-start)+target[end:]
    heavy,light=(placeholder,light) if role=='antibody_heavy' else (heavy,placeholder)
    chains=[BioSeqChain(heavy,'antibody_heavy')]
    if light:chains.append(BioSeqChain(light,'antibody_light'))
    record=BioSeqRecord(chains,task_type='antibody',source='native_v5_cdr')
    return cdr_inputs(model,record,role,start,end)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.check_only:
        from downstream.v5_native_eval import load,cdr_input as old_input
        from downstream.v7_native import cdr_argmax
        model,_=load(a.run/'checkpoint','cpu')
        source=ROOT/'data/downstream/cdr_infilling/sabdab_kong/cdrh3/fold_0/test.json'
        rows=[json.loads(s) for s in source.read_text().splitlines()]
        row=next(r for r in rows if not r['light_chain_seq'])
        h=row['heavy_chain_seq'].replace('J','L');ref=row['cdrh3_seq'].replace('J','L');start=h.index(ref);end=start+len(ref)
        f,mask=cdr_input(model,h,'','antibody_heavy',start,end)
        altered=h[:start]+'G'*len(ref)+h[end:]
        g,other=cdr_input(model,altered,'','antibody_heavy',start,end)
        assert mask.equal(other) and all(f[k].equal(g[k]) for k in f)
        output,audit=cdr_argmax(model,f,mask);assert int(mask.sum())==len(ref)
        # Nonempty light path must remain exactly the original adapter.
        light='DIQMTQSPSSLSASVGDRVTITCRASQSIYSSLAWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPYTFGQGTKVEIK'
        x,m=cdr_input(model,h,light,'antibody_heavy',start,end);y,n=old_input(model,h,light,'antibody_heavy',start,end)
        assert m.equal(n) and all(x[k].equal(y[k]) for k in x)
        proof={'status':'passed','empty_light_forward':True,'hidden_target_invariance':True,'paired_path_unchanged':True,'source':str(source),'masked_residues':int(mask.sum())}
        (a.run/'kong_retry_checks.json').write_text(json.dumps(proof,indent=2)+'\n');print(proof);return
    from scripts.downstream import run_v5_cdr_eval as cdr
    cdr.cdr_input=cdr_input
    from scripts.downstream import run_v5_suite_task as task
    sys.argv=[task.__file__,'--task','cdr-kong','--run',str(a.run)]
    task.main()

if __name__=='__main__':main()
