"""Run an unchanged AB probe or T3-broad on one derived final-layer checkpoint.

Activate pllm/protenix_abtcr; from repository root:
python -B scripts/downstream/evaluate_final_preservation_probe.py \
  --checkpoint-dir ABS_PILOT_ARM --out ABS_NEW_TASK_DIR --task specificity
Tasks: specificity, m396, t3broad. Local GPU, existing heads/splits/defaults.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--task',choices=('specificity','m396','t3broad'),required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    status=dict(status='running',task=args.task,pid=os.getpid(),host=socket.gethostname(),started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',status)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        from downstream.final_layer_preservation_eval import load
        from downstream.fixed_chain_eval import FixedChainBackbone
        model,identity=load(args.checkpoint_dir.resolve())
        backbone=FixedChainBackbone(model,identity)
        backbone.name='fixed50000_preservation_'+args.checkpoint_dir.name
        write(out/'model_identity.json',identity)
        from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain
        sources=[Path(__file__),ROOT/'downstream/final_layer_preservation_eval.py',ROOT/'downstream/fixed_chain_eval.py']
        if args.task in ('specificity','m396'):
            from downstream.grammar import ab_probes as runner
            sources.append(Path(runner.__file__))
            def features(pairs,**kwargs):
                chunks=[]
                for start in range(0,len(pairs),8):
                    records=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],
                              task_type='antibody',source='native_v5_probe') for h,l in pairs[start:start+8]]
                    chunks.append(backbone.vectors(records))
                    if start%1024==0:
                        status.update(stage='features',completed=min(start+8,len(pairs)),total=len(pairs))
                        write(out/'status.json',status)
                        print('features',status['completed'],len(pairs),flush=True)
                x=np.concatenate(chunks)
                manifest={**identity,'pool':'observed_residue_mean','n':len(pairs),'input_audit':kwargs.get('input_audit')}
                write(out/'feature_manifest.json',manifest)
                np.save(out/'features.npy',x)
                status.update(stage='head_fit');write(out/'status.json',status)
                return torch.from_numpy(x),manifest
            runner.cached_features=features
            sys.argv=[runner.__file__,'--task',args.task,'--checkpoint',str(args.checkpoint_dir.resolve()),
                      '--out-dir',str(out),'--epochs','100','--embedding-batch-size','8','--input-policy','fixed-v2-fv']
            runner.main()
            path=out/'run_manifest.json';manifest=json.loads(path.read_text())
            manifest['model']=backbone.name;manifest['adaptation_identity']=identity;write(path,manifest)
            path=out/'metrics.json';metrics=json.loads(path.read_text())
            metrics['model']=backbone.name;metrics['run_manifest_sha256']=sha(out/'run_manifest.json');write(path,metrics)
        else:
            from downstream.grammar.tcr_features import TCRRuntimeEmbedder
            from scripts.downstream import run_tcr_native_repr as runner
            runner.CKPT=args.checkpoint_dir.resolve()
            runner.TCRRuntimeEmbedder=lambda *a,**kw:TCRRuntimeEmbedder(backbone=backbone,batch_size=8)
            sources += [Path(runner.__file__),ROOT/'downstream/grammar/tcr_features.py',ROOT/'downstream/benchmark/common/fewshot.py']
            _,paths,_=runner.inputs('t3broad')
            write(out/'data_identity.json',{str(p):sha(p) for p in paths})
            runner.broad(out)
        write(out/'execution.json',dict(task=args.task,checkpoint=str(args.checkpoint_dir.resolve()),identity=identity,
              code_sha256={str(p):sha(p) for p in sources},protocol='unchanged existing task splits, heads, seed and budgets'))
        status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc(),finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:write(out/'status.json',status)


if __name__=='__main__':main()
