"""Evaluate parent/derived final features for matched attention controls.

Activate pllm/protenix_abtcr; from repository root run:
python -B scripts/downstream/evaluate_final_attention_control.py \
  --checkpoint-dir ABS_COMPLETED_ARM --out ABS_NEW_TASK_DIR \
  --task specificity --attention-mode no_mask_keys
Use --parent for the original fixed50000 checkpoint. Tasks: specificity, m396, t3broad. Existing heads/splits/budgets are unchanged.
Reproduce the chosen saved GDP feature cache bitwise before extracting another
 task, so a native/no-mask mismatch cannot silently enter the comparison.
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
    parser.add_argument('--attention-mode',required=True,choices=('native','no_mask_keys'))
    parser.add_argument('--parent',action='store_true')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    status=dict(status='running',task=args.task,attention_mode=args.attention_mode,pid=os.getpid(),host=socket.gethostname(),started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',status)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        if args.parent:
            from downstream.fixed_chain_eval import load
        else:
            from downstream.final_layer_preservation_eval import load
        from downstream.fixed_chain_eval import FixedChainBackbone
        model,identity=load(args.checkpoint_dir.resolve())
        class NoMaskBackbone(FixedChainBackbone):
            def _final_hidden(self,batch):
                modified={**batch,'attention_mask':batch['attention_mask'] & batch['input_ids'].ne(self.model.tokenizer.mask_token_id)}
                assert not (batch['residue_mask'] & batch['attention_mask'] & ~modified['attention_mask']).any()
                return super()._final_hidden(modified)
        cls=FixedChainBackbone if args.attention_mode=='native' else NoMaskBackbone
        identity={**identity,'attention_mode':args.attention_mode,
                  'readout':'actual final RMSNorm hidden; observed-residue mean; '+args.attention_mode}
        backbone=cls(model,identity)
        backbone.name='fixed50000_'+args.checkpoint_dir.parent.name+'_'+args.attention_mode
        write(out/'model_identity.json',identity)
        from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain
        arm=args.checkpoint_dir.resolve()
        if not args.parent:
            verified=json.loads((arm/'verification.json').read_text())
            assert verified['status']=='passed' and verified['reloaded_features_bitwise_equal']
            assert json.loads((arm.parent/'status.json').read_text())['status']=='success'
        else:
            assert arm==ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
        from downstream.grammar.ab_probes import prepare_task
        _,pairs,_,_=prepare_task('gdp_a1')
        records=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],
                             task_type='antibody',source='native_v5_probe') for h,l in pairs]
        cache_key=None
        if args.parent:
            cache=ROOT/'output/diagnostics/fixed50000_depth_20261018/features.npz'
            cache_key=args.attention_mode+'_final_norm'
            expected=np.load(cache)[cache_key]
        else:
            cache=arm/('final_features.npy' if args.attention_mode=='native' else 'final_no_mask_features.npy')
            expected=np.load(cache)
        np.testing.assert_array_equal(backbone.vectors(records),expected)
        write(out/'readout_verification.json',dict(status='passed',attention_mode=args.attention_mode,
              saved_gdp_features_bitwise_equal=True,rows=len(records),cache=str(cache),cache_key=cache_key,cache_sha256=sha(cache)))
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
            class RecordingEmbedder:
                def __init__(self,*a,**kw):
                    self.embedder=TCRRuntimeEmbedder(backbone=backbone,batch_size=8)
                    self.calls=0
                def embed_pairs(self,beta,alpha):
                    x=self.embedder.embed_pairs(beta,alpha)
                    np.save(out/f'tcr_features_{self.calls}.npy',x)
                    write(out/f'tcr_inputs_{self.calls}.json',dict(cdr3b=beta,cdr3a=alpha))
                    self.calls+=1
                    assert self.calls<=2
                    return x
            runner.TCRRuntimeEmbedder=RecordingEmbedder
            sources += [Path(runner.__file__),ROOT/'downstream/grammar/tcr_features.py',ROOT/'downstream/benchmark/common/fewshot.py']
            _,paths,_=runner.inputs('t3broad')
            write(out/'data_identity.json',{str(p):sha(p) for p in paths})
            runner.broad(out)
        write(out/'execution.json',dict(task=args.task,checkpoint=str(args.checkpoint_dir.resolve()),identity=identity,
              attention_mode=args.attention_mode,parent_control=args.parent,code_sha256={str(p):sha(p) for p in sources},protocol='unchanged existing task splits, heads, seed and budgets'))
        status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc(),finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:write(out/'status.json',status)


if __name__=='__main__':main()
