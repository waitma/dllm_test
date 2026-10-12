"""Run one native V5 downstream task on a GPU: python -B SCRIPT --task NAME --run ABS.

Independent output paths, fixed checkpoint, original splits/scoring. No backbone
training, no checkpoint conversion, and no training-job mutation.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback
import time

ROOT=Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
sys.path.insert(0,str(ROOT))
BENCH=ROOT/'downstream/benchmark'
sys.path.insert(0,str(BENCH))
from downstream.v5_native_eval import load, NativeBackbone, sha


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n');temp.replace(path)


def invoke(module,args):
    sys.argv=[module.__file__,*map(str,args)];module.main()


def install(backbone):
    # Explicit per-process bridge; original model loaders and training files stay untouched.
    from common import model_api
    import downstream.benchmark.common.model_api as qualified
    original=model_api.build_embedder
    def factory(spec,**kwargs):
        if spec.startswith(('grammar:','native-v5:','tcr-v5:')):
            backbone.batch_size=kwargs.get('batch_size',8)
            if spec.startswith('tcr-v5:'):
                from downstream.grammar.tcr_features import TCRRuntimeEmbedder
                return TCRRuntimeEmbedder(backbone=backbone,batch_size=backbone.batch_size)
            return backbone
        return original(spec,**kwargs)
    model_api.build_embedder=qualified.build_embedder=factory
    def assets(checkpoint):
        return {p.name:sha(p) for p in Path(checkpoint).glob('*.json')}
    import tcr_binding.run_retrained_ours as binding
    binding._model_assets=assets
    binding.OURS_METHOD='Ours native V5 object302 frozen observed-mean + MLP'
    return binding


def probes(task,checkpoint,out,backbone):
    import torch
    from downstream.grammar import ab_probes as runner
    from dllm.pipelines.immune_llada.data.records import BioSeqChain,BioSeqRecord
    def features(pairs,**kwargs):
        records=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],task_type='antibody',source='native_v5_probe') for h,l in pairs]
        vectors=torch.from_numpy(backbone.vectors(records))
        identity={**backbone.identity,'pool':'observed_residue_mean','input_audit':kwargs.get('input_audit'),'n':len(pairs)}
        write(out/'feature_manifest.json',identity)
        return vectors,identity
    runner.cached_features=features
    args=['--task',task,'--checkpoint',checkpoint,'--out-dir',out,'--epochs','100','--embedding-batch-size','8']
    if task in ('specificity','m396'):args+=['--input-policy','fixed-v2-fv']
    invoke(runner,args)
    p=out/'metrics.json'; d=json.loads(p.read_text());d['model']='native_v5_object302_75.6M';write(p,d)


def representation(task,checkpoint,out,backbone,tag):
    from downstream.grammar.tcr_features import TCRRuntimeEmbedder
    from scripts.downstream import run_tcr_native_repr as runner
    runner.CKPT=checkpoint
    runner.TCRRuntimeEmbedder=lambda *a,**kw:TCRRuntimeEmbedder(backbone=backbone,batch_size=8)
    if task=='t3broad':runner.broad(out);return
    if task=='t2a':
        from tcr_clustering import run as module
        args=['--method','embed-threshold']
    elif task=='t2b':
        from tcr_clustering import run_embed_bench as module
        args=['--algos','kmeans']
    else:
        from tcr_representation import run_paper6 as module
        args=['--method','embed','--columns','cdr3b','cdr3a']
    invoke(module,[*args,'--embedder','tcr-v5:'+str(checkpoint),'--tag',tag])
    directory=runner.DATASETS[task][0]
    write(out/'result_location.json',{'path':str(BENCH/'outputs'/directory/tag)})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    out=a.run/a.task;out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists(): raise FileExistsError('Task already attempted; inspect status before a separately authorized retry')
    status={'status':'running','task':a.task,'started_unix':time.time()};write(out/'status.json',status)
    from scripts.downstream.report_v5_suite import update
    update(a.run)
    try:
        checkpoint=a.run/'checkpoint';model,identity=load(checkpoint);write(out/'model_identity.json',identity)
        backbone=NativeBackbone(model,identity);binding=install(backbone)
        tag=a.run.name+'_'+a.task
        if a.task in ('specificity','gdp_a1','m396'):probes(a.task,checkpoint,out,backbone)
        elif a.task.startswith('t1-'):
            binding.OUT_ROOT=out/'protocol_outputs'
            if a.task=='t1-cdr3b':
                import shutil
                baseline=BENCH/'outputs/tcr_binding_nm2025_retrained/comparison_AS.csv'
                binding.OUT_ROOT.mkdir(parents=True,exist_ok=True)
                shutil.copy2(baseline,binding.OUT_ROOT/baseline.name)
                write(out/'baseline_identity.json',dict(path=str(baseline),sha256=sha(baseline)))
            invoke(binding,['--checkpoint',checkpoint,'--tag',tag,'--track',a.task[3:],'--fold','all','--eval-set','all','--neg-source','AS','--embedding-batch-size','8'])
        elif a.task=='uptcr':
            import tcr_binding.run_uptcr_ours as module
            invoke(module,['--checkpoint',checkpoint,'--output',out/'protocol_outputs','--batch-size','8'])
        elif a.task in ('t2a','t2b','t3broad','t3deep'):representation(a.task,checkpoint,out,backbone,tag)
        elif a.task.startswith('cdr-'):
            from scripts.downstream import run_v5_cdr_eval as module
            # The entry's loader reuses this exact already-verified model.
            module.load=lambda *args:(model,identity)
            invoke(module,['--checkpoint',checkpoint,'--dataset',a.task[4:],'--output-root',out/'results'])
        else:
            from downstream.v5_native_generation_eval import run
            run(a.task,out,model,backbone,identity)
        status.update(status='success',finished_unix=time.time())
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc(),finished_unix=time.time());raise
    finally:
        write(out/'status.json',status)
        from scripts.downstream.report_v5_suite import update
        update(a.run)
        if status['status']=='success':
            # Keep predictions, metrics and identities; discard derived heads/features.
            for pattern in ('**/*.pt','**/*.joblib','**/*features*.npy','**/*.tmp'):
                for path in out.glob(pattern):path.unlink()

if __name__=='__main__':main()
