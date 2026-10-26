"""Connect the pinned no-ESMC checkpoint to model-independent benchmark interfaces.

GPU/Volc only: python -m downstream.benchmark.ab_antigen.native_adapter --task binding
  --data DATA.jsonl --out-dir NEW_RUN
For frozen supervised probes use sequence_eval; this adapter returns zero-shot
scores or paired generation. It does not implement external baseline models.
"""
import argparse
import json
from pathlib import Path
from downstream.benchmark.ab_antigen.interfaces import load_dataset,run_adapter,evaluate,save_json


class NativeNoEsmcAdapter:
    def __init__(self,checkpoint=None,*,pll_batch_size=2):
        import torch
        from downstream.benchmark.ab_antigen.sequence_eval import CHECKPOINT,EXPECTED_SHA,sha
        from examples.llada.load_fusion_checkpoint import load_fusion_for_eval
        checkpoint=Path(checkpoint or CHECKPOINT)
        if pll_batch_size<1:raise ValueError('PLL batch size must be positive')
        config=json.loads((checkpoint/'fusion_config.json').read_text())
        state=json.loads((checkpoint/'eval_snapshot_manifest.json').read_text())
        if config.get('residue_cond_mode')!='token' or config.get('condition_hidden_size')!=0:
            raise ValueError('Expected no-ESMC configuration')
        if state.get('global_step')!=150000 or sha(checkpoint/'model.safetensors')!=EXPECTED_SHA:
            raise ValueError('Expected approved immutable step150000 weights')
        if not torch.cuda.is_available():raise RuntimeError('Use a Volc GPU job for actual checkpoint inference')
        self.bundle=load_fusion_for_eval(checkpoint,device='cuda',max_length=2048)
        self.bundle.model.eval().requires_grad_(False)
        self.pll_batch_size=pll_batch_size
        self.identity=dict(name='native_noesmc150000',checkpoint_sha256=EXPECTED_SHA,
            config_sha256=sha(checkpoint/'fusion_config.json'),tokenizer_sha256=sha(checkpoint/'tokenizer.json'),
            scorer_sha256=sha(Path(__file__).with_name('sequence_eval.py')),
            adapter_sha256=sha(__file__),pll_batch_size=pll_batch_size,max_length=2048)

    def score_pair(self,request):
        from downstream.benchmark.ab_antigen.sequence_eval import score_pair
        value,_=score_pair(self.bundle,request.antigen,request.heavy,request.light,
            batch_size=self.pll_batch_size,features=False)
        return value

    def generate(self,request):
        from downstream.benchmark.ab_antigen.sequence_eval import generate_one
        return generate_one(self.bundle,request.antigen,request.seed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['binding','spr','abbibench','chimera'],required=True)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--assay');parser.add_argument('--pll-batch-size',type=int,default=2)
    parser.add_argument('--include-extra-assays',action='store_true',help='Explicit supplementary AbBiBench evaluation')
    args=parser.parse_args()
    if args.task=='chimera':
        parser.error('This legacy adapter generates full VH/VL. CHIMERA now uses the H3 adapter: run_v5_suite_task.py --task ag-chimera')
    rows=load_dataset(args.data,args.task,split='test' if args.task=='chimera' else None,assay=args.assay,include_extra_assays=args.include_extra_assays)
    adapter=NativeNoEsmcAdapter(args.checkpoint,pll_batch_size=args.pll_batch_size)
    predictions=run_adapter(args.task,rows,adapter,args.out_dir,model_identity=adapter.identity)
    report=evaluate(args.task,rows,predictions)
    save_json(args.out_dir/'interface_metrics.json',report)
    if args.task=='chimera':
        from downstream.benchmark.ab_antigen.sequence_metrics import main as score_generation
        # Existing reference scorer reads predictions.jsonl; preserve the interface journal.
        destination=args.out_dir/'predictions.jsonl'
        content=''.join(json.dumps(r)+'\n' for r in predictions)
        if destination.exists() and destination.read_text()!=content:raise ValueError('Reference-scoring input mismatch')
        destination.write_text(content)
        score_generation(args.out_dir,reference_path=args.data)


if __name__=='__main__':main()
