"""Profile single-record all-layer feedback at requested canvas lengths.

In pllm: USE_TF=0 python -B -m scripts.research.profile_structure_mlm
  --bert ABS_CHECKPOINT --requests ABS_JSON --output ABS_JSON --lengths 310 1024 2048
Extra context is synthetic poly-A solely for allocation/throughput testing;
it is never sent to the teacher or saved as biological training data.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator,corrupt
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel,PairConfig
from scripts.research.run_stage1_distillation import write_json


def run(args):
    torch.set_num_threads(8)
    torch.manual_seed(42)
    model=StructureMLMModel.from_bert(args.bert,PairConfig()).cuda().train()
    base=BioSeqRecord.from_prepared_dict(json.loads(Path(args.requests).read_text())["requests"][0]["record"])
    collator=FullTokenMLMCollator(model.config)
    baseline_length=collator([base])["input_ids"].shape[1]
    def extended(n):
        return BioSeqRecord([BioSeqChain("A"*n,"antigen"),*base.chains],
                            "antibody_antigen","engineering_allocation_only")
    extra_overhead=collator([extended(1)])["input_ids"].shape[1]-1
    results=[]
    for length in args.lengths:
        if length < baseline_length or length > model.config.max_length:
            raise ValueError("Requested canvas outside supported bounds")
        record=base if length==baseline_length else extended(length-extra_overhead)
        clean=collator([record])
        if clean["input_ids"].shape[1] != length:
            raise ValueError("Profile canvas length differs")
        noisy=corrupt(clean,model.tokenizer,torch.Generator().manual_seed(51))
        noisy={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in noisy.items()}
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.time()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out=model(**noisy)
            loss=out["loss_sum"]/out["mlm_count"].clamp_min(1)
        loss.backward()
        torch.cuda.synchronize()
        finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if not finite or not torch.isfinite(loss):
            raise ValueError("Non-finite profile forward/backward")
        result=dict(canvas_tokens=length,batch_size=1,precision="bf16",seconds=time.time()-started,
                    peak_cuda_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                    all_eight_feedback_gradients=all(p.weight.grad.abs().sum()>0 for p in model.pair_bias_outputs),
                    status="passed")
        result["all_eight_feedback_gradients"]=bool(result["all_eight_feedback_gradients"])
        results.append(result)
        print(json.dumps(result),flush=True)
        del out,loss,noisy
    write_json(args.output,dict(scope="allocation_smoke_only",gpu=torch.cuda.get_device_name(),
                               pair_config=model.pair_config.__dict__,results=results))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("bert","requests","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--lengths",nargs="+",type=int,default=[310,1024,2048])
    run(parser.parse_args())
