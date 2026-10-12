"""Bounded CPU toy optimization check, not a biological or downstream benchmark.

Run after activating pllm:
python scripts/smoke_immune_v3_cpu.py --output-dir /absolute/new/output --steps 200
Uses synthetic strings, a 64-wide/two-layer scratch model, and no corpus/checkpoint.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from dllm.pipelines.immune_llada.data.records import BioSeqChain,BioSeqRecord
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig,ImmuneModel
from scripts.profile_immune_v3 import code_hashes


@torch.no_grad()
def measure(model,batch):
    model.eval()
    noise=torch.ones_like(batch["input_ids"],dtype=torch.float)
    losses=model(**batch,token_t=noise)
    features=model.feature_args(batch)
    features["input_ids"]=features["input_ids"].masked_fill(batch["eligible_mask"],model.tokenizer.mask_token_id)
    logits,_=model.denoise(**features)
    predictions=logits.argmax(-1)
    metrics={key:float(losses[key]) for key in ("loss","aa_loss","terminal_loss")}
    for name,mask in (("aa",batch["residue_mask"]),("terminal",batch["terminal_mask"])):
        mask=mask & batch["eligible_mask"]
        metrics[name+"_accuracy"]=float((predictions[mask]==batch["labels"][mask]).float().mean())
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--steps",type=int,default=200)
    args=p.parse_args()
    if not 1<=args.steps<=500:
        raise ValueError("CPU toy run is capped at 500 steps")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    rows=[BioSeqRecord([BioSeqChain("CAAAAAAF","tcr_beta"),BioSeqChain("CGGGW","tcr_alpha")],"tcr","synthetic_toy")]
    report=dict(kind="cpu_synthetic_toy_optimization",device="cpu",seed=42,steps=args.steps,
                biological_quality_claim=False,source_sha256=code_hashes(),runs={})
    for ablation in ("A1","A4"):
        torch.manual_seed(42)
        config=ImmuneConfig(ablation=ablation,d_model=64,n_layers=2,n_heads=4,mlp_hidden_size=128,
                            max_length=512,full_capacity=24,fragment_capacity=16)
        model=ImmuneModel(config)
        collate=ImmuneCollator(ablation,max_length=512,full_capacity=24,fragment_capacity=16)
        batch=collate(rows)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
        start=time.monotonic()
        initial=measure(model,batch)
        trace=[]
        for step in range(1,args.steps+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            level=(.25,.5,.75,1.)[(step-1)%4]
            result=model(**batch,token_t=torch.full_like(batch["input_ids"],level,dtype=torch.float))
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError(f"{ablation} nonfinite loss at {step}")
            result["loss"].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if step%50==0 or step==args.steps:
                metrics=measure(model,batch)
                trace.append(dict(step=step,gradient_norm=float(norm),**metrics))
                print(json.dumps(dict(ablation=ablation,**trace[-1])),flush=True)
        generation_batch=ImmuneCollator(ablation,mode="generation",max_length=512,full_capacity=24,fragment_capacity=16)(rows)
        model.eval()
        generated={str(n):model.generate(generation_batch,steps=n)["chains"] for n in (1,8)}
        result=dict(config=asdict(config),parameters=sum(p.numel() for p in model.parameters()),initial=initial,
                    final=measure(model,batch),trace=trace,generation=generated,elapsed_seconds=time.monotonic()-start)
        result["loss_decreased"]=result["final"]["loss"]<initial["loss"]
        report["runs"][ablation]=result
        (args.output_dir/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({a:{"initial_loss":r["initial"]["loss"],"final_loss":r["final"]["loss"],
                         "aa_accuracy":r["final"]["aa_accuracy"],"end_accuracy":r["final"]["terminal_accuracy"]}
                      for a,r in report["runs"].items()}),flush=True)


if __name__=="__main__":
    main()
