"""Bounded single-GPU memory sweep; use train_jobs/immune_v3_profile_20260930.yml.

Each batch candidate gets a fresh process. Stop at OOM or >=80% occupancy;
this profiles synthetic tensors only and never bypasses the training data gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def code_hashes():
    files=[ROOT/"examples/llada/immune_pretrain_v3.py",Path(__file__),
           ROOT/"dllm/pipelines/llada/models/modeling_llada.py",
           ROOT/"dllm/pipelines/immune_llada/data/grammar.py",
           ROOT/"dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py"]
    files+=list((ROOT/"dllm/pipelines/immune_llada/v3").glob("*.py"))
    files+=list((ROOT/"dllm/pipelines/immune_llada/v3").glob("*.json"))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def verify_memory_profile(directory,ablation,batch):
    """Validate exactly the measured code and production precision/batch."""
    proof=json.loads((directory/"COMPLETE.json").read_text())
    summary_path=directory/"summary.json"
    if (proof.get("status")!="passed" or
            proof.get("summary_sha256")!=hashlib.sha256(summary_path.read_bytes()).hexdigest()):
        raise ValueError("Memory profile is incomplete or changed")
    if json.loads((directory/"source_sha256.json").read_text())!=code_hashes():
        raise ValueError("Implementation changed since memory profiling")
    summary=json.loads(summary_path.read_text())
    measured=[r for r in summary[ablation]["measurements"] if r["batch_size"]==batch and r["status"]=="passed"]
    if len(measured)!=1 or measured[0].get("precision")!="fp32_parameters_adam_bf16_autocast":
        raise ValueError("Requested training batch/precision has no passing profile")
    return measured[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fingerprints=code_hashes()
    (args.output_dir/"source_sha256.json").write_text(json.dumps(fingerprints,indent=2)+"\n")
    summary={}
    for ablation in ["A0","A1","A2","A3","A4"]:
        results=[]
        for batch in [1,2,4,8,16,32,64,128,256]:
            out=args.output_dir/f"{ablation}_bs{batch}"
            out.mkdir(exist_ok=True)
            command=[sys.executable,str(ROOT/"examples/llada/immune_pretrain_v3.py"),"--ablation",ablation,
                     "--profile","--batch-size",str(batch),"--max-length","2048","--output-dir",str(out)]
            with (out/"profile.log").open("w") as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            if result.returncode:
                tail=(out/"profile.log").read_text()[-12000:]
                if "out of memory" in tail.lower():
                    results.append({"batch_size":batch,"status":"oom"})
                    break
                raise RuntimeError(f"Profile failed: {out/'profile.log'}")
            measured=json.loads((out/"profile.json").read_text())
            results.append(measured)
            print(json.dumps({"ablation":ablation,"batch":batch,"mem_peak_gib":measured["max_allocated_gib"]}),flush=True)
            if measured["max_allocated_gib"]/measured["total_gib"] >= .8:
                break
        passed=[r for r in results if r["status"]=="passed"]
        if not passed:
            raise RuntimeError(f"No feasible batch for {ablation}")
        preferred=[r for r in passed if r["max_allocated_gib"]/r["total_gib"] <= .85]
        choice=(preferred or passed)[-1]
        summary[ablation]={"measurements":results,"recommended_micro_batch":choice["batch_size"],
                           "minimum_accumulation_1gpu":256//choice["batch_size"],"profile_only":True}
        (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if code_hashes()!=fingerprints:
        raise RuntimeError("Profile source changed during execution; measurements cannot authorize pilots")
    (args.output_dir/"COMPLETE.json").write_text(json.dumps({"status":"passed","ablations":list(summary),
        "summary_sha256":hashlib.sha256((args.output_dir/"summary.json").read_bytes()).hexdigest()})+"\n")


if __name__=="__main__":
    main()
