"""Create A0-A4 pilot YAMLs only after data and measured memory gates pass.

Run from the repository root after profiling and audit completion:
python scripts/prepare_immune_v3_pilots.py --profile-dir /absolute/profile --data-dir /absolute/immune_v7_mhc_sequence_v1
Submit generated YAMLs with the repository Volc workflow and log each task ID.
This script never submits jobs or changes existing data/checkpoints.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml
from examples.llada.immune_pretrain_v3 import assert_ready, sha
from scripts.profile_immune_v3 import code_hashes


def make_jobs(profile_dir,data_dir,destination,stop_step=100):
    if not 1<=stop_step<=1000:
        raise ValueError("Initial validation is limited to 1-1000 steps")
    assert_ready(data_dir)
    proof=json.loads((profile_dir/"COMPLETE.json").read_text())
    if proof.get("status")!="passed" or proof.get("summary_sha256")!=sha(profile_dir/"summary.json"):
        raise ValueError("Memory profile is incomplete or changed")
    if json.loads((profile_dir/"source_sha256.json").read_text())!=code_hashes():
        raise ValueError("Implementation changed since memory profiling")
    summary=json.loads((profile_dir/"summary.json").read_text())
    destination.mkdir(parents=True,exist_ok=True)
    paths=[]
    for ablation in ("A0","A1","A2","A3","A4"):
        item=summary[ablation]
        batch=item["recommended_micro_batch"]
        measured=[r for r in item["measurements"] if r["batch_size"]==batch and r["status"]=="passed"]
        if (len(measured)!=1 or batch<1 or 256%batch or
                measured[0].get("precision")!="fp32_parameters_adam_bf16_autocast"):
            raise ValueError(f"Invalid measured batch for {ablation}")
        config=yaml.safe_load((ROOT/"train_jobs/immune_v3_profile_20261001_queue012.yml").read_text())
        dataset_tag=data_dir.name
        output=ROOT/"output"/f"immune_v3_{ablation.lower()}_{dataset_tag}_seed42_pilot"
        command=[sys.executable,str(ROOT/"examples/llada/immune_pretrain_v3.py"),"--ablation",ablation,
                 "--data-dir",str(data_dir.resolve()),"--output-dir",str(output),"--batch-size",str(batch),
                 "--grad-accum",str(256//batch),"--max-steps","10000","--pilot-stop-step",str(stop_step),"--seed","42",
                 "--profile-dir",str(profile_dir.resolve())]
        # Resolve Python in the configured worker environment, not the machine
        # generating this YAML.
        command[0]="python"
        config["TaskName"]=f"immune-v3-{ablation.lower()}-{dataset_tag.replace('_','-')}-seed42-pilot"
        config["Description"]=(f"{dataset_tag} {ablation}; stop/checkpoint/eval at {stop_step} with fixed 10000-step cosine; "
                               f"global batch 256={batch}x{256//batch}; measured peak {measured[0]['max_allocated_gib']:.2f} GiB")
        prefix=config["Entrypoint"].split("  python -u scripts/profile_immune_v3.py")[0]
        # YAML block scalars normally remove indentation when parsed.
        prefix=prefix.split("python -u scripts/profile_immune_v3.py")[0]
        config["Entrypoint"]=prefix+shlex.join(command)+"\n"
        config["Tags"]=["immune","noesmc","v3",ablation.lower(),"pilot"]
        config["Preemptible"]=True
        config["ActiveDeadlineSeconds"]=7200
        path=destination/f"immune_v3_{ablation.lower()}_{dataset_tag}_seed42_pilot.yml"
        content=yaml.safe_dump(config,sort_keys=False,allow_unicode=True)
        if path.exists() and path.read_text()!=content:
            raise FileExistsError(f"Different pilot configuration already exists: {path}")
        path.write_text(content)
        paths.append(str(path))
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile-dir",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,default=ROOT/"data/prepared/immune_v7_mhc_sequence_v1")
    p.add_argument("--destination",type=Path,default=ROOT/"train_jobs")
    p.add_argument("--stop-step",type=int,default=100)
    args=p.parse_args()
    print(json.dumps(make_jobs(args.profile_dir,args.data_dir,args.destination,args.stop_step),indent=2))


if __name__=="__main__":
    main()
