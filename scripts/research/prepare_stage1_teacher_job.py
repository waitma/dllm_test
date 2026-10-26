"""Prepare an offline ESMFold2 cache job using an already frozen teacher script.

Activate pllm; python -B -m scripts.research.prepare_stage1_teacher_job --help.
The generated Volc job loads one frozen teacher per GPU in the isolated teacher
environment. --verify-only checks pinned requests, code and weight hashes on CPU.
"""
import argparse
import json
from pathlib import Path
import shlex

import yaml

from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.prepare_stage1_training import ROOT, STORAGE


def prepare(args):
    bundle = Path(args.student_bundle).resolve()
    launch = json.loads((bundle/"launch.json").read_text())
    if sha(bundle/"plan.json") != launch["plan_sha256"]:
        raise ValueError("Frozen student bundle changed")
    plan = json.loads((bundle/"plan.json").read_text())
    script = Path(plan["source_root"])/"scripts/research/stage1_teacher.py"
    if sha(script) != plan["sources"][str(script)]:
        raise ValueError("Frozen teacher source changed")
    requests, weights, control, output = map(lambda x: Path(x).resolve(),
        (args.requests, args.model_dir, args.control, args.output))
    receipt = json.loads(requests.with_suffix(".receipt.json").read_text())
    if receipt["requests_sha256"] != sha(requests) or receipt["data_index_sha256"] != plan["data_index_sha256"]:
        raise ValueError("Requests are not bound to this corpus")
    provenance = json.loads((weights/"download_provenance.json").read_text())
    contract = plan["teacher_contract"]
    if provenance["revision"] != contract["revision"] or provenance["repository"] != contract["repository"]:
        raise ValueError("Wrong teacher revision")
    references = {str(script): sha(script), str(requests): sha(requests),
        str(weights/"config.json"): contract["config_sha256"],
        str(weights/"model.safetensors.index.json"): sha(weights/"model.safetensors.index.json"),
        **{str(weights/name): digest for name,digest in contract["weight_sha256"].items()}}
    control.mkdir(parents=True, exist_ok=False)
    config = dict(schema="immune.stage1.teacher_job.v1", script=str(script), requests=str(requests),
        output=str(output), model_dir=str(weights), references=references, workers=8,
        pairs_per_chain_pair=1024, num_loops=3, seed=42, submitted=False,
        quality_status="unvalidated", split_counts=receipt["counts"])
    write_json(control/"plan.json", config)
    references[str(control/"plan.json")] = sha(control/"plan.json")
    checks = "\n".join(f"{digest}  {path}" for path,digest in references.items())
    python = STORAGE/"conda/envs/stage1_esmfold2/bin/python"
    entry = f'''#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(Path(plan['source_root'])))}
source {STORAGE}/miniforge3/etc/profile.d/conda.sh
conda activate {STORAGE}/conda/envs/pllm
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sha256sum -c <<'TEACHER_SHA'
{checks}
TEACHER_SHA
if [[ " $* " == *" --verify-only "* ]]; then exit 0; fi
exec {python} -I -B -m torch.distributed.run --standalone --nproc_per_node=8 {shlex.quote(str(script))} --model-dir {shlex.quote(str(weights))} --requests {shlex.quote(str(requests))} --output {shlex.quote(str(output))} --resume-cache --num-loops 3 --pairs-per-chain-pair 1024 --seed 42 "$@"
'''
    (control/"entrypoint.sh").write_text(entry)
    (control/"entrypoint.sh").chmod(0o755)
    template = yaml.safe_load((ROOT/"train_jobs/bert_diffusion_edit_resume28000_nonspot_zenith_20261022.yml").read_text())
    template.update(TaskName=args.name, Entrypoint="bash "+str(control/"entrypoint.sh"),
        Description="Offline Stage1 teacher cache; 8 independent frozen ESMFold2-Fast workers; inference batch 1 x 8 GPUs, no gradient accumulation. mem_peak 待 profile; quality remains unvalidated. Requests preserve train/valid bindings; partial/synthetic chains excluded.",
        Tags=["bioseq", "stage1-teacher-cache", "8gpu", "nonpreemptible"], ActiveDeadlineSeconds=86400)
    yaml_path = ROOT/"train_jobs"/(args.name+".yml")
    with yaml_path.open("x") as stream:
        yaml.safe_dump(template, stream, sort_keys=False, allow_unicode=True)
    write_json(control/"submission.json", dict(yaml=str(yaml_path), yaml_sha256=sha(yaml_path), submitted=False))
    print(json.dumps(dict(control=str(control), yaml=str(yaml_path), submitted=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("student-bundle", "requests", "control", "output", "name"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--model-dir", default=str(STORAGE/"model_weights/esmfold2_fast/45fe8656f5b3ef493c17fcf9abe9a2968902e712"))
    prepare(parser.parse_args())
