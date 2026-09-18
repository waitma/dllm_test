"""Prepare, but never submit, independent non-preemptible single-GPU AB jobs.

Run: python scripts/downstream/prepare_ab_native_jobs.py --checkpoint /abs/selected_ckpt --tag ab_native_YYYYMMDD
The checkpoint must be explicitly chosen. Optionally --snapshot-checkpoint keeps
an eval-only copy (weights/tokenizer, no optimizer) safe from training top-k pruning.
Submit each emitted YAML with scripts/volc-no-proxy.sh and update PROJECT_PROCESS.md.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNTIME = Path("/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr")
PLLM = Path("/vepfs-mlp2/c20250601/251105016/conda/envs/pllm")


def build_jobs(checkpoint: Path, tag: str, queue: str = "c20250601"):
    """Pure config construction: six pairing, two CDR datasets, three probes."""
    template = yaml.safe_load((ROOT / "eval_jobs/eval_immune_270m_42000_pairing_reference_v3.yml").read_text())
    out = ROOT / "output/downstream_generation" / tag
    common = f"""set -euo pipefail
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate {PLLM}
export PYTHONPATH={ROOT}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/vepfs-mlp2/c20250601/251105016/conda/cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {ROOT}
conda activate {RUNTIME}
export LD_LIBRARY_PATH={RUNTIME}/lib:${{LD_LIBRARY_PATH:-}}
test -f {shlex.quote(str(checkpoint / 'model.safetensors'))}
python -c 'import torch; assert torch.cuda.is_available(), "GPU required"; print(torch.cuda.get_device_name(0))'
python -m pytest scripts/tests/bioseq/test_pairing_generation_protocol.py scripts/tests/bioseq/test_sampling_bioseq.py scripts/tests/bioseq/test_ab_native_probes.py -q
"""
    jobs = {}

    def add(key, commands):
        config = copy.deepcopy(template)
        config.update(TaskName=f"{tag}-{key}".replace("_", "-"), Description=f"Native ESMC+LLaDA AB: {key}; independent one GPU, non-preemptible",
                      Entrypoint=common + commands, Tags=["immune", "ab", "native", "1gpu"],
                      ResourceQueueName=queue, Preemptible=False, ActiveDeadlineSeconds=43200)
        config["TaskRoleSpecs"] = [{"RoleName": "worker", "RoleReplicas": 1, "Flavor": "ml.pni2.3xlarge"}]
        jobs[key] = config

    qckpt = shlex.quote(str(checkpoint))
    for prompt in (0, 3):
        for cfg in (0.0, 1.0, 1.5):
            key = f"pair-p{prompt}-cfg{cfg:g}".replace(".", "p")
            prefix = out / key / "holdout500"
            add(key, f"""mkdir -p {shlex.quote(str(prefix.parent))}
python -u -m downstream.grammar.light_chain_pairing --checkpoint-path {qckpt} \\
  --csv-path {ROOT}/data/downstream/comp_chain/test_data_oas_holdout.csv \\
  --output-csv {shlex.quote(str(prefix))}.csv --device cuda --heavy-batch-size 2 \\
  --num-seqs 8 --light-prompt-tokens {prompt} --cfg-scale {cfg:g} --light-length-mode reference \\
  --max-iter 124 --sampling-strategy gumbel_argmax --temperature 1 --seed 42 --skip-eval \\
  2>&1 | tee {shlex.quote(str(prefix))}_generation.log
python -u scripts/downstream/score_pairing_generation.py --csv {shlex.quote(str(prefix))}_n8.csv \\
  --metrics {shlex.quote(str(prefix))}_metrics.json --num-seqs 8 \\
  2>&1 | tee {shlex.quote(str(prefix))}_scoring.log
""")
    for task in ("specificity", "gdp_a1", "m396"):
        taskout = out / task
        add(task, f"""mkdir -p {shlex.quote(str(taskout))}
python -u -m downstream.grammar.ab_probes --task {task} --checkpoint {qckpt} \\
  --out-dir {shlex.quote(str(taskout))} --device cuda --embedding-batch-size 16 --max-length 1024 \\
  --epochs 100 --head-batch-size 8 --lr 5e-5 --seed 42 --split-seed 2023 --ridge-alpha 0.01 \\
  2>&1 | tee {shlex.quote(str(taskout))}/run.log
""")
    for name, dataset, modes, folds in (
        ("cdr-kong", "sabdab_kong", "cdrh1 cdrh2 cdrh3", 10),
        ("cdr-sab23", "sab23h2_converted", "cdrh1 cdrh2 cdrh3 cdrl1 cdrl2 cdrl3", 1),
    ):
        taskout = out / name
        add(name, f"""mkdir -p {shlex.quote(str(taskout))}
for AB_CDR_MODE in {modes}; do
  python -u -m downstream.grammar.cdr_infill --checkpoint-path {qckpt} \\
    --test-set {ROOT}/data/downstream/cdr_infilling/{dataset} --mode "${{AB_CDR_MODE}}" \\
    --device cuda --num-folds {folds} --sampling-strategy argmax --max-iter 1 \\
    2>&1 | tee {shlex.quote(str(taskout))}/"${{AB_CDR_MODE}}.log"
done
""")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--queue", default="c20250601")
    parser.add_argument("--snapshot-checkpoint", action="store_true")
    parser.add_argument("--with-preflight", action="store_true", help="Also prepare a separate GPU gate; run it before the eleven jobs")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,42}", args.tag):
        parser.error("tag must use lowercase letters/digits/_/- and be at most 43 chars")
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        parser.error("Checkpoint directory must contain model.safetensors")
    output = ROOT / "output/downstream_generation" / args.tag
    if output.exists():
        raise FileExistsError(f"Use a fresh tag, output exists: {output}")
    paths = [ROOT / "eval_jobs" / f"eval_{args.tag}_{key}.yml" for key in build_jobs(checkpoint, args.tag)]
    if any(p.exists() for p in paths):
        raise FileExistsError("YAML tag already exists")
    if args.snapshot_checkpoint:
        from downstream.grammar.ab_features import file_sha256, write_json
        source = checkpoint
        checkpoint = ROOT / "output/ab_eval_checkpoints" / args.tag
        checkpoint.mkdir(parents=True, exist_ok=False)
        original_sha = file_sha256(source / "model.safetensors")
        for path in source.iterdir():
            if path.is_file() and (path.name == "model.safetensors" or path.suffix in (".json", ".txt", ".model")):
                shutil.copy2(path, checkpoint / path.name)
        if file_sha256(checkpoint / "model.safetensors") != original_sha:
            raise ValueError("Checkpoint changed during snapshot")
        write_json(checkpoint / "eval_snapshot_manifest.json", {"source": str(source), "sha256": original_sha})
    output.mkdir(parents=True)
    jobs = build_jobs(checkpoint, args.tag, args.queue)
    for (key, config), path in zip(jobs.items(), paths):
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        print(f"bash {ROOT}/scripts/volc-no-proxy.sh ml_task submit --conf {path}")
    preflight_path = None
    if args.with_preflight:
        preflight_path = ROOT / "eval_jobs" / f"eval_{args.tag}_preflight.yml"
        if preflight_path.exists():
            raise FileExistsError(preflight_path)
        config = copy.deepcopy(next(iter(jobs.values())))
        config.update(TaskName=f"{args.tag}-preflight".replace("_", "-"),
                      Description="Native AB checkpoint GPU gate; not quality evaluation",
                      ActiveDeadlineSeconds=3600)
        config["Entrypoint"] = config["Entrypoint"].split("mkdir -p", 1)[0] + (
            f"python -u scripts/downstream/preflight_ab_native.py --checkpoint {shlex.quote(str(checkpoint))} "
            f"--out-dir {shlex.quote(str(output / 'preflight'))}\n")
        preflight_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        print(f"RUN FIRST: bash {ROOT}/scripts/volc-no-proxy.sh ml_task submit --conf {preflight_path}")
    (output / "job_plan.json").write_text(json.dumps({"checkpoint": str(checkpoint), "queue": args.queue,
        "preemptible": False, "gpus_per_job": 1, "jobs": [str(p) for p in paths],
        "preflight": str(preflight_path) if preflight_path else None, "status": "prepared_not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
