"""Freeze Stage 1 code, parent/data/cache identities, and Volc submission files.

Activate pllm; run `python -B -m scripts.research.prepare_stage1_training --help`.
Creates a new package only. --phase online performs joint GPU qualification,
online replay smoke and exploratory training in one submitted shell entrypoint.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import shutil

import yaml

from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import PairConfig
from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.stage1_data import content_sha


ROOT = Path(__file__).resolve().parents[2]
STORAGE = Path("/project_c20250601/251105016")
PYTHON = STORAGE/"conda/envs/pllm/bin/python"
DEFAULT_RUN = ROOT/"output/stage1_training_20261022"
DEFAULT_PARENT = ROOT/"output/bert_diffusion_edit_v7_from287000_20261019/checkpoint"
DEFAULT_INDEX = ROOT/"output/bert_mlm_v7_300k_20261016/data/index_test_clean.json"


def prepare(args):
    bundle, output = Path(args.bundle).resolve(), Path(args.output).resolve()
    bundle.mkdir(parents=True, exist_ok=False)
    source = bundle/"source"
    # Copy Python source only; training never imports the changing working tree.
    for folder in ("dllm", "scripts", "examples"):
        for origin in (ROOT/folder).rglob("*.py"):
            relative = origin.relative_to(ROOT)
            target = source/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
    sources = {str(p): sha(p) for p in sorted(source.rglob("*.py"))}
    parent = Path(args.bert).resolve()
    model = json.loads((parent/"immune_config.json").read_text())
    receipt = json.loads((parent/"checkpoint_receipt.json").read_text())
    manifest_path = Path(args.cache_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["data_index_sha256"] != sha(args.data_index):
        raise ValueError("Cache index belongs to another corpus")
    profile_path, smoke_path = Path(args.profile_receipt).resolve(), Path(args.smoke_receipt).resolve()
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else None
    micro = profile["micro_batch"] if profile else 1
    settings = dict(max_steps=args.steps, global_batch=512, micro_batch=micro, loader_workers=4,
        backbone_lr=1e-5, pair_lr=1e-4, weight_decay=.01, warmup_steps=min(500, args.steps),
        kd_warmup_steps=1000, intra_weight=.05, inter_weight=.05, temperature=1., clip_norm=1.,
        precision="bf16", seed=42, log_every=args.log_every, save_every=500, eval_every=500,
        sampler="complete_epoch_natural_ratio", objective="all_observed_tokens_plus_chain_aware_kd")
    if args.phase == "smoke":
        settings.update(max_steps=3, global_batch=8, micro_batch=1, warmup_steps=1, loader_workers=0)
    online = None
    if args.phase == "online":
        settings["micro_batch"] = 1  # Selected by the joint GPU profile at launch.
        online = dict(mode="on_demand_write_through", residency="same_gpu_persistent_process",
            cache_root=str(Path(args.online_cache).resolve() / content_sha(manifest["teacher_contract"])),
            # Resolving a venv's Python symlink selects its base environment.
            model_dir=str(Path(args.teacher_model_dir).resolve()), python=str(Path(args.teacher_python).absolute()),
            request_timeout_seconds=3600, inference_batch_size=1)
        if not Path(online["python"]).is_file():
            raise FileNotFoundError(online["python"])
        if sha(source/"scripts/research/stage1_teacher.py") != manifest["teacher_contract"]["fast_compat_source_sha256"]:
            raise ValueError("Seed-cache teacher implementation differs from frozen teacher")
    pair = asdict(PairConfig())
    relevant = {str(p.relative_to(source)): sources[str(p)] for p in source.rglob("*.py")
        if str(p.relative_to(source)).startswith(("dllm/pipelines/immune_llada/", "dllm/pipelines/llada/"))
        or p.name in {"run_stage1_training.py", "run_stage1_distillation.py", "stage1_data.py", "stage1_corpus.py",
                      "run_bert_mlm.py", "bert_mlm_data.py", "bert_mlm_eval.py", "profile_bert_mlm.py",
                      "stage1_online_cache.py", "stage1_online_teacher.py", "stage1_teacher_worker.py", "run_stage1_online.py"}}
    # Allocation/resume mechanics depend on target format and pair ceiling.
    # The complete teacher contract and cache bytes remain separately frozen in
    # each training plan/checkpoint and in its scientific quality receipt.
    qualification = dict(model=model, pair=pair, teacher_output={k:manifest["teacher_contract"][k]
        for k in ("repository", "revision", "distogram_bins")},
        max_pairs_per_record=args.max_pairs_per_record, implementation_sha256=content_sha(relevant),
        precision=settings["precision"], optimizer="AdamW_foreach_false", activation_checkpointing="whole_layer",
        optimizer_settings={k: settings[k] for k in ("backbone_lr", "pair_lr", "weight_decay", "clip_norm")},
        world_size=8, max_length=model["max_length"])
    if online is not None:
        qualification["online_teacher"] = online
    gates = {"profile": str(profile_path), "smoke": str(smoke_path)}
    gate_hashes = {k: sha(v) if Path(v).exists() else None for k, v in gates.items()}
    quality = dict(path=str(Path(args.quality).resolve()), sha256=sha(args.quality)) if args.quality else None
    plan = dict(schema="immune.stage1.launch.v1", created_utc=datetime.now(timezone.utc).isoformat(),
        phase=args.phase, source_root=str(source), output_dir=str(output), sources=sources,
        model=model, pair=pair, training=settings, world_size=8, parent_checkpoint=str(parent),
        parent_model_sha256=sha(parent/"model.safetensors"), parent_step=receipt["step"],
        parent_receipt_sha256=sha(parent/"checkpoint_receipt.json"),
        data_index=str(Path(args.data_index).resolve()), data_index_sha256=sha(args.data_index),
        cache_manifest=str(manifest_path), cache_manifest_sha256=sha(manifest_path),
        teacher_contract=manifest["teacher_contract"], max_pairs_per_record=args.max_pairs_per_record,
        quality=quality, gates=gates, gate_hashes=gate_hashes, qualification=qualification,
        minimum_train_cache_coverage=args.minimum_train_cache_coverage, minimum_valid_caches=args.minimum_valid_caches)
    if online is not None:
        plan.update(online_teacher=online, teacher_quality_policy="exploratory_unvalidated",
                    cache_coverage_policy="generate_every_eligible_miss_before_update")
    write_json(bundle/"plan.json", plan)
    write_json(bundle/"launch.json", dict(plan_sha256=sha(bundle/"plan.json")))
    launcher = source/"scripts/research/launch_stage1_training.py"
    checks = "\n".join(f"{sha(p)}  {p}" for p in (bundle/"launch.json", bundle/"plan.json", launcher))
    entry = f'''#!/usr/bin/env bash
set -euo pipefail
test -d {shlex.quote(str(ROOT))}
cd {shlex.quote(str(source))}
export OPENSSL_CONF={STORAGE}/miniforge3/ssl/openssl.cnf
export SSL_CERT_FILE={STORAGE}/miniforge3/ssl/cert.pem
export SSL_CERT_DIR={STORAGE}/miniforge3/ssl/certs
source {STORAGE}/miniforge3/etc/profile.d/conda.sh
conda activate {STORAGE}/conda/envs/pllm
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -s 8192
sha256sum -c <<'STAGE1_SHA'
{checks}
STAGE1_SHA
if [[ " $* " == *" --verify-only "* ]]; then
  exec {PYTHON} -I -B {shlex.quote(str(launcher))} --bundle {shlex.quote(str(bundle))} "$@"
fi
exec {PYTHON} -I -B -m torch.distributed.run --standalone --nproc_per_node=8 {shlex.quote(str(launcher))} --bundle {shlex.quote(str(bundle))} "$@"
'''
    (bundle/"entrypoint.sh").write_text(entry)
    (bundle/"entrypoint.sh").chmod(0o755)
    template = yaml.safe_load((ROOT/"train_jobs/bert_diffusion_edit_resume28000_nonspot_zenith_20261022.yml").read_text())
    peak = max(x["reserved_gib"] for x in profile["attempts"][-1]) if profile and online is None else None
    global_batch = settings["global_batch"]
    per_rank = global_batch//8
    run_micro = settings["micro_batch"]
    partitions = [min(run_micro, per_rank-i) for i in range(0, per_rank, run_micro)]
    memory = f"profile reserved {peak:.2f} GiB" if peak is not None else "待 profile"
    batch_description = ("target global batch 512; bs candidates 64/48/32/24/16/12/8/4/2/1; "
        "ga=ceil(64/bs) x 8 GPU with remainder microbatch; three AdamW updates per candidate"
        if args.phase == "profile" else f"global batch {global_batch}; bs {run_micro} x ga {math.ceil(per_rank/run_micro)} x 8 GPU"
        + (f", tail micro {partitions[-1]}" if per_rank % run_micro else ""))
    if online is not None:
        batch_description = ("target global batch 512; joint teacher/student profile selects bs from 24/16/12/8/4/2/1; "
            "ga=ceil(64/bs) x8 with remainder microbatch; then online replay smoke and training")
    template.update(TaskName=args.name, Description=f"Stage1 {args.phase}; BERT287000 + all-layer pair feedback. "
        f"{batch_description}; "
        f"max_length {model['max_length']}; {memory}. "
        + ("On-demand ESMFold2 targets with atomic cache reuse; exploratory quality unvalidated. " if online is not None
           else "Sparse cached ESMFold2-Fast targets; ") + "independent full-state resume.",
        Entrypoint="bash "+str(bundle/"entrypoint.sh"), Tags=["bioseq", "stage1-structure", args.phase, "8gpu", "nonpreemptible"],
        ActiveDeadlineSeconds=2592000 if args.phase in {"train", "online"} else 86400)
    yaml_path = ROOT/"train_jobs"/(args.name+".yml")
    if yaml_path.exists():
        raise FileExistsError(yaml_path)
    yaml_path.write_text(yaml.safe_dump(template, sort_keys=False, allow_unicode=True))
    write_json(bundle/"submission.json", dict(yaml=str(yaml_path), yaml_sha256=sha(yaml_path),
        entrypoint=str(bundle/"entrypoint.sh"), submitted=False,
        command=f"volc ml_task submit --conf {yaml_path}", phase=args.phase,
        note=("Construction only. Online entrypoint qualifies joint GPU memory and DDP replay before exploratory training; "
              "missing eligible targets are generated; biological accuracy remains unvalidated." if online is not None else
              "Construction only. Formal training is gated by new profile, DDP replay, cache coverage and teacher quality.")))
    print(json.dumps(dict(bundle=str(bundle), yaml=str(yaml_path), phase=args.phase, submitted=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("profile", "smoke", "train", "online"), required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--bert", default=str(DEFAULT_PARENT))
    parser.add_argument("--data-index", default=str(DEFAULT_INDEX))
    parser.add_argument("--profile-receipt", default=str(DEFAULT_RUN/"profile/COMPLETE.json"))
    parser.add_argument("--smoke-receipt", default=str(DEFAULT_RUN/"smoke/COMPLETE.json"))
    parser.add_argument("--quality")
    parser.add_argument("--online-cache", default=str(DEFAULT_RUN/"online_teacher_cache"))
    parser.add_argument("--teacher-model-dir", default=str(STORAGE/"model_weights/esmfold2_fast/45fe8656f5b3ef493c17fcf9abe9a2968902e712"))
    parser.add_argument("--teacher-python", default=str(STORAGE/"conda/envs/stage1_esmfold2/bin/python"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100, help="Rank-zero training metrics interval in optimizer steps")
    parser.add_argument("--max-pairs-per-record", type=int, default=8192)
    parser.add_argument("--minimum-train-cache-coverage", type=float, default=.1)
    parser.add_argument("--minimum-valid-caches", type=int, default=128)
    args = parser.parse_args()
    if args.steps < 1 or args.log_every < 1 or args.max_pairs_per_record < 1 or not 0 < args.minimum_train_cache_coverage <= 1 or args.minimum_valid_caches < 1:
        parser.error("Invalid step/pair/coverage settings")
    prepare(args)
