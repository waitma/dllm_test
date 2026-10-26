"""Prepare (do not submit) official AB+TCR eval YAMLs for two v2 immune_v6 models.

Creates prune-safe hardlink snapshots plus independent SHA-locked copies, then
writes 32 preemptible 1-GPU YAMLs covering the RESULTS official matrix.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import yaml

from downstream.grammar.ab_features import file_sha256, write_json

ROOT = Path("/project_c20250601/251105016/project/dllm_test")
RUNTIME = Path("/project_c20250601/251105016/conda/envs/protenix_abtcr")
PLLM = Path("/project_c20250601/251105016/conda/envs/pllm")
DATE = "20260927"
QUEUE = "queue012"  # this identity cannot CreateCustomTask on c20250601
TEMPLATE = ROOT / "eval_jobs/eval_immune_270m_42000_pairing_reference_v3.yml"

T1_INPUT_MODE = {
    "cdr3b": "cdr3b",
    "others_cdr3b": "cdr3b",
    "cdr3ab": "cdr3ab",
    "others_longab": "longab",
}

MODELS = [
    {
        "key": "esmc",
        "label": "V6 ESMC fusion / fixed-canvas v2",
        "source": ROOT / "output/protein_esmc_llada270m_diffusion_v2_immune_v6_2node4gpu/checkpoint-15000",
        "train_root": ROOT / "output/protein_esmc_llada270m_diffusion_v2_immune_v6_2node4gpu",
        "step": 15000,
        "val_loss": 0.7391467690467834,
        "residue_cond_mode": "add",
        "source_balanced": True,
        "ab_tag": f"ab_v2_esmc_v6_15000_{DATE}",
        "tcr_tag": f"tcr_v2_esmc_v6_15000_{DATE}",
        "snap_name": f"v2_esmc_immune_v6_15000_{DATE}",
        "train_task": "t-20260923185956-wsvpq",
    },
    {
        "key": "noesmc",
        "label": "V6 no-ESMC decoder-only / fixed-canvas v2",
        "source": ROOT / "output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/checkpoint-110000",
        "train_root": ROOT / "output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m",
        "step": 110000,
        "val_loss": 0.7124133684805461,
        "residue_cond_mode": "token",
        "source_balanced": False,
        "ab_tag": f"ab_v2_noesmc_v6_110000_{DATE}",
        "tcr_tag": f"tcr_v2_noesmc_v6_110000_{DATE}",
        "snap_name": f"v2_noesmc_immune_v6_110000_{DATE}",
        "train_task": "t-20260924185250-w22zf",
    },
]


def common_env(checkpoint: Path) -> str:
    return f"""set -euo pipefail
source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate {PLLM}
conda activate {RUNTIME}
export LD_LIBRARY_PATH={RUNTIME}/lib:${{LD_LIBRARY_PATH:-}}
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
export HF_HOME=/project_c20250601/251105016/conda/cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {ROOT}
test -f {checkpoint}/model.safetensors
python -c 'import torch; assert torch.cuda.is_available(), "GPU required"; print(torch.cuda.get_device_name(0))'
"""


def job_config(task_name, description, entrypoint, tags, deadline=43200):
    config = copy.deepcopy(yaml.safe_load(TEMPLATE.read_text()))
    config.update(
        TaskName=task_name,
        Description=description,
        Entrypoint=entrypoint,
        Tags=tags,
        ResourceQueueName=QUEUE,
        Preemptible=True,
        ActiveDeadlineSeconds=deadline,
        Priority=6,
    )
    config["TaskRoleSpecs"] = [
        {"RoleName": "worker", "RoleReplicas": 1, "Flavor": "ml.pni2.3xlarge"}
    ]
    return config


def snapshot_model(model):
    source = model["source"]
    if not (source / "model.safetensors").is_file():
        raise FileNotFoundError(source)
    if not (source / "fusion_config.json").is_file():
        raise FileNotFoundError(source / "fusion_config.json")
    train_snap = model["train_root"] / f"eval_snapshot_{model['step']}"
    independent = ROOT / "output/ab_eval_checkpoints" / model["snap_name"]
    if train_snap.exists() or independent.exists():
        raise FileExistsError(f"snapshot already exists: {train_snap} or {independent}")
    for dest in (train_snap, independent):
        dest.mkdir(parents=True)
        os.link(source / "model.safetensors", dest / "model.safetensors")
        for path in source.iterdir():
            if path.is_file() and path.name != "model.safetensors" and path.suffix in {".json", ".txt", ".model"}:
                shutil.copy2(path, dest / path.name)
    sha = file_sha256(independent / "model.safetensors")
    if file_sha256(train_snap / "model.safetensors") != sha:
        raise ValueError("hardlink SHA mismatch")
    fusion = json.loads((independent / "fusion_config.json").read_text())
    if fusion.get("residue_cond_mode") != model["residue_cond_mode"]:
        raise ValueError(
            f"{model['key']} residue_cond_mode {fusion.get('residue_cond_mode')} "
            f"!= {model['residue_cond_mode']}"
        )
    if not fusion.get("fixed_receptor_lengths"):
        raise ValueError(f"{model['key']} is not a v2 fixed-canvas checkpoint")
    trainer_state = {
        "global_step": model["step"],
        "eval_snapshot": True,
        "source_checkpoint_was_weights_only": True,
        "source": str(source),
        "topk_val_loss": model["val_loss"],
        "reason": "best live topk_val_manifest eval_loss; trainer_state absent on slimmed top-k weights",
    }
    write_json(independent / "trainer_state.json", trainer_state)
    write_json(train_snap / "trainer_state.json", trainer_state)
    manifest = {
        "source": str(source),
        "train_hardlink": str(train_snap),
        "independent": str(independent),
        "sha256": sha,
        "global_step": model["step"],
        "topk_val_loss": model["val_loss"],
        "residue_cond_mode": model["residue_cond_mode"],
        "source_balanced_training": model["source_balanced"],
        "nlink_model_safetensors": (independent / "model.safetensors").stat().st_nlink,
        "selection": "best_val_not_latest",
        "date": DATE,
    }
    write_json(independent / "eval_snapshot_manifest.json", manifest)
    write_json(train_snap / "eval_snapshot_manifest.json", manifest)
    model["checkpoint"] = independent
    model["sha"] = sha
    model["train_snap"] = train_snap
    model["manifest"] = manifest
    return model


def write_yaml(name, config):
    path = ROOT / "eval_jobs" / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return path


def build_ab_jobs(model):
    ckpt = model["checkpoint"]
    tag = model["ab_tag"]
    out = ROOT / "output/downstream_generation" / tag
    out.mkdir(parents=True, exist_ok=True)
    env = common_env(ckpt)
    paths = {}
    pytest = (
        "python -m pytest scripts/tests/bioseq/test_pairing_generation_protocol.py "
        "scripts/tests/bioseq/test_sampling_bioseq.py scripts/tests/bioseq/test_ab_native_probes.py -q\n"
    )
    for name, dataset, modes, folds in (
        ("cdr-kong", "sabdab_kong", "cdrh1 cdrh2 cdrh3", 10),
        ("cdr-sab23", "sab23h2_converted", "cdrh1 cdrh2 cdrh3 cdrl1 cdrl2 cdrl3", 1),
    ):
        taskout = out / name
        command = env + pytest + f"""mkdir -p {taskout}
for AB_CDR_MODE in {modes}; do
  python -u -m downstream.grammar.cdr_infill --checkpoint-path {ckpt} \\
    --test-set {ROOT}/data/downstream/cdr_infilling/{dataset} --mode "${{AB_CDR_MODE}}" \\
    --device cuda --num-folds {folds} --sampling-strategy argmax --max-iter 2 \\
    --output-dir {taskout}/"${{AB_CDR_MODE}}" \\
    2>&1 | tee {taskout}/"${{AB_CDR_MODE}}.log"
done
"""
        paths[name] = write_yaml(
            f"eval_{tag}_{name}.yml",
            job_config(
                f"{tag}-{name}".replace("_", "-"),
                f"{model['label']} AB {name}; 1 GPU preemptible; v2 ckpt {model['step']}",
                command,
                ["immune", "ab", "v2", "spot", "1gpu"],
                86400,
            ),
        )
    prefix = out / "pair-p0-cfg0" / "holdout500"
    pairing = env + pytest + f"""mkdir -p {prefix.parent}
if [ -f {prefix}_n8.csv ]; then
  python scripts/downstream/validate_pairing_reuse.py --csv {prefix}_n8.csv --checkpoint {ckpt}
else
python -u -m downstream.grammar.light_chain_pairing --checkpoint-path {ckpt} \\
  --csv-path {ROOT}/data/downstream/comp_chain/test_data_oas_holdout.csv \\
  --output-csv {prefix}.csv --device cuda --heavy-batch-size 2 \\
  --num-seqs 8 --light-prompt-tokens 0 --cfg-scale 0 --light-length-mode auto \\
  --max-iter 16 --sampling-strategy gumbel_argmax --temperature 1 --seed 42 --skip-eval \\
  2>&1 | tee {prefix}_generation.log
fi
python -u scripts/downstream/score_pairing_generation.py --csv {prefix}_n8.csv \\
  --metrics {prefix}_metrics.json --num-seqs 8 \\
  2>&1 | tee {prefix}_scoring.log
"""
    paths["pair-p0-cfg0"] = write_yaml(
        f"eval_{tag}_pair-p0-cfg0.yml",
        job_config(
            f"{tag}-pair-p0-cfg0".replace("_", "-"),
            f"{model['label']} AB pairing p0/CFG0/iter16/auto; 1 GPU preemptible",
            pairing,
            ["immune", "ab", "v2", "spot", "1gpu"],
            86400,
        ),
    )
    for task in ("specificity", "gdp_a1", "m396"):
        taskout = out / (task + "-fv-v1" if task in {"specificity", "m396"} else task)
        input_policy = " --input-policy fixed-v2-fv" if task in {"specificity", "m396"} else ""
        command = env + pytest + f"""mkdir -p {taskout}
python -u -m downstream.grammar.ab_probes --task {task}{input_policy} --checkpoint {ckpt} \\
  --out-dir {taskout} --device cuda --embedding-batch-size 16 --max-length 1024 \\
  --epochs 100 --head-batch-size 8 --lr 5e-5 --seed 42 --split-seed 2023 --ridge-alpha 0.01 \\
  2>&1 | tee {taskout}/run.log
"""
        paths[task] = write_yaml(
            f"eval_{tag}_{task}.yml",
            job_config(
                f"{tag}-{task}".replace("_", "-"),
                f"{model['label']} AB {task}; 1 GPU preemptible; v2 ckpt {model['step']}",
                command,
                ["immune", "ab", "v2", "spot", "1gpu"],
                43200,
            ),
        )
    return paths


def build_tcr_jobs(model):
    ckpt = model["checkpoint"]
    tag = model["tcr_tag"]
    sha = model["sha"]
    step = model["step"]
    out = ROOT / "output/downstream_generation" / tag
    out.mkdir(parents=True, exist_ok=True)
    env = common_env(ckpt) + f"""export TCR_EVAL_TAG={tag}
export TCR_EVAL_CKPT={ckpt}
export TCR_EVAL_STEP={step}
export TCR_EVAL_SHA={sha}
"""
    paths = {}
    for track in ("cdr3b", "others_cdr3b", "cdr3ab", "others_longab"):
        key = f"t1-{track}"
        mode = T1_INPUT_MODE[track]
        from downstream.benchmark.tcr_binding.query_protocol import TCRBindingQueryProtocol
        protocol_sha = TCRBindingQueryProtocol(input_mode=mode).sha256
        guard = (
            "import json; from pathlib import Path; "
            "from downstream.benchmark.tcr_binding.retrained_protocol import sha256_file; "
            "from downstream.benchmark.tcr_binding.query_protocol import TCRBindingQueryProtocol; "
            f"assert sha256_file(Path({str(ckpt)!r})/'model.safetensors') == {sha!r}; "
            f"assert TCRBindingQueryProtocol(input_mode={mode!r}).sha256 == {protocol_sha!r}"
        )
        command = env + f"""mkdir -p {out}/{key}
python -c {json.dumps(guard)}
python -u downstream/benchmark/tcr_binding/run_retrained_ours.py \\
  --checkpoint {ckpt} --tag ours_{tag}_{track} --track {track} \\
  --neg-source AS --fold all --eval-set all --seed 0 --cdr3-format junction --max-length 1024 \\
  --embedding-batch-size 16 --feature-chunk-size 512 --head-hidden 256 --head-batch-size 512 \\
  --dropout 0.3 --learning-rate 0.001 --weight-decay 0.00001 \\
  --max-epochs 200 --patience 15 --val-fraction 0.1 \\
  2>&1 | tee {out}/{key}/run.log
"""
        paths[key] = write_yaml(
            f"eval_{tag}_{key}.yml",
            job_config(
                f"{tag}-{key}".replace("_", "-"),
                f"{model['label']} TCR {key}; 1 GPU preemptible; v2 ckpt {step}",
                command,
                ["tcr", "v2", "spot", "1gpu"],
                43200,
            ),
        )
    for task in ("t2a", "t3broad", "t3deep"):
        key = f"{task}-v2"
        gate = out / key / "gate.json"
        command = env + f"""mkdir -p {out}/{key}
python -c 'import torch; assert torch.cuda.is_available()'
python -m pytest -q scripts/tests/immune_llada/test_tcr_scoring_audit.py
python -u scripts/downstream/run_tcr_native_repr.py --task preflight --gate {gate}
python -u scripts/downstream/run_tcr_native_repr.py --task {task} --gate {gate} \\
  2>&1 | tee {out}/{key}/run.log
"""
        paths[key] = write_yaml(
            f"eval_{tag}_{key}.yml",
            job_config(
                f"{tag}-{key}".replace("_", "-"),
                f"{model['label']} TCR {key}; in-job GPU gate; 1 GPU preemptible",
                command,
                ["tcr", "v2", "spot", "1gpu"],
                43200,
            ),
        )
    for task in ("benchmark14", "held20"):
        key = f"t4-{task}"
        command = env + f"""mkdir -p {out}/{key}
python -u scripts/downstream/run_tcr_v2_generation_eval.py --task {task} --restart-incomplete \\
  2>&1 | tee {out}/{key}/run.log
"""
        paths[key] = write_yaml(
            f"eval_{tag}_{key}.yml",
            job_config(
                f"{tag}-{key}".replace("_", "-"),
                f"{model['label']} TCR {key} v2 EOS protocol; 1 GPU preemptible",
                command,
                ["tcr", "v2", "spot", "1gpu"],
                86400,
            ),
        )
    return paths


def main():
    plan_dir = ROOT / f"output/downstream_generation/v2_immune_v6_eval_{DATE}"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan = {"date": DATE, "queue": QUEUE, "preemptible": True, "models": [], "jobs": []}
    for model in MODELS:
        snapshot_model(model)
        ab = build_ab_jobs(model)
        tcr = build_tcr_jobs(model)
        record = {
            "key": model["key"],
            "label": model["label"],
            "step": model["step"],
            "val_loss": model["val_loss"],
            "sha256": model["sha"],
            "checkpoint": str(model["checkpoint"]),
            "train_hardlink": str(model["train_snap"]),
            "source": str(model["source"]),
            "residue_cond_mode": model["residue_cond_mode"],
            "source_balanced_training": model["source_balanced"],
            "train_task": model["train_task"],
            "ab_jobs": {k: str(v) for k, v in ab.items()},
            "tcr_jobs": {k: str(v) for k, v in tcr.items()},
        }
        plan["models"].append(record)
        for group in (ab, tcr):
            for key, path in group.items():
                plan["jobs"].append(
                    {"model": model["key"], "task": key, "yaml": str(path), "status": "prepared_not_submitted"}
                )
        print(f"SNAP {model['key']} step={model['step']} sha={model['sha']} nlink={model['manifest']['nlink_model_safetensors']}")
        for key, path in {**ab, **tcr}.items():
            print(f"YAML {key} {path}")
    write_json(plan_dir / "prepare_plan.json", plan)
    print("PLAN", plan_dir / "prepare_plan.json")


if __name__ == "__main__":
    main()
