"""Freeze V7 BERT packages for profiling, startup checks and 300000-step training.

After activating pllm, use the training environment:
  USE_TF=0 python -B -m scripts.research.prepare_bert_mlm_v7_launch --phase profile
Then --phase smoke, and --phase train after eight-rank checks pass.
This prepares a pool ticket and a queue012 YAML; it never enqueues either itself.
"""
import argparse
import json
from pathlib import Path
import shutil

from scripts.research.bert_mlm_data import sha, step_budget, write_json
from scripts.research.bert_mlm_v7_data import release_binding

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "output/bert_mlm_v7_300k_20261016"
DATA = ROOT / "data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_mhc_completed_20261015"
VENDOR = ROOT / "output/bert_mlm_10epoch_20261016/vendor"
PYTHON = "/project_c20250601/251105016/conda/envs/protenix_abtcr/bin/python"
PROFILE_BUNDLE = ROOT / "output/v5_full_binding_20261007/launch_bert_mlm_v7_profile_20261016"
MODEL_FILES = ("dllm/pipelines/immune_llada/fixed_chain/mlm.py",
               "dllm/pipelines/immune_llada/fixed_chain/model.py",
               "dllm/pipelines/immune_llada/fixed_chain/layout.py",
               "dllm/pipelines/immune_llada/research_next/model.py",
               "dllm/pipelines/immune_llada/research_refined/model.py",
               "dllm/pipelines/immune_llada/research/model.py",
               "dllm/pipelines/immune_llada/v3/model.py",
               "dllm/pipelines/llada/models/modeling_llada.py")
TRAIN_FILES = (*MODEL_FILES, "scripts/research/run_bert_mlm.py", "scripts/research/bert_mlm_data.py",
               "scripts/research/bert_mlm_v7_data.py", "scripts/research/bert_mlm_eval.py",
               "scripts/research/launch_bert_mlm.py")


def same_code(source, names):
    for relative in names:
        if sha(ROOT / relative) != sha(source / relative):
            raise ValueError("Code differs from tested snapshot: " + relative)


def prepare(phase, suffix, smoke_dir=None, *, objective="receptor_aa"):
    # Distinct snapshots, output paths and W&B runs keep the two objectives separate.
    full_token = objective == "all_observed_tokens"
    if objective not in ("receptor_aa", "all_observed_tokens"):
        raise ValueError("Unknown MLM objective")
    date = "20261017" if full_token else "20261016"
    name = "bert_full_token_v7" if full_token else "bert_mlm_v7"
    run = ROOT / f"output/{name}_300k_{date}"
    profile_bundle = ROOT / f"output/v5_full_binding_20261007/launch_{name}_profile_{date}"
    smoke_dir = smoke_dir or run / "smoke"
    extra_files = ("dllm/pipelines/immune_llada/fixed_chain/full_token_mlm.py",
                   "scripts/research/bert_mlm_objectives.py") if full_token else ()
    model_files, train_files = (*MODEL_FILES, *extra_files), (*TRAIN_FILES, *extra_files)
    bound = release_binding(DATA)
    model = bound["model"]
    profile = None
    if phase != "profile":
        profile = json.loads((run / "profile/profile.json").read_text())
        profiled = json.loads((profile_bundle / "plan.json").read_text())
        if (profile["status"] != "passed" or profile["world_size"] != 8
                or profile["sequence_length"] != model["max_length"] or profiled["model"] != model
                or profiled.get("objective", "receptor_aa") != objective):
            raise ValueError("Missing eight-rank memory profile for this V7 config")
        same_code(Path(profiled["source_root"]), (*model_files, "scripts/research/profile_bert_mlm.py"))
    budget = step_budget(bound["records"]["train"], 512, 300000)
    training = dict(budget_mode="steps", minimum_epochs=10, epochs=budget["epochs_started"],
                    max_steps=300000, global_batch=512, micro_batch=profile["micro_batch"] if profile else None,
                    grad_accum=profile["grad_accum"] if profile else None, seed=42, learning_rate=1e-4,
                    warmup_steps=15000, scheduler="cosine", weight_decay=.01,
                    adam_betas=[.9, .999], adam_epsilon=1e-8, gradient_clip=1., bf16=True,
                    eval_every=1000, log_every=25, loader_workers=4,
                    sampler="complete_epoch_no_replacement_no_padding_no_drop_last_then_partial_final_epoch")
    # Both objectives use exactly the same immutable records and protected probe split.
    index_path = RUN / "data/index_test_clean.json"
    if phase != "profile":
        index = json.loads(index_path.read_text())
        if index["records"] != bound["records"] or index["manifest_sha256"] != bound["manifest_sha256"]:
            raise ValueError("BERT index is not the complete V7 release")
    if phase == "train":
        smoke = json.loads((smoke_dir / "COMPLETE.json").read_text())
        tested = json.loads((smoke_dir / "run_provenance.json").read_text())
        if (smoke["status"] != "passed" or smoke["world_size"] != 8 or tested["model"] != model
                or tested["training"] != training or tested["data_index_sha256"] != sha(index_path)
                or tested.get("objective", "receptor_aa") != objective):
            raise ValueError("Eight-rank startup test did not cover this data/model/budget")
        same_code(Path(tested["source_root"]), train_files)
        for rank in range(8):
            receipt = json.loads((smoke_dir / f"smoke_rank{rank}.json").read_text())
            if not (receipt["status"] == "passed" and receipt["restored_state_exact"]
                    and receipt["corruption_exact"] and receipt["tail_records"] == budget["tail_records"]
                    and receipt["math_oracle"]["status"] == "passed"):
                raise ValueError("Incomplete checkpoint restore or V7 tail test")
    bundle = ROOT / f"output/v5_full_binding_20261007/launch_{name}_{phase}_{date}{suffix}"
    bundle.mkdir(exist_ok=False)
    source = bundle / "source"
    for directory in ("dllm", "scripts", "examples"):
        for path in (ROOT / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    # The persistent container uses this legacy transport name for all bundles.
    launcher = source / "scripts/research/launch_v5_direct.py"
    shutil.copy2(ROOT / "scripts/research/launch_bert_mlm.py", launcher)
    (source / "evaluate_bert.py").write_text('''"""Evaluate this frozen BERT version: python -I -B THIS_FILE --help."""
import sys
import runpy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
runpy.run_module('scripts.research.bert_mlm_eval', run_name='__main__')
''')
    run_prefix = "bertv7full" if full_token else "bertv7mlm"
    wandb = dict(username="waitma", entity="pressure_jian", project="dllm-immune",
                 run_id=run_prefix+("300k" if phase == "train" else "check")+date[-4:]+suffix,
                 name=f"{name}-75m-mhc-300k-{date}"+("-startup" if phase == "smoke" else ""),
                 credential_file="/project_c20250601/251105016/.secrets/wandb_api_key")
    plan = dict(schema="immune.bert_mlm.launch.v1", phase=phase, source_root=str(source),
                output_dir=str(run / (phase+suffix)), model=model, training=training, objective=objective,
                data_manifest_sha256=bound["manifest_sha256"], data_dir=str(DATA), budget=budget,
                wandb=wandb, wandb_vendor=str(VENDOR),
                sources={str(p): sha(p) for p in source.rglob("*.py")},
                vendor_files={str(p): sha(p) for p in VENDOR.rglob("*") if p.is_file()
                              and "__pycache__" not in p.parts and p.suffix != ".pyc"})
    if not plan["vendor_files"]:
        raise ValueError("Missing isolated W&B SDK")
    if phase != "profile":
        plan.update(data_index=str(index_path), data_index_sha256=sha(index_path),
                    profile=str(run/"profile/profile.json"), profile_sha256=sha(run/"profile/profile.json"))
    if phase == "train":
        plan["smoke_receipt"] = dict(path=str(smoke_dir/"COMPLETE.json"), sha256=sha(smoke_dir/"COMPLETE.json"))
    write_json(bundle / "plan.json", plan)
    write_json(bundle / "direct_launch.json", dict(plan_sha256=sha(bundle/"plan.json"),
               runtime_dir=str(run / (phase+suffix+"_runtime"))))
    ticket = dict(bundle=str(bundle), launcher_sha256=sha(launcher),
                  purpose=f"User-authorized V7 BERT {objective}, scratch / 300000 steps: {phase}")
    write_json(run / "preparation" / (phase+suffix+"_ticket.json"), ticket)
    if phase == "train":
        config_name = "v7_bert_full_token" if full_token else "v7_bert_mlm"
        config = ROOT / f"configs/models/{config_name}_75m_300k_{date}{suffix}"
        config.mkdir(exist_ok=False)
        write_json(config / "immune_config.json", model)
        write_json(config / "training_config.json", dict(**training, objective=objective, wandb=wandb, budget=budget,
                   bundle=str(bundle), output_dir=plan["output_dir"], data_dir=str(DATA),
                   data_index=str(index_path), data_index_sha256=plan["data_index_sha256"],
                   evaluation_script=str(source/"evaluate_bert.py")))
        peaks = [r["attempts"][-1]["reserved_gib"] for r in profile["reports"]]
        yaml = f'''TaskName: immune-{name.replace('_', '-')}-75m-300k-8gpu-{date}{suffix}
Description: >-
  Scratch BERT {objective}, cleaned V7 with completed MHC and slots167/135, exactly 300000 steps.
  Global batch 512 = {profile['micro_batch']} x GA{profile['grad_accum']} x 8; 24.576 data epochs.
  A10080 BF16 profile length2048 peak reserved {max(peaks):.2f}GiB per GPU.
  Uses existing container t-20261007153431-vhx2f; this YAML is an equivalent new allocation, do not submit twice.
Storages:
- Type: Vepfs
  MountPath: /project_c20250601/251105016
  VepfsId: vepfs-cnbj2c98dea54433
  SubPath: c20250601/251105016
Entrypoint: |
  set -euo pipefail
  source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
  conda activate pllm
  export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export WANDB_PROJECT=dllm-immune WANDB_ENTITY=pressure_jian WANDB_MODE=online
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 IMMUNE_SDPA_BACKEND=efficient
  export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec {PYTHON} -I -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 --no_python {PYTHON} -I -B -u {launcher} --bundle {bundle}
ImageUrl: cr-mlp-cn-beijing.cr.volces.com/public/airgen:v1
Framework: Custom
ResourceQueueName: queue012
TaskRoleSpecs:
- RoleName: worker
  RoleReplicas: 1
  Flavor: ml.pni2.28xlarge
ActiveDeadlineSeconds: 2592000
DelayExitTimeSeconds: 0
AccessType: Private
Preemptible: false
Priority: 6
RetryOptions:
  EnableRetry: false
  MaxRetryTimes: 0
'''
        job_name = "bert_full_token_75m_v7" if full_token else "bert_fixed_chain_75m_v7"
        (ROOT / f"train_jobs/{job_name}_300k_8gpu_{date}{suffix}.yml").write_text(yaml)
    print(json.dumps(dict(bundle=str(bundle), phase=phase, max_steps=300000,
                          micro_batch=training["micro_batch"], grad_accum=training["grad_accum"],
                          budget=budget, enqueued=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("profile", "smoke", "train"), required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--objective", choices=("receptor_aa", "all_observed_tokens"), default="receptor_aa")
    args = parser.parse_args()
    prepare(args.phase, args.suffix, args.smoke_dir, objective=args.objective)
