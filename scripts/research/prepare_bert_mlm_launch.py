"""Freeze a reviewed BERT smoke/train bundle; never enqueue implicitly.

Run: python -B -m scripts.research.prepare_bert_mlm_launch --phase smoke
Formal: --phase train --entity TEAM --run-id NEW_ID (after the smoke passes).
"""
import argparse
import json
import math
from pathlib import Path
import shutil

from scripts.research.bert_mlm_data import sha, write_json

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "output/bert_mlm_10epoch_20261016"
PYTHON = "/project_c20250601/251105016/conda/envs/protenix_abtcr/bin/python"


def prepare(phase, suffix, entity, run_id, smoke_dir):
    profile = json.loads((RUN / "profile/profile.json").read_text())
    if profile["status"] != "passed" or profile["world_size"] != 8 or profile["sequence_length"] != 2048:
        raise ValueError("Missing eight-rank final-length profile")
    if phase == "train":
        smoke = json.loads((smoke_dir / "COMPLETE.json").read_text())
        if smoke["status"] != "passed" or smoke["world_size"] != 8:
            raise ValueError("Eight-rank resume smoke did not pass")
        tested = json.loads((smoke_dir / "run_provenance.json").read_text())
        for relative in ("dllm/pipelines/immune_llada/fixed_chain/mlm.py",
                         "dllm/pipelines/immune_llada/fixed_chain/model.py",
                         "dllm/pipelines/immune_llada/fixed_chain/layout.py",
                         "scripts/research/run_bert_mlm.py", "scripts/research/bert_mlm_data.py",
                         "scripts/research/bert_mlm_eval.py"):
            if sha(ROOT / relative) != sha(Path(tested["source_root"]) / relative):
                raise ValueError("Training/evaluation changed after the eight-rank test: " + relative)
        for rank in range(8):
            receipt = json.loads((smoke_dir / f"smoke_rank{rank}.json").read_text())
            if not (receipt["status"] == "passed" and receipt["restored_state_exact"]
                    and receipt["corruption_exact"] and receipt["tail_records"] == 133):
                raise ValueError("Incomplete native resume/tail test")
    bundle = ROOT / ("output/v5_full_binding_20261007/launch_bert_mlm_"+phase+"_20261016"+suffix)
    bundle.mkdir(exist_ok=False)
    source = bundle / "source"
    for directory in ("dllm", "scripts", "examples"):
        for path in (ROOT / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    launcher = source / "scripts/research/launch_v5_direct.py"
    shutil.copy2(ROOT / "scripts/research/launch_bert_mlm.py", launcher)
    (source / "evaluate_bert.py").write_text('''"""Evaluate this bundle's BERT checkpoint: python -I -B THIS_FILE --help."""
import sys
import runpy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
runpy.run_module('scripts.research.bert_mlm_eval', run_name='__main__')
''')
    profiled_source = ROOT / "output/v5_full_binding_20261007/launch_bert_mlm_profile_20261016/source"
    for name in ("dllm/pipelines/immune_llada/fixed_chain/mlm.py", "dllm/pipelines/immune_llada/fixed_chain/model.py",
                 "dllm/pipelines/immune_llada/fixed_chain/layout.py", "scripts/research/profile_bert_mlm.py"):
        if sha(source / name) != sha(profiled_source / name):
            raise ValueError("Profiled model/corruption changed")
    index_path = RUN / "data/index.json"
    index = json.loads(index_path.read_text())
    model = json.loads((ROOT / "configs/models/v5_fixed_chain_f011_75m_20261012/immune_config.json").read_text())
    steps = math.ceil(index["records"]["train"]/512)*10
    training = dict(epochs=10, max_steps=steps, global_batch=512, micro_batch=profile["micro_batch"],
                    grad_accum=profile["grad_accum"], seed=42, learning_rate=1e-4,
                    warmup_steps=math.ceil(.05*steps), scheduler="cosine", weight_decay=.01,
                    adam_betas=[.9, .999], adam_epsilon=1e-8, gradient_clip=1., bf16=True,
                    eval_every=1000, log_every=25, loader_workers=4,
                    sampler="complete_epoch_no_replacement_no_padding_no_drop_last")
    wandb = dict(username="waitma", entity=entity, project="dllm-immune", run_id=run_id,
                 name="bert-fixed-chain-75m-v5-10epoch-20261016"+("-startup" if phase == "smoke" else ""),
                 credential_file="/project_c20250601/251105016/.secrets/wandb_api_key")
    vendor = RUN / "vendor"
    plan = dict(schema="immune.bert_mlm.launch.v1", phase=phase, source_root=str(source),
                output_dir=str(RUN / ("train" if phase == "train" else "smoke"+suffix)),
                model=model, training=training, data_index=str(index_path), data_index_sha256=sha(index_path),
                wandb=wandb, wandb_vendor=str(vendor), profile=str(RUN/"profile/profile.json"),
                profile_sha256=sha(RUN/"profile/profile.json"),
                sources={str(p): sha(p) for p in source.rglob("*.py")},
                vendor_files={str(p): sha(p) for p in vendor.rglob("*") if p.is_file()
                              and "__pycache__" not in p.parts and p.suffix != ".pyc"})
    if phase == "train":
        plan["smoke_receipt"] = dict(path=str(smoke_dir/"COMPLETE.json"), sha256=sha(smoke_dir/"COMPLETE.json"))
    write_json(bundle / "plan.json", plan)
    write_json(bundle / "direct_launch.json", dict(plan_sha256=sha(bundle/"plan.json"),
               runtime_dir=str(RUN / (phase+suffix+"_runtime"))))
    ticket = dict(bundle=str(bundle), launcher_sha256=sha(launcher),
                  purpose="User-authorized 10 complete epoch BERT representation pretraining: "+phase)
    write_json(RUN / "preparation" / (phase+suffix+"_ticket.json"), ticket)
    if phase == "train":
        config = ROOT / "configs/models/v5_bert_mlm_75m_10epoch_20261016"
        config.mkdir(exist_ok=False)
        write_json(config / "immune_config.json", model)
        write_json(config / "training_config.json", dict(**training, wandb=wandb,
                   bundle=str(bundle), output_dir=plan["output_dir"], data_index=str(index_path),
                   data_index_sha256=plan["data_index_sha256"], evaluation_script=str(source/"evaluate_bert.py")))
        peaks = [r["attempts"][-1]["reserved_gib"] for r in profile["reports"]]
        # This recipe uses the existing allocated container. YAML is an explicit
        # equivalent launch, for a future allocation only; never submit both.
        yaml = f'''TaskName: immune-bert75m-v5-10epoch-8gpu-20261016
Description: >-
  Scratch BERT representation training, V5 all seven sources, 10 complete epochs / {steps} steps.
  Global batch 512 = {profile['micro_batch']} x GA{profile['grad_accum']} x 8.
  A10080 BF16 profile length2048 peak reserved {max(peaks):.2f}GiB per GPU.
  Executed in existing container t-20261007153431-vhx2f; do not duplicate submission.
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
  export WANDB_PROJECT=dllm-immune WANDB_ENTITY={entity} WANDB_MODE=online
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
        (ROOT / "train_jobs/bert_fixed_chain_75m_v5_10epoch_8gpu_20261016.yml").write_text(yaml)
    print(json.dumps(dict(bundle=str(bundle), phase=phase, max_steps=steps,
                          micro_batch=profile["micro_batch"], grad_accum=profile["grad_accum"], enqueued=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "train"), required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--entity", default="pressure_jian")
    parser.add_argument("--run-id", default="bertv5mlm1016")
    parser.add_argument("--smoke-dir", type=Path, default=RUN/"smoke_r2")
    args = parser.parse_args()
    prepare(args.phase, args.suffix, args.entity, args.run_id, args.smoke_dir)
