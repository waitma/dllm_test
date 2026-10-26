"""Freeze diffusion/edit profile, smoke, focal pilots, or formal continuation.

Activate pllm; python -B -m scripts.research.prepare_diffusion_continuation
--phase profile [--suffix _r2] [--enqueue]. Uses the existing retained eight-GPU
allocation; it never submits or cancels a platform allocation.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.diffusion_continuation_objective import default_recipe

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/"output/bert_diffusion_edit_v7_from287000_20261019"
POOL=ROOT/"output/immune_training_container_c20250601_20261007"
FILES=("diffusion_continuation_objective.py","diffusion_continuation_data.py",
       "diffusion_continuation_losses.py","diffusion_continuation_eval.py",
       "run_diffusion_continuation.py","launch_diffusion_continuation.py",
       "prepare_diffusion_continuation.py")


def prepare(phase,suffix="",enqueue=False):
    parent=RUN/"checkpoint"
    parent_prov=json.loads((parent/"run_provenance.json").read_text())
    parent_source=Path(parent_prov["source_root"])
    receipt=json.loads((parent/"checkpoint_receipt.json").read_text())
    recipe=default_recipe(0. if phase=="pilot_ce" else 1.)
    gates=[]
    micro=None
    registry=json.loads((RUN/"packages.json").read_text()) if (RUN/"packages.json").exists() else {}
    if phase!="profile":
        profile_path=Path(registry["profile"]["output"])/"COMPLETE.json"
        profile=json.loads(profile_path.read_text())
        if profile["status"]!="passed" or profile["world_size"]!=8 or profile["model"]!=parent_prov["model"]:
            raise ValueError("Missing successful actual-shape eight-GPU profile")
        tested=Path(registry["profile"]["source"])
        # Exact objective and profile code; recipe differs only by EOS focal coefficient.
        for name in ("diffusion_continuation_objective.py","diffusion_continuation_losses.py","run_diffusion_continuation.py"):
            if sha(ROOT/"scripts/research"/name)!=sha(tested/"scripts/research"/name):
                raise ValueError("Code differs from GPU profile: "+name)
        micro=profile["micro_batch_views"]
        gates.append(dict(path=str(profile_path),sha256=sha(profile_path)))
    if phase in ("pilot_ce","pilot_focal","train"):
        smoke_path=Path(registry["smoke"]["output"])/"COMPLETE.json"
        smoke=json.loads(smoke_path.read_text())
        if smoke["status"]!="passed" or smoke["world_size"]!=8:
            raise ValueError("Missing successful eight-rank checkpoint replay")
        for rank in range(8):
            path=smoke_path.parent/f"smoke_rank{rank}.json"
            if json.loads(path.read_text())["status"]!="passed":
                raise ValueError("Incomplete rank replay")
            gates.append(dict(path=str(path),sha256=sha(path)))
        for name in FILES:
            if sha(ROOT/"scripts/research"/name)!=sha(Path(registry["smoke"]["source"])/"scripts/research"/name):
                raise ValueError("Source changed after startup validation: "+name)
        gates.append(dict(path=str(smoke_path),sha256=sha(smoke_path)))
    if phase=="train":
        choice_path=RUN/"focal_selection.json"
        choice=json.loads(choice_path.read_text())
        if choice["status"]!="complete" or choice["parent_model_sha256"]!=receipt["files"]["model.safetensors"]:
            raise ValueError("Missing paired focal pilot selection")
        recipe=default_recipe(choice["selected_eos_gamma"])
        gates.append(dict(path=str(choice_path),sha256=sha(choice_path)))
    training=dict(max_steps=200 if phase.startswith("pilot_") else 10000,
                  warmup_steps=40 if phase.startswith("pilot_") else 1000,
                  learning_rate=2e-5,global_batch=512,micro_batch_views=micro,
                  complementary_views=2,seed=42,loader_workers=4,
                  log_every=25,eval_every=200 if phase.startswith("pilot_") else 500,
                  weight_decay=.01,gradient_clip=1.,scheduler="warmup_cosine_0.01_to_1_to_0.1",
                  sampler="equal_source_continuous_without_replacement_then_reshuffle",
                  optimizer_reset=True)
    bundle=ROOT/f"output/v5_full_binding_20261007/launch_diffusion_edit287000_{phase}_20261019{suffix}"
    bundle.mkdir(exist_ok=False)
    source=bundle/"source"
    # Preserve the actual pretrained implementation, then overlay the new objective/driver.
    for directory in ("dllm","scripts","examples"):
        for path in (parent_source/directory).rglob("*.py"):
            target=source/path.relative_to(parent_source)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    for name in FILES:
        shutil.copy2(ROOT/"scripts/research"/name,source/"scripts/research"/name)
    launcher=source/"scripts/research/launch_v5_direct.py"
    shutil.copy2(ROOT/"scripts/research/launch_diffusion_continuation.py",launcher)
    output=RUN/(phase+suffix)
    vendor=ROOT/"output/bert_mlm_10epoch_20261016/vendor"
    plan=dict(schema="immune.diffusion_edit.launch.v1",phase="train" if phase.startswith("pilot_") else phase,
              stage=phase,source_root=str(source),output_dir=str(output),model=parent_prov["model"],
              training=training,recipe=recipe,parent_checkpoint=str(parent),
              parent_model_sha256=receipt["files"]["model.safetensors"],parent_step=287000,
              data_index=parent_prov["data_index"],data_index_sha256=parent_prov["data_index_sha256"],
              sources={str(p):sha(p) for p in source.rglob("*.py")},gates=gates,
              created_utc=datetime.now(timezone.utc).isoformat())
    if phase=="train":
        plan.update(wandb=dict(username="waitma",entity="pressure_jian",project="dllm-immune",
                 run_id="diffedit287k1019"+suffix,name="V7-BERT287k-diffusion-edit-balanced"+suffix,
                 credential_file=str(ROOT.parents[1]/".secrets/wandb_api_key")),
                 wandb_vendor=str(vendor),vendor_files={str(p):sha(p) for p in vendor.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc"})
    write_json(bundle/"plan.json",plan)
    write_json(bundle/"direct_launch.json",dict(plan_sha256=sha(bundle/"plan.json"),runtime_dir=str(RUN/(phase+suffix+"_runtime"))))
    ticket=dict(bundle=str(bundle),launcher_sha256=sha(launcher),purpose="User-authorized BERT287000 conditional diffusion/edit: "+phase)
    write_json(bundle/"ticket.json",ticket)
    registry[phase]=dict(bundle=str(bundle),source=str(source),output=str(output),ticket=str(bundle/"ticket.json"))
    write_json(RUN/"packages.json",registry)
    if enqueue:
        state=json.loads((POOL/"state.json").read_text())
        if state["state"]!="standby" or list((POOL/"pending").glob("*.json")) or list((POOL/"running").glob("*.json")):
            raise ValueError("Pool is not idle; package prepared but not enqueued")
        name=f"diffusion_edit287000_{phase}_20261019{suffix}.json"
        if (POOL/"finished"/name).exists():
            raise ValueError("Ticket was already attempted; use a new suffix")
        write_json(POOL/"pending"/name,ticket)
        (POOL/"PAUSE_TRAINING").unlink(missing_ok=True)
    print(json.dumps(dict(**registry[phase],enqueued=enqueue),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("profile","smoke","pilot_ce","pilot_focal","train"),required=True)
    parser.add_argument("--suffix",default="")
    parser.add_argument("--enqueue",action="store_true")
    args=parser.parse_args()
    prepare(args.phase,args.suffix,args.enqueue)
