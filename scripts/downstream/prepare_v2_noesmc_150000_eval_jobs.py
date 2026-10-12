"""Prepare official AB+TCR eval YAMLs for no-ESMC immune_v6 checkpoint-150000.

Reuses snapshot + YAML builders from prepare_v2_immune_v6_eval_jobs.py.
Does not submit. Does not touch training checkpoints except hardlinking weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/downstream"))

import prepare_v2_immune_v6_eval_jobs as base
from downstream.grammar.ab_features import write_json

DATE = "20260928"
base.DATE = DATE

MODEL = {
    "key": "noesmc",
    "label": "V6 no-ESMC decoder-only / fixed-canvas v2",
    "source": ROOT / "output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/checkpoint-150000",
    "train_root": ROOT / "output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m",
    "step": 150000,
    "val_loss": 0.7099563160112926,
    "residue_cond_mode": "token",
    "source_balanced": False,
    "ab_tag": f"ab_v2_noesmc_v6_150000_{DATE}",
    "tcr_tag": f"tcr_v2_noesmc_v6_150000_{DATE}",
    "snap_name": f"v2_noesmc_immune_v6_150000_{DATE}",
    "train_task": "t-20260924185250-w22zf",
}


def main():
    plan_dir = ROOT / f"output/downstream_generation/v2_immune_v6_eval_{DATE}"
    plan_dir.mkdir(parents=True, exist_ok=True)
    model = dict(MODEL)
    base.snapshot_model(model)
    ab = base.build_ab_jobs(model)
    tcr = base.build_tcr_jobs(model)
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
    jobs = []
    for group in (ab, tcr):
        for key, path in group.items():
            jobs.append(
                {
                    "model": "noesmc",
                    "step": 150000,
                    "task": key,
                    "yaml": str(path),
                    "status": "prepared_not_submitted",
                }
            )
    plan = {
        "date": DATE,
        "queue": base.QUEUE,
        "preemptible": True,
        "expected_jobs": 16,
        "models": [record],
        "jobs": jobs,
    }
    write_json(plan_dir / "prepare_plan.json", plan)
    print(f"SNAP noesmc step={model['step']} sha={model['sha']} nlink={model['manifest']['nlink_model_safetensors']}")
    print(f"INDEPENDENT {model['checkpoint']}")
    print(f"TRAIN_SNAP {model['train_snap']}")
    for key, path in {**ab, **tcr}.items():
        print(f"YAML {key} {path}")
    print("PLAN", plan_dir / "prepare_plan.json")


if __name__ == "__main__":
    main()
