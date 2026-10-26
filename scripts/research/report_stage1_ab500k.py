"""Compare completed AB500K arms without mixing MLM and KD objectives.

Activate pllm; python -B scripts/research/report_stage1_ab500k.py --plan ABS_PLAN
Optional --output ABS_DIR selects the report destination. Teacher GPU hours are
read from the sixteen bound worker completion receipts, including resume sessions.
Produces JSON plus standalone PNG/PDF learning curves; never trains or submits jobs.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from datetime import datetime
import json
import math
import numpy as np

from scripts.research.bert_mlm_data import sha, write_json, step_budget


def load_arm(root, arm, settings=None):
    settings = settings or dict(max_steps=3000, global_batch=512, eval_every=250)
    max_steps = settings["max_steps"]
    directory = Path(root)/arm
    complete = json.loads((directory/"COMPLETE.json").read_text())
    if complete.get("status") != "success" or complete.get("step") != max_steps:
        raise ValueError("Comparison requires both arms to complete the planned budget")
    if complete.get("seen_records") != step_budget(500000, settings["global_batch"], max_steps)["seen_records"]:
        raise ValueError("Completed exposure differs from the planned budget")
    provenance = json.loads((directory/"run_provenance.json").read_text())
    values = sorted((json.loads(path.read_text()) for path in directory.glob("validation_step*.json")),
                    key=lambda item: item["step"])
    expected = sorted(set(range(0, max_steps+1, settings["eval_every"])) | {max_steps})
    by_step = {item["step"]: item for item in values}
    if not set(expected).issubset(by_step):
        raise ValueError("Missing a scheduled validation checkpoint")
    # STOP-triggered validation is diagnostic and does not change the frozen comparison grid.
    values = [by_step[step] for step in expected]
    for value in values:
        if value["records"] != 4096 or value["masks"] != 3:
            raise ValueError("Validation population differs")
    return dict(complete=complete, provenance=provenance, validation=values)


def first_target_step(values, target):
    # Two successive scheduled observations make a one-point fluctuation insufficient.
    for first, second in zip(values, values[1:]):
        if first["step"] and first["aa_ce"] <= target and second["aa_ce"] <= target:
            return first["step"]
    return None


def paired_bootstrap(a_values, d_values, repeats=2000, seed=20261022):
    """Resample held-out paired-sequence records; this does not estimate training-seed variance."""
    matrices = []
    identities = None
    for values in (a_values[-3:], d_values[-3:]):
        total = None
        for value in values:
            ref = value["record_statistics"]
            if sha(ref["path"]) != ref["sha256"]:
                raise ValueError("Validation record statistics changed")
            with np.load(ref["path"]) as saved:
                rows, stats = saved["row_ids"], saved["statistics"]
                if identities is None:
                    identities = rows.copy()
                elif not np.array_equal(identities, rows):
                    raise ValueError("Bootstrap requires identical held-out records and order")
                total = stats.copy() if total is None else total + stats
        matrices.append(total)
    a, d = matrices
    if not np.array_equal(a[:, [1, 3, 5]], d[:, [1, 3, 5]]):
        raise ValueError("A/D validation masks or token denominators differ")
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(repeats):
        indices = rng.integers(0, len(a), len(a))
        aa, dd = a[indices].sum(0), d[indices].sum(0)
        loss_a, loss_d = aa[2]/aa[3], dd[2]/dd[3]
        estimates.append((loss_a-loss_d)/loss_a)
    return dict(method="paired_record_bootstrap", repeats=repeats, seed=seed,
        relative_aa_ce_improvement_ci95=np.quantile(estimates, [.025, .975]).tolist(),
        scope="held-out record uncertainty; single training seed; no training-seed confidence claim")


def compare(a, d):
    if a["provenance"].get("initialization") != d["provenance"].get("initialization"):
        raise ValueError("A/D initialization differs")
    for key in ("plan_sha256", "subset_manifest_sha256", "parent_model_sha256",
                "training", "noise_recipe", "noise_parent_records", "recipe", "validation_seeds", "world_size"):
        if a["provenance"][key] != d["provenance"][key]:
            raise ValueError("A/D comparison contract differs: " + key)
    if a["provenance"].get("cache_manifest_sha256") is not None:
        raise ValueError("Independent MLM must not bind teacher cache")
    if not d["provenance"].get("cache_manifest_sha256"):
        raise ValueError("Pair/KD requires a bound teacher cache")
    av, dv = a["validation"], d["validation"]
    if not math.isclose(av[0]["aa_ce"], dv[0]["aa_ce"], rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("A/D initial MLM predictions differ")
    final = {arm: {metric: float(np.mean([row[metric] for row in values[-3:]]))
                   for metric in ("aa_ce", "full_token_ce")}
             for arm, values in (("mlm", av), ("pair_kd", dv))}
    reduction = 1-final["pair_kd"]["aa_ce"]/final["mlm"]["aa_ce"]
    full_ce_nonworse = final["pair_kd"]["full_token_ce"] <= final["mlm"]["full_token_ce"]
    ce_signal = reduction >= .01 and full_ce_nonworse
    target = av[0]["aa_ce"]*.98
    steps = {arm: first_target_step(values, target) for arm, values in (("mlm", av), ("pair_kd", dv))}
    speed_signal = (all(step is not None for step in steps.values())
                    and steps["pair_kd"] <= .8*steps["mlm"] and full_ce_nonworse)
    return dict(schema="immune.stage1.ab500k.comparison.v1", status="complete",
        conclusion="promising_single_seed_signal" if ce_signal or speed_signal else "no_prespecified_positive_signal",
        last_three_steps=[row["step"] for row in av[-3:]], last_three_mean=final,
        relative_aa_ce_reduction=reduction, full_token_ce_nonworse=full_ce_nonworse, lower_ce_signal=ce_signal,
        convergence=dict(target_aa_ce=target, definition="0.98 times shared initial AA CE; two successive scheduled observations",
                         first_target_step=steps, twenty_percent_fewer_steps_signal=bool(speed_signal)),
        uncertainty=paired_bootstrap(av, dv),
        attribution="A/D tests the full pair-feedback plus KD package; architecture versus teacher attribution needs C",
        teacher_structure_accuracy="unvalidated", training_seed=42,
        next_action="Repeat A/D with an independent seed and add C before scaling or attributing benefit" if ce_signal or speed_signal
                    else "Keep the 500K scope and investigate; do not infer a full-data gain")


def render_plot(a, d, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, panels = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = panels.ravel()
    for data, label, color in ((a, "A: MLM BERT", "#3567a7"), (d, "D: Pair feedback + KD", "#c45535")):
        values = data["validation"]
        for axis, field, title in ((axes[0], "aa_ce", "Antibody residue CE"),
                                   (axes[1], "full_token_ce", "Full-token MLM CE")):
            axis.plot([row["step"] for row in values], [row[field] for row in values], label=label, color=color)
            axis.set(xlabel="Optimizer updates", ylabel=title)
            axis.grid(alpha=.2)
        axes[2].plot([row["wall_seconds"]/3600 for row in values], [row["aa_ce"] for row in values], label=label, color=color)
        axes[3].plot([row["seen_records"] for row in values], [row["aa_ce"] for row in values], label=label, color=color)
    axes[0].legend(frameon=False)
    axes[2].set(xlabel="Student elapsed hours (including validation)", ylabel="Antibody residue CE")
    axes[2].grid(alpha=.2)
    axes[3].set(xlabel="Training records seen", ylabel="Antibody residue CE")
    axes[3].grid(alpha=.2)
    fig.savefig(output/"learning_curves.png", dpi=180)
    fig.savefig(output/"learning_curves.pdf")
    plt.close(fig)


def teacher_timing(plan):
    """Account for every GPU worker; interrupted-heartbeat totals remain lower bounds."""
    subset = json.loads(Path(plan["subset_manifest"]).read_text())
    manifest = json.loads(Path(plan["cache_manifest"]).read_text())
    bound = {item["path"]: item["sha256"] for item in manifest.get("workers", [])}
    seconds, receipts, starts, ends, exact = 0., [], [], [], True
    for spec in sorted(subset["request_shards"], key=lambda item: item["worker_index"]):
        path = Path(plan["cache_root"])/f"worker{spec['worker_index']:02d}"/"complete.json"
        digest = sha(path)
        if bound and bound.get(str(path)) != digest:
            raise ValueError("Teacher completion receipt changed")
        value = json.loads(path.read_text())
        if (value["status"] != "passed" or value["count"] != spec["count"]
                or value["requests_sha256"] != spec["sha256"]
                or value["subset_manifest_sha256"] != sha(plan["subset_manifest"])
                or value["teacher_contract"] != manifest["teacher_contract"]):
            raise ValueError("Teacher timing receipt belongs to another workload")
        elapsed = value["seconds"]
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid teacher GPU time")
        seconds += elapsed
        exact &= value.get("timing_complete", False)
        if value.get("start_utc") and value.get("end_utc"):
            starts.append(datetime.fromisoformat(value["start_utc"].replace("Z", "+00:00")))
            ends.append(datetime.fromisoformat(value["end_utc"].replace("Z", "+00:00")))
        receipts.append(dict(path=str(path), sha256=digest, seconds=elapsed,
                             timing_complete=value.get("timing_complete", False), timing_note=value.get("timing_note")))
    if len(receipts) != 16:
        raise ValueError("Teacher timing requires all sixteen workers")
    return dict(gpu_hours=seconds/3600, status="measured" if exact else "lower_bound_after_unclean_exit",
        wall_seconds=(max(ends)-min(starts)).total_seconds() if len(starts) == len(ends) == 16 else None,
        workers=receipts, receipt_binding="cache_manifest" if bound else "verified_content_and_report_hashes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    a, d = (load_arm(plan["output_root"], arm, plan["training"]) for arm in ("mlm", "pair_kd"))
    if a["provenance"]["plan_sha256"] != sha(args.plan):
        raise ValueError("Report plan differs from the training plan")
    if d["provenance"]["cache_manifest_sha256"] != sha(plan["cache_manifest"]):
        raise ValueError("Pair/KD cache differs from report cache")
    result = compare(a, d)
    result["initialization"] = plan.get("initialization")
    result["training_budget"] = plan["training"]
    result["timing"] = {arm: {key: row["complete"].get(key) for key in
        ("wall_seconds", "io_seconds", "update_seconds", "validation_seconds", "checkpoint_seconds",
         "student_gpu_hours", "student_with_qualification_gpu_hours")}
        for arm, row in (("mlm", a), ("pair_kd", d))}
    teacher = teacher_timing(plan)
    result["timing"]["teacher"] = teacher
    result["timing"]["scope"] = ("Recorded GPU-worker compute-stage hours, not platform billing: teacher includes model loading; "
        "student includes completed profile/replay qualification and training/IO/validation/checkpoint phases. "
        "Queueing, node startup, launcher CPU preflight, process teardown and unrecorded failed qualification attempts are excluded.")
    result["timing"]["pair_kd_end_to_end_gpu_hours"] = teacher["gpu_hours"]+d["complete"]["student_with_qualification_gpu_hours"]
    result["timing"]["pair_kd_end_to_end_interpretation"] = "Sum of recorded teacher and student compute-stage GPU-hour estimates; not allocated-resource billing"
    for arm, row in (("mlm", a), ("pair_kd", d)):
        result["timing"][arm].update(timing_complete=row["complete"].get("timing_complete", False),
                                    timing_note=row["complete"].get("timing_note"))
    result["timing"]["end_to_end_cost_status"] = ("measured" if teacher["status"] == "measured"
        and d["complete"].get("timing_complete", False) else "lower_bound_after_resume_or_unclean_exit")
    output = args.output or Path(plan["output_root"]).parent/"report"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"comparison.json", result)
    render_plot(a, d, output)
    print(json.dumps(dict(status="complete", report=str(output/"comparison.json"), conclusion=result["conclusion"])))


if __name__ == "__main__":
    main()
