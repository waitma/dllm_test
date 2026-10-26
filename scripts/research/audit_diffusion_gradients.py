"""Audit logged diffusion gradients, validation drift and source exposure.

Activate pllm; python -B -m scripts.research.audit_diffusion_gradients --output ABS.
Reads existing logs only; emits JSON, CSV and standalone PNG/PDF figures.
Logged gradient statistics cover sampled updates, not every optimizer step.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from scripts.research.bert_mlm_data import sha, write_json

ROOT = Path(__file__).resolve().parents[2]


def audit(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output.mkdir(parents=True, exist_ok=True)
    original = ROOT / "output/bert_diffusion_edit_v7_from287000_20261019/train"
    extended = ROOT / "output/bert_diffusion_edit_v7_from287000_200k_20261020/train"
    train, valid, duplicates = {}, {}, 0
    paths = [original / "metrics.jsonl", extended / "metrics.jsonl"]
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            step = row["step"]
            if "train/grad_norm" in row:
                duplicates += int(step in train)
                train[step] = row
            if "valid/loss" in row:
                valid[step] = row
    rows = [train[k] for k in sorted(train)]
    valid_rows = [valid[k] for k in sorted(valid)]
    for name, values in (("training", rows), ("validation", valid_rows)):
        with (output / (name + ".csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    windows = []
    bounds = [0, 1000, 2500, 5000, 10000, 11000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
    for low, high in zip(bounds[:-1], bounds[1:]):
        subset = [r for r in rows if low < r["step"] <= high]
        if not subset:
            continue
        gradients = [r["train/grad_norm"] for r in subset]
        windows.append(dict(start_exclusive=low, end_inclusive=high, logged_updates=len(subset),
            grad_median=statistics.median(gradients), grad_max=max(gradients),
            fraction_above_clip=statistics.mean(g > 1 for g in gradients),
            mean_train_loss=statistics.mean(r["train/loss"] for r in subset)))
    first = min(valid_rows, key=lambda r: r["valid/loss"])
    last = valid_rows[-1]
    plan_path = ROOT / "output/v5_full_binding_20261007/launch_diffusion_edit287000_extend200k_train_20261020/plan.json"
    plan = json.loads(plan_path.read_text())
    index_path = Path(plan["data_index"])
    index = json.loads(index_path.read_text())
    sizes = {}
    for shard in index["shards"]:
        if shard["split"] == "train":
            sizes[shard["source"]] = sizes.get(shard["source"], 0) + shard["records"]
    sources = {}
    for offset, (name, size) in enumerate(sizes.items()):
        key = f"valid/source/{name}/loss"
        count = (last["step"] * plan["training"]["global_batch"] + len(sizes) - 1 - offset) // len(sizes)
        sources[name] = dict(records=size, exposures=count, exposure_epochs=count / size,
            best_checkpoint_ce=first[key], latest_checkpoint_ce=last[key],
            delta_ce=last[key] - first[key], macro_delta_contribution=(last[key] - first[key]) / len(sizes))
    monitor = extended.parent / "nonspot_20261022/monitor_status.json"
    status = json.loads(monitor.read_text())
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Historical logged updates; no live platform status query; no new training",
        inputs={str(p): sha(p) for p in [*paths, plan_path, index_path, monitor]},
        logged_updates=len(rows), removed_replayed_updates=duplicates, validation_points=len(valid_rows),
        latest_logged_step=rows[-1]["step"], best_valid_step=first["step"], latest_valid_step=last["step"],
        best_valid_loss=first["valid/loss"], latest_valid_loss=last["valid/loss"],
        relative_validation_regression=last["valid/loss"] / first["valid/loss"] - 1,
        windows=windows, sources=sources,
        last_platform_snapshot={k: status[k] for k in ("updated_utc", "platform_status", "latest_complete_step")},
        interpretation="Gradients rise gradually but logged post-startup norms remain below 1; fixed validation regresses. Oversampling is a risk, not a proven causal attribution.")
    write_json(output / "summary.json", report)
    steps = np.array([r["step"] for r in rows])
    norms = np.array([r["train/grad_norm"] for r in rows])
    rolling = [statistics.median(norms[max(0, i-39):i+1]) for i in range(len(norms))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(steps, norms, s=3, alpha=.25, label="Logged pre-clip norm")
    ax.plot(steps, rolling, label="Trailing 40 logged updates: median")
    ax.axhline(1, color="red", ls="--", label="Clip threshold")
    ax.set(ylim=(0, 1.1), title="Gradient norm (startup peaks up to 4.19 omitted)")
    ax.legend(fontsize=8)
    axes[0, 1].plot(steps, [r["train/learning_rate"] for r in rows])
    axes[0, 1].set(title="Actual learning rate per logged update", ylabel="LR")
    ax = axes[1, 0]
    ax.plot([r["step"] for r in valid_rows], [r["valid/loss"] for r in valid_rows], label="Fixed-validation CE")
    ax.axvline(first["step"], color="grey", ls="--", label="Best: step 2500")
    ax.set(title="Validation CE (different objective from train loss)", ylim=(1.07, 1.32))
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    names = list(sources)
    ax.barh(names, [sources[n]["delta_ce"] for n in names])
    ax.set(title="Per-source validation change: 2500 to 76500", xlabel="CE increase")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Diffusion optimizer step")
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("gradient_audit." + suffix), dpi=170)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ("logged_updates", "removed_replayed_updates", "relative_validation_regression")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.output)
