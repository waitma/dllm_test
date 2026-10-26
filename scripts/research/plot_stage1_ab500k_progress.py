"""Plot observed AB500K training and validation losses without touching training.

Activate pllm, then run:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/plot_stage1_ab500k_progress.py --plan ABS_PLAN --output ABS_NEW_OUTPUT_DIRECTORY
Produces standalone PNG/PDF figures, raw CSV tables, and a frozen JSON snapshot.
Both completed and in-progress arms are supported; missing steps are not imputed.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np


COLORS = {"mlm": "#2563eb", "pair_kd": "#ea580c"}
LABELS = {"mlm": "MLM", "pair_kd": "Pair + KD"}


def read_arm(root, arm):
    directory = root / arm
    payload = (directory / "metrics.jsonl").read_bytes()
    # A writer may still be appending its last line. Only consume complete lines.
    lines = payload.splitlines(keepends=True)
    dropped_partial_line = bool(lines and not lines[-1].endswith(b"\n"))
    if dropped_partial_line:
        lines = lines[:-1]
    rows = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if row["arm"] != arm:
            raise ValueError("Metrics belong to another arm")
        if not all(math.isfinite(row[key]) for key in ("mlm", "objective", "intra_kl")):
            raise ValueError("Non-finite loss in training log")
        # A restart can replay uncommitted updates: keep the newest trajectory.
        while rows and rows[-1]["step"] >= row["step"]:
            rows.pop()
        rows.append(row)
    if not rows:
        raise ValueError(f"No complete training metrics for {arm}")
    validation = sorted(
        (json.loads(path.read_text()) for path in directory.glob("validation_step*.json")),
        key=lambda row: row["step"],
    )
    validation = [row for row in validation if row["step"] <= rows[-1]["step"]]
    for row in validation:
        if row["records"] != 4096 or row["masks"] != 3:
            raise ValueError("Unexpected validation population")
        if not all(math.isfinite(row[key]) for key in ("aa_ce", "full_token_ce")):
            raise ValueError("Non-finite validation loss")
    provenance = json.loads((directory / "run_provenance.json").read_text())
    complete = directory / "COMPLETE.json"
    completion = json.loads(complete.read_text()) if complete.exists() else None
    return dict(
        training=rows, validation=validation, provenance=provenance,
        completion=completion, last_step=rows[-1]["step"],
        metrics_path=str(directory / "metrics.jsonl"),
        metrics_snapshot_sha256=hashlib.sha256(payload).hexdigest(),
        ignored_incomplete_last_line=dropped_partial_line,
    )


def trailing_mean(values, window=5):
    return [float(np.mean(values[max(0, i-window+1):i+1])) for i in range(len(values))]


def style_axis(axis, title, ylabel):
    axis.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
    axis.grid(axis="y", alpha=.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)


def train_line(axis, rows, key, color, label, linestyle="-"):
    steps = [row["step"] for row in rows]
    values = [row[key] for row in rows]
    axis.plot(steps, values, color=color, alpha=.24, linewidth=1)
    axis.plot(steps, trailing_mean(values), color=color, linewidth=2,
              linestyle=linestyle, label=label)


def save_figure(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def render(arms, output, timestamp):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                         "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    shared = min(data["last_step"] for data in arms.values())
    maximum = max(data["last_step"] for data in arms.values())
    subtitle = (f"Snapshot {timestamp} UTC  |  MLM: {arms['mlm']['last_step']}/3000"
                f"  |  Pair + KD: {arms['pair_kd']['last_step']}/3000")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.suptitle("AB500K | Training MLM loss", fontsize=17, y=.97)
    fig.text(.5, .905, subtitle, ha="center", fontsize=10, color="#475569")
    for axis, limit, title in zip(axes, (maximum, shared),
                                 ("A  Full observed range", "B  Same-step comparison")):
        for arm, data in arms.items():
            rows = [row for row in data["training"] if row["step"] <= limit]
            train_line(axis, rows, "mlm", COLORS[arm], LABELS[arm],
                       "--" if arm == "pair_kd" else "-")
        style_axis(axis, title, "Training MLM cross-entropy")
        axis.set_xlim(0, limit)
        axis.legend(frameon=False, loc="best")
    if shared < maximum:
        axes[0].axvline(shared, color="#64748b", linewidth=1, linestyle=":")
    fig.text(.07, .055, "Faint: individual logged batches. Bold: trailing mean of 5 logged points (not every training batch).",
             fontsize=9, color="#475569")
    fig.text(.07, .02, "Both arms use the MLM loss here. Overlapping lines indicate similar values; unfinished steps are omitted.",
             fontsize=9, color="#475569")
    fig.subplots_adjust(left=.07, right=.98, bottom=.18, top=.80, wspace=.24)
    save_figure(fig, output, "training_loss_curves")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.suptitle("AB500K | Validation and Pair/KD loss components", fontsize=17, y=.98)
    fig.text(.5, .94, subtitle, ha="center", fontsize=10, color="#475569")
    for axis, field, title in ((axes[0, 0], "aa_ce", "A  Validation amino-acid CE"),
                               (axes[0, 1], "full_token_ce", "B  Validation full-token CE")):
        for arm, data in arms.items():
            values = data["validation"]
            axis.plot([row["step"] for row in values], [row[field] for row in values],
                      color=COLORS[arm], linestyle="--" if arm == "pair_kd" else "-",
                      marker="o", markersize=4, linewidth=1.8, label=LABELS[arm])
        axis.axhline(arms["mlm"]["validation"][0][field], color="#64748b", linestyle=":",
                     linewidth=1, label="Shared initial model")
        style_axis(axis, title, "Cross-entropy (zoomed y-axis)")
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
        axis.set_xlim(0, maximum)
        axis.legend(frameon=False, fontsize=9)
    pair = arms["pair_kd"]
    train_line(axes[1, 0], pair["training"], "objective", COLORS["pair_kd"], "Pair total objective")
    train_line(axes[1, 0], pair["training"], "mlm", COLORS["mlm"], "Pair MLM component", "--")
    style_axis(axes[1, 0], "C  Pair training objective", "Loss")
    axes[1, 0].legend(frameon=False, fontsize=9)
    train_line(axes[1, 1], pair["training"], "intra_kl", COLORS["pair_kd"], "Training KD (5-point mean)")
    values = pair["validation"]
    axes[1, 1].plot([row["step"] for row in values], [row["intra_kl"] for row in values],
                    color="#1e293b", marker="D", markersize=4, linestyle="--", label="Validation KD")
    style_axis(axes[1, 1], "D  Pair raw intra-chain KD", "Unweighted KL loss")
    axes[1, 1].legend(frameon=False, fontsize=9)
    for axis in axes[1]:
        axis.set_xlim(0, pair["last_step"])
    fig.text(.08, .05, "Validation: fixed 4096 records x 3 masks; values are unsmoothed. Training panels use the same 5-point mean.",
             fontsize=9, color="#475569")
    fig.text(.08, .022, "Pair objective = MLM + 0.05 x KD warmup scale x intra-chain KL. Compare arms with the common MLM/validation metrics.",
             fontsize=9, color="#475569")
    fig.subplots_adjust(left=.10, right=.98, bottom=.13, top=.86, hspace=.37, wspace=.28)
    save_figure(fig, output, "validation_and_kd_curves")


def write_csv(path, arms, kind, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", *columns], extrasaction="ignore")
        writer.writeheader()
        for arm, data in arms.items():
            for row in data[kind]:
                writer.writerow({**row, "arm": arm})


def paired_validation_diagnostics(arms):
    """One-checkpoint paired-record bootstrap; exploratory, not seed uncertainty."""
    av = {row["step"]: row for row in arms["mlm"]["validation"]}
    dv = {row["step"]: row for row in arms["pair_kd"]["validation"]}
    shared = sorted(av.keys() & dv.keys())
    comparisons = [dict(step=step, mlm_aa_ce=av[step]["aa_ce"], pair_aa_ce=dv[step]["aa_ce"],
                        improvement_percent=100*(av[step]["aa_ce"]-dv[step]["aa_ce"])/av[step]["aa_ce"])
                   for step in shared]
    arrays, identities = [], []
    for row in (av[shared[-1]], dv[shared[-1]]):
        reference = row["record_statistics"]
        raw = Path(reference["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("Validation statistics hash differs")
        with np.load(reference["path"]) as saved:
            identities.append(saved["row_ids"].copy())
            arrays.append(saved["statistics"].copy())
    if not np.array_equal(*identities) or not np.array_equal(arrays[0][:, [1, 3, 5]], arrays[1][:, [1, 3, 5]]):
        raise ValueError("Paired validation records/masks differ")
    a, d = arrays
    rng = np.random.default_rng(20261023)
    estimates = []
    for _ in range(2000):
        indices = rng.integers(0, len(a), len(a))
        aa, dd = a[indices].sum(0), d[indices].sum(0)
        a_loss, d_loss = aa[2]/aa[3], dd[2]/dd[3]
        estimates.append(100*(a_loss-d_loss)/a_loss)
    pair_values = arms["pair_kd"]["validation"]
    first, last = pair_values[0]["intra_kl"], pair_values[-1]["intra_kl"]
    return dict(shared_validation=comparisons, bootstrap_step=shared[-1],
                improvement_percent_ci95=np.quantile(estimates, [.025, .975]).tolist(),
                bootstrap_repeats=2000, bootstrap_seed=20261023,
                uncertainty_scope="Exploratory paired-record bootstrap; excludes homologous-cluster dependence, training-seed variance and checkpoint-selection uncertainty.",
                validation_kd_first=first, validation_kd_last=last,
                validation_kd_reduction_percent=100*(first-last)/first,
                total_objective_note="The KD coefficient increases during warmup; rising combined loss alone does not establish divergence.")


def render_explained_chinese(arms, output, timestamp, diagnostics):
    """Use a practical percent scale and separate task performance from KD fit."""
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font.exists():
        raise FileNotFoundError("Chinese plot font is required: " + str(font))
    font_manager.fontManager.addfont(str(font))
    bold = font.with_name("NotoSansCJK-Bold.ttc")
    if bold.exists():
        font_manager.fontManager.addfont(str(bold))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False,
                         "font.size": 11, "axes.titlesize": 13, "pdf.fonttype": 42})
    subtitle = (f"{timestamp} UTC 快照  |  纯 MLM：{arms['mlm']['last_step']}/3000 步"
                f"  |  Pair + KD：{arms['pair_kd']['last_step']}/3000 步")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.3))
    fig.suptitle("如何读这次实验：更像教师，不等于氨基酸预测更好", fontsize=19, y=.98)
    fig.text(.5, .938, subtitle, ha="center", fontsize=10, color="#475569")

    axis = axes[0, 0]
    values = diagnostics["shared_validation"]
    axis.axhspan(0, 1.15, color="#dcfce7", alpha=.45)
    axis.axhspan(-1.15, 0, color="#fee2e2", alpha=.45)
    axis.axhline(0, color="#475569", linewidth=1)
    axis.axhline(1, color="#15803d", linestyle=":", linewidth=1, label="1% 改善参考线")
    axis.plot([v["step"] for v in values], [v["improvement_percent"] for v in values],
              marker="o", color=COLORS["pair_kd"], linewidth=2, label="同一步：Pair 相对 MLM")
    last = values[-1]
    lo, hi = diagnostics["improvement_percent_ci95"]
    axis.vlines(last["step"], lo, hi, color="#7c2d12", linewidth=2)
    axis.set(title="① 相同训练步数：氨基酸预测是否改善", xlabel="训练步数", ylabel="验证 AA loss 相对改善（%）",
             ylim=(-1.15, 1.15), xlim=(0, max(250, last["step"])*1.06))
    axis.text(.04, .08, f"最新共同验证：step {last['step']}\n改善 {last['improvement_percent']:+.6f}%\n正数更好，负数更差", transform=axis.transAxes,
              fontsize=10, bbox=dict(boxstyle="round,pad=.5", facecolor="white", alpha=.92, edgecolor="#cbd5e1"))
    axis.legend(frameon=False, loc="upper right", fontsize=9)

    axis = axes[0, 1]
    rows = arms["pair_kd"]["validation"]
    axis.plot([r["step"] for r in rows], [r["intra_kl"] for r in rows], color=COLORS["pair_kd"], marker="o", linewidth=2)
    axis.set(title="② 验证 KD：与教师距离分布的差异", xlabel="训练步数", ylabel="原始 KD / KL（越低越好）",
             ylim=(0, max(r["intra_kl"] for r in rows)*1.18))
    reduction = diagnostics["validation_kd_reduction_percent"]
    axis.text(.48, .78, f"{rows[0]['intra_kl']:.3f} → {rows[-1]['intra_kl']:.3f}\n下降 {reduction:.1f}%", transform=axis.transAxes,
              fontsize=13, color="#9a3412")
    axis.text(.05, .08, "这表示更像教师，\n尚不等于结构真值更准或 MLM 更好。", transform=axis.transAxes, fontsize=10)

    axis = axes[1, 0]
    rows = arms["pair_kd"]["training"]
    x = [r["step"] for r in rows]
    mlm = trailing_mean([r["mlm"] for r in rows])
    kd = trailing_mean([r["objective"]-r["mlm"] for r in rows])
    axis.stackplot(x, mlm, kd, colors=["#93c5fd", "#fdba74"], labels=["MLM 部分", "加权 KD 部分"], alpha=.85)
    axis.plot(x, np.asarray(mlm)+np.asarray(kd), color="#9a3412", linewidth=1.8, label="联合总 loss")
    axis.set(title="③ 总 loss 的上升：KD 权重也在逐渐增加", xlabel="训练步数", ylabel="Pair 训练目标（5 个日志点平滑）",
             ylim=(0, max(np.asarray(mlm)+np.asarray(kd))*1.38), xlim=(0, rows[-1]["step"]))
    axis.text(.04, .83, "总 loss = MLM + 0.05 × warmup 系数 × KD\n它不能直接和纯 MLM loss 比大小。", transform=axis.transAxes, fontsize=10)
    axis.legend(frameon=False, fontsize=9, loc="lower right")

    axis = axes[1, 1]
    for y, arm in enumerate(("pair_kd", "mlm")):
        axis.barh(y, 3000, color="#e2e8f0", height=.42)
        axis.barh(y, arms[arm]["last_step"], color=COLORS[arm], height=.42)
        axis.text(arms[arm]["last_step"]+45, y, f"{arms[arm]['last_step']} 步", va="center", fontsize=11)
    axis.set(title="④ 训练进度：比较两组共同完成的 step", xlabel="已完成训练步数", yticks=[0, 1],
             yticklabels=["Pair + KD", "纯 MLM"], xlim=(0, 3550), ylim=(-.65, 1.65))
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
    fig.text(.07, .047, "验证均使用固定 4096 条样本 × 3 组 MASK；① 的改善 = (MLM loss − Pair loss) / MLM loss。", fontsize=9, color="#475569")
    fig.text(.07, .021, "① 小竖线为最新共同点的探索性记录 bootstrap 区间；不包含训练随机种子差异，稳健结论仍需独立重复。", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.075, right=.975, bottom=.12, top=.865, hspace=.37, wspace=.25)
    save_figure(fig, output, "loss_explained_zh")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("两组 MLM 训练 loss：曲线重合表示数值接近", fontsize=18, y=.97)
    fig.text(.5, .90, subtitle, ha="center", fontsize=10, color="#475569")
    limits = [max(d["last_step"] for d in arms.values()), min(d["last_step"] for d in arms.values())]
    for axis, limit, title in zip(axes, limits, ["全部已记录进度", "共同训练步数：公平对照"]):
        for arm, data in arms.items():
            rows = [row for row in data["training"] if row["step"] <= limit]
            train_line(axis, rows, "mlm", COLORS[arm], "纯 MLM" if arm == "mlm" else "Pair 的 MLM 部分", "--" if arm == "pair_kd" else "-")
        style_axis(axis, title, "MLM loss（越低越好）")
        axis.set_xlabel("训练步数")
        axis.set_xlim(0, limit)
        axis.legend(frameon=False, fontsize=10)
    fig.text(.07, .045, "浅线：已记录的单批 loss。粗线：5 个日志点的后向移动平均，不是每个训练批次的均值。", fontsize=10, color="#475569")
    fig.subplots_adjust(left=.075, right=.98, bottom=.18, top=.79, wspace=.24)
    save_figure(fig, output, "training_comparison_zh")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.plan.is_absolute() or not args.output.is_absolute():
        raise ValueError("Use absolute paths")
    plan = json.loads(args.plan.read_text())
    arms = {arm: read_arm(Path(plan["output_root"]), arm) for arm in ("mlm", "pair_kd")}
    for key in ("plan_sha256", "subset_manifest_sha256", "parent_model_sha256", "training",
                "noise_recipe", "noise_parent_records", "recipe", "validation_seeds", "world_size"):
        if arms["mlm"]["provenance"][key] != arms["pair_kd"]["provenance"][key]:
            raise ValueError(f"Comparison contract differs: {key}")
    plan_hash = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    if arms["mlm"]["provenance"]["plan_sha256"] != plan_hash:
        raise ValueError("Plot plan differs from training plan")
    av = {row["step"]: row for row in arms["mlm"]["validation"]}
    dv = {row["step"]: row for row in arms["pair_kd"]["validation"]}
    shared_steps = sorted(av.keys() & dv.keys())
    for step in shared_steps:
        for key in ("records", "masks", "seeds", "selected_tokens", "aa_tokens"):
            if av[step][key] != dv[step][key]:
                raise ValueError(f"Validation masks/population differ at step {step}: {key}")
    now = datetime.now(timezone.utc)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = dict(observed_utc=now.isoformat(), plan=str(args.plan), plan_sha256=plan_hash,
                    arms=arms, shared_validation_steps=shared_steps,
                    smoothing="trailing mean of five logged batches; raw data also displayed and exported",
                    scope="Observed progress only; not the final completed-arm experiment comparison")
    (args.output / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+"\n")
    write_csv(args.output / "training_losses.csv", arms, "training",
              ["step", "mlm", "objective", "intra_kl", "kd_scale", "backbone_lr", "grad_norm", "seen_records", "wall_seconds"])
    write_csv(args.output / "validation_losses.csv", arms, "validation",
              ["step", "full_token_ce", "aa_ce", "mask_accuracy", "intra_kl", "records", "masks", "wall_seconds"])
    render(arms, args.output, now.strftime("%Y-%m-%d %H:%M:%S"))
    diagnostics = paired_validation_diagnostics(arms)
    (args.output / "interpretation.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2)+"\n")
    render_explained_chinese(arms, args.output, now.strftime("%Y-%m-%d %H:%M:%S"), diagnostics)
    latest_shared = shared_steps[-1]
    summary = dict(output=str(args.output), observed_utc=now.isoformat(),
                   last_step={arm: data["last_step"] for arm, data in arms.items()},
                   completion={arm: data["completion"] for arm, data in arms.items()},
                   latest_shared_validation={"step": latest_shared, "mlm": av[latest_shared]["aa_ce"], "pair_kd": dv[latest_shared]["aa_ce"]},
                   last_validation={arm: data["validation"][-1] for arm, data in arms.items()})
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("completion", "last_validation")}))


if __name__ == "__main__":
    main()
