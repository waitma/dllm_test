"""Activate pllm; python -B THIS_FILE --root ABS_DIAGNOSTICS --output ABS_NEW_DIR.

Summarize measured Pair speed candidates, retaining failed numerical gates.
Produces CSV/JSON plus a standalone PNG/PDF from the confirmation measurements.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"report_source.py").write_bytes(Path(__file__).read_bytes())
    bindings = {}

    def read(relative):
        path = args.root/relative
        raw = path.read_bytes()
        bindings[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    parity = {name: read(name+"/verification.json") for name in
              ("parity", "parity_precise", "parity_confirmation")}
    gates = {}
    for name, report in parity.items():
        for row in report["results"]:
            gates.setdefault(row["variant"], []).append(dict(batch=name, status=row["status"]))
    rows = []
    confirmations = {}
    for group in ("baseline", "candidates", "symmetry", "precise", "confirmation"):
        report = read(group+"/results.json")
        for result in report["results"]:
            variant = result["variant"]
            row = dict(group=group, variant=variant, status=result["status"],
                seconds=result.get("median_seconds"), assembly_seconds=result.get("assembly_median_seconds"),
                allocated_gib=result.get("peak_allocated_gib"), reserved_gib=result.get("peak_reserved_gib"),
                numerical_gate=";".join(g["batch"]+":"+g["status"] for g in gates.get(variant, [])))
            rows.append(row)
            if group == "confirmation":
                confirmations[variant] = result
    with (args.output/"measurements.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidates = ["symmetric_cast", "symmetric_cast_no_pair_checkpoint"]
    confirmed = {row["variant"]: row for row in parity["parity_confirmation"]["results"]}
    for name in candidates:
        assert confirmed[name]["status"] == "passed_bf16_tolerance"
        assert confirmations[name]["status"] == "passed"
    base = confirmations["baseline"]
    summary = dict(scope="Single GPU, B64/L310, no DDP; two real batches checked; no production deployment",
        evidence_sha256=bindings, measurements=rows, candidates={})
    for name in candidates:
        result = confirmations[name]
        summary["candidates"][name] = dict(
            update_speedup=base["median_seconds"]/result["median_seconds"],
            update_time_reduction=1-result["median_seconds"]/base["median_seconds"],
            assembly_plus_update_speedup=(base["median_seconds"]+base["assembly_median_seconds"])/
                (result["median_seconds"]+result["assembly_median_seconds"]),
            allocated_gib=result["peak_allocated_gib"],
            logits_rms=confirmed[name]["selected_logits_rms_error"],
            max_gradient_relative_l2=max(g["relative_l2_error"] for g in confirmed[name]["gradients"].values()))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 11})
    names = ["baseline", *candidates]
    labels = ["原 Pair＋KD", "对称＋提前 dtype 转换\n保留 Pair 重计算", "对称＋提前 dtype 转换\n关闭 Pair 重计算"]
    colors = ["#64748b", "#2563eb", "#059669"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, title, limit, unit in (
        (axes[0], "median_seconds", "更新耗时：越低越快", 3.0, "秒"),
        (axes[1], "peak_allocated_gib", "峰值 allocated 显存：越低越省", 65, "GiB")):
        values = [confirmations[name][key] for name in names]
        ax.barh(labels, values, color=colors, height=.55)
        ax.invert_yaxis()
        for index, value in enumerate(values):
            ax.text(value+limit*.02, index, f"{value:.2f} {unit}", va="center")
        ax.set_xlim(0, limit)
        ax.set_title(title, pad=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    axes[1].set_yticklabels([])
    fig.suptitle("Pair 单卡优化实测：约 1.72× / 2.05× 更新速度", fontsize=16)
    fig.text(.5, .035, "A100 80GB · 每卡64条 · 长度310 · 第二批真实数据 · 5次预热＋10次计时，中位数\n"
             "耗时含前后向和优化器，不含CPU组batch、读取、DDP、验证及保存；候选通过有限批次输出/梯度检查", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .94))
    for ext in ("png", "pdf"):
        fig.savefig(args.output/("pair_speed_comparison."+ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary["candidates"], indent=2))


if __name__ == "__main__":
    main()
