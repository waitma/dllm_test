"""Run current local T2/T3 tasks with the gated v5 runtime protocol.

In protenix_abtcr after pllm: python scripts/downstream/run_tcr_native_repr.py
--task preflight (first), then --task t2a|t2b|t3deep|t3broad.
Only the pinned TCR snapshot (default 49000; TCR_EVAL_* overrides) is accepted. Original CSVs are read-only.
T3 is explicitly a local control, not a paper-exact dataset reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from downstream.grammar.ab_features import file_sha256, write_json
from downstream.grammar.tcr_features import TCRRepresentationProtocol, TCRRuntimeEmbedder
from scripts.downstream.prepare_tcr_native_jobs import ROOT, CKPT, OUT, TAG
from scripts.downstream.tcr_eval_pin import PIN
from scripts.downstream.preflight_tcr_native import EXPECTED_SHA

BENCH = ROOT / "downstream/benchmark"
DATASETS = {
    "t2a": ("tcr_clustering", ["tcrs.csv"], False),
    "t2b": ("tcr_clustering_embed", ["tcrs.csv"], False),
    "t3deep": ("tcr_representation_paper6", ["target_binders.csv", "background_pool.csv"], True),
    "t3broad": ("tcr_representation", ["train.csv", "test.csv"], True),
}


def inputs(task):
    directory, names, paired = DATASETS[task]
    paths = [BENCH / "data" / directory / name for name in names]
    frames = [pd.read_csv(path).fillna("") for path in paths]
    for frame in frames:
        if "cdr3b" not in frame or (paired and "cdr3a" not in frame):
            raise ValueError("missing required receptor column")
    return frames, paths, paired


def identity(task):
    frames, paths, paired = inputs(task)
    protocol = TCRRepresentationProtocol(input_mode="cdr3ab" if paired else "cdr3b")
    scripts = [Path(__file__), BENCH / "common/fewshot.py", BENCH / "common/metrics.py",
        BENCH / "tcr_clustering/run.py", BENCH / "tcr_clustering/run_embed_bench.py",
        BENCH / "tcr_representation/run_paper6.py"]
    return {"checkpoint_sha256": PIN.sha, "global_step": PIN.step,
        "input_protocol_sha256": protocol.sha256, "input_protocol": protocol.provenance,
        "data_sha256": {str(p): file_sha256(p) for p in paths},
        "code_sha256": {str(p): file_sha256(p) for p in scripts},
        "rows": [len(f) for f in frames],
        "missing_alpha_rows": [int(f.cdr3a.eq("").sum()) if paired else None for f in frames],
        "interpretation": "local_control_not_paper_exact" if task.startswith("t3") else "local_harness",
        "pretraining_overlap": "not_fully_cleared_do_not_claim_contamination_free"}


def assert_checkpoint():
    if file_sha256(CKPT / "model.safetensors") != PIN.sha:
        raise ValueError("not the approved checkpoint")
    if json.loads((CKPT / "trainer_state.json").read_text())["global_step"] != PIN.step:
        raise ValueError(f"not step {PIN.step}")


def receptor_groups(frame):
    """One receptor identity, all known positive labels; no majority collapse."""
    grouped = frame.groupby(["cdr3b", "cdr3a"], sort=False, dropna=False)
    return grouped.peptide.agg(lambda x: tuple(sorted(set(x)))).reset_index()


def multilabel_fewshot(dist, train, test, shots=(1, 2, 5, 10, 20, 50, 100), seeds=5):
    from sklearn.metrics import roc_auc_score
    epitopes = sorted(set(e for es in train.peptide for e in es) &
                      set(e for es in test.peptide for e in es))
    result = {"protocol": "local_broad_unique_pair_multilabel_nn_v1", "shots": list(shots),
        "seeds": seeds, "agg": "min", "auroc_by_shot": {}, "per_epitope_by_shot": {},
        "n_epitopes_by_shot": {}, "episodes": []}
    for k in shots:
        per_ep = {}
        for ep in epitopes:
            pool = np.array([i for i, labels in enumerate(train.peptide) if ep in labels])
            y = np.array([ep in labels for labels in test.peptide], dtype=int)
            if len(pool) < k or len(set(y)) < 2:
                continue
            scores = []
            for seed in range(seeds):
                selected = np.random.default_rng(seed + 1000 * k).choice(pool, k, replace=False)
                scores.append(roc_auc_score(y, -dist[:, selected].min(axis=1)))
                result["episodes"].append({"k": k, "epitope": ep, "seed": seed,
                    "support_indices": selected.tolist(), "auroc": float(scores[-1])})
            per_ep[ep] = float(np.mean(scores))
        vals = list(per_ep.values())
        result["per_epitope_by_shot"][k] = per_ep
        result["n_epitopes_by_shot"][k] = len(vals)
        result["auroc_by_shot"][k] = {"mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None}
    return result


def broad(output):
    from downstream.benchmark.common.fewshot import cosine_distance_matrix
    frames, _, _ = inputs("t3broad")
    train, test = [receptor_groups(frame) for frame in frames]
    keys = lambda f: set(zip(f.cdr3b, f.cdr3a))
    if keys(train) & keys(test):
        raise ValueError("support/query receptor overlap")
    emb = TCRRuntimeEmbedder(str(CKPT), device="cuda", batch_size=8)
    xtr, xte = [emb.embed_pairs(f.cdr3b.tolist(), f.cdr3a.tolist()) for f in (train, test)]
    result = multilabel_fewshot(cosine_distance_matrix(xte, xtr), train, test)
    result.update(n_train=len(train), n_test=len(test),
        multilabel_pairs=[int(f.peptide.map(len).gt(1).sum()) for f in (train, test)])
    write_json(output / "fewshot.json", result)
    # Retain the built auxiliary row-annotation probe, clearly separate from
    # multi-label few-shot. It cannot be interpreted as exclusive specificity.
    from downstream.benchmark.common import metrics
    x_by_split = []
    for original, grouped, vectors in zip(frames, (train, test), (xtr, xte)):
        lut = dict(zip(zip(grouped.cdr3b, grouped.cdr3a), vectors))
        x_by_split.append(np.stack([lut[key] for key in zip(original.cdr3b, original.cdr3a)]))
    classes = sorted(set(frames[0].peptide) | set(frames[1].peptide))
    cls2id = {c: i for i, c in enumerate(classes)}
    ytr, yte = [f.peptide.map(cls2id).to_numpy() for f in frames]
    probe = metrics.linear_probe_metrics(x_by_split[0], ytr, x_by_split[1], yte, multiclass=True, seed=0)
    probe.update(metrics.knn_top1_metric(x_by_split[0], ytr, x_by_split[1], yte, metric="cosine"))
    probe["interpretation"] = "auxiliary_row_annotation_diagnostic_not_exclusive_binding"
    write_json(output / "probe_diagnostic.json", probe)


def preflight(target=None):
    target = target or OUT / "repr_preflight_v2/passed.json"
    if target.exists():
        raise FileExistsError(target)
    assert_checkpoint()
    emb = TCRRuntimeEmbedder(str(CKPT), device="cuda", batch_size=8)
    reports = {}
    for task in DATASETS:
        frames, _, paired = inputs(task)
        frame = pd.concat(frames, ignore_index=True)
        keys = list(dict.fromkeys(zip(frame.cdr3b, frame.cdr3a if paired else [None] * len(frame))))
        p = emb.protocols["cdr3ab" if paired else "cdr3b"]
        lengths = [sum(len(c.sequence) for c in p.record(*key).chains) + 8 for key in keys]
        selected = list(dict.fromkeys([keys[int(np.argmax(lengths))]] + keys[:15]))
        if paired and frame.cdr3a.eq("").any():
            selected += [key for key in keys if key[1] == ""]
        def features(rows, bs):
            # Fresh cache every time: batch/order test must perform new forwards.
            wrap = TCRRuntimeEmbedder(backbone=emb.backbone, batch_size=bs)
            return wrap.embed_pairs([b for b, _ in rows], [a for _, a in rows] if paired else None)
        a, b, rev = features(selected, 8), features(selected, 1), features(selected[::-1], 8)[::-1]
        unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        delta = float(np.max(np.abs(unit(a) - unit(b))))
        reverse = float(np.max(np.abs(unit(a) - unit(rev))))
        cosine = float(np.min((unit(a) * unit(b)).sum(1)))
        if not np.isfinite(a).all() or max(delta, reverse) > .005 or cosine < .99995:
            raise ValueError(f"{task} GPU batch/order gate failed: {delta=} {reverse=} {cosine=}")
        reports[task] = {"identity": identity(task), "max_input_length": max(lengths),
            "unique_inputs": len(keys), "normalized_max_delta": delta,
            "reverse_normalized_max_delta": reverse, "min_cosine": cosine}
        print(f"PASS {task}: n={len(keys)}, {delta=}", flush=True)
    write_json(target, {"status": "passed", "tasks": reports, "quality_evaluation": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["preflight", *DATASETS], required=True)
    p.add_argument("--gate", type=Path, default=OUT / "repr_preflight_v2/passed.json")
    args = p.parse_args()
    if args.task == "preflight":
        preflight(args.gate)
        return
    assert_checkpoint()
    gate = json.loads(args.gate.read_text())
    ident = identity(args.task)
    if gate["status"] != "passed" or ident != gate["tasks"][args.task]["identity"]:
        raise ValueError("source/data/protocol changed after GPU gate")
    directory = DATASETS[args.task][0]
    tag = "ours_" + TAG + "_runtime_v2"
    output = BENCH / "outputs" / directory / tag
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "task": args.task, "identity": ident,
        "checkpoint": str(CKPT), "output": str(output)}
    write_json(output / "run_manifest.json", manifest)
    try:
        spec = "tcr-v5:" + str(CKPT)
        commands = {
            "t2a": ["tcr_clustering/run.py", "--method", "embed-threshold"],
            "t2b": ["tcr_clustering/run_embed_bench.py", "--algos", "kmeans"],
            "t3deep": ["tcr_representation/run_paper6.py", "--method", "embed", "--columns", "cdr3b", "cdr3a"],
        }
        if args.task == "t3broad":
            broad(output)
        else:
            command = commands[args.task]
            subprocess.run([sys.executable, str(BENCH / command[0]), *command[1:],
                "--embedder", spec, "--tag", tag], check=True)
        manifest["status"] = "success"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
