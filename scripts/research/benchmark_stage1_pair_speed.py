"""Measure AB500K micro-updates on one GPU without saving training weights.

Activate pllm; CUDA_VISIBLE_DEVICES=0 USE_TF=0 IMMUNE_SDPA_BACKEND=efficient
python -B THIS_FILE --plan ABS_PLAN --output ABS_NEW_DIR --variants mlm baseline
Inputs come from the frozen plan; optimization variants affect only RAM objects.
Timings exclude dataset loading/batch assembly/DDP; assembly is measured separately.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["mlm", "baseline", "kd_only", "feedback_only",
        "chunk64", "chunk128", "chunk310", "no_pair_checkpoint", "no_checkpoints"])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--mask-seed", type=int, default=10042)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not args.plan.is_absolute() or not args.output.is_absolute():
        raise ValueError("Use absolute paths")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"benchmark_source.py").write_bytes(Path(__file__).read_bytes())
    candidates_path = Path(__file__).with_name("stage1_pair_speed_candidates.py")
    if candidates_path.exists():
        (args.output/"candidates_source.py").write_bytes(candidates_path.read_bytes())
        candidate_spec = importlib.util.spec_from_file_location("speed_candidates", candidates_path)
        candidates = importlib.util.module_from_spec(candidate_spec)
        candidate_spec.loader.exec_module(candidates)
    plan = json.loads(args.plan.read_text())
    sys.path.insert(0, plan["source_root"])
    import numpy as np
    import torch
    from scripts.research import run_stage1_ab500k as training
    torch.set_num_threads(1)
    os.environ.setdefault("IMMUNE_SDPA_BACKEND", "efficient")
    device = torch.device("cuda:0")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one idle GPU for this diagnostic")
    for path, digest in plan["sources"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen source changed: " + path)
    subset = json.loads(Path(plan["subset_manifest"]).read_text())
    dataset = training.PilotDataset(subset, plan["cache_manifest"], "train")
    ordinals = np.random.default_rng(args.seed).choice(len(dataset), args.batch, replace=False).tolist()
    started = time.perf_counter()
    rows = [dataset[index] for index in ordinals]
    load_seconds = time.perf_counter()-started
    environment = dict(observed_utc=datetime.now(timezone.utc).isoformat(), gpu=torch.cuda.get_device_name(),
        gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30, torch=torch.__version__,
        cuda=torch.version.cuda, plan=str(args.plan), plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        batch=args.batch, precision="bf16", ordinals=ordinals, source_root=plan["source_root"],
        data_load_seconds=load_seconds, warmup=args.warmup, repeats=args.repeats,
        data_seed=args.seed, mask_seed=args.mask_seed,
        scope="Single-GPU in-memory micro-update; includes H2D, loss, backward, clipping, AdamW; excludes DDP, loader and batch assembly; no saved weights")
    write(args.output/"environment.json", environment)
    results = []
    for variant in args.variants:
        model = optimizer = scheduler = inputs = None
        try:
            arm = "mlm" if variant == "mlm" else "pair_kd"
            model = training.new_model(plan, arm, device)
            changes = {}
            if variant.startswith("chunk"):
                changes["query_chunk"] = int(variant.removeprefix("chunk"))
            elif variant in ("no_pair_checkpoint", "no_checkpoints"):
                changes.update(query_chunk=64, checkpoint_pair_chunks=False)
            elif variant == "kd_only":
                changes["feedback"] = "none"
            elif variant.startswith("symmetric"):
                candidates.install_symmetric(model)
                if "cast" in variant:
                    candidates.install_cast(model)
                if "compiled" in variant:
                    candidates.install_compile(model, precise="precise" in variant)
                if "no_pair_checkpoint" in variant:
                    changes["checkpoint_pair_chunks"] = False
            elif variant in ("cast", "cast_no_pair_checkpoint", "compiled", "compiled_cast", "compiled_no_pair_checkpoint"):
                if "no_pair_checkpoint" in variant:
                    changes.update(query_chunk=64, checkpoint_pair_chunks=False)
                if variant.startswith("compiled"):
                    candidates.install_compile(model, cast=variant == "compiled_cast")
                else:
                    candidates.install_cast(model)
            elif variant not in ("mlm", "baseline", "feedback_only"):
                raise ValueError("Unknown variant: " + variant)
            if changes:
                model.pair_config = replace(model.pair_config, **changes)
                model.pair_encoder.config = model.pair_config
            if variant == "no_checkpoints":
                model.backbone.set_activation_checkpointing(None)
            settings = dict(plan["training"])
            if variant == "feedback_only":
                settings["intra_weight"] = 0.
            optimizer = training.new_optimizer(model, settings, arm)
            scheduler = training.scheduler_for(optimizer, settings)
            assembly = []
            for _ in range(3):
                start = time.perf_counter()
                inputs = training.build_micros(model, rows, args.batch, args.mask_seed, arm)
                assembly.append(time.perf_counter()-start)
            if variant == "feedback_only":
                inputs = [(batch, pairs[:0], teacher[:0]) for batch, pairs, teacher in inputs]
            def update():
                # Fixed post-warmup KD scale; only disposable in-memory parameters change.
                return training.optimizer_update(model, optimizer, scheduler, inputs, settings, arm, 1000)
            warmup_start = time.perf_counter()
            for _ in range(args.warmup):
                update()
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter()-warmup_start
            torch.cuda.reset_peak_memory_stats()
            elapsed = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                scores = update()
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-start)
            result = dict(variant=variant, status="passed", seconds=elapsed,
                median_seconds=statistics.median(elapsed), min_seconds=min(elapsed), max_seconds=max(elapsed),
                warmup_seconds=warmup_seconds,
                assembly_median_seconds=statistics.median(assembly),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                sequence_length=inputs[0][0]["input_ids"].shape[1], pair_targets=len(inputs[0][1]),
                settings=changes, final_probe_scores=scores,
                interpretation="architecture ablation, not an equivalent optimization" if variant in
                    ("kd_only", "feedback_only") else "same mathematical model; numerical parity requires separate verification")
            if args.profile and variant != "mlm":
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    update()
                    torch.cuda.synchronize()
                events = profiler.key_averages()
                (args.output/(variant+"_cuda_operators.txt")).write_text(events.table(sort_by="self_cuda_time_total", row_limit=35))
                (args.output/(variant+"_cpu_operators.txt")).write_text(events.table(sort_by="self_cpu_time_total", row_limit=35))
                profiler.export_chrome_trace(str(args.output/(variant+"_trace.json.gz")))
        except torch.OutOfMemoryError as error:
            result = dict(variant=variant, status="oom", error=str(error)[:1200])
        except Exception as error:
            result = dict(variant=variant, status="failed", error=repr(error))
        results.append(result)
        write(args.output/(variant+".json"), result)
        print(json.dumps(result), flush=True)
        model = optimizer = scheduler = inputs = None
        gc.collect()
        torch.cuda.empty_cache()
    write(args.output/"results.json", dict(environment=environment, results=results))


if __name__ == "__main__":
    main()
