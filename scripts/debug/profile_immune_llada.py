"""Bounded, offline, single-GPU profiling; never submits jobs or saves checkpoints.

Run from the repository with the protenix_abtcr Python::

    python -B scripts/debug/profile_immune_llada.py --check-gpu
    python -B scripts/debug/profile_immune_llada.py --run --background \
        --io-clear-file refactor_baseline/local_profile_20260909/IO_CLEAR

Only AFTER coordinating with the full-parity job, create IO_CLEAR containing
``main-agent-cleared``. The runner does not create this permission itself. It
requires three idle GPU observations, then calls the real training entry's
model construction with a profiling-only Trainer override. --smoke reduces the
step counts, but is explicitly NOT a benchmark. All writes stay in the artifact
root, including subprocess logs, PID/status, and library caches. The supervisor
has a hard wall-clock deadline and only terminates its own child's process group.
"""

from __future__ import annotations

import argparse

import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any
from unittest.mock import patch
from uuid import UUID


class ProfileDeadline(TimeoutError):
    """Raised by the supervisor's wall-clock alarm, including during GPU queries."""

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "refactor_baseline/local_profile_20260909"
PYTHON = "/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python"
SNAPSHOT = "/vepfs-mlp2/c20250601/251105016/conda/cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
SOURCES = ("oas", "ots", "asd_antibody", "trait", "tcr_native", "tcr_papers", "tcr_repertoire")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def gpu_snapshot() -> dict[str, Any]:
    """No torch/CUDA initialization; unknown or failed queries fail closed."""
    try:
        def query(fields, kind):
            return subprocess.run(
                ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        rows = []
        for line in query("index,uuid,name,memory.used,memory.total,utilization.gpu", "gpu").splitlines():
            index, uuid, name, memory, total, util = [x.strip() for x in line.split(",")]
            rows.append(dict(index=int(index), uuid=uuid, name=name,
                             memory_used_mib=float(memory), memory_total_mib=float(total),
                             utilization_pct=float(util)))
        apps = []
        for line in query("gpu_uuid,pid,process_name,used_memory", "compute-apps").splitlines():
            if line.strip():
                uuid, pid, name, memory = [x.strip() for x in line.split(",", 3)]
                apps.append(dict(uuid=uuid, pid=int(pid), name=name, memory_mib=memory))
        return dict(timestamp=time.time(), gpus=rows, compute_apps=apps, error=None)
    except ProfileDeadline:
        raise
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        return dict(timestamp=time.time(), gpus=[], compute_apps=[], error=str(exc))


def gpu_card(snapshot, gpu_index=None, *, gpu_uuid=None):
    """Resolve an NVML card, rejecting missing/ambiguous identities before CUDA use."""
    if snapshot.get("error"):
        raise RuntimeError(snapshot["error"])
    key, value = ("uuid", gpu_uuid) if gpu_uuid is not None else ("index", gpu_index)
    cards = [x for x in snapshot["gpus"] if x.get(key) == value]
    if len(cards) != 1:
        raise RuntimeError(f"GPU {key}={value} not uniquely visible")
    card = cards[0]
    uuid = card.get("uuid", "")
    try:
        valid = uuid.startswith("GPU-") and str(UUID(uuid[4:])) == uuid[4:].lower()
    except (ValueError, AttributeError):
        valid = False
    if not valid or sum(x.get("uuid") == uuid for x in snapshot["gpus"]) != 1:
        raise RuntimeError(f"GPU has missing/invalid/ambiguous UUID: {uuid!r}")
    return card


def idle_reasons(snapshot, gpu_index=None, *, gpu_uuid=None):
    try:
        card = gpu_card(snapshot, gpu_index, gpu_uuid=gpu_uuid)
    except RuntimeError as exc:
        return [str(exc)]
    reasons = []
    apps = [x for x in snapshot["compute_apps"] if x["uuid"] == card["uuid"]]
    if apps:
        reasons.append(f"foreign GPU compute processes: {apps}")
    if not all(math.isfinite(card[key]) for key in ("memory_used_mib", "utilization_pct")):
        return ["GPU memory/utilization telemetry is nonfinite"]
    if card["memory_used_mib"] > 512:
        reasons.append(f"memory already used: {card['memory_used_mib']} MiB")
    if card["utilization_pct"] > 5:
        reasons.append(f"GPU utilization: {card['utilization_pct']}%")
    return reasons


def child_gate_reasons(snapshot, cfg):
    uuid = cfg.get("gpu_uuid")
    if not uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        return ["CUDA visibility must match the supervisor's verified GPU UUID"]
    return idle_reasons(snapshot, gpu_uuid=uuid)


def io_is_clear(path):
    try:
        return Path(path).read_text().strip() == "main-agent-cleared"
    except OSError:
        return False


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0)
    def percentile(q):
        pos = (len(values) - 1) * q
        low = math.floor(pos)
        high = math.ceil(pos)
        return values[low] + (values[high] - values[low]) * (pos - low)
    return dict(count=len(values), mean=statistics.mean(values), min=values[0],
                p50=percentile(.5), p95=percentile(.95), max=values[-1])


def select_samples(parts, sources, seed, candidates_per_source, samples_per_source):
    """Sample semantic rows, not tokens; candidate lengths are NOT corpus extrema."""
    rng = random.Random(seed)
    selected, report = {}, {}
    offset = 0
    for source, dataset in zip(sources, parts):
        n = len(dataset)
        count = min(n, candidates_per_source)
        if not count:
            raise ValueError(f"empty source {source}")
        # Spread candidate draws over the entire source, not a prefix or one OAS row.
        indices = [rng.randrange(i * n // count, (i + 1) * n // count) for i in range(count)]
        candidates = []
        for index in indices:
            record = dataset[index]
            lengths = [len(chain.sequence) for chain in record.chains]
            candidates.append(dict(index=offset + index, local_index=index, source=source,
                                   record_source=record.source, residues=sum(lengths),
                                   max_chain_length=max(lengths), chain_roles=record.chain_roles,
                                   semantic_sha256=hashlib.sha256(json.dumps(record.to_dict(), sort_keys=True).encode()).hexdigest()))
        candidates.sort(key=lambda x: (x["residues"], x["index"]))
        take = min(samples_per_source, len(candidates))
        chosen = [candidates[i * (len(candidates) - 1) // max(1, take - 1)] for i in range(take)]
        selected[source] = chosen
        report[source] = dict(total_records=n, candidate_count=count, selected=chosen,
                              candidate_residues=distribution([x["residues"] for x in candidates]),
                              candidate_max_chain_length=distribution([x["max_chain_length"] for x in candidates]))
        offset += n
    return selected, report


def make_plan(selected, batch_size, warmup, steps, seed):
    """Cycle 7 homogeneous sources plus mixed, across three length terciles."""
    rng = random.Random(seed)
    sources = list(selected)
    pools = {}
    for source, samples in selected.items():
        for bucket in range(3):
            rows = samples[bucket * len(samples) // 3:(bucket + 1) * len(samples) // 3]
            pools[source, bucket] = rows or samples
    plan = []
    # 8 groups * 3 length terciles = 24-step coverage cycle for seven sources.
    groups = sources + ["mixed"]
    for step in range(warmup + steps):
        group = groups[step % len(groups)]
        bucket = (step // len(groups)) % 3
        members = []
        for j in range(batch_size):
            source = sources[(step // len(groups) + j) % len(sources)] if group == "mixed" else group
            # Mixed batches intentionally combine different lengths (real padding).
            length_bucket = (bucket + j) % 3 if group == "mixed" else bucket
            members.append(rng.choice(pools[source, length_bucket]))
        plan.append(dict(step=step, phase="warmup" if step < warmup else "measured",
                         group=group, length_bucket=("short", "middle", "long")[bucket],
                         indices=[x["index"] for x in members],
                         sources=[x["source"] for x in members],
                         semantic_residues=sum(x["residues"] for x in members)))
    return plan


def batch_tensor_digest(batch):
    """Hash every tensor in the collator's flat CPU batch, including its schema."""
    import torch

    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if value.device.type != "cpu":
            raise ValueError("batch tensor digest requires CPU tensors before H2D")
        header = json.dumps([key, str(value.dtype), list(value.shape)]).encode()
        # A byte view also supports BF16, which NumPy cannot represent directly.
        payload = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (header, payload):
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    return digest.hexdigest()


class TimedCollator:
    """Always invokes the actual grammar/remap collator on semantic records."""
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, records):
        started = time.perf_counter()
        batch = self.collator(records)
        collate_s = time.perf_counter() - started
        started = time.perf_counter()
        tensor_sha256 = batch_tensor_digest(batch)
        batch["_profile"] = dict(
            collate_s=collate_s, tensor_sha256=tensor_sha256,
            samples=len(records), residues=int(batch["residue_mask"].sum()),
            eligible_tokens=int(batch["diffusion_eligible_mask"].sum()),
            eligible_residues=int((batch["diffusion_eligible_mask"] & batch["residue_mask"]).sum()),
            decoder_shape=list(batch["input_ids"].shape), encoder_shape=list(batch["encoder_input_ids"].shape),
            padding_tokens=int((~batch["attention_mask"]).sum()),
            record_sources=list(batch["sources"]),
            instrumentation_s=time.perf_counter() - started,
        )
        return batch


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


class Monitor:
    """Process-tree CPU/RSS and device-wide GPU measurements off the hot path."""
    def __init__(self, output, interval, gpu_uuid, marker):
        self.output = output
        self.interval = interval
        self.gpu_uuid = gpu_uuid
        self.marker = marker
        self.stop = threading.Event()
        self.problem = None
        self.allowed_pids = set()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.phase = "startup"
        self.samples = 0
        self.baseline_ready = False
        self.progress = {}

    def establish_own_gpu_processes(self):
        snapshot = gpu_snapshot()
        uuid = gpu_card(snapshot, gpu_uuid=self.gpu_uuid)["uuid"]
        self.allowed_pids = {x["pid"] for x in snapshot["compute_apps"] if x["uuid"] == uuid}
        if len(self.allowed_pids) != 1:
            raise RuntimeError(f"cannot uniquely attribute newly initialized CUDA context: {snapshot}")
        self.baseline_ready = True
        write_json(self.output / "gpu_after_context_init.json", snapshot)

    def run(self):
        import psutil
        root = psutil.Process()
        previous = {}
        with (self.output / "resources.jsonl").open("a") as handle:
            while not self.stop.is_set():
                started = time.perf_counter()
                processes = []
                for process in [root] + root.children(recursive=True):
                    try:
                        with process.oneshot():
                            cpu = process.cpu_times()
                            total = cpu.user + cpu.system
                            old = previous.get(process.pid)
                            pct = 100 * (total - old[1]) / (started - old[0]) if old else None
                            previous[process.pid] = (started, total)
                            processes.append(dict(pid=process.pid, rss_mib=process.memory_info().rss / 2**20,
                                                  cpu_pct_one_core=pct, cpu_seconds=total))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                gpu = gpu_snapshot()
                if not io_is_clear(self.marker):
                    self.problem = "IO permission revoked; stopping only this profiling job"
                if gpu["error"]:
                    self.problem = "GPU monitoring failed: " + gpu["error"]
                elif self.baseline_ready:
                    try:
                        card = gpu_card(gpu, gpu_uuid=self.gpu_uuid)
                    except RuntimeError as exc:
                        self.problem = f"GPU monitoring failed: {exc}"
                    else:
                        foreign = [x for x in gpu["compute_apps"] if x["uuid"] == card["uuid"] and x["pid"] not in self.allowed_pids]
                        if foreign:
                            self.problem = f"new foreign GPU process observed; result invalid: {foreign}"
                row = dict(timestamp=time.time(), phase=self.phase, processes=processes,
                           tree_rss_mib=sum(x["rss_mib"] for x in processes),
                           system_cpu_pct=psutil.cpu_percent(), gpu=gpu, problem=self.problem)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                write_json(self.output / "progress.json", dict(**self.progress, phase=self.phase))
                self.samples += 1
                self.stop.wait(max(0.05, self.interval - (time.perf_counter() - started)))

    def check(self):
        if not self.thread.is_alive():
            raise RuntimeError("resource monitor exited unexpectedly")
        if self.problem:
            raise RuntimeError(self.problem)


def aggregate(rows):
    elapsed = sum(x["step_time_s"] for x in rows)
    return dict(steps=len(rows), next_wait_s=distribution([x["next_wait_s"] for x in rows]),
                step_time_s=distribution([x["step_time_s"] for x in rows]),
                collate_s=distribution([x["collate_s"] for x in rows]),
                samples_per_sec=sum(x["samples"] for x in rows) / elapsed,
                residues_per_sec=sum(x["residues"] for x in rows) / elapsed,
                exposed_next_fraction=sum(x["next_wait_s"] for x in rows) / elapsed)


def compute_profile_loss(trainer, model, batch):
    # Trainer's loss context alone enables only CPU AMP. This loop bypasses the
    # Accelerate model preparation that normally installs CUDA forward autocast.
    with trainer.accelerator.autocast(), trainer.compute_loss_context_manager():
        return trainer.compute_loss(model, batch)


def profile_trainer(trainer, cfg, output, monitor):
    import torch
    import transformers
    from torch.utils.data import DataLoader

    model = trainer.model
    if trainer.args.world_size != 1 or not str(trainer.args.device).startswith("cuda"):
        raise RuntimeError("this harness requires exactly one local CUDA process; no FSDP/DDP claim")
    decoder_cfg = model.decoder.config
    expected = dict(d_model=768, n_layers=8, n_heads=12, mlp_hidden_size=3072)
    if any(getattr(decoder_cfg, key) != value for key, value in expected.items()):
        raise RuntimeError(f"unexpected decoder dimensions; required {expected}")
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("ESMC must remain frozen")
    parts = list(getattr(trainer.train_dataset, "datasets", [trainer.train_dataset]))
    if len(parts) != len(SOURCES):
        raise RuntimeError("all seven prepared source datasets are required")
    monitor.phase = "semantic_sampling"
    started = time.perf_counter()
    selected, selection = select_samples(parts, SOURCES, cfg["seed"], cfg["candidates_per_source"], cfg["samples_per_source"])
    plan = make_plan(selected, cfg["batch_size"], cfg["warmup"], cfg["steps"], cfg["seed"])
    write_json(output / "sample_plan.json", dict(selection=selection, plan=plan,
               selection_seconds=time.perf_counter() - started,
               caveat="Balanced length-stratified diagnostic mix; not corpus-frequency-weighted throughput. Length extrema are from bounded candidates, not a full semantic scan."))
    # Preserve initial weights only in RAM. No checkpoint or model-ready cache.
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    metadata = dict(decoder_parameters=sum(p.numel() for p in model.decoder.parameters()),
                    encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    decoder_config=model.decoder.config.to_dict(), dtype=str(next(model.parameters()).dtype),
                    device=str(trainer.args.device), encoder_frozen=True)
    write_json(output / "model.json", metadata)
    reference_hashes = None
    results = []
    for workers in cfg["workers"]:
        monitor.check()
        monitor.phase = f"workers={workers}:reset"
        trainer.optimizer = None
        model.zero_grad(set_to_none=True)
        model.load_state_dict(initial)
        trainer.create_optimizer()
        optimizer = trainer.optimizer
        if optimizer is None:
            raise RuntimeError("Trainer did not construct an optimizer")
        transformers.set_seed(cfg["seed"])
        model.train()
        kwargs: dict[str, Any] = dict(dataset=trainer.train_dataset, batch_sampler=[row["indices"] for row in plan],
                      collate_fn=TimedCollator(trainer.data_collator), num_workers=workers,
                      pin_memory=True, generator=torch.Generator().manual_seed(cfg["seed"]),
                      worker_init_fn=seed_worker)
        if workers:
            kwargs.update(persistent_workers=True, prefetch_factor=cfg["prefetch_factor"],
                          multiprocessing_context="spawn", timeout=cfg["loader_timeout"])
        started = time.perf_counter()
        loader = DataLoader(**kwargs)
        construction_s = time.perf_counter() - started
        started = time.perf_counter()
        iterator = iter(loader)
        iterator_s = time.perf_counter() - started
        rows, hashes = [], []
        previous_end = None
        try:
            for entry in plan:
                monitor.check()
                monitor.phase = f"workers={workers}:{entry['phase']}"
                monitor.progress = dict(workers=workers, step=entry["step"])
                if entry["step"] == cfg["warmup"]:
                    torch.cuda.reset_peak_memory_stats()
                # Previous iteration has completed; no artificial loader-only timing.
                start = time.perf_counter()
                batch = next(iterator)
                next_wait = time.perf_counter() - start
                gap = start - previous_end if previous_end is not None else None
                info = batch.pop("_profile")
                hashes.append(info["tensor_sha256"])
                if reference_hashes is not None and hashes[-1] != reference_hashes[entry["step"]]:
                    raise RuntimeError("DataLoader tensors changed across worker counts")
                batch = trainer._prepare_inputs(batch)
                optimizer.zero_grad(set_to_none=True)
                loss = compute_profile_loss(trainer, model, batch)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.args.max_grad_norm)
                optimizer.step()
                torch.cuda.synchronize()
                previous_end = time.perf_counter()
                elapsed = previous_end - start
                loss_value = float(loss.detach())
                norm_value = float(grad_norm)
                if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                    raise RuntimeError(f"nonfinite loss/gradient: {loss_value}, {norm_value}")
                row = dict(**entry, **info, workers=workers, next_wait_s=next_wait,
                           step_time_s=elapsed, gap_before_step_s=gap, loss=loss_value, grad_norm=norm_value,
                           samples_per_sec=info["samples"] / elapsed,
                           residues_per_sec=info["residues"] / elapsed,
                           cuda_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                           cuda_reserved_mib=torch.cuda.memory_reserved() / 2**20,
                           cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
                rows.append(row)
                del batch, loss
        finally:
            if workers:
                # Private API is needed to close persistent workers on failure as well.
                getattr(iterator, "_shutdown_workers")()
            # Do not insert filesystem writes between steps: that hides loader waits.
            with (output / f"steps_workers_{workers}.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            del iterator, loader
        if reference_hashes is None:
            reference_hashes = hashes
        measured = [x for x in rows if x["phase"] == "measured"]
        groups = sorted({x["group"] for x in measured})
        wall_s = sum(x["step_time_s"] for x in measured) + sum(x["gap_before_step_s"] or 0 for x in measured[1:])
        result = dict(measured_loop_wall_s=wall_s,
                      wall_samples_per_sec=sum(x["samples"] for x in measured) / wall_s,
                      wall_residues_per_sec=sum(x["residues"] for x in measured) / wall_s,
                      warmup_summary=aggregate([x for x in rows if x["phase"] == "warmup"]),
                      measured_windows=[aggregate(measured[i:i + 24]) for i in range(0, len(measured), 24)],
                      workers=workers, loader_construction_s=construction_s, iterator_start_s=iterator_s,
                      first_next_wait_s=rows[0]["next_wait_s"],
                      overall=aggregate(measured),
                      per_group={g: aggregate([x for x in measured if x["group"] == g]) for g in groups},
                      per_length_bucket={g: aggregate([x for x in measured if x["length_bucket"] == g])
                                         for g in sorted({x["length_bucket"] for x in measured})},
                      loss_trace=[x["loss"] for x in rows], batch_hashes=hashes)
        results.append(result)
        write_json(output / f"profile_workers_{workers}.json", result)
        del optimizer
        trainer.optimizer = None
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    monitor.check()
    reference_losses = results[0]["loss_trace"]
    return dict(model=metadata, profiles=results, identical_batches_across_workers=True,
                loss_trace_max_abs_difference={str(x["workers"]): max(abs(a - b) for a, b in zip(reference_losses, x["loss_trace"])) for x in results})


class ProfileFinished(Exception):
    """Exit the real entry before its final save_model/tokenizer.save_pretrained."""


def train_cli(cfg, output):
    return ["profile_immune_llada.py",
            "--model_name_or_path", SNAPSHOT, "--decoder_init", "scratch",
            "--decoder_d_model", "768", "--decoder_n_layers", "8", "--decoder_n_heads", "12",
            "--decoder_mlp_hidden", "3072",
            "--decoder_weight_tying", "False", "--prepared_data_dir", str(ROOT / "data/prepared/immune_v3_heterotypic"),
            "--dataset_args", "+".join(SOURCES), "--esmc_path", str(ROOT / "model_weights/esmc/ESMC-300M"),
            "--max_length", "1024", "--max_protein_length", "1024",
            "--decoder_grad_ckpt", "True", "--freeze_encoder", "True", "--residue_cond_mode", "add",
            "--train_objective", "diffusion", "--bf16", "True",
            "--output_dir", str(output / "trainer"), "--report_to", "none",
            "--save_strategy", "no", "--eval_strategy", "no", "--save_top_k", "0", "--slim_checkpoints", "False",
            "--optim", "adamw_torch", "--learning_rate", "0.0001", "--weight_decay", "0.0",
            "--seed", str(cfg["seed"]), "--disable_tqdm", "True"]


def run_child(cfg, output):
    snap = gpu_snapshot()
    reasons = child_gate_reasons(snap, cfg)
    if reasons or not io_is_clear(cfg["io_clear_file"]):
        raise RuntimeError(f"pre-import safety gate failed: {reasons}; IO clear={io_is_clear(cfg['io_clear_file'])}")
    sys.path.insert(0, str(ROOT))
    import torch
    from examples.llada import protein_pretrain_esmc as entry

    torch.set_num_threads(cfg["cpu_threads"])
    monitor = Monitor(output, cfg["monitor_interval"], cfg["gpu_uuid"], cfg["io_clear_file"])
    # Establish the only newly appearing GPU PID immediately, before indexing or
    # model loading, so a process arriving during those slow stages is not trusted.
    reasons = child_gate_reasons(gpu_snapshot(), cfg)
    if reasons or not io_is_clear(cfg["io_clear_file"]):
        raise RuntimeError(f"post-import safety gate failed: {reasons}")
    context_probe = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    monitor.establish_own_gpu_processes()
    del context_probe
    monitor.thread.start()
    dataset_calls = []
    original_loader = entry.load_prepared_dataset
    started = time.perf_counter()
    result = None

    def timed_loader(*args, **kwargs):
        monitor.check()
        monitor.phase = f"dataset_index:{kwargs.get('split')}:{kwargs.get('source')}"
        start = time.perf_counter()
        dataset = original_loader(*args, **kwargs)
        dataset_calls.append(dict(split=kwargs.get("split"), source=kwargs.get("source"),
                                  records=len(dataset), init_seconds=time.perf_counter() - start))
        write_json(output / "dataset_initialization.json", dataset_calls)
        return dataset

    class ProfileTrainer(entry.FusionTrainer):
        def train(self, *args, **kwargs):
            nonlocal result
            result = profile_trainer(self, cfg, output, monitor)
            raise ProfileFinished()

    argv = train_cli(cfg, output)
    write_json(output / "training_construction_argv.json", argv)
    try:
        with patch.object(entry, "FusionTrainer", ProfileTrainer), patch.object(entry, "load_prepared_dataset", timed_loader), patch.object(sys, "argv", argv):
            try:
                entry.train()
            except ProfileFinished:
                pass
        if result is None:
            raise RuntimeError("real train entry never reached profile loop")
        result.update(status="smoke_complete" if cfg["smoke"] else "complete",
                      configuration=cfg, dataset_initialization=dataset_calls,
                      elapsed_s=time.perf_counter() - started,
                      torch_version=torch.__version__, cuda_version=getattr(torch, "version").cuda,
                      fsdp_or_multi_gpu_verified=False, model_ready_cache=False,
                      limitations=["Single GPU only; no FSDP/DDP validation.",
                                   "Fixed-LR AdamW diagnostic loop, without a scheduler or the production Trainer training loop; not Trainer equivalence.",
                                   "Single fixed-order worker comparison; no repeated/counterbalanced trials or robust fastest-worker claim.",
                                   "Balanced source/length diagnostic sampling, not natural corpus mixture throughput.",
                                   "OS page cache is not flushed; index time is observed, not a cold-cache guarantee.",
                                   "next_wait is wall time around next(iterator), with real model work between calls.",
                                   "step_time includes next, H2D, masking, forward, backward, clip, optimizer and CUDA completion; excludes JSON logging.",
                                   "Collator instrumentation overhead is recorded, and included in exposed wait when not hidden by prefetch.",
                                   "CPU percentages use one-core=100%; summed process RSS may double-count shared pages.",
                                   "Production diffusion eligibility is preserved, including generated grammar tokens; residues/sec counts residues only.",
                                   "GPU utilization is device-wide sampled telemetry, not per-kernel or per-process attribution.",
                                   "On PID-namespaced hosts the single GPU PID appearing immediately after gated CUDA init is treated as this child; attribution is not host-PID identity proof.",
                                   "Dataset initialization is performed once, shared across all worker arms; train and valid timings are separate."])
        monitor.check()
        write_json(output / "result.json", result)
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=25)


def stop_owned_group(child):
    """Clean even orphaned workers if the session leader has already exited."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # A dead leader does not imply its spawned workers exited. Only this private
    # session/group is addressed; never search for or signal processes by name.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=5)


def supervise(cfg, output):
    (output / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    started = time.monotonic()
    deadline = started + cfg["max_seconds"]
    wait_deadline = min(deadline, started + cfg["wait_seconds"])
    streak = 0
    gpu_uuid = None
    status: dict[str, Any] = dict(status="waiting", supervisor_pid=os.getpid(), hard_limit_seconds=cfg["max_seconds"])
    child = None
    temporary_fd = None
    def terminate(signum, frame):
        raise InterruptedError(f"supervisor received signal {signum}")
    def alarm(signum, frame):
        raise ProfileDeadline("hard wall-clock deadline reached")
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, cfg["max_seconds"])
    try:
        while time.monotonic() < wait_deadline:
            snap = gpu_snapshot()
            reasons = idle_reasons(snap, cfg["gpu_index"])
            if not io_is_clear(cfg["io_clear_file"]):
                reasons.append("IO clearance absent; do not overlap the main-agent parity scan")
            if reasons:
                streak = 0
            else:
                observed_uuid = gpu_card(snap, cfg["gpu_index"])["uuid"]
                streak = streak + 1 if observed_uuid == gpu_uuid else 1
                gpu_uuid = observed_uuid
            status.update(gpu=snap, reasons=reasons, idle_streak=streak, gpu_uuid=gpu_uuid)
            write_json(output / "status.json", status)
            if streak >= 3:
                break
            time.sleep(min(3, max(0, wait_deadline - time.monotonic())))
        if streak < 3:
            status["status"] = "blocked"
            write_json(output / "status.json", status)
            return 3
        # Persist the same verified identity used by the idle streak. CUDA ordinal
        # order is not guaranteed to match nvidia-smi/NVML index order.
        cfg = dict(cfg, gpu_uuid=gpu_uuid)
        write_json(output / "configuration.json", cfg)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES=gpu_uuid,
                   OMP_NUM_THREADS=str(cfg["cpu_threads"]), MKL_NUM_THREADS=str(cfg["cpu_threads"]),
                   HF_MODULES_CACHE=str(output / "hf_modules"), TRITON_CACHE_DIR=str(output / "triton"),
                   TORCHINDUCTOR_CACHE_DIR=str(output / "torchinductor"), TMPDIR=str(output / "tmp"),
                   WANDB_MODE="disabled", CUDA_CACHE_PATH=str(output / "cuda_cache"))
        for key in ("HF_MODULES_CACHE", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR", "CUDA_CACHE_PATH"):
            Path(env[key]).mkdir(exist_ok=True)
        # Python multiprocessing uses pathname Unix sockets (limit ~108 bytes).
        # Keep writes in the allowed artifact directory but address it through a
        # short /proc FD alias, valid until this supervisor finishes cleanup.
        temporary_fd = os.open(output / "tmp", os.O_RDONLY | os.O_DIRECTORY)
        env["TMPDIR"] = f"/proc/{os.getpid()}/fd/{temporary_fd}"
        env["PYTHONHASHSEED"] = str(cfg["seed"] % 2**32)
        with (output / "child.log").open("w") as log:
            child = subprocess.Popen([PYTHON, "-B", str(Path(__file__).resolve()), "--_child", str(output)],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (output / "child.pid").write_text(str(child.pid) + "\n")
            status.update(status="running", child_pid=child.pid)
            write_json(output / "status.json", status)
            try:
                rc = child.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                stop_owned_group(child)
                status.update(status="timeout", reason="hard wall-clock deadline reached; only own process group terminated")
                rc = 124
        if status["status"] != "timeout":
            status.update(status=("smoke_complete" if cfg["smoke"] else "complete") if rc == 0 else "failed", returncode=rc)
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "status.json", status)
        return rc
    except ProfileDeadline as exc:
        status.update(status="timeout", reason=str(exc), returncode=124,
                      elapsed_seconds=time.monotonic() - started)
        write_json(output / "status.json", status)
        return 124
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, InterruptedError) else "failed", error=repr(exc))
        write_json(output / "status.json", status)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if child is not None:
            stop_owned_group(child)
        if temporary_fd is not None:
            os.close(temporary_fd)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-gpu", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--background", action="store_true", help="detach bounded supervisor; prints artifact path/PID")
    parser.add_argument("--smoke", action="store_true", help="2 warmup + 8 measured steps, all workers; not a benchmark")
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4], choices=[0, 1, 2, 4])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--candidates-per-source", type=int, default=1024)
    parser.add_argument("--samples-per-source", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--loader-timeout", type=int, default=120, help="fail rather than silently hang waiting for a worker batch")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--monitor-interval", type=float, default=2)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--wait-seconds", type=int, default=300)
    parser.add_argument("--max-seconds", type=int, default=3600, help="hard wall clock including idle waiting and dataset indexing")
    parser.add_argument("--io-clear-file", default=str(ARTIFACT_ROOT / "IO_CLEAR"))
    parser.add_argument("--_supervise", help=argparse.SUPPRESS)
    parser.add_argument("--_child", help=argparse.SUPPRESS)
    cfg = vars(parser.parse_args(argv))
    for name in ("batch_size", "warmup", "steps", "candidates_per_source", "samples_per_source", "prefetch_factor", "loader_timeout", "cpu_threads", "wait_seconds", "max_seconds", "monitor_interval"):
        if cfg[name] <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if cfg["samples_per_source"] < 3 or cfg["candidates_per_source"] < cfg["samples_per_source"]:
        parser.error("require candidates-per-source >= samples-per-source >= 3")
    if len(set(cfg["workers"])) != len(cfg["workers"]):
        parser.error("worker counts must be unique")
    if cfg["smoke"]:
        cfg.update(warmup=2, steps=8)
    elif cfg["warmup"] < 24 or cfg["steps"] < 24:
        parser.error("benchmark requires >=24 warmup and >=24 measured steps for source/length coverage; use --smoke otherwise")
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    if cfg["_supervise"] or cfg["_child"]:
        output = Path(cfg["_supervise"] or cfg["_child"]).resolve()
        if not output.is_relative_to(ARTIFACT_ROOT.resolve()):
            raise ValueError("internal output path must remain inside the profiling artifact root")
        saved = json.loads((output / "configuration.json").read_text())
        if cfg["_supervise"]:
            return supervise(saved, output)
        try:
            run_child(saved, output)
            return 0
        except BaseException:
            (output / "error.txt").write_text(traceback.format_exc())
            raise
    if cfg["check_gpu"]:
        snap = gpu_snapshot()
        print(json.dumps(dict(snapshot=snap, blocking_reasons=idle_reasons(snap, cfg["gpu_index"])), indent=2))
    if not cfg["run"]:
        if not cfg["check_gpu"]:
            raise SystemExit("choose --check-gpu or --run (use --help for details)")
        return 0
    if any(os.environ.get(key, "1") != "1" for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        raise SystemExit("refusing a distributed launcher environment")
    cfg["io_clear_file"] = str(Path(cfg["io_clear_file"]).resolve())
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    output = ARTIFACT_ROOT / (time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"_{os.getpid()}")
    output.mkdir()  # Never overwrite an earlier run, including failed runs.
    write_json(output / "configuration.json", cfg)
    write_json(output / "status.json", dict(status="launching"))
    with (output / "supervisor.log").open("w") as log:
        process = subprocess.Popen([PYTHON, "-B", str(Path(__file__).resolve()), "--_supervise", str(output)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    print(json.dumps(dict(output=str(output), supervisor_pid=process.pid, hard_limit_seconds=cfg["max_seconds"]), indent=2), flush=True)
    if cfg["background"]:
        return 0
    try:
        return process.wait()
    except KeyboardInterrupt:
        # Supervisor's SIGTERM handler cleans up only its own child group.
        process.terminate()
        return process.wait(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
