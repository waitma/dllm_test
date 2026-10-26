"""Run the frozen AB500K A/D pilot; activate pllm before invoking.

CPU: python -B scripts/research/run_stage1_ab500k.py --plan ABS --arm mlm --verify-only
GPU: torchrun --standalone --nproc_per_node=8 THIS_FILE --plan ABS --arm pair_kd
The entrypoint profiles each arm, checks complete-state replay, then trains.
It never submits a job. Teacher targets remain scientifically unvalidated.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import gc
import hashlib
import json
import math
import os
import random
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import (
    FullTokenMLMCollator, FullTokenMLMModel, RECIPE, corrupt)
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel, PairConfig
from dllm.pipelines.immune_llada.fixed_chain.structure_losses import chain_aware_kd, normalized_objective
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from scripts.research.bert_mlm_data import IndexedPreparedDataset, CompleteEpochBatchSampler, step_budget, sha, write_json
from scripts.research.bert_mlm_eval import fixed_corruption, read_samples
from scripts.research.run_bert_mlm import (rank_zero, save_checkpoint, load_checkpoint,
    capture_rng, restore_rng, optimizer_update_difference)
from scripts.research.run_stage1_training import scheduler_for
from scripts.research.run_stage1_distillation import make_optimizer, autocast_context
from scripts.research.stage1_data import batch_teacher_targets
from scripts.research.stage1_ab500k_initialization import (
    build_scratch_model, shared_state_hash, verify_scratch_initialization)


DEFAULTS = dict(max_steps=3000, global_batch=512, seed=42, backbone_lr=1e-5,
    pair_lr=1e-4, weight_decay=.01, warmup_steps=500, kd_warmup_steps=1000,
    intra_weight=.05, inter_weight=0., temperature=1., clip_norm=1., precision="bf16",
    eval_every=250, save_every=250, log_every=25, loader_workers=4)
NOISE_RECIPE = "full_token_per_parent_row_epoch.v1"
VALID_SEEDS = (41016, 51016, 61016)


def identity(rows):
    return rows


def training_loader(dataset, sampler, settings, rank, epoch):
    """Do not fork a rank that has already initialized CUDA/NCCL threads."""
    workers = int(settings["loader_workers"])
    options = (dict(multiprocessing_context="spawn", timeout=120, prefetch_factor=1)
               if workers else {})
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=identity,
        num_workers=workers, generator=torch.Generator().manual_seed(settings["seed"]+epoch+rank),
        **options)


def record_training_metrics(out, arm, state, scores, stopped=False):
    """Preserve cumulative token counters and separately name this batch's counts."""
    metrics = {**scores, **state, "arm": arm,
               "batch_selected_tokens": scores["selected_tokens"],
               "batch_eligible_tokens": scores["eligible_tokens"]}
    with (Path(out)/"metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(metrics, allow_nan=False)+"\n")
    write_json(Path(out)/"status.json", dict(status="stopped" if stopped else "training", **metrics))
    print(json.dumps(metrics, allow_nan=False), flush=True)


def training_phase(out, arm, rank, stage, step):
    """Record each rank's startup position before potentially blocking work."""
    event = dict(event="training_phase", arm=arm, rank=rank, stage=stage,
                 step=step, pid=os.getpid(), time_unix=time.time())
    write_json(Path(out)/f"runtime_rank{rank}.json", event)
    if rank == 0:
        print(json.dumps(event), flush=True)


def collective_sum(value, world):
    if world > 1:
        torch.distributed.all_reduce(value)
    return value


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reference(manifest, name):
    if name in manifest.get("files", {}):
        return manifest["files"][name]
    return dict(path=manifest[name], sha256=manifest[name+"_sha256"],
                records=manifest["counts"]["valid" if name != "train_rows" else "train"])


def verify_plan(path, arm="pair_kd"):
    """Validate immutable inputs; missing not-yet-generated data is pending, not passed."""
    if arm not in {"mlm", "pair_kd"}:
        raise ValueError("Unknown arm")
    needs_cache = arm == "pair_kd"
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") not in {"immune.stage1.ab500k_pilot.v1", "immune.stage1.ab500k_training.v1"}:
        raise ValueError("Wrong AB500K plan schema")
    settings = {**DEFAULTS, **plan.get("training", {})}
    scratch = plan.get("initialization", {}).get("mode") == "scratch"
    expected = {**DEFAULTS, "backbone_lr": 1e-4} if scratch else DEFAULTS
    variable = {"loader_workers", "max_steps"} if scratch else {"loader_workers"}
    if any(settings[k] != v for k, v in expected.items() if k not in variable):
        raise ValueError("Pilot recipe differs from the declared A/D experiment")
    if type(settings["max_steps"]) is not int or settings["max_steps"] <= settings["warmup_steps"]:
        raise ValueError("Training budget must exceed warmup")
    if plan.get("world_size", 8) != 8:
        raise ValueError("Pilot requires eight student ranks")
    if Path(plan["source_root"]).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("Execute this plan's frozen source entrypoint")
    for source, digest in plan["sources"].items():
        if sha(source) != digest:
            raise ValueError("Frozen source changed: " + source)
    if scratch:
        verify_scratch_initialization(plan)
    else:
        parent = Path(plan["parent_checkpoint"])
        if sha(parent/"model.safetensors") != plan["parent_model_sha256"]:
            raise ValueError("Wrong BERT parent weights")
        if sha(parent/"checkpoint_receipt.json") != plan["parent_receipt_sha256"]:
            raise ValueError("Parent receipt changed")
        receipt = json.loads((parent/"checkpoint_receipt.json").read_text())
        if receipt.get("status") != "complete" or receipt.get("step") != 287000:
            raise ValueError("Parent is not complete BERT287000")
        for name in ("immune_config.json", "research_config.json", "vocabulary.json"):
            if sha(parent/name) != receipt["files"][name]:
                raise ValueError("Parent model configuration changed")
    pending = []
    subset_path, cache_path = Path(plan["subset_manifest"]), Path(plan["cache_manifest"])
    if not subset_path.is_file():
        pending.append("subset_manifest")
    if needs_cache and not cache_path.is_file():
        pending.append("complete_offline_cache")
    subset = cache = None
    if subset_path.is_file():
        from scripts.research.stage1_ab500k_data import validate_subset_files
        validate_subset_files(subset_path)
        subset = json.loads(subset_path.read_text())
        validation_path = plan.get("subset_validation")
        if not validation_path or not Path(validation_path).is_file():
            pending.append("subset_deep_validation_receipt")
        else:
            if sha(validation_path) != plan["subset_validation_sha256"]:
                raise ValueError("Subset deep validation receipt changed")
            validated = json.loads(Path(validation_path).read_text())
            if validated.get("status") != "passed" or validated.get("manifest_sha256") != sha(subset_path):
                raise ValueError("Subset lacks a matching deep validation receipt")
        expected = plan.get("subset_manifest_sha256")
        if expected and sha(subset_path) != expected:
            raise ValueError("Frozen subset changed")
        if subset["counts"]["train"] != 500000 or subset["counts"]["valid"] != 4096:
            raise ValueError("AB500K sample counts differ")
    if needs_cache and cache_path.is_file():
        from scripts.research.stage1_ab500k_cache import validate_cache_manifest
        # The atomic cache manifest is published only after the full payload audit.
        # Bind its immutable files/complete row index here; the reader checks and
        # decodes every payload again when used, without repeating all 100 GB on GPUs.
        cache = validate_cache_manifest(cache_path, verify_payloads=False)
        if (cache["status"] != "passed" or cache["quality_status"] != "unvalidated"
                or cache["counts"]["train"] != 500000 or cache["counts"]["valid"] != 4096
                or cache["subset_manifest_sha256"] != sha(subset_path)):
            raise ValueError("Cache must cover the exact complete subset; quality remains unvalidated")
        if plan.get("teacher_contract") and cache["teacher_contract"] != plan["teacher_contract"]:
            raise ValueError("Teacher contract differs")
    return dict(status="not_ready" if pending else "passed", pending=pending,
                plan=plan, subset=subset, cache=cache, settings=settings,
                teacher_quality="unvalidated")


class PilotDataset(Dataset):
    def __init__(self, subset, cache_manifest, split="train", *, targets=True):
        self.dataset = IndexedPreparedDataset(subset["data_index"], split)
        self.rows = np.load(reference(subset, split+"_rows")["path"], mmap_mode="r")
        self.split = split
        self.store = None
        if targets:
            from scripts.research.stage1_ab500k_cache import PackedCacheStore
            self.store = PackedCacheStore(cache_manifest)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        record = self.dataset[row]
        return row, record, self.store.lookup(self.split, row, record) if self.store else None


def training_noise_seed(seed, epoch, parent_train_records):
    return int(seed + 10000 + epoch * parent_train_records)


def prepare_batch(model, rows, *, seed, targets=True):
    """Each clean record has its own RNG, independent of microbatch and padding."""
    clean = FullTokenMLMCollator(model.config)([item[1] for item in rows])
    noisy = fixed_corruption(clean, model.tokenizer, [int(item[0]) for item in rows],
                              seed=seed, corrupt_fn=corrupt)
    if targets:
        if any(item[2] is None for item in rows):
            raise ValueError("Missing offline KD target; this experiment does not downgrade to MLM-only")
        pairs, teacher, _ = batch_teacher_targets([item[2] for item in rows], clean, "cpu")
        if teacher is None:
            raise ValueError("Offline KD batch contains no verified targets")
    else:
        pairs, teacher = torch.empty((0, 3), dtype=torch.long), torch.empty((0, 64))
    return noisy, pairs, teacher


def build_micros(model, rows, micro, seed, arm):
    return [prepare_batch(model, rows[i:i+micro], seed=seed, targets=arm == "pair_kd")
            for i in range(0, len(rows), micro)]


def optimizer_update(model, optimizer, scheduler, micros, settings, arm, step, world=1, precision=None):
    """Same global MLM denominator for A/D, with optional chain-normalized KD."""
    device = next(model.parameters()).device
    counts = torch.zeros(3, dtype=torch.float64, device=device)
    for batch, pairs, teacher in micros:
        counts[0] += batch["labels"].ne(-100).sum().to(device)
        if len(pairs):
            row, i, j = pairs.unbind(-1)
            same = batch["chain_ids"][row, i] == batch["chain_ids"][row, j]
            counts[1] += row[same].unique().numel()
            counts[2] += row[~same].unique().numel()
    collective_sum(counts, world)
    if not counts[0]:
        raise ValueError("No global MLM targets")
    optimizer.zero_grad(set_to_none=True)
    values = torch.zeros(5, dtype=torch.float64, device=device)
    scale = min(1., (step+1)/max(1, settings["kd_warmup_steps"]))
    for number, (raw, pairs, teacher) in enumerate(micros):
        sync = model.no_sync() if hasattr(model, "no_sync") and number+1 < len(micros) else nullcontext()
        with sync:
            batch = {k: v.to(device) for k, v in raw.items()}
            pairs, teacher = pairs.to(device), teacher.to(device)
            with autocast_context(device, precision or settings["precision"]):
                result = model(pair_indices=pairs, **batch) if arm == "pair_kd" else model(**batch)
            if arm == "pair_kd":
                kd = chain_aware_kd(result["pair_logits"], teacher, pairs, batch["chain_ids"],
                                   batch["noise_kind"], settings["temperature"])
                loss = normalized_objective(result["loss_sum"], batch["labels"].ne(-100).sum(), kd,
                    settings["intra_weight"]*scale, 0., denominators=counts, world_size=world)
                values[1] += kd["intra_sum"].detach().double()
                values[2] += len(pairs)
            else:
                loss = result["loss_sum"] * world / counts[0]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite pilot loss")
            loss.backward()
            values[0] += result["loss_sum"].detach().double()
            values[3] += len(batch["input_ids"])
            values[4] += batch["eligible_mask"].sum()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_norm"], error_if_nonfinite=True)
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    collective_sum(values, world)
    mean_mlm, mean_kd = float(values[0]/counts[0]), float(values[1]/counts[1].clamp_min(1))
    return dict(mlm=mean_mlm, intra_kl=mean_kd,
        objective=mean_mlm + (settings["intra_weight"]*scale*mean_kd if arm == "pair_kd" else 0.),
        selected_tokens=int(counts[0]), kd_records=int(counts[1]), pairs=int(values[2]),
        records=int(values[3]), eligible_tokens=int(values[4]), kd_scale=scale if arm == "pair_kd" else 0.,
        backbone_lr=lr, grad_norm=float(norm))


def new_model(plan, arm, device):
    torch.manual_seed(plan["training"]["seed"])
    if plan.get("initialization", {}).get("mode") == "scratch":
        spec = verify_scratch_initialization(plan)
        model = build_scratch_model(plan["model"], arm, spec["seed"], plan.get("pair"))
        if shared_state_hash(model) != spec["shared_state_sha256"]:
            raise ValueError("Actual random initialization differs from frozen shared tensors")
    else:
        model = (FullTokenMLMModel.from_pretrained(plan["parent_checkpoint"]) if arm == "mlm" else
                 StructureMLMModel.from_bert(plan["parent_checkpoint"], PairConfig(**plan.get("pair", {}))))
    if arm == "pair_kd":
        if model.pair_config.feedback != "all":
            raise ValueError("Pilot D requires all eight feedback layers")
        model.teacher_contract = plan["teacher_contract"]
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    return model.to(device).train()


def new_optimizer(model, settings, arm):
    if arm == "pair_kd":
        return make_optimizer(model, settings["backbone_lr"], settings["pair_lr"], settings["weight_decay"])
    return torch.optim.AdamW(model.parameters(), lr=settings["backbone_lr"],
                            weight_decay=settings["weight_decay"], foreach=False)


def distributed_model(model, device):
    return DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                  broadcast_buffers=False, gradient_as_bucket_view=True)


def profile_attempt(plan, arm, device, rows, micro):
    model = new_model(plan, arm, device)
    optimizer = new_optimizer(model, plan["training"], arm)
    scheduler = scheduler_for(optimizer, plan["training"])
    inputs = build_micros(model, rows * micro, micro, 10042, arm)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    for step in range(3):
        optimizer_update(model, optimizer, scheduler, inputs, plan["training"], arm, step)
    torch.cuda.synchronize(device)
    total = torch.cuda.get_device_properties(device).total_memory/2**30
    reserved = torch.cuda.max_memory_reserved(device)/2**30
    return dict(status="passed" if reserved <= .85*total else "insufficient_headroom", micro_batch=micro,
        reserved_gib=reserved, allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        total_gib=total, sequence_length=inputs[0][0]["input_ids"].shape[1],
        seconds_per_micro_update=(time.monotonic()-start)/3, optimizer_steps=3)


def profile(plan, arm, dataset, out, device, rank, world, contract):
    receipt_path = out/"profile.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("contract") != contract or receipt.get("status") != "passed":
            raise ValueError("Existing arm profile belongs to another experiment")
        return receipt
    # OAS-only paired records share the same fixed H/L slots and no antigen context.
    # The data manifest binds this profile shape; evaluate full valid/training shapes below.
    rows = [dataset[0]]
    attempts, started = [], time.monotonic()
    for micro in (64, 48, 32, 24, 16, 12, 8, 4, 2, 1):
        try:
            result = profile_attempt(plan, arm, device, rows, micro)
        except torch.cuda.OutOfMemoryError:
            result = dict(status="oom", micro_batch=micro)
        gc.collect(); torch.cuda.empty_cache()
        gathered = [None]*world
        torch.distributed.all_gather_object(gathered, result)
        attempts.append(gathered)
        if rank == 0:
            print(json.dumps(dict(event="profile", arm=arm, micro_batch=micro, ranks=gathered)), flush=True)
        if all(value["status"] == "passed" for value in gathered):
            if result["sequence_length"] != plan["student_profile_length"]:
                raise ValueError("AB fixed-slot profile does not match the verified data shape")
            receipt = dict(status="passed", contract=contract, micro_batch=micro, attempts=attempts,
                seconds=time.monotonic()-started, sequence_length=result["sequence_length"],
                world_size=world, global_batch=512, scope="three_real_updates; per_arm_maximum_safe_micro")
            rank_zero(lambda: write_json(receipt_path, receipt), rank)
            return receipt
    raise RuntimeError("No safe microbatch on every rank")


def retire_partial(out, step):
    partial = out/f".checkpoint-{step}.tmp"
    if partial.exists():
        partial.rename(out/f".abandoned-{step}-{time.time_ns()}")


def validate_resume_cursor(state, records, global_batch):
    """Step zero is a real saved initialization with no data consumed."""
    if state["step"] < 0:
        raise ValueError("Resume optimizer step must be nonnegative")
    expected = (dict(completed_epochs=0, next_batch=0, seen_records=0) if state["step"] == 0
                else step_budget(records, global_batch, state["step"]))
    if any(state[key] != expected[field] for key, field in
           (("epoch", "completed_epochs"), ("next_batch", "next_batch"), ("seen_records", "seen_records"))):
        raise ValueError("Resume subset cursor differs")


def recover_completion(out, state, arm, settings):
    """A committed final checkpoint can outlive an interrupted completion-marker write."""
    if state["step"] != settings["max_steps"]:
        raise ValueError("Only the complete frozen budget can reconstruct completion")
    for step in range(0, settings["max_steps"]+1, settings["eval_every"]):
        result = json.loads((out/f"validation_step{step}.json").read_text())
        statistics = result["record_statistics"]
        if (result["step"] != step or result["records"] != 4096 or result["masks"] != 3
                or sha(statistics["path"]) != statistics["sha256"]):
            raise ValueError("Cannot reconstruct completion with missing or modified validation")
    write_json(out/"COMPLETE.json", dict(status="success", arm=arm, **state,
        teacher_quality="unvalidated", max_steps=settings["max_steps"],
        completion_reconstructed_from_checkpoint=True))


def replay_smoke(plan, arm, dataset, out, device, rank, world, provenance):
    receipt_path = out/"smoke.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("contract") != provenance or receipt.get("status") != "passed":
            raise ValueError("Existing replay qualification differs")
        return receipt
    smoke_out = out/"smoke"
    def prepare_directory():
        if smoke_out.exists():
            smoke_out.rename(out/f"smoke.abandoned-{time.time_ns()}")
        smoke_out.mkdir()
    rank_zero(prepare_directory, rank)
    started = time.monotonic()
    model = new_model(plan, arm, device)
    for block in model.backbone.transformer.blocks:
        block.sdpa_backend = "math"
    ddp = distributed_model(model, device)
    optimizer = new_optimizer(model, plan["training"], arm)
    scheduler = scheduler_for(optimizer, plan["training"])
    noise = torch.Generator().manual_seed(42+rank)
    rows = [dataset[rank]]
    inputs = build_micros(model, rows, 1, 10042, arm)
    optimizer_update(ddp, optimizer, scheduler, inputs, plan["training"], arm, 0, world, "fp32")
    state = dict(step=1, epoch=0, next_batch=1, seen_records=world, best_step=1, best_score=1e30)
    checkpoint = save_checkpoint(smoke_out, model, optimizer, scheduler, noise, state, provenance, rank, world)
    expected_inputs = build_micros(model, rows, 1, 10043, arm)
    optimizer_update(ddp, optimizer, scheduler, expected_inputs, plan["training"], arm, 1, world, "fp32")
    expected = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    moments = {key: {name: value.detach().cpu().clone() for name, value in item.items()}
               for key, item in optimizer.state_dict()["state"].items()}
    restored = load_checkpoint(checkpoint, model, optimizer, scheduler, noise, provenance, rank)
    if restored != state:
        raise ValueError("Smoke data cursor differs")
    actual_inputs = build_micros(model, rows, 1, 10043, arm)
    if any(not torch.equal(value, actual_inputs[0][0][key]) for key, value in expected_inputs[0][0].items()):
        raise ValueError("Smoke corruption replay differs")
    optimizer_update(ddp, optimizer, scheduler, actual_inputs, plan["training"], arm, 1, world, "fp32")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), expected[key], atol=1e-8, rtol=1e-6)
    differences = optimizer_update_difference(moments, optimizer.state_dict()["state"])
    if any(item["relative_l2"] > 1e-5 for item in differences.values()):
        raise ValueError("Smoke AdamW replay differs")
    write_json(out/f"smoke_rank{rank}.json", dict(status="passed", optimizer=differences, corruption_exact=True))
    torch.distributed.barrier()
    receipt = dict(status="passed", contract=provenance, seconds=time.monotonic()-started,
                   world_size=world, scope="engineering_resume_only")
    rank_zero(lambda: write_json(receipt_path, receipt), rank)
    return receipt


@torch.no_grad()
def validate(model, dataset, settings, arm, device, rank, world, micro, output, step):
    """Three immutable masks, token-weighted losses, per-record bootstrap sufficient statistics."""
    previous = model.training
    model.eval()
    local = []
    try:
        indices = list(range(rank, len(dataset), world))
        for start in range(0, len(indices), micro):
            ordinals = indices[start:start+micro]
            rows = [dataset[i] for i in ordinals]
            statistics = torch.zeros((len(rows), 8), dtype=torch.float64, device=device)
            for seed in VALID_SEEDS:
                raw, pairs, teacher = prepare_batch(model, rows, seed=seed, targets=arm == "pair_kd")
                batch = {key: value.to(device) for key, value in raw.items()}
                pairs, teacher = pairs.to(device), teacher.to(device)
                with autocast_context(device, settings["precision"]):
                    result = model(pair_indices=pairs, **batch) if arm == "pair_kd" else model(**batch)
                logits = result["logits"].float()
                for row in range(len(rows)):
                    selected = batch["labels"][row].ne(-100)
                    aa = selected & batch["residue_mask"][row] & batch["target_slot_mask"][row]
                    masked = batch["noise_kind"][row].eq(1)
                    for mask, column in ((selected, 0), (aa, 2)):
                        if mask.any():
                            statistics[row, column] += F.cross_entropy(logits[row, mask], batch["labels"][row, mask], reduction="sum")
                            statistics[row, column+1] += mask.sum()
                    statistics[row, 4] += (logits[row, masked].argmax(-1) == batch["labels"][row, masked]).sum()
                    statistics[row, 5] += masked.sum()
                if arm == "pair_kd":
                    for row in range(len(rows)):
                        use = pairs[:, 0].eq(row)
                        kd = chain_aware_kd(result["pair_logits"][use], teacher[use], pairs[use],
                            batch["chain_ids"], batch["noise_kind"], settings["temperature"])
                        statistics[row, 6] += kd["intra_sum"]
                        statistics[row, 7] += kd["intra_count"]
            local.extend((i, int(rows[j][0]), statistics[j].cpu().numpy()) for j, i in enumerate(ordinals))
        gathered = [None]*world
        if world > 1:
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered[0] = local
        ordered = sorted((item for group in gathered for item in group), key=lambda item: item[0])
        if [item[0] for item in ordered] != list(range(len(dataset))):
            raise ValueError("Validation rank coverage differs")
        arrays = np.stack([item[2] for item in ordered])
        sums = arrays.sum(0)
        metrics = dict(step=step, records=len(ordered), masks=3, seeds=list(VALID_SEEDS),
            full_token_ce=float(sums[0]/max(1, sums[1])), aa_ce=float(sums[2]/max(1, sums[3])),
            mask_accuracy=float(sums[4]/max(1, sums[5])),
            intra_kl=float(sums[6]/max(1, sums[7])) if arm == "pair_kd" else None,
            selected_tokens=int(sums[1]), aa_tokens=int(sums[3]),
            record_statistics_columns=["full_sum", "full_count", "aa_sum", "aa_count", "mask_correct", "mask_count", "kd_sum", "kd_count"])
        if rank == 0 and output is not None:
            target = Path(output)/f"validation_step{step}.records.npz"
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, row_ids=np.array([item[1] for item in ordered], dtype=np.int64), statistics=arrays)
            temporary.replace(target)
            metrics["record_statistics"] = dict(path=str(target), sha256=sha(target))
            write_json(Path(output)/f"validation_step{step}.json", metrics)
        return metrics
    finally:
        model.train(previous)


def run(plan_path, arm, resume):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires the submitted eight-GPU job; use --verify-only on CPU")
    rank, local_rank, world = (int(os.environ.get(key, default)) for key, default in
                               (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    if world != 8:
        raise ValueError("Run with torchrun --nproc_per_node=8")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl", timeout=timedelta(hours=3), device_id=device)
    lock = None
    try:
        checked = rank_zero(lambda: verify_plan(plan_path, arm), rank)
        if checked["status"] != "passed":
            raise ValueError("Offline preparation incomplete: " + str(checked["pending"]))
        plan, subset, cache, settings = (checked[key] for key in ("plan", "subset", "cache", "settings"))
        plan["training"] = settings
        if arm == "pair_kd":
            plan["teacher_contract"] = cache["teacher_contract"]
        plan["student_profile_length"] = subset["student_max_length"]
        out = Path(plan["output_root"])/arm
        def acquire():
            nonlocal lock
            out.mkdir(parents=True, exist_ok=True)
            lock = (out/"run.lock").open("a+")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        rank_zero(acquire, rank)
        dataset = PilotDataset(subset, cache, targets=arm == "pair_kd")
        validation = PilotDataset(subset, cache, "valid", targets=arm == "pair_kd")
        parent_count = json.loads(Path(subset["data_index"]).read_text())["records"]["train"]
        contract = dict(schema="immune.stage1.ab500k.run.v1", arm=arm, plan_sha256=sha(plan_path),
            subset_manifest_sha256=sha(plan["subset_manifest"]), cache_manifest_sha256=sha(plan["cache_manifest"]) if arm == "pair_kd" else None,
            parent_model_sha256=plan["parent_model_sha256"], initialization=plan.get("initialization"),
            training=settings, pair=plan.get("pair", {}),
            noise_recipe=NOISE_RECIPE, noise_parent_records=parent_count, recipe=RECIPE,
            validation_seeds=list(VALID_SEEDS), teacher_quality="unvalidated", world_size=world,
            torch=torch.__version__, source_root=plan["source_root"], sources=plan["sources"])
        profile_receipt = profile(plan, arm, dataset, out, device, rank, world, contract)
        micro = profile_receipt["micro_batch"]
        provenance = {**contract, "micro_batch": micro, "profile_sha256": sha(out/"profile.json")}
        smoke = replay_smoke(plan, arm, dataset, out, device, rank, world, provenance)
        gc.collect(); torch.cuda.empty_cache()
        latest_path = out/"latest.json"
        selected_resume = None
        if resume == "latest" and latest_path.exists():
            selected_resume = json.loads(latest_path.read_text())["checkpoint"]
        elif resume and resume != "latest":
            selected_resume = resume
        elif latest_path.exists():
            raise ValueError("Existing training checkpoints require --resume latest")
        if (out/"run_provenance.json").exists() and json.loads((out/"run_provenance.json").read_text()) != provenance:
            raise ValueError("Output directory belongs to another run")
        random.seed(settings["seed"]); np.random.seed(settings["seed"])
        model = new_model(plan, arm, device)
        ddp = distributed_model(model, device)
        optimizer = new_optimizer(model, settings, arm)
        scheduler = scheduler_for(optimizer, settings)
        noise = torch.Generator().manual_seed(settings["seed"]+rank)
        state = dict(step=0, epoch=0, next_batch=0, seen_records=0, selected_tokens=0, eligible_tokens=0,
            best_score=1e30, best_step=0, wall_seconds=0., io_seconds=0., update_seconds=0.,
            validation_seconds=0., checkpoint_seconds=0., timing_complete=True,
            timing_note="successful_session_measurement; excludes launch/preflight and process teardown")
        if selected_resume:
            state = load_checkpoint(selected_resume, model, optimizer, scheduler, noise, provenance, rank)
            validate_resume_cursor(state, len(dataset), 512)
            state.update(timing_complete=False,
                timing_note="lower_bound_after_resume; uncommitted work and the previous checkpoint's own save time may be absent")
        rank_zero(lambda: write_json(out/"run_provenance.json", provenance), rank)
        if state["step"] >= settings["max_steps"]:
            if state["step"] != settings["max_steps"]:
                raise ValueError("Checkpoint exceeds the frozen budget")
            if not (out/"COMPLETE.json").exists():
                rank_zero(lambda: recover_completion(out, state, arm, settings), rank)
            return
        wall_before, start = state["wall_seconds"], time.monotonic()
        def stamp():
            state["wall_seconds"] = wall_before + time.monotonic()-start
            state["student_gpu_hours"] = world*state["wall_seconds"]/3600
            state["student_with_qualification_gpu_hours"] = world*(state["wall_seconds"]+profile_receipt["seconds"]+smoke["seconds"])/3600
        def evaluate():
            began = time.monotonic()
            rng = capture_rng(noise)
            try:
                result = validate(model, validation, settings, arm, device, rank, world, micro, out, state["step"])
                if result["aa_ce"] < state["best_score"]:
                    state.update(best_score=result["aa_ce"], best_step=state["step"])
            finally:
                restore_rng(rng, noise)
            state["validation_seconds"] += time.monotonic()-began
            stamp()
            result.update(seen_records=state["seen_records"], wall_seconds=state["wall_seconds"],
                          student_gpu_hours=state["student_gpu_hours"])
            rank_zero(lambda: write_json(out/f"validation_step{state['step']}.json", result), rank)
        if state["step"] == 0:
            evaluate()
            if not selected_resume:
                began = time.monotonic()
                rank_zero(lambda: retire_partial(out, 0), rank)
                save_checkpoint(out, model, optimizer, scheduler, noise, state, provenance, rank, world)
                state["checkpoint_seconds"] += time.monotonic()-began
                stamp()
        stopped = False
        while state["step"] < settings["max_steps"] and not stopped:
            training_phase(out, arm, rank, "creating_loader", state["step"])
            sampler = CompleteEpochBatchSampler(len(dataset), 512, rank, world, settings["seed"],
                state["epoch"], state["next_batch"], settings["max_steps"]-state["step"])
            loader = training_loader(dataset, sampler, settings, rank, state["epoch"])
            iterator = iter(loader)
            while True:
                trace_step = state["step"] < 3 or state["step"] % settings["log_every"] == 0
                if trace_step:
                    training_phase(out, arm, rank, "loading_batch", state["step"])
                began = time.monotonic()
                try:
                    rows = next(iterator)
                except StopIteration:
                    break
                state["io_seconds"] += time.monotonic()-began
                if trace_step:
                    training_phase(out, arm, rank, "preparing_batch", state["step"])
                began = time.monotonic()
                seed = training_noise_seed(settings["seed"], state["epoch"], parent_count)
                micros = build_micros(model, rows, micro, seed, arm)
                if any(item[0]["input_ids"].shape[1] > profile_receipt["sequence_length"] for item in micros):
                    raise ValueError("Actual sequence exceeds qualified AB profile shape")
                if trace_step:
                    training_phase(out, arm, rank, "optimizer_update", state["step"])
                scores = optimizer_update(ddp, optimizer, scheduler, micros, settings, arm, state["step"], world)
                state["update_seconds"] += time.monotonic()-began
                state.update(step=state["step"]+1, next_batch=state["next_batch"]+1,
                    seen_records=state["seen_records"]+scores["records"],
                    selected_tokens=state["selected_tokens"]+scores["selected_tokens"],
                    eligible_tokens=state["eligible_tokens"]+scores["eligible_tokens"])
                if state["next_batch"] == math.ceil(len(dataset)/512):
                    state.update(epoch=state["epoch"]+1, next_batch=0)
                stopped = rank_zero(lambda: (out/"STOP").exists(), rank)
                finished = state["step"] == settings["max_steps"]
                if state["step"] % settings["eval_every"] == 0 or finished or stopped:
                    evaluate()
                stamp()
                if state["step"] % settings["save_every"] == 0 or finished or stopped:
                    began = time.monotonic()
                    rank_zero(lambda: retire_partial(out, state["step"]), rank)
                    save_checkpoint(out, model, optimizer, scheduler, noise, state, provenance, rank, world)
                    state["checkpoint_seconds"] += time.monotonic()-began
                    stamp()
                if state["step"] <= 3 or state["step"] % settings["log_every"] == 0 or finished or stopped:
                    rank_zero(lambda: record_training_metrics(out, arm, state, scores, stopped), rank)
                    training_phase(out, arm, rank, "step_complete", state["step"])
                if stopped or finished:
                    break
        stamp()
        rank_zero(lambda: write_json(out/("STOPPED.json" if stopped else "COMPLETE.json"),
            dict(status="stopped" if stopped else "success", arm=arm, **state,
                 teacher_quality="unvalidated", max_steps=settings["max_steps"])), rank)
    except BaseException:
        # Show the original failure before distributed cleanup can wait on peers.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if lock is not None:
            lock.close()
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", choices=("mlm", "pair_kd"), required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_plan(args.plan, args.arm)
        print(json.dumps({key: value for key, value in result.items() if key not in {"plan", "subset", "cache", "settings"}}))
        return 0 if result["status"] == "passed" else 2
    run(args.plan, args.arm, args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
