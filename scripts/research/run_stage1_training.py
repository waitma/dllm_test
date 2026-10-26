"""Full-corpus Stage 1 DDP training, global MLM/KD means, and exact state resume.

Activate pllm; execute via the frozen launch_stage1_training.py package.
Profile and smoke are engineering phases. Formal training requires independently
qualified teacher caches and uses the original BERT natural-ratio epoch sampler.
"""
from contextlib import nullcontext
import gc
import json
import math
from pathlib import Path
import random
import sqlite3
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel, PairConfig
from dllm.pipelines.immune_llada.fixed_chain.structure_losses import chain_aware_kd, normalized_objective
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator, corrupt, RECIPE
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from scripts.research.bert_mlm_data import CompleteEpochBatchSampler, step_budget, write_json
from scripts.research.bert_mlm_eval import token_validation
from scripts.research.run_bert_mlm import (rank_zero, save_checkpoint, load_checkpoint,
    capture_rng, restore_rng, optimizer_update_difference)
from scripts.research.run_stage1_distillation import make_optimizer, prepare_microbatch, autocast_context
from scripts.research.stage1_corpus import CachedCorpus, identity_batch


def scheduler_for(optimizer, settings):
    def scale(step):
        # First update is nonzero; scheduler cursor equals completed updates.
        if step < settings["warmup_steps"]:
            return max(.01, (step+1)/max(1, settings["warmup_steps"]))
        progress = min(1., (step-settings["warmup_steps"])/max(1, settings["max_steps"]-settings["warmup_steps"]))
        return .1 + .9 * .5 * (1 + math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def build_micros(model, rows, noise, micro):
    return [prepare_microbatch(model, [r[1] for r in rows[i:i+micro]],
            [r[2] for r in rows[i:i+micro]], noise, "cpu")
            for i in range(0, len(rows), micro)]


def step_counts(micros):
    counts = torch.zeros(3, dtype=torch.float64)
    for batch, pairs, teacher, _ in micros:
        counts[0] += batch["labels"].ne(-100).sum()
        if len(pairs):
            row, i, j = pairs.unbind(-1)
            same = batch["chain_ids"][row, i].eq(batch["chain_ids"][row, j])
            counts[1] += row[same].unique().numel()
            counts[2] += row[~same].unique().numel()
    return counts


def update(ddp, optimizer, scheduler, micros, settings, world, step, *, precision=None):
    """Sum microbatch numerators over step-wide global counts, including empty KD ranks."""
    device = next(ddp.parameters()).device
    totals = step_counts(micros).to(device)
    torch.distributed.all_reduce(totals)
    if not totals[0]:
        raise ValueError("No MLM targets in global optimizer step")
    kd_scale = min(1., (step+1)/max(1, settings["kd_warmup_steps"]))
    intra, inter = settings["intra_weight"]*kd_scale, settings["inter_weight"]*kd_scale
    optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(6, dtype=torch.float64, device=device)
    for number, (batch, pairs, teacher, _) in enumerate(micros):
        last = number == len(micros)-1
        sync = nullcontext() if last or not hasattr(ddp, "no_sync") else ddp.no_sync()
        with sync:
            batch = {k: v.to(device) for k, v in batch.items()}
            pairs, teacher = pairs.to(device), teacher.to(device)
            with autocast_context(device, precision or settings["precision"]):
                result = ddp(pair_indices=pairs, **batch)
            kd = chain_aware_kd(result["pair_logits"], teacher, pairs, batch["chain_ids"],
                               batch["noise_kind"], settings["temperature"])
            loss = normalized_objective(result["loss_sum"], result["mlm_count"], kd,
                intra, inter, denominators=totals, world_size=world)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite Stage 1 objective")
            loss.backward()
            sums[:3] += torch.stack([result["loss_sum"].detach(), kd["intra_sum"].detach(), kd["inter_sum"].detach()]).double()
            sums[3] += len(pairs)
            sums[4] += pairs[:, 0].unique().numel()
            sums[5] += len(batch["input_ids"])
            del result, loss, kd, batch, pairs, teacher
    norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), settings["clip_norm"], error_if_nonfinite=True)
    lrs = [group["lr"] for group in optimizer.param_groups]
    optimizer.step()
    scheduler.step()
    torch.distributed.all_reduce(sums)
    means = sums[:3]/totals.clamp_min(1)
    return dict(mlm=float(means[0]), intra_kl=float(means[1]), inter_kl=float(means[2]),
        loss=float(means[0]+intra*means[1]+inter*means[2]), counts=totals.tolist(),
        pairs=int(sums[3]), cached_records=int(sums[4]), records=int(sums[5]),
        kd_coverage=float(sums[4]/sums[5].clamp_min(1)), kd_scale=kd_scale,
        backbone_lr=lrs[0], pair_lr=lrs[1], grad_norm=float(norm))


def new_model(plan, device):
    model = StructureMLMModel.from_bert(plan["parent_checkpoint"], PairConfig(**plan["pair"])).to(device)
    model.teacher_contract = plan["teacher_contract"]
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    return model


@torch.no_grad()
def structure_validation(model, plan, manifest, device, rank, world):
    connection = sqlite3.connect(Path(manifest["database"]).as_uri()+"?mode=ro&immutable=1", uri=True)
    rows = [r[0] for r in connection.execute("SELECT row_index FROM targets WHERE split='valid' ORDER BY row_index")]
    connection.close()
    dataset = CachedCorpus(plan["data_index"], manifest, "valid", rows)
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    previous = model.training
    model.eval()
    try:
        for ordinal in range(rank, len(dataset), world):
            item = dataset[ordinal]
            batch, pairs, teacher, _ = prepare_microbatch(model, [item[1]], [item[2]],
                torch.Generator().manual_seed(41016+item[0]), device)
            with autocast_context(device, plan["training"]["precision"]):
                out = model(pair_indices=pairs, **batch)
            kd = chain_aware_kd(out["pair_logits"], teacher, pairs, batch["chain_ids"], batch["noise_kind"], plan["training"]["temperature"])
            totals += torch.stack([kd["intra_sum"], kd["inter_sum"], kd["intra_count"], kd["inter_count"]]).double()
        torch.distributed.all_reduce(totals)
    finally:
        model.train(previous)
    return dict(intra_kl=float(totals[0]/totals[2].clamp_min(1)), inter_kl=float(totals[1]/totals[3].clamp_min(1)),
                intra_records=int(totals[2]), inter_records=int(totals[3]), records=len(rows))


def smoke_replay(plan, ddp, optimizer, scheduler, noise, provenance, state, manifest, device, rank, world, online=None):
    settings, model = plan["training"], ddp.module
    connection = sqlite3.connect(manifest["database"])
    cached = [x[0] for x in connection.execute("SELECT row_index FROM targets WHERE split='train' ORDER BY row_index LIMIT 8")]
    connection.close()
    if not cached:
        raise ValueError("Smoke requires real teacher targets")
    dataset = CachedCorpus(plan["data_index"], manifest)
    # Odd ranks deliberately have no KD; test changing branch participation.
    rows = [dataset[cached[rank % len(cached)]]]
    online_report = None
    if online is not None:
        # Ignore the static seed lookup here to exercise the live miss/hit path.
        reference = rows[0][2]["tensors"]
        unresolved = [(r[0], r[1], None) for r in rows]
        cpu_rng, cuda_rng, noise_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device), noise.get_state()
        torch.cuda.empty_cache()
        rows, first = online.resolve(unresolved, model.config)
        reused, second = online.resolve(unresolved, model.config)
        for key, value in reference.items():
            if not torch.equal(value, rows[0][2]["tensors"][key]) or not torch.equal(value, reused[0][2]["tensors"][key]):
                raise ValueError("Online teacher differs from pinned pilot/reused targets")
        if (not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
                or not torch.equal(noise_rng, noise.get_state()) or second["generated"] != 0 or second["online_hits"] != 1):
            raise ValueError("Online cache reuse perturbs student RNG or repeats inference")
        online_report = dict(first=first, second=second, pilot_targets_exact=True, student_rng_unchanged=True)
    if rank % 2:
        rows = [(r[0], r[1], None) for r in rows]
    update(ddp, optimizer, scheduler, build_micros(model, rows, noise, 1), settings, world, 0)
    state.update(step=1, seen_records=world, next_batch=1)
    checkpoint = save_checkpoint(Path(plan["output_dir"]), model, optimizer, scheduler, noise, state, provenance, rank, world)
    for block in model.backbone.transformer.blocks:
        block.sdpa_backend = "math"
    expected_inputs = build_micros(model, rows, noise, 1)
    update(ddp, optimizer, scheduler, expected_inputs, settings, world, 1, precision="fp32")
    expected = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    moments = {n: {k: v.detach().cpu().clone() for k, v in s.items()} for n, s in optimizer.state_dict()["state"].items()}
    load_checkpoint(checkpoint, model, optimizer, scheduler, noise, provenance, rank)
    actual_inputs = build_micros(model, rows, noise, 1)
    if any(not torch.equal(v, actual_inputs[0][0][k]) for k, v in expected_inputs[0][0].items()):
        raise ValueError("Restored corruption differs")
    update(ddp, optimizer, scheduler, actual_inputs, settings, world, 1, precision="fp32")
    for name, value in model.named_parameters():
        torch.testing.assert_close(value.detach().cpu(), expected[name], atol=1e-8, rtol=1e-6)
    differences = optimizer_update_difference(moments, optimizer.state_dict()["state"])
    if any(v["relative_l2"] > 1e-5 for v in differences.values()):
        raise ValueError("Restored Adam update differs")
    write_json(Path(plan["output_dir"])/f"smoke_rank{rank}.json", dict(status="passed", optimizer=differences,
        corruption_exact=True, kd_present=rank % 2 == 0, online_cache=online_report))
    torch.distributed.barrier()
    rank_zero(lambda: write_json(Path(plan["output_dir"])/"COMPLETE.json", dict(status="passed",
        world_size=world, qualification=plan["qualification"], scope="engineering_resume_only")), rank)


def main(plan, device, rank, world, resume=None, online=None):
    settings, out = plan["training"], Path(plan["output_dir"])
    rank_zero(lambda: out.mkdir(parents=True, exist_ok=bool(resume)), rank)
    seed = settings["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    manifest = json.loads(Path(plan["cache_manifest"]).read_text())
    model = new_model(plan, device).train()
    ddp = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False, find_unused_parameters=plan["pair"]["feedback"] != "all", gradient_as_bucket_view=True)
    optimizer = make_optimizer(model, settings["backbone_lr"], settings["pair_lr"], settings["weight_decay"])
    scheduler = scheduler_for(optimizer, settings)
    noise = torch.Generator().manual_seed(seed+10000+rank)
    provenance = {k: plan[k] for k in ("model", "pair", "training", "sources", "source_root", "data_index",
        "data_index_sha256", "parent_checkpoint", "parent_model_sha256", "cache_manifest", "cache_manifest_sha256",
        "teacher_contract", "quality", "qualification")}
    provenance.update(schema="immune.stage1.training.v1", world_size=world, recipe=RECIPE,
                      torch=torch.__version__, normalization="global_tokens_and_eligible_records")
    if online is not None:
        provenance.update(online_teacher=plan["online_teacher"], teacher_quality_policy=plan["teacher_quality_policy"],
                          online_launch_sha256=plan["online_launch_sha256"])
    state = dict(step=0, epoch=0, next_batch=0, seen_records=0, best_score=1e30, best_step=0)
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, noise, provenance, rank)
    rank_zero(lambda: write_json(out/"run_provenance.json", provenance), rank)
    if plan["phase"] == "smoke":
        if resume:
            raise ValueError("Smoke owns its internal replay; do not resume it externally")
        return smoke_replay(plan, ddp, optimizer, scheduler, noise, provenance, state, manifest, device, rank, world, online)
    dataset = CachedCorpus(plan["data_index"], manifest)
    index = json.loads(Path(plan["data_index"]).read_text())
    if state["step"]:
        expected = step_budget(len(dataset), settings["global_batch"], state["step"])
        if (state["epoch"] != expected["completed_epochs"] or state["next_batch"] != expected["next_batch"]
                or state["seen_records"] != expected["seen_records"]):
            raise ValueError("Resume corpus cursor differs from completed optimizer steps")
    if state["step"] >= settings["max_steps"]:
        raise ValueError("Resume already reached the frozen budget")

    def validate():
        rng = capture_rng(noise)
        try:
            mlm = token_validation(model, index, device, rank, world, batch_size=1)
            structure = structure_validation(model, plan, manifest, device, rank, world)
            if mlm["all"]["loss"] < state["best_score"]:
                state.update(best_score=mlm["all"]["loss"], best_step=state["step"])
            rank_zero(lambda: write_json(out/f"validation_step{state['step']}.json", dict(mlm=mlm, structure=structure)), rank)
        finally:
            restore_rng(rng, noise)

    if state["step"] == 0:
        validate()
    stopped = False
    while state["step"] < settings["max_steps"] and not stopped:
        sampler = CompleteEpochBatchSampler(len(dataset), settings["global_batch"], rank, world,
            seed, state["epoch"], state["next_batch"], settings["max_steps"]-state["step"])
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=identity_batch,
            num_workers=settings["loader_workers"], generator=torch.Generator().manual_seed(seed+rank+state["epoch"]),
            **({"multiprocessing_context": "spawn"} if online is not None and settings["loader_workers"] else {}))
        for rows in loader:
            started = time.monotonic()
            online_metrics = None
            if online is not None:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                rows, online_metrics = online.resolve(rows, model.config)
            metrics = update(ddp, optimizer, scheduler, build_micros(model, rows, noise, settings["micro_batch"]),
                             settings, world, state["step"])
            if online_metrics is not None:
                keys = tuple(online_metrics)
                counts = torch.tensor([online_metrics[k] for k in keys], dtype=torch.float64, device=device)
                torch.distributed.all_reduce(counts)
                metrics["online_teacher"] = dict(zip(keys, counts.tolist()))
            state.update(step=state["step"]+1, next_batch=state["next_batch"]+1,
                         seen_records=state["seen_records"]+metrics["records"])
            if state["next_batch"] == math.ceil(len(dataset)/settings["global_batch"]):
                state.update(epoch=state["epoch"]+1, next_batch=0)
            stopped = rank_zero(lambda: (out/"STOP").exists(), rank)
            finished = state["step"] == settings["max_steps"]
            if rank == 0 and (state["step"] <= 3 or state["step"] % settings["log_every"] == 0 or finished):
                values = dict(step=state["step"], seconds=time.monotonic()-started, **metrics)
                with (out/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(values, allow_nan=False)+"\n")
                write_json(out/"status.json", dict(status="training", **state, **metrics, time_unix=time.time()))
                print(json.dumps(values), flush=True)
            if state["step"] % settings["eval_every"] == 0 or finished or stopped:
                validate()
            if state["step"] % settings["save_every"] == 0 or finished or stopped:
                # Keep interrupted staging directories for diagnosis; never treat them as checkpoints.
                def retire_partial():
                    partial = out/f".checkpoint-{state['step']}.tmp"
                    if partial.exists():
                        partial.rename(out/f".abandoned-{state['step']}-{time.time_ns()}")
                rank_zero(retire_partial, rank)
                save_checkpoint(out, model, optimizer, scheduler, noise, state, provenance, rank, world)
            if stopped or finished:
                break
    rank_zero(lambda: write_json(out/("STOPPED.json" if stopped else "COMPLETE.json"),
        dict(status="stopped" if stopped else "success", **state, world_size=world)), rank)


def profile_attempt(plan, device, micro, online=None):
    from scripts.research import profile_bert_mlm as base
    base.BertMLMCollator = FullTokenMLMCollator
    free, total_bytes = torch.cuda.mem_get_info(device)
    external_bytes = total_bytes - free - torch.cuda.memory_reserved(device)
    model = new_model(plan, device).train()
    settings = plan["training"]
    optimizer = make_optimizer(model, settings["backbone_lr"], settings["pair_lr"], settings["weight_decay"])
    scheduler = scheduler_for(optimizer, settings)
    clean, tokenizer = base.synthetic_batch(model.config, micro, model.config.max_length)
    noisy = corrupt(clean, tokenizer, torch.Generator().manual_seed(42))
    positions = clean["residue_mask"][0].nonzero().flatten()
    candidates = torch.combinations(positions, r=2)
    pair = candidates[torch.randperm(len(candidates), generator=torch.Generator().manual_seed(5))[:plan["max_pairs_per_record"]]]
    if len(pair) != plan["max_pairs_per_record"]:
        raise ValueError("Allocation profile cannot cover the requested pair ceiling")
    indices = torch.cat([torch.cat([torch.full((len(pair), 1), r), pair], 1) for r in range(micro)])
    teacher = torch.zeros(len(indices), plan["pair"]["distogram_bins"])
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    # Independent per-rank attempts avoid stranding peers in collectives on OOM.
    # DDP synchronization is qualified separately by the eight-rank smoke phase.
    from scripts.research.run_stage1_distillation import optimizer_step
    moved = [({k:v.to(device) for k,v in noisy.items()}, indices.to(device), teacher.to(device), None)]
    for _ in range(3):
        optimizer_step(model, optimizer, moved, precision=settings["precision"],
            intra_weight=settings["intra_weight"], inter_weight=settings["inter_weight"], temperature=settings["temperature"])
        scheduler.step()
    torch.cuda.synchronize(device)
    allocated, reserved = torch.cuda.max_memory_allocated(device)/2**30, torch.cuda.max_memory_reserved(device)/2**30
    total = torch.cuda.get_device_properties(device).total_memory/2**30
    result = dict(status="passed" if reserved <= .85*total else "insufficient_headroom",
        micro_batch=micro, allocated_gib=allocated, reserved_gib=reserved, total_gib=total,
        seconds_per_step=(time.monotonic()-started)/3, optimizer_steps=3)
    if online is not None:
        student_joint_peak = reserved + external_bytes/2**30
        result.update(student_joint_peak_estimate_gib=student_joint_peak)
        if student_joint_peak > .85*total:
            result["status"] = "insufficient_headroom"
            return result
        # Run the actual teacher with the trained student's model + Adam state
        # still resident; no training activations or allocator slack remain.
        optimizer.zero_grad(set_to_none=True)
        del moved
        gc.collect(); torch.cuda.empty_cache()
        teacher_memory = online.call(dict(op="profile"))
        joint_peak = max(student_joint_peak, teacher_memory["joint_peak_reserved_estimate_gib"])
        result.update(teacher=teacher_memory, joint_peak_estimate_gib=joint_peak,
            status="passed" if joint_peak <= .85*total else "insufficient_headroom")
    return result


def profile(plan, device, rank, world, online=None):
    out = Path(plan["output_dir"])
    rank_zero(lambda: out.mkdir(parents=True, exist_ok=False), rank)
    attempts, selected = [], None
    for micro in ((24, 16, 12, 8, 4, 2, 1) if online is not None else (64, 48, 32, 24, 16, 12, 8, 4, 2, 1)):
        try:
            result = profile_attempt(plan, device, micro, online)
        except torch.cuda.OutOfMemoryError:
            result = dict(status="oom", micro_batch=micro)
        gc.collect(); torch.cuda.empty_cache()
        gathered = [None]*world
        torch.distributed.all_gather_object(gathered, result)
        attempts.append(gathered)
        if rank == 0:
            print(json.dumps(dict(event="stage1_profile", micro_batch=micro, ranks=gathered)), flush=True)
        if all(x["status"] == "passed" for x in gathered):
            selected = micro
            break
    if selected is None:
        raise RuntimeError("No safe Stage 1 microbatch")
    rank_zero(lambda: write_json(out/"COMPLETE.json", dict(status="passed", world_size=world,
        qualification=plan["qualification"], micro_batch=selected, max_length=plan["model"]["max_length"],
        max_pairs_per_record=plan["max_pairs_per_record"], attempts=attempts,
        gpu=torch.cuda.get_device_name(device), scope="online_teacher_and_student" if online is not None
        else "allocation_only; no biological targets")), rank)
