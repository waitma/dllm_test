"""Train AB500K with memory-only Protenix supervision; activate pllm first.

Use the frozen launch/train_mlm.sh or train_pair_kd.sh in the generated run.
--verify-only is CPU-only. --qualify-only runs joint GPU profiling and complete
student-checkpoint replay without starting the experiment. No teacher targets
or ESM embeddings are written to disk, including during validation/replay.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from datetime import timedelta
import fcntl
import gc
import importlib.metadata
import json
import math
import os
import random
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import RECIPE
from scripts.research.bert_mlm_data import CompleteEpochBatchSampler, sha, write_json, step_budget
from scripts.research.run_bert_mlm import (rank_zero, save_checkpoint, load_checkpoint,
    capture_rng, restore_rng, optimizer_update_difference)
from scripts.research.run_stage1_training import scheduler_for
from scripts.research.run_stage1_distillation import autocast_context
from scripts.research.run_stage1_ab500k import (
    DEFAULTS, VALID_SEEDS, NOISE_RECIPE, PilotDataset, new_model, new_optimizer,
    distributed_model, training_loader, training_phase, record_training_metrics,
    training_noise_seed, retire_partial, validate_resume_cursor, recover_completion,
    reference)
from scripts.research.stage1_ab500k_initialization import verify_scratch_initialization
from scripts.research.stage1_protenix_online import (
    OnlineProtenixTeacher, build_micros, prepare_micro, optimizer_update, chain_aware_kd)


SCHEMA = "immune.stage1.protenix_online.v1"


def verify_plan(path, arm="pair_kd"):
    plan = json.loads(Path(path).read_text())
    if (plan.get("schema") != SCHEMA or plan.get("teacher_target_storage") != "none"
            or any(plan.get(k) is not None for k in ("cache_root", "cache_manifest"))):
        raise ValueError("Use a memory-only online plan, not an offline or write-through-cache plan")
    if arm not in {"mlm", "pair_kd"} or plan.get("world_size") != 8:
        raise ValueError("Expected the frozen eight-rank AB500K experiment")
    if Path(plan["source_root"]).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("Execute this plan's frozen source")
    for path, expected in plan["sources"].items():
        if sha(path) != expected:
            raise ValueError("Frozen source changed: " + path)
    for path, expected in plan["teacher_assets"].items():
        if arm == "pair_kd" and sha(path) != expected:
            raise ValueError("Frozen teacher asset changed: " + path)
    for key in ("subset_manifest", "subset_validation", "data_index"):
        if sha(plan[key]) != plan[key+"_sha256"]:
            raise ValueError("Frozen data binding changed: " + key)
    subset = json.loads(Path(plan["subset_manifest"]).read_text())
    receipt = json.loads(Path(plan["subset_validation"]).read_text())
    if (receipt.get("status") != "passed" or receipt.get("manifest_sha256") != sha(plan["subset_manifest"])
            or subset["counts"] != dict(train=500000, valid=4096)
            or subset["data_index"] != plan["data_index"]):
        raise ValueError("Invalid AB500K data qualification")
    for split in ("train", "valid"):
        binding = reference(subset, split+"_rows")
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Selected row identities changed")
    # Full prepared-source fingerprints were recorded by the completed subset
    # audit; bind them at entry, not only the byte-offset index files.
    for item in subset["verified_sources"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Prepared input sequences changed: " + item["path"])
    settings = {**DEFAULTS, **plan["training"]}
    if (settings["global_batch"] != 512 or settings["max_steps"] <= settings["warmup_steps"]
            or min(settings["intra_weight"], settings["inter_weight"]) < 0
            or settings["precision"] != "bf16" or settings["loader_workers"] < 0):
        raise ValueError("Invalid online training recipe")
    verify_scratch_initialization(plan)
    return dict(status="passed", pending=[], plan=plan, subset=subset, settings=settings)


def profile_attempt(plan, arm, device, rows, micro, teacher):
    model = new_model(plan, arm, device)
    optimizer = new_optimizer(model, plan["training"], arm)
    scheduler = scheduler_for(optimizer, plan["training"])
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for step in range(3):
        # Exercise maximum total length, individual-chain length and atom count.
        row = rows[step % len(rows)]
        inputs = build_micros(model, [row]*micro, micro, 10042, arm)
        optimizer_update(model, optimizer, scheduler, inputs, plan["training"], arm,
                         step, teacher=teacher)
    torch.cuda.synchronize(device)
    total = torch.cuda.get_device_properties(device).total_memory/2**30
    reserved = torch.cuda.max_memory_reserved(device)/2**30
    return dict(status="passed" if reserved <= .85*total else "insufficient_headroom",
        micro_batch=micro, reserved_gib=reserved, total_gib=total,
        allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        seconds_per_micro_update=(time.monotonic()-started)/3,
        sequence_length=inputs[0][0]["input_ids"].shape[1], optimizer_steps=3,
        teacher_resident=teacher is not None, teacher_target_disk_bytes=0)


def profile(plan, arm, dataset, out, device, rank, world, contract, teacher):
    path = out/"profile.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result.get("contract") != contract or result.get("status") != "passed":
            raise ValueError("Joint profile belongs to another experiment or GPU")
        return result
    rows = [dataset[i] for i in plan["profile_train_ordinals"]]
    started, attempts = time.monotonic(), []
    for micro in (64, 48, 32, 24, 16, 12, 8, 4, 2, 1):
        try:
            result = profile_attempt(plan, arm, device, rows, micro, teacher)
        except torch.cuda.OutOfMemoryError:
            result = dict(status="oom", micro_batch=micro)
        gc.collect()
        torch.cuda.empty_cache()
        gathered = [None]*world
        torch.distributed.all_gather_object(gathered, result)
        attempts.append(gathered)
        if rank == 0:
            print(json.dumps(dict(event="online_joint_profile", arm=arm, ranks=gathered)), flush=True)
        if all(x["status"] == "passed" for x in gathered):
            if result["sequence_length"] != plan["student_profile_length"]:
                raise ValueError("Student fixed-slot profile shape changed")
            result = dict(status="passed", contract=contract, micro_batch=micro, attempts=attempts,
                seconds=time.monotonic()-started, sequence_length=result["sequence_length"],
                world_size=world, global_batch=512, teacher_target_disk_bytes=0,
                scope="joint_teacher_student_three_updates; longest_complex_chain_and_largest_atom_count",
                micro_cap_reason="global512/world8=64; smaller candidates use accumulation")
            rank_zero(lambda: write_json(path, result), rank)
            return result
    raise RuntimeError("No jointly safe microbatch")


@torch.no_grad()
def validate(model, dataset, settings, arm, device, rank, world, micro, output, step, teacher):
    previous = model.training
    model.eval()
    local = []
    try:
        indices = list(range(rank, len(dataset), world))
        for start in range(0, len(indices), micro):
            ordinals = indices[start:start+micro]
            rows = [dataset[i] for i in ordinals]
            statistics = torch.zeros((len(rows), 12), dtype=torch.float64, device=device)
            _, cpu_pairs, requests = prepare_micro(model, rows, VALID_SEEDS[0], arm == "pair_kd")
            pairs = cpu_pairs.to(device)
            if arm == "pair_kd":
                targets, confidence = teacher.targets(requests)
            # Reuse a single micro's teacher tensors for the three fixed masks.
            # They are released before starting the next micro or validation.
            for seed in VALID_SEEDS:
                raw, _, _ = prepare_micro(model, rows, seed, False)
                batch = {k: v.to(device) for k, v in raw.items()}
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
                        use = pairs[:, 0].eq(row)
                        kd_args = (result["pair_logits"][use], targets[use], pairs[use], batch["chain_ids"],
                                   batch["noise_kind"], settings["temperature"])
                        raw_kd = chain_aware_kd(*kd_args)
                        kd = chain_aware_kd(*kd_args, pair_confidence=confidence[use]) if settings["confidence_weighting"] else raw_kd
                        for column, name in ((6, "intra"), (8, "inter")):
                            statistics[row, column] += kd[name+"_sum"]
                            statistics[row, column+1] += kd[name+"_count"]
                        statistics[row, 10] += raw_kd["intra_sum"]
                        statistics[row, 11] += raw_kd["inter_sum"]
                        del kd_args, kd, raw_kd
                del result, batch, logits
            local.extend((i, int(rows[j][0]), statistics[j].cpu().numpy()) for j, i in enumerate(ordinals))
            if arm == "pair_kd":
                del targets, confidence
        gathered = [None]*world
        torch.distributed.all_gather_object(gathered, local)
        ordered = sorted((x for group in gathered for x in group), key=lambda x: x[0])
        if [x[0] for x in ordered] != list(range(len(dataset))):
            raise ValueError("Validation rank coverage differs")
        arrays = np.stack([x[2] for x in ordered])
        sums = arrays.sum(0)
        metrics = dict(step=step, records=len(ordered), masks=3, seeds=list(VALID_SEEDS),
            full_token_ce=float(sums[0]/max(1, sums[1])), aa_ce=float(sums[2]/max(1, sums[3])),
            mask_accuracy=float(sums[4]/max(1, sums[5])),
            intra_kl=float(sums[6]/max(1, sums[7])) if arm == "pair_kd" else None,
            inter_kl=float(sums[8]/max(1, sums[9])) if arm == "pair_kd" else None,
            raw_intra_kl=float(sums[10]/max(1, sums[7])) if arm == "pair_kd" else None,
            raw_inter_kl=float(sums[11]/max(1, sums[9])) if arm == "pair_kd" else None,
            selected_tokens=int(sums[1]), aa_tokens=int(sums[3]), teacher_target_disk_bytes=0,
            record_statistics_columns=["full_sum", "full_count", "aa_sum", "aa_count", "mask_correct", "mask_count",
                                       "intra_sum", "intra_count", "inter_sum", "inter_count", "raw_intra_sum", "raw_inter_sum"])
        if rank == 0 and output is not None:
            # Sufficient statistics for comparing validation CE, not teacher
            # features, coordinates, confidence arrays, or distogram targets.
            target = Path(output)/f"validation_step{step}.records.npz"
            with target.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream, row_ids=np.array([x[1] for x in ordered]), statistics=arrays)
            target.with_suffix(".tmp").replace(target)
            metrics["record_statistics"] = dict(path=str(target), sha256=sha(target))
            write_json(Path(output)/f"validation_step{step}.json", metrics)
        return metrics
    finally:
        model.train(previous)


def replay_smoke(plan, arm, dataset, out, device, rank, world, provenance, teacher):
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
    ddp = distributed_model(model, device)
    optimizer = new_optimizer(model, plan["training"], arm)
    scheduler = scheduler_for(optimizer, plan["training"])
    noise = torch.Generator().manual_seed(42+rank)
    rows = [dataset[rank]]
    inputs = build_micros(model, rows, 1, 10042, arm)
    optimizer_update(ddp, optimizer, scheduler, inputs, plan["training"], arm, 0, world, teacher=teacher)
    state = dict(step=1, epoch=0, next_batch=1, seen_records=world, best_step=1, best_score=1e30)
    checkpoint = save_checkpoint(smoke_out, model, optimizer, scheduler, noise, state, provenance, rank, world)
    expected_inputs = build_micros(model, rows, 1, 10043, arm)
    expected_targets = []
    optimizer_update(ddp, optimizer, scheduler, expected_inputs, plan["training"], arm, 1, world, teacher=teacher, target_digests=expected_targets)
    expected = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    moments = {key: {name: value.detach().cpu().clone() for name, value in item.items()}
               for key, item in optimizer.state_dict()["state"].items()}
    restored = load_checkpoint(checkpoint, model, optimizer, scheduler, noise, provenance, rank)
    if restored != state:
        raise ValueError("Smoke data cursor differs")
    actual_inputs = build_micros(model, rows, 1, 10043, arm)
    if any(not torch.equal(value, actual_inputs[0][0][key]) for key, value in expected_inputs[0][0].items()):
        raise ValueError("Smoke corruption replay differs")
    actual_targets = []
    optimizer_update(ddp, optimizer, scheduler, actual_inputs, plan["training"], arm, 1, world, teacher=teacher, target_digests=actual_targets)
    if actual_targets != expected_targets:
        raise ValueError("Online regenerated targets differ after resume")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), expected[key], atol=1e-8, rtol=1e-6)
    differences = optimizer_update_difference(moments, optimizer.state_dict()["state"])
    if any(item["relative_l2"] > 1e-5 for item in differences.values()):
        raise ValueError("Smoke AdamW replay differs")
    write_json(out/f"smoke_rank{rank}.json", dict(status="passed", optimizer=differences, corruption_exact=True, teacher_targets_exact=actual_targets == expected_targets, teacher_target_disk_bytes=0))
    torch.distributed.barrier()
    receipt = dict(status="passed", contract=provenance, seconds=time.monotonic()-started,
                   world_size=world, scope="joint_online_complete_state_replay_actual_precision_and_sdpa; targets_regenerated")
    rank_zero(lambda: write_json(receipt_path, receipt), rank)
    return receipt


def run(plan_path, arm, resume, qualify_only=False):
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
            raise ValueError("Online input verification incomplete: " + str(checked["pending"]))
        plan, subset, settings = (checked[key] for key in ("plan", "subset", "settings"))
        plan["training"] = settings
        plan["student_profile_length"] = subset["student_max_length"]
        out = Path(plan["output_root"])/arm
        def acquire():
            nonlocal lock
            out.mkdir(parents=True, exist_ok=True)
            lock = (out/"run.lock").open("a+")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        rank_zero(acquire, rank)
        dataset = PilotDataset(subset, None, targets=False)
        validation = PilotDataset(subset, None, "valid", targets=False)
        parent_count = json.loads(Path(subset["data_index"]).read_text())["records"]["train"]
        teacher = (OnlineProtenixTeacher(device, base_seed=settings["seed"],
            confidence_threshold=plan["teacher_contract"]["confidence_threshold_A"]) if arm == "pair_kd" else None)
        if teacher is not None:
            if any(teacher.contract.get(k) != v for k, v in plan["teacher_contract"].items()):
                raise ValueError("Actual teacher differs from frozen plan")
        contract = dict(schema=SCHEMA, arm=arm, plan_sha256=sha(plan_path),
            subset_manifest_sha256=sha(plan["subset_manifest"]), teacher_target_storage="none", teacher_contract=teacher.contract if teacher else None,
            teacher_assets=plan["teacher_assets"], gpu_name=torch.cuda.get_device_name(device),
            gpu_memory=torch.cuda.get_device_properties(device).total_memory,
            parent_model_sha256=plan["parent_model_sha256"], initialization=plan.get("initialization"),
            training=settings, pair=plan.get("pair", {}),
            noise_recipe=NOISE_RECIPE, noise_parent_records=parent_count, recipe=RECIPE,
            validation_seeds=list(VALID_SEEDS), teacher_quality="unvalidated", world_size=world,
            torch=torch.__version__, source_root=plan["source_root"], sources=plan["sources"])
        contract["runtime"] = {name: importlib.metadata.version(name) for name in
            ("transformers", "numpy", "safetensors", "biotite", "rdkit", "fair-esm")}
        profile_receipt = profile(plan, arm, dataset, out, device, rank, world, contract, teacher)
        micro = profile_receipt["micro_batch"]
        provenance = {**contract, "micro_batch": micro, "profile_sha256": sha(out/"profile.json")}
        smoke = replay_smoke(plan, arm, dataset, out, device, rank, world, provenance, teacher)
        if qualify_only:
            return
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
            timing_note="worker GPU hours include resident teacher and student; excludes launch/preflight and teardown")
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
            state["worker_gpu_hours"] = world*state["wall_seconds"]/3600
            state["worker_with_qualification_gpu_hours"] = world*(state["wall_seconds"]+profile_receipt["seconds"]+smoke["seconds"])/3600
        def evaluate():
            began = time.monotonic()
            rng = capture_rng(noise)
            try:
                result = validate(model, validation, settings, arm, device, rank, world, micro, out, state["step"], teacher)
                if result["aa_ce"] < state["best_score"]:
                    state.update(best_score=result["aa_ce"], best_step=state["step"])
            finally:
                restore_rng(rng, noise)
            state["validation_seconds"] += time.monotonic()-began
            stamp()
            result.update(seen_records=state["seen_records"], wall_seconds=state["wall_seconds"],
                          worker_gpu_hours=state["worker_gpu_hours"])
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
                    training_phase(out, arm, rank, "online_teacher_then_optimizer_update", state["step"])
                scores = optimizer_update(ddp, optimizer, scheduler, micros, settings, arm, state["step"], world, teacher=teacher)
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
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_plan(args.plan, args.arm)
        print(json.dumps({key: value for key, value in result.items() if key not in {"plan", "subset", "cache", "settings"}}))
        return 0 if result["status"] == "passed" else 2
    run(args.plan, args.arm, args.resume, args.qualify_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
