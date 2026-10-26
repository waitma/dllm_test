"""Profile, verify resume, or train the conditional diffusion/edit continuation.

Activate pllm; execute only via a frozen launch_diffusion_continuation package
with eight torchrun ranks. New stages start from BERT; recorded extensions restore
the full completed diffusion state. STOP saves a checkpoint before a clean exit.
"""
from contextlib import nullcontext
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from scripts.research.bert_mlm_data import IndexedPreparedDataset, sha, write_json
from scripts.research.run_bert_mlm import (rank_zero, data_gate, save_checkpoint, load_checkpoint,
    capture_rng, restore_rng, parameter_digest, optimizer_update_difference)
from scripts.research.diffusion_continuation_data import BalancedSourceBatchSampler, source_lengths, warmup_cosine_scheduler
from scripts.research.diffusion_continuation_objective import DiffusionEditModel, DiffusionEditCollator, corrupt
from scripts.research.diffusion_continuation_eval import evaluate


def update(ddp, batch, optimizer, scheduler, micro, world, bf16=True, gradient_clip=1.):
    """Average per-view objectives once across all ranks and all microbatches."""
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("gradient_clip must be finite and positive")
    device = next(ddp.parameters()).device
    count = torch.tensor(len(batch["input_ids"]), device=device, dtype=torch.long)
    torch.distributed.all_reduce(count)
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    scope_losses = torch.zeros(4, device=device, dtype=torch.float64)
    names = ("loss_sum", "ce_sum", "eos_ce_sum", "tokens", "eos_tokens")
    for offset in range(0, len(batch["input_ids"]), micro):
        last = offset + micro >= len(batch["input_ids"])
        with nullcontext() if last else ddp.no_sync():
            part = {key: value[offset:offset+micro].to(device) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                result = ddp(**part)
                loss = result["loss_sum"] * world / count
            loss.backward()
            totals += torch.stack([result[key].detach().double() for key in names])
            if "mask_scope" in part:
                scope, edited = part["mask_scope"].flatten(), part["edit_task"].flatten()
                masks = (edited, scope.eq(0) & ~edited, scope.eq(1), scope.eq(2))
                scope_losses += torch.stack([result["view_loss_sums"][m].double().sum() for m in masks])
            del part, result, loss
    parameters = list(ddp.parameters())
    norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
    # clip_grad_norm_ returns the PRE-clip norm. Measure the applied gradient too.
    post_norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(p.grad.detach().float()) for p in parameters if p.grad is not None]))
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    torch.distributed.all_reduce(totals)
    cdr_metrics = {}
    if "mask_scope" in batch:
        scopes = batch["mask_scope"].flatten()
        edits = batch["edit_task"].flatten()
        counts = torch.tensor([int(edits.sum()), int((scopes.eq(0) & ~edits).sum()),
            int(scopes.eq(1).sum()), int(scopes.eq(2).sum()),
            int(batch["cdr_full_mask"].sum()), int((batch["cdr_eligible"].flatten() & ~edits).sum())],
            device=device, dtype=torch.long)
        torch.distributed.all_reduce(counts)
        torch.distributed.all_reduce(scope_losses)
        cdr_metrics = {"task_views_"+name: int(n) for name, n in zip(
            ("edit", "chain_mask", "cdr_single", "cdr_multi", "cdr_full_mask", "cdr_eligible_mask"), counts)}
        cdr_metrics.update({"task_loss_"+name: float(loss/n) for name, loss, n in zip(
            ("edit", "chain_mask", "cdr_single", "cdr_multi"), scope_losses, counts[:4]) if n > 0})
    return dict(loss=float(totals[0]/count), unweighted_ce=float(totals[1]/totals[3].clamp_min(1)),
                eos_ce=float(totals[2]/totals[4].clamp_min(1)), selected_tokens=int(totals[3]),
                record_views=int(count), grad_norm=float(norm), learning_rate=lr,
                grad_norm_pre_clip=float(norm), grad_norm_post_clip=float(post_norm),
                grad_clip_coefficient=float((gradient_clip / (norm + 1e-6)).clamp(max=1.)),
                grad_clipped=bool(norm > gradient_clip), **cdr_metrics)


def verify_parent(plan):
    parent = Path(plan["parent_checkpoint"])
    receipt = json.loads((parent / "checkpoint_receipt.json").read_text())
    if receipt["status"] != "complete" or receipt["files"]["model.safetensors"] != plan["parent_model_sha256"]:
        raise ValueError("Parent checkpoint identity changed")
    for name in ("model.safetensors", "research_config.json", "immune_config.json", "vocabulary.json", "state.json", "run_provenance.json"):
        if sha(parent / name) != receipt["files"][name]:
            raise ValueError("Parent checkpoint file changed: " + name)
    if len(set(receipt["rank_parameter_sha256"])) != 1 or receipt["step"] != 287000:
        raise ValueError("Parent rank parameters/step differ")
    return dict(status="passed", step=receipt["step"], model_sha256=plan["parent_model_sha256"])


def new_model(plan, device):
    model = DiffusionEditModel.from_parent(plan["parent_checkpoint"], plan["recipe"]).to(device).train()
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    return model


def profile(plan, device, rank, world):
    from scripts.research import profile_bert_mlm as original
    original.BertMLMCollator = DiffusionEditCollator
    attempts = []
    selected = None
    for micro in (64, 48, 32, 16):
        model = optimizer = batch = loss = None
        try:
            model = new_model(plan, device)
            raw, tok = original.synthetic_batch(model.config, micro//2, model.config.max_length)
            batch = {key: value.to(device) for key, value in corrupt(raw, tok,
                     torch.Generator().manual_seed(42), plan["recipe"]).items()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan["training"]["learning_rate"])
            torch.cuda.reset_peak_memory_stats(device)
            losses = []
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch)["loss_sum"] / micro
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            total = torch.cuda.get_device_properties(device).total_memory / 2**30
            allocated = torch.cuda.max_memory_allocated(device)/2**30
            reserved = torch.cuda.max_memory_reserved(device)/2**30
            report = dict(status="passed" if allocated < .88*total and reserved < .92*total else "insufficient_headroom",
                          micro_batch_views=micro, length=model.config.max_length, losses=losses,
                          allocated_gib=allocated, reserved_gib=reserved, total_gib=total)
        except torch.cuda.OutOfMemoryError:
            report = dict(status="oom", micro_batch_views=micro)
        finally:
            del model, optimizer, batch, loss
            gc.collect()
            torch.cuda.empty_cache()
        reports = [None]*world
        torch.distributed.all_gather_object(reports, report)
        attempts.append(reports)
        if rank == 0:
            write_json(Path(plan["output_dir"])/"profile_progress.json", dict(attempts=attempts))
        if all(row["status"] == "passed" for row in reports):
            selected = micro
            break
    if selected is None:
        raise RuntimeError("No safe microbatch at the actual maximum sequence length")
    rank_zero(lambda: write_json(Path(plan["output_dir"])/"COMPLETE.json", dict(
        status="passed", world_size=world, micro_batch_views=selected, attempts=attempts,
        recipe=plan["recipe"], model=plan["model"], max_length=plan["model"]["max_length"])), rank)


def wandb_start(plan, out, resume):
    import os
    import wandb
    spec = plan["wandb"]
    key = Path(spec["credential_file"]).read_text().strip()
    os.environ.update(WANDB_API_KEY=key, WANDB_MODE="online", WANDB_BASE_URL="https://api.wandb.ai")
    api = wandb.Api(api_key=key, timeout=30)
    viewer = api.viewer
    username = viewer.get("username") if isinstance(viewer,dict) else viewer.username
    if username != spec["username"]:
        raise ValueError("W&B account differs")
    run = wandb.init(entity=spec["entity"], project=spec["project"], id=spec["run_id"], name=spec["name"],
                     resume="must" if resume else "never", dir=str(out),
                     config={key: plan[key] for key in ("model","training","recipe","parent_model_sha256","data_index_sha256","extension","repair") if key in plan},
                     settings=wandb.Settings(init_timeout=120, save_code=False, disable_git=True))
    write_json(out/"wandb_run.json",dict(url=run.url,id=run.id,username=username))
    return run


def main(plan, device, rank, world):
    out = Path(plan["output_dir"])
    resume = plan.get("resume_checkpoint")
    rank_zero(lambda: out.mkdir(parents=True, exist_ok=bool(resume)), rank)
    parent_gate = rank_zero(lambda: verify_parent(plan), rank)
    if plan["phase"] == "profile":
        profile(plan,device,rank,world)
        return
    settings = plan["training"]
    gate = rank_zero(lambda: data_gate(plan),rank)
    index = json.loads(Path(plan["data_index"]).read_text())
    dataset = IndexedPreparedDataset(plan["data_index"])
    lengths = source_lengths(index)
    provenance = {key: plan[key] for key in ("model","training","recipe","source_root","sources",
                   "data_index","data_index_sha256","parent_checkpoint","parent_model_sha256")}
    provenance.update(schema="immune.diffusion_edit.training.v1", world_size=world,
                      normalization="global record-view mean; one original record has two complementary views")
    extension = plan.get("extension")
    if extension:
        provenance["extension"] = extension
    if plan.get("repair"):
        provenance["repair"] = plan["repair"]
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    model = new_model(plan,device)
    initial_digest = parameter_digest(model)
    expected_digest = json.loads((Path(plan["parent_checkpoint"])/"checkpoint_receipt.json").read_text())["rank_parameter_sha256"][rank]
    if initial_digest != expected_digest:
        raise ValueError("Initial parameters differ from pretrained BERT")
    ddp = DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False,
                                  find_unused_parameters=False,gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=.01,betas=(.9,.999),eps=1e-8)
    if extension:
        from scripts.research.diffusion_extension import extension_scheduler, restore_extension
        scheduler = extension_scheduler(optimizer,extension)
    else:
        scheduler = warmup_cosine_scheduler(optimizer,settings["max_steps"],settings["warmup_steps"])
    noise = torch.Generator().manual_seed(settings["seed"]+10000+rank)
    state = dict(step=0,best_step=0,best_score=1e30,seen_records=0,parent_step=287000)
    guard = settings.get("validation_lr_guard")
    if resume and guard:
        from scripts.research.diffusion_training_guard import ValidationLRCeiling
        scheduler = ValidationLRCeiling(scheduler, guard)
    if resume:
        state = load_checkpoint(resume,model,optimizer,scheduler,noise,provenance,rank)
    elif plan.get("repair"):
        from scripts.research.diffusion_guard_repair import restore_repair
        state = restore_repair(plan, provenance, model, optimizer, scheduler, noise, rank)
    elif extension:
        state = restore_extension(plan,provenance,model,optimizer,scheduler,noise,rank)
    if guard and not resume:
        from scripts.research.diffusion_training_guard import ValidationLRCeiling
        scheduler = ValidationLRCeiling(scheduler, guard, best=state["best_score"],
            initial_scale=plan.get("repair", {}).get("initial_lr_scale", 1.))
    starting_step = state["step"]
    from scripts.research.diffusion_source_sampling import make_sampler
    sampler = make_sampler(list(lengths.values()), settings, rank=rank, world=world,
                           start_step=state["step"])
    loader = DataLoader(dataset,batch_sampler=sampler,collate_fn=DiffusionEditCollator(model.config),
        num_workers=settings["loader_workers"],generator=torch.Generator().manual_seed(settings["seed"]+rank))
    micro = settings["micro_batch_views"]
    run = None
    if rank == 0:
        write_json(out/"run_provenance.json",provenance)
        write_json(out/"data_gate.json",gate)
        write_json(out/"parent_gate.json",dict(**parent_gate,initial_parameter_digest=initial_digest))
    if plan.get("wandb"):
        holder=[]
        rank_zero(lambda: holder.append(wandb_start(plan,out,resume)),rank)
        run = holder[0] if rank==0 else None

    def log(values):
        if rank==0:
            values={"step":state["step"],**values}
            with (out/"metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(values,allow_nan=False)+"\n")
            if run:
                allowed={"train/loss","train/learning_rate","train/grad_norm",
                         "train/grad_norm_pre_clip","train/grad_norm_post_clip",
                         "train/grad_clip_coefficient","train/grad_clipped",
                         "valid/next_learning_rate","valid/lr_reductions","valid/loss",
                         "valid/eos_loss","valid/first_eos_accuracy","valid/relation_loss",
                         "valid/edit_accuracy","valid/keep_error_rate"}
                run.log({k:v for k,v in values.items() if k=="step" or k in allowed
                         or k.startswith("train/task_views_") or k.startswith("train/task_loss_")
                         or k.startswith("valid/cdr_")
                         or (k.startswith("valid/source/") and k.endswith("/loss"))},step=state["step"])

    def validate():
        rng=capture_rng(noise)
        try:
            result=evaluate(model,index,device,rank,world,batch_size=8)
            if guard:
                scheduler.observe(result["summary"]["loss"], state["step"])
                result["lr_guard"] = dict(next_learning_rate=scheduler.get_last_lr(),
                    reductions=scheduler.reductions, bad_validations=scheduler.bad_validations)
            if result["summary"]["loss"]<state["best_score"]:
                state.update(best_score=result["summary"]["loss"],best_step=state["step"])
            if rank==0:
                write_json(out/f"validation_step{state['step']}.json",result)
            log({**{"valid/"+k:v for k,v in result["summary"].items()},
                 **({"valid/next_learning_rate": scheduler.get_last_lr()[0],
                     "valid/lr_reductions": scheduler.reductions} if guard else {}),
                 **{"valid/source/"+k+"/loss":v for k,v in result["source_losses"].items()}})
            return result
        finally:
            restore_rng(rng,noise)

    if plan["phase"] in ("smoke","extension_smoke"):
        iterator=iter(loader)
        raw=next(iterator)
        first=corrupt(raw,model.tokenizer,noise,model.recipe)
        metrics=update(ddp,first,optimizer,scheduler,micro,world,
                       gradient_clip=settings["gradient_clip"])
        state.update(step=state["step"]+1,seen_records=state["seen_records"]+settings["global_batch"])
        checkpoint=save_checkpoint(out,model,optimizer,scheduler,noise,state,provenance,rank,world)
        raw={k:v[:2] for k,v in next(iterator).items()}
        blocks=list(model.backbone.transformer.blocks)
        for block in blocks:
            block.sdpa_backend="math"
        load_checkpoint(checkpoint,model,optimizer,scheduler,noise,provenance,rank)
        expected=corrupt(raw,model.tokenizer,noise,model.recipe)
        update(ddp,expected,optimizer,scheduler,1,world,bf16=False,
               gradient_clip=settings["gradient_clip"])
        parameters={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
        moments={p:{k:v.detach().cpu().clone() for k,v in values.items()}
                 for p,values in optimizer.state_dict()["state"].items()}
        load_checkpoint(checkpoint,model,optimizer,scheduler,noise,provenance,rank)
        actual=corrupt(raw,model.tokenizer,noise,model.recipe)
        if any(not torch.equal(v,actual[k]) for k,v in expected.items()):
            raise ValueError("Corruption differs after restoring checkpoint RNG")
        update(ddp,actual,optimizer,scheduler,1,world,bf16=False,
               gradient_clip=settings["gradient_clip"])
        for name,value in model.named_parameters():
            torch.testing.assert_close(value.detach().cpu(),parameters[name],atol=1e-8,rtol=1e-6)
        difference=optimizer_update_difference(moments,optimizer.state_dict()["state"])
        if any(value["relative_l2"]>1e-5 for value in difference.values()):
            raise ValueError("Adam moments differ after replay")
        write_json(out/f"smoke_rank{rank}.json",dict(status="passed",rng_exact=True,
            parameters_restored=True,optimizer_replay=difference,first_update=metrics,
            initial_matches_parent=True,global_batch=settings["global_batch"],starting_step=starting_step))
        torch.distributed.barrier()
        rank_zero(lambda: write_json(out/"COMPLETE.json",dict(status="passed",world_size=world,
            recipe=plan["recipe"],training=settings,parent_model_sha256=plan["parent_model_sha256"],
            starting_step=starting_step,extension=extension)),rank)
        return

    if state["step"]==0 or ((extension or plan.get("repair")) and not resume):
        validate()
    stopped=False
    for raw in loader:
        started=time.monotonic()
        batch=corrupt(raw,model.tokenizer,noise,model.recipe)
        metrics=update(ddp,batch,optimizer,scheduler,micro,world,
                       gradient_clip=settings["gradient_clip"])
        state["step"]+=1
        state["seen_records"]+=settings["global_batch"]
        finished=state["step"]==settings["max_steps"]
        if state["step"]<=starting_step+3 or state["step"]%settings["log_every"]==0 or finished:
            stopped=rank_zero(lambda:(out/"STOP").exists(),rank)
            elapsed=time.monotonic()-started
            log({"train/"+k:v for k,v in metrics.items()})
            if rank==0:
                write_json(out/"status.json",dict(status="training",**state,target_steps=settings["max_steps"],
                    seconds_per_step=elapsed,time_unix=time.time(),world_size=world,**metrics,
                    source_exposures=dict(zip(lengths,sampler.exposures(state["step"])))))
        periodic=state["step"]%settings["eval_every"]==0
        if periodic or finished or stopped:
            validate()
        if periodic or finished or stopped or state["step"]==starting_step+25:
            save_checkpoint(out,model,optimizer,scheduler,noise,state,provenance,rank,world)
        if stopped:
            break
    rank_zero(lambda:write_json(out/("STOPPED.json" if stopped else "COMPLETE.json"),dict(
        status="stopped_by_user" if stopped else "success",world_size=world,**state,time_unix=time.time())),rank)
    rank_zero(lambda:write_json(out/"status.json",dict(status="stopped_by_user" if stopped else "success",
        **state,target_steps=settings["max_steps"],world_size=world,time_unix=time.time())),rank)
    if run:
        run.finish()
