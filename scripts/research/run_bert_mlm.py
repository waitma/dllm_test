"""Train fixed-chain BERT with exact step budgets, global token means and native DDP.

Run through the frozen launch_bert_mlm.py package. The smoke phase verifies
eight-rank save/resume equivalence before a separately initialized formal run.
"""
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.mlm import BertMLMCollator, BertMLMModel, RECIPE, corrupt
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from scripts.research.bert_mlm_data import (
    CompleteEpochBatchSampler, IndexedPreparedDataset, sha, step_budget, write_json)
from scripts.research.bert_mlm_eval import token_validation, representation_validation
from scripts.research.bert_mlm_objectives import components


def capture_rng(noise):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                corruption=noise.get_state())


def restore_rng(state, noise):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())
    noise.set_state(state["corruption"].cpu())


def make_scheduler(optimizer, total, warmup):
    def multiplier(step):
        if step < warmup:
            return step/max(1, warmup)
        progress = min(1., (step-warmup)/max(1, total-warmup))
        return .5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_update_difference(expected, actual):
    """Compare Adam moments after replay; a tiny warmup LR must not hide drift."""
    totals = {key: dict(squared_error=0., squared_reference=0., max_absolute=0.)
              for key in ("exp_avg", "exp_avg_sq")}
    if expected.keys() != actual.keys():
        raise ValueError("Replayed optimizer parameter set differs")
    for parameter, values in expected.items():
        if not torch.equal(values["step"].cpu(), actual[parameter]["step"].cpu()):
            raise ValueError("Replayed Adam step differs")
        for key, total in totals.items():
            reference = values[key].cpu().double()
            difference = actual[parameter][key].detach().cpu().double() - reference
            total["squared_error"] += float(difference.square().sum())
            total["squared_reference"] += float(reference.square().sum())
            total["max_absolute"] = max(total["max_absolute"], float(difference.abs().max()))
    return {key: dict(relative_l2=math.sqrt(x["squared_error"]/max(x["squared_reference"], 1e-300)),
                      max_absolute=x["max_absolute"]) for key, x in totals.items()}


def rank_zero(operation, rank):
    """Broadcast rank-zero errors so peers never continue after a failed gate."""
    packet = [None]
    if rank == 0:
        try:
            packet[0] = {"value": operation(), "error": None}
        except Exception as exc:
            packet[0] = {"value": None, "error": repr(exc)}
    torch.distributed.broadcast_object_list(packet, src=0)
    if packet[0]["error"]:
        raise RuntimeError(packet[0]["error"])
    return packet[0]["value"]


def data_gate(plan):
    if sha(plan["data_index"]) != plan["data_index_sha256"]:
        raise ValueError("Frozen data index changed")
    index = json.loads(Path(plan["data_index"]).read_text())
    if index.get("binding_kind") in ("immune.bert_mlm.v7_binding.v1", "immune.bert_mlm.v8_binding.v1"):
        if index["binding_kind"] == "immune.bert_mlm.v8_binding.v1":
            from scripts.research.bert_mlm_v8_data import verify_binding
        else:
            from scripts.research.bert_mlm_v7_data import verify_binding
        binding = verify_binding(index["binding"], index["binding_sha256"], plan["model"])
        expected = {x["path"]: x for x in binding["shards"]}
        if set(expected) != {x["path"] for x in index["shards"]}:
            raise ValueError("BERT index does not cover exactly the bound V7 shards")
        for shard in index["shards"]:
            reference = expected[shard["path"]]
            if any(shard[k] != reference[k] for k in ("sha256", "resolved_path", "records", "source", "split")):
                raise ValueError("V7 index shard differs from the accepted release")
    elif "binding_kind" not in index:
        from scripts.research.fixed_chain_data_binding import verify_binding
        verify_binding(Path(index["binding"]), index["binding_sha256"], Path(index["data_dir"]),
                       343, "fixed_chain_full_global", full_hash=False)
    else:
        raise ValueError("Unknown BERT data binding")
    if sha(index["manifest"]) != index["manifest_sha256"]:
        raise ValueError("Data manifest changed")
    def check(shard):
        if shard.get("resolved_path", str(Path(shard["path"]).resolve())) != str(Path(shard["path"]).resolve()):
            raise ValueError("Bound shard symlink target changed")
        if sha(shard["path"]) != shard["sha256"] or sha(shard["index"]) != shard["index_sha256"]:
            raise ValueError("Shard/index content changed: " + shard["path"])
        return shard["records"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(check, index["shards"]))
    for reference in index["files"].values():
        if sha(reference["path"]) != reference["sha256"]:
            raise ValueError("Development set changed")
    if any(sum(x["records"] for x in index["shards"] if x["split"] == s) != count
           for s, count in index["records"].items()):
        raise ValueError("Indexed split totals differ")
    return dict(status="passed", shards=len(counts), rows=sum(counts),
                train=index["records"]["train"], manifest_sha256=index["manifest_sha256"])


def save_checkpoint(output, model, optimizer, scheduler, noise, state, provenance, rank, world):
    checkpoint = output / ("checkpoint-" + str(state["step"]))
    temporary = output / (".checkpoint-" + str(state["step"]) + ".tmp")
    def start():
        temporary.mkdir(exist_ok=False)
        model.save_pretrained(temporary)
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        write_json(temporary / "state.json", state)
        write_json(temporary / "run_provenance.json", provenance)
    rank_zero(start, rank)
    torch.save(capture_rng(noise), temporary / f"rng_rank{rank}.pt")
    digests = [None]*world
    torch.distributed.all_gather_object(digests, parameter_digest(model))
    if len(set(digests)) != 1:
        raise RuntimeError("Model parameters diverged across ranks")
    def commit():
        required = [temporary / f"rng_rank{r}.pt" for r in range(world)]
        if not all(path.is_file() for path in required):
            raise ValueError("Missing rank RNG state")
        from safetensors.torch import load_file
        saved = load_file(str(temporary / "model.safetensors"))
        if any(not torch.equal(v.detach().cpu(), saved[k]) for k, v in model.state_dict().items()):
            raise ValueError("Saved model differs from rank parameters")
        files = {p.name: sha(p) for p in temporary.iterdir() if p.is_file()}
        write_json(temporary / "checkpoint_receipt.json", dict(status="complete", step=state["step"],
                   files=files, world_size=world, rank_parameter_sha256=digests))
        temporary.rename(checkpoint)
        write_json(output / "latest.json", dict(checkpoint=str(checkpoint), **state))
        # Only checkpoints created by this run are eligible for automatic retention.
        candidates = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        keep = set(candidates[-2:]) | {output / ("checkpoint-" + str(state["best_step"]))}
        for old in candidates:
            if old not in keep and (old / "checkpoint_receipt.json").exists():
                if json.loads((old / "run_provenance.json").read_text()) != provenance:
                    raise ValueError("Refusing to prune a checkpoint from another run")
                shutil.rmtree(old)
        return str(checkpoint)
    rank_zero(commit, rank)
    return checkpoint


def load_checkpoint(checkpoint, model, optimizer, scheduler, noise, provenance, rank):
    checkpoint = Path(checkpoint)
    receipt = json.loads((checkpoint / "checkpoint_receipt.json").read_text())
    def verify():
        if receipt["status"] != "complete" or any(sha(checkpoint/name) != value for name, value in receipt["files"].items()):
            raise ValueError("Incomplete or modified checkpoint")
        if json.loads((checkpoint / "run_provenance.json").read_text()) != provenance:
            raise ValueError("Resume recipe/code/data differs")
    rank_zero(verify, rank)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    if parameter_digest(model) != receipt["rank_parameter_sha256"][rank]:
        raise ValueError("Restored model tensors differ from the saved rank")
    # AdamW's non-capturable step counters stay on CPU. load_state_dict moves
    # moment tensors to each parameter's device; mapping every tensor to CUDA
    # changes bias-correction arithmetic after resume.
    saved_optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved_optimizer)
    restored_optimizer = optimizer.state_dict()
    if saved_optimizer["param_groups"] != restored_optimizer["param_groups"]:
        raise ValueError("Restored optimizer groups differ")
    for parameter, values in saved_optimizer["state"].items():
        for key, value in values.items():
            restored = restored_optimizer["state"][parameter][key]
            if isinstance(value, torch.Tensor) and not torch.equal(value.cpu(), restored.cpu()):
                raise ValueError("Restored AdamW tensor differs: " + key)
    saved_scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=False)
    scheduler.load_state_dict(saved_scheduler)
    if scheduler.state_dict() != saved_scheduler:
        raise ValueError("Restored scheduler differs")
    saved_rng = torch.load(checkpoint / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False)
    restore_rng(saved_rng, noise)
    restored_rng = capture_rng(noise)
    for key in ("cpu", "corruption", "cuda"):
        if saved_rng[key] is not None and not torch.equal(saved_rng[key], restored_rng[key]):
            raise ValueError("Restored rank RNG differs: " + key)
    state = json.loads((checkpoint / "state.json").read_text())
    if scheduler.last_epoch != state["step"]:
        raise ValueError("Restored scheduler cursor differs")
    return state


def update(ddp, batch, optimizer, scheduler, micro, world, *, bf16=True):
    """One optimizer step: normalize every microbatch by the same global count."""
    device = next(ddp.parameters()).device
    denominator = batch["labels"].ne(-100).sum().to(device)
    torch.distributed.all_reduce(denominator)
    if not denominator.item():
        raise ValueError("Optimizer step has no selected target on any rank")
    optimizer.zero_grad(set_to_none=True)
    local_sum = torch.zeros((), device=device, dtype=torch.float64)
    rows = len(batch["input_ids"])
    for offset in range(0, rows, micro):
        last = offset + micro >= rows
        with nullcontext() if last else ddp.no_sync():
            part = {k: v[offset:offset+micro].to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                result = ddp(**part)
                loss = result["loss_sum"] * world / denominator
            loss.backward()
            local_sum += result["loss_sum"].detach().double()
            del result, loss, part
    norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1., error_if_nonfinite=True)
    lr_used = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    torch.distributed.all_reduce(local_sum)
    return dict(loss=float(local_sum/denominator), selected_tokens=int(denominator),
                grad_norm=float(norm), learning_rate=lr_used)


def math_resume_oracle(checkpoint, ddp, optimizer, scheduler, noise, provenance, raw, micro, rank, world):
    """Independently replay saved state using FP32 and the math attention backend.

    This startup-only oracle separates serialization errors from BF16 fused
    backward variability. Formal training keeps its profiled efficient backend.
    """
    model = ddp.module
    blocks = list(model.backbone.transformer.blocks)
    policies = [block.sdpa_backend for block in blocks]
    try:
        for block in blocks:
            block.sdpa_backend = "math"
        load_checkpoint(checkpoint, model, optimizer, scheduler, noise, provenance, rank)
        expected_batch = corrupt(raw, model.tokenizer, noise)
        expected_metrics = update(ddp, expected_batch, optimizer, scheduler, micro, world, bf16=False)
        parameters = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
        moments = {p: {k: v.detach().cpu().clone() for k, v in values.items()}
                   for p, values in optimizer.state_dict()["state"].items()}
        load_checkpoint(checkpoint, model, optimizer, scheduler, noise, provenance, rank)
        actual_batch = corrupt(raw, model.tokenizer, noise)
        if any(not torch.equal(value, actual_batch[key]) for key, value in expected_batch.items()):
            raise ValueError("Math oracle replay inputs differ")
        actual_metrics = update(ddp, actual_batch, optimizer, scheduler, micro, world, bf16=False)
        difference = optimizer_update_difference(moments, optimizer.state_dict()["state"])
        max_parameter = 0.
        for key, value in model.named_parameters():
            value = value.detach().cpu()
            torch.testing.assert_close(value, parameters[key], atol=1e-8, rtol=1e-6)
            max_parameter = max(max_parameter, float((value-parameters[key]).abs().max()))
        if any(value["relative_l2"] > 1e-5 for value in difference.values()):
            raise ValueError("FP32/math replay Adam moments differ")
        for key in ("loss", "selected_tokens", "learning_rate"):
            if expected_metrics[key] != actual_metrics[key]:
                raise ValueError("FP32/math replay objective or schedule differs")
        if not math.isclose(expected_metrics["grad_norm"], actual_metrics["grad_norm"], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("FP32/math replay gradient norm differs")
        return dict(status="passed", dtype="float32", attention="math", inputs_exact=True,
                    max_parameter_difference=max_parameter, optimizer_moments=difference,
                    optimizer_relative_l2_tolerance=1e-5,
                    expected_metrics=expected_metrics, actual_metrics=actual_metrics)
    finally:
        for block, policy in zip(blocks, policies):
            block.sdpa_backend = policy


def wandb_start(plan, provenance, output, resume):
    import wandb
    spec = plan["wandb"]
    key = Path(spec["credential_file"]).read_text().strip()
    os.environ.update(WANDB_API_KEY=key, WANDB_PROJECT=spec["project"], WANDB_ENTITY=spec["entity"],
                      WANDB_MODE="online", WANDB_DIR=str(output), WANDB_BASE_URL="https://api.wandb.ai")
    os.environ.pop("WANDB_RUN_ID", None)
    api = wandb.Api(api_key=key, timeout=30)
    viewer = api.viewer
    username = viewer.get("username") if isinstance(viewer, dict) else viewer.username
    if username != spec["username"]:
        raise ValueError("W&B credential identity differs from the selected account")
    run = wandb.init(entity=spec["entity"], project=spec["project"], id=spec["run_id"],
                     name=spec["name"], resume="must" if resume else "never", dir=str(output),
                     config={"model": plan["model"], "training": plan["training"], "recipe": RECIPE,
                             "data_index_sha256": plan["data_index_sha256"],
                             "evaluation_script": str(Path(plan["source_root"])/"scripts/research/bert_mlm_eval.py")},
                     settings=wandb.Settings(init_timeout=120, save_code=False, disable_git=True))
    write_json(output / "wandb_run.json", dict(url=run.url, id=run.id, entity=run.entity,
                                               project=run.project, username=username, sdk_version=wandb.__version__))
    return run


def verify_initial_checkpoint(plan):
    """Verify the original full-token BERT287000 before a new V8 stage."""
    from dllm.pipelines.immune_llada.fixed_chain.unbinding_mlm import OBJECTIVE
    if plan.get("objective") != OBJECTIVE or plan.get("parent_step") != 287000:
        raise ValueError("Parent initialization requires the explicit V8/BERT287000 contract")
    parent = Path(plan["parent_checkpoint"])
    if sha(parent / "checkpoint_receipt.json") != plan["parent_receipt_sha256"]:
        raise ValueError("Parent checkpoint receipt changed")
    receipt = json.loads((parent / "checkpoint_receipt.json").read_text())
    if (receipt["status"] != "complete" or receipt["step"] != 287000
            or len(receipt["rank_parameter_sha256"]) != receipt["world_size"]
            or len(set(receipt["rank_parameter_sha256"])) != 1):
        raise ValueError("Parent checkpoint is not a completed, synchronized BERT287000")
    for name in ("model.safetensors", "research_config.json", "immune_config.json", "vocabulary.json",
                 "state.json", "run_provenance.json"):
        if sha(parent / name) != receipt["files"][name]:
            raise ValueError("Parent checkpoint file changed: " + name)
    if (receipt["files"]["model.safetensors"] != plan["parent_model_sha256"]
            or json.loads((parent / "immune_config.json").read_text()) != plan["model"]
            or json.loads((parent / "state.json").read_text())["step"] != 287000):
        raise ValueError("Parent model/config/step differs")
    return dict(status="passed", step=287000, model_sha256=plan["parent_model_sha256"])


def main(plan, device, rank, world):
    # Bind one immutable objective per process, including resume/oracle helpers.
    global BertMLMCollator, BertMLMModel, corrupt, RECIPE
    BertMLMCollator, BertMLMModel, corrupt, RECIPE = components(plan.get("objective", "receptor_aa"))
    settings = plan["training"]
    out = Path(plan["output_dir"])
    resume = plan.get("resume_checkpoint")
    if settings["global_batch"] != settings["micro_batch"]*settings["grad_accum"]*world:
        raise ValueError("Global batch differs from microbatch × accumulation × world size")
    rank_zero(lambda: out.mkdir(parents=True, exist_ok=bool(resume)), rank)
    gate = rank_zero(lambda: data_gate(plan), rank)
    index = json.loads(Path(plan["data_index"]).read_text())
    dataset = IndexedPreparedDataset(plan["data_index"])
    total = settings["max_steps"]
    budget = step_budget(len(dataset), settings["global_batch"], total)
    steps_per_epoch = budget["steps_per_epoch"]
    if budget["completed_epochs"] < settings.get("minimum_epochs", 10) or len(dataset) != gate["train"]:
        raise ValueError("Bound training budget must cover at least ten complete data epochs")
    mode = settings.get("budget_mode", "epochs")
    if mode not in ("epochs", "steps") or (mode == "epochs" and total != steps_per_epoch*settings["epochs"]):
        raise ValueError("Epoch/step budget differs from the actual dataset")
    provenance = {k: plan[k] for k in ("model", "training", "source_root", "sources", "data_index", "data_index_sha256")}
    provenance.update(recipe=RECIPE, schema="immune.bert_mlm.training.v1", world_size=world,
                      objective=plan.get("objective", "receptor_aa"),
                      eval_script=str(Path(plan["source_root"])/"scripts/research/bert_mlm_eval.py"),
                      wandb={k: v for k, v in plan["wandb"].items() if k != "credential_file"})
    if plan.get("parent_checkpoint"):
        rank_zero(lambda: verify_initial_checkpoint(plan), rank)
        provenance.update({k: plan[k] for k in ("parent_checkpoint", "parent_step",
                           "parent_receipt_sha256", "parent_model_sha256")})
        provenance["initialization"] = "parent_weights; fresh_optimizer_scheduler_sampler_rng"
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    model = (BertMLMModel.from_parent(plan["parent_checkpoint"]) if plan.get("parent_checkpoint")
             else BertMLMModel(FixedChainConfig(**plan["model"]))).to(device)
    if sum(p.numel() for p in model.parameters()) != 75601176:
        raise ValueError("Parameter count differs from the 75.6M architecture")
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    ddp = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False,
                                  find_unused_parameters=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 betas=(.9, .999), eps=1e-8, weight_decay=.01)
    scheduler = make_scheduler(optimizer, total, settings["warmup_steps"])
    noise = torch.Generator().manual_seed(settings["seed"] + 10000 + rank)
    state = dict(step=0, epoch=0, next_batch=0, seen_records=0, best_score=-1., best_step=0)
    if plan.get("parent_checkpoint"):
        state["parent_step"] = plan["parent_step"]
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, noise, provenance, rank)
    run_holder = []
    def start_logging():
        write_json(out / "run_provenance.json", provenance)
        write_json(out / "data_gate.json", gate)
        write_json(out / "training_budget.json", dict(max_steps=total, **budget))
        if plan["phase"] in ("train", "smoke"):
            run_holder.append(wandb_start(plan, provenance, out, resume))
        return True
    rank_zero(start_logging, rank)
    run = run_holder[0] if run_holder else None
    def log(metrics):
        if rank == 0:
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+"\n")
            print(json.dumps(metrics), flush=True)
            if run:
                # Keep full diagnostics locally; only chart core validation metrics.
                wandb_metrics = {key: value for key, value in metrics.items()
                                 if not key.startswith("valid/")
                                 or key in ("valid/loss", "valid/mask_accuracy")
                                 or (key.count("/") == 2 and key.endswith("/loss"))}
                run.log(wandb_metrics, step=state["step"])
    def evaluate(probe):
        rng = capture_rng(noise)
        try:
            tokens = token_validation(model, index, device, rank, world)
            values = {"valid/"+key: value for key, value in tokens["all"].items()}
            for source, metrics in tokens.items():
                if source != "all":
                    values.update({"valid/"+source+"/"+key: value for key, value in metrics.items()})
            if probe:
                result = representation_validation(model, index, device, rank, world)
                values["probe/macro_auroc"] = result["macro_auroc"]
                for source, metrics in result["sources"].items():
                    values.update({"probe/"+source+"/"+key: value for key, value in metrics.items()})
                if result["macro_auroc"] > state["best_score"]:
                    state.update(best_score=result["macro_auroc"], best_step=state["step"])
                if rank == 0:
                    write_json(out / f"probe_step{state['step']}.json", result)
            if rank == 0:
                write_json(out / f"validation_step{state['step']}.json", tokens)
            log({"step": state["step"], "epoch": state["epoch"], **values})
        finally:
            restore_rng(rng, noise)
    collator = BertMLMCollator(model.config)
    if plan["phase"] == "train" and not resume:
        evaluate(True)
        save_checkpoint(out, model, optimizer, scheduler, noise, state, provenance, rank, world)
    if plan["phase"] == "smoke":
        # Real bound records exercise collation, DDP, AdamW, scheduling and checkpoint I/O.
        smoke_indices = plan.get("smoke_indices", list(range(settings["global_batch"]*3)))
        if len(smoke_indices) != settings["global_batch"]*3 or len(set(smoke_indices)) != len(smoke_indices):
            raise ValueError("Smoke input coverage differs")
        records = [dataset[i] for i in smoke_indices[rank::world]]
        raw = collator(records)
        per_rank = settings["global_batch"]//world
        batches = [{k: v[start:start+per_rank] for k, v in raw.items()} for start in (0, per_rank, 2*per_rank)]
        state.update(step=1, next_batch=1, seen_records=settings["global_batch"])
        update(ddp, corrupt(batches[0], model.tokenizer, noise), optimizer, scheduler, settings["micro_batch"], world)
        checkpoint = save_checkpoint(out, model, optimizer, scheduler, noise, state, provenance, rank, world)
        expected_batch = corrupt(batches[1], model.tokenizer, noise)
        expected_metrics = update(ddp, expected_batch, optimizer, scheduler, settings["micro_batch"], world)
        expected = parameter_digest(model)
        expected_parameters = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
        expected_optimizer = {p: {k: v.detach().cpu().clone() for k, v in values.items()}
                              for p, values in optimizer.state_dict()["state"].items()}
        load_checkpoint(checkpoint, model, optimizer, scheduler, noise, provenance, rank)
        actual_batch = corrupt(batches[1], model.tokenizer, noise)
        if any(not torch.equal(expected_batch[k], actual_batch[k]) for k in expected_batch):
            raise ValueError("Resumed inputs/corruption/labels differ")
        actual_metrics = update(ddp, actual_batch, optimizer, scheduler, settings["micro_batch"], world)
        actual = parameter_digest(model)
        differences = {k: float((v.detach().cpu()-expected_parameters[k]).abs().max())
                       for k, v in model.named_parameters()}
        moment_differences = optimizer_update_difference(expected_optimizer, optimizer.state_dict()["state"])
        write_json(out / f"resume_diagnostic_rank{rank}.json", dict(expected_metrics=expected_metrics,
                   actual_metrics=actual_metrics, exact_parameters=actual == expected,
                   max_parameter_difference=max(differences.values()),
                   optimizer_moments=moment_differences, optimizer_relative_l2_tolerance=1e-3,
                   grad_norm_tolerance=dict(rtol=1e-5, atol=1e-6),
                   largest_differences=sorted(differences.items(), key=lambda x: -x[1])[:8]))
        # Efficient SDPA / distributed reductions can differ by one FP32 ULP.
        # Saved state and corrupted inputs must be exact; subsequent arithmetic
        # is checked numerically with an explicit, recorded tolerance.
        for key, value in model.named_parameters():
            torch.testing.assert_close(value.detach().cpu(), expected_parameters[key], atol=1e-8, rtol=1e-6)
        for key in ("loss", "selected_tokens", "learning_rate"):
            if expected_metrics[key] != actual_metrics[key]:
                raise ValueError("Resumed objective/selected count/schedule differs: " + key)
        # The norm reduces 75M FP32 entries; DDP bucket rebuild/reduction order
        # can exceed 1e-6 relative error even when the objective is identical.
        if not math.isclose(expected_metrics["grad_norm"], actual_metrics["grad_norm"], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("Resumed gradient norm differs beyond floating-point tolerance")
        # BF16 efficient-backend updates are checked numerically; a separate
        # FP32/math replay below retains the stricter moment tolerance.
        if any(value["relative_l2"] > 1e-3 for value in moment_differences.values()):
            raise ValueError("Replayed Adam moments differ beyond floating-point tolerance")
        del expected_parameters, expected_optimizer
        oracle = math_resume_oracle(checkpoint, ddp, optimizer, scheduler, noise, provenance,
                                    batches[1], settings["micro_batch"], rank, world)
        write_json(out / f"math_resume_oracle_rank{rank}.json", oracle)
        state.update(step=2, next_batch=2, seen_records=2*settings["global_batch"])
        # Exercise the spawned loader and the uneven, unpadded epoch tail.
        tail_sampler = CompleteEpochBatchSampler(len(dataset), settings["global_batch"], rank, world,
                                                  settings["seed"], 0, steps_per_epoch-1)
        tail_loader = DataLoader(dataset, batch_sampler=tail_sampler, collate_fn=collator,
                                num_workers=settings["loader_workers"], multiprocessing_context="spawn",
                                generator=torch.Generator().manual_seed(77))
        tail_batches = list(tail_loader)
        if len(tail_batches) != 1:
            raise ValueError("Tail loader coverage differs")
        tail = tail_batches[0]
        tail_count = torch.tensor(len(tail["input_ids"]), device=device)
        torch.distributed.all_reduce(tail_count)
        if int(tail_count) != len(dataset) % settings["global_batch"]:
            raise ValueError("Distributed tail repeats or drops records")
        tail_metrics = update(ddp, corrupt(tail, model.tokenizer, noise), optimizer, scheduler,
                              settings["micro_batch"], world)
        state.update(step=3)
        write_json(out / f"smoke_rank{rank}.json", dict(status="passed", restored_state_exact=True,
                   corruption_exact=True, next_update_parameters_exact=actual == expected,
                   next_update_max_difference=max(differences.values()), tolerance=dict(atol=1e-8, rtol=1e-6),
                   optimizer_moments=moment_differences, optimizer_relative_l2_tolerance=1e-3,
                   grad_norm_tolerance=dict(rtol=1e-5, atol=1e-6),
                   math_oracle=oracle,
                   metrics=actual_metrics, tail_records=int(tail_count),
                   tail_metrics=tail_metrics,
                   allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                   reserved_gib=torch.cuda.max_memory_reserved(device)/2**30))
        evaluate(True)
        rank_zero(lambda: write_json(out / "COMPLETE.json", dict(status="passed", world_size=world)), rank)
        if run:
            run.finish()
        return
    started = time.monotonic()
    while state["step"] < total:
        epoch, start_batch = state["epoch"], state["next_batch"]
        sampler = CompleteEpochBatchSampler(len(dataset), settings["global_batch"], rank, world,
                                             settings["seed"], epoch, start_batch,
                                             max_batches=total-state["step"])
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator,
                            num_workers=settings["loader_workers"], pin_memory=True,
                            generator=torch.Generator().manual_seed(settings["seed"]+epoch),
                            **({"prefetch_factor": 2, "multiprocessing_context": "spawn"}
                               if settings["loader_workers"] else {}))
        for batch_index, clean in enumerate(loader, start=start_batch):
            step_start = time.monotonic()
            batch = corrupt(clean, model.tokenizer, noise)
            scores = update(ddp, batch, optimizer, scheduler, settings["micro_batch"], world)
            state["step"] += 1
            state["next_batch"] = batch_index+1
            n = min(settings["global_batch"], len(dataset)-batch_index*settings["global_batch"])
            state["seen_records"] += n
            end_epoch = state["next_batch"] == steps_per_epoch
            if end_epoch:
                state["epoch"] += 1
                state["next_batch"] = 0
            finished = state["step"] == total
            status = dict(status="training", step=state["step"], target_steps=total,
                          completed_epochs=state["epoch"], epoch_fraction=state["epoch"]+state["next_batch"]/steps_per_epoch,
                          seen_records=state["seen_records"], world_size=world,
                          time_unix=time.time(), seconds_per_step=time.monotonic()-step_start,
                          allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                          reserved_gib=torch.cuda.max_memory_reserved(device)/2**30, **scores)
            if state["step"] <= 3 or state["step"] % settings["log_every"] == 0 or end_epoch or finished:
                if rank == 0:
                    write_json(out / "status.json", status)
                log({"step": state["step"], "epoch": status["epoch_fraction"],
                     **{"train/"+k: v for k, v in scores.items()},
                     "train/seen_records": state["seen_records"], "train/seconds_per_step": status["seconds_per_step"]})
            periodic = state["step"] % settings["eval_every"] == 0
            if periodic or end_epoch or finished:
                evaluate(end_epoch or finished)
            if periodic or end_epoch or finished or state["step"] == 50:
                save_checkpoint(out, model, optimizer, scheduler, noise, state, provenance, rank, world)
            del batch, clean
        del loader, sampler
        gc.collect()
    if (state["step"] != total or state["seen_records"] != budget["seen_records"]
            or state["epoch"] != budget["completed_epochs"] or state["next_batch"] != budget["next_batch"]):
        raise ValueError("Training ended without the exact step budget and record coverage")
    rank_zero(lambda: write_json(out / "status.json", dict(status="complete", **state,
                                target_steps=total, world_size=world, time_unix=time.time())), rank)
    rank_zero(lambda: write_json(out / "COMPLETE.json", dict(status="success", **state,
                                target_steps=total, world_size=world, seconds=time.monotonic()-started)), rank)
    if run:
        run.finish()
