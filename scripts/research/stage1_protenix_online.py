"""Memory-only Protenix targets; activate pllm before using the joint teacher env.

Run tests via python -B -m pytest on test_protenix_online.py. This module has no
target writer, cache directory, file-backed queue, or dataset-sized memoization.
Only the frozen teacher weights stay resident between calls.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import logging
import os
import random
import time

import numpy as np
import torch

from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator, corrupt
from dllm.pipelines.immune_llada.fixed_chain.structure_losses import chain_aware_kd, normalized_objective
from scripts.research.bert_mlm_eval import fixed_corruption
from scripts.research.stage1_data import make_teacher_request
from scripts.research.run_stage1_distillation import autocast_context


@contextmanager
def teacher_rng(seed, device):
    """Teacher loading, featurization and sampling cannot advance student RNG."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    py, np_state = random.getstate(), np.random.get_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, cudnn_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def sequence_seed(sequences, base_seed):
    # Stable across ranks, epochs, microbatch boundaries and resume. Sequence
    # order is part of the identity; identical H/L copies get the same target.
    encoded = json.dumps([base_seed, sequences], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def make_request(record, config):
    request = make_teacher_request(record, config)
    if (request["excluded"] or len(request["chains"]) != 2
            or [c["role"] for c in request["chains"]] != ["antibody_heavy", "antibody_light"]):
        raise ValueError("Online AB pilot requires two complete native H/L chains")
    # This is pairing provenance, not evidence of an experimentally solved interface.
    request["inter_evidence"] = [{"chains": [c["record_chain"] for c in request["chains"]],
                                 "evidence": "frozen_oas_native_pair_membership"}]
    from scripts.research.stage1_data import content_sha
    request["request_sha256"] = content_sha({k: v for k, v in request.items() if k != "request_sha256"})
    return request


def prepare_micro(model, rows, seed, needs_teacher):
    clean = FullTokenMLMCollator(model.config)([x[1] for x in rows])
    noisy = fixed_corruption(clean, model.tokenizer, [int(x[0]) for x in rows], seed=seed, corrupt_fn=corrupt)
    requests, indices = [], []
    if needs_teacher:
        for row, (_, record, _) in enumerate(rows):
            request = make_request(record, model.config)
            positions = torch.tensor([p for c in request["chains"] for p in c["student_positions"]])
            expected = model.tokenizer.encode_residues("".join(c["sequence"] for c in request["chains"]))
            if clean["input_ids"][row, positions].tolist() != expected:
                raise ValueError("Clean sequence-to-student mapping changed")
            for chain in request["chains"]:
                if not clean["chain_ids"][row, chain["student_positions"]].eq(chain["student_chain"]).all():
                    raise ValueError("Chain order changed in this batch")
            pair = torch.triu_indices(len(positions), len(positions), offset=1).T
            indices.append(torch.cat((torch.full((len(pair), 1), row), positions[pair]), dim=1))
            requests.append(request)
    pairs = torch.cat(indices) if indices else torch.empty((0, 3), dtype=torch.long)
    return noisy, pairs, requests


def build_micros(model, rows, micro, seed, arm):
    # Only small CPU input tensors and mappings are retained for an optimizer
    # update. Teacher tensors are generated inside optimizer_update, one micro
    # at a time; no CUDA work is delegated to DataLoader workers.
    return [prepare_micro(model, rows[i:i+micro], seed, arm == "pair_kd")
            for i in range(0, len(rows), micro)]


class OnlineProtenixTeacher:
    def __init__(self, device, *, base_seed=42, confidence_threshold=2., adapter_factory=None):
        self.device = torch.device(device)
        self.base_seed, self.confidence_threshold = base_seed, confidence_threshold
        if confidence_threshold not in (1., 2., 4.):
            raise ValueError("Use a native PDE CDF threshold: 1, 2 or 4 A")
        if adapter_factory is None:
            from scripts.research.general_teacher_adapters import ProtenixMini
            adapter_factory = ProtenixMini
        with teacher_rng(base_seed, self.device), torch.no_grad():
            self.adapter = adapter_factory(self.device)
        # Native inference emits multiple INFO lines per record. At 500K x
        # epochs these are enormous logs; use a bounded progress event instead.
        logging.getLogger("protenix").setLevel(logging.WARNING)
        for name, logger in list(logging.Logger.manager.loggerDict.items()):
            if name.startswith("protenix.") and isinstance(logger, logging.Logger):
                logger.setLevel(logging.WARNING)
        self.completed_records = 0
        self.last_progress = time.monotonic()
        for name in ("model", "esm"):
            module = getattr(self.adapter, name, None)
            if module is not None and (module.training or any(p.requires_grad for p in module.parameters())):
                raise ValueError("Teacher must be frozen and in eval mode")
        self.contract = {**self.adapter.contract, "delivery": "memory_only_no_target_cache",
            "base_seed": base_seed, "confidence_threshold_A": confidence_threshold,
            "confidence_symmetry": "mean_of_both_directions", "confidence_calibrated": False,
            "pair_coverage": "all_unordered_nonself_intra_and_inter", "student_input": "sequence_only"}

    @torch.no_grad()
    def targets(self, requests):
        logits, confidence = [], []
        for ordinal, request in enumerate(requests):
            sequences = [c["sequence"] for c in request["chains"]]
            length = sum(map(len, sequences))
            seed = sequence_seed(sequences, self.base_seed)
            example = dict(id=f"online_{seed:08x}", sequences=sequences, chain_lengths=list(map(len, sequences)))
            with teacher_rng(seed, self.device):
                prepared = self.adapter.prepare(example)
                result = self.adapter.predict_with_confidence(prepared, self.confidence_threshold)
            head, q = result["logits"], result["confidence"]
            if (head.shape != (length, length, 64) or q.shape != (length, length)
                    or not torch.isfinite(head).all() or not torch.isfinite(q).all()
                    or (q < 0).any() or (q > 1).any()):
                raise ValueError("Invalid online distogram/confidence; never silently omit a record")
            i, j = torch.triu_indices(length, length, offset=1, device=head.device)
            logits.append(head[i, j].detach())
            confidence.append(((q[i, j] + q[j, i]) * .5).detach())
            # Coordinates, CDF diagnostics and native feature tensors are not
            # retained. Advanced indexing above owns independent tensor storage.
            del result, prepared, head, q
            self.completed_records += 1
            if time.monotonic()-self.last_progress >= 60:
                print(json.dumps(dict(event="online_teacher_progress", rank=int(os.environ.get("RANK", "0")),
                    completed_records=self.completed_records, micro_done=ordinal+1,
                    micro_records=len(requests), teacher_target_storage="none")), flush=True)
                self.last_progress = time.monotonic()
        if not logits:
            return (torch.empty((0, 64), device=self.device), torch.empty((0,), device=self.device))
        return torch.cat(logits), torch.cat(confidence)


def update_counts(micros, device, world):
    counts = torch.zeros(3, device=device, dtype=torch.float64)
    for batch, pairs, _ in micros:
        counts[0] += int(batch["labels"].ne(-100).sum())
        if len(pairs):
            row, i, j = pairs.unbind(-1)
            same = batch["chain_ids"][row, i] == batch["chain_ids"][row, j]
            counts[1] += row[same].unique().numel()
            counts[2] += row[~same].unique().numel()
    if world > 1:
        torch.distributed.all_reduce(counts)
    if counts[0] <= 0:
        raise ValueError("No global MLM targets")
    return counts


def optimizer_update(model, optimizer, scheduler, micros, settings, arm, step,
                     world=1, precision=None, *, teacher=None, target_digests=None):
    device = next(model.parameters()).device
    counts = update_counts(micros, device, world)
    optimizer.zero_grad(set_to_none=True)
    # MLM; weighted intra/inter; raw intra/inter; pairs; records; tokens;
    # confidence sums/counts by topology; VV/VM/MM weighted sums/counts.
    values = torch.zeros(18, device=device, dtype=torch.float64)
    scale = min(1., (step+1)/max(1, settings["kd_warmup_steps"]))
    teacher_seconds = 0.
    for number, (raw, cpu_pairs, requests) in enumerate(micros):
        if arm == "pair_kd":
            if teacher is None:
                raise ValueError("An online teacher is required")
            started = time.monotonic()
            targets, confidence = teacher.targets(requests)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            teacher_seconds += time.monotonic()-started
            if len(targets) != len(cpu_pairs):
                raise ValueError("Online pair coverage differs")
            if target_digests is not None:
                digest = hashlib.sha256()
                for tensor in (targets, confidence):
                    digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
                target_digests.append(digest.hexdigest())
        sync = model.no_sync() if hasattr(model, "no_sync") and number+1 < len(micros) else nullcontext()
        with sync:
            batch = {k: v.to(device) for k, v in raw.items()}
            pairs = cpu_pairs.to(device)
            with autocast_context(device, precision or settings["precision"]):
                result = model(pair_indices=pairs, **batch) if arm == "pair_kd" else model(**batch)
            if arm == "pair_kd":
                weight = confidence if settings.get("confidence_weighting", True) else None
                kd = chain_aware_kd(result["pair_logits"], targets, pairs, batch["chain_ids"],
                    batch["noise_kind"], settings["temperature"], pair_confidence=weight)
                loss = normalized_objective(result["loss_sum"], batch["labels"].ne(-100).sum(), kd,
                    settings["intra_weight"]*scale, settings["inter_weight"]*scale,
                    denominators=counts, world_size=world)
                with torch.no_grad():
                    raw_kd = chain_aware_kd(result["pair_logits"].detach(), targets, pairs,
                        batch["chain_ids"], batch["noise_kind"], settings["temperature"])
                    for column, name in ((1, "intra_sum"), (2, "inter_sum")):
                        values[column] += kd[name].detach()
                        values[column+2] += raw_kd[name]
                    same = batch["chain_ids"][pairs[:, 0], pairs[:, 1]] == batch["chain_ids"][pairs[:, 0], pairs[:, 2]]
                    for column, select in ((8, same), (10, ~same)):
                        values[column] += confidence[select].double().sum()
                        values[column+1] += select.sum()
                    for column, name in ((12, "vv"), (14, "vm"), (16, "mm")):
                        values[column] += kd[name+"_sum"]
                        values[column+1] += kd[name+"_count"]
                    values[5] += len(pairs)
            else:
                loss = result["loss_sum"] * world / counts[0]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite online training loss")
            loss.backward()
            values[0] += result["loss_sum"].detach().double()
            values[6] += len(batch["input_ids"])
            values[7] += batch["eligible_mask"].sum()
        del result, loss, batch, pairs
        if arm == "pair_kd":
            del targets, confidence, weight, kd, raw_kd
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_norm"], error_if_nonfinite=True)
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    if world > 1:
        torch.distributed.all_reduce(values)
    mlm = float(values[0]/counts[0])
    intra, inter = (float(values[i]/counts[i].clamp_min(1)) for i in (1, 2))
    metrics = dict(mlm=mlm, intra_kl=intra, inter_kl=inter,
        raw_intra_kl=float(values[3]/counts[1].clamp_min(1)),
        raw_inter_kl=float(values[4]/counts[2].clamp_min(1)),
        objective=mlm+scale*(settings["intra_weight"]*intra+settings["inter_weight"]*inter),
        selected_tokens=int(counts[0]), kd_records=int(counts[1]), inter_records=int(counts[2]),
        pairs=int(values[5]), records=int(values[6]), eligible_tokens=int(values[7]),
        kd_scale=scale if arm == "pair_kd" else 0., backbone_lr=lr, grad_norm=float(norm),
        teacher_seconds_local=teacher_seconds, teacher_target_disk_bytes=0)
    for column, name in ((8, "intra_confidence"), (10, "inter_confidence"),
                         (12, "vv_kl"), (14, "vm_kl"), (16, "mm_kl")):
        metrics[name] = float(values[column]/values[column+1].clamp_min(1))
        metrics[name+"_pairs"] = int(values[column+1])
    return metrics
