"""Generate AB500K compact targets with four independent four-GPU tasks.

Activate pllm, then use the isolated teacher Python with torchrun --nproc_per_node=4
THIS_FILE --plan PLAN --job-index 0 (use job-index 0, 1, 2, 3 for the four tasks).
``--verify-only`` checks frozen files and runtime on CPU without loading a model.
"""
import argparse
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from scripts.research import stage1_teacher as teacher
from scripts.research.stage1_ab500k_cache import (
    CODEC, ShardWriter, WorkerTimer, atomic_json, content_sha, decode_record, encode_record,
    precision_metrics, read_index, sha, validate_pilot_receipt, verify_request, verify_subset_receipt,
)


def worker_number(job_index, local_rank, local_world_size=4):
    if job_index not in range(4) or local_rank not in range(4) or local_world_size != 4:
        raise ValueError("Exactly four tasks of four independent GPU workers are required")
    return 4 * job_index + local_rank


def verify_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    verify_subset_receipt(plan)
    subset_path = Path(plan["subset_manifest"])
    if plan.get("subset_manifest_sha256", sha(subset_path)) != sha(subset_path):
        raise ValueError("Subset manifest differs from frozen plan")
    for source, digest in plan.get("sources", {}).items():
        if sha(source) != digest:
            raise ValueError("Frozen source changed: " + source)
    subset = json.loads(subset_path.read_text())
    if sha(subset["requests"]) != subset["requests_sha256"]:
        raise ValueError("Subset requests changed")
    specs = subset["request_shards"]
    if len(specs) != 16 or {s["worker_index"] for s in specs} != set(range(16)):
        raise ValueError("Subset does not contain 16 exclusive worker partitions")
    if sum(s["pilot_count"] for s in specs) != 2048 or subset["pilot_count"] != 2048:
        raise ValueError("The globally stratified teacher pilot must contain 2048 records")
    for spec in specs:
        if sha(spec["path"]) != spec["sha256"] or sha(spec["pilot_path"]) != spec["pilot_sha256"]:
            raise ValueError("Teacher worker partition changed")
        if spec["pilot_count"] < 1:
            raise ValueError("Every worker needs a nonempty pilot partition")
    from transformers import EsmFold2Model
    from transformers.models.esmfold2.protein_utils import prepare_protein_features
    import transformers
    actual = dict(transformers=transformers.__version__, torch=torch.__version__,
        model_source_sha256=sha(inspect.getfile(EsmFold2Model)),
        featurizer_source_sha256=sha(inspect.getfile(prepare_protein_features)),
        fast_compat_source_sha256=sha(teacher.__file__))
    expected = plan["teacher_contract"]
    if any(expected.get(k) != v for k, v in actual.items()):
        raise ValueError("Teacher runtime differs from frozen teacher contract")
    if (expected["num_loops"] != 3 or expected["base_seed"] != 42
            or expected["pairs_per_chain_pair"] != 1024 or expected["distogram_bins"] != 64
            or expected["mode"] != "trunk_only"):
        raise ValueError("AB500K must retain the existing teacher supervision protocol")
    model_dir = Path(plan["model_dir"])
    if sha(model_dir / "config.json") != expected["config_sha256"]:
        raise ValueError("Teacher model config changed")
    for name, digest in expected["weight_sha256"].items():
        if sha(model_dir / name) != digest:
            raise ValueError("Teacher weights changed: " + name)
    return plan, subset


class OfflineTeacher:
    def __init__(self, plan, device):
        from transformers.models.esmfold2.protein_utils import prepare_protein_features
        self.device = device
        self.model, self.contract, self.loading_info = teacher.load_teacher(plan["model_dir"], device)
        expected = plan["teacher_contract"]
        self.contract.update(num_loops=expected["num_loops"], base_seed=expected["base_seed"],
            pairs_per_chain_pair=expected["pairs_per_chain_pair"],
            featurizer_source_sha256=sha(inspect.getfile(prepare_protein_features)))
        if self.contract != expected:
            raise ValueError("Loaded teacher differs from bound contract")

    def predict(self, request, *, replay=False):
        verify_request(request)
        started = time.monotonic()
        sequences = [chain["sequence"] for chain in request["chains"]]
        if not sequences or request.get("excluded") or request.get("inter_evidence"):
            raise ValueError("AB pilot requires complete observed chains with intra-chain-only targets")
        features = teacher.prepare_multichain_features(sequences, self.device)
        kwargs = teacher.trunk_arguments(features)
        seed = (self.contract["base_seed"] + int(request["request_sha256"][:8], 16)) % (2**31)

        def forward():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                return self.model(**kwargs, num_loops=self.contract["num_loops"]).distogram_logits[0].float().cpu()

        logits = forward()
        length = sum(map(len, sequences))
        if logits.shape != (length, length, 64) or not torch.isfinite(logits).all():
            raise ValueError("Invalid teacher output")
        symmetry = float((logits - logits.transpose(0, 1)).abs().max())
        if symmetry > 1e-5:
            raise ValueError("Teacher distogram is not symmetric")
        repeat = None
        if replay:
            repeat = float((logits - forward()).abs().max())
            if repeat > 1e-5:
                raise ValueError("Deterministic teacher replay differs")
        pairs = teacher.sample_pairs(request, self.contract["pairs_per_chain_pair"], seed)
        positions = torch.tensor([p for c in request["chains"] for p in c["student_positions"]], dtype=torch.long)
        selected = logits[pairs[:, 0], pairs[:, 1]].contiguous()
        metadata = dict(schema="immune.structure_teacher.sparse_cache.packed.v1",
            request_sha256=request["request_sha256"], teacher_contract=self.contract,
            seed=seed, chain_lengths=list(map(len, sequences)), pair_count=len(pairs),
            symmetry_max_abs=symmetry, repeat_max_abs=repeat, seconds=time.monotonic()-started,
            teacher_requires_grad=False, quality_status=request["quality_status"],
            **{name: features[name][0].cpu().tolist() for name in ("asym_id", "entity_id", "sym_id")})
        return dict(request=request, metadata=metadata, tensors=dict(teacher_logits=selected,
            teacher_pair_indices=pairs.contiguous(), student_positions=positions))


def run(args):
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    plan, subset = verify_plan(args.plan)
    if args.verify_only:
        print(json.dumps(dict(status="passed", scope="cpu_teacher_preflight", workers=16, codec=CODEC)), flush=True)
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    worker = worker_number(args.job_index, local_rank, int(os.environ.get("LOCAL_WORLD_SIZE", "4")))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.cuda.reset_peak_memory_stats(device)
    spec = next(s for s in subset["request_shards"] if s["worker_index"] == worker)
    model = OfflineTeacher(plan, device)
    print(json.dumps(dict(event="teacher_loaded", worker=worker, seconds=time.monotonic()-started)), flush=True)
    root = Path(plan["cache_root"])
    with ShardWriter(root, worker, model.contract, budget=plan.get("worker_budget_bytes", 8_750_000_000)) as writer, WorkerTimer(writer.directory, started, started_utc) as timer:
        pilot_path = writer.directory / "pilot.json"
        pilot_binding = dict(subset_manifest_sha256=sha(plan["subset_manifest"]),
            pilot_sha256=spec["pilot_sha256"], teacher_contract_sha256=content_sha(model.contract), codec=CODEC)
        if pilot_path.exists():
            pilot = json.loads(pilot_path.read_text())
            if any(pilot.get(k) != v for k, v in pilot_binding.items()):
                raise ValueError("Existing pilot does not pass this run's precision gate")
            validate_pilot_receipt(pilot, spec, model.contract)
        else:
            pilot_started = time.monotonic()
            weighted_kl, pair_count, maximum, measurements = 0., 0, 0., []
            for ordinal, envelope in enumerate(read_index(spec["pilot_path"])):
                cache = model.predict(envelope["request"], replay=ordinal < plan.get("replay_per_worker", 2))
                blob = encode_record(cache)
                decoded = decode_record(blob)
                metrics = precision_metrics(cache["tensors"]["teacher_logits"], decoded["tensors"]["teacher_logits"])
                weighted_kl += metrics["mean_kl"] * metrics["pairs"]
                pair_count += metrics["pairs"]
                maximum = max(maximum, metrics["max_pair_kl"])
                measurements.append(dict(request_sha256=cache["request"]["request_sha256"],
                    length=sum(cache["metadata"]["chain_lengths"]), seconds=cache["metadata"]["seconds"],
                    packed_bytes=len(blob), repeat_max_abs=cache["metadata"]["repeat_max_abs"], **metrics))
                writer.append(envelope, cache)
                measurements[-1]["stored_record_bytes"] = writer.record_bytes[(envelope["split"], envelope["row_index"])]
                timer.update()
            if len(measurements) != spec["pilot_count"] or not pair_count:
                raise ValueError("Pilot coverage differs")
            pilot = dict(**pilot_binding, status="passed", worker_index=worker, count=len(measurements),
                mean_kl=weighted_kl/pair_count, max_pair_kl=maximum, pairs=pair_count,
                seconds=time.monotonic()-pilot_started, measurements=measurements,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            mean_record_bytes = sum(x["stored_record_bytes"] for x in measurements)/len(measurements)
            pilot.update(mean_stored_record_bytes=mean_record_bytes,
                projected_worker_bytes=math.ceil(mean_record_bytes*spec["count"] + math.ceil(spec["count"]/1024)*4096 + (2 << 20)),
                worker_budget_bytes=writer.budget,
                remaining_worker_budget_bytes=writer.budget-writer.committed_bytes-writer.pending_bytes)
            if pilot["mean_kl"] > 1e-5 or maximum > 1e-3:
                pilot["status"] = "failed"
                atomic_json(pilot_path, pilot)
                raise ValueError("FP16 precision gate failed; do not produce remaining targets")
            if pilot["projected_worker_bytes"] > writer.budget:
                pilot.update(status="failed", reason="projected_worker_storage_exceeds_quota")
                writer.commit()
                atomic_json(pilot_path, pilot)
                raise RuntimeError("Pilot projects worker storage above quota; remaining inference is stopped")
            validate_pilot_receipt(pilot, spec, model.contract)
            writer.commit()
            atomic_json(pilot_path, pilot)
        if pilot["projected_worker_bytes"] > writer.budget:
            raise RuntimeError("Resumed pilot storage projection exceeds this worker's quota")
        print(json.dumps(dict(event="pilot_passed", worker=worker, mean_kl=pilot["mean_kl"], max_pair_kl=pilot["max_pair_kl"])), flush=True)
        processed, generated = 0, 0
        last_progress = time.monotonic()
        for envelope in read_index(spec["path"]):
            if not writer.contains(envelope):
                writer.append(envelope, model.predict(envelope["request"]))
                generated += 1
            processed += 1
            timer.update()
            if time.monotonic()-last_progress >= 60 or processed == spec["count"]:
                print(json.dumps(dict(event="teacher_progress", worker=worker, processed=processed,
                    total=spec["count"], generated=generated, elapsed_seconds=time.monotonic()-started)), flush=True)
                last_progress = time.monotonic()
        if processed != spec["count"]:
            raise ValueError("Worker input count differs")
        writer.finish(processed, dict(requests_sha256=spec["sha256"],
            subset_manifest_sha256=sha(plan["subset_manifest"]), pilot_sha256=sha(pilot_path),
            start_utc=started_utc, end_utc=datetime.now(timezone.utc).isoformat(),
            seconds=time.monotonic()-started, gpu_worker_hours=(time.monotonic()-started)/3600,
            worker_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            worker_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30))
        stored = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        if stored > 140_000_000_000:
            raise RuntimeError("Global teacher cache exceeds 140 GB payload quota")
    print(json.dumps(dict(event="worker_complete", worker=worker, count=processed)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--job-index", type=int, choices=range(4))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and args.job_index is None:
        parser.error("--job-index is required for teacher generation")
    run(args)


if __name__ == "__main__":
    main()
