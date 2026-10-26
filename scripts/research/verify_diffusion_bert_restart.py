"""Check frozen BERT initialization, real-row supervision and A100 maximum shape.

Activate pllm; USE_TF=0 python -B FROZEN_THIS_FILE --bundle ABS.
Runs one real-batch update and two max-length updates on one idle GPU, writes
diagnostics only. Production multi-rank qualification remains a separate gate.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    source = args.bundle / "source"
    sys.path.insert(0, str(source))
    os.environ.update(USE_TF="0", IMMUNE_SDPA_BACKEND="efficient", OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    from scripts.research.launch_diffusion_bert_restart import verify, production_world_size
    plan = verify(args.bundle, "smoke")
    production_world = production_world_size(plan)
    import numpy as np
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from scripts.research.bert_mlm_data import IndexedPreparedDataset, ROLES, sha, write_json
    from scripts.research.bert_mlm_eval import read_samples
    from scripts.research.diffusion_continuation_data import source_lengths, warmup_cosine_scheduler
    from scripts.research.diffusion_source_sampling import make_sampler
    from scripts.research.diffusion_continuation_objective import DiffusionEditCollator, corrupt
    from scripts.research.diffusion_training_guard import ValidationLRCeiling
    from scripts.research.run_diffusion_continuation import new_model, update, parameter_digest, verify_parent
    from scripts.research import profile_bert_mlm

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    index = json.loads(Path(plan["data_index"]).read_text())
    dataset = IndexedPreparedDataset(plan["data_index"])
    lengths = source_lengths(index)
    settings = plan["training"]
    per_rank_records = settings["global_batch"] // production_world
    per_rank_views = per_rank_records * settings["complementary_views"]
    verify_parent(plan)
    model = new_model(plan, device)
    parent_receipt = json.loads((Path(plan["parent_checkpoint"]) / "checkpoint_receipt.json").read_text())
    assert parameter_digest(model) == parent_receipt["rank_parameter_sha256"][0]
    collator = DiffusionEditCollator(model.config)
    counts, offset, rows = {}, 0, []
    for name, size in lengths.items():
        selected = np.linspace(0, size-1, min(128, size), dtype=np.int64).tolist()
        rows.extend(dataset[offset+i] for i in selected)
        counts[name] = len(selected)
        offset += size
    rows.extend(row["record"] for row in read_samples(index["files"]["token_valid"]))
    noise = torch.Generator().manual_seed(261026)
    cdr_scope_counts = Counter()
    for start in range(0, len(rows), 16):
        records = rows[start:start+16]
        raw = collator(records)
        for i, row in enumerate(records):
            # IndexedPreparedDataset returns BioSeqRecord; normalize for counting.
            if not isinstance(row, dict):
                chains = [(c.role, c.sequence, c.metadata) for c in row.chains]
            else:
                chains = [(role, seq, row.get("chain_metadata", {}).get(str(j), {}))
                          for j, (role, seq) in enumerate(zip(row["chain_roles"], row["chains"]))]
            eligible = 0
            for role, seq, metadata in chains:
                if role in ROLES:
                    observed = len(seq) - sum(metadata.get("synthetic_residue_mask", []))
                    eligible += observed + int(observed > 0)
            assert int(raw["eligible_mask"][i].sum()) == eligible
        noisy = corrupt(raw, model.tokenizer, noise, model.recipe)
        assert torch.isfinite(noisy["loss_weights"]).all()
        assert not (noisy["active_population"] & ~noisy["eligible_mask"]).any()
        if "cdr_tasks" in plan["recipe"]:
            scoped = noisy["mask_scope"].ne(0).expand_as(noisy["input_ids"])
            context = scoped & ~noisy["active_population"] & ~noisy["relation_mask"]
            assert torch.equal(noisy["input_ids"][context], raw["input_ids"].repeat(2,1)[context])
            assert noisy["labels"][context].eq(-100).all()
            cdr_scope_counts.update(noisy["mask_scope"].flatten().tolist())
    out = Path(plan["profile_receipt"]).parent
    write_json(out / "collator_audit.json", dict(status="passed", sampled_train_by_source=counts,
        validation_records=index["files"]["token_valid"]["records"], total_records=len(rows),
        synthetic_supervision_mismatches=0, nonfinite_weights=0, cdr_scope_view_counts=dict(cdr_scope_counts),
        plan_sha256=sha(args.bundle / "plan.json")))
    with tempfile.TemporaryDirectory(prefix="diffusion-bert-restart-") as temp:
        torch.distributed.init_process_group("nccl", init_method="file://" + temp + "/init", rank=0, world_size=1)
        try:
            ddp = DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False,
                                          gradient_as_bucket_view=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=.01)
            scheduler = ValidationLRCeiling(warmup_cosine_scheduler(optimizer,
                settings["max_steps"], settings["warmup_steps"]), settings["validation_lr_guard"])
            assert not optimizer.state and scheduler.last_epoch == 0
            assert abs(optimizer.param_groups[0]["lr"] - 2e-7) < 1e-15
            sampler = make_sampler(list(lengths.values()), settings, rank=0, world=production_world, start_step=0)
            raw = collator([dataset[i] for i in next(iter(sampler))])
            actual = corrupt(raw, model.tokenizer, noise, model.recipe)
            real_metrics = update(ddp, actual, optimizer, scheduler, settings["micro_batch_views"], 1,
                                  gradient_clip=settings["gradient_clip"])
            cdr_updates = {}
            if "cdr_tasks" in plan["recipe"]:
                from scripts.research.diffusion_cdr_tasks import CDRS, cdr_spans
                from scripts.research.diffusion_cdr_eval import evaluate_cdr
                qualified = [row for row in rows if len(cdr_spans(row)) == 6][:2]
                assert len(qualified) == 2
                cdr_raw = collator(qualified)
                for scope, targets in (("cdr_single", ["H3"]), ("cdr_multi", list(CDRS))):
                    forced = corrupt(cdr_raw, model.tokenizer, noise, model.recipe,
                                     mask_scope=scope, target_cdrs=targets, ratio=1., edit=False)
                    cdr_updates[scope] = update(ddp, forced, optimizer, scheduler, 4, 1,
                                               gradient_clip=settings["gradient_clip"])
                    assert cdr_updates[scope]["task_views_" + scope] == 4
                cdr_eval = evaluate_cdr(model, [{"record": row} for row in qualified], device, batch_size=2)
                assert cdr_eval["eligible_records"] == 2
                assert all(value is not None for value in cdr_eval["summary"].values())
                write_json(out / "cdr_validation_check.json", cdr_eval)
            profile_bert_mlm.BertMLMCollator = DiffusionEditCollator
            raw, tok = profile_bert_mlm.synthetic_batch(model.config, per_rank_records,
                                                      model.config.max_length)
            batch = corrupt(raw, tok, noise, model.recipe)
            assert batch["input_ids"].shape == (per_rank_views, model.config.max_length)
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            metrics = [update(ddp, batch, optimizer, scheduler, settings["micro_batch_views"], 1,
                              gradient_clip=settings["gradient_clip"]) for _ in range(2)]
            torch.cuda.synchronize()
            assert all(m["grad_norm_post_clip"] <= 1.00001 for m in [real_metrics, *metrics])
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            allocated, reserved = (torch.cuda.max_memory_allocated()/2**30, torch.cuda.max_memory_reserved()/2**30)
            assert allocated < .9 * total and reserved < .93 * total
            write_json(Path(plan["profile_receipt"]), dict(status="passed", device=torch.cuda.get_device_name(0),
                plan_sha256=sha(args.bundle / "plan.json"), world_size=1, production_world_size=production_world,
                initial_parameters_equal_bert287000=True, initial_optimizer_empty=True,
                diffusion_start_step=0, initial_learning_rate=2e-7, real_batch=real_metrics,
                cdr_task_updates=cdr_updates,
                max_length=model.config.max_length, per_rank_records=per_rank_records, per_rank_views=per_rank_views,
                micro_views=[min(settings["micro_batch_views"], per_rank_views-offset)
                             for offset in range(0, per_rank_views, settings["micro_batch_views"])],
                metrics=metrics, allocated_gib=allocated,
                reserved_gib=reserved, total_gib=total, seconds_per_update=(time.monotonic()-begin)/2,
                production_rank_smoke="pending; run by cloud job before formal training", checkpoints_written=False))
            print(json.dumps(dict(status="passed", checked_records=len(rows), allocated_gib=allocated,
                                  reserved_gib=reserved, initial_parameters_equal_bert287000=True)))
        finally:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
