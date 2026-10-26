"""Compare old/new updates on copied real weights, then check the guarded LR.

Activate pllm; USE_TF=0 python -B -m scripts.research.verify_diffusion_gradient_update
--checkpoint ABS --output ABS. Requires one idle GPU. Saves diagnostics only;
does not write model checkpoints or qualify eight-GPU production training.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

import torch
from torch.nn.parallel import DistributedDataParallel

from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.bert_mlm_eval import read_samples
from scripts.research.diffusion_continuation_objective import DiffusionEditModel, DiffusionEditCollator, corrupt
from scripts.research.diffusion_extension import extension_scheduler
from scripts.research.diffusion_training_guard import ValidationLRCeiling
from scripts.research.run_bert_mlm import load_checkpoint
from scripts.research.run_diffusion_continuation import update


def verify(checkpoint, output):
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    output.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((checkpoint / "run_provenance.json").read_text())
    old_path = Path(provenance["source_root"]) / "scripts/research/run_diffusion_continuation.py"
    if sha(old_path) != provenance["sources"][str(old_path)]:
        raise ValueError("Original update source changed")
    spec = importlib.util.spec_from_file_location("original_diffusion_update", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    index = json.loads(Path(provenance["data_index"]).read_text())
    samples = read_samples(index["files"]["token_valid"])
    chosen = []
    for source in ("oas", "trait"):
        chosen.append(next(row for row in samples if row["source_group"] == source
                           and sum(len(chain) for chain in row["record"]["chains"]) < 400))
    with tempfile.TemporaryDirectory(prefix="diffusion-gradient-check-") as temporary:
        torch.distributed.init_process_group("nccl", init_method="file://" + temporary + "/rendezvous",
                                             rank=0, world_size=1)
        try:
            model = DiffusionEditModel.from_pretrained(checkpoint).cuda().train()
            for block in model.backbone.transformer.blocks:
                block.sdpa_backend = "math"
            ddp = DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=provenance["training"]["learning_rate"],
                                          weight_decay=.01, betas=(.9, .999), eps=1e-8)
            scheduler = extension_scheduler(optimizer, provenance["extension"])
            noise = torch.Generator()
            collator = DiffusionEditCollator(model.config)
            raw = collator([row["record"] for row in chosen])
            batch = corrupt(raw, model.tokenizer, torch.Generator().manual_seed(261026), model.recipe)

            def restore():
                return load_checkpoint(checkpoint, model, optimizer, scheduler, noise, provenance, 0)

            restore()
            baseline = old.update(ddp, batch, optimizer, scheduler, 2, 1, bf16=False)
            expected = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
            state = restore()
            actual = update(ddp, batch, optimizer, scheduler, 2, 1, bf16=False, gradient_clip=1.)
            largest = max(float((parameter.detach().cpu() - expected[name]).abs().max())
                          for name, parameter in model.named_parameters())
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.detach().cpu(), expected[name], rtol=1e-6, atol=1e-8)
            assert abs(baseline["grad_norm"] - actual["grad_norm_pre_clip"]) < 1e-6
            assert actual["grad_norm_post_clip"] <= 1.00001
            restore()
            before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
            guard = ValidationLRCeiling(scheduler, dict(factor=.5, patience=3, min_lr=2e-6,
                relative_threshold=.001), best=state["best_score"], initial_scale=.5)
            guarded = update(ddp, batch, optimizer, guard, 2, 1, bf16=False)
            assert guarded["learning_rate"] == baseline["learning_rate"] * .5
            maximum_half_error = 0.
            for name, parameter in model.named_parameters():
                target = before[name] + .5 * (expected[name] - before[name])
                maximum_half_error = max(maximum_half_error,
                    float((parameter.detach().cpu() - target).abs().max()))
                torch.testing.assert_close(parameter.detach().cpu(), target, rtol=1e-6, atol=2e-7)
            bf16 = update(ddp, batch, optimizer, guard, 2, 1, bf16=True)
            write_json(output / "verification.json", dict(status="passed", world_size=1,
                production_eight_rank_qualification=False, checkpoint=str(checkpoint),
                checkpoint_receipt_sha256=sha(checkpoint / "checkpoint_receipt.json"),
                diagnostic_records=len(chosen), sources=[x["source_group"] for x in chosen],
                views=len(batch["input_ids"]), original=baseline, telemetry_fix=actual,
                max_parameter_difference=largest, guarded=guarded,
                max_half_update_error=maximum_half_error, bf16=bf16,
                model_checkpoints_written=False))
            print(json.dumps(dict(status="passed", max_parameter_difference=largest,
                                  max_half_update_error=maximum_half_error)))
        finally:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.checkpoint, args.output)
