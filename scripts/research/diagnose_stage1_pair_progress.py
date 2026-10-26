"""Read-only CPU diagnostics for an existing AB500K Pair/KD checkpoint.

Activate pllm and run this absolute script path with --plan ABS_PLAN --output
ABS_NEW_DIRECTORY. Uses four fixed validation records and FP32; never trains,
updates weights, writes checkpoints, or submits GPU jobs. This is a small local
mechanistic probe, not a replacement for full BF16 validation or ablation training.
"""
from pathlib import Path
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["IMMUNE_SDPA_BACKEND"] = "math"
    os.environ["USE_TF"] = "0"
    plan = json.loads(args.plan.read_text())
    sys.path.insert(0, plan["source_root"])
    import numpy as np
    import torch
    from safetensors import safe_open
    from scripts.research.run_stage1_ab500k import PilotDataset, prepare_batch
    from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel
    from dllm.pipelines.immune_llada.fixed_chain.structure_losses import chain_aware_kd
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    directory = Path(plan["output_root"]) / "pair_kd"
    checkpoint = Path(json.loads((directory / "latest.json").read_text())["checkpoint"])
    state = json.loads((checkpoint / "state.json").read_text())
    receipt = json.loads((checkpoint / "checkpoint_receipt.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"phase": "loading_checkpoint", "checkpoint": str(checkpoint)}), flush=True)
    model_path = checkpoint / "model.safetensors"
    h = hashlib.sha256()
    with model_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if receipt["status"] != "complete" or digest != receipt["files"]["model.safetensors"]:
        raise ValueError("Checkpoint model receipt mismatch")
    model = StructureMLMModel.from_pretrained(checkpoint).cpu().float().eval()
    # This encoder has only Linear/LayerNorm/SiLU/Embedding modules: enabling its
    # training flag activates chunk recomputation without introducing dropout.
    model.pair_encoder.train(True)
    dataset = PilotDataset(json.loads(Path(plan["subset_manifest"]).read_text()),
                           json.loads(Path(plan["cache_manifest"]).read_text()), "valid", targets=True)
    ordinals = sorted(np.random.default_rng(20261023).choice(len(dataset), 4, replace=False).tolist())
    rows = [dataset[index] for index in ordinals]
    batch, pairs, teacher = prepare_batch(model, rows, seed=41016, targets=True)
    names, parameters = zip(*model.named_parameters())

    def group(name):
        if name.startswith("pair_encoder."):
            return "pair_encoder"
        if name.startswith("pair_bias_outputs."):
            return "pair_feedback_outputs"
        if name.startswith("distogram_head."):
            return "distogram_head"
        return "inherited_backbone"

    weights = {}
    with safe_open(str(directory / "checkpoint-0/model.safetensors"), framework="pt", device="cpu") as original:
        for name, parameter in zip(names, parameters):
            if group(name) == "inherited_backbone":
                continue
            initial = original.get_tensor(name)
            weights[name] = {"parameters": parameter.numel(), "l2_norm": float(parameter.detach().norm()),
                             "delta_from_step0_l2": float((parameter.detach()-initial).norm()),
                             "max_abs": float(parameter.detach().abs().max())}
    print(json.dumps({"phase": "forward_and_gradient_probe", "records": len(rows), "pairs": len(pairs)}), flush=True)
    result = model(pair_indices=pairs, **batch)
    mlm = result["loss_sum"] / result["mlm_count"]
    kd_terms = chain_aware_kd(result["pair_logits"], teacher, pairs, batch["chain_ids"],
                             batch["noise_kind"], plan["training"]["temperature"])
    kd = kd_terms["intra_sum"] / kd_terms["intra_count"].clamp_min(1)
    mlm_gradients = torch.autograd.grad(mlm, parameters, retain_graph=True, allow_unused=True)
    kd_gradients = torch.autograd.grad(kd, parameters, allow_unused=True)
    effective_weight = plan["training"]["intra_weight"] * min(1., state["step"] / plan["training"]["kd_warmup_steps"])
    aggregates = {}
    for name, gm, gk in zip(names, mlm_gradients, kd_gradients):
        item = aggregates.setdefault(group(name), {"mlm_squared": 0., "kd_squared": 0., "dot": 0.,
                                                   "mlm_connected_tensors": 0, "kd_connected_tensors": 0})
        if gm is not None:
            item["mlm_squared"] += float(gm.square().sum())
            item["mlm_connected_tensors"] += 1
        if gk is not None:
            item["kd_squared"] += float(gk.square().sum())
            item["kd_connected_tensors"] += 1
        if gm is not None and gk is not None:
            item["dot"] += float((gm*gk).sum())
    gradients = {}
    for key, item in aggregates.items():
        m, k = math.sqrt(item["mlm_squared"]), math.sqrt(item["kd_squared"])
        gradients[key] = dict(mlm_l2=m, raw_kd_l2=k, weighted_kd_l2=k*effective_weight,
                             weighted_kd_to_mlm_ratio=k*effective_weight/m if m else None,
                             cosine=item["dot"]/(m*k) if m*k else None,
                             mlm_connected_tensors=item["mlm_connected_tensors"], kd_connected_tensors=item["kd_connected_tensors"])
    del mlm_gradients, kd_gradients
    selected = batch["labels"].ne(-100)
    allowed = model.allowed_target
    original_logits = result["logits"].detach()[selected][:, allowed]
    original_ce, original_kd = float(mlm.detach()), float(kd.detach())
    print(json.dumps({"phase": "feedback_ablation"}), flush=True)
    model.pair_config = replace(model.pair_config, feedback="none")
    with torch.no_grad():
        ablated = model(pair_indices=pairs, **batch)
        ablated_ce = float(ablated["loss_sum"]/ablated["mlm_count"])
        ablated_logits = ablated["logits"][selected][:, allowed]
        difference = original_logits-ablated_logits
        feedback = dict(mlm_ce_enabled=original_ce, mlm_ce_disabled=ablated_ce,
                        disabled_minus_enabled_ce=ablated_ce-original_ce,
                        selected_tokens=int(selected.sum()),
                        selected_token_prediction_changes=int((original_logits.argmax(-1) != ablated_logits.argmax(-1)).sum()),
                        selected_logits_difference_rms=float(difference.square().mean().sqrt()),
                        selected_logits_difference_max_abs=float(difference.abs().max()))
    output = dict(observed_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint),
                  checkpoint_step=state["step"], checkpoint_model_sha256=digest,
                  device="cpu", precision="float32", validation_ordinals=ordinals,
                  original_row_ids=[int(row[0]) for row in rows], mask_seed=41016,
                  records=len(rows), pair_targets=len(pairs), effective_kd_weight=effective_weight,
                  raw_kd_loss=original_kd, gradients=gradients, feedback_ablation=feedback,
                  new_parameter_changes=weights,
                  limitations=["Four records, one fixed mask, CPU FP32; not population or training-seed inference.",
                               "Disabling feedback is inference-only; it is not a separately trained KD-only control.",
                               "Gradient norms are local un-clipped loss gradients, not AdamW parameter update magnitudes.",
                               "No claim of structural accuracy or downstream improvement."])
    (args.output / "diagnostic.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"checkpoint_step":state["step"], "gradients":gradients,"feedback_ablation":feedback}), flush=True)


if __name__ == "__main__":
    main()
