"""Check full-size CUDA outputs/gradients of RAM-only Pair speed candidates.

Activate pllm; CUDA_VISIBLE_DEVICES=0 USE_TF=0 IMMUNE_SDPA_BACKEND=efficient
python -B THIS_FILE --plan ABS_PLAN --checkpoint ABS_NONZERO_PAIR_CKPT
  --benchmark-environment ABS_JSON --output ABS_NEW_DIR
Reads trained weights to test nonzero feedback, but never updates or saves weights.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "checkpoint", "benchmark-environment", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["baseline_repeat", "cast", "symmetric", "compiled", "symmetric_compiled"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"verification_source.py").write_bytes(Path(__file__).read_bytes())
    path = Path(__file__).with_name("stage1_pair_speed_candidates.py")
    (args.output/"candidates_source.py").write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location("speed_candidates", path)
    candidates = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidates)
    plan = json.loads(args.plan.read_text())
    sys.path.insert(0, plan["source_root"])
    import torch
    from scripts.research import run_stage1_ab500k as training
    from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel
    from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
    torch.set_num_threads(1)
    model_sha = hashlib.sha256((args.checkpoint/"model.safetensors").read_bytes()).hexdigest()
    receipt = json.loads((args.checkpoint/"checkpoint_receipt.json").read_text())
    if receipt["files"]["model.safetensors"] != model_sha:
        raise ValueError("Checkpoint fingerprint mismatch")
    model = StructureMLMModel.from_pretrained(args.checkpoint).cuda().train()
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    assert all(torch.count_nonzero(p.weight) for p in model.pair_bias_outputs)
    base_pairs, base_attention, base_config = model.pair_encoder.pairs, model.pair_encoder.attention_bias, model.pair_config
    environment = json.loads(args.benchmark_environment.read_text())
    subset = json.loads(Path(plan["subset_manifest"]).read_text())
    data = training.PilotDataset(subset, plan["cache_manifest"], "train")
    rows = [data[index] for index in environment["ordinals"]]
    raw, pairs, teacher = training.prepare_batch(model, rows, seed=environment.get("mask_seed", 10042))
    batch = {key: value.cuda() for key, value in raw.items()}
    pairs, teacher = pairs.cuda(), teacher.cuda()
    def evaluate():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pair_indices=pairs, **batch)
        kd = training.chain_aware_kd(out["pair_logits"], teacher, pairs, batch["chain_ids"], batch["noise_kind"])
        mlm = out["loss_sum"]/batch["labels"].ne(-100).sum()
        kl = kd["intra_sum"]/kd["intra_count"]
        loss = mlm + .05*kl
        loss.backward()
        torch.cuda.synchronize()
        selected = out["logits"][batch["labels"].ne(-100)][:, model.allowed_target].detach().float().cpu()
        gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
        return dict(scores={"mlm": float(mlm.detach()), "kd": float(kl.detach()), "objective": float(loss.detach())},
                    selected_logits=selected, gradients=gradients)
    reference = evaluate()
    results = []
    for variant in args.variants:
        model.pair_encoder.pairs, model.pair_encoder.attention_bias = base_pairs, base_attention
        model.pair_config = model.pair_encoder.config = base_config
        if variant.startswith("symmetric"):
            candidates.install_symmetric(model)
        if "cast" in variant:
            candidates.install_cast(model)
        if "compiled" in variant:
            candidates.install_compile(model, precise="precise" in variant)
        if "no_pair_checkpoint" in variant:
            model.pair_config = model.pair_encoder.config = replace(base_config, checkpoint_pair_chunks=False)
        actual = evaluate()
        if set(reference["gradients"]) != set(actual["gradients"]):
            raise ValueError("Gradient connectivity changed")
        groups = {}
        for name, grad in actual["gradients"].items():
            group = name.split('.')[0] if name.startswith(("pair_encoder.", "pair_bias_outputs.", "distogram_head.")) else "backbone"
            row = groups.setdefault(group, dict(reference_sq=0., candidate_sq=0., error_sq=0., dot=0., max_abs_error=0.))
            base = reference["gradients"][name]
            delta = grad-base
            for key, value in (("reference_sq", base.square().sum()), ("candidate_sq", grad.square().sum()),
                               ("error_sq", delta.square().sum()), ("dot", (grad*base).sum())):
                row[key] += float(value)
            row["max_abs_error"] = max(row["max_abs_error"], float(delta.abs().max()))
        for row in groups.values():
            row["relative_l2_error"] = (row["error_sq"]/max(row["reference_sq"], 1e-30))**.5
            row["cosine"] = row["dot"]/max((row["reference_sq"]*row["candidate_sq"])**.5, 1e-30)
        diff = actual["selected_logits"]-reference["selected_logits"]
        score_errors = {key: actual["scores"][key]-value for key, value in reference["scores"].items()}
        passed = (all(abs(value) <= .001*abs(reference["scores"][key])+1e-6 for key, value in score_errors.items())
                  and float(diff.square().mean().sqrt()) <= .01
                  and all(row["relative_l2_error"] <= .03 for row in groups.values()))
        result = dict(variant=variant, status="passed_bf16_tolerance" if passed else "failed",
            scores=actual["scores"], score_errors=score_errors, gradients=groups,
            selected_logits_rms_error=float(diff.square().mean().sqrt()),
            selected_logits_max_abs_error=float(diff.abs().max()),
            selected_token_argmax_changes=int((actual["selected_logits"].argmax(-1) != reference["selected_logits"].argmax(-1)).sum()))
        results.append(result)
        (args.output/(variant+".json")).write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        del actual
    summary = dict(status="passed_bf16_tolerance" if all(r["status"] == "passed_bf16_tolerance" for r in results) else "failed",
        observed_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(args.checkpoint), checkpoint_model_sha256=model_sha,
        gpu=torch.cuda.get_device_name(), batch=len(rows), sequence_length=batch["input_ids"].shape[1], pair_targets=len(pairs),
        reference_scores=reference["scores"], results=results,
        tolerance=dict(loss_relative=.001, selected_logits_rms=.01, per_group_gradient_relative_l2=.03),
        scope="Same trained nonzero-feedback weights; one actual batch; FP32 CPU algebra tests separate; not long-run or distributed equivalence")
    (args.output/"verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    if summary["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
