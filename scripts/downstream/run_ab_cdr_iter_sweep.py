"""Local SAb23H2 CDR iteration diagnostic, retaining per-sequence predictions.

Run in protenix_abtcr (activate pllm first):
  python scripts/downstream/run_ab_cdr_iter_sweep.py --checkpoint /abs/snapshot \
    --out-dir /abs/new_run --iterations 1 2 4 8
Uses the production fusion loader, CDR masking and sampler, with batch size one.
Never selects a headline iteration or overwrites an existing run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import time
import traceback

import numpy as np
import torch

from downstream.grammar.ab_features import ROOT, file_sha256, write_json
from downstream.grammar.cdr_infill import (
    SAbDabDataset, _build_cdr_partial_mask, _chain_role_for_mode,
    build_antibody_record, resolve_sabdab_test_json,
)
from downstream.grammar.common import run_grammar_generate
from downstream.grammar.light_chain_pairing import _protocol_source_hashes
from downstream.grammar.metrics import masked_token_accuracy
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import build_generation_mask
from examples.llada.load_fusion_checkpoint import load_fusion_for_eval

MODES = ("cdrh1", "cdrh2", "cdrh3", "cdrl1", "cdrl2", "cdrl3")
PAPER_AAR_PERCENT = dict(zip(MODES, (74.8, 68.6, 36.8, 81.1, 80.1, 73.7)))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def target_encoder_slots(batch, target):
    """Independent per-residue mapping for checking actual encoder inputs."""
    positions = []
    for row, column in target.nonzero().tolist():
        chain = int(batch["chain_ids"][row, column])
        inner = int(batch["position_ids_inner"][row, column])
        slots = batch["encoder_residue_mask"][row, chain].nonzero().flatten()
        positions.append((row, column, chain, int(slots[inner])))
    return positions


def generate(model, batch, partial, steps, *, audit=False):
    """Delegate unchanged generation, optionally checking both forward boundaries."""
    target = build_generation_mask(batch, partial)
    if not audit:
        return run_grammar_generate(model, batch, partial_mask=partial, max_iter=steps,
                                    sampling_strategy="argmax", temperature=1.0)[0], []
    slots = target_encoder_slots(batch, target)
    original_denoise = model._denoise
    decoder_mask = int(model.config.mask_token_id)
    encoder_mask = int(model.config.encoder_mask_token_id)
    trace = []
    expected_for_encoder = {}
    previous = []

    def checked_denoise(**kwargs):
        tokens = kwargs["input_ids"]
        encoder = kwargs["encoder_input_ids"]
        expected = batch["encoder_input_ids"].clone()
        accepted_ids = set(model._residue_token_ids.tolist())
        for row, column, chain, position in slots:
            token = int(tokens[row, column])
            expected[row, chain, position] = (
                model.llada_to_grammar_ids[token] if token in accepted_ids else encoder_mask
            )
        if not torch.equal(encoder, expected):
            raise AssertionError("Encoder does not mirror generated CDR state")
        if not torch.equal(tokens[~target], batch["input_ids"][~target]):
            raise AssertionError("Fixed CDR context changed")
        if previous:
            committed = target & previous[-1].ne(decoder_mask)
            if not torch.equal(tokens[committed], previous[-1][committed]):
                raise AssertionError("Committed residues were rewritten")
        previous.append(tokens.clone())
        pending = int((tokens.eq(decoder_mask) & target).sum())
        trace.append({"forward": len(trace) + 1, "pending_mask": pending,
                      "generated_visible": int(target.sum()) - pending})
        expected_for_encoder["ids"] = expected.reshape(-1, expected.size(-1))
        return original_denoise(**kwargs)

    def encoder_hook(module, args, kwargs):
        if not torch.equal(kwargs["input_ids"], expected_for_encoder["ids"]):
            raise AssertionError("Actual ESMC call differs from checked fusion input")

    handle = model.encoder.register_forward_pre_hook(encoder_hook, with_kwargs=True)
    model._denoise = checked_denoise
    try:
        tokens, _ = run_grammar_generate(model, batch, partial_mask=partial, max_iter=steps,
                                        sampling_strategy="argmax", temperature=1.0)
    finally:
        model._denoise = original_denoise
        handle.remove()
    if not trace or trace[0]["pending_mask"] != int(target.sum()):
        raise AssertionError("Initial CDR targets were visible")
    return tokens, trace


def summarize_rows(rows):
    if not rows:
        raise ValueError("Cannot summarize empty predictions")
    folds = sorted({r["fold"] for r in rows})
    fold_aar = [float(np.mean([r["aar"] for r in rows if r["fold"] == fold]) * 100)
                for fold in folds]
    return {"n_sequences": len(rows), "average_aar_all_folds": float(np.mean(fold_aar)),
            "average_aar": float(np.mean([r["aar"] for r in rows]) * 100),
            "fold_aar_percent": fold_aar, "fold_sd_ddof0": float(np.std(fold_aar))}


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU required")
    if args.out_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    start = time.monotonic()
    state = {"state": "loading", "started": now(), "pid": os.getpid(),
             "host": socket.gethostname(), "completed_modes": [], "completed_predictions": 0}

    def status(**update):
        state.update(update, updated=now(), elapsed_seconds=time.monotonic() - start)
        write_json(args.out_dir / "status.json", state)

    try:
        status()
        checkpoint_hash = file_sha256(args.checkpoint / "model.safetensors")
        snapshot = json.loads((args.checkpoint / "eval_snapshot_manifest.json").read_text())
        if checkpoint_hash != snapshot["sha256"]:
            raise ValueError("Retained checkpoint hash mismatch")
        sources = _protocol_source_hashes()
        for path in (Path(__file__), ROOT / "downstream/grammar/cdr_infill.py",
                     ROOT / "downstream/grammar/metrics.py", ROOT / "downstream/grammar/ab_features.py"):
            sources[str(path.resolve())] = file_sha256(path)
        data_files = sorted({Path(resolve_sabdab_test_json(str(args.test_set), mode, 0)).resolve()
                             for mode in MODES})
        manifest = {"protocol": "cdr_iter_sweep_encoder_feedback_v1", "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_hash, "dataset": "SAb23H2", "n_sequences_per_mode": 60,
                    "iterations": args.iterations, "modes": MODES, "batch_size": 1,
                    "sampling_strategy": "argmax", "cfg_scale": 0, "temperature": 1, "seed": 42,
                    "data_sha256": {str(p): file_sha256(p) for p in data_files}, "source_sha256": sources,
                    "torch_version": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
                    "headline_selection": "none; diagnostic only", "paper_source": "/root/oph_paper/oph.pdf Table 1"}
        write_json(args.out_dir / "manifest.json", manifest)
        torch.manual_seed(42)
        bundle = load_fusion_for_eval(args.checkpoint, device="cuda")
        residue_names = {bundle.grammar_tokenizer.encode_residues(aa)[0]: aa
                         for aa in "ACDEFGHIKLMNPQRSTVWYXBUZO"}
        summary = {"complete": False, "protocol": manifest["protocol"], "checkpoint_sha256": checkpoint_hash,
                   "paper_aar_percent": PAPER_AAR_PERCENT, "metrics": {}}
        all_audits = {}
        for mode in MODES:
            dataset = SAbDabDataset(str(args.test_set), mode, 0)
            if len(dataset) != 60:
                raise ValueError(f"Expected full SAb23H2 (60), got {len(dataset)}")
            rows_by_iter = {steps: [] for steps in args.iterations}
            status(state="running", mode=mode, mode_completed=0)
            for index in range(len(dataset)):
                heavy, light, target_string, pos, _ = dataset[index]
                record = build_antibody_record(heavy, light)
                batch = bundle.collator([record])
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                role, cdr = _chain_role_for_mode(mode)
                partial = _build_cdr_partial_mask(batch, bundle.grammar_tokenizer, record, role, cdr, target_string)
                target = build_generation_mask(batch, partial)
                if int(target.sum()) != len(target_string):
                    raise AssertionError("Wrong CDR mask length")
                for steps in args.iterations:
                    tokens, trace = generate(bundle.model, batch, partial, steps, audit=index == 0)
                    predicted_ids = tokens[target].tolist()
                    truth_ids = batch["labels"][target].tolist()
                    prediction_tokens = [residue_names.get(i, bundle.grammar_tokenizer.token(i))
                                         for i in predicted_ids]
                    aar = float(masked_token_accuracy(tokens, batch["labels"], target).item())
                    row = {"fold": 0, "row": index, "mode": mode, "max_iter": steps,
                           "heavy": heavy, "light": light, "target": target_string, "position": pos,
                           "prediction_tokens": prediction_tokens, "prediction_ids": predicted_ids,
                           "prediction": "".join(prediction_tokens),
                           "target_ids": truth_ids, "aar": aar,
                           "correct": sum(a == b for a, b in zip(predicted_ids, truth_ids)),
                           "target_length": len(truth_ids)}
                    rows_by_iter[steps].append(row)
                    if index == 0:
                        all_audits[f"{mode}/iter{steps}"] = trace
                        if steps == max(args.iterations):
                            poisoned = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
                            residue_id = int(bundle.model._residue_token_ids[0])
                            encoder_id = int(bundle.model.llada_to_grammar_ids[residue_id])
                            poisoned["input_ids"][target] = residue_id
                            poisoned["labels"][target] = encoder_id
                            for r, col, chain, epos in target_encoder_slots(batch, target):
                                poisoned["encoder_input_ids"][r, chain, epos] = encoder_id
                            alternate, alternate_trace = generate(bundle.model, poisoned, partial, steps, audit=True)
                            if not torch.equal(tokens, alternate) or trace != alternate_trace:
                                raise AssertionError("Hidden target perturbation changed generation")
                            all_audits[f"{mode}/hidden_reference_invariance"] = True
                    state["completed_predictions"] += 1
                if (index + 1) % 5 == 0:
                    status(mode_completed=index + 1)
                    print(f"{now()} {mode} {index + 1}/60, predictions={state['completed_predictions']}", flush=True)
            for steps, rows in rows_by_iter.items():
                path = args.out_dir / mode / f"iter{steps}"
                write_json(path / "predictions.json", rows)
                metrics = summarize_rows(rows)
                metrics.update(mode=mode, max_iter=steps, checkpoint_sha256=checkpoint_hash)
                if steps == 1 and args.reference_logs:
                    log_path = args.reference_logs / f"{mode}.log"
                    matches = re.findall(r"^Average AAR all folds:\s*([0-9.eE+-]+)", log_path.read_text(), re.M)
                    if len(matches) != 1:
                        raise ValueError(f"Missing or ambiguous previous iter1 result: {log_path}")
                    delta = metrics["average_aar_all_folds"] - float(matches[0])
                    metrics["previous_iter1_delta_pp"] = delta
                    if abs(delta) > 1e-4:
                        raise AssertionError(f"iter1 no longer reproduces previous result: {mode}, delta={delta}")
                write_json(path / "metrics.json", metrics)
                summary["metrics"][f"{mode}/iter{steps}"] = metrics
                print(f"RESULT {mode} iter{steps} AAR={metrics['average_aar_all_folds']:.6f}", flush=True)
            state["completed_modes"].append(mode)
            write_json(args.out_dir / "feedback_audit.json", all_audits)
            write_json(args.out_dir / "summary.json", summary)
            status()
        for path, digest in sources.items():
            if file_sha256(Path(path)) != digest:
                raise RuntimeError(f"Source changed during sweep: {path}")
        summary["complete"] = True
        summary["elapsed_seconds"] = time.monotonic() - start
        write_json(args.out_dir / "summary.json", summary)
        status(state="complete", mode=None, peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    except BaseException:
        status(state="failed", error=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--test-set", type=Path, default=ROOT / "data/downstream/cdr_infilling/sab23h2_converted")
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--reference-logs", type=Path)
    args = parser.parse_args()
    if not args.iterations or len(set(args.iterations)) != len(args.iterations) or min(args.iterations) < 1:
        parser.error("iterations must be distinct positive integers")
    run(args)


if __name__ == "__main__":
    main()
