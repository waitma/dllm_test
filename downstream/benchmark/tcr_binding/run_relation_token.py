"""Score official NM2025 AS tests without a fitted head.

Run from the repository root after activating pllm and the evaluation environment:
python downstream/benchmark/tcr_binding/run_relation_token.py --checkpoint PATH --output PATH
The default track is the full others beta-only test protocol, not a subsample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "downstream/benchmark")]

import numpy as np
import pandas as pd
import torch
from tcr_binding.query_protocol import TCRBindingQueryProtocol
from tcr_binding.retrained_protocol import load_official_test, sha256_file
from tcr_binding.run_retrained_ours import (
    EVAL_SETS, _atomic_csv, _atomic_json, _validate_frame, _verify_data_archive,
    data_track_for_track, input_columns, input_mode_for_track, paper_metrics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", choices=["cdr3b", "others_cdr3b", "cdr3ab", "others_longab"], default="others_cdr3b")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = TCRBindingQueryProtocol(input_mode=input_mode_for_track(args.track))
    manifest = dict(status="loading", checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_sha256=sha256_file(args.checkpoint / "model.safetensors"),
                    track=args.track, archive=_verify_data_archive(),
                    query_protocol=protocol.provenance, script_sha256=sha256_file(Path(__file__)),
                    scoring="logit(binding)-logit(nonbinding); no CFG; no fitted head",
                    batch_size=args.batch_size, tests={})
    frames = {}
    for fold in range(1, 6):
        for name in EVAL_SETS:
            key = f"fold_{fold}/{name}"
            frame, _, meta = load_official_test(eval_set=name, neg_source="AS", fold=fold, track=data_track_for_track(args.track))
            if fold != 1 and name != "seen_test":
                assert meta["test_member_sha256"] == manifest["tests"][f"fold_1/{name}"]["test_member_sha256"]
                continue
            _validate_frame(frame, key, args.track)
            frames[key] = frame
            manifest["tests"][key] = meta
    columns = input_columns(args.track)
    pairs = list(dict.fromkeys(pair for frame in frames.values()
                              for pair in frame[columns].astype(str).itertuples(index=False, name=None)))
    manifest["unique_inputs"] = len(pairs)
    _atomic_json(args.output / "manifest.json", manifest)
    print(f"Loaded full tests: {len(pairs)} unique inputs; rows={[len(f) for f in frames.values()]}", flush=True)
    from examples.llada.load_fusion_checkpoint import load_fusion_for_eval
    from scripts.downstream.score_binding_relation import _token_id, apply_decoder_corruption_to_encoder
    bundle = load_fusion_for_eval(args.checkpoint, device="cuda", max_length=protocol.max_length)
    model = bundle.model.eval()
    binding = _token_id(bundle.llada_tokenizer, "<binding>")
    nonbinding = _token_id(bundle.llada_tokenizer, "<nonbinding>")
    assert binding != nonbinding
    scores, argmax_ids = [], []
    start_time = time.monotonic()
    kwargs = ("input_ids", "attention_mask", "chain_ids", "position_ids_inner", "position_ids_chain",
              "residue_mask", "encoder_input_ids", "encoder_attention_mask", "encoder_residue_mask",
              "encoder_chain_mask", "encoder_position_ids")
    with torch.inference_mode():
        for start in range(0, len(pairs), args.batch_size):
            batch = protocol.collate([protocol.record(*pair) for pair in pairs[start:start+args.batch_size]],
                                     bundle.collator, decoder_mask_token_id=model._decoder_mask_token_id)
            batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
            query = batch["relation_target_mask"].bool()
            assert bool(batch["input_ids"][query].eq(model._decoder_mask_token_id).all())
            batch["encoder_input_ids"] = apply_decoder_corruption_to_encoder(
                batch=batch, corruption_mask=query, mask_token_id=model._encoder_mask_token_id)
            output = model._denoise(**{k: batch.get(k) for k in kwargs})
            logits = output.logits[query].float()
            scores.extend((logits[:, binding] - logits[:, nonbinding]).cpu().tolist())
            argmax_ids.extend(logits.argmax(-1).cpu().tolist())
            del output, logits, batch
            if start == 0 or (start // args.batch_size) % 50 == 0:
                print(f"Scored {len(scores)}/{len(pairs)} in {time.monotonic()-start_time:.1f}s", flush=True)
    assert np.isfinite(scores).all()
    lookup = dict(zip(pairs, scores))
    unique = pd.DataFrame(pairs, columns=columns)
    unique["logit_margin"] = scores
    unique["full_vocab_argmax_id"] = argmax_ids
    _atomic_csv(args.output / "unique_scores.csv", unique)
    metrics = {}
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score, average_precision_score
    from common.metrics import precrec_pr_auc
    for key, frame in frames.items():
        frame = frame.copy()
        margin = np.array([lookup[pair] for pair in frame[columns].astype(str).itertuples(index=False, name=None)])
        labels = frame["Affinity"].astype(int).to_numpy()
        frame["logit_margin"] = margin
        frame["conditional_binding_probability"] = expit(margin)
        result = paper_metrics(labels, expit(margin))
        # Rank metrics use raw margins: avoid sigmoid saturation creating ties.
        result.update(auroc=float(roc_auc_score(labels, margin)),
                      auprc_precrec=float(precrec_pr_auc(labels, margin)),
                      average_precision_sklearn=float(average_precision_score(labels, margin)),
                      n=len(frame), prevalence=float(labels.mean()))
        metrics[key] = result
        _atomic_csv(args.output / key / "predictions.csv", frame)
    seen = [metrics[f"fold_{fold}/seen_test"] for fold in range(1, 6)]
    summary = {"status": "complete", "track": args.track, "checkpoint_sha256": manifest["checkpoint_sha256"],
               "unique_inputs": len(pairs), "seconds_scoring": time.monotonic()-start_time,
               "seen_test_mean": {k: float(np.mean([r[k] for r in seen])) for k in seen[0]},
               "seen_test_sd_ddof1": {k: float(np.std([r[k] for r in seen], ddof=1)) for k in seen[0]},
               "seen_independent": metrics["fold_1/seen_independent"],
               "unseen_independent": metrics["fold_1/unseen_independent"], "per_test": metrics,
               "notes": "One frozen checkpoint; five test-fold partitions, not five fitted models. Independent sets scored once. Pretraining overlap unverified."}
    _atomic_json(args.output / "summary.json", summary)
    manifest["status"] = "complete"
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
