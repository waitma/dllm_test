"""Check the selected TCR checkpoint and real GPU inputs before submitting jobs.

Run in protenix_abtcr (activate pllm first): python scripts/downstream/preflight_tcr_native.py
  --checkpoint /abs/ckpt --out-dir /abs/new_gate --scope binding
This is an execution/invariance gate, not a model-quality experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from downstream.benchmark.common.model_api import FusionGrammarEmbedder
from downstream.benchmark.tcr_binding.query_protocol import TCRBindingEmbedder, TCRBindingQueryProtocol
from downstream.benchmark.tcr_binding.run_retrained_ours import (
    FOLDS, _verify_data_archive, build_pair_universe, input_mode_for_track,
    load_protocol, protocol_overlap_audit,
)
from downstream.benchmark.tcr_binding.retrained_protocol import sha256_file
from downstream.grammar.ab_features import write_json
from scripts.downstream.tcr_eval_pin import PIN

EXPECTED_SHA = PIN.sha


def binding_gate(backbone):
    reports = {}
    _verify_data_archive()
    for track in ("cdr3b", "cdr3ab", "others_cdr3b", "others_longab"):
        frames, data = load_protocol("AS", track)
        overlap = protocol_overlap_audit(frames, track)
        pairs = build_pair_universe(frames, track)
        protocol = TCRBindingQueryProtocol(input_mode=input_mode_for_track(track))
        # Check every record for validity/length before allocating expensive jobs.
        max_length, longest = 0, None
        for pair in pairs:
            record = protocol.record(*pair)
            length = sum(len(c.sequence) for c in record.chains) + 8
            if length > max_length:
                max_length, longest = length, pair
        selected = list(dict.fromkeys([longest] + pairs[:15]))
        def embed(rows, bs):
            backbone.batch_size = bs
            wrapped = TCRBindingEmbedder(backbone, protocol)
            return wrapped.embed_pairs(beta=[r[1] for r in rows],
                alpha=[r[2] for r in rows] if len(rows[0]) == 3 else None,
                peptide=[r[0] for r in rows])
        large = embed(selected, 16)
        small = embed(selected, 1)
        rev = embed(selected[::-1], 16)[::-1]
        assert np.isfinite(large).all() and large.shape == (len(selected), backbone.hidden)
        unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        delta = float(np.max(np.abs(unit(large) - unit(small))))
        cosine = float(np.min((unit(large) * unit(small)).sum(1)))
        reverse_delta = float(np.max(np.abs(unit(large) - unit(rev))))
        if delta > .005 or cosine < .99995 or reverse_delta > .005:
            raise ValueError(f"batch consistency failed {track}: {delta=} {cosine=} {reverse_delta=}")
        reports[track] = {"status": "passed", "n_unique_inputs": len(pairs),
            "max_input_length": max_length, "batch_size_tested": [1, 16],
            "normalized_max_abs_delta": delta, "min_cosine": cosine,
            "reverse_normalized_max_abs_delta": reverse_delta,
            "input_protocol_sha256": protocol.sha256, "input_protocol": protocol.provenance,
            "data": data, "split_overlap": overlap}
        print(f"PASS binding {track}: {len(pairs)} unique inputs, {max_length=}, {delta=}", flush=True)
        del frames, pairs
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--scope", choices=["binding"], default="binding")
    args = p.parse_args()
    target = args.out_dir / "passed.json"
    if target.exists():
        raise FileExistsError("Use a fresh preflight directory")
    assert torch.cuda.is_available(), "GPU required"
    weights_sha = sha256_file(args.checkpoint / "model.safetensors")
    assert weights_sha == PIN.sha, "not the pinned TCR eval snapshot"
    state = json.loads((args.checkpoint / "trainer_state.json").read_text())
    assert state["global_step"] == PIN.step
    backbone = FusionGrammarEmbedder(str(args.checkpoint), device="cuda", batch_size=16)
    result = binding_gate(backbone)
    write_json(target, {"status": "passed", "scope": args.scope,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": weights_sha,
        "global_step": PIN.step, "quality_evaluation": False, "tracks": result,
        "gpu": torch.cuda.get_device_name(0), "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
    print(f"TCR_BINDING_{PIN.step}_GPU_GATE_PASSED", flush=True)


if __name__ == "__main__":
    main()
