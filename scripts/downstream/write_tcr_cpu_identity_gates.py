"""Write CPU identity stubs so a new checkpoint can reuse the 49000 job generator.

These files lock checkpoint SHA, step, and input-protocol hashes. They are not
a GPU batch-consistency gate and must not be described as one.
"""
from __future__ import annotations

from downstream.benchmark.tcr_binding.query_protocol import TCRBindingQueryProtocol
from downstream.benchmark.tcr_binding.run_retrained_ours import input_mode_for_track
from downstream.grammar.ab_features import file_sha256, write_json
from scripts.downstream.run_tcr_native_generation import identity as generation_identity
from scripts.downstream.run_tcr_native_repr import DATASETS, identity as repr_identity
from scripts.downstream.tcr_eval_pin import PIN


def main() -> None:
    weights = PIN.ckpt / "model.safetensors"
    if file_sha256(weights) != PIN.sha:
        raise ValueError("snapshot SHA does not match TCR_EVAL_SHA")
    PIN.out.mkdir(parents=True, exist_ok=False)
    tracks = {}
    for track in ("cdr3b", "cdr3ab", "others_cdr3b", "others_longab"):
        protocol = TCRBindingQueryProtocol(input_mode=input_mode_for_track(track))
        tracks[track] = {
            "status": "cpu_identity_only",
            "input_protocol": protocol.provenance,
            "input_protocol_sha256": protocol.sha256,
        }
    binding = PIN.out / "binding_preflight/passed.json"
    write_json(binding, {
        "status": "passed",
        "scope": "binding",
        "checkpoint": str(PIN.ckpt),
        "checkpoint_sha256": PIN.sha,
        "global_step": PIN.step,
        "quality_evaluation": False,
        "gate_kind": "cpu_identity_from_snapshot",
        "tracks": tracks,
    })
    write_json(PIN.out / "repr_preflight_v2/passed.json", {
        "status": "passed",
        "quality_evaluation": False,
        "gate_kind": "cpu_identity_from_snapshot",
        "tasks": {task: {"identity": repr_identity(task)} for task in DATASETS},
    })
    generation = {
        "status": "passed",
        "identity": generation_identity(),
        "quality_evaluation": False,
        "gate_kind": "cpu_identity_from_snapshot",
    }
    write_json(PIN.out / "generation_preflight/passed.json", generation)
    write_json(PIN.out / "generation_preflight_v2/passed.json", generation)
    print(f"wrote CPU identity gates under {PIN.out}")


if __name__ == "__main__":
    main()
