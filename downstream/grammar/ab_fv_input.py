"""Audited Fv inputs for fixed-v2 probes; run via ab_probes --input-policy fixed-v2-fv.

Build the boundary manifest with scripts/downstream/prepare_ab_fv_input.py first.
Raw benchmark rows, targets and folds are never changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/downstream/ab_fixed_v2_fv_manifest.json"


def adapt_pairs(task, pairs, *, manifest_path=MANIFEST):
    manifest = json.loads(Path(manifest_path).read_text())
    result = []
    changed = 0
    for pair in pairs:
        output = []
        for role, seq, cap in zip(("h", "l"), pair, (166, 134)):
            if task == "m396":
                rule = manifest["m396"][role]
                start, end = rule["start"], rule["end"]
                wt = rule["sequence"]
                if len(seq) != len(wt) or seq[:start] != wt[:start] or seq[end:] != wt[end:]:
                    raise ValueError("m396 mutation/length change outside audited Fv; refusing to discard it")
                value = seq[start:end]
            elif len(seq) > cap:
                key = hashlib.sha256(seq.encode()).hexdigest()
                rule = manifest["overflow"].get(key)
                if rule is None or rule["role"] != role or rule["sequence"] != seq:
                    raise ValueError("Unverified oversized antibody; rebuild ANARCI boundary manifest")
                value = seq[rule["start"]:rule["end"]]
            else:
                value = seq
            if not value or len(value) > cap:
                raise ValueError(f"Fv does not fit fixed-v2 {role} canvas")
            output.append(value)
        result.append(tuple(output))
        changed += tuple(output) != tuple(pair)
    return result, {"policy": "fixed-v2-fv", "boundary_method": "ANARCI_IMGT",
                    "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
                    "changed_pairs": changed, "n_pairs": len(pairs),
                    "m396_outside_fv_mutations": 0 if task == "m396" else None}
