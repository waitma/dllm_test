"""Validate completed pairing before score-only retry.

python scripts/downstream/validate_pairing_reuse.py --csv PATH --checkpoint PATH
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(csv, checkpoint):
    csv, checkpoint = Path(csv), Path(checkpoint)
    manifest = json.loads(csv.with_name(csv.stem + "_manifest.json").read_text())
    if Path(manifest["checkpoint_path"]).resolve() != checkpoint.resolve():
        raise ValueError("Pairing checkpoint path mismatch")
    if manifest["checkpoint_sha256"] != sha(checkpoint / "model.safetensors"):
        raise ValueError("Pairing checkpoint weights changed")
    if manifest["input_csv_sha256"] != sha(manifest["csv_path"]):
        raise ValueError("Pairing input changed")
    expected = dict(dataset_len=500, num_seqs=8, max_iter=16, light_prompt_tokens=0,
                    cfg_scale=0.0, temperature=1.0, seed=42, start_index=0,
                    light_length_mode="fixed_v2", sampling_strategy="gumbel_argmax")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Pairing protocol mismatch: {key}")
    rows = pd.read_csv(csv)
    source = pd.read_csv(manifest["csv_path"])
    if len(source) != 500 or len(rows) != 4000 or rows.gen_l_sequence.isna().any():
        raise ValueError("Incomplete pairing CSV")
    for i, (_, ref) in enumerate(source.iterrows()):
        block = rows.iloc[i * 8:(i + 1) * 8]
        if (not block.h_sequence.eq(ref[manifest['heavy_col']]).all()
                or not block.raw_l_sequence.eq(ref[manifest['light_col']]).all()
                or block.variant_idx.tolist() != list(range(8))):
            raise ValueError(f"Pairing row identity mismatch at heavy {i}")
    # Preserve original generation provenance; scoring does not regenerate tokens.
    return {"rows": len(rows), "csv_sha256": sha(csv), "generation_manifest": manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    audit = validate(args.csv, args.checkpoint)
    args.csv.with_name(args.csv.stem + "_reuse_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Validated {audit['rows']} existing samples; reuse for scoring")
