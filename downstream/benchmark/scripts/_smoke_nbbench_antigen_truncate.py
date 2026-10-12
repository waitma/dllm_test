#!/usr/bin/env python
"""Smoke: 1273-aa antigen through NbBench featurize + post-LLaDA grammar embed (no crash)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import SMOKE_FUSION_CHECKPOINT  # noqa: E402
from nbbench.run import featurize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=SMOKE_FUSION_CHECKPOINT)
    args = ap.parse_args()
    ckpt = args.checkpoint
    # Fusion checkpoints are Trainer directories; retired grammar_v2 ones were
    # single .pt files. Accept either so a restored .pt still works.
    if not Path(ckpt).exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    long_ag = "A" * 1273
    vhh = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK"
    train = pd.DataFrame({"VHH_sequence": [vhh], "Ag_sequence": [long_ag], "label": [0]})
    test = pd.DataFrame({"VHH_sequence": [vhh], "Ag_sequence": [long_ag], "label": [1]})
    emb = f"grammar:decoder:{ckpt}"

    xtr, xte, name = featurize(emb, train, test, ("VHH_sequence", "Ag_sequence"))
    assert xtr.shape == xte.shape == (1, xtr.shape[1])
    assert xtr.shape[1] > 0
    print(f"ok: embedder={name} xtr.shape={xtr.shape} finite={bool(xtr.flags.writeable)}")


if __name__ == "__main__":
    main()
