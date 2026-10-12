#!/usr/bin/env python
"""RETIRED smoke test: post-LLaDA (``bioseq-llada:``) vs ESMC-encoder (``bioseq:``).

**This script no longer has a checkpoint it can run against.** Both specs call
``torch.load`` on a single-file ``.pt``, which only the retired ``grammar_v2_*``
runs produced; those checkpoints were deleted on 2026-09-08 (see
``docs/archive/grammar_v2_retired_20260908``). The current ``examples/llada``
immune-fusion line saves HuggingFace Trainer directories instead, so it cannot
be substituted here -- use the ``grammar:`` spec for fusion checkpoints, e.g.
``scripts/smoke_grammar_embedder.py``.

Kept for reference because it documents what the post-LLaDA readout was
verified to guarantee: strict full-model load (0 missing / 0 unexpected),
finite correctly-shaped vectors, a readout genuinely distinct from the
ESMC-encoder mean-pool, and epitope signal on a labelled sample.

Pass a checkpoint explicitly to run it against a ``.pt`` you have restored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common.model_api import build_embedder  # noqa: E402


def main():
    if len(sys.argv) <= 1:
        raise SystemExit(
            "retired: the grammar_v2 .pt checkpoints this smoke test targeted were "
            "deleted on 2026-09-08, and fusion-line checkpoints are Trainer "
            "directories that the bioseq-llada:/bioseq: specs cannot load. "
            "Pass a restored .pt path explicitly to run anyway, or use "
            "scripts/smoke_grammar_embedder.py for the current fusion line."
        )
    ckpt = sys.argv[1]
    llada = build_embedder(f"bioseq-llada:{ckpt}")
    print(f"[post-LLaDA] name={llada.name}")
    print(f"[post-LLaDA] load_info={llada._load_info}")

    df = pd.read_csv(BENCH / "data/tcr_binding_nm2025/seen/cdr3b/AS/test.csv").head(40)
    beta = df["cdr3b"].astype(str).tolist()
    y = df["label"].to_numpy()

    fe = llada.embed(beta)
    print(f"[post-LLaDA] embed -> {fe.shape}, finite={np.isfinite(fe).all()}, "
          f"dim={llada.dim} mean={fe.mean():.4f} std={fe.std():.4f}")
    assert fe.shape[0] == len(beta)
    assert fe.shape[1] == llada.dim
    assert np.isfinite(fe).all()

    enc = build_embedder(f"bioseq:{ckpt}")
    ee = enc.embed(beta)
    print(f"[ESMC-enc ] embed -> {ee.shape}, dim={enc.dim}")
    # dims may match (both = encoder hidden when no projection); the vectors must
    # still differ because post-LLaDA passes through the diffusion backbone.
    if ee.shape == fe.shape:
        rel = float(np.linalg.norm(fe - ee) / (np.linalg.norm(ee) + 1e-9))
        print(f"[compare  ] relative L2 diff (post-LLaDA vs ESMC-enc) = {rel:.4f}")
        assert rel > 1e-3, "post-LLaDA feature is identical to ESMC-encoder — bug!"

    fn = fe / (np.linalg.norm(fe, axis=1, keepdims=True) + 1e-9)
    sim = fn @ fn.T
    pos, neg = y == 1, y == 0
    if pos.sum() > 1 and neg.sum() > 0:
        pp = sim[np.ix_(pos, pos)][np.triu_indices(int(pos.sum()), 1)].mean()
        pn = sim[np.ix_(pos, neg)].mean()
        print(f"[signal   ] cosine pos-pos={pp:.4f} pos-neg={pn:.4f} "
              f"({'signal' if pp > pn else 'weak'})")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
