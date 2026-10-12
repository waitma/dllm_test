#!/usr/bin/env python
"""Smoke: post-LLaDA GrammarEmbedder canonical口径 after the tcr_pair/zero-fill fix.

Checks, on the default fusion smoke checkpoint (override as ``argv[1]``):
  1. strict load info (0 missing / 0 unexpected via load_grammar_checkpoint),
  2. single-seq embed() (tcr_single, post-LLaDA) -> finite, non-degenerate,
  3. embed_pairs() with **mixed** alpha (some rows β-only) -> no KeyError,
  4. determinism on re-embed.

The original version of check 3 also asserted that β-only rows carry an
all-zero alpha block. That is a ``pool_mode="segment_concat"`` property, and
``segment_concat`` only ever existed on the retired ``grammar_v2`` ``.pt``
checkpoints (deleted 2026-09-08). ``FusionGrammarEmbedder`` rejects anything
but ``decoder`` + ``global`` mean-pool, which emits one vector over the whole
joint record with no per-chain blocks, so that assertion was dropped rather
than silently reinterpreted. Pass a restored ``.pt`` to exercise segment mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from common import SMOKE_FUSION_CHECKPOINT  # noqa: E402
from common.model_api import build_embedder  # noqa: E402


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else SMOKE_FUSION_CHECKPOINT
    emb = build_embedder(f"grammar:decoder:{ckpt}")
    print(f"name={emb.name} feature_source={emb.feature_source} hidden={emb.hidden} enc_hidden={emb.enc_hidden}")

    # 1) single-seq post-LLaDA
    seqs = ["CASSLGTGELFF", "CASSIRSSYEQYF", "CASSPQGADTQYF", "GILGFVFTL"]
    x = emb.embed(seqs)
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    off = (xn @ xn.T)[~np.eye(len(x), dtype=bool)]
    print(f"[single] shape={x.shape} finite={np.isfinite(x).all()} "
          f"offdiag_cos[min/mean/max]={off.min():.3f}/{off.mean():.3f}/{off.max():.3f}")

    # 2) mixed-alpha embed_pairs (rows 1,3 β-only) -- the case that crashed nck2s
    beta = ["CASSLGTGELFF", "CASSIRSSYEQYF", "CASSPQGADTQYF", "CASSLAPGATNEKLFF"]
    alpha = ["CAVRDSNYQLIW", "", "CAGAGSQGNLIF", ""]
    xp = emb.embed_pairs(beta=beta, alpha=alpha, peptide=None, order=("beta", "alpha"))
    print(f"[pairs ] shape={xp.shape} finite={np.isfinite(xp).all()} "
          f"pool_mode={emb.pool_mode}")

    if emb.pool_mode == "segment_concat":
        half = xp.shape[1] // 2
        # β-only rows (idx 1,3) must have a zero alpha block (second half)
        z1 = float(np.abs(xp[1, half:]).sum())
        z3 = float(np.abs(xp[3, half:]).sum())
        a0 = float(np.abs(xp[0, half:]).sum())
        print(f"[pairs ] alpha-block |sum|: paired row0={a0:.3g} | "
              f"β-only row1={z1:.3g} row3={z3:.3g} (β-only alpha block should be 0)")
        blocks_ok = z1 == 0.0 and z3 == 0.0 and a0 > 0
    else:
        # global mean-pool emits one whole-record vector: no per-chain blocks to
        # check, but β-only rows must still differ from paired rows.
        blocks_ok = not np.allclose(xp[0], xp[1], atol=1e-4)
        print(f"[pairs ] global pool: paired row0 differs from β-only row1 = {blocks_ok}")

    # 3) determinism
    xp2 = emb.embed_pairs(beta=beta, alpha=alpha, peptide=None, order=("beta", "alpha"))
    same = np.allclose(xp, xp2, atol=1e-4)

    ok = (np.isfinite(x).all() and np.isfinite(xp).all() and off.std() > 1e-4
          and blocks_ok and same)
    print(f"\nSMOKE {'PASS' if ok else 'FAIL'} (determinism={same})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
