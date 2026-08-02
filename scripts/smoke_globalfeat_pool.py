#!/usr/bin/env python
"""Smoke test: pool_mode=global returns [N,H] and differs from segment concat."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "downstream" / "benchmark"))

CKPT = ROOT / "output/grammar_v2_esmc300m_cmp500k_llada/best.pt"


def main() -> int:
    from common.model_api import build_embedder

    if not CKPT.exists():
        print(f"MISSING checkpoint: {CKPT}")
        return 1

    beta = ["CASSLGQGAEAFF", "CASSQETQYF"]
    alpha = ["CAVRDSSYKLIF", "CAVNFGGGKLIF"]
    peptide = ["GILGFVFTL", "NLVPMVATV"]

    global_emb = build_embedder(f"grammar:decoder:global:{CKPT}")
    seg_emb = build_embedder(f"grammar:decoder:segment:{CKPT}")

    # T1-style joint β+peptide
    g_bind = global_emb.embed_pairs(beta=beta, alpha=None, peptide=peptide)
    s_bind = seg_emb.embed_pairs(beta=beta, alpha=None, peptide=peptide, order=("beta", "peptide"))
    print(f"T1 bind  global shape={g_bind.shape}  segment_concat shape={s_bind.shape}")
    assert g_bind.shape == (2, global_emb.hidden), g_bind.shape
    assert s_bind.shape[1] == 2 * global_emb.hidden, s_bind.shape
    assert not np.allclose(g_bind, s_bind[:, : global_emb.hidden]), "global must differ from beta segment only"

    # T3-style paired TCR (no peptide)
    g_pair = global_emb.embed_pairs(beta=beta, alpha=alpha, peptide=None)
    s_pair = seg_emb.embed_pairs(beta=beta, alpha=alpha, peptide=None, order=("beta", "alpha"))
    print(f"T3 pair  global shape={g_pair.shape}  segment_concat shape={s_pair.shape}")
    assert g_pair.shape == (2, global_emb.hidden), g_pair.shape
    assert s_pair.shape[1] == 2 * global_emb.hidden, s_pair.shape
    h = global_emb.hidden
    seg_b, seg_a = s_pair[:, :h], s_pair[:, h:]
    assert not np.allclose(g_pair, seg_b), "global != beta segment pool"
    assert not np.allclose(g_pair, seg_a), "global != alpha segment pool"
    assert not np.allclose(g_pair, (seg_b + seg_a) / 2), "global != mean of segment pools"
    print(f"T3 pair  global distinct from per-segment pools (max|Δ|="
          f"{max(np.max(np.abs(g_pair - seg_b)), np.max(np.abs(g_pair - seg_a))):.4f})")

    # Default spec (no pool tag) must be global
    default_emb = build_embedder(f"grammar:decoder:{CKPT}")
    assert default_emb.pool_mode == "global", default_emb.pool_mode
    g_def = default_emb.embed_pairs(beta=beta[:1], alpha=alpha[:1], peptide=None)
    assert g_def.shape == (1, default_emb.hidden)

    print("SMOKE OK: global pool [N,H], distinct from segment concat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
