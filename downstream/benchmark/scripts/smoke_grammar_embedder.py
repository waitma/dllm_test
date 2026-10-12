#!/usr/bin/env python
"""Smoke test: the grammar embedder loads and yields finite, separable features.

Usage (single GPU)::

    cd downstream/benchmark
    python scripts/smoke_grammar_embedder.py                 # decoder features
    python scripts/smoke_grammar_embedder.py encoder         # encoder condition
    python scripts/smoke_grammar_embedder.py decoder <ckpt>  # explicit checkpoint

The default checkpoint is an immune-fusion directory, so ``build_embedder``
routes to ``FusionGrammarEmbedder``. The retired ``grammar_v2_*`` ``.pt``
checkpoints this script originally used were deleted on 2026-09-08.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common.model_api import build_embedder  # noqa: E402

CKPT = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/"
    "protein_esmc_llada270m_diffusion_immune_v3_8gpu_2m/eval_snapshot_44000"
)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "decoder"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else CKPT
    if not Path(ckpt).exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    emb = build_embedder(f"grammar:{src}:{ckpt}")
    print(f"embedder={emb.name} feature_source={emb.feature_source} "
          f"hidden={emb.hidden} enc_hidden={emb.enc_hidden}")

    # A handful of real AS-seen rows. The file is label-sorted, so take a
    # balanced slice from each class -- head(24) would be all positives and
    # would make the pos-vs-neg cosine below undefined.
    full = pd.read_csv(BENCH / "data/tcr_binding_nm2025/seen/cdr3b/AS/test.csv")
    df = pd.concat(
        [full[full["label"] == 1].head(12), full[full["label"] == 0].head(12)]
    ).reset_index(drop=True)
    beta = df["cdr3b"].tolist()
    pep = df["peptide"].tolist()
    feats = emb.embed_pairs(beta, alpha=None, peptide=pep)
    print(f"embed_pairs -> {feats.shape}, finite={np.isfinite(feats).all()}, "
          f"dim={emb.dim} mean={feats.mean():.4f} std={feats.std():.4f}")
    assert feats.shape[0] == len(df)
    assert np.isfinite(feats).all()
    assert feats.shape[1] == emb.dim

    # cosine separability: mean pos-vs-pos vs pos-vs-neg similarity
    y = df["label"].to_numpy()
    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    sim = fn @ fn.T
    pos = y == 1
    neg = y == 0
    pp = sim[np.ix_(pos, pos)][np.triu_indices(pos.sum(), 1)].mean()
    pn = sim[np.ix_(pos, neg)].mean()
    print(f"cosine: pos-pos={pp:.4f} pos-neg={pn:.4f} (pos-pos>pos-neg indicates signal)")

    # single-seq embed() fallback
    e1 = emb.embed(beta[:4])
    print(f"embed(single) -> {e1.shape}, finite={np.isfinite(e1).all()}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
