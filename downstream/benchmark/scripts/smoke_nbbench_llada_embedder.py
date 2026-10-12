#!/usr/bin/env python
"""Smoke: NbBench POST-LLaDA embedder (embed + embed_residues alignment).

Verifies that a spec loads a checkpoint, yields finite pooled/residue
features, and keeps 1:1 residue alignment (incl. CDR infilling gap ``-``).

Usage (single GPU, ~1–2 min on the 270m fusion snapshot):
    cd downstream/benchmark
    python scripts/smoke_nbbench_llada_embedder.py
    python scripts/smoke_nbbench_llada_embedder.py grammar <ckpt>

Only the ``grammar:`` spec runs by default. ``bioseq-llada:`` ``torch.load``s a
single ``.pt``, which only the retired ``grammar_v2_*`` runs produced (deleted
2026-09-08); pass such a path explicitly as the second argument to test it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import SMOKE_FUSION_CHECKPOINT  # noqa: E402
from common.model_api import build_embedder  # noqa: E402

# Short VHH-ish snippets: plain, with gap, with non-standard char.
SEQS = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK",
    "EVQLV--GGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK",
    "MKTA",
]


def _check_spec(label: str, spec: str) -> None:
    print(f"\n=== {label} spec={spec} ===")
    emb = build_embedder(spec)
    print(f"name={emb.name} dim={emb.dim} load={getattr(emb, '_load_info', {})}")

    pooled = emb.embed(SEQS)
    assert pooled.shape == (len(SEQS), emb.dim), pooled.shape
    assert np.isfinite(pooled).all(), "non-finite pooled features"

    residues = emb.embed_residues(SEQS)
    assert len(residues) == len(SEQS)
    for s, r in zip(SEQS, residues):
        assert r.ndim == 2 and r.shape[1] == emb.dim, r.shape
        assert np.isfinite(r).all(), "non-finite residue features"
        assert r.shape[0] == len(s), f"alignment {r.shape[0]} != {len(s)} for len={len(s)}"
        print(f"  len(seq)={len(s):4d} residues={r.shape} ok")

    # pooled ≈ mean(residues) for plain sequences (gap rows use placeholder 'A').
    for i, (s, r) in enumerate(zip(SEQS, residues)):
        if "-" not in s and all(c.isalpha() for c in s):
            diff = np.abs(pooled[i] - r.mean(0)).max()
            print(f"  pooled vs mean(residues) max|Δ|={diff:.6f}")
    print(f"{label} SMOKE OK")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "grammar"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else SMOKE_FUSION_CHECKPOINT
    if not Path(ckpt).exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    if mode == "bioseq-llada":
        if Path(ckpt).is_dir():
            raise SystemExit(
                "bioseq-llada: needs a single .pt checkpoint; the fusion default "
                "is a Trainer directory. Pass a restored grammar_v2 .pt explicitly."
            )
        _check_spec("bioseq-llada", f"bioseq-llada:{ckpt}")
    else:
        _check_spec("grammar:decoder", f"grammar:decoder:{ckpt}")
    print("\nALL SMOKE OK")


if __name__ == "__main__":
    main()
