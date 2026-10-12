"""OLGA human TCR-beta wrapper: unconditional sampling + generation probability.

Uses the official ``statbiophys/OLGA`` code and its shipped ``human_T_beta``
default model (no retraining). Provides:

  * ``sample_cdr3b(n)``  -- n productive CDR3b amino-acid sequences.
  * ``pgen(seq)``        -- P_gen of a CDR3b amino-acid sequence.

pgen is the generation probability under the VDJ recombination model; it
calibrates whether generated CDR3b look like they came from a real repertoire
(too-high pgen = generic/public, zero pgen = non-recombinable / invalid).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
OLGA_REPO = BENCH / "baselines" / "OLGA"
OLGA_MODEL = OLGA_REPO / "olga" / "default_models" / "human_T_beta"


@lru_cache(maxsize=1)
def _load():
    if str(OLGA_REPO) not in sys.path:
        sys.path.insert(0, str(OLGA_REPO))
    import olga.load_model as load_model
    import olga.sequence_generation as seq_gen
    import olga.generation_probability as pgen_mod

    params = str(OLGA_MODEL / "model_params.txt")
    marginals = str(OLGA_MODEL / "model_marginals.txt")
    v_anchor = str(OLGA_MODEL / "V_gene_CDR3_anchors.csv")
    j_anchor = str(OLGA_MODEL / "J_gene_CDR3_anchors.csv")

    genomic_data = load_model.GenomicDataVDJ()
    genomic_data.load_igor_genomic_data(params, v_anchor, j_anchor)
    generative_model = load_model.GenerativeModelVDJ()
    generative_model.load_and_process_igor_model(marginals)

    sg = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)
    pg = pgen_mod.GenerationProbabilityVDJ(generative_model, genomic_data)
    return sg, pg


def sample_cdr3b(n: int, seed: int | None = None) -> list[str]:
    """Sample ``n`` productive CDR3b amino-acid sequences from OLGA human_T_beta."""
    sg, _ = _load()
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
    out = []
    for _ in range(int(n)):
        _, aa, _, _ = sg.gen_rnd_prod_CDR3()
        if aa:
            out.append(aa)
    return out


def pgen(seq: str) -> float:
    """Generation probability of a CDR3b amino-acid sequence (0 if non-recombinable)."""
    _, pg = _load()
    try:
        return float(pg.compute_aa_CDR3_pgen(seq))
    except Exception:
        return 0.0


def pgen_stats(seqs, cap: int = 3000, seed: int = 0) -> dict:
    """Summary pgen stats over a (subsampled) sequence list.

    Reports fraction with pgen>0 (recombinable / valid), and mean/median log10
    pgen over the recombinable subset.
    """
    import numpy as np

    seqs = [s for s in seqs if s]
    if not seqs:
        return {"n": 0, "frac_pgen_positive": float("nan"),
                "mean_log10_pgen": float("nan"), "median_log10_pgen": float("nan")}
    rng = np.random.default_rng(seed)
    if len(seqs) > cap:
        seqs = list(rng.choice(seqs, size=cap, replace=False))
    vals = [pgen(s) for s in seqs]
    pos = [v for v in vals if v and v > 0]
    logs = np.log10(pos) if pos else np.array([])
    return {
        "n": int(len(seqs)),
        "frac_pgen_positive": float(len(pos) / len(vals)),
        "mean_log10_pgen": float(np.mean(logs)) if len(logs) else float("nan"),
        "median_log10_pgen": float(np.median(logs)) if len(logs) else float("nan"),
    }
