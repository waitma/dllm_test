#!/usr/bin/env python
"""Run TCR-VALID (Nat Commun 2024) on our NAR-GAB curated clustering dataset.

Runs inside the dedicated ``tcrvalid`` conda env (TF 2.8 / py3.8). Two outputs:

1. TCR-VALID **latent embeddings** of every paired TCR's TRB chain, saved as
   ``outputs/tcr_clustering/tcrvalid/latents.npy`` (row-aligned to
   ``data/tcr_clustering/tcrs.csv``). These feed our unified ``embed-threshold``
   Purity-Retention protocol in the main env, comparable to Ours-BioSeq.

2. TCR-VALID's **native clustering protocol** on those latents: the paper's own
   ``dbscan_loop`` (DBSCAN over the VAE latent, manhattan metric) scored with the
   paper's own ``clustering_scoring`` -> ``tcrvalid_native/native_scores.csv``.

Input-format adaptation (documented honestly): TCR-VALID consumes ``CDR2-CDR3core``
strings. We derive CDR2beta from the pair's TRBV gene via the packaged
``TRBV_reference.csv`` and strip the conserved C.../.F|W anchors from CDR3beta to
get the core. This is the standard TCR-VALID featurization; it differs from the
NAR-GAB CDR3-only clustering, so TCR-VALID is reported as an additional
representation baseline, not a NAR-GAB re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH / "baselines" / "TCR-VALID"
DATA = BENCH / "data" / "tcr_clustering"
OUT = BENCH / "outputs" / "tcr_clustering"
TRB_REF = REPO / "tcrvalid" / "data" / "TRBV_reference.csv"

TRB_MODEL = "1_2_full_40"       # final full-data TRB model (paper's headline TRB)


def _core_cdr3(s: str) -> str:
    """Strip the conserved leading C and trailing F/W anchors -> CDR3 core."""
    s = str(s)
    if s and s[0] == "C":
        s = s[1:]
    if s and s[-1] in "FW":
        s = s[:-1]
    return s


def build_features(pairs: pd.DataFrame):
    ref = pd.read_csv(TRB_REF)
    v2cdr2 = dict(zip(ref["new_meta_vcall"].astype(str), ref["cdr2_no_gaps"].astype(str)))
    # our vb values look like 'TRBV5-6' == new_meta_vcall; try a *01 fallback too.
    ref01 = dict(zip(ref["new_v_call"].astype(str), ref["cdr2_no_gaps"].astype(str)))

    cdr2, ok = [], 0
    for vb in pairs["vb"].astype(str):
        c = v2cdr2.get(vb) or ref01.get(vb + "*01") or ref01.get(vb)
        if c is not None:
            ok += 1
        cdr2.append(c if c is not None else "")
    pairs = pairs.copy()
    pairs["cdr2b"] = cdr2
    pairs["pre_feature"] = pairs["cdr2b"] + "-" + pairs["cdr3b"].map(_core_cdr3)
    print(f"V->CDR2beta mapped for {ok}/{len(pairs)} pairs "
          f"({100*ok/len(pairs):.1f}%); unmapped get empty CDR2.")
    return pairs


def main():
    import sys
    sys.path.insert(0, str(REPO))
    from tcrvalid.load_models import load_named_models
    from tcrvalid.physio_embedding import SeqArrayDictConverter
    from tcrvalid.cluster_loop import dbscan_loop

    pairs = pd.read_csv(DATA / "tcrs.csv").fillna("")
    pairs = build_features(pairs)

    mapping = SeqArrayDictConverter()
    model = load_named_models(TRB_MODEL, chain="TRB", as_keras=True)[TRB_MODEL]

    x = mapping.seqs_to_array(pairs["pre_feature"].values, maxlen=28)
    z, _, _ = model.predict(x, batch_size=512)
    z = np.asarray(z, dtype=np.float32)
    print("TCR-VALID TRB latent shape:", z.shape)

    (OUT / "tcrvalid").mkdir(parents=True, exist_ok=True)
    (OUT / "tcrvalid_native").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "tcrvalid" / "latents.npy", z)
    pairs[["pair_id", "cdr3b", "peptide", "cdr2b"]].to_csv(
        OUT / "tcrvalid" / "row_order.csv", index=False)

    # --- native protocol: TCR-VALID's own DBSCAN loop + scoring ---
    df = pairs.rename(columns={"pair_id": "sequence_id"}).copy()
    df["clono_id"] = df["sequence_id"]
    eps_range = np.round(np.linspace(0.5, 10.0, 20), 2)
    scores = dbscan_loop(df, z, eps_range, metric="manhattan",
                         min_samples=2, eps_format=":.2f")
    scores.to_csv(OUT / "tcrvalid_native" / "native_scores.csv")
    keep = ["epsilon", "n_clusters", "percent_clustered", "mean_purity",
            "well_clustered", "well_clustered_total", "n_epitopes_captured"]
    print("\n=== TCR-VALID native DBSCAN scoring (their metrics, %) ===")
    with pd.option_context("display.width", 200):
        print(scores[keep].round(2).to_string(index=False))

    meta = {
        "model": f"TCR-VALID TRB {TRB_MODEL}",
        "repo": "github.com/peterghawkins-regn/tcrvalid",
        "commit": "475ed9647f924d60fa49ac6e83288f89b06275b1",
        "latent_dim": int(z.shape[1]),
        "n_pairs": int(len(pairs)),
        "input_format": "CDR2beta-CDR3beta_core (V-gene -> CDR2 via TRBV_reference)",
        "native_protocol": "DBSCAN(manhattan, min_samples=2) over VAE latent; "
                           "paper clustering_scoring (mean_purity / well_clustered / "
                           "percent_clustered, all in %).",
        "note": ("TCR-VALID uses CDR2-CDR3core featurization, differing from the "
                 "NAR-GAB CDR3-only clustering; reported as an additional "
                 "representation baseline. Latents also scored under our unified "
                 "embed-threshold protocol (outputs/tcr_clustering/tcrvalid/)."),
    }
    with (OUT / "tcrvalid_native" / "metrics.json").open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nsaved latents -> {OUT}/tcrvalid/latents.npy")
    print(f"native scores -> {OUT}/tcrvalid_native/native_scores.csv")


if __name__ == "__main__":
    main()
