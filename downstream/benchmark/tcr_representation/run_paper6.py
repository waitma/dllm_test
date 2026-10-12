#!/usr/bin/env python
"""T3-deep -- SCEPTR (Cell Systems 2024) 6-pMHC deep few-shot benchmark.

Reproduces the SCEPTR Methods III.1 / Table SI protocol on its own data
(``data/tcr_representation_paper6/``), independent of the broad 24-epitope T3.

Universe = ``target_binders.csv`` (6 targets, deduped binders) ++
``background_pool.csv`` (rest of filtered VDJdb, pure negatives). For each
target ``e`` and shot ``k`` we sample ``k`` reference binders, score every
universe row by nearest-neighbour closeness to the reference, and take the
per-pMHC AUROC of ``e``'s held-out binders vs **the entire background** (paper
口径, not the other-target-only negatives of the broad T3).

Backends (all official source / official weights, via existing wrappers):
- ``levenshtein`` : CDR3b(+CDR3a) Levenshtein NN (rapidfuzz).
- ``tcrdist``     : official tcrdist3 rectangular distances (universe TCRrep
                    built once, then rectangular distances to each epitope).
- ``sceptr``      : official SCEPTR 64-d vectors + L2 (euclidean, == cdist p=2).
- ``embed``       : an embedder via ``--embedder`` (esm2_150m / protbert /
                    tcrbert / bioseq:/abs/best.pt), per-column embeddings
                    concatenated, cosine closeness.

Results -> ``outputs/tcr_representation_paper6/<tag>/fewshot.json`` with the full
per-pMHC × per-k AUROC grid + the 6-pMHC macro mean.

Examples::

    python tcr_representation/run_paper6.py --method levenshtein
    python tcr_representation/run_paper6.py --method tcrdist
    python tcr_representation/run_paper6.py --method sceptr
    python tcr_representation/run_paper6.py --method embed --embedder esm2_150m
    python tcr_representation/run_paper6.py --method embed --embedder tcrbert --columns cdr3b cdr3a
    python tcr_representation/run_paper6.py --method embed --embedder bioseq:/abs/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import fewshot as fs  # noqa: E402
from common.provenance import checkpoint_provenance  # noqa: E402

DATA = BENCH / "data" / "tcr_representation_paper6"
OUT = BENCH / "outputs" / "tcr_representation_paper6"
TARGETS = ["GILGFVFTL", "NLVPMVATV", "SPRWYFYYL", "TFEYVSQPFLMDLE", "TTDPSFLGRY", "YLQPRTFLL"]
DEFAULT_SHOTS = [1, 2, 5, 10, 20, 50, 100, 200]


def _load_universe():
    """target_binders first (so binder indices are contiguous), then background."""
    tgt = pd.read_csv(DATA / "target_binders.csv").fillna("")
    bg = pd.read_csv(DATA / "background_pool.csv").fillna("")
    uni = pd.concat([tgt, bg], ignore_index=True)
    return uni, tgt, bg


def _tag_for(method: str, embedder: str | None) -> str:
    if method != "embed":
        return method
    spec = embedder or ""
    if spec.startswith("grammar:"):
        # grammar:decoder:/abs/best.pt  or  grammar:/abs/best.pt (defaults decoder)
        rest = spec.split(":", 1)[1]
        if rest.split(":", 1)[0] in ("decoder", "encoder"):
            source, path = rest.split(":", 1)
        else:
            source, path = "decoder", rest
        ckpt = Path(path)
        run = (ckpt.parent.name or ckpt.stem).replace("grammar_v2_", "")
        # decoder = post-LLaDA joint peptide-free paired record (grammar_name
        # "tcr_pair"); encoder = ESMC condition mean-pool. Tag reflects口径.
        return f"grammar_{'tcrpair' if source == 'decoder' else 'enc'}_{run}"
    if spec.startswith("bioseq:"):
        ckpt = Path(spec.split(":", 1)[1])
        return "bioseq_" + (ckpt.parent.name or ckpt.stem)
    return spec.replace("/", "_").replace(":", "_")


# ---------------------------------------------------------------------------
# Backend builders -> either universe_vec (+metric) or a pairwise_fn(cand_idx)
# ---------------------------------------------------------------------------

def _editdist_pairwise(uni: pd.DataFrame, columns):
    from rapidfuzz.distance import Levenshtein as Lev
    from rapidfuzz import process as rp
    cols = {c: uni[c].fillna("").astype(str).tolist() for c in columns}

    def pairwise_fn(cand_idx):
        d = None
        for c in columns:
            full = cols[c]
            ref = [full[i] for i in cand_idx]
            m = rp.cdist(full, ref, scorer=Lev.distance, dtype=np.float32)
            d = m if d is None else d + m
        return d
    return pairwise_fn


def _tcrdist_pairwise(uni: pd.DataFrame, chains=("alpha", "beta")):
    """Official tcrdist3 rectangular distances; universe germline built once."""
    from tcrdist.repertoire import TCRrep
    from common.model_api import TcrdistDistance

    uni_td = TcrdistDistance._to_tcrdist_df(uni)
    tr = TCRrep(cell_df=uni_td, organism="human", chains=list(chains),
                compute_distances=False, deduplicate=False)
    universe_clones = tr.clone_df.where(tr.clone_df.notnull(), "")
    assert len(universe_clones) == len(uni), "tcrdist clone_df misaligned with universe"

    def pairwise_fn(cand_idx):
        cand_clones = universe_clones.iloc[cand_idx]
        tr.compute_rect_distances(df=universe_clones, df2=cand_clones)
        d = None
        for ch in chains:
            m = np.asarray(getattr(tr, f"rw_{ch}"), dtype=np.float32)
            d = m if d is None else d + m
        return d
    return pairwise_fn


def _sceptr_vectors(uni: pd.DataFrame):
    from common.model_api import build_distance_source
    src = build_distance_source("sceptr")
    vec = src.embed(uni)                     # [N, 64] SCEPTR representation
    return np.asarray(vec, dtype=np.float32), src.name


# Fields each non-embed backend actually consumes, independent of ``--columns``.
# SCEPTR takes TRAV/CDR3A/TRBV/CDR3B (``ScepterSource._to_sceptr_df``) and
# tcrdist3 additionally takes both J genes (``TcrdistSource``), so neither is a
# CDR3-only method even though ``--columns`` defaults to the two CDR3 loops.
_INPUT_FIELDS = {
    "levenshtein": ["cdr3b", "cdr3a"],
    "sceptr": ["cdr3b", "cdr3a", "vb", "va"],
    "tcrdist": ["cdr3b", "cdr3a", "vb", "va", "jb", "ja"],
}


def _embed_vectors(uni: pd.DataFrame, embedder_spec: str, columns):
    """Per-column embed over unique strings, concatenate (matches broad T3).

    Exception: a **grammar decoder** embedder (``feature_source == "decoder"``,
    our LLaDA multi-chain representation) is not a per-column encoder — it is
    routed through ``fs.grammar_pair_embed``, which renders each TCR in the
    training grammar format as a β(+α) record, runs the clean/unmasked
    LLaDA-decoder forward, and mean-pools the last-layer hidden states over the
    whole record. All other embedders (esm2/protbert/tcrbert/kmer/bioseq
    encoder-only) keep the original per-column path.

    On epitope leakage: ``grammar_pair_embed`` passes ``peptide=None``, which
    ``embed_pairs`` turns into an empty string, and its ``order`` omits the
    ``peptide`` segment altogether — so the epitope is absent from the rendered
    record rather than present as a constant placeholder. Input fields are
    exactly ``cdr3b`` (+ ``cdr3a``); V/J genes are never supplied.
    """
    if embedder_spec == "kmer":
        from common.featurizers import KmerFeaturizer
        kf = KmerFeaturizer(k=3)
        emb = type("K", (), {"embed": lambda self, s: kf.transform([str(x) for x in s]),
                             "name": "kmer-3"})()
    else:
        from common.model_api import build_embedder
        emb = build_embedder(embedder_spec)

    if getattr(emb, "feature_source", None) == "decoder":
        vec = fs.grammar_pair_embed(emb, uni, columns)
        return np.asarray(vec, dtype=np.float32), getattr(emb, "name", embedder_spec)

    blocks = []
    for c in columns:
        vals = uni[c].fillna("").astype(str)
        uniq = sorted(set(vals))
        e = emb.embed(uniq)
        lut = {u: e[i] for i, u in enumerate(uniq)}
        blocks.append(np.stack([lut[v] for v in vals]))
    vec = np.concatenate(blocks, axis=1).astype(np.float32)
    return vec, getattr(emb, "name", embedder_spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["levenshtein", "tcrdist", "sceptr", "embed"],
                    default="levenshtein")
    ap.add_argument("--embedder", default=None,
                    help="for --method embed: esm2_150m / protbert / tcrbert / kmer / "
                         "bioseq:/abs/best.pt (ESMC encoder-only mean-pool) / "
                         "grammar:decoder:/abs/best.pt (LLaDA decoder, multi-chain, our correct口径)")
    ap.add_argument("--columns", nargs="+", default=["cdr3b", "cdr3a"])
    ap.add_argument("--shots", nargs="+", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--max-ref-pool", type=int, default=2000,
                    help="cap reference-candidate binders per target (0=no cap)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.method == "embed" and not args.embedder:
        ap.error("--method embed requires --embedder")

    uni, tgt, bg = _load_universe()
    peptides = uni["peptide"].to_numpy()
    tag = args.tag or _tag_for(args.method, args.embedder)
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    max_ref_pool = None if args.max_ref_pool in (0, None) else int(args.max_ref_pool)

    n_by_target = {t: int((peptides == t).sum()) for t in TARGETS}
    print(f"[paper6] method={args.method} tag={tag} universe={len(uni)} "
          f"(targets={len(tgt)} background={len(bg)}) columns={args.columns}")
    print(f"[paper6] per-target binders: {n_by_target}")

    t0 = time.time()
    universe_vec = None
    pairwise_fn = None
    metric = "cosine"
    model_name = args.method

    if args.method == "levenshtein":
        pairwise_fn = _editdist_pairwise(uni, args.columns)
        model_name = "CDR3-Levenshtein-NN"
    elif args.method == "tcrdist":
        pairwise_fn = _tcrdist_pairwise(uni)
        model_name = "TCRdist"
    elif args.method == "sceptr":
        universe_vec, model_name = _sceptr_vectors(uni)
        metric = "euclidean"
    elif args.method == "embed":
        universe_vec, model_name = _embed_vectors(uni, args.embedder, args.columns)
        metric = "cosine"
    build_sec = round(time.time() - t0, 1)
    print(f"[paper6] backend ready ({build_sec}s) model={model_name} metric={metric}")

    t1 = time.time()
    res = fs.few_shot_paper_auroc(
        peptides, TARGETS, shots=args.shots, seeds=args.seeds,
        universe_vec=universe_vec, metric=metric, pairwise_fn=pairwise_fn,
        max_ref_pool=max_ref_pool,
    )
    res["model"] = model_name
    res["method"] = args.method
    res["embedder"] = args.embedder
    res["columns"] = list(args.columns)
    # ``--columns`` only drives the per-column embed / editdist paths. The
    # ``sceptr`` and ``tcrdist`` backends are handed the whole dataframe and read
    # germline calls straight out of it, so recording ``args.columns`` for them
    # understates what they saw. Store the fields actually consumed so the
    # artifact cannot be misread as an equal-input comparison.
    res["input_fields"] = _INPUT_FIELDS.get(args.method, list(args.columns))
    res["metric"] = metric
    res["tag"] = tag
    res["n_universe"] = int(len(uni))
    res["n_background"] = int(len(bg))
    res["per_target_binders"] = n_by_target
    res["max_ref_pool"] = max_ref_pool
    res["build_sec"] = build_sec
    res["score_sec"] = round(time.time() - t1, 1)
    if args.method == "embed" and args.embedder:
        prov = checkpoint_provenance(args.embedder)
        if prov is not None:
            res["checkpoint"] = prov

    with (out_dir / "fewshot.json").open("w") as fh:
        json.dump(res, fh, indent=2)

    print(f"\n=== paper6 deep few-shot per-pMHC NN AUROC :: {model_name} "
          f"(build {build_sec}s + score {res['score_sec']}s) ===")
    hdr = "k".ljust(6) + "macro".rjust(9) + "".join(t[:6].rjust(9) for t in TARGETS)
    print(hdr)
    print("-" * len(hdr))
    for k in res["shots"]:
        s = res["auroc_by_shot"][k]
        row = str(k).ljust(6) + f"{s['mean']:.3f}".rjust(9)
        pe = res["per_epitope_by_shot"][k]
        for t in TARGETS:
            v = pe.get(t)
            row += (f"{v['mean']:.3f}".rjust(9) if v else "-".rjust(9))
        print(row)
    print(f"  -> {out_dir}/fewshot.json")


if __name__ == "__main__":
    main()
