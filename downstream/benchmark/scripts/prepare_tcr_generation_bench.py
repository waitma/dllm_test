#!/usr/bin/env python
"""Build the TCRT5-anchored TCR generation benchmark + BioSeq leakage report.

Main evaluation set (aligned with TCRT5, Nat Mach Intell 2025):
  * ``held20``       -- 20 target-rich held-out pMHC from ``topk_val_df.csv``
                        (the paper's held-out top-20 validation split).
  * ``benchmark14``  -- 14 sparse benchmark pMHC from ``benchmark_data_w_preds.csv``
                        (the paper's 13 sparse benchmark epitopes; the file ships
                        14 pMHC and stores official TCRT5/GRATCR/ER predictions,
                        which we keep for calibration cross-validation).

Outputs (``data/tcr_generation_bench/``):
  eval_conditional.json   {pMHC: {epitope, allele, mhc_pseudo, category,
                                  dataset, ref_binders, n_ref}}
  benchmark_preds.json    {pMHC: {tcrt5, gratcr, er, *_greedy}}  (14 only)
  train_binders.csv       epitope,cdr3b  conditioning/train pool (TCRT5 topk train)
  leakage_report.json     eval vs BioSeq training corpus + eval vs conditioning
  manifest.json           data versions, counts, K, seed provenance

Run:
    conda activate protenix_abtcr
    python downstream/benchmark/scripts/prepare_tcr_generation_bench.py
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/project_c20250601/251105016/project/dllm_test")
BENCH = Path(__file__).resolve().parents[1]
TCRT5 = BENCH / "baselines" / "TCRT5_tcr_translate"
OUT = BENCH / "data" / "tcr_generation_bench"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _norm(seq) -> str:
    return "".join(str(seq or "").split()).upper()


def _parse_list(cell):
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        return list(ast.literal_eval(cell))
    except Exception:
        return []


def load_hla_pseudo() -> dict[str, str]:
    ps = pd.read_csv(TCRT5 / "refs" / "hla_pseudo_seqs.csv", index_col=0)
    return {str(k): str(v) for k, v in ps["pseudo-sequence"].to_dict().items()}


def build_held20(pseudo_map: dict[str, str]) -> dict[str, dict]:
    """20 target-rich held-out pMHC: ref binders = unique CDR3b per pMHC."""
    val = pd.read_csv(TCRT5 / "data" / "topk_val_df.csv", low_memory=False)
    val = val[val["CDR3b"].notna()].copy()
    val["pMHC"] = val["Epitope"].astype(str) + "_" + val["Allele"].astype(str)
    entries = {}
    for (ep, allele), grp in val.groupby(["Epitope", "Allele"]):
        pmhc = f"{ep}_{allele}"
        binders = sorted({_norm(x) for x in grp["CDR3b"].tolist() if _norm(x)})
        entries[pmhc] = {
            "epitope": _norm(ep),
            "allele": str(allele),
            "mhc_pseudo": pseudo_map.get(str(allele), ""),
            "category": "held20",
            "dataset": "TCRT5_topk_val",
            "ref_binders": binders,
            "n_ref": len(binders),
        }
    return entries


def build_benchmark14(pseudo_map: dict[str, str]) -> tuple[dict[str, dict], dict[str, dict]]:
    """14 sparse benchmark pMHC + stored official predictions."""
    bm = pd.read_csv(TCRT5 / "data" / "benchmark_data_w_preds.csv", low_memory=False)
    entries, preds = {}, {}
    for _, row in bm.iterrows():
        ep = _norm(row["Epitope"])
        allele = f"HLA-{row['Allele']}" if not str(row["Allele"]).startswith("HLA") else str(row["Allele"])
        pmhc = f"{ep}_{row['Allele']}"
        refs = sorted({_norm(x) for x in _parse_list(row["reference_translations"]) if _norm(x)})
        entries[pmhc] = {
            "epitope": ep,
            "allele": allele,
            "mhc_pseudo": pseudo_map.get(allele, ""),
            "category": "benchmark14",
            "dataset": str(row["Dataset"]),
            "ref_binders": refs,
            "n_ref": len(refs),
        }
        preds[pmhc] = {
            "tcrt5": [_norm(x) for x in _parse_list(row["tcrt5_translations"])],
            "gratcr": [_norm(x) for x in _parse_list(row["gratcr_translations"])],
            "er": [_norm(x) for x in _parse_list(row["er_translations"])],
            "tcrt5_greedy": _norm(row.get("tcrt5_greedy")),
            "gratcr_greedy": _norm(row.get("gratcr_greedy")),
            "er_greedy": _norm(row.get("er_greedy")),
        }
    return entries, preds


def build_train_binders() -> pd.DataFrame:
    """Conditioning/train binder pool from TCRT5 topk train (epitope, cdr3b)."""
    src = (TCRT5 / "data" / "pmhc2tcr_topk_train_source_dedup_uf.txt").read_text().splitlines()
    trg = (TCRT5 / "data" / "pmhc2tcr_topk_train_target_dedup_uf.txt").read_text().splitlines()
    rows = []
    for s, t in zip(src, trg):
        ep = _norm(s.split()[0]) if s.strip() else ""
        cdr3b = _norm(t)
        if ep and cdr3b:
            rows.append((ep, cdr3b))
    df = pd.DataFrame(rows, columns=["epitope", "cdr3b"]).drop_duplicates()
    return df


def load_bioseq_tcr_corpus() -> tuple[set[str], set[tuple[str, str]], set[str]]:
    """Extract (epitope set, (epitope,cdr3b) pairs, cdr3b set) from BioSeq tcr source.

    BioSeq trains on ``--sources oas,ots,tcr,ppi``; the leakage-relevant TCR
    conditioning signal lives in the ``tcr`` grammar source (roles antigen /
    tcr_beta / tcr_alpha). oas/ots are antibody repertoires (no TCR:epitope
    pairs) so they cannot leak epitope-conditioned binders.
    """
    from dllm.pipelines.immune_llada.data.grammar import GrammarArrowSource, GrammarArrowSourceConfig

    epitopes: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    cdr3bs: set[str] = set()
    for split in ("train", "valid"):
        src = GrammarArrowSource(GrammarArrowSourceConfig(name="tcr", split=split))
        for rec in src.iter_records():
            ep = ""
            beta = ""
            for c in rec.chains:
                role = c.role.lower()
                if role in {"antigen", "peptide", "epitope"}:
                    ep = _norm(c.sequence)
                elif role == "tcr_beta":
                    beta = _norm(c.sequence)
            if ep:
                epitopes.add(ep)
            if beta:
                cdr3bs.add(beta)
            if ep and beta:
                pairs.add((ep, beta))
    return epitopes, pairs, cdr3bs


def build_leakage_report(
    eval_entries: dict[str, dict],
    train_df: pd.DataFrame,
    corpus: tuple[set[str], set[tuple[str, str]], set[str]],
) -> dict:
    tr_eps, tr_pairs, tr_cdr3b = corpus
    print(f"  BioSeq tcr corpus: {len(tr_eps)} epitopes, {len(tr_pairs)} (ep,cdr3b) pairs, {len(tr_cdr3b)} unique cdr3b")

    # conditioning (TCRT5 train) pools for ref-vs-train overlap
    cond_pairs = set(zip(train_df["epitope"], train_df["cdr3b"]))
    cond_by_ep = defaultdict(set)
    for ep, cb in cond_pairs:
        cond_by_ep[ep].add(cb)

    per_pmhc = {}
    tot = {"n_ref": 0, "ref_in_bioseq_pair": 0, "ref_cdr3b_in_bioseq": 0, "ref_in_cond_pair": 0}
    eval_eps = set()
    for pmhc, e in eval_entries.items():
        ep = e["epitope"]
        eval_eps.add(ep)
        refs = e["ref_binders"]
        ref_pair_leak = sum(1 for r in refs if (ep, r) in tr_pairs)
        ref_cdr3b_leak = sum(1 for r in refs if r in tr_cdr3b)
        ref_cond_leak = sum(1 for r in refs if r in cond_by_ep.get(ep, set()))
        per_pmhc[pmhc] = {
            "epitope": ep,
            "category": e["category"],
            "n_ref": len(refs),
            "epitope_in_bioseq_train": ep in tr_eps,
            "ref_binders_in_bioseq_pair": ref_pair_leak,
            "ref_binders_in_bioseq_pair_frac": round(ref_pair_leak / len(refs), 4) if refs else 0.0,
            "ref_cdr3b_in_bioseq_anywhere": ref_cdr3b_leak,
            "ref_cdr3b_in_bioseq_anywhere_frac": round(ref_cdr3b_leak / len(refs), 4) if refs else 0.0,
            "ref_binders_in_conditioning_pool": ref_cond_leak,
        }
        tot["n_ref"] += len(refs)
        tot["ref_in_bioseq_pair"] += ref_pair_leak
        tot["ref_cdr3b_in_bioseq"] += ref_cdr3b_leak
        tot["ref_in_cond_pair"] += ref_cond_leak

    eval_ep_in_bioseq = sorted(eval_eps & tr_eps)
    return {
        "description": "Leakage of TCRT5 eval epitopes/binders into (a) BioSeq training corpus "
                       "(sources oas,ots,tcr,ppi; only tcr carries epitope-conditioned binders) "
                       "and (b) the TCRT5 conditioning/train pool.",
        "bioseq_corpus": {
            "n_epitopes": len(tr_eps),
            "n_pairs": len(tr_pairs),
            "n_unique_cdr3b": len(tr_cdr3b),
        },
        "eval_summary": {
            "n_eval_pmhc": len(eval_entries),
            "n_eval_epitopes": len(eval_eps),
            "n_eval_epitopes_in_bioseq_train": len(eval_ep_in_bioseq),
            "eval_epitopes_in_bioseq_train": eval_ep_in_bioseq,
            "total_ref_binders": tot["n_ref"],
            "total_ref_binders_leaked_pair": tot["ref_in_bioseq_pair"],
            "total_ref_binders_leaked_pair_frac": round(tot["ref_in_bioseq_pair"] / tot["n_ref"], 4) if tot["n_ref"] else 0.0,
            "total_ref_cdr3b_in_bioseq_anywhere": tot["ref_cdr3b_in_bioseq"],
            "total_ref_cdr3b_in_bioseq_anywhere_frac": round(tot["ref_cdr3b_in_bioseq"] / tot["n_ref"], 4) if tot["n_ref"] else 0.0,
            "total_ref_binders_in_conditioning_pool": tot["ref_in_cond_pair"],
        },
        "per_pmhc": per_pmhc,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pseudo = load_hla_pseudo()

    print("building held20 eval (topk_val_df.csv) ...")
    held20 = build_held20(pseudo)
    print(f"  held20: {len(held20)} pMHC")

    print("building benchmark14 eval (benchmark_data_w_preds.csv) ...")
    bench14, preds = build_benchmark14(pseudo)
    print(f"  benchmark14: {len(bench14)} pMHC")

    eval_entries = {**held20, **bench14}
    (OUT / "eval_conditional.json").write_text(json.dumps(eval_entries, indent=1))
    (OUT / "benchmark_preds.json").write_text(json.dumps(preds, indent=1))

    print("building conditioning/train binder pool (TCRT5 topk train) ...")
    train_df = build_train_binders()
    train_df.to_csv(OUT / "train_binders.csv", index=False)
    print(f"  train_binders: {len(train_df)} (epitope,cdr3b) rows, {train_df['epitope'].nunique()} epitopes")

    print("extracting BioSeq tcr training corpus (loads the arrow shard) ...")
    corpus = load_bioseq_tcr_corpus()
    print("building leakage report vs BioSeq training corpus ...")
    leak = build_leakage_report(eval_entries, train_df, corpus)
    (OUT / "leakage_report.json").write_text(json.dumps(leak, indent=1))

    # Persist which eval pMHC are BioSeq-unseen (truly zero-shot for our model)
    bioseq_unseen = [p for p, e in eval_entries.items()
                     if not leak["per_pmhc"][p]["epitope_in_bioseq_train"]]
    (OUT / "bioseq_unseen_pmhc.json").write_text(json.dumps(sorted(bioseq_unseen), indent=1))

    manifest = {
        "benchmark_version": "irbench.tcr_generation.v2",
        "anchor": "TCRT5 (Nat Mach Intell 2025)",
        "eval_sets": {
            "held20": {"n_pmhc": len(held20), "source": "topk_val_df.csv",
                       "note": "target-rich held-out top-20 validation split"},
            "benchmark14": {"n_pmhc": len(bench14), "source": "benchmark_data_w_preds.csv",
                            "note": "sparse benchmark epitopes with official stored preds"},
        },
        "total_eval_pmhc": len(eval_entries),
        "conditioning_pool": {"source": "pmhc2tcr_topk_train_*_dedup_uf.txt",
                              "n_pairs": int(len(train_df))},
        "unconditional_setting_A": {"train": "data/tcr_generation/train_cdr3b.txt",
                                    "holdout": "data/tcr_generation/holdout_cdr3b.txt",
                                    "note": "reused verbatim from prior IRBench build"},
        "default_K": 100,
        "default_seed": 42,
        "leakage_report": "leakage_report.json",
        "bioseq_unseen_pmhc": "bioseq_unseen_pmhc.json",
        "n_bioseq_unseen_pmhc": int(len(bioseq_unseen)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print("\n=== leakage summary ===")
    s = leak["eval_summary"]
    print(f"  eval epitopes in BioSeq train: {s['n_eval_epitopes_in_bioseq_train']}/{s['n_eval_epitopes']}")
    print(f"  ref binders leaked as (ep,cdr3b) pair into BioSeq: "
          f"{s['total_ref_binders_leaked_pair']}/{s['total_ref_binders']} "
          f"({s['total_ref_binders_leaked_pair_frac']*100:.2f}%)")
    print(f"  ref cdr3b present anywhere in BioSeq: "
          f"{s['total_ref_cdr3b_in_bioseq_anywhere']}/{s['total_ref_binders']} "
          f"({s['total_ref_cdr3b_in_bioseq_anywhere_frac']*100:.2f}%)")
    print(f"  ref binders in TCRT5 conditioning pool: {s['total_ref_binders_in_conditioning_pool']}")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
