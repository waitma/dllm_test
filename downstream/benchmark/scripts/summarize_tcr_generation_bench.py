#!/usr/bin/env python
"""Consolidate all tcr_generation_bench metrics into one leaderboard JSON + tables.

Reads ``outputs/tcr_generation_bench/setting_{A,B,C}/<tag>/metrics.json`` and
prints markdown-ready tables for RESULTS.md / downstream.md, plus writes
``outputs/tcr_generation_bench/_summary.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
OUT = BENCH / "outputs" / "tcr_generation_bench"


def _g(d, k):
    v = d.get(k)
    if v is None or (isinstance(v, float) and v != v):
        return None
    return v


def _fmt(v, nd=4):
    return "nan" if v is None else f"{v:.{nd}f}"


def load_all():
    rows = {"A": [], "B": [], "C": []}
    for setting in ("A", "B", "C"):
        sdir = OUT / f"setting_{setting}"
        if not sdir.exists():
            continue
        for mj in sorted(sdir.glob("*/metrics.json")):
            data = json.loads(mj.read_text())
            rows[setting].append((mj.parent.name, data))
    return rows


def summarize_B_row(tag, data):
    """Return dict of the key views for a Setting B/C entry."""
    summ = data.get("summary", {})
    out = {
        "tag": tag, "model": data.get("model", tag), "k": data.get("k"),
        "eval_set": data.get("eval_set"),
        "baseline_provenance": data.get("baseline_provenance"),
    }
    for view in ("overall", "held20", "paper_sparse13", "benchmark14",
                 "bioseq_seen", "bioseq_unseen", "bioseq_unseen_common"):
        if view in summ:
            s = summ[view]
            out[view] = {m: _g(s, m) for m in
                         ("precision", "recall", "f1", "d_edit", "seq_recovery",
                          "char_bleu", "diversity", "novelty_vs_conditioning",
                          "n_pmhc", "n_pmhc_scored")}
    if "fulllength_validity" in data:
        v = data["fulllength_validity"]
        out["validity"] = {"frac_valid_aa": _g(v, "frac_valid_aa"),
                           "frac_cdr3_extracted": _g(v, "frac_cdr3_extracted"),
                           "n_generated": v.get("n_generated")}
    if "cdr3b_distribution" in data and data["cdr3b_distribution"]:
        out["cdr3b_distribution"] = data["cdr3b_distribution"]
    if "conditioning" in data:
        out["conditioning"] = data["conditioning"]
    return out


def main():
    rows = load_all()
    summary = {"setting_A": [], "setting_B": [], "setting_C": []}

    # Setting A
    print("\n" + "=" * 90)
    print("SETTING A — unconditional CDR3b repertoire")
    print("=" * 90)
    print(f"{'model':32s} {'n':>6s} {'uniq':>6s} {'novelty':>8s} {'NNdist':>7s} {'JSD':>7s} {'pgen+':>7s} {'medlogP':>8s}")
    for tag, data in rows["A"]:
        m = data.get("metrics", {})
        pg = m.get("pgen_generated", {})
        print(f"{data.get('model', tag)[:32]:32s} {m.get('n_generated','?'):>6} "
              f"{m.get('n_unique','?'):>6} {_fmt(_g(m,'novelty'),3):>8} {_fmt(_g(m,'mean_nn_distance'),2):>7} "
              f"{_fmt(_g(m,'kmer_jsd'),4):>7} {_fmt(_g(pg,'frac_pgen_positive'),3):>7} {_fmt(_g(pg,'median_log10_pgen'),2):>8}")
        summary["setting_A"].append({"tag": tag, "model": data.get("model", tag), "metrics": m})

    # Setting B / C
    for setting, label in (("B", "epitope-conditioned CDR3b design"),
                           ("C", "full-length alpha/beta -> CDR3b")):
        print("\n" + "=" * 90)
        print(f"SETTING {setting} — {label}")
        print("=" * 90)
        for view in ("paper_sparse13", "benchmark14", "bioseq_unseen_common",
                     "bioseq_unseen", "held20", "overall"):
            printed_header = False
            for tag, data in rows[setting]:
                srow = summarize_B_row(tag, data)
                if view not in srow:
                    continue
                if not printed_header:
                    print(f"\n-- view: {view} --")
                    print(f"{'model':30s} {'n_ep':>8s} {'prec':>7s} {'recall':>7s} {'f1':>7s} "
                          f"{'d_edit':>7s} {'seqrec':>7s} {'bleu':>7s} {'div':>6s}")
                    print(f"{'':30s} {'scr/tot':>8s}")
                    printed_header = True
                s = srow[view]
                n_ep = f"{s.get('n_pmhc_scored', '?')}/{s.get('n_pmhc', '?')}"
                print(f"{srow['model'][:30]:30s} {n_ep:>8} "
                      f"{_fmt(s['precision']):>7} {_fmt(s['recall']):>7} {_fmt(s['f1']):>7} "
                      f"{_fmt(s['d_edit'],3):>7} {_fmt(s['seq_recovery']):>7} {_fmt(s['char_bleu']):>7} "
                      f"{_fmt(s['diversity'],3):>6}")
        # Setting C: also print the unconditional full-length validity + extracted-CDR3b distribution
        if setting == "C":
            vrows = [(tag, summarize_B_row(tag, data)) for tag, data in rows[setting]]
            vrows = [(t, s) for t, s in vrows if "validity" in s or "cdr3b_distribution" in s]
            if vrows:
                print("\n-- unconditional full-length validity + extracted-CDR3b distribution --")
                print(f"{'model':30s} {'cond':>12s} {'n':>5s} {'validAA':>8s} {'cdr3ext':>8s} "
                      f"{'JSD':>7s} {'novel':>6s} {'NNd':>6s} {'pgen+':>6s}")
                for tag, s in vrows:
                    v = s.get("validity", {})
                    d = s.get("cdr3b_distribution", {}) or {}
                    pg = d.get("pgen_generated", {}) or {}
                    print(f"{s['model'][:30]:30s} {str(s.get('conditioning','?'))[:12]:>12s} "
                          f"{str(v.get('n_generated','?')):>5} {_fmt(_g(v,'frac_valid_aa'),3):>8} "
                          f"{_fmt(_g(v,'frac_cdr3_extracted'),3):>8} {_fmt(_g(d,'kmer_jsd'),4):>7} "
                          f"{_fmt(_g(d,'novelty'),3):>6} {_fmt(_g(d,'mean_nn_distance'),2):>6} "
                          f"{_fmt(_g(pg,'frac_pgen_positive'),3):>6}")

        for tag, data in rows[setting]:
            summary[f"setting_{setting}"].append(summarize_B_row(tag, data))

    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {OUT}/_summary.json")


if __name__ == "__main__":
    main()
