#!/usr/bin/env python
"""Score BioSeq *unconditional* full-length alpha/beta pairs (Setting C).

BioSeq can only assemble full-length TCRs unconditionally (its ``ots`` source
has no epitope), so Setting C for BioSeq is an honest capability view:

  1. full-length validity   -> frac valid AA (alpha / beta), frac CDR3b extractable
  2. extract CDR3b (ANARCI, IMGT 104-118; regex fallback)
  3. treat extracted CDR3b as an unconditional repertoire -> Setting A metrics
     (k-mer JSD vs OTS holdout, novelty, mean-NN edit dist, OLGA pgen calibration)

Reads   outputs/tcr_generation_bench/setting_C/<tag>/full.jsonl
Writes  outputs/tcr_generation_bench/setting_C/<tag>/metrics.json

Epitope-conditioned full-length remains TcrDesign's job (--method tcrdesign,
scored with the conditional Setting C path in tcr_generation_bench/run.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from tcr_generation_bench import scoring  # noqa: E402

OUT = BENCH / "outputs" / "tcr_generation_bench"


def _validity(seqs: list[str]) -> dict:
    seqs = [s for s in seqs if s]
    valid = [s for s in seqs if scoring.is_valid_aa(s)]
    return {"n": len(seqs), "n_valid_aa": len(valid),
            "frac_valid_aa": float(len(valid) / len(seqs)) if seqs else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="ours_bioseq")
    ap.add_argument("--samples-file", default=None,
                    help="full.jsonl (default: setting_C/<tag>/full.jsonl)")
    ap.add_argument("--no-anarci", action="store_true")
    ap.add_argument("--no-pgen", action="store_true")
    ap.add_argument("--pgen-cap", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = OUT / "setting_C" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = Path(args.samples_file) if args.samples_file else out_dir / "full.jsonl"

    rows = [json.loads(l) for l in samples.read_text().splitlines() if l.strip()]
    alphas = [str(r.get("tcr_alpha", "")).upper() for r in rows]
    betas = [str(r.get("tcr_beta", "")).upper() for r in rows]
    print(f"[load] {len(rows)} pairs from {samples}")

    beta_valid = [b for b in betas if b and scoring.is_valid_aa(b)]
    if args.no_anarci:
        cdr3s = [scoring.extract_cdr3b_regex(b) for b in beta_valid]
    else:
        cdr3s = scoring.extract_cdr3b_anarci(beta_valid)
    cdr3s = [c for c in cdr3s if c]
    print(f"[extract] {len(cdr3s)}/{len(beta_valid)} CDR3b extracted from valid beta chains")

    validity = {
        "n_generated": len(rows),
        "alpha": _validity(alphas),
        "beta": _validity(betas),
        "frac_valid_aa": _validity(betas)["frac_valid_aa"],  # beta primary
        "n_cdr3_extracted": len(cdr3s),
        "frac_cdr3_extracted": float(len(cdr3s) / len(beta_valid)) if beta_valid else float("nan"),
    }

    dist = None
    if cdr3s:
        dist = scoring.score_unconditional(cdr3s, with_pgen=not args.no_pgen,
                                           pgen_cap=args.pgen_cap, seed=args.seed)

    result = {
        "setting": "C",
        "method": "file",
        "conditioning": "unconditional",
        "model": "BioSeq unconditional full-length alpha/beta",
        "tag": args.tag,
        "n": len(rows),
        "fulllength_validity": validity,
        "cdr3b_distribution": dist,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    print("\n=== BioSeq unconditional full-length (Setting C) ===")
    print(f"  n={validity['n_generated']} "
          f"valid_aa(beta)={validity['frac_valid_aa']:.3f} "
          f"valid_aa(alpha)={validity['alpha']['frac_valid_aa']:.3f} "
          f"cdr3_extracted={validity['frac_cdr3_extracted']:.3f}")
    if dist:
        print(f"  extracted CDR3b: kmer_JSD={dist.get('kmer_jsd'):.4f} "
              f"novelty={dist.get('novelty'):.3f} NN_dist={dist.get('mean_nn_distance'):.2f}")
        if dist.get("pgen_generated"):
            pg = dist["pgen_generated"]
            print(f"  pgen: frac_recombinable={pg['frac_pgen_positive']:.3f} "
                  f"med_log10={pg['median_log10_pgen']:.2f}")
    print(f"\n-> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
