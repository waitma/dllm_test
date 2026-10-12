#!/usr/bin/env python
"""Unified TCR generation benchmark harness (TCRT5-anchored, 3 settings).

Settings
--------
  A  unconditional CDR3b repertoire distribution match (+ OLGA pgen)
  B  epitope-conditioned CDR3b design (main leaderboard, TCRT5 protocol)
  C  full-length alpha/beta assembly -> validity + CDR3b -> Setting B metrics

Methods
-------
  tcrt5     TCRT5 (HF) beam-search (B, cond) / sampling (A, uncond)
  olga      OLGA human_T_beta unconditional sampler (A; B lower bound)
  sonnia    soNNia/SONIA post-selected sampler (A; B lower bound; official API)
  tcrdesign read a samples file produced by the TcrDesign wrapper (its own env)
  file      read a samples file (BioSeq adapters / any external generator)

Samples file formats
--------------------
  conditional (B/C): JSONL, one obj per epitope/pMHC:
      {"pmhc": "GILGFVFTL_HLA-A*02:01", "epitope": "GILGFVFTL", "sequences": [...]}
      (pMHC key preferred; "epitope" is matched against the eval set otherwise)
  unconditional (A): plain .txt (one CDR3b/line) or JSONL with a "sequences" list.

Examples
--------
  # Setting B, TCRT5 official baseline, K=100, benchmark14 subset
  python tcr_generation_bench/run.py --setting B --method tcrt5 --eval-set benchmark14 -k 100
  # Setting A, OLGA
  python tcr_generation_bench/run.py --setting A --method olga --n 5000
  # Setting B, our BioSeq designs
  python tcr_generation_bench/run.py --setting B --method file --samples-file bioseq.jsonl --tag ours_bioseq
  # Setting C, full-length designs
  python tcr_generation_bench/run.py --setting C --method file --samples-file bioseq_full.jsonl --tag ours_bioseq_full
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


# ---------------------------------------------------------------------------
# samples-file readers
# ---------------------------------------------------------------------------

def read_conditional_samples(path: str, eval_entries: dict[str, dict]) -> tuple[dict, dict]:
    """Return ({pMHC: [seqs]}, {pMHC: greedy}) matched to the eval set."""
    ep_to_pmhc: dict[str, list[str]] = {}
    for pmhc, e in eval_entries.items():
        ep_to_pmhc.setdefault(e["epitope"].upper(), []).append(pmhc)

    designs: dict[str, list[str]] = {}
    greedy: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        seqs = [str(s).upper() for s in obj.get("sequences", [])]
        if not seqs and obj.get("full_rows"):
            seqs = [
                str(r.get("betaCDR3", "")).strip().upper()
                for r in obj["full_rows"]
                if r.get("betaCDR3")
            ]
        targets = []
        if obj.get("pmhc") and obj["pmhc"] in eval_entries:
            targets = [obj["pmhc"]]
        elif obj.get("epitope"):
            targets = ep_to_pmhc.get(str(obj["epitope"]).upper(), [])
        for t in targets:
            designs[t] = seqs
            if obj.get("greedy"):
                greedy[t] = str(obj["greedy"]).upper()
    return designs, greedy


def read_unconditional_samples(path: str) -> list[str]:
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".jsonl" or text.lstrip().startswith("{"):
        seqs: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seqs.extend(str(s).upper() for s in obj.get("sequences", []))
        return seqs
    return [s.strip().upper() for s in text.splitlines() if s.strip()]


def subset_eval(eval_entries: dict, which: str) -> dict:
    if which == "all":
        return eval_entries
    return {p: e for p, e in eval_entries.items() if e["category"] == which}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=["A", "B", "C"], required=True)
    ap.add_argument("--method", choices=["tcrt5", "olga", "sonnia", "tcrdesign", "file"], required=True)
    ap.add_argument("--eval-set", choices=["held20", "benchmark14", "all"], default="all",
                    help="conditional eval subset (Setting B/C)")
    ap.add_argument("--samples-file", default=None, help="for --method file/tcrdesign")
    ap.add_argument("-k", "--k", type=int, default=100, help="designs per epitope (Setting B/C)")
    ap.add_argument("--n", type=int, default=5000, help="repertoire size (Setting A)")
    ap.add_argument("--num-beams-mult", type=int, default=3, help="TCRT5 num_beams = mult*k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pgen", action="store_true", help="skip OLGA pgen in Setting A")
    ap.add_argument("--no-anarci", action="store_true", help="regex-only CDR3 extraction (Setting C)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or args.method
    out_dir = OUT / f"setting_{args.setting}" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.setting == "A":
        result = run_setting_a(args, out_dir)
    else:
        result = run_conditional(args, out_dir)

    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"\n-> {out_dir}/metrics.json")


def run_setting_a(args, out_dir: Path) -> dict:
    if args.method == "tcrt5":
        from tcr_generation_bench import generators
        gen = generators.tcrt5_unconditional(n=args.n, device=args.device, seed=args.seed)
        model_name = "TCRT5 (uncond sampling)"
    elif args.method == "olga":
        from tcr_generation_bench import generators
        gen = generators.olga_unconditional(n=args.n, seed=args.seed)
        model_name = "OLGA human_T_beta"
    elif args.method == "sonnia":
        from tcr_generation_bench import generators
        gen = generators.sonia_unconditional(n=args.n, seed=args.seed)
        model_name = "soNNia/SONIA (post-selected)"
    elif args.method in ("file", "tcrdesign"):
        assert args.samples_file, "--samples-file required"
        gen = read_unconditional_samples(args.samples_file)
        model_name = f"{args.method}:{Path(args.samples_file).name}"
    else:
        raise ValueError(args.method)

    print(f"[Setting A] {model_name}: {len(gen)} generated")
    (out_dir / "samples.txt").write_text("\n".join(gen) + "\n")
    m = scoring.score_unconditional(gen, with_pgen=not args.no_pgen, seed=args.seed)
    result = {"setting": "A", "method": args.method, "model": model_name,
              "n_requested": args.n, "metrics": m}
    _print_setting_a(model_name, m)
    return result


def run_conditional(args, out_dir: Path) -> dict:
    eval_all = scoring.load_eval_conditional()
    eval_entries = subset_eval(eval_all, args.eval_set)
    print(f"[Setting {args.setting}] eval_set={args.eval_set}: {len(eval_entries)} pMHC")

    greedy_map: dict[str, str] = {}
    full_length = args.setting == "C"

    if args.method == "tcrt5":
        from tcr_generation_bench import generators
        designs, greedy_map = generators.tcrt5_conditional(
            eval_entries, k=args.k, num_beams_mult=args.num_beams_mult, device=args.device)
        model_name = "TCRT5 (beam)"
    elif args.method in ("olga", "sonnia"):
        from tcr_generation_bench import generators
        pool = (generators.olga_unconditional(n=args.k, seed=args.seed)
                if args.method == "olga"
                else generators.sonia_unconditional(n=args.k, seed=args.seed))
        designs = generators.replicate_pool_per_pmhc(pool, eval_entries)
        model_name = f"{args.method} (uncond lower bound)"
    elif args.method in ("file", "tcrdesign"):
        assert args.samples_file, "--samples-file required"
        designs, greedy_map = read_conditional_samples(args.samples_file, eval_entries)
        model_name = f"{args.method}:{Path(args.samples_file).name}"
    else:
        raise ValueError(args.method)

    n_seqs = sum(len(v) for v in designs.values())
    covered = sum(1 for p in eval_entries if designs.get(p))
    print(f"  {model_name}: {n_seqs} designs across {covered}/{len(eval_entries)} pMHC")

    # persist raw designs
    with (out_dir / "designs.jsonl").open("w") as fh:
        for pmhc, seqs in designs.items():
            obj = {"pmhc": pmhc, "epitope": eval_entries[pmhc]["epitope"], "sequences": seqs}
            if pmhc in greedy_map:
                obj["greedy"] = greedy_map[pmhc]
            fh.write(json.dumps(obj) + "\n")

    if full_length and args.method == "tcrdesign":
        # TcrDesign tcrdesign.py outputs designed betaCDR3 (V/J/CDR3 rows), not
        # full-length beta chains — score via Setting B + pipeline validity.
        scored = scoring.score_pipeline_cdr3beta(designs, eval_entries, k=args.k)
    elif full_length:
        scored = scoring.score_fulllength(designs, eval_entries, k=args.k, use_anarci=not args.no_anarci)
    else:
        scored = scoring.score_conditional(designs, eval_entries, k=args.k, greedy_map=greedy_map)

    result = {"setting": args.setting, "method": args.method, "model": model_name,
              "eval_set": args.eval_set, "k": args.k, **scored}
    _print_conditional(model_name, scored, full_length)
    return result


def _fmt(s: dict) -> str:
    def g(k):
        v = s.get(k)
        return "nan" if v is None or (isinstance(v, float) and v != v) else f"{v:.4f}"
    return (f"prec={g('precision')} recall={g('recall')} f1={g('f1')} "
            f"d_edit={g('d_edit')} seqrec={g('seq_recovery')} bleu={g('char_bleu')} "
            f"div={g('diversity')} nov={g('novelty_vs_conditioning')} "
            f"(n={s.get('n_pmhc_scored')}/{s.get('n_pmhc')} scored/total)")


def _print_conditional(model_name: str, scored: dict, full_length: bool):
    print(f"\n=== {model_name} ===")
    if full_length and "fulllength_validity" in scored:
        v = scored["fulllength_validity"]
        print(f"  full-length: valid_aa={v['frac_valid_aa']:.3f} "
              f"cdr3_extracted={v['frac_cdr3_extracted']:.3f} (n={v['n_generated']})")
    for cat in ("overall", "held20", "benchmark14", "bioseq_seen", "bioseq_unseen"):
        if cat in scored["summary"]:
            print(f"  {cat:14s}: {_fmt(scored['summary'][cat])}")


def _print_setting_a(model_name: str, m: dict):
    print(f"\n=== {model_name} ===")
    print(f"  n={m['n_generated']} unique={m['n_unique']} novelty={m.get('novelty'):.3f} "
          f"NN_dist={m.get('mean_nn_distance'):.2f} kmer_JSD={m.get('kmer_jsd'):.4f}")
    if "pgen_generated" in m:
        pg = m["pgen_generated"]
        print(f"  pgen(gen): frac_recombinable={pg['frac_pgen_positive']:.3f} "
              f"med_log10_pgen={pg['median_log10_pgen']:.2f}")


if __name__ == "__main__":
    main()
