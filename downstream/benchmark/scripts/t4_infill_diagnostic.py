#!/usr/bin/env python
"""T4 diagnostic -- does the model model TCR sequence, or only fail to use the epitope?

Why this exists
---------------
Setting B cannot separate two very different failure modes. Our best model beats
OLGA -- a pure unconditional VDJ-recombination sampler that never sees the
epitope -- by only 0.046 edit distance, which is consistent with *either*:

  (H1) the model cannot model CDR3b sequence at all, or
  (H2) the model models CDR3b fine but ignores the epitope.

RESULTS.md section 0.0(h) argues for H2 from the training mix: only 3.76% of the
5.65M training rows carry an epitope block, so a strong unconditional prior is
the loss-optimal solution. This script tests that argument directly by removing
the epitope from the task entirely: re-fill a centred window of a *real*
held-out CDR3b from its own flanking residues, with a bare ``tcr_single``
record. If H2 holds, infilling should be clearly above the position-only
baselines; if H1 holds, it should not.

The baselines are the whole point
---------------------------------
Raw AAR on CDR3b is meaningless without them, because CDR3b is a hypervariable
junction: its centre is largely random N-insertion, and its flanks are strongly
stereotyped (``CASS`` at the N-side, ``...F`` at the C-side). Three
context-blind references, all scored on the *same* rows and the *same* masked
positions:

* ``pwm``      -- per (length, position) argmax over the 1.72M-sequence train
                  reference. Knows sequence length and where in the junction it
                  is, but nothing about the actual flanks. **This is the
                  reference that matters**: beating it requires using context.
* ``marginal`` -- global most-frequent residue, one letter everywhere.
* ``random``   -- 1/20 uniform, reported analytically.

A model that merely memorised CDR3b positional statistics lands on ``pwm``. The
interpretation of "we beat OLGA by 0.046" is that we learned something close to
an unconditional prior; the question here is whether that prior is *better* than
positional statistics, i.e. whether it uses context at all.

Usage
-----
    # 1) generate (needs GPU; one run per mask width)
    python -m downstream.grammar.tcr_generation --mode infill --mask-width 3 \
        --checkpoint <ckpt> --out <dir>/infill_w3.jsonl

    # 2) score against the context-blind baselines
    python scripts/t4_infill_diagnostic.py --pred <dir>/infill_w3.jsonl

Output:
    outputs/tcr_generation_bench/infill_diagnostic/<tag>.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
FULLREF = BENCH / "data" / "tcr_generation_fullref"
OUT = BENCH / "outputs" / "tcr_generation_bench" / "infill_diagnostic"


def _anchor(core: str) -> str:
    """Re-attach the C/F anchors the fullref files strip, matching the sampler."""
    s = ("C" if not core.startswith("C") else "") + core
    return s + ("F" if not s.endswith(("F", "W")) else "")


def mcnemar(b: int, c: int) -> dict:
    """Exact paired test on the disagreement counts (``b`` = ours only correct).

    Under the null "ours and the PWM are equally likely to win a position they
    disagree on", ``b ~ Binomial(b + c, 0.5)``. Reported instead of a
    two-proportion z-test because the two methods are scored on identical
    positions, which the unpaired test cannot exploit -- with n in the hundreds
    the unpaired interval is wide enough to call a real gap inconclusive.
    """
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return {"b_ours_only": b, "c_pwm_only": c, "p_two_sided": None,
                "verdict": "no disagreements"}
    p = float(binomtest(b, n, 0.5).pvalue)
    if p >= 0.05:
        verdict = "tie (not significant at 0.05)"
    else:
        verdict = "ours better" if b > c else "PWM better"
    return {"b_ours_only": b, "c_pwm_only": c, "n_discordant": n,
            "p_two_sided": p, "verdict": verdict}


def build_pwm(train_path: Path) -> tuple[dict, str]:
    """(length, position) -> most frequent residue, plus the global mode residue.

    Conditioning on length matters: a position index means something different in
    a 10-mer than in a 20-mer, so an unconditional positional PWM would be a
    weaker straw man than the model deserves.
    """
    by_len_pos: dict[tuple[int, int], Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    for line in train_path.read_text().splitlines():
        core = line.strip()
        if not core:
            continue
        seq = _anchor(core)
        for i, aa in enumerate(seq):
            by_len_pos[(len(seq), i)][aa] += 1
            overall[aa] += 1
    pwm = {k: c.most_common(1)[0][0] for k, c in by_len_pos.items()}
    return pwm, overall.most_common(1)[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="infill_*.jsonl from --mode infill")
    ap.add_argument("--train-ref", default=str(FULLREF / "train_cdr3b.txt"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--keep-train-seen", action="store_true",
                    help="do NOT drop rows whose CDR3b also occurs in training. Off by "
                         "default: the tcr_single holdout is disjoint from train, but "
                         "the OTS holdout used by the tcr_pair layout shares 10.5%% of "
                         "its beta chains with OTS train, and leaving those in would "
                         "flatter the pair layout in the very comparison it exists for.")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    rows = [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {pred_path}")

    print(f"building PWM from {args.train_ref} ...", flush=True)
    pwm, mode_aa = build_pwm(Path(args.train_ref))
    print(f"  PWM cells {len(pwm):,}, global mode residue {mode_aa!r}")

    train_cores = {l.strip() for l in
                   Path(args.train_ref).read_text().splitlines() if l.strip()}

    def _core(r: dict) -> str:
        if "cdr3b_core" in r:
            return r["cdr3b_core"]
        # tcr_single truths are ``C + core + F``; strip the anchors back off.
        t = r["truth"]
        return t[1:-1] if t.startswith("C") and t.endswith(("F", "W")) else t

    n_before = len(rows)
    seen_rows = [r for r in rows if _core(r) in train_cores]
    unseen_rows = [r for r in rows if _core(r) not in train_cores]
    n_dropped = len(seen_rows)
    if not args.keep_train_seen:
        rows = unseen_rows
        print(f"  dropped {n_dropped}/{n_before} rows whose CDR3b occurs in train "
              f"({n_dropped / n_before * 100:.2f}%)")
        if not rows:
            raise SystemExit("every row was train-seen; nothing left to score")

    def score(subset: list[dict]) -> dict:
        ok = [r for r in subset if r["length_ok"] and r["truth_window"]]
        if not ok:
            return {}
        n_pos = sum(len(r["truth_window"]) for r in ok)
        hits = {"ours": 0, "pwm": 0, "marginal": 0}
        pwm_miss = 0
        # McNemar cells: positions where exactly one of (ours, pwm) is correct.
        # ours and pwm are scored on the *same* positions, so an unpaired
        # two-proportion test would overstate the variance; the paired
        # disagreement counts are what carry the evidence.
        b = c = 0
        for r in ok:
            lo, _ = r["span"]
            if "cdr3b_core" in r:
                # tcr_pair rows carry full-length beta coordinates, but the PWM is
                # keyed on the anchored junction. Map back: the core starts at
                # ``offset`` in the chain and at index 1 in ``C + core + F``.
                core = r["cdr3b_core"]
                offset = r["truth"].find(core)
                L = len(core) + 2
                lo = lo - offset + 1
            else:
                L = len(r["truth"])
            for j, true_aa in enumerate(r["truth_window"]):
                ours_ok = r["pred_window"][j] == true_aa
                if ours_ok:
                    hits["ours"] += 1
                cell = pwm.get((L, lo + j))
                if cell is None:
                    pwm_miss += 1
                pwm_ok = cell is not None and cell == true_aa
                if pwm_ok:
                    hits["pwm"] += 1
                if ours_ok and not pwm_ok:
                    b += 1
                elif pwm_ok and not ours_ok:
                    c += 1
                if mode_aa == true_aa:
                    hits["marginal"] += 1
        aar = {k: round(v / max(1, n_pos) * 100, 2) for k, v in hits.items()}
        aar["random"] = round(100 / 20, 2)
        return {
            "n_rows": len(subset),
            "n_length_preserved": len(ok),
            "length_preserved_frac": round(len(ok) / len(subset), 4),
            "n_masked_positions": n_pos,
            "pwm_cells_missing": pwm_miss,
            "aar": aar,
            "ours_minus_pwm": round(aar["ours"] - aar["pwm"], 2),
            "mcnemar_vs_pwm": mcnemar(b, c),
        }

    main_res = score(rows)
    res = {
        "pred_file": str(pred_path),
        "layout": "tcr_pair" if "cdr3b_core" in rows[0] else "tcr_single",
        "mask_width": rows[0]["span"][1] - rows[0]["span"][0],
        "n_rows_before_split": n_before,
        "n_train_seen": n_dropped,
        "train_seen_frac": round(n_dropped / n_before, 4),
        **main_res,
        # Upper-bound probe, not a performance claim: rows whose CDR3b the model was
        # actually trained on. If those also lose to the PWM, the deficit is a fitting
        # failure rather than a generalisation failure -- the two need different fixes.
        "train_seen_probe": score(seen_rows) if seen_rows else
                            {"note": "no train-seen rows in this file"},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or pred_path.stem
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=2))

    print(f"\n=== {tag} [{res['layout']}] (mask_width={res['mask_width']}, "
          f"{main_res['n_masked_positions']:,} masked positions) ===")
    print(f"  length preserved   {main_res['n_length_preserved']}/{main_res['n_rows']} "
          f"({main_res['length_preserved_frac'] * 100:.1f}%)")
    for k in ("ours", "pwm", "marginal", "random"):
        print(f"  {k:10s} AAR {main_res['aar'][k]:6.2f}%")
    mc = main_res["mcnemar_vs_pwm"]
    pstr = "n/a" if mc["p_two_sided"] is None else f"{mc['p_two_sided']:.2e}"
    print(f"\n  ours - pwm = {main_res['ours_minus_pwm']:+.2f} pp   "
          f"McNemar b={mc['b_ours_only']} c={mc['c_pwm_only']} p={pstr}")
    print(f"  -> {mc['verdict']}")
    probe = res["train_seen_probe"]
    if "aar" in probe:
        print(f"  [train-seen probe, n={probe['n_rows']}] "
              f"ours {probe['aar']['ours']:.2f}% vs pwm {probe['aar']['pwm']:.2f}% "
              f"-> {probe['ours_minus_pwm']:+.2f} pp, "
              f"{probe['mcnemar_vs_pwm']['verdict']}")
    print(f"-> {OUT / f'{tag}.json'}")


if __name__ == "__main__":
    main()
