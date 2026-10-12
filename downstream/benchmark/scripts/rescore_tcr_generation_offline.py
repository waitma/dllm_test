#!/usr/bin/env python
"""Recompute Setting B/C summaries from the per-pMHC values already on disk.

No generation and no model forward pass happens here: every ``metrics.json``
already stores a complete ``per_pmhc`` block, and the summary is a pure function
of that block.  Re-deriving it picks up two aggregation fixes:

* ``n_pmhc_scored`` -- category membership is not the number of pMHC that
  entered the macro average, because NaN entries are dropped.  A run conditioned
  on one view emits designs only for that view, so the other categories report a
  membership count far larger than what they averaged.
* the reserved simulation pMHC is excluded from the cross-cutting
  ``bioseq_seen`` / ``bioseq_unseen`` views.

Dry-run by default; pass ``--write`` to update the files in place.

    python downstream/benchmark/scripts/rescore_tcr_generation_offline.py
    python downstream/benchmark/scripts/rescore_tcr_generation_offline.py --write
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from tcr_generation_bench import scoring  # noqa: E402

OUT = BENCH / "outputs" / "tcr_generation_bench"


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def rescore_setting_a(write: bool) -> None:
    """Rescore Setting A from ``samples.txt`` under both reference builds.

    pgen is not recomputed: it is a property of the generated sequences alone and
    does not depend on the reference, so the stored block is reused.
    """
    runs = sorted(p for p in (OUT / "setting_A").iterdir() if (p / "samples.txt").exists())
    print(f"\n{'='*100}\nSETTING A -- rescore vs both reference builds\n{'='*100}")
    print(f"{'run':46s} {'build':10s} {'novelty':>8s} {'JSD':>8s} {'NNdist':>8s} "
          f"{'nov_ref':>10s} {'holdout':>8s}")
    for run in runs:
        gen = [s.strip() for s in (run / "samples.txt").read_text().splitlines() if s.strip()]
        data = json.loads((run / "metrics.json").read_text())
        old = data.get("metrics", {})
        print(f"{run.name[:46]:46s} {'STORED':10s} {_fmt(old.get('novelty')):>8s} "
              f"{_fmt(old.get('kmer_jsd')):>8s} {_fmt(old.get('mean_nn_distance')):>8s} "
              f"{'?':>10s} {'?':>8s}")
        fresh = {}
        for label, full in (("capped", False), ("fullref", True)):
            m = scoring.score_unconditional(gen, with_pgen=False, full_reference=full)
            fresh[label] = m
            print(f"{'':46s} {label:10s} {_fmt(m['novelty']):>8s} {_fmt(m['kmer_jsd']):>8s} "
                  f"{_fmt(m['mean_nn_distance']):>8s} {m['n_novelty_reference']:>10,} "
                  f"{m['n_holdout']:>8,}")
        if write:
            keep_pgen = {k: old[k] for k in ("pgen_generated", "pgen_note") if k in old}
            # Keep the block being replaced. The capped rescore reproduces the
            # stored value for a run whose samples.txt still matches what was
            # scored, so a disagreement here is a provenance signal worth an
            # audit trail rather than a silent overwrite.
            reproduced = _fmt(old.get("kmer_jsd")) == _fmt(fresh["capped"]["kmer_jsd"])
            if "previous_stored_metrics" not in data:
                data["previous_stored_metrics"] = old
                data["previous_stored_reproduced_by_capped_rescore"] = reproduced
            data["metrics"] = {**fresh["fullref"], **keep_pgen}
            data["metrics_capped_reference_diagnostic"] = fresh["capped"]
            data["rescore_note"] = (
                "Setting A rescored offline 2026-08-30 from samples.txt. Primary metrics use the "
                "repaired tcr_generation_fullref build (holdout disjoint from the full OTS corpus) "
                "and an exact full-corpus novelty reference. pgen is reused unchanged: it does not "
                "depend on the reference set. The prior block is kept under "
                "previous_stored_metrics; when the capped rescore reproduces it, the change is "
                "purely the reference swap."
            )
            (run / "metrics.json").write_text(json.dumps(data, indent=2))
    print(f"\n{'rewrote' if write else 'dry run for'} {len(runs)} Setting A runs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="update metrics.json in place (default: dry run)")
    ap.add_argument("--setting-a", action="store_true",
                    help="also rescore Setting A against the repaired reference build")
    args = ap.parse_args()

    if args.setting_a:
        rescore_setting_a(args.write)
        return

    paths = sorted(OUT.rglob("metrics.json"))
    if not paths:
        raise SystemExit(f"no metrics.json under {OUT}")

    n_changed = 0
    for path in paths:
        data = json.loads(path.read_text())
        per = data.get("per_pmhc")
        if not per:
            continue

        old = data.get("summary", {})
        new = scoring._aggregate_conditional(per)

        rows = []
        for view in sorted(set(old) | set(new)):
            o, n = old.get(view, {}), new.get(view, {})
            o_n, n_n = o.get("n_pmhc"), n.get("n_pmhc")
            n_scored = n.get("n_pmhc_scored")
            o_de, n_de = o.get("d_edit"), n.get("d_edit")
            moved = (o_n != n_n) or (_fmt(o_de) != _fmt(n_de)) or (n_scored != n_n)
            if moved:
                rows.append((view, o_n, n_n, n_scored, o_de, n_de))

        if rows:
            n_changed += 1
            print(f"\n{path.relative_to(OUT)}")
            print(f"  {'view':18s} {'n_old':>6s} {'n_new':>6s} {'scored':>7s} "
                  f"{'d_edit_old':>11s} {'d_edit_new':>11s}")
            for view, o_n, n_n, n_scored, o_de, n_de in rows:
                print(f"  {view:18s} {str(o_n):>6s} {str(n_n):>6s} {str(n_scored):>7s} "
                      f"{_fmt(o_de):>11s} {_fmt(n_de):>11s}")

        if args.write:
            data["summary"] = new
            data["reserved_simulation_in_eval_set"] = (
                scoring.RESERVED_SIMULATION_PMHC in per)
            path.write_text(json.dumps(data, indent=2))

    verb = "rewrote" if args.write else "would change"
    print(f"\n{verb} {n_changed}/{len(paths)} metrics.json")
    if not args.write:
        print("dry run -- pass --write to apply")


if __name__ == "__main__":
    main()
