"""Exact per-source record counts for the immune training mix.

Reuses the training entry's own spec builder (benchmark blocklists + length
filters) but streams rows instead of retaining them, so it can count the full
7.6M-row mix without the ~10GB the real in-memory dataset needs.

Usage (all fields default to the production values, notably the 1024 caps)::

    python scripts/count_immune_mix.py train \\
      "oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire" \\
      tcr_papers_dir=.../data/tcr_papers_v2/dataset

Three traps this script exists to avoid:

1. ``tcr_papers_dir`` defaulted to the *v1* corpus until 2026-08-29, so a
   flagless run before that date silently measured v1 (408,037 rows) instead of
   the v2 corpus v3 actually trains on. The default now points at v2, but any
   count produced earlier may be on the old basis. Pass trailing ``field=value``
   args to reproduce a specific job.
2. **Counting one token is not the same as counting it inside the mix.**
   ``replaces_trait`` supersession keys are only assembled when the superseding
   sources are also requested, so ``trait`` alone reports more rows than the
   same source inside the full mix. Always pass the job's whole token list when
   you want per-source numbers that match a training log.
3. **Corpora and blocklists move independently, and reports do not always
   follow.** ``tcr_repertoire``'s ``build_report.json::split_counts`` went stale
   when the near-duplicate migration rewrote its CSVs, and ``trait`` dropped
   from 34,872 to 31,515 rows when ``trait_benchmark_blocklist`` was rebuilt
   from 862 to 59,212 keys. Re-run this script rather than quoting any document.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "examples" / "llada"))

from dllm.pipelines.bioseq.datasets import _iter_csv_records, _source_split_path
from protein_pretrain_esmc import DataArguments, build_immune_specs


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    tokens = sys.argv[2] if len(sys.argv) > 2 else "oas+ots+asd_antibody+trait+tcr_native"
    # Trailing ``field=value`` args override DataArguments. Needed because the
    # per-source *_dir defaults are not always what a job trains on: the v3 yml
    # passes --tcr_papers_dir .../tcr_papers_v2/dataset while the default still
    # points at v1, so a flagless count silently measures the v2 corpus.
    overrides: dict[str, str] = {}
    for arg in sys.argv[3:]:
        key, _, value = arg.partition("=")
        overrides[key] = value
    # Everything else left at its DataArguments default on purpose: the defaults
    # are the production values (1024 caps, all benchmark blocklists at their
    # real paths). Overriding those is what made earlier counts disagree with
    # the training logs.
    data_args = DataArguments(dataset_args=tokens, **overrides)
    if overrides:
        print(f"overrides: {overrides}")
    specs = build_immune_specs(data_args)

    total = 0
    rows = []
    for spec in specs:
        path = _source_split_path(spec, split)
        raw = kept = 0
        n_chains = n_res = n_gen_res = 0
        for record in _iter_csv_records(path, spec.row_to_record, None):
            kept += 1
            chains = record["chains"]
            roles = record.get("roles") or []
            n_chains += len(chains)
            n_res += sum(len(s) for s in chains)
            fixed = {"antigen", "mhc", "pmhc", "hla", "peptide", "epitope"}
            for i, seq in enumerate(chains):
                role = roles[i].lower() if i < len(roles) else ""
                if role not in fixed:
                    n_gen_res += len(seq)
        with path.open(newline="") as fh:
            raw = sum(1 for _ in fh) - 1
        total += kept
        rows.append((spec.name, raw, kept, n_res, n_gen_res, n_chains))
        print(f"  {spec.name:14s} raw={raw:>9,}  kept={kept:>9,}  "
              f"res={n_res:>12,}  gen_res={n_gen_res:>12,}", flush=True)

    print(f"\nsplit={split}  TOTAL kept = {total:,}")
    tot_res = sum(r[3] for r in rows)
    tot_gen = sum(r[4] for r in rows)
    print(f"total residues = {tot_res:,}   generated-chain residues = {tot_gen:,}\n")
    print(f"{'source':14s} {'records':>10s} {'rec%':>7s} {'res%':>7s} {'gen_res%':>9s} {'res/rec':>8s}")
    for name, _raw, kept, n_res, n_gen, _nc in rows:
        print(f"{name:14s} {kept:>10,} {100*kept/total:>6.2f}% "
              f"{100*n_res/tot_res:>6.2f}% {100*n_gen/tot_gen:>8.2f}% {n_res/kept:>8.0f}")


if __name__ == "__main__":
    main()
