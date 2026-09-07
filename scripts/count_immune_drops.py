"""Per-source attribution of how many rows the immune mix actually drops.

Reuses the training entry's own ``build_immune_specs`` so every predicate is the
production one. Four nested configurations are built and evaluated against the
SAME parsed CSV row in a single pass, which keeps the 2.9 GB OAS file to one
read instead of four:

  stage 0  raw          rows in the CSV
  stage 1  schema       ``row_to_record`` accepts (no blocklist, no length cap)
  stage 2  +length      max_protein_length / max_length caps applied
  stage 3  +supersede   replaces_trait blocklist (TRAIT rows superseded by native)
  stage 4  +decontam    benchmark blocklists = FINAL training rows

Usage:
  python scripts/count_immune_drops.py [--scenario as_run|current] [--split train]

``as_run`` reproduces the four completed v1 runs (ASD blocklist disabled, T4 and
T2/T3 blocklists did not exist yet, no tcr_papers). ``current`` is the config on
disk now (all blocklists active, tcr_papers included).

Both scenarios run at ``max_length = max_protein_length = 1024``, matching every
``train_jobs/*.yml``. Do not lower these to compare against older reports: at 512
the ASD antigen budget collapses to ~268 aa and ``asd_antibody`` reads 159,331
rows instead of its real 276,412.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "examples" / "llada"))

from dllm.pipelines.bioseq.datasets import _source_split_path  # noqa: E402
from protein_pretrain_esmc import DataArguments, build_immune_specs  # noqa: E402

csv.field_size_limit(2**31 - 1)

# "none" (not "") -- load_exclusion_keys rejects an empty path outright, since a
# blocklist that silently does nothing is indistinguishable from a working one.
_OFF = {
    "ots_benchmark_blocklist": "none",
    "oas_benchmark_blocklist": "none",
    "asd_antibody_benchmark_blocklist": "none",
    "asd_nanobody_benchmark_blocklist": "none",
    "trait_benchmark_blocklist": "none",
    "t4_refbinder_blocklist": "none",
    "t2t3_eval_blocklist": "none",
}


def _args(tokens: str, **over) -> DataArguments:
    # Length caps and blocklist paths are left at their DataArguments defaults,
    # which are the production values. Anything this script needs to turn off it
    # turns off explicitly via ``over``.
    base = dict(dataset_args=tokens)
    base.update(over)
    return DataArguments(**base)


def build_stages(tokens: str, *, t4: bool, asd: bool) -> list[tuple[str, dict]]:
    """Four nested configs -> {source_name: row_to_record}."""
    final_off = {}
    if not t4:
        final_off["t4_refbinder_blocklist"] = "none"
        final_off["t2t3_eval_blocklist"] = "none"
    if not asd:
        final_off["asd_antibody_benchmark_blocklist"] = "none"

    cfgs = [
        # stage 1: nothing on. Length caps off via 0/0 so with_length_filter is a no-op.
        ("schema", _args(tokens, max_length=0, max_protein_length=0,
                         replaces_trait_blocklist="none", **_OFF)),
        # stage 2: length caps only.
        ("length", _args(tokens, replaces_trait_blocklist="none", **_OFF)),
        # stage 3: + replaces_trait supersession.
        ("supersede", _args(tokens, **_OFF)),
        # stage 4: + benchmark decontamination (production).
        ("decontam", _args(tokens, **final_off)),
    ]
    out = []
    for name, da in cfgs:
        specs = build_immune_specs(da)
        out.append((name, {s.name: s for s in specs}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["as_run", "current"], default="as_run")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    if args.scenario == "as_run":
        # The four completed (v1) runs: no tcr_papers, and both the T4/T2-T3 and
        # the ASD antibody blocklists inert.
        tokens = "oas+ots+asd_antibody+trait+tcr_native"
        t4 = asd = False
    else:
        tokens = "oas+ots+asd_antibody+trait+tcr_native+tcr_papers"
        t4 = asd = True

    stages = build_stages(tokens, t4=t4, asd=asd)
    stage_names = [n for n, _ in stages]
    order = list(stages[-1][1].keys())

    print(f"scenario={args.scenario}  split={args.split}  tokens={tokens}")
    print(f"t4_refbinder+t2t3_eval={'ON' if t4 else 'OFF'}  "
          f"asd_antibody_benchmark_blocklist={'ON' if asd else 'OFF'}  "
          f"max_length=max_protein_length=1024\n")

    rows_out = []
    for name in order:
        spec = stages[-1][1][name]
        path = _source_split_path(spec, args.split)
        if not path.is_file():
            print(f"  {name:14s} MISSING {path}")
            continue
        fns = [(sn, sm[name].row_to_record) for sn, sm in stages if name in sm]
        counts = {sn: 0 for sn, _ in fns}
        raw = 0
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                raw += 1
                for sn, fn in fns:
                    if fn(row) is not None:
                        counts[sn] += 1
        rows_out.append((name, raw, counts))
        kept = counts[stage_names[-1]]
        print(f"  {name:14s} raw={raw:>9,} -> kept={kept:>9,}  "
              f"dropped={raw - kept:>9,} ({100 * (raw - kept) / raw:>5.2f}%)", flush=True)

    print(f"\n{'source':14s} {'raw':>10s} {'schema':>10s} {'+length':>10s} "
          f"{'+supersede':>11s} {'+decontam':>10s} | {'d_schema':>9s} {'d_length':>9s} "
          f"{'d_supers':>9s} {'d_decont':>9s} {'kept%':>7s}")
    tot = {k: 0 for k in ["raw"] + stage_names}
    for name, raw, c in rows_out:
        d_schema = raw - c["schema"]
        d_len = c["schema"] - c["length"]
        d_sup = c["length"] - c["supersede"]
        d_dec = c["supersede"] - c["decontam"]
        print(f"{name:14s} {raw:>10,} {c['schema']:>10,} {c['length']:>10,} "
              f"{c['supersede']:>11,} {c['decontam']:>10,} | {d_schema:>9,} {d_len:>9,} "
              f"{d_sup:>9,} {d_dec:>9,} {100 * c['decontam'] / raw:>6.2f}%")
        tot["raw"] += raw
        for k in stage_names:
            tot[k] += c[k]

    print(f"{'TOTAL':14s} {tot['raw']:>10,} {tot['schema']:>10,} {tot['length']:>10,} "
          f"{tot['supersede']:>11,} {tot['decontam']:>10,} | "
          f"{tot['raw'] - tot['schema']:>9,} {tot['schema'] - tot['length']:>9,} "
          f"{tot['length'] - tot['supersede']:>9,} {tot['supersede'] - tot['decontam']:>9,} "
          f"{100 * tot['decontam'] / tot['raw']:>6.2f}%")
    print(f"\ntotal dropped = {tot['raw'] - tot['decontam']:,} of {tot['raw']:,} "
          f"({100 * (tot['raw'] - tot['decontam']) / tot['raw']:.2f}%)")


if __name__ == "__main__":
    main()
