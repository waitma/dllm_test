"""Which grammar layouts does the model actually see, per source?

The T4 epitope-conditioned generation benchmark decodes with a
``noMHC + peptide + beta-only`` prompt, which the renderer names
``tcr_peptide``. If training almost never produces that layout the model is
being asked at inference time for a token arrangement it never fit, so this
script measures the real per-source layout distribution.

Sampled (``--per-source``) because rendering the full 7.8M-row mix is far more
work than the distribution warrants.

The sample must be a RESERVOIR sample, not a prefix. Layouts are a function of
which fields a row populates, and several sources are concatenations of
heterogeneous corpora whose layout mix varies by position: ``tcr_papers_v2`` is
seven paper corpora written back to back, and its first 30,000 rows are 100%
``tcr_peptide`` while the full split is 80.3% ``tcr_pmhc`` / 19.7%
``tcr_peptide``. Prefix sampling therefore mis-attributed 547,274 rows to the
wrong layout and understated ``tcr_pmhc`` by ~4x.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "examples" / "llada"))

from dllm.pipelines.bioseq.datasets import _source_split_path  # noqa: E402
from dllm.pipelines.qwen3_vl_arch.data.grammar import (  # noqa: E402
    BioSeqChain,
    BioSeqRecord,
    GrammarRenderer,
    GrammarTokenizer,
)
from protein_pretrain_esmc import DataArguments, build_immune_specs  # noqa: E402

csv.field_size_limit(2**31 - 1)


def to_record(rec: dict) -> BioSeqRecord:
    chains = [
        BioSeqChain(sequence=s, role=r)
        for s, r in zip(rec["chains"], rec.get("roles") or [""] * len(rec["chains"]))
    ]
    return BioSeqRecord(
        chains=chains,
        task_type=str(rec.get("task_type") or ""),
        source=str(rec.get("source") or ""),
        labels={"relation": rec.get("relation", "unknown")},
        weight=float(rec.get("weight", 1.0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokens",
                    default="oas+ots+asd_antibody+trait+tcr_native+tcr_papers")
    ap.add_argument("--tcr-papers-dir", default=None,
                    help="override, e.g. the tcr_papers_v2 corpus")
    ap.add_argument("--tcr-repertoire-dir", default=None)
    args = ap.parse_args()

    overrides = {}
    if args.tcr_papers_dir:
        overrides["tcr_papers_dir"] = args.tcr_papers_dir
    if args.tcr_repertoire_dir:
        overrides["tcr_repertoire_dir"] = args.tcr_repertoire_dir
    data_args = DataArguments(
        dataset_args=args.tokens,
        max_length=1024,
        max_protein_length=1024,
        **overrides,
    )
    specs = build_immune_specs(data_args)
    gt = GrammarTokenizer()
    renderer = GrammarRenderer(gt)

    grand = Counter()
    gen_res = Counter()
    # Mix-weighted view: a per-source cap over-represents the small TCR sources.
    # Scale each source's sampled shares by its true row count so the numbers
    # reflect what the model actually spends its loss budget on.
    w_rec: Counter = Counter()
    w_gen: Counter = Counter()
    print(f"sampling up to {args.per_source:,} rows/source  split={args.split}\n")
    for spec in specs:
        path = _source_split_path(spec, args.split)
        if not path.is_file():
            continue
        # Reservoir sample over the kept rows, then render only the reservoir.
        rng = random.Random(args.seed)
        reservoir: list[dict] = []
        kept = 0
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                rec = spec.row_to_record(row)
                if rec is None:
                    continue
                kept += 1
                if len(reservoir) < args.per_source:
                    reservoir.append(rec)
                else:
                    j = rng.randrange(kept)
                    if j < args.per_source:
                        reservoir[j] = rec

        local = Counter()
        local_gen = Counter()
        n = 0
        for rec in reservoir:
            try:
                out = renderer.encode(to_record(rec))
            except Exception:
                local["<render_error>"] += 1
                continue
            name = out["grammar_name"]
            local[name] += 1
            local_gen[name] += sum(out["diffusion_loss_mask"])
            n += 1
        total_rows = kept
        scale = total_rows / n if n else 0.0
        print(f"  {spec.name:14s} n={n:>7,}  kept={total_rows:>9,}  x{scale:>6.1f}  " + "  ".join(
            f"{k}={v:,}" for k, v in local.most_common()))
        grand.update(local)
        gen_res.update(local_gen)
        for k, v in local.items():
            w_rec[k] += v * scale
        for k, v in local_gen.items():
            w_gen[k] += v * scale

    def table(title, rec, gen, note):
        tot, tot_gen = sum(rec.values()), sum(gen.values())
        print(f"\n=== {title} ===")
        print(f"{'layout':18s} {'records':>13s} {'rec%':>7s} {'gen_tokens':>15s} {'gen%':>7s}")
        for name, cnt in rec.most_common():
            print(f"{name:18s} {cnt:>13,.0f} {100 * cnt / tot:>6.2f}% "
                  f"{gen[name]:>15,.0f} {100 * gen[name] / tot_gen:>6.2f}%")
        print(note)

    table("per-source capped (each source counted equally)", grand, gen_res,
          "  -> flatters the small TCR sources; not what training saw.")
    table("MIX-WEIGHTED (scaled to true row counts)", w_rec, w_gen,
          "  -> this is the real share of the training loss budget.")
    print("\nThe T4 decode prompt renders as 'tcr_peptide' with a single beta chain.")


if __name__ == "__main__":
    main()
