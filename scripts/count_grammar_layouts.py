"""Measure grammar layouts in a prepared immune LLaDA corpus, without models.

Usage::

    python scripts/count_grammar_layouts.py --prepared-data-dir data/prepared/immune_v3 \
        --split train --per-source 40000 --seed 0

Sample uniformly without replacement from each source's prepared row indexes
(default cap 40000, seed 0). This has the same sampling distribution as reservoir
sampling and avoids prefix bias without decoding every row. The prepared loader
still scans shards to build its offset index. --all streams every record and is
mutually exclusive with --per-source/--seed. Rendering errors fail immediately.

The weighted tables scale each source's sample by prepared_records/sample_size.
They estimate one pass over the selected prepared corpus, including duplicates;
record weights, training samplers and stochastic masking are not applied.
generated_tokens counts diffusion-eligible grammar positions, INCLUDING generated
structure/separator tokens, not only residues or realized training loss tokens.
The built-in residue tokenizer requires no weights or network. Token IDs are not
cached, and no new length cap or filtering policy is applied to prepared rows.
--sources selects the existing mix; raw --tokens/--*-dir overrides are retired.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dllm.pipelines.immune_llada.data.grammar import GrammarRenderer, GrammarTokenizer
from scripts.count_immune_mix import (
    counting_parser,
    load_manifest,
    load_source_dataset,
    percentage,
    selected_sources,
)


def sample_indices(population: int, limit: int, seed: int) -> Sequence[int]:
    if limit <= 0:
        raise ValueError("--per-source must be positive")
    if population <= limit:
        return range(population)
    return sorted(random.Random(seed).sample(range(population), limit))


def count_layouts(
    dataset_dir: str | Path, *, split: str = "train", sources: str | None = None,
    per_source: int | None = None, seed: int | None = None, all_records: bool = False,
) -> dict[str, Any]:
    if all_records and (per_source is not None or seed is not None):
        raise ValueError("--all cannot be combined with --per-source or --seed")
    limit = 40000 if per_source is None else per_source
    if limit <= 0:
        raise ValueError("--per-source must be positive")
    sample_seed = 0 if seed is None else seed
    root = Path(dataset_dir)
    manifest = load_manifest(root)
    selected = selected_sources(manifest, split, sources)
    # Length policy already ran offline. Do not impose the renderer's default
    # 1024 cap on a prepared corpus published with a different budget.
    renderer = GrammarRenderer(GrammarTokenizer(), ppi_max_protein_length=0)
    weighted_records: dict[str, float] = defaultdict(float)
    weighted_generated_tokens: dict[str, float] = defaultdict(float)
    source_rows: list[dict[str, Any]] = []
    for source in selected:
        dataset = load_source_dataset(root, manifest, split, source)
        population = len(dataset) if dataset is not None else 0
        indexes = range(population) if all_records else sample_indices(population, limit, sample_seed)
        layouts: Counter[str] = Counter()
        generated_tokens: Counter[str] = Counter()
        for index in indexes:
            assert dataset is not None
            try:
                rendered = renderer.encode(dataset[index])
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                raise ValueError(f"Cannot render prepared {split}/{source} record {index}: {exc}") from exc
            layout = str(rendered["grammar_name"])
            layouts[layout] += 1
            generated_tokens[layout] += sum(rendered["diffusion_loss_mask"])
        scale = population / len(indexes) if indexes else 0.0
        source_rows.append({
            "source": source, "prepared_records": population,
            "sampled_records": len(indexes), "scale": scale,
            "layouts": dict(layouts), "generated_tokens": dict(generated_tokens),
        })
        for name, count in layouts.items():
            weighted_records[name] += count * scale
        for name, count in generated_tokens.items():
            weighted_generated_tokens[name] += count * scale
        del dataset
    return {
        "prepared_data_dir": str(root), "split": split, "sources": selected,
        "sampling": "all_records" if all_records else "uniform_without_replacement",
        "per_source": None if all_records else limit,
        "seed": None if all_records else sample_seed,
        "basis": "prepared_corpus_record_counts_without_sample_weights",
        "generated_tokens_basis": "diffusion_eligible_positions_including_structure",
        "budget": manifest.get("budget"), "source_rows": source_rows,
        "weighted_layouts": dict(weighted_records),
        "weighted_generated_tokens": dict(weighted_generated_tokens),
    }


def _print_report(result: dict[str, Any]) -> None:
    full = result["sampling"] == "all_records"
    mode = "all prepared records" if full else f"uniform sample up to {result['per_source']:,}/source, seed={result['seed']}"
    print(f"prepared={result['prepared_data_dir']}  split={result['split']}  sources={'+'.join(result['sources'])}")
    print(f"sampling={mode}")
    print("Record-count scaling; weights/sampler exposure/random masking are not applied.")
    print("Generated tokens include diffusion-eligible grammar structure and separators.\n")
    for row in result["source_rows"]:
        print(
            f"{row['source']:16s} prepared={row['prepared_records']:>10,} sampled={row['sampled_records']:>8,} "
            f"x{row['scale']:>7.1f}  "
            + "  ".join(f"{name}={count:,}" for name, count in sorted(row["layouts"].items()))
        )
    records, generated = result["weighted_layouts"], result["weighted_generated_tokens"]
    total_records, total_generated = sum(records.values()), sum(generated.values())
    print("\n" + ("Full corpus counts" if full else "Estimated corpus counts"))
    print("layout                         records      rec%   generated_tokens      gen%")
    for name, count in sorted(records.items(), key=lambda item: (-item[1], item[0])):
        print(
            f"{name:24s} {count:>14,.2f} {percentage(count, total_records):>8.2f}% "
            f"{generated.get(name, 0):>18,.2f} {percentage(generated.get(name, 0), total_generated):>8.2f}%"
        )
    print(f"TOTAL records={total_records:,.2f} generated_tokens={total_generated:,.2f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = counting_parser(__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--per-source", type=int, help="positive sample cap per source (default: 40000)")
    selection.add_argument("--all", action="store_true", dest="all_records", help="stream/render every record")
    parser.add_argument("--seed", type=int, help="sampling seed (default: 0); invalid with --all")
    args = parser.parse_args(argv)
    try:
        result = count_layouts(
            args.prepared_data_dir, split=args.split, sources=args.sources,
            per_source=args.per_source, seed=args.seed, all_records=args.all_records,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
