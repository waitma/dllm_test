"""Offline teacher-forced binding / nonbinding relation-token eval.

The recognition relation is a sequence token (``<binding>`` / ``<nonbinding>``),
not a classification head. This script never imports the training step and does
not modify any file the queued trainer will execute. It loads a saved fusion
checkpoint, collates prepared validation records, masks only the supervised
relation site, and scores that site.

Positions MUST be taken from ``relation_target_mask == 1``.
``relation_token_mask`` also lights up MHC→peptide presentation ``<binding>``
and the OAS/OTS ``<unknown>`` null prefix; those are fixed context and would
contaminate the metric.

This is not pairing PLL. Do not reuse ``pairing_auroc`` / ``mean_token_logprob``.

Usage::

    /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \\
        scripts/downstream/score_binding_relation.py \\
        --checkpoint output/protein_esmc_llada270m_diffusion_immune_v3/checkpoint-42000 \\
        --prepared-data-dir data/prepared/immune_v5_receptor_completion \\
        --max-records 400 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dllm.pipelines.immune_llada.data.preprocessing.validators import (
    validate_prepared_row,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    apply_decoder_corruption_to_encoder,
)
from examples.llada.load_fusion_checkpoint import load_fusion_for_eval

logger = logging.getLogger("score_binding_relation")

DEFAULT_PREPARED = ROOT / "data/prepared/immune_v5_receptor_completion"
DEFAULT_SOURCES = ("asd_antibody", "tcr_papers", "tcr_native", "trait")
SCORED_RELATIONS = frozenset({"binding", "nonbinding"})


def _relation_label(record: Any) -> str:
    raw = record.labels.get("relation") or record.metadata.get("relation") or ""
    return str(raw).strip().lower().replace(" ", "_").replace("-", "_")


def is_supervised_relation_record(record: Any) -> bool:
    """Same gate as ``grammar.py`` L442: metadata flag plus a scored token."""

    return bool(record.metadata.get("relation_supervised")) and _relation_label(
        record
    ) in SCORED_RELATIONS


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def mann_whitney_auroc(scores: list[float], labels: list[int]) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie). Empty class → NaN."""

    positive = [s for s, y in zip(scores, labels) if y == 1]
    negative = [s for s, y in zip(scores, labels) if y == 0]
    if not positive or not negative:
        return float("nan")
    pos = torch.tensor(positive, dtype=torch.float64)
    neg = torch.tensor(negative, dtype=torch.float64)
    greater = pos.unsqueeze(1) > neg.unsqueeze(0)
    tied = pos.unsqueeze(1) == neg.unsqueeze(0)
    return float((greater.double() + 0.5 * tied.double()).mean())


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def _fmt(value: float, digits: int = 4) -> str:
    if value != value:  # NaN
        return "nan"
    return f"{value:.{digits}f}"


def shard_paths(prepared_dir: Path, *, split: str, source: str) -> list[Path]:
    manifest_path = prepared_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_info = manifest["splits"].get(split)
    if split_info is None:
        raise KeyError(f"split {split!r} missing from {manifest_path}")
    paths = [
        prepared_dir / str(shard["path"])
        for shard in split_info.get("shards", [])
        if shard.get("source") == source
    ]
    if not paths:
        raise ValueError(f"no {split} shards for source {source!r} in {manifest_path}")
    return paths


def iter_prepared_records(paths: Iterable[Path]) -> Iterable[Any]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield validate_prepared_row(json.loads(line))


def load_supervised_records(
    prepared_dir: Path,
    *,
    split: str,
    sources: list[str],
    max_records: int,
) -> list[Any]:
    """Stream prepared shards; keep supervised binding/nonbinding rows.

    When ``max_records > 0`` the budget is split across sources, and each
    source tries to take half binding / half nonbinding so a binding-first
    shard (e.g. asd_antibody) does not hide the negative class.
    """

    per_source = 0
    extra = 0
    if max_records > 0:
        per_source = max_records // len(sources)
        extra = max_records % len(sources)
        if per_source == 0:
            raise ValueError(
                f"--max-records {max_records} is smaller than {len(sources)} sources"
            )

    collected: list[Any] = []
    for index, source in enumerate(sources):
        quota = 0 if max_records <= 0 else per_source + (1 if index < extra else 0)
        want_pos = quota // 2 if quota else 0
        want_neg = quota - want_pos if quota else 0
        pos: list[Any] = []
        neg: list[Any] = []
        overflow: list[Any] = []
        for record in iter_prepared_records(shard_paths(prepared_dir, split=split, source=source)):
            if not is_supervised_relation_record(record):
                continue
            # Prepared tcr_papers rows still store source="tcr_native"; keep the
            # shard name so per-source metrics are not swallowed.
            record.metadata["eval_shard_source"] = source
            bucket = pos if _relation_label(record) == "binding" else neg
            if quota <= 0:
                bucket.append(record)
                continue
            if _relation_label(record) == "binding":
                if len(pos) < want_pos:
                    pos.append(record)
                else:
                    overflow.append(record)
            else:
                if len(neg) < want_neg:
                    neg.append(record)
                else:
                    overflow.append(record)
            if len(pos) >= want_pos and len(neg) >= want_neg:
                break
        chosen = pos + neg
        if quota > 0 and len(chosen) < quota:
            chosen.extend(overflow[: quota - len(chosen)])
        logger.info(
            "source %s: kept %d supervised (binding=%d nonbinding=%d)",
            source,
            len(chosen),
            sum(1 for r in chosen if _relation_label(r) == "binding"),
            sum(1 for r in chosen if _relation_label(r) == "nonbinding"),
        )
        collected.extend(chosen)
    return collected


def _first_matching_record(
    prepared_dir: Path,
    *,
    split: str,
    source: str,
    predicate,
) -> Any | None:
    try:
        paths = shard_paths(prepared_dir, split=split, source=source)
    except (KeyError, ValueError):
        return None
    for record in iter_prepared_records(paths):
        if predicate(record):
            return record
    return None


def _contrast_records_for_masks(prepared_dir: Path, *, split: str) -> list[Any]:
    """OAS null-prefix and MHC presentation cases that must not be scored."""

    extras: list[Any] = []
    oas = _first_matching_record(
        prepared_dir, split=split, source="oas", predicate=lambda rec: True
    )
    if oas is not None:
        oas.metadata["eval_shard_source"] = "oas"
        extras.append(oas)
    mhc_trait = _first_matching_record(
        prepared_dir,
        split=split,
        source="trait",
        predicate=lambda rec: any(
            chain.role.lower() in {"mhc", "pmhc", "hla"} for chain in rec.chains
        )
        and is_supervised_relation_record(rec),
    )
    if mhc_trait is not None:
        mhc_trait.metadata["eval_shard_source"] = "trait"
        extras.append(mhc_trait)
    return extras


def _report_source(record: Any) -> str:
    return str(record.metadata.get("eval_shard_source") or record.source)


def describe_mask_semantics(
    records: list[Any],
    collator: Any,
    id_to_token,
    *,
    max_examples: int = 8,
) -> list[dict[str, Any]]:
    """Collate a few records and report target vs any-relation token sites."""

    examples: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for record in records:
        source_name = _report_source(record)
        if source_name in seen_sources and len(examples) >= 4:
            continue
        if len(examples) >= max_examples:
            break
        seen_sources.add(source_name)
        batch = collator([record])
        target = batch["relation_target_mask"][0].bool()
        relation = batch["relation_token_mask"][0].bool()
        fixed = batch["fixed_context_mask"][0].bool()
        loss = batch["diffusion_loss_mask"][0].bool()
        ids = batch["input_ids"][0].tolist()
        target_idx = torch.nonzero(target, as_tuple=False).flatten().tolist()
        extra_idx = torch.nonzero(relation & ~target, as_tuple=False).flatten().tolist()
        extra_tokens = [str(id_to_token(int(ids[index]))) for index in extra_idx]
        examples.append(
            {
                "source": _report_source(record),
                "relation": _relation_label(record),
                "relation_supervised": bool(record.metadata.get("relation_supervised")),
                "n_relation_token": int(relation.sum().item()),
                "n_relation_target": int(target.sum().item()),
                "target_indices": target_idx,
                "target_fixed": [bool(fixed[i]) for i in target_idx],
                "target_in_loss": [bool(loss[i]) for i in target_idx],
                "extra_relation_indices": extra_idx,
                "extra_relation_tokens": extra_tokens,
                "extra_fixed": [bool(fixed[i]) for i in extra_idx],
                "extra_in_loss": [bool(loss[i]) for i in extra_idx],
            }
        )
    return examples


def _token_id(tokenizer: Any, token: str) -> int:
    token_id = int(tokenizer.convert_tokens_to_ids(token))
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and token_id == int(unk):
        raise RuntimeError(f"tokenizer maps {token!r} to unk ({unk})")
    return token_id


@torch.no_grad()
def score_records(
    bundle: Any,
    records: list[Any],
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Mask only ``relation_target_mask`` sites and score those logits."""

    model = bundle.model
    collator = bundle.collator
    binding_id = _token_id(bundle.llada_tokenizer, "<binding>")
    nonbinding_id = _token_id(bundle.llada_tokenizer, "<nonbinding>")
    mask_id = int(model._decoder_mask_token_id)
    encoder_mask_id = int(model._encoder_mask_token_id)
    rows: list[dict[str, Any]] = []
    model.eval()

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        batch = _move_batch(collator(chunk), device)
        target = batch["relation_target_mask"].bool()
        if batch.get("attention_mask") is not None:
            target = target & batch["attention_mask"].bool()
        counts = target.sum(dim=1)
        if bool((counts == 0).any()):
            missing = [chunk[i].source for i in range(len(chunk)) if int(counts[i]) == 0]
            raise RuntimeError(
                "record(s) passed the metadata gate but have no relation_target_mask "
                f"site after collation: {missing[:8]}"
            )

        clean_ids = batch["input_ids"]
        noised_ids = clean_ids.clone()
        noised_ids[target] = mask_id
        encoder_ids = apply_decoder_corruption_to_encoder(
            batch=batch,
            corruption_mask=target,
            mask_token_id=encoder_mask_id,
        )
        output = model._denoise(
            input_ids=noised_ids,
            attention_mask=batch.get("attention_mask"),
            chain_ids=batch.get("chain_ids"),
            position_ids_inner=batch.get("position_ids_inner"),
            position_ids_chain=batch.get("position_ids_chain"),
            residue_mask=batch.get("residue_mask"),
            encoder_input_ids=encoder_ids,
            encoder_attention_mask=batch.get("encoder_attention_mask"),
            encoder_residue_mask=batch.get("encoder_residue_mask"),
            encoder_chain_mask=batch.get("encoder_chain_mask"),
            encoder_position_ids=batch.get("encoder_position_ids"),
        )
        logits = output.logits
        log_probs = F.log_softmax(logits.float(), dim=-1)

        for row_index, record in enumerate(chunk):
            positions = torch.nonzero(target[row_index], as_tuple=False).flatten()
            gold_ids = clean_ids[row_index, positions]
            gold_nll = -log_probs[row_index, positions, gold_ids]
            pred_ids = logits[row_index, positions].argmax(dim=-1)
            bind_logit = logits[row_index, positions, binding_id].float()
            nonbind_logit = logits[row_index, positions, nonbinding_id].float()
            scores = bind_logit - nonbind_logit
            gold_label = _relation_label(record)
            for pos_i, position in enumerate(positions.tolist()):
                nll = float(gold_nll[pos_i].item())
                score = float(scores[pos_i].item())
                pred_id = int(pred_ids[pos_i].item())
                gold_id = int(gold_ids[pos_i].item())
                rows.append(
                    {
                        "source": _report_source(record),
                        "gold_relation": gold_label,
                        "label": 1 if gold_label == "binding" else 0,
                        "position": int(position),
                        "nll": nll,
                        "correct": 1.0 if pred_id == gold_id else 0.0,
                        "score": score,
                        "logit_binding": float(bind_logit[pos_i].item()),
                        "logit_nonbinding": float(nonbind_logit[pos_i].item()),
                        "pred_id": pred_id,
                        "gold_id": gold_id,
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    groups["ALL"] = rows
    summary: dict[str, dict[str, Any]] = {}
    for name, items in groups.items():
        labels = [int(item["label"]) for item in items]
        scores = [float(item["score"]) for item in items]
        nlls = [float(item["nll"]) for item in items]
        pos_scores = [item["score"] for item in items if item["label"] == 1]
        neg_scores = [item["score"] for item in items if item["label"] == 0]
        n = len(items)
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        summary[name] = {
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_frac": (n_pos / n) if n else float("nan"),
            "mean_nll": _mean(nlls),
            "accuracy": _mean([float(item["correct"]) for item in items]),
            "auroc": mann_whitney_auroc(scores, labels),
            "score_pos_mean": _mean(pos_scores),
            "score_neg_mean": _mean(neg_scores),
            "score_pos_median": _median(pos_scores),
            "score_neg_median": _median(neg_scores),
        }
    return summary


def print_table(summary: dict[str, dict[str, Any]], source_order: list[str]) -> None:
    headers = (
        "source",
        "n",
        "pos",
        "neg",
        "pos%",
        "nll",
        "acc",
        "auroc",
        "score_pos_mean",
        "score_neg_mean",
        "score_pos_med",
        "score_neg_med",
    )
    print("\t".join(headers))
    for name in list(source_order) + ["ALL"]:
        if name not in summary:
            continue
        row = summary[name]
        print(
            "\t".join(
                [
                    name,
                    str(row["n"]),
                    str(row["n_pos"]),
                    str(row["n_neg"]),
                    _fmt(100.0 * row["pos_frac"], 1),
                    _fmt(row["mean_nll"]),
                    _fmt(row["accuracy"]),
                    _fmt(row["auroc"]),
                    _fmt(row["score_pos_mean"]),
                    _fmt(row["score_neg_mean"]),
                    _fmt(row["score_pos_median"]),
                    _fmt(row["score_neg_median"]),
                ]
            )
        )


def print_mask_examples(examples: list[dict[str, Any]]) -> None:
    print("# mask semantics (collated prepared records)")
    print(
        "rule: score only relation_target_mask==1; relation_token_mask also "
        "includes presentation <binding> and null-prefix <unknown>"
    )
    for ex in examples:
        extra = ",".join(
            f"{idx}:{tok}(fixed={fix},loss={loss})"
            for idx, tok, fix, loss in zip(
                ex["extra_relation_indices"],
                ex["extra_relation_tokens"],
                ex["extra_fixed"],
                ex["extra_in_loss"],
            )
        ) or "-"
        print(
            f"  {ex['source']} gold={ex['relation']} "
            f"rel_tokens={ex['n_relation_token']} targets={ex['n_relation_target']} "
            f"target_idx={ex['target_indices']} target_fixed={ex['target_fixed']} "
            f"target_loss={ex['target_in_loss']} extra=[{extra}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="fusion Trainer dir")
    parser.add_argument(
        "--prepared-data-dir",
        type=Path,
        default=DEFAULT_PREPARED,
        help="prepared immune_llada directory with dataset_manifest.json",
    )
    parser.add_argument("--split", default="valid")
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="comma-separated prepared source names",
    )
    parser.add_argument("--max-records", type=int, default=0, help="0 = all supervised rows")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--esmc-path", default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional JSON dump; write outside the repo or into an ignored dir",
    )
    parser.add_argument(
        "--verify-masks-only",
        action="store_true",
        help="collate a few records and print mask semantics; do not load weights",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    prepared_dir = Path(args.prepared_data_dir)
    if not (prepared_dir / "dataset_manifest.json").is_file():
        raise FileNotFoundError(prepared_dir / "dataset_manifest.json")
    sources = [part.strip() for part in str(args.sources).split(",") if part.strip()]
    if not sources:
        raise ValueError("--sources is empty")

    records = load_supervised_records(
        prepared_dir,
        split=str(args.split),
        sources=sources,
        max_records=int(args.max_records),
    )
    if not records:
        raise RuntimeError(
            f"no relation_supervised binding/nonbinding rows in {prepared_dir} "
            f"split={args.split} sources={sources}"
        )
    logger.info("scoring %d records from %s", len(records), prepared_dir)
    contrast = _contrast_records_for_masks(prepared_dir, split=str(args.split))

    if args.verify_masks_only:
        from dllm.pipelines.immune_llada.data import (
            GrammarBioSeqCollator,
            GrammarTokenizer,
        )

        collator = GrammarBioSeqCollator(
            tokenizer=GrammarTokenizer(),
            max_sequence_length=1024,
            max_protein_length=1024,
        )
        examples = describe_mask_semantics(
            contrast + records,
            collator,
            GrammarTokenizer().token,
            max_examples=10,
        )
        print_mask_examples(examples)
        return

    bundle = load_fusion_for_eval(
        args.checkpoint,
        device=args.device,
        esmc_path=args.esmc_path,
    )
    device = next(bundle.model.parameters()).device
    examples = describe_mask_semantics(
        contrast + records,
        bundle.collator,
        bundle.llada_tokenizer.convert_ids_to_tokens,
        max_examples=10,
    )
    print_mask_examples(examples)

    rows = score_records(
        bundle, records, batch_size=int(args.batch_size), device=device
    )
    summary = summarize(rows)
    print(
        f"# checkpoint {Path(args.checkpoint).resolve()}  "
        f"n_records={len(records)} n_sites={len(rows)}  "
        f"input=mask relation_target_mask sites only"
    )
    print_table(summary, sources)

    if args.output_json is not None:
        payload = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "prepared_data_dir": str(prepared_dir.resolve()),
            "split": args.split,
            "sources": sources,
            "n_records": len(records),
            "n_sites": len(rows),
            "construction": (
                "mask only relation_target_mask positions with decoder <|mdm_mask|>; "
                "mirror via apply_decoder_corruption_to_encoder; gold tokens stay in labels"
            ),
            "mask_examples": examples,
            "summary": summary,
            "rows": rows,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("wrote %s", args.output_json)

    if any(
        (not math.isfinite(summary[name]["mean_nll"]))
        or summary[name]["mean_nll"] < 0
        or not (0.0 <= summary[name]["accuracy"] <= 1.0)
        or (
            summary[name]["auroc"] == summary[name]["auroc"]
            and not (0.0 <= summary[name]["auroc"] <= 1.0)
        )
        for name in summary
        if summary[name]["n"]
    ):
        raise RuntimeError("metric out of expected range (NLL>=0, acc/AUROC in [0,1])")


if __name__ == "__main__":
    main()
