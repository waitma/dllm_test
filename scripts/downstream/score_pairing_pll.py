"""Pairing pseudo-likelihood difference: p(L|H) − p(L) on OAS holdout.

DecoderTCR-style relation score for the fusion model. One denoise pass with
light masked (heavy clean) vs both chains masked. ESMC sees the same
corruption via ``apply_decoder_corruption_to_encoder``.

Headline pairing for generation remains ImmunoMatch
(``run_immune_fusion_pairing.sh``). This script is the model-native diagnostic.

Usage::

    python scripts/downstream/score_pairing_pll.py \\
        --checkpoint output/protein_esmc_llada270m_diffusion_chainratio_immune_v3_4gpu/checkpoint-final \\
        --output-json output/downstream_generation/ours_fusion_v3_chainratio_pll.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, BioSeqRecord
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    apply_decoder_corruption_to_encoder,
)
from dllm.pipelines.qwen3_vl_arch.relation_aux import (
    generated_heavy_light_masks,
    mean_token_logprob,
    pairing_auroc,
)
from examples.llada.load_fusion_checkpoint import load_fusion_for_eval

logger = logging.getLogger("score_pairing_pll")

DEFAULT_CSV = ROOT / "data/downstream/comp_chain/test_data_oas_holdout.csv"


def _oas_row_to_record(row: dict[str, str]) -> BioSeqRecord | None:
    heavy = (row.get("cleaned_h_sequence") or "").strip()
    light = (row.get("cleaned_l_sequence") or "").strip()
    if not heavy or not light:
        return None
    return BioSeqRecord(
        chains=[
            BioSeqChain(heavy, "antibody_heavy"),
            BioSeqChain(light, "antibody_light"),
        ],
        task_type="antibody",
        source="oas_holdout",
        labels={"relation": "binding"},
    )


def _rotate_lights(records: list[BioSeqRecord]) -> list[BioSeqRecord]:
    if len(records) < 2:
        raise ValueError("need at least 2 pairs to rotate lights")
    rotated: list[BioSeqRecord] = []
    for i, rec in enumerate(records):
        other = records[(i + 1) % len(records)]
        rotated.append(
            BioSeqRecord(
                chains=[rec.chains[0], other.chains[1]],
                task_type=rec.task_type,
                source=rec.source,
                labels={"relation": "nonbinding"},
            )
        )
    return rotated


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def _light_mean_logp(
    model: Any,
    batch: dict[str, Any],
    *,
    mask_heavy: bool,
) -> torch.Tensor:
    """Mean log p of the true light residues. ``[B]``."""

    heavy_mask, light_mask = generated_heavy_light_masks(batch)
    corruption = light_mask.clone()
    if mask_heavy:
        corruption = corruption | heavy_mask
    noised = batch["input_ids"].clone()
    noised[corruption] = int(model._decoder_mask_token_id)
    encoder_ids = apply_decoder_corruption_to_encoder(
        batch=batch,
        corruption_mask=corruption,
        mask_token_id=int(model._encoder_mask_token_id),
    )
    output = model._denoise(
        input_ids=noised,
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
    return mean_token_logprob(output.logits, batch["input_ids"], light_mask)


@torch.no_grad()
def score_records(
    bundle: Any,
    records: list[BioSeqRecord],
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float]]:
    model = bundle.model
    collator = bundle.collator
    rows: list[dict[str, float]] = []
    model.eval()
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        batch = _move_batch(collator(chunk), device)
        cond = _light_mean_logp(model, batch, mask_heavy=False)
        marg = _light_mean_logp(model, batch, mask_heavy=True)
        delta = cond - marg
        for i in range(len(chunk)):
            rows.append(
                {
                    "cond": float(cond[i].item()),
                    "marg": float(marg[i].item()),
                    "delta": float(delta[i].item()),
                }
            )
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="fusion Trainer dir")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all rows")
    parser.add_argument("--esmc-path", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    records: list[BioSeqRecord] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rec = _oas_row_to_record(row)
            if rec is None:
                continue
            records.append(rec)
            if args.max_pairs and len(records) >= int(args.max_pairs):
                break
    if len(records) < 2:
        raise RuntimeError(f"need ≥2 valid OAS pairs in {csv_path}, got {len(records)}")

    logger.info("Loaded %d OAS pairs from %s", len(records), csv_path)
    bundle = load_fusion_for_eval(
        args.checkpoint,
        device=args.device,
        esmc_path=args.esmc_path,
    )
    device = next(bundle.model.parameters()).device
    true_rows = score_records(
        bundle, records, batch_size=int(args.batch_size), device=device
    )
    mismatch_rows = score_records(
        bundle,
        _rotate_lights(records),
        batch_size=int(args.batch_size),
        device=device,
    )
    true_delta = [row["delta"] for row in true_rows]
    fake_delta = [row["delta"] for row in mismatch_rows]
    pairwise = [
        1.0 if t > f else 0.5 if t == f else 0.0
        for t, f in zip(true_delta, fake_delta)
    ]
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "csv": str(csv_path.resolve()),
        "n_pairs": len(records),
        "mean_cond_true": _mean([row["cond"] for row in true_rows]),
        "mean_marg_true": _mean([row["marg"] for row in true_rows]),
        "mean_delta_true": _mean(true_delta),
        "mean_cond_mismatch": _mean([row["cond"] for row in mismatch_rows]),
        "mean_marg_mismatch": _mean([row["marg"] for row in mismatch_rows]),
        "mean_delta_mismatch": _mean(fake_delta),
        "pairwise_acc": _mean(pairwise),
        "auroc_delta": pairing_auroc(true_delta, fake_delta),
        "true": true_rows,
        "mismatch": mismatch_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(
        "n=%d delta_true=%.4f delta_mismatch=%.4f pairwise=%.4f auroc=%.4f -> %s",
        summary["n_pairs"],
        summary["mean_delta_true"],
        summary["mean_delta_mismatch"],
        summary["pairwise_acc"],
        summary["auroc_delta"],
        args.output_json,
    )


if __name__ == "__main__":
    main()
