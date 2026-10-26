"""Deterministic conditional diffusion/edit validation on the frozen V7 holdout.

Activate pllm; python -B -m scripts.research.diffusion_continuation_eval
--checkpoint ABS --index ABS --output ABS --device cuda
Uses only token_valid; sampling/loss weights never alter reported CE metrics.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import json

import torch
from torch.nn import functional as F

from scripts.research.bert_mlm_eval import read_samples, autocast
from scripts.research.bert_mlm_data import write_json
from scripts.research.diffusion_continuation_objective import DiffusionEditCollator, DiffusionEditModel, corrupt

MODES = (("mask25", "joint", .25, False), ("mask50", "joint", .5, False),
         ("mask75", "joint", .75, False), ("mask100", "joint", 1., False),
         ("edit15", "joint", .15, True), ("edit50", "joint", .5, True),
         ("first_to_second", "first_to_second", 1., False),
         ("second_to_first", "second_to_first", 1., False))
FIELDS = ("ce", "tokens", "correct", "eos_ce", "eos_tokens", "eos_correct",
          "relation_ce", "relation_tokens", "relation_correct", "aa_tokens", "early_eos",
          "keep_tokens", "keep_edits", "chains", "first_correct", "missing_eos",
          "length_error", "length_compared", "records")


def fixed_batch(collator, samples, indices, recipe, mode, ratio, edit, seed=61019,
                *, mask_scope="chain", target_cdrs=None):
    """Independent per-record RNG before padding: world/batch partition invariant."""
    batch = collator([samples[i]["record"] for i in indices])
    result = {key: value.clone() for key, value in batch.items()}
    for key in ("noise_kind", "loss_weights", "active_population"):
        result[key] = torch.zeros_like(batch["input_ids"], dtype=(
            torch.float32 if key == "loss_weights" else torch.bool if key == "active_population" else torch.long))
    for row, index in enumerate(indices):
        length = int(batch["attention_mask"][row].sum())
        one = {key: value[row:row+1, :length] for key, value in batch.items()}
        noisy = corrupt(one, collator.tokenizer, torch.Generator().manual_seed(seed+index),
                        recipe, mode=mode, ratio=ratio, edit=edit, paired=False,
                        mask_scope=mask_scope, target_cdrs=target_cdrs)
        for key in ("input_ids", "labels", "noise_kind", "loss_weights", "active_population"):
            result[key][row, :length] = noisy[key][0]
    return result


def summarize(values):
    v = dict(zip(FIELDS, values))
    def div(a, b):
        return v[a]/v[b] if v[b] else None
    return dict(loss=div("ce", "tokens"), accuracy=div("correct", "tokens"),
                eos_loss=div("eos_ce", "eos_tokens"), eos_accuracy=div("eos_correct", "eos_tokens"),
                relation_loss=div("relation_ce", "relation_tokens"),
                relation_accuracy=div("relation_correct", "relation_tokens"),
                early_eos_rate=div("early_eos", "aa_tokens"),
                keep_error_rate=div("keep_edits", "keep_tokens"),
                first_eos_accuracy=div("first_correct", "chains"),
                missing_eos_rate=div("missing_eos", "chains"),
                length_mae=div("length_error", "length_compared"), counts=v)


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values)/len(values) if values else None


@torch.no_grad()
def evaluate(model, index, device, rank=0, world=1, batch_size=8):
    samples = read_samples(index["files"]["token_valid"])
    sources = sorted({row["source_group"] for row in samples})
    collator = DiffusionEditCollator(model.config)
    was_training = model.training
    model.eval()
    records = list(range(rank, len(samples), world))
    results = {}
    try:
        for name, mode, ratio, edit in MODES:
            totals = torch.zeros((len(sources), len(FIELDS)), device=device, dtype=torch.float64)
            for start in range(0, len(records), batch_size):
                indices = records[start:start+batch_size]
                batch = {key: value.to(device) for key, value in fixed_batch(
                    collator, samples, indices, model.recipe, mode, ratio, edit).items()}
                with autocast(device):
                    logits, _ = model.denoise(**model.feature_args(batch))
                pred = logits.argmax(-1)
                labels = batch["labels"]
                for row, sample_index in enumerate(indices):
                    v = defaultdict(float)
                    receptor = batch["noise_kind"][row].eq(1) | batch["noise_kind"][row].eq(2)
                    eos = receptor & labels[row].eq(model.tokenizer.eos_token_id)
                    relation = batch["relation_mask"][row]
                    for mask, prefix in ((receptor, ""), (eos, "eos_"), (relation, "relation_")):
                        count = int(mask.sum())
                        v[prefix+"tokens"] = count
                        if count:
                            v[prefix+"ce"] = float(F.cross_entropy(logits[row, mask].float(), labels[row, mask], reduction="sum"))
                            v[prefix+"correct"] = int(pred[row, mask].eq(labels[row, mask]).sum())
                    aa = receptor & batch["residue_mask"][row]
                    v["aa_tokens"] = int(aa.sum())
                    v["early_eos"] = int((aa & pred[row].eq(model.tokenizer.eos_token_id)).sum())
                    kept = batch["noise_kind"][row].eq(3)
                    v["keep_tokens"] = int(kept.sum())
                    v["keep_edits"] = int((kept & pred[row].ne(labels[row])).sum())
                    if not edit and ratio == 1.:
                        terminal = batch["terminal_mask"][row] & batch["active_population"][row]
                        for chain in batch["chain_ids"][row, terminal].unique().tolist():
                            positions = (batch["target_slot_mask"][row] & batch["chain_ids"][row].eq(chain)).nonzero().flatten()
                            end = batch["terminal_mask"][row, positions].nonzero().flatten()
                            if len(end) != 1 or not batch["residue_mask"][row, positions[:int(end[0])]].all():
                                continue  # Report endpoint errors on wholly observed chains only.
                            truth_end = int(end[0])
                            predicted = pred[row, positions].eq(model.tokenizer.eos_token_id).nonzero().flatten()
                            v["chains"] += 1
                            if len(predicted):
                                found = int(predicted[0])
                                v["first_correct"] += int(found == truth_end)
                                v["length_error"] += abs(found-truth_end)
                                v["length_compared"] += 1
                            else:
                                v["missing_eos"] += 1
                    v["records"] = 1
                    totals[sources.index(samples[sample_index]["source_group"])] += torch.tensor(
                        [v[key] for key in FIELDS], device=device, dtype=torch.float64)
            if world > 1:
                torch.distributed.all_reduce(totals)
            values = totals.cpu().tolist()
            results[name] = dict(all=summarize(totals.sum(0).cpu().tolist()),
                                 sources={source: summarize(value) for source, value in zip(sources, values)})
        source_loss = {source: mean([results[name]["sources"][source]["loss"]
                                    for name in ("mask25", "mask50", "mask75", "mask100")]) for source in sources}
        endpoint = results["mask100"]["all"]
        summary = dict(loss=mean(list(source_loss.values())),
                       eos_loss=endpoint["eos_loss"], first_eos_accuracy=endpoint["first_eos_accuracy"],
                       early_eos_rate=endpoint["early_eos_rate"], missing_eos_rate=endpoint["missing_eos_rate"],
                       length_mae=endpoint["length_mae"],
                       relation_loss=results["mask50"]["all"]["relation_loss"],
                       edit_accuracy=results["edit15"]["all"]["accuracy"],
                       keep_error_rate=results["edit15"]["all"]["keep_error_rate"])
        cdr_results = {}
        if "cdr_tasks" in model.recipe:
            from scripts.research.diffusion_cdr_eval import evaluate_cdr
            cdr_results = evaluate_cdr(model, samples, device, rank, world, batch_size)
            summary.update(cdr_results["summary"])
        return dict(schema="immune.diffusion_edit.validation.v1", summary=summary,
                    source_losses=source_loss, modes=results, records=len(samples),
                    cdr_tasks=cdr_results,
                    scope="frozen internal V7 validation; not downstream performance",
                    endpoint_population="fully observed chains; first EOS over entire fixed canvas")
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = DiffusionEditModel.from_pretrained(args.checkpoint).to(args.device)
    result = evaluate(model, json.loads(args.index.read_text()), torch.device(args.device))
    write_json(args.output, result)


if __name__ == "__main__":
    main()
