"""Fixed-noise MLM validation and clean, frozen-backbone development probes.

Run after activating pllm: python -B -m scripts.research.bert_mlm_eval --help
Use the source snapshot bound to the checkpoint for subsequent evaluation.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dllm.pipelines.immune_llada.fixed_chain.mlm import BertMLMCollator, BertMLMModel, corrupt
from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.bert_mlm_objectives import components, for_model


def read_samples(reference):
    if sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Frozen validation data changed")
    samples = [json.loads(line) for line in Path(reference["path"]).read_text().splitlines()]
    if len(samples) != reference["records"]:
        raise ValueError("Validation count changed")
    return samples


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def fixed_corruption(batch, tokenizer, sample_indices, seed=41016, corrupt_fn=corrupt):
    """Per-record noise remains identical across batch sizes and world sizes."""
    result = {key: value.clone() for key, value in batch.items()}
    result["noise_kind"] = torch.zeros_like(batch["input_ids"])
    result["labels"].fill_(-100)
    for row, sample in enumerate(sample_indices):
        length = int(batch["attention_mask"][row].sum())
        item = {key: value[row:row+1, :length] for key, value in batch.items()}
        noisy = corrupt_fn(item, tokenizer, torch.Generator().manual_seed(seed+sample))
        for key in ("input_ids", "labels", "noise_kind"):
            result[key][row, :length] = noisy[key][0]
    return result


@torch.no_grad()
def token_validation(model, index, device, rank=0, world=1, batch_size=8):
    collator_cls, _, corrupt_fn, _ = for_model(model)
    samples = read_samples(index["files"]["token_valid"])
    sources = sorted({x["source_group"] for x in samples})
    groups = ("receptor_aa", "context_aa", "eos", "relation", "type", "separator", "null_context")
    totals = torch.zeros((len(sources)+1, 17+3*len(groups)), dtype=torch.float64, device=device)
    collator = collator_cls(model.config)
    was_training = model.training
    model.eval()
    selected_indices = list(range(rank, len(samples), world))
    for start in range(0, len(selected_indices), batch_size):
        indices = selected_indices[start:start+batch_size]
        clean = collator([samples[i]["record"] for i in indices])
        noisy = fixed_corruption(clean, model.tokenizer, indices, corrupt_fn=corrupt_fn)
        batch = {k: v.to(device) for k, v in noisy.items()}
        with autocast(device):
            logits, _ = model.denoise(**model.feature_args(batch))
        pred = logits.argmax(-1)
        truth = clean["input_ids"].to(device)
        probability = logits.float().softmax(-1).amax(-1)
        for row, sample_id in enumerate(indices):
            values = torch.zeros(17+3*len(groups), dtype=torch.float64, device=device)
            for kind in (1, 2, 3):
                mask = batch["noise_kind"][row].eq(kind)
                column = (kind-1)*3
                if mask.any():
                    values[column] = F.cross_entropy(logits[row, mask].float(), truth[row, mask], reduction="sum")
                    values[column+1] = mask.sum()
                    values[column+2] = pred[row, mask].eq(truth[row, mask]).sum()
            correct_input = batch["eligible_mask"][row] & (batch["noise_kind"][row].eq(0) | batch["noise_kind"][row].eq(3))
            edited = correct_input & pred[row].ne(truth[row]) & (probability[row] >= .9)
            values[9] = correct_input.sum()
            values[10] = edited.sum()
            values[11] = batch["eligible_mask"][row].sum()
            values[12] = batch["noise_kind"][row].ne(0).sum()
            values[13] = 1
            editable = batch["eligible_mask"][row] & ~batch["noise_kind"][row].eq(1)
            edits = editable & pred[row].ne(batch["input_ids"][row]) & (probability[row] >= .9)
            values[14] = editable.sum()
            values[15] = edits.sum()
            values[16] = (edits & batch["noise_kind"][row].eq(2) & pred[row].eq(truth[row])).sum()
            type_mask = torch.zeros_like(truth[row], dtype=torch.bool)
            for token in ("<ab>", "<tcr>", "<nb>", "<pep>"):
                type_mask |= truth[row].eq(model.tokenizer.special_id(token))
            masks = (batch["residue_mask"][row] & batch["target_slot_mask"][row],
                     batch["residue_mask"][row] & ~batch["target_slot_mask"][row],
                     truth[row].eq(model.tokenizer.eos_token_id),
                     batch.get("relation_mask", torch.zeros_like(truth, dtype=torch.bool))[row],
                     type_mask, truth[row].eq(model.tokenizer.chain_separator_id()),
                     truth[row].eq(model.tokenizer.special_id("<null>")))
            for number, population in enumerate(masks):
                mask = population & batch["labels"][row].ne(-100)
                if mask.any():
                    column = 17+number*3
                    values[column] = F.cross_entropy(logits[row, mask].float(), truth[row, mask], reduction="sum")
                    values[column+1] = mask.sum()
                    values[column+2] = pred[row, mask].eq(truth[row, mask]).sum()
            totals[0] += values
            totals[sources.index(samples[sample_id]["source_group"])+1] += values
    if world > 1:
        torch.distributed.all_reduce(totals)
    model.train(was_training)
    result = {}
    for name, values in zip(["all", *sources], totals.cpu().tolist()):
        count = sum(values[i] for i in (1, 4, 7))
        metrics = {"loss": sum(values[i] for i in (0, 3, 6))/max(count, 1),
                   "selected_tokens": int(count), "records": int(values[13]),
                   "selected_fraction": values[12]/max(values[11], 1),
                   "false_edit_rate_at_0.9": values[10]/max(values[9], 1),
                   "false_edits_at_0.9": int(values[10]), "correct_input_tokens": int(values[9]),
                   "edit_coverage_at_0.9": values[15]/max(values[14], 1),
                   "random_repair_recall_at_0.9": values[16]/max(values[4], 1)}
        for i, kind in enumerate(("mask", "random_other", "unchanged")):
            metrics[kind+"_accuracy"] = values[3*i+2]/max(values[3*i+1], 1)
            metrics[kind+"_tokens"] = int(values[3*i+1])
        for number, group in enumerate(groups):
            column = 17+number*3
            metrics[group+"_loss"] = values[column]/max(values[column+1], 1)
            metrics[group+"_tokens"] = int(values[column+1])
            metrics[group+"_accuracy"] = values[column+2]/max(values[column+1], 1)
        result[name] = metrics
    return result


def fit_relation_probes(vectors, samples):
    """Fit separate fixed-C heads by source; validation never fits a head."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    result = {}
    sources = sorted({x["source_group"] for x in samples["valid"]})
    for source in sources:
        indices = {split: [i for i, s in enumerate(samples[split]) if s["source_group"] == source]
                   for split in ("train", "valid")}
        labels = {split: np.array([samples[split][i]["record"]["labels"]["relation"] == "binding"
                                  for i in indices[split]], dtype=np.int64) for split in indices}
        if any(len(set(y)) != 2 for y in labels.values()):
            raise ValueError("Probe needs both classes in each partition")
        x = {s: vectors[s][indices[s]] for s in indices}
        def head(y):
            estimator = make_pipeline(StandardScaler(), LogisticRegression(
                C=.1, class_weight="balanced", max_iter=2000, random_state=42))
            estimator.fit(x["train"], y)
            return estimator.predict_proba(x["valid"])[:, 1]
        probabilities = head(labels["train"])
        null = head(np.random.default_rng(918).permutation(labels["train"]))
        result[source] = dict(auroc=float(roc_auc_score(labels["valid"], probabilities)),
                              auprc=float(average_precision_score(labels["valid"], probabilities)),
                              shuffled_auroc=float(roc_auc_score(labels["valid"], null)),
                              train_records=len(labels["train"]), valid_records=len(labels["valid"]))
    return dict(sources=result, macro_auroc=float(np.mean([x["auroc"] for x in result.values()])),
                scope="V5 relation development diagnostic, not final downstream benchmark", C=.1)


@torch.no_grad()
def representation_validation(model, index, device, rank=0, world=1, batch_size=16):
    samples = {s: read_samples(index["files"]["probe_"+s]) for s in ("train", "valid")}
    vectors = {}
    collator_cls, _, _, _ = for_model(model)
    from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator
    collator = collator_cls(model.config, **({"hide_relation": True}
                           if issubclass(collator_cls, FullTokenMLMCollator) else {}))
    was_training = model.training
    model.eval()
    for split, rows in samples.items():
        indices, local = list(range(rank, len(rows), world)), []
        for start in range(0, len(indices), batch_size):
            part = indices[start:start+batch_size]
            batch = {k: v.to(device) for k, v in collator([rows[i]["record"] for i in part]).items()}
            with autocast(device):
                values = model.receptor_vectors(batch).float().cpu().numpy()
            local.extend(zip(part, values))
        packets = [None]*world
        if world > 1:
            torch.distributed.all_gather_object(packets, local)
        else:
            packets[0] = local
        if rank == 0:
            ordered = sorted([item for packet in packets for item in packet], key=lambda x: x[0])
            if [i for i, _ in ordered] != list(range(len(rows))):
                raise ValueError("Probe rank coverage differs")
            vectors[split] = np.stack([value for _, value in ordered])
    result = [fit_relation_probes(vectors, samples) if rank == 0 else None]
    if world > 1:
        torch.distributed.broadcast_object_list(result, src=0)
    model.train(was_training)
    return result[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    provenance = json.loads((args.checkpoint / "run_provenance.json").read_text())
    source = Path(provenance["source_root"])
    if Path(__file__).resolve() != source / "scripts/research/bert_mlm_eval.py":
        raise ValueError("Run the evaluation script from this checkpoint's frozen source root")
    for path, expected in provenance["sources"].items():
        if sha(path) != expected:
            raise ValueError("Frozen training/evaluation code changed")
    if sha(provenance["data_index"]) != provenance["data_index_sha256"]:
        raise ValueError("Checkpoint data binding changed")
    index = json.loads(Path(provenance["data_index"]).read_text())
    device = torch.device(args.device)
    _, model_cls, _, _ = components(provenance.get("objective", "receptor_aa"))
    model = model_cls.from_pretrained(args.checkpoint).to(device).eval()
    write_json(args.output, dict(token=token_validation(model, index, device),
                                representation=representation_validation(model, index, device)))
