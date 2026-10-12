"""TCR CDR infilling eval — antibody-style AAR on paired α/β.

Given a full-length OTS pair, mask one IMGT CDR (α1/α2/α3 or β1/β2/β3),
argmax-fill, report residue recovery. Layout is ``tcr_pair`` (no epitope),
the same grammar training uses for OTS.

  python -m downstream.grammar.tcr_cdr_infill \
    --test-set data/downstream/cdr_infilling/tcr \
    --mode cdr_beta3 --checkpoint-path <ckpt> \
    --device cuda --num-folds 10 \
    --sampling-strategy argmax --max-iter 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.immune_llada.data.records import normalize_sequence
from downstream.grammar.cdr_infill import resolve_sabdab_test_json
from downstream.grammar.common import (
    build_eval_collator,
    load_grammar_checkpoint,
    run_grammar_generate,
    tcr_pair_record,
)
from downstream.grammar.masks import cdr3b_span_partial_mask
from downstream.grammar.metrics import masked_token_accuracy

MODES = (
    "cdr_beta1",
    "cdr_beta2",
    "cdr_beta3",
    "cdr_alpha1",
    "cdr_alpha2",
    "cdr_alpha3",
)
_MODE_CHAIN = {
    "cdr_beta1": ("beta", 1, "CDR1"),
    "cdr_beta2": ("beta", 1, "CDR2"),
    "cdr_beta3": ("beta", 1, "CDR3"),
    "cdr_alpha1": ("alpha", 0, "CDR1"),
    "cdr_alpha2": ("alpha", 0, "CDR2"),
    "cdr_alpha3": ("alpha", 0, "CDR3"),
}
_DEFAULT_TRAIN = PROJECT_ROOT / "data" / "ots_paired_clean" / "final" / "train.csv"


class TcrCdrDataset(Dataset):
    def __init__(self, file_path: str, mode: str, fold: int = 0):
        if mode not in _MODE_CHAIN:
            raise ValueError(f"unknown TCR CDR mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        path = resolve_sabdab_test_json(file_path, mode, fold)
        data = pd.read_json(path, lines=True)
        self.alpha = data["alpha_chain_seq"].astype(str).map(normalize_sequence).tolist()
        self.beta = data["beta_chain_seq"].astype(str).map(normalize_sequence).tolist()
        self.target = data[f"{mode}_seq"].astype(str).map(normalize_sequence).tolist()
        self.pos = data[f"{mode}_pos"].tolist()

    def __len__(self) -> int:
        return len(self.alpha)

    def __getitem__(self, index: int):
        start, end = self.pos[index]
        return {
            "alpha": self.alpha[index],
            "beta": self.beta[index],
            "target": self.target[index],
            "start": int(start),
            "end_inclusive": int(end),
        }


def _collate_eval(rows: list[dict]) -> dict:
    return {
        "alpha": [row["alpha"] for row in rows],
        "beta": [row["beta"] for row in rows],
        "target": [row["target"] for row in rows],
        "start": [row["start"] for row in rows],
        "end_inclusive": [row["end_inclusive"] for row in rows],
    }


def _is_beta_row(row: dict[str, str], prefix: str) -> bool:
    anarci = str(row.get(f"{prefix}_anarci_type", "")).strip().upper()
    chain_type = str(row.get(f"{prefix}_type", "")).strip().lower()
    return anarci == "B" or chain_type in {"beta", "trb"}


def build_pwm(train_csv: Path, mode: str) -> dict[tuple[int, int], str]:
    """(length, position) → most frequent residue for this CDR, from OTS train."""

    _role, _chain_index, cdr_name = _MODE_CHAIN[mode]
    want_beta = "beta" in mode
    col = f"CDR{cdr_name[-1]}"
    counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with train_csv.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for prefix in ("chain1", "chain2"):
                is_beta = _is_beta_row(row, prefix)
                if is_beta != want_beta:
                    continue
                seq = normalize_sequence(row.get(f"{prefix}_{col}"))
                if not seq:
                    continue
                for i, aa in enumerate(seq):
                    counts[(len(seq), i)][aa] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in counts.items()}


def pwm_accuracy(targets: list[str], pwm: dict[tuple[int, int], str]) -> float:
    hit = tot = 0
    for seq in targets:
        for i, aa in enumerate(seq):
            tot += 1
            if pwm.get((len(seq), i)) == aa:
                hit += 1
    return float(hit / tot) if tot else float("nan")


def evaluate(args) -> dict:
    device = torch.device(args.device)
    model, tokenizer = load_grammar_checkpoint(args.checkpoint_path, device=device)
    collator = build_eval_collator(model, tokenizer)
    _role, chain_index, _cdr_name = _MODE_CHAIN[args.mode]

    fold_jsons: list[str] = []
    for fold in range(args.num_folds):
        path = resolve_sabdab_test_json(args.test_set, args.mode, fold)
        if path in fold_jsons:
            break
        fold_jsons.append(path)

    pwm = None
    if args.pwm_train:
        print(f"building PWM from {args.pwm_train} for {args.mode} ...", flush=True)
        pwm = build_pwm(Path(args.pwm_train), args.mode)
        print(f"  PWM cells {len(pwm):,}", flush=True)

    fold_aars: list[float] = []
    fold_pwms: list[float] = []
    all_aars: list[float] = []
    n_scored = 0
    n_skipped = 0

    for fold, _json_path in enumerate(fold_jsons):
        dataset = TcrCdrDataset(args.test_set, args.mode, fold=fold)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate_eval,
        )
        inner: list[float] = []
        inner_targets: list[str] = []
        seen = 0
        for batch_rows in loader:
            if args.max_samples is not None and seen >= args.max_samples:
                break
            take = len(batch_rows["alpha"])
            if args.max_samples is not None:
                take = min(take, args.max_samples - seen)
            alphas = batch_rows["alpha"][:take]
            betas = batch_rows["beta"][:take]
            targets = batch_rows["target"][:take]
            starts = batch_rows["start"][:take]
            ends = batch_rows["end_inclusive"][:take]
            seen += take

            records = []
            spans: list[tuple[int, int]] = []
            keep_targets: list[str] = []
            for alpha, beta, target, start, end in zip(alphas, betas, targets, starts, ends):
                chain_seq = beta if chain_index == 1 else alpha
                if chain_seq[start : end + 1] != target:
                    n_skipped += 1
                    continue
                records.append(tcr_pair_record(alpha, beta))
                spans.append((start, end + 1))
                keep_targets.append(target)
            if not records:
                continue

            batch = collator(records)
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            try:
                partial = cdr3b_span_partial_mask(batch, chain_index=chain_index, span=spans)
            except ValueError as exc:
                n_skipped += len(records)
                print(f"  skip batch fold={fold}: {exc}", flush=True)
                continue
            labels = batch["labels"]
            generation_mask = partial.logical_not() & batch["attention_mask"] & batch["residue_mask"]
            output_tokens, _ = run_grammar_generate(
                model,
                batch,
                partial_mask=partial,
                max_iter=args.max_iter,
                sampling_strategy=args.sampling_strategy,
                temperature=args.temperature,
            )
            aars = masked_token_accuracy(output_tokens, labels, generation_mask)
            inner.extend(float(x) for x in aars.detach().cpu().tolist())
            inner_targets.extend(keep_targets)
            n_scored += len(records)

        fold_mean = float(np.mean(inner) * 100.0) if inner else 0.0
        fold_aars.append(fold_mean)
        all_aars.extend(inner)
        if pwm is not None and inner_targets:
            fold_pwms.append(pwm_accuracy(inner_targets, pwm) * 100.0)
        print(
            f"fold {fold}: n={len(inner)} AAR={fold_mean:.4f}"
            + (f" PWM={fold_pwms[-1]:.4f}" if fold_pwms else ""),
            flush=True,
        )

    micro = float(np.mean(all_aars) * 100.0) if all_aars else 0.0
    macro = float(np.mean(fold_aars)) if fold_aars else 0.0
    std = float(np.std(fold_aars)) if fold_aars else 0.0
    print("Average AAR:", micro)
    print("Average AAR all folds:", macro)
    print("AAR Standard deviation across all folds:", std)
    result = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint_path),
        "n_scored": n_scored,
        "n_skipped": n_skipped,
        "average_aar": micro,
        "average_aar_all_folds": macro,
        "aar_std_all_folds": std,
        "fold_aars": fold_aars,
        "max_iter": args.max_iter,
        "sampling_strategy": args.sampling_strategy,
    }
    if fold_pwms:
        pwm_macro = float(np.mean(fold_pwms))
        print("PWM AAR all folds:", pwm_macro)
        result["pwm_aar_all_folds"] = pwm_macro
        result["fold_pwm_aars"] = fold_pwms
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="cdr_beta3", choices=list(MODES))
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-strategy", type=str, default="argmax")
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--pwm-train", type=str, default=str(_DEFAULT_TRAIN))
    parser.add_argument("--no-pwm", action="store_true")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()
    if args.no_pwm:
        args.pwm_train = ""
    evaluate(args)


if __name__ == "__main__":
    main()
