"""CDR infilling eval adapter for grammar BioSeq / LLaDA-fusion models.

Run: python -m downstream.grammar.cdr_infill --smoke --device cpu
Persist evaluations with --output-dir <NEW_DIRECTORY>; rescore without inference:
python -m downstream.grammar.cdr_result_io --score-only <DIRECTORY>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.immune_llada.data import BioSeqChain, BioSeqRecord
from downstream.grammar.common import (
    antibody_pair_record,
    build_eval_collator,
    build_grammar_collator,
    build_grammar_tokenizer,
    labels_to_grammar,
    load_grammar_checkpoint,
    load_untrained_no_encoder,
    run_grammar_generate,
)
from downstream.grammar.masks import cdr_generation_partial_mask, cdr_generation_partial_mask_from_subsequence
from downstream.grammar.metrics import masked_token_accuracy
from downstream.grammar.cdr_result_io import CDRResultWriter, checkpoint_identity, score_artifacts
from downstream.benchmark.common.reporting import file_identity


def resolve_sabdab_test_json(file_path: str, mode: str, fold: int) -> str:
    candidates = [
        os.path.join(file_path, f"fold_{fold}", "test.json"),
        os.path.join(file_path, mode, f"fold_{fold}", "test.json"),
        os.path.join(file_path, mode, "test.json"),
        os.path.join(file_path, "test.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"CDR test.json not found for mode={mode!r} fold={fold}; tried {candidates}"
    )


class SAbDabDataset(Dataset):
    def __init__(self, file_path: str, mode: str, fold: int = 0):
        self.mode = mode
        data = pd.read_json(resolve_sabdab_test_json(file_path, mode, fold), lines=True)
        self.heavy = data["heavy_chain_seq"].astype(str).str.replace("J", "L").tolist()
        self.light = data["light_chain_seq"].astype(str).str.replace("J", "L").tolist()
        self.target = data[f"{mode}_seq"].tolist()
        self.pos = data[f"{mode}_pos"].tolist()

    def __len__(self) -> int:
        return len(self.heavy)

    def __getitem__(self, index: int):
        return self.heavy[index], self.light[index], self.target[index], self.pos[index], self.mode


def build_antibody_record(heavy: str, light: str) -> BioSeqRecord:
    """Record for one antibody, tolerating a missing light chain.

    MEAN's `filter 111` keeps a complex when the light *chain id* is present, which is
    not the same as the light *sequence* being non-empty: the Kong-vintage SAbDab split
    has one such entry (4kv5, light_chain='I', light_chain_seq=''). The official
    `zeroshot_cdr.py` absorbs it silently because it tokenizes the empty string into a
    fixed 128-slot eos-padded buffer, whereas `BioSeqChain` rejects an empty sequence.
    Emitting a heavy-only record keeps that complex in the denominator, matching the
    official AAR, instead of dropping a row and changing the metric.
    """

    heavy = str(heavy).strip()
    light = str(light).strip()
    if not light:
        return BioSeqRecord(
            chains=[BioSeqChain(heavy, "antibody_heavy")],
            task_type="antibody",
            source="oas_paired",
        )
    return antibody_pair_record(heavy, light)


def _chain_role_for_mode(mode: str) -> tuple[str, str]:
    normalized = mode.lower()
    if normalized.startswith("cdrl"):
        return "light", normalized.upper().replace("CDRL", "CDR")
    return "heavy", normalized.upper().replace("CDRH", "CDR")


def _build_cdr_partial_mask(batch, tokenizer, record, chain_role: str, cdr_name: str, target_subsequence: str | None):
    chain_index = 0 if chain_role == "heavy" else 1
    if chain_index >= len(record.chains):
        raise ValueError(
            f"Cannot infill {cdr_name} on the {chain_role} chain: this record has only "
            f"{len(record.chains)} chain(s) (light sequence was empty)."
        )
    chain = record.chains[chain_index]
    if chain.regions.get(cdr_name.upper()) or chain.regions.get(cdr_name):
        return cdr_generation_partial_mask(batch, tokenizer, chain, chain_role, cdr_name)  # type: ignore[arg-type]
    if target_subsequence:
        return cdr_generation_partial_mask_from_subsequence(
            batch,
            tokenizer,
            chain,
            chain_role,  # type: ignore[arg-type]
            target_subsequence,
        )
    raise ValueError(f"No {cdr_name} regions or target subsequence for chain {chain.role}")


def evaluate_record(
    model,
    heavy: str,
    light: str,
    mode: str,
    *,
    max_iter: int,
    sampling_strategy: str,
    temperature: float,
    device: str,
    target_subsequence: str | None = None,
) -> float:
    tokenizer = build_grammar_tokenizer()
    collator = build_eval_collator(model, tokenizer)
    chain_role, cdr_name = _chain_role_for_mode(mode)
    record = antibody_pair_record(heavy, light)
    batch = collator([record])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    partial_mask = _build_cdr_partial_mask(batch, tokenizer, record, chain_role, cdr_name, target_subsequence)
    labels = labels_to_grammar(model, batch["labels"])
    generation_mask = partial_mask.logical_not() & batch["attention_mask"] & batch["residue_mask"]
    output_tokens, _ = run_grammar_generate(
        model,
        batch,
        partial_mask=partial_mask,
        max_iter=max_iter,
        sampling_strategy=sampling_strategy,
        temperature=temperature,
    )
    aar = masked_token_accuracy(output_tokens, labels, generation_mask)
    return float(aar.item())


def smoke_eval(device: str = "cpu") -> float:
    tokenizer = build_grammar_tokenizer()
    heavy = BioSeqChain(
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGWGPGTLVTVSS",
        "antibody_heavy",
        regions={
            "FR1": "EVQLVESGGGLVQPGGSLRLSCAAS",
            "CDR1": "GFTFSSYA",
            "FR2": "MSWVRQAPGKGLEWVSA",
            "CDR2": "ISGSGGST",
            "FR3": "YYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
            "CDR3": "ARDRGWGPGTLVTVSS",
            "FR4": "",
        },
    )
    light = BioSeqChain("DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK", "antibody_light")
    record = BioSeqRecord(chains=[heavy, light], task_type="antibody", source="unit")
    model = load_untrained_no_encoder(tokenizer.vocab_size, mask_token_id=tokenizer.mask_token_id).to(device)
    batch = build_grammar_collator(tokenizer)([record])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    partial_mask = cdr_generation_partial_mask(batch, tokenizer, heavy, "heavy", "CDR3")
    labels = labels_to_grammar(model, batch["labels"])
    generation_mask = partial_mask.logical_not() & batch["attention_mask"] & batch["residue_mask"]
    output_tokens, _ = run_grammar_generate(
        model,
        batch,
        partial_mask=partial_mask,
        max_iter=8,
        sampling_strategy="argmax",
        temperature=1.0,
    )
    aar = masked_token_accuracy(output_tokens, labels, generation_mask)
    return float(aar.item())


def _load_eval_model(args, device: torch.device):
    if args.checkpoint_path:
        model, tokenizer = load_grammar_checkpoint(args.checkpoint_path, device=device)
        return model, tokenizer
    tokenizer = build_grammar_tokenizer()
    model = load_untrained_no_encoder(tokenizer.vocab_size, mask_token_id=tokenizer.mask_token_id).to(device)
    return model, tokenizer


def evaluate(args) -> None:
    if args.num_folds <= 0 or (args.max_samples is not None and args.max_samples <= 0):
        raise ValueError("num-folds and max-samples must be positive")
    if not args.checkpoint_path:
        raise ValueError("Dataset evaluation requires a checkpoint; use --smoke for an untrained model")
    if getattr(args, "output_dir", None) and Path(args.output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    fold_jsons: list[str] = []
    for fold in range(args.num_folds):
        path = str(Path(resolve_sabdab_test_json(args.test_set, args.mode, fold)).resolve())
        if path in fold_jsons:
            break
        fold_jsons.append(path)

    datasets = [SAbDabDataset(args.test_set, args.mode, fold=fold) for fold in range(len(fold_jsons))]
    if not datasets or any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("CDR evaluation contains an empty fold")
    device = torch.device(args.device)
    model, tokenizer = _load_eval_model(args, device)
    collator = build_eval_collator(model, tokenizer)
    writer = None
    if getattr(args, "output_dir", None):
        code_paths = [
            "downstream/grammar/cdr_infill.py", "downstream/grammar/cdr_result_io.py",
            "downstream/grammar/common.py", "downstream/grammar/masks.py", "downstream/grammar/metrics.py",
            "downstream/benchmark/common/reporting.py",
            "dllm/pipelines/immune_llada/data/grammar.py", "dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py",
            "examples/llada/load_fusion_checkpoint.py", "examples/llada/protein_fusion_model.py",
        ]
        identity = {
            "checkpoint": checkpoint_identity(args.checkpoint_path),
            "datasets": [{**file_identity(path), "fold": fold, "n_rows": len(datasets[fold])}
                         for fold, path in enumerate(fold_jsons)],
            "tokenizer": {"class": type(tokenizer).__qualname__,
                          "tokens": [tokenizer.token(i) for i in range(tokenizer.vocab_size)],
                          "mask_token_id": tokenizer.mask_token_id, "pad_token_id": tokenizer.pad_token_id},
            "code": [file_identity(PROJECT_ROOT / path) for path in code_paths],
            "parameters": {"mode": args.mode, "num_folds_requested": args.num_folds,
                           "resolved_folds": len(fold_jsons), "max_samples": args.max_samples,
                           "max_iter": args.max_iter, "sampling_strategy": args.sampling_strategy,
                           "temperature": args.temperature, "device": args.device,
                           "torch_initial_seed": torch.initial_seed(),
                           "torch_rng_state_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                           "torch_version": torch.__version__, "normalization": "heavy/light J->L (legacy unchanged)"},
        }
        writer = CDRResultWriter(args.output_dir, identity)

    outer_aars = []
    all_aars = []
    for fold, _json_path in enumerate(fold_jsons):
        dataset = datasets[fold]
        inner_aars = []
        for sample_idx, (heavy, light, target, _pos, mode) in enumerate(DataLoader(dataset, batch_size=1)):
            if args.max_samples is not None and sample_idx >= args.max_samples:
                break
            heavy_str = heavy[0]
            light_str = light[0]
            target_str = target[0]
            record = build_antibody_record(heavy_str, light_str)
            chain_role, cdr_name = _chain_role_for_mode(mode[0])
            batch = collator([record])
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            partial_mask = _build_cdr_partial_mask(
                batch,
                tokenizer,
                record,
                chain_role,
                cdr_name,
                target_str,
            )
            labels = labels_to_grammar(model, batch["labels"])
            generation_mask = partial_mask.logical_not() & batch["attention_mask"] & batch["residue_mask"]
            if not bool(generation_mask.any()):
                raise ValueError(f"Empty CDR target at fold={fold} row={sample_idx}")
            output_tokens, _ = run_grammar_generate(
                model,
                batch,
                partial_mask=partial_mask,
                max_iter=args.max_iter,
                sampling_strategy=args.sampling_strategy,
                temperature=args.temperature,
            )
            aar = masked_token_accuracy(output_tokens, labels, generation_mask)
            if writer is not None:
                writer.append(fold=fold, row=sample_idx,
                              reference_ids=labels[generation_mask].tolist(),
                              prediction_ids=output_tokens[generation_mask].tolist(),
                              masked_positions=generation_mask[0].nonzero().flatten().tolist(),
                              aar=float(aar.item()))
            inner_aars.append(float(aar.item()))
        outer_aars.append(float(np.mean(inner_aars) * 100.0) if inner_aars else 0.0)
        all_aars.extend(inner_aars)

    if writer is not None:
        result = writer.finish(allow_partial=args.max_samples is not None)
        print("Artifact status:", result["status"], "directory:", str(writer.directory))
    print("Average AAR:", float(np.mean(all_aars) * 100.0) if all_aars else 0.0)
    print("Average AAR all folds:", float(np.mean(outer_aars)) if outer_aars else 0.0)
    print("AAR Standard deviation across all folds:", float(np.std(outer_aars)) if outer_aars else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=str, default="")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--mode", type=str, default="cdrh3")
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-strategy", type=str, default="argmax")
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="New artifact directory (never overwritten)")
    parser.add_argument("--score-only", type=Path, help="Rescore saved artifacts without inference")
    parser.add_argument("--allow-partial", action="store_true", help="Allow diagnostic scoring of incomplete artifacts")
    args = parser.parse_args()

    if args.score_only:
        print(json.dumps(score_artifacts(args.score_only, allow_partial=args.allow_partial), indent=2))
        return
    if args.smoke or not args.test_set:
        print(f"smoke AAR={smoke_eval(device=args.device):.4f}")
        return
    evaluate(args)


if __name__ == "__main__":
    main()
