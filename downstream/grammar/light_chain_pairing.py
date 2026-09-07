"""Light-chain pairing eval adapter for grammar BioSeq / LLaDA-fusion models.

Generate + evaluate (ImmunoMatch, diversity, ANARCI chain/V/J metrics):
  conda activate protenix_abtcr
  python -m downstream.grammar.light_chain_pairing \\
    --csv-path /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/comp_chain/test_data_oas_holdout.csv \\
    --checkpoint-path /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v1_esmc300m/latest.pt \\
    --output-csv /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/grammar_v1_esmc300m_light_pairing.csv \\
    --device cuda --num-seqs 8 --light-prompt-tokens 3

Target-length discipline (``--light-length-mode``)
--------------------------------------------------
The grammar-v2 record renders a chain as exactly ``len(sequence)`` residue slots
and there is no in-block terminator the model could emit, so the number of masked
slots *is* the generated length.  Building the record from the reference light
chain therefore leaks the target length.

``reference``
    Legacy behaviour: allocate ``len(reference_light)`` slots.  **Leaks the
    target length** -> generated light is a near-reconstruction of the reference
    (measured: 100% length match, ~0.95 identity).  Kept only for reproducing
    older diagnostics; never use it for a pairing capability claim.

``prior`` (default)
    Draw the light-chain length from a reference-independent prior histogram
    built from the OAS *training* split
    (``data/downstream/comp_chain/oas_train_light_length_prior.json``).  This is
    the grammar-side analogue of the official AirGen protocol
    (``AirGen-Dev/downstream/comp_chain/generate_light_from_csv.py``), which
    allocates a fixed 128-token light buffer, masks everything after the prompt
    and lets the model emit its own ``<eos>``.  Our model has no terminator, so
    the length comes from the marginal prior instead of from this row's
    reference.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DEFAULT_HOLDOUT_CSV = PROJECT_ROOT / "data" / "downstream" / "comp_chain" / "test_data_oas_holdout.csv"
DEFAULT_LENGTH_PRIOR_JSON = (
    PROJECT_ROOT / "data" / "downstream" / "comp_chain" / "oas_train_light_length_prior.json"
)
# Fully masked filler; only the slot count reaches the model, never this residue.
LENGTH_PRIOR_FILLER = "A"
# Bump whenever the generation protocol or decoding path changes semantically, so a
# stale `.progress.pt` from the previous protocol can never be resumed into a new run.
# v2 = light_length_mode + the ESMC encoder-stream leak fix (encoder now mirrors the
# whole generation mask instead of the shrinking corruption mask).
GENERATION_PROTOCOL_VERSION = 2
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.grammar.common import (
    antibody_pair_record,
    build_eval_collator,
    build_grammar_collator,
    build_grammar_tokenizer,
    collate_records,
    load_grammar_checkpoint,
    load_sample_oas_record,
    load_untrained_no_encoder,
    run_grammar_generate,
)
from downstream.grammar.masks import light_chain_generation_partial_mask
from downstream.grammar.metrics import extract_chain_sequence


class HeavyLightCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        heavy_col: str = "h_sequence",
        light_col: str = "l_sequence",
        start_index: int = 0,
        end_index: int | None = None,
    ):
        frame = pd.read_csv(csv_path)
        frame = frame[~frame[heavy_col].isna()].copy()
        frame["_input_row_idx"] = frame.index
        frame = frame.iloc[start_index:end_index].copy()
        self.heavy_col = heavy_col
        self.light_col = light_col
        self.heavy = frame[heavy_col].astype(str).str.replace("-", "").tolist()
        self.light = frame.get(light_col, "").fillna("").astype(str).str.replace("-", "").tolist()
        self.metadata = frame.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.heavy)

    def __getitem__(self, index: int):
        return self.heavy[index], self.light[index], self.metadata[index]


class LightLengthPrior:
    """Reference-independent light-chain length prior (OAS train histogram)."""

    def __init__(self, path: Path):
        payload = json.loads(Path(path).read_text())
        histogram = payload["histogram"]
        self.path = Path(path)
        self.n_rows = int(payload.get("n_rows", 0))
        self.lengths = np.array(sorted(int(k) for k in histogram), dtype=np.int64)
        weights = np.array([float(histogram[str(int(k))]) for k in self.lengths], dtype=np.float64)
        self.probs = weights / weights.sum()

    def sample(self, size: int) -> np.ndarray:
        return np.random.choice(self.lengths, size=size, p=self.probs)

    def signature(self) -> dict[str, Any]:
        return {
            "path": str(self.path.resolve()),
            "n_rows": self.n_rows,
            "n_bins": int(self.lengths.size),
            "min_length": int(self.lengths.min()),
            "max_length": int(self.lengths.max()),
        }


def build_placeholder_light(reference_light: str, target_length: int, prompt_residues: int) -> str:
    """Prompt residues from the reference, remaining slots filled and later masked."""

    prompt_residues = max(int(prompt_residues), 0)
    target_length = max(int(target_length), prompt_residues, 1)
    prompt = reference_light[:prompt_residues]
    return prompt + LENGTH_PRIOR_FILLER * (target_length - len(prompt))


def generation_csv_path(output_path: Path, num_seqs: int) -> Path:
    """Return the public, complete-only CSV path for one generation run."""

    if output_path.suffix:
        base, ext = output_path.stem, output_path.suffix
        return output_path.with_name(f"{base}_n{num_seqs}{ext}")
    return Path(f"{output_path}_n{num_seqs}.csv")


def save_generation_csv(rows: list[dict], output_path: Path, num_seqs: int) -> Path:
    """Atomically publish a complete generation CSV."""

    saved_path = generation_csv_path(output_path, num_seqs)
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = saved_path.with_suffix(saved_path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(saved_path)
    return saved_path


def _progress_path(saved_csv: Path) -> Path:
    return saved_csv.with_name(f"{saved_csv.stem}.progress.pt")


def _generation_signature(args, dataset_len: int) -> dict[str, Any]:
    """Fields that must be identical before a partial run can be resumed."""

    csv_path = Path(args.csv_path).resolve()
    csv_stat = csv_path.stat()
    checkpoint_path = Path(args.checkpoint_path).resolve() if args.checkpoint_path else None
    checkpoint_stat = checkpoint_path.stat() if checkpoint_path is not None else None
    return {
        "csv_path": str(csv_path),
        "csv_size": int(csv_stat.st_size),
        "csv_mtime_ns": int(csv_stat.st_mtime_ns),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
        "checkpoint_size": int(checkpoint_stat.st_size) if checkpoint_stat is not None else 0,
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns) if checkpoint_stat is not None else 0,
        "heavy_col": args.heavy_col,
        "light_col": args.light_col,
        "start_index": int(args.start_index),
        "max_samples": args.max_samples,
        "dataset_len": int(dataset_len),
        "heavy_batch_size": int(args.heavy_batch_size),
        "num_seqs": int(args.num_seqs),
        "max_iter": int(args.max_iter),
        "sampling_strategy": args.sampling_strategy,
        "temperature": float(args.temperature),
        "light_prompt_tokens": int(args.light_prompt_tokens),
        "light_length_mode": str(args.light_length_mode),
        "protocol_version": int(GENERATION_PROTOCOL_VERSION),
        "seed": args.seed,
    }


def _save_progress(
    progress_path: Path,
    *,
    signature: dict[str, Any],
    next_index: int,
    rows: list[dict],
) -> None:
    """Atomically checkpoint rows and all RNG streams after a completed batch."""

    payload: dict[str, Any] = {
        "version": 1,
        "signature": signature,
        "next_index": int(next_index),
        "rows": rows,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(progress_path)


def _load_progress(
    progress_path: Path,
    *,
    signature: dict[str, Any],
    dataset_len: int,
    num_seqs: int,
) -> tuple[int, list[dict]]:
    """Validate a progress checkpoint and restore its exact RNG continuation."""

    payload = torch.load(progress_path, map_location="cpu", weights_only=False)
    if payload.get("version") != 1 or payload.get("signature") != signature:
        raise RuntimeError(
            f"Refusing incompatible light-pairing progress checkpoint: {progress_path}"
        )
    next_index = int(payload["next_index"])
    rows = list(payload["rows"])
    if not (0 <= next_index <= dataset_len):
        raise RuntimeError(f"Invalid next_index={next_index} in {progress_path}")
    expected_rows = next_index * int(num_seqs)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Progress row mismatch in {progress_path}: got {len(rows)}, expected {expected_rows}"
        )
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_state = payload.get("torch_cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    return next_index, rows


def generate_for_batch(
    samples: list[tuple[str, str, dict]],
    *,
    model,
    collator,
    tokenizer,
    device: torch.device,
    num_seqs: int,
    max_iter: int,
    sampling_strategy: str,
    temperature: float,
    light_prompt_tokens: int,
    light_length_mode: str,
    length_prior: LightLengthPrior | None,
) -> list[dict]:
    records = []
    metadata_rows: list[tuple[str, str, int, dict, int]] = []

    if light_length_mode == "prior":
        if length_prior is None:
            raise ValueError("light_length_mode='prior' requires a length prior")
        drawn = length_prior.sample(len(samples) * num_seqs)
    else:
        drawn = None

    cursor = 0
    for heavy, light, metadata in samples:
        for variant_idx in range(num_seqs):
            if drawn is not None:
                target_length = int(drawn[cursor])
                slot_light = build_placeholder_light(light, target_length, light_prompt_tokens)
            else:
                target_length = len(light)
                slot_light = light
            cursor += 1
            records.append(antibody_pair_record(heavy, slot_light))
            metadata_rows.append((heavy, light, variant_idx, metadata, target_length))

    batch = collator(records)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    partial_mask = light_chain_generation_partial_mask(
        batch,
        tokenizer,
        prompt_residues=light_prompt_tokens,
    )
    output_tokens, _ = run_grammar_generate(
        model,
        batch,
        partial_mask=partial_mask,
        max_iter=max_iter,
        sampling_strategy=sampling_strategy,
        temperature=temperature,
    )

    rows: list[dict] = []
    for row_idx, (heavy, light, variant_idx, metadata, target_length) in enumerate(metadata_rows):
        generated_light = extract_chain_sequence(
            output_tokens[row_idx],
            batch["attention_mask"][row_idx],
            batch["residue_mask"][row_idx],
            tokenizer,
            chain="light",
        )
        result = {
            "h_sequence": heavy,
            "gen_l_sequence": generated_light,
            "raw_l_sequence": light,
            "variant_idx": variant_idx,
            # Audit trail for the length-leakage fix: where the slot count came from.
            "light_length_mode": light_length_mode,
            "target_light_length": int(target_length),
            "ref_light_length": len(light),
        }
        for key, value in metadata.items():
            if key not in result:
                result[key] = value
        rows.append(result)
    return rows


def run_comp_chain_eval(generated_csv: Path, num_seqs: int) -> dict:
    eval_dir = PROJECT_ROOT / "downstream" / "comp_chain" / "eval_scripts"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from generation_eval import main as generation_eval_main

    return generation_eval_main(
        generate_results_file=str(generated_csv),
        gen_col_name="gen_l_sequence",
        ref_col_name="raw_l_sequence",
        detailed_comparison=True,
        heavy_col_name="h_sequence",
        expected_count=num_seqs,
    )


def run_generation(args) -> Path:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    length_prior: LightLengthPrior | None = None
    if args.light_length_mode == "prior":
        length_prior = LightLengthPrior(Path(args.length_prior_json))
        print(
            f"Light-length prior: {length_prior.signature()} "
            f"(reference-independent; target length is NOT taken from the reference)",
            flush=True,
        )
    else:
        print(
            "WARNING: --light-length-mode=reference allocates len(reference_light) slots and "
            "therefore LEAKS the target length. Diagnostic only; not a pairing capability claim.",
            flush=True,
        )

    if args.checkpoint_path:
        model, tokenizer = load_grammar_checkpoint(args.checkpoint_path, device=device)
    else:
        tokenizer = build_grammar_tokenizer()
        model = load_untrained_no_encoder(tokenizer.vocab_size, mask_token_id=tokenizer.mask_token_id).to(device)

    collator = build_eval_collator(model, tokenizer)
    end_index = None if args.max_samples is None else args.start_index + args.max_samples
    dataset = HeavyLightCsvDataset(
        args.csv_path,
        heavy_col=args.heavy_col,
        light_col=args.light_col,
        start_index=args.start_index,
        end_index=end_index,
    )
    heavy_batch_size = max(1, args.heavy_batch_size)
    output_path = Path(args.output_csv)
    saved_path = generation_csv_path(output_path, args.num_seqs)
    progress_path = _progress_path(saved_path)
    signature = _generation_signature(args, len(dataset))
    resume_index = 0
    all_rows: list[dict] = []
    if progress_path.is_file():
        resume_index, all_rows = _load_progress(
            progress_path,
            signature=signature,
            dataset_len=len(dataset),
            num_seqs=args.num_seqs,
        )
        print(
            f"Resuming light-pairing generation at heavy index {resume_index}/{len(dataset)} "
            f"from {progress_path}",
            flush=True,
        )

    for start_idx in tqdm(
        range(resume_index, len(dataset), heavy_batch_size),
        desc="grammar-heavy2light",
    ):
        batch_samples = [
            dataset[seq_idx]
            for seq_idx in range(start_idx, min(start_idx + heavy_batch_size, len(dataset)))
        ]
        all_rows.extend(generate_for_batch(
            batch_samples,
            model=model,
            collator=collator,
            tokenizer=tokenizer,
            device=device,
            num_seqs=args.num_seqs,
            max_iter=args.max_iter,
            sampling_strategy=args.sampling_strategy,
            temperature=args.temperature,
            light_prompt_tokens=args.light_prompt_tokens,
            light_length_mode=args.light_length_mode,
            length_prior=length_prior,
        ))
        next_index = min(start_idx + heavy_batch_size, len(dataset))
        _save_progress(
            progress_path,
            signature=signature,
            next_index=next_index,
            rows=all_rows,
        )

    saved_path = save_generation_csv(all_rows, output_path, args.num_seqs)
    print(f"Saved generation CSV: {saved_path} ({len(all_rows)} rows)")
    return saved_path


def smoke_eval(device: str = "cpu") -> dict[str, float | list[str]]:
    tokenizer = build_grammar_tokenizer()
    record = load_sample_oas_record(split="valid", index=0)
    batch = collate_records([record])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    model = load_untrained_no_encoder(tokenizer.vocab_size, mask_token_id=tokenizer.mask_token_id).to(device)
    partial_mask = light_chain_generation_partial_mask(batch, tokenizer, prompt_residues=3)
    output_tokens, _ = run_grammar_generate(
        model,
        batch,
        partial_mask=partial_mask,
        max_iter=8,
        sampling_strategy="argmax",
        temperature=1.0,
    )
    generated_light = extract_chain_sequence(
        output_tokens[0],
        batch["attention_mask"][0],
        batch["residue_mask"][0],
        tokenizer,
        chain="light",
    )
    return {"generated_light_sequences": [generated_light]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grammar light-chain pairing generate + comp_chain eval.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_HOLDOUT_CSV))
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--heavy-col", type=str, default="h_sequence")
    parser.add_argument("--light-col", type=str, default="l_sequence")
    parser.add_argument("--heavy-batch-size", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=32)
    parser.add_argument("--sampling-strategy", type=str, default="gumbel_argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--light-prompt-tokens", type=int, default=3)
    parser.add_argument(
        "--light-length-mode",
        type=str,
        default="prior",
        choices=("prior", "reference"),
        help=(
            "prior: draw the light length from the OAS-train histogram (no target-length "
            "leakage, default). reference: allocate len(reference_light) slots (LEAKS length; "
            "diagnostic only)."
        ),
    )
    parser.add_argument("--length-prior-json", type=str, default=str(DEFAULT_LENGTH_PRIOR_JSON))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-json", type=str, default="")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        result = smoke_eval(device="cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
        print(f"sample light={result['generated_light_sequences'][0][:80]}")
        return

    if not args.output_csv:
        args.output_csv = str(
            PROJECT_ROOT / "output" / "downstream_generation" / "grammar_v1_esmc300m_light_pairing.csv"
        )

    saved_csv = run_generation(args)
    if args.skip_eval:
        _progress_path(saved_csv).unlink(missing_ok=True)
        return

    metrics = run_comp_chain_eval(saved_csv, args.num_seqs)
    metrics_path = Path(args.metrics_json) if args.metrics_json else saved_csv.with_name(saved_csv.stem + "_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _progress_path(saved_csv).unlink(missing_ok=True)
    print(f"Saved comp_chain metrics: {metrics_path}")


if __name__ == "__main__":
    main()
