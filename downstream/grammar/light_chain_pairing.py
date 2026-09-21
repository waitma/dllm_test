"""Light-chain pairing eval adapter for grammar BioSeq / LLaDA-fusion models.

Generate + evaluate (ImmunoMatch, diversity, ANARCI chain/V/J metrics):
  conda activate pllm
  python -m downstream.grammar.light_chain_pairing \\
    --csv-path /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/comp_chain/test_data_oas_holdout.csv \\
    --checkpoint-path /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune/checkpoint-42000 \\
    --output-csv /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/pairing_reflen_v3.csv \\
    --device cuda --num-seqs 8 --light-prompt-tokens 3

Target-length discipline (``--light-length-mode``)
--------------------------------------------------
The grammar record preallocates light residue slots and fixes a trailing
``<protd>``. The parser may truncate earlier if a generated slot emits another
``<protd>``; this adapter does not implement unconstrained length generation.

``auto`` (default)
    Selects ``fixed_v2`` for fixed-canvas checkpoints and ``reference`` for
    legacy checkpoints. The resolved mode is recorded in generation metadata.

``prior`` (optional diagnostic)
    Draw the light-chain length from a reference-independent prior histogram
    built from the OAS *training* split
    (``data/downstream/comp_chain/oas_train_light_length_prior.json``).  This is
    the grammar-side analogue of the official AirGen protocol
    (``AirGen-Dev/downstream/comp_chain/generate_light_from_csv.py``), which
    allocates a fixed 128-token light buffer, masks everything after the prompt
    and lets the model emit its own ``<eos>``. Here the allocated slot budget
    comes from the marginal prior instead of from this row's
    reference. This is a different length condition, not an exact reproduction
    of native EOS generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
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
# v3 = committed-state sampler fix + explicit reference-length protocol.
# v4 = prompt0/prompt3 + explicit residue-condition CFG (structure preserved).
# v5 = feed committed generated residues back into ESMC with vocabulary remapping;
# pending positions stay masked in both streams, never restored from the reference.
GENERATION_PROTOCOL_VERSION = 5
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.grammar.common import (
    antibody_pair_record,
    build_eval_collator,

    build_grammar_tokenizer,
    collate_records,
    collator_fixed_receptor_lengths,
    load_grammar_checkpoint,
    load_sample_oas_record,
    load_untrained_no_encoder,
    run_grammar_generate,
)
from downstream.grammar.masks import light_chain_generation_partial_mask
from downstream.grammar.metrics import extract_chain_sequence


def _model_uses_fixed_v2(model: Any) -> bool:
    """Return the checkpoint contract bit without assuming a model class."""

    config = getattr(model, "config", None)
    marker = getattr(config, "fixed_receptor_lengths", None)
    if marker is not None:
        return bool(marker)
    marker = getattr(model, "fixed_receptor_lengths", None)
    if marker is not None:
        return bool(marker)
    # Fusion checkpoints expose the same contract through their eval collator;
    # this fallback keeps older composite model wrappers usable while the model
    # config remains the canonical source for newly loaded checkpoints.
    collator = getattr(model, "_fusion_eval_collator", None)
    marker = collator_fixed_receptor_lengths(collator)
    if marker is not None:
        return marker
    return bool(getattr(model, "_predict_eos", False))


def resolve_light_length_mode(model: Any, requested: str) -> str:
    """Resolve and validate the pairing canvas contract for a checkpoint.

    ``auto`` is intentionally the CLI default: fixed-canvas v2 checkpoints use
    their 134-residue light canvas, while legacy checkpoints retain the
    reference-length protocol. Explicitly selecting the other contract is an
    error rather than a silent change in conditioning semantics.
    """

    requested = str(requested).lower()
    if requested not in {"auto", "fixed_v2", "reference", "prior"}:
        raise ValueError(f"unknown light_length_mode={requested!r}")
    fixed_v2 = _model_uses_fixed_v2(model)
    if requested == "auto":
        return "fixed_v2" if fixed_v2 else "reference"
    if fixed_v2 and requested != "fixed_v2":
        raise ValueError(
            f"light_length_mode={requested!r} is incompatible with a fixed_receptor_lengths v2 checkpoint; "
            "use fixed_v2 or auto"
        )
    if not fixed_v2 and requested == "fixed_v2":
        raise ValueError(
            "light_length_mode='fixed_v2' requires a checkpoint with "
            "model.config.fixed_receptor_lengths=True; use reference or auto for legacy checkpoints"
        )
    return requested


def _validate_pairing_collator(collator: Any, mode: str) -> None:
    """Reject a legacy canvas before collating for a fixed-v2 checkpoint."""

    if mode == "fixed_v2" and not collator_fixed_receptor_lengths(collator):
        raise ValueError(
            "Fixed-v2 pairing requires collator.fixed_receptor_lengths=True; "
            "a legacy collator is incompatible with the checkpoint's fixed canvas"
        )


def _length_condition_label(mode: str) -> str:
    if mode == "fixed_v2":
        return "fixed_canvas_eos"
    if mode == "reference":
        return "reference"
    if mode == "prior":
        return "train_prior"
    raise ValueError(f"unresolved light-length mode: {mode!r}")


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
        if light_col not in frame:
            raise ValueError(f"Pairing evaluation requires reference column {light_col!r}")
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
    """Prompt residues from the reference; all remaining slots are filler."""

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
        "cfg_scale": float(args.cfg_scale),
        "cfg_condition": "observed_residues_both_streams_structure_preserved",
        "encoder_state": "committed_generated_residues_pending_mask_no_reference_targets",
        "light_prompt_tokens": int(args.light_prompt_tokens),
        "light_length_mode": str(args.light_length_mode),
        "protocol_version": int(GENERATION_PROTOCOL_VERSION),
        "seed": args.seed,
        "implementation_sha256": _protocol_source_hashes(),
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
    cfg_scale: float = 0.0,
) -> list[dict]:
    effective_length_mode = resolve_light_length_mode(model, light_length_mode)
    _validate_pairing_collator(collator, effective_length_mode)
    records = []
    metadata_rows: list[tuple[str, str, int, dict, int]] = []
    if not math.isfinite(cfg_scale) or cfg_scale < 0:
        raise ValueError("cfg_scale must be finite and nonnegative")
    if light_prompt_tokens < 0:
        raise ValueError("light_prompt_tokens must be nonnegative")

    if effective_length_mode == "prior":
        if length_prior is None:
            raise ValueError("light_length_mode='prior' requires a length prior")
        drawn = length_prior.sample(len(samples) * num_seqs)
    else:
        drawn = None

    cursor = 0
    for heavy, light, metadata in samples:
        for variant_idx in range(num_seqs):
            if effective_length_mode == "reference" and len(light) <= max(int(light_prompt_tokens), 0):
                raise ValueError("Reference light must contain residues beyond the declared prompt")
            if len(light) < max(int(light_prompt_tokens), 0):
                raise ValueError("Reference light is shorter than the declared prompt")
            if effective_length_mode == "fixed_v2":
                if light_prompt_tokens > 134:
                    raise ValueError("fixed_v2 light canvas has 134 amino-acid slots; prompt exceeds the canvas")
                # 134 amino-acid placeholders plus the renderer's EOS slot make
                # the complete 135-slot light canvas. The reference suffix is
                # never rendered, so neither its length nor its residues enter
                # the model condition.
                target_length = 134
            elif effective_length_mode == "prior":
                if drawn is None:
                    raise RuntimeError("prior mode did not produce sampled lengths")
                target_length = int(drawn[cursor])
            else:
                if len(light) <= max(int(light_prompt_tokens), 0):
                    raise ValueError("Reference light must contain residues beyond the declared prompt")
                target_length = len(light)
            # Never render the hidden reference suffix, even in reference-length
            # mode. The clean answer is kept only in the scoring metadata.
            slot_light = build_placeholder_light(light, target_length, light_prompt_tokens)
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
        cfg_scale=cfg_scale,
    )

    rows: list[dict] = []
    for row_idx, (heavy, light, variant_idx, metadata, target_length) in enumerate(metadata_rows):
        position_ids_chain = batch.get("position_ids_chain")
        chain_slot_mask = batch.get("chain_slot_mask")
        generated_light = extract_chain_sequence(
            output_tokens[row_idx],
            batch["attention_mask"][row_idx],
            batch["residue_mask"][row_idx],
            tokenizer,
            chain="light",
            position_ids_chain=(
                position_ids_chain[row_idx] if position_ids_chain is not None else None
            ),
            chain_slot_mask=(
                chain_slot_mask[row_idx] if chain_slot_mask is not None else None
            ),
        )
        result = {
            "h_sequence": heavy,
            "gen_l_sequence": generated_light,
            "raw_l_sequence": light,
            "variant_idx": variant_idx,
            # Audit trail for the length-leakage fix: where the slot count came from.
            "light_length_mode": effective_length_mode,
            "target_light_length": int(target_length),
            "ref_light_length": len(light),
            "generated_light_length": len(generated_light),
            "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
            "light_prompt_tokens": int(light_prompt_tokens),
            "cfg_scale": float(cfg_scale),
            "length_condition": _length_condition_label(effective_length_mode),
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_source_hashes() -> dict[str, str]:
    source_paths = [
        Path(__file__),
        PROJECT_ROOT / "dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py",
        PROJECT_ROOT / "dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py",
        PROJECT_ROOT / "dllm/pipelines/llada/models/modeling_llada.py",
        PROJECT_ROOT / "dllm/pipelines/immune_llada/data/grammar.py",
        PROJECT_ROOT / "examples/llada/protein_fusion_model.py",
        PROJECT_ROOT / "examples/llada/load_fusion_checkpoint.py",
        PROJECT_ROOT / "downstream/grammar/masks.py",
        PROJECT_ROOT / "downstream/grammar/common.py",
    ]
    return {str(path): _file_sha256(path) for path in source_paths}


def _write_run_manifest(args, saved_path: Path, signature: dict[str, Any]) -> None:
    """Record exact weights/data and active source content, including dirty edits."""
    checkpoint = Path(args.checkpoint_path) if args.checkpoint_path else None
    weights = checkpoint / "model.safetensors" if checkpoint and checkpoint.is_dir() else checkpoint
    payload = {
        **signature,
        "protocol_id": f"ab_pairing_{args.light_length_mode}_length_prompt{args.light_prompt_tokens}_cfg{args.cfg_scale:g}_v{GENERATION_PROTOCOL_VERSION}",
        "length_condition": _length_condition_label(args.light_length_mode),
        "sampler_state": "committed_tokens_only_v1",
        "cfg_scale": float(args.cfg_scale),
        "cfg_formula": "cond + scale * (cond - uncond)",
        "cfg_condition": "observed_residues_both_streams_structure_preserved",
        "checkpoint_sha256": _file_sha256(weights) if weights else None,
        "input_csv_sha256": _file_sha256(Path(args.csv_path)),
        "source_sha256": signature["implementation_sha256"],
        "torch_version": torch.__version__,
        "known_deviations": (
            ["known reference length; not native-EOS protocol"]
            if args.light_length_mode == "reference"
            else ["length sampled from training prior"]
            if args.light_length_mode == "prior"
            else []
        ),
    }
    manifest_path = saved_path.with_name(saved_path.stem + "_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_generation(args) -> Path:
    if not math.isfinite(args.cfg_scale) or args.cfg_scale < 0 or args.light_prompt_tokens < 0:
        raise ValueError("CFG must be finite and nonnegative; prompt length must be nonnegative")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.checkpoint_path:
        model, tokenizer = load_grammar_checkpoint(args.checkpoint_path, device=device)
    else:
        tokenizer = build_grammar_tokenizer()
        model = load_untrained_no_encoder(tokenizer.vocab_size, mask_token_id=tokenizer.mask_token_id).to(device)
    args.light_length_mode = resolve_light_length_mode(model, args.light_length_mode)

    length_prior: LightLengthPrior | None = None
    if args.light_length_mode == "prior":
        length_prior = LightLengthPrior(Path(args.length_prior_json))
        print(
            f"Light-length prior: {length_prior.signature()} "
            f"(reference-independent; target length is NOT taken from the reference)",
            flush=True,
        )
    elif args.light_length_mode == "fixed_v2":
        print(
            "Fixed-v2 light canvas: 134 amino-acid slots plus renderer EOS; "
            "hidden reference suffix is not rendered.",
            flush=True,
        )
    else:
        print(
            "Known-reference-length pairing: target length is an explicit condition; "
            "only the declared light prompt is visible. Not a native-EOS comparison.",
            flush=True,
        )

    collator = build_eval_collator(model, tokenizer)
    _validate_pairing_collator(collator, args.light_length_mode)
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
    if saved_path.exists() and not progress_path.exists():
        raise FileExistsError(f"Refusing to overwrite a completed generation CSV: {saved_path}")
    if not len(dataset):
        raise ValueError("No heavy/light samples selected for generation")
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

    _write_run_manifest(args, saved_path, signature)

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
            cfg_scale=args.cfg_scale,
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


def smoke_eval(device: str = "cpu") -> dict[str, list[str]]:
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
    parser.add_argument("--max-iter", type=int, default=124)
    parser.add_argument("--sampling-strategy", type=str, default="gumbel_argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0,
                        help="cond + s*(cond-uncond); uncond masks observed residues in both streams, preserving grammar")
    parser.add_argument("--light-prompt-tokens", type=int, default=3)
    parser.add_argument(
        "--light-length-mode",
        type=str,
        default="auto",
        choices=("auto", "fixed_v2", "prior", "reference"),
        help=(
            "auto selects fixed_v2 for model.config.fixed_receptor_lengths=True and "
            "reference for legacy checkpoints; explicit mismatches are rejected."
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
