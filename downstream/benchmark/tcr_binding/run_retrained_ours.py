#!/usr/bin/env python
"""Train frozen-BioSeq MLP heads on the official NM2025 retrain folds.

The backbone is never updated. Each peptide--CDR3beta pair (optionally with
observed CDR3alpha from the official ``others`` track) is rendered as one
joint ``task_type=tcr`` grammar record with runtime v5 receptor completion and
a masked recognition query. The full ESMC -> LLaDA checkpoint is frozen;
only observed residue hidden states are pooled (synthetic X is excluded).
A fresh two-layer MLP head is then
trained on each official fold's ``i_1_1train.csv`` and evaluated on that fold's
seen test plus the shared seen/unseen independent tests.

Run the complete AS five-fold experiment on one GPU with::

    source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
    conda activate /project_c20250601/251105016/conda/envs/pllm
    python /project_c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_ours.py \
      --checkpoint /path/to/checkpoint --tag v5_masked_runtime_new \
      --fold all --eval-set all --neg-source AS --cdr3-format junction

Add ``--track cdr3ab`` to use the official others/AS folds with only
Epitope, CDR3B and CDR3A. MHC, V/J and LongA/LongB are ignored.
Use ``--track others_cdr3b`` for a beta-only control on those SAME others rows.
This does not select the larger default cdr3b dataset or rewrite its CSVs.

Frozen features are stored in a versioned resumable memory-mapped cache; the
original CSVs are not rewritten/preprocessed. Published baseline values remain
the primary reference; local official-checkpoint reruns are diagnostic controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parents[1]))

from tcr_binding.query_protocol import (  # noqa: E402
    DEFAULT_COMPLETION_MANIFEST, INPUT_RECORD, POOLING,
    TCRBindingEmbedder, TCRBindingQueryProtocol,
    input_columns_for_mode, validate_long_chain, _sequence,
)
from tcr_binding.retrained_protocol import (  # noqa: E402
    EXPECTED_ARCHIVES,
    FIGSHARE_DOI,
    RETRAIN_DATA_ZIP,
    SUPPLEMENTARY_TABLE_7,
    T1_CITATION,
    load_official_test,
    load_official_train,
    sha256_file,
)


OUT_ROOT = BENCH / "outputs" / "tcr_binding_nm2025_retrained"
OURS_METHOD = "Ours fusion frozen runtime-v5 masked-query observed-mean + MLP"
EVAL_SETS = ("seen_test", "seen_independent", "unseen_independent")
FOLDS = (1, 2, 3, 4, 5)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    frame.to_csv(partial, index=False)
    partial.replace(path)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(content, encoding="utf-8")
    partial.replace(path)


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - artifact identity, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _model_assets(checkpoint: Path) -> dict[str, Any]:
    """Hash local tokenizer/config assets as well as the separately hashed weights."""
    from examples.llada.load_fusion_checkpoint import DEFAULT_ESMC, DEFAULT_LLADA_SNAP

    ckpt_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
    roots = (ckpt_dir, DEFAULT_ESMC, DEFAULT_LLADA_SNAP)
    names = (
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.txt",
        "vocab.json", "merges.txt", "tokenizer.model",
    )
    return {str(root.resolve()): {
        name: sha256_file(root / name) if (root / name).is_file() else None
        for name in names
    } for root in roots}


def _feature_identity(manifest: dict[str, Any]) -> str:
    return _canonical_hash({key: manifest[key] for key in (
        "checkpoint_sha256", "pairs_sha256", "input_protocol_sha256",
        "model_assets_sha256", "feature_dim", "pooling", "input_record",
    )})


def _require_provenance(actual: dict, expected: dict, artifact: str) -> None:
    mismatch = {key: (actual.get(key), value) for key, value in expected.items()
                if actual.get(key) != value}
    if mismatch:
        raise RuntimeError(f"{artifact} provenance mismatch; use a new --tag: {mismatch}")


def _verify_data_archive() -> dict[str, Any]:
    expected = EXPECTED_ARCHIVES[RETRAIN_DATA_ZIP]
    if not RETRAIN_DATA_ZIP.is_file():
        raise FileNotFoundError(f"official retrain data is missing: {RETRAIN_DATA_ZIP}")
    size = RETRAIN_DATA_ZIP.stat().st_size
    if size != expected["size"]:
        raise RuntimeError(
            f"wrong retrain.zip size: {size} != {expected['size']}"
        )
    md5 = _md5_file(RETRAIN_DATA_ZIP)
    if md5 != expected["md5"]:
        raise RuntimeError(f"wrong retrain.zip MD5: {md5} != {expected['md5']}")
    return {"path": str(RETRAIN_DATA_ZIP), "size": size, "md5": md5}


def input_mode_for_track(track: str) -> str:
    """Separate model-visible chains from the official data selection."""
    if track not in {"cdr3b", "cdr3ab", "others_cdr3b", "others_longab"}:
        raise ValueError(f"unsupported input track: {track}")
    if track == "others_longab":
        return "longab"
    return "cdr3ab" if track == "cdr3ab" else "cdr3b"


def data_track_for_track(track: str) -> str:
    input_mode_for_track(track)  # Validate even when called independently.
    return "cdr3b" if track == "cdr3b" else "cdr3ab"


def input_columns(track: str = "cdr3b") -> list[str]:
    mode = input_mode_for_track(track)
    return input_columns_for_mode(mode)


def _frame_keys(frame: pd.DataFrame, track: str = "cdr3b"):
    return frame[input_columns(track)].astype(str).itertuples(index=False, name=None)


def _validate_frame(frame: pd.DataFrame, name: str, track: str = "cdr3b") -> None:
    required = set(input_columns(track)) | {"Affinity"}
    if track == "others_longab":
        required |= {"CDR3A", "CDR3B"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} lacks required columns: {sorted(missing)}")
    labels = frame["Affinity"].astype(int)
    if not set(labels.unique()) <= {0, 1}:
        raise RuntimeError(f"{name} contains non-binary Affinity labels")
    for column in input_columns(track):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().str.len().eq(0).any():
            raise RuntimeError(f"{name} contains an empty {column}")
    if track == "others_longab":
        for index, row in frame.iterrows():
            try:
                _sequence(row["Epitope"], "Epitope")
                validate_long_chain(row["LongB"], row["CDR3B"], "LongB")
                validate_long_chain(row["LongA"], row["CDR3A"], "LongA")
            except ValueError as exc:
                raise RuntimeError(f"{name} row {index}: {exc}") from exc


def load_protocol(neg_source: str, track: str = "cdr3b") -> tuple[dict[int, dict[str, pd.DataFrame]], dict]:
    """Load exact official bytes for all folds and return frames + provenance."""

    frames: dict[int, dict[str, pd.DataFrame]] = {}
    data_track = data_track_for_track(track)
    provenance: dict[str, Any] = {
        "folds": {}, "track": track, "data_track": data_track,
        "input_mode": input_mode_for_track(track), "input_columns": input_columns(track),
    }
    for fold in FOLDS:
        train, _, train_meta = load_official_train(
            neg_source=neg_source, fold=fold, track=data_track,
        )
        _validate_frame(train, f"fold_{fold}/train", track)
        fold_frames: dict[str, pd.DataFrame] = {"train": train}
        fold_meta: dict[str, Any] = {"train": train_meta, "tests": {}}
        for eval_set in EVAL_SETS:
            test, _, test_meta = load_official_test(
                eval_set=eval_set, neg_source=neg_source, fold=fold, track=data_track,
            )
            _validate_frame(test, f"fold_{fold}/{eval_set}", track)
            fold_frames[eval_set] = test
            fold_meta["tests"][eval_set] = test_meta
        frames[fold] = fold_frames
        provenance["folds"][str(fold)] = fold_meta
    return frames, provenance


def _unique_pairs(frame: pd.DataFrame, track: str = "cdr3b") -> set[tuple[str, ...]]:
    return set(_frame_keys(frame, track))


def protocol_overlap_audit(
    frames: dict[int, dict[str, pd.DataFrame]],
    track: str = "cdr3b",
) -> dict[str, Any]:
    """Report train/test overlap without treating intended seen epitopes as errors."""

    columns = input_columns(track)
    result: dict[str, Any] = {"folds": {}, "input_columns": columns}
    for fold in FOLDS:
        train = frames[fold]["train"]
        train_epitopes = set(train["Epitope"].astype(str))
        train_pairs = _unique_pairs(train, track)
        train_tcrs = {pair[1:] for pair in train_pairs}
        duplicates = int(train.duplicated(columns).sum())
        conflicts = int(
            train.groupby(columns)["Affinity"]
            .nunique().gt(1).sum()
        )
        fold_result: dict[str, Any] = {
            "train_rows": int(len(train)),
            "train_duplicate_pair_rows": duplicates,
            "train_conflicting_pair_labels": conflicts,
            "tests": {},
        }
        if duplicates or conflicts:
            raise RuntimeError(
                f"fold {fold} official train has duplicate/conflicting pair labels"
            )
        for eval_set in EVAL_SETS:
            test = frames[fold][eval_set]
            test_epitopes = set(test["Epitope"].astype(str))
            test_pairs = _unique_pairs(test, track)
            test_tcrs = {pair[1:] for pair in test_pairs}
            pair_overlap = train_pairs & test_pairs
            fold_result["tests"][eval_set] = {
                "test_rows": int(len(test)),
                "shared_unique_epitopes": int(len(train_epitopes & test_epitopes)),
                "shared_unique_clonotypes": int(len(train_tcrs & test_tcrs)),
                "shared_unique_exact_pairs": int(len(pair_overlap)),
                "test_rows_with_exact_pair_in_train": int(sum(
                    pair in train_pairs
                    for pair in _frame_keys(test, track)
                )),
            }
        result["folds"][str(fold)] = fold_result
    result["interpretation"] = {
        "seen_epitope_overlap": "expected by the official seen protocol",
        "exact_pair_or_clonotype_overlap": "must be reported; zero is preferred",
        "unseen_epitope_overlap": "must be zero for a genuinely unseen-epitope split",
    }
    return result


def build_pair_universe(
    frames: dict[int, dict[str, pd.DataFrame]],
    track: str = "cdr3b",
) -> list[tuple[str, ...]]:
    """Return the deterministic union used by the shared frozen-feature cache."""

    pairs: set[tuple[str, ...]] = set()
    for fold in FOLDS:
        for split in ("train", *EVAL_SETS):
            pairs.update(_unique_pairs(frames[fold][split], track))
    return sorted(pairs)


def pair_universe_sha256(pairs: list[tuple[str, ...]]) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(b"\0".join(value.encode("utf-8") for value in pair))
        digest.update(b"\n")
    return digest.hexdigest()


def _cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "pairs": cache_dir / "pairs.csv",
        "features": cache_dir / "features.npy",
        "partial": cache_dir / "features.partial.npy",
        "state": cache_dir / "feature_state.json",
        "manifest": cache_dir / "feature_manifest.json",
    }


def _check_cache_manifest(
    manifest: dict[str, Any], *, checkpoint_sha256: str,
    pairs_sha256: str, n_pairs: int, input_protocol_sha256: str,
    model_assets_sha256: str, input_record: str = INPUT_RECORD,
) -> None:
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "pairs_sha256": pairs_sha256,
        "n_pairs": n_pairs,
        "feature_source": "post_llada_final_hidden_state",
        "pooling": POOLING,
        "input_record": input_record,
        "input_protocol_sha256": input_protocol_sha256,
        "model_assets_sha256": model_assets_sha256,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "feature cache provenance mismatch; use a new --tag/cache or "
            f"--force-features: {mismatches}"
        )


def extract_or_load_features(
    *, pairs: list[tuple[str, ...]], checkpoint: Path,
    checkpoint_sha256: str, cache_dir: Path, embedding_batch_size: int,
    feature_chunk_size: int, force: bool,
    query_protocol: TCRBindingQueryProtocol,
    model_assets: dict[str, Any],
) -> tuple[np.memmap, dict[str, Any]]:
    """Materialize resumable post-LLaDA global-mean features for every pair."""

    paths = _cache_paths(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs_sha = pair_universe_sha256(pairs)
    assets_sha = _canonical_hash(model_assets)
    if force:
        for path in paths.values():
            path.unlink(missing_ok=True)

    if paths["manifest"].is_file():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        _check_cache_manifest(
            manifest,
            checkpoint_sha256=checkpoint_sha256,
            pairs_sha256=pairs_sha,
            n_pairs=len(pairs),
            input_protocol_sha256=query_protocol.sha256,
            model_assets_sha256=assets_sha,
            input_record=query_protocol.provenance["input_record"],
        )
        if manifest.get("status") == "complete" and paths["features"].is_file():
            features = np.load(paths["features"], mmap_mode="r")
            expected_shape = (len(pairs), int(manifest["feature_dim"]))
            if features.shape != expected_shape or features.dtype != np.float32:
                raise RuntimeError(
                    f"invalid complete feature cache: {features.shape}/{features.dtype}"
                )
            print(f"[features] reuse {features.shape} from {paths['features']}")
            return features, manifest

    if paths["features"].exists():
        raise RuntimeError(
            f"feature file exists without a matching complete manifest: "
            f"{paths['features']}"
        )

    # Keep the human-auditable row map synchronized even after a failed run
    # reused the same directory before its partial-state manifest was written.
    _atomic_csv(
        paths["pairs"],
        pd.DataFrame(pairs, columns=query_protocol.provenance["input_columns"]),
    )

    from common.model_api import build_embedder
    import torch

    print(f"[features] load frozen backbone: {checkpoint}")
    backbone = build_embedder(
        f"grammar:decoder:global:{checkpoint}",
        batch_size=embedding_batch_size,
        max_length=query_protocol.max_length,
    )
    embedder = TCRBindingEmbedder(backbone, query_protocol)
    embedder.model.eval()
    for parameter in embedder.model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in embedder.model.parameters()):
        raise RuntimeError("backbone freeze invariant failed")
    if embedder.feature_source != "decoder" or embedder.pool_mode != "global":
        raise RuntimeError("wrong feature source/pooling for the canonical protocol")
    feature_dim = int(embedder.hidden)

    if paths["state"].is_file() and paths["partial"].is_file():
        state = json.loads(paths["state"].read_text(encoding="utf-8"))
        _check_cache_manifest(
            state,
            checkpoint_sha256=checkpoint_sha256,
            pairs_sha256=pairs_sha,
            n_pairs=len(pairs),
            input_protocol_sha256=query_protocol.sha256,
            model_assets_sha256=assets_sha,
            input_record=query_protocol.provenance["input_record"],
        )
        if int(state["feature_dim"]) != feature_dim:
            raise RuntimeError("partial feature dimension does not match checkpoint")
        next_index = int(state["next_index"])
        mmap = np.lib.format.open_memmap(paths["partial"], mode="r+")
        if mmap.shape != (len(pairs), feature_dim) or mmap.dtype != np.float32:
            raise RuntimeError("partial feature mmap has the wrong shape/dtype")
        print(f"[features] resume at {next_index}/{len(pairs)}")
    else:
        paths["partial"].unlink(missing_ok=True)
        paths["state"].unlink(missing_ok=True)
        next_index = 0
        mmap = np.lib.format.open_memmap(
            paths["partial"], mode="w+", dtype=np.float32,
            shape=(len(pairs), feature_dim),
        )

    base_manifest: dict[str, Any] = {
        "status": "extracting",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "data_archive": str(RETRAIN_DATA_ZIP),
        "data_archive_md5": EXPECTED_ARCHIVES[RETRAIN_DATA_ZIP]["md5"],
        "pairs_sha256": pairs_sha,
        "n_pairs": len(pairs),
        "feature_dim": feature_dim,
        "dtype": "float32",
        "feature_source": "post_llada_final_hidden_state",
        "pooling": POOLING,
        "input_record": query_protocol.provenance["input_record"],
        "input_protocol": query_protocol.provenance,
        "input_protocol_sha256": query_protocol.sha256,
        "model_assets": model_assets,
        "model_assets_sha256": assets_sha,
        "backbone_trainable_parameters": 0,
        "requested_embedding_batch_size": embedding_batch_size,
        "effective_embedding_batch_size": int(embedder.batch_size),
        "feature_chunk_size": feature_chunk_size,
        "next_index": next_index,
    }
    _atomic_json(paths["state"], base_manifest)
    started = time.time()
    for start in range(next_index, len(pairs), feature_chunk_size):
        end = min(start + feature_chunk_size, len(pairs))
        chunk = pairs[start:end]
        while True:
            try:
                vectors = embedder.embed_pairs(
                    beta=[pair[1] for pair in chunk],
                    alpha=[pair[2] for pair in chunk] if query_protocol.input_mode in {"cdr3ab", "longab"} else None,
                    peptide=[pair[0] for pair in chunk],
                    order=("beta", "peptide"),
                )
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                    "out of memory" in str(exc).lower()
                )
                if not is_oom or embedder.batch_size <= 1:
                    raise
                embedder._cache.clear()
                torch.cuda.empty_cache()
                embedder.batch_size = max(1, embedder.batch_size // 2)
                print(
                    f"[features] CUDA OOM; retry chunk with batch="
                    f"{embedder.batch_size}"
                )
        if vectors.shape != (end - start, feature_dim):
            raise RuntimeError(f"unexpected feature shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise RuntimeError(f"non-finite frozen feature in rows [{start}, {end})")
        mmap[start:end] = vectors
        mmap.flush()
        embedder._cache.clear()
        base_manifest.update({
            "next_index": end,
            "effective_embedding_batch_size": int(embedder.batch_size),
        })
        _atomic_json(paths["state"], base_manifest)
        rate = end / max(time.time() - started, 1e-6)
        print(
            f"[features] {end}/{len(pairs)} ({100 * end / len(pairs):.1f}%) "
            f"current-run-rate={rate:.1f} pairs/s"
        )

    mmap.flush()
    del mmap
    paths["partial"].replace(paths["features"])
    base_manifest.update({
        "status": "complete",
        "next_index": len(pairs),
        "features_path": str(paths["features"]),
        "pairs_path": str(paths["pairs"]),
        "elapsed_sec_current_process": round(time.time() - started, 3),
    })
    _atomic_json(paths["manifest"], base_manifest)
    paths["state"].unlink(missing_ok=True)
    del embedder, backbone
    torch.cuda.empty_cache()
    features = np.load(paths["features"], mmap_mode="r")
    return features, base_manifest


def frame_feature_indices(
    frame: pd.DataFrame, lookup: dict[tuple[str, ...], int], track: str = "cdr3b",
) -> np.ndarray:
    try:
        return np.fromiter(
            (
                lookup[pair] for pair in _frame_keys(frame, track)
            ),
            dtype=np.int64,
            count=len(frame),
        )
    except KeyError as exc:
        raise RuntimeError(f"pair missing from frozen-feature universe: {exc}") from exc


def _make_mlp(input_dim: int, hidden: int, dropout: float, device: str):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1),
    ).to(device)


def train_head(
    features: np.memmap, feature_indices: np.ndarray, labels: np.ndarray,
    *, seed: int, hidden: int, dropout: float, learning_rate: float,
    weight_decay: float, batch_size: int, max_epochs: int, patience: int,
    val_fraction: float, validation_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit only the MLP; return CPU state plus train-only normalization."""

    import torch
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels, dtype=np.int64)
    rows = np.arange(len(labels), dtype=np.int64)
    if validation_mask is None:
        train_rows, val_rows = train_test_split(
            rows, test_size=val_fraction, random_state=seed,
            shuffle=True, stratify=labels,
        )
    else:
        validation_mask = np.asarray(validation_mask)
        if validation_mask.dtype != np.bool_ or validation_mask.shape != labels.shape:
            raise ValueError("validation_mask must be boolean and match labels")
        train_rows, val_rows = rows[~validation_mask], rows[validation_mask]
        if not len(train_rows) or not len(val_rows):
            raise ValueError("explicit training and validation partitions must be nonempty")
    x = np.asarray(features[feature_indices], dtype=np.float32)
    mean = x[train_rows].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x[train_rows].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    x -= mean
    x /= std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _make_mlp(x.shape[1], hidden, dropout, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(labels.astype(np.float32)).to(device)
    train_tensor = torch.from_numpy(train_rows).to(device)
    val_tensor = torch.from_numpy(val_rows).to(device)
    del x

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(
            len(train_tensor), generator=generator, device=device,
        )
        total_loss = 0.0
        total_rows = 0
        for offset in range(0, len(order), batch_size):
            batch_rows = train_tensor[order[offset:offset + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[batch_rows]).squeeze(-1)
            loss = loss_function(logits, y_tensor[batch_rows])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_rows)
            total_rows += len(batch_rows)
        model.eval()
        with torch.no_grad():
            val_loss = float(
                loss_function(
                    model(x_tensor[val_tensor]).squeeze(-1),
                    y_tensor[val_tensor],
                )
            )
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(total_rows, 1),
            "val_loss": val_loss,
        })
        print(
            f"      epoch={epoch:03d} train_loss={history[-1]['train_loss']:.6f} "
            f"val_loss={val_loss:.6f}"
        )
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP training did not produce a finite validation state")
    return {
        "state_dict": best_state,
        "feature_mean": mean,
        "feature_std": std,
        "input_dim": int(features.shape[1]),
        "hidden": hidden,
        "dropout": dropout,
        "seed": seed,
        "best_val_loss": best_loss,
        "epochs_run": len(history),
        "n_train": int(len(train_rows)),
        "n_validation": int(len(val_rows)),
        "elapsed_sec": round(time.time() - started, 3),
        "history": history,
    }


def predict_head(
    head: dict[str, Any], features: np.memmap, feature_indices: np.ndarray,
    *, batch_size: int,
) -> np.ndarray:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _make_mlp(
        int(head["input_dim"]), int(head["hidden"]),
        float(head["dropout"]), device,
    )
    model.load_state_dict(head["state_dict"])
    model.eval()
    mean = np.asarray(head["feature_mean"], dtype=np.float32)
    std = np.asarray(head["feature_std"], dtype=np.float32)
    output = np.empty(len(feature_indices), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(feature_indices), batch_size):
            end = min(start + batch_size, len(feature_indices))
            x = np.asarray(features[feature_indices[start:end]], dtype=np.float32)
            x = (x - mean) / std
            logits = model(torch.from_numpy(x).to(device)).squeeze(-1)
            output[start:end] = torch.sigmoid(logits).cpu().numpy()
    return output


def paper_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from common.metrics import precrec_pr_auc
    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
        precision_score, recall_score, roc_auc_score,
    )

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores > 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc_precrec": float(precrec_pr_auc(labels, scores)),
        "average_precision_sklearn": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def _experiment_root(tag: str, neg_source: str, track: str = "cdr3b") -> Path:
    return OUT_ROOT / tag / track / neg_source


def _fold_dir(tag: str, neg_source: str, fold: int, track: str = "cdr3b") -> Path:
    return _experiment_root(tag, neg_source, track) / f"fold_{fold}"


def _head_config(args: argparse.Namespace, fold_seed: int) -> dict[str, Any]:
    return {
        "architecture": (
            f"input-{args.head_hidden}-{args.head_hidden // 2}-1_relu_dropout"
        ),
        "hidden": args.head_hidden,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.head_batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "validation_fraction": args.val_fraction,
        "validation_split": "train-only stratified random split",
        "fold_seed": fold_seed,
    }


def run_fold(
    *, fold: int, eval_sets: list[str], frames: dict[int, dict[str, pd.DataFrame]],
    data_provenance: dict[str, Any], overlap_audit: dict[str, Any],
    features: np.memmap, pair_lookup: dict[tuple[str, ...], int],
    feature_manifest: dict[str, Any], checkpoint_sha256: str,
    args: argparse.Namespace,
) -> None:
    import torch

    fold_seed = args.seed + fold
    fold_dir = _fold_dir(args.tag, args.neg_source, fold, args.track)
    fold_dir.mkdir(parents=True, exist_ok=True)
    head_path = fold_dir / "head.pt"
    train = frames[fold]["train"]
    train_meta = data_provenance["folds"][str(fold)]["train"]
    config = _head_config(args, fold_seed)
    feature_identity = _feature_identity(feature_manifest)

    if head_path.is_file() and not args.force_heads:
        head = torch.load(head_path, map_location="cpu", weights_only=False)
        expected = {
            "checkpoint_sha256": checkpoint_sha256,
            "pairs_sha256": feature_manifest["pairs_sha256"],
            "train_member_sha256": train_meta["train_member_sha256"],
            "fold": fold,
            "head_config": config,
            "feature_identity": feature_identity,
        }
        mismatch = {
            key: (head.get(key), value)
            for key, value in expected.items()
            if head.get(key) != value
        }
        if mismatch:
            raise RuntimeError(
                f"fold {fold} cached head provenance mismatch; "
                f"use --force-heads: {mismatch}"
            )
        print(f"[head] reuse fold={fold}: {head_path}")
    else:
        indices = frame_feature_indices(train, pair_lookup, args.track)
        labels = train["Affinity"].astype(int).to_numpy()
        print(
            f"[head] train fold={fold} rows={len(train)} seed={fold_seed} "
            f"input_dim={features.shape[1]}"
        )
        head = train_head(
            features, indices, labels,
            seed=fold_seed,
            hidden=args.head_hidden,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.head_batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            val_fraction=args.val_fraction,
        )
        head.update({
            "status": "complete",
            "protocol": "nm2025_official_retrain_frozen_bioseq_mlp",
            "fold": fold,
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "pairs_sha256": feature_manifest["pairs_sha256"],
            "feature_identity": feature_identity,
            "input_protocol": feature_manifest["input_protocol"],
            "input_protocol_sha256": feature_manifest["input_protocol_sha256"],
            "train_member": train_meta["train_member"],
            "train_member_sha256": train_meta["train_member_sha256"],
            "head_config": config,
            "backbone_frozen": True,
            "backbone_trainable_parameters": 0,
        })
        partial = head_path.with_name(head_path.name + ".tmp")
        torch.save(head, partial)
        partial.replace(head_path)
        print(
            f"[head] fold={fold} best_val={head['best_val_loss']:.6f} "
            f"epochs={head['epochs_run']}"
        )

    head_sha = sha256_file(head_path)
    for eval_set in eval_sets:
        out_dir = fold_dir / eval_set
        metrics_path = out_dir / "metrics.json"
        test_meta = data_provenance["folds"][str(fold)]["tests"][eval_set]
        prediction_identity = {
            "feature_identity": feature_identity,
            "head_sha256": head_sha,
            "test_member_sha256": test_meta["test_member_sha256"],
            "train_member_sha256": train_meta["train_member_sha256"],
            "fold": fold,
            "eval_set": eval_set,
        }
        if metrics_path.is_file() and not (args.force_eval or args.force_heads):
            prior = json.loads(metrics_path.read_text(encoding="utf-8"))
            _require_provenance(prior.get("prediction_identity", {}),
                                prediction_identity, "cached predictions")
            print(f"[eval] reuse fold={fold} {eval_set}: {metrics_path}")
            continue
        test = frames[fold][eval_set]
        test_indices = frame_feature_indices(test, pair_lookup, args.track)
        labels = test["Affinity"].astype(int).to_numpy()
        scores = predict_head(
            head, features, test_indices, batch_size=args.predict_batch_size,
        )
        metrics = paper_metrics(labels, scores)
        predictions = pd.DataFrame({
            "row_id": np.arange(len(test), dtype=np.int64),
            "peptide": test["Epitope"].astype(str),
            "cdr3b": test["CDR3B"].astype(str),
            "label": labels,
            "score": scores,
        })
        if data_track_for_track(args.track) == "cdr3ab":
            # Preserve source-row identity for auditing, not as model input.
            predictions["cdr3a"] = test["CDR3A"].astype(str)
        if args.track == "others_longab":
            predictions["longb"] = test["LongB"].astype(str)
            predictions["longa"] = test["LongA"].astype(str)
        _atomic_csv(out_dir / "predictions.csv", predictions)
        test_meta = data_provenance["folds"][str(fold)]["tests"][eval_set]
        payload = {
            "status": "ours_frozen_backbone_mlp_complete",
            "method": OURS_METHOD,
            "tag": args.tag,
            "track": args.track,
            "data_track": data_track_for_track(args.track),
            "input_mode": input_mode_for_track(args.track),
            "input_columns": input_columns(args.track),
            "neg_source": args.neg_source,
            "fold": fold,
            "eval_set": eval_set,
            "n_input": int(len(test)),
            "n_scored": int(len(scores)),
            "prediction_identity": prediction_identity,
            "metrics": metrics,
            "model_protocol": {
                "checkpoint_path": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "backbone_frozen": True,
                "backbone_trainable_parameters": 0,
                "input": feature_manifest["input_record"],
                "input_protocol": feature_manifest["input_protocol"],
                "input_protocol_sha256": feature_manifest["input_protocol_sha256"],
                "feature_source": "final post-LLaDA hidden state",
                "pooling": POOLING,
                "classifier": config,
                "head_path": str(head_path),
                "head_sha256": head_sha,
            },
            "data_protocol": {
                "name": "nm2025_official_retrained_five_fold",
                "artifact_source": FIGSHARE_DOI,
                "citation": T1_CITATION,
                "data_archive": str(RETRAIN_DATA_ZIP),
                "data_archive_md5": EXPECTED_ARCHIVES[RETRAIN_DATA_ZIP]["md5"],
                **train_meta,
                **test_meta,
                "overlap_audit": overlap_audit["folds"][str(fold)]["tests"][eval_set],
            },
            "metric_protocol": {
                "headline_auprc": "precrec::evalmod default-compatible PRC-AUC",
                "precrec_x_bins": 1000,
                "diagnostic_only": "sklearn average_precision_score",
                "aggregation": "five-fold mean and sample SD",
            },
        }
        _atomic_json(metrics_path, payload)
        print(
            f"[eval] fold={fold} {eval_set} AUROC={metrics['auroc']:.6f} "
            f"AUPRC={metrics['auprc_precrec']:.6f} n={len(scores)}"
        )


def summarize(tag: str, neg_source: str, *, feature_identity: str,
              track: str = "cdr3b") -> dict[str, Any]:
    summary: dict[str, Any] = {
        "tag": tag,
        "method": OURS_METHOD,
        "protocol": "nm2025_official_retrain_frozen_bioseq_mlp",
        "feature_identity": feature_identity,
        "track": track,
        "data_track": data_track_for_track(track),
        "input_mode": input_mode_for_track(track),
        "input_columns": input_columns(track),
        "folds_expected": list(FOLDS),
        "eval_sets": {},
    }
    for eval_set in EVAL_SETS:
        records = []
        for fold in FOLDS:
            path = _fold_dir(tag, neg_source, fold, track) / eval_set / "metrics.json"
            if path.is_file():
                record = json.loads(path.read_text(encoding="utf-8"))
                _require_provenance(record.get("prediction_identity", {}), {
                    "feature_identity": feature_identity, "fold": fold,
                    "eval_set": eval_set,
                }, "fold metrics")
                records.append(record)
        entry: dict[str, Any] = {
            "folds_complete": [record["fold"] for record in records],
        }
        if len(records) == len(FOLDS):
            auroc = np.asarray([record["metrics"]["auroc"] for record in records])
            auprc = np.asarray([
                record["metrics"]["auprc_precrec"] for record in records
            ])
            ap = np.asarray([
                record["metrics"]["average_precision_sklearn"]
                for record in records
            ])
            entry["local"] = {
                "auroc_mean": float(auroc.mean()),
                "auroc_std_sample": float(auroc.std(ddof=1)),
                "auprc_precrec_mean": float(auprc.mean()),
                "auprc_precrec_std_sample": float(auprc.std(ddof=1)),
                "average_precision_mean": float(ap.mean()),
                "average_precision_std_sample": float(ap.std(ddof=1)),
            }
        summary["eval_sets"][eval_set] = entry
    _atomic_json(_experiment_root(tag, neg_source, track) / "summary.json", summary)
    return summary


def _ranking_frame(
    summary: dict[str, Any], comparison: pd.DataFrame, *, source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if source == "local_release":
        metric_columns = {
            "AUROC": ("local_auroc_mean", "local_auroc_std"),
            "AUPRC": ("local_auprc_mean", "local_auprc_std"),
        }
        baseline_source = "official_retrained_checkpoint_local_rerun"
        scope = "same released retrain.zip test bytes"
    elif source == "paper_reference":
        metric_columns = {
            "AUROC": ("paper_auroc_mean", "paper_auroc_std"),
            "AUPRC": ("paper_auprc_mean", "paper_auprc_std"),
        }
        baseline_source = "paper_reported_retrained"
        scope = "mixed-source reference only; paper seen data differ from release"
    else:
        raise ValueError(source)

    for eval_set in EVAL_SETS:
        ours = summary["eval_sets"][eval_set].get("local")
        if ours is None:
            continue
        baseline = comparison[comparison["eval_set"] == eval_set]
        for metric, (mean_col, std_col) in metric_columns.items():
            metric_rows = [
                {
                    "tag": row.tag,
                    "method": row.method,
                    "mean": float(getattr(row, mean_col)),
                    "std_sample": float(getattr(row, std_col)),
                    "result_source": baseline_source,
                    "n_note": _baseline_n_note(row.tag, eval_set),
                }
                for row in baseline.itertuples(index=False)
            ]
            ours_mean = (
                ours["auroc_mean"] if metric == "AUROC"
                else ours["auprc_precrec_mean"]
            )
            ours_std = (
                ours["auroc_std_sample"] if metric == "AUROC"
                else ours["auprc_precrec_std_sample"]
            )
            metric_rows.append({
                "tag": summary["tag"],
                "method": summary["method"],
                "mean": ours_mean,
                "std_sample": ours_std,
                "result_source": "local_frozen_backbone_mlp",
                "n_note": "all official rows",
            })
            ranked = pd.DataFrame(metric_rows)
            ranked["rank"] = ranked["mean"].rank(
                method="min", ascending=False,
            ).astype(int)
            ranked = ranked.sort_values(["rank", "method"])
            for record in ranked.to_dict("records"):
                rows.append({
                    "eval_set": eval_set,
                    "metric": metric,
                    "rank": record["rank"],
                    "n_methods": len(ranked),
                    "tag": record["tag"],
                    "method": record["method"],
                    "mean": record["mean"],
                    "std_sample": record["std_sample"],
                    "result_source": record["result_source"],
                    "comparison_scope": scope,
                    "mixed_source": source == "paper_reference",
                    "n_note": record["n_note"],
                })
    return pd.DataFrame(rows)


def _baseline_n_note(tag: str, eval_set: str) -> str:
    """Describe official row drops that make one local rerun slightly non-identical."""

    if tag != "ergo_ae":
        return "all official rows"
    if eval_set == "seen_test":
        return (
            "official ERGO-AE batch-50 tail-drop by fold: "
            "30650/30670,30650/30660,30350/30384,"
            "30250/30282,30250/30268"
        )
    if eval_set == "seen_independent":
        return "5850/5882 rows per fold (official ERGO-AE batch-50 tail-drop)"
    return "3150/3162 rows per fold (official ERGO-AE batch-50 tail-drop)"


def _wide_comparison(
    ranking: pd.DataFrame, *, tag_order: list[str],
) -> pd.DataFrame:
    """Pivot the long metric ranking into one row per model for direct review."""

    rows: list[dict[str, Any]] = []
    for tag in tag_order:
        model_rows = ranking[ranking["tag"] == tag]
        if model_rows.empty:
            continue
        if len(model_rows) != len(EVAL_SETS) * 2:
            raise RuntimeError(f"incomplete ranking rows for {tag}: {len(model_rows)}")
        row: dict[str, Any] = {
            "tag": tag,
            "method": str(model_rows["method"].iloc[0]),
            "result_source": str(model_rows["result_source"].iloc[0]),
            "comparison_scope": str(model_rows["comparison_scope"].iloc[0]),
            "mixed_source": bool(model_rows["mixed_source"].iloc[0]),
        }
        for eval_set in EVAL_SETS:
            split_rows = model_rows[model_rows["eval_set"] == eval_set]
            notes = split_rows["n_note"].unique().tolist()
            if len(notes) != 1:
                raise RuntimeError(f"inconsistent n_note for {tag}/{eval_set}")
            row[f"{eval_set}_n_note"] = notes[0]
            for metric in ("AUROC", "AUPRC"):
                record = split_rows[split_rows["metric"] == metric]
                if len(record) != 1:
                    raise RuntimeError(f"missing {tag}/{eval_set}/{metric}")
                prefix = f"{eval_set}_{metric.lower()}"
                row[f"{prefix}_mean"] = float(record["mean"].iloc[0])
                row[f"{prefix}_std_sample"] = float(record["std_sample"].iloc[0])
                row[f"{prefix}_rank"] = int(record["rank"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _comparison_markdown(frame: pd.DataFrame, *, title: str, note: str) -> str:
    def cell(row, eval_set: str, metric: str) -> str:
        prefix = f"{eval_set}_{metric}"
        return (
            f"{row[f'{prefix}_mean']:.4f}±{row[f'{prefix}_std_sample']:.4f} "
            f"(#{int(row[f'{prefix}_rank'])})"
        )

    lines = [
        f"# {title}",
        "",
        note,
        "",
        "| Method | Seen test AUROC | Seen test AUPRC | Seen independent AUROC | Seen independent AUPRC | Unseen AUROC | Unseen AUPRC | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| " + " | ".join([
                str(row["method"]),
                cell(row, "seen_test", "auroc"),
                cell(row, "seen_test", "auprc"),
                cell(row, "seen_independent", "auroc"),
                cell(row, "seen_independent", "auprc"),
                cell(row, "unseen_independent", "auroc"),
                cell(row, "unseen_independent", "auprc"),
                str(row["result_source"]),
            ]) + " |"
        )
    lines.extend([
        "",
        "All AUPRC values use the precrec-compatible metric; ranks are among the nine displayed methods.",
        "ERGO-AE official tail-drop row counts are retained in the CSV `*_n_note` columns.",
        "",
    ])
    return "\n".join(lines)


def _seen_split_comparison(
    wide: pd.DataFrame, *, eval_set: str, ours_tag: str,
) -> pd.DataFrame:
    """One seen table: baselines by descending AUPRC, with Ours always last."""

    if eval_set not in {"seen_test", "seen_independent"}:
        raise ValueError(f"seen presentation does not accept {eval_set}")
    columns = {
        "tag": "tag",
        "method": "method",
        f"{eval_set}_auroc_mean": "auroc_mean",
        f"{eval_set}_auroc_std_sample": "auroc_std_sample",
        f"{eval_set}_auroc_rank": "auroc_rank",
        f"{eval_set}_auprc_mean": "auprc_mean",
        f"{eval_set}_auprc_std_sample": "auprc_std_sample",
        f"{eval_set}_auprc_rank": "auprc_rank",
        "result_source": "result_source",
        f"{eval_set}_n_note": "n_note",
    }
    table = wide[list(columns)].rename(columns=columns)
    ours = table[table["tag"] == ours_tag]
    baselines = table[table["tag"] != ours_tag].sort_values(
        ["auprc_mean", "auroc_mean", "method"],
        ascending=[False, False, True],
    )
    if len(ours) != 1:
        raise RuntimeError(f"expected one Ours row, found {len(ours)}")
    result = pd.concat([baselines, ours], ignore_index=True)
    result.insert(0, "display_order", np.arange(1, len(result) + 1))
    result.insert(1, "display_rule", "baselines_by_auprc_desc_then_ours_last")
    return result


def _seen_split_markdown(frame: pd.DataFrame, *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Baselines are sorted by descending AUPRC; Ours is fixed to the final row.",
        "",
        "| Method | AUROC | AUPRC | Source |",
        "|---|---:|---:|---|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['method']} | "
            f"{row['auroc_mean']:.4f}±{row['auroc_std_sample']:.4f} "
            f"(#{int(row['auroc_rank'])}) | "
            f"{row['auprc_mean']:.4f}±{row['auprc_std_sample']:.4f} "
            f"(#{int(row['auprc_rank'])}) | {row['result_source']} |"
        )
    lines.extend([
        "",
        "AUPRC uses the precrec-compatible metric; ranks remain the ranks among all nine displayed methods.",
        "Official row-drop details, where applicable, are stored in the CSV `n_note` column.",
        "",
    ])
    return "\n".join(lines)


def write_rankings(summary: dict[str, Any], neg_source: str) -> None:
    if summary.get("track", "cdr3b") != "cdr3b":
        print("[ranking] others results are separate; no larger beta-only dataset ranking")
        return
    if neg_source != "AS":
        print("[ranking] paper/local comparison is currently defined only for AS")
        return
    comparison_path = OUT_ROOT / f"comparison_{neg_source}.csv"
    if not comparison_path.is_file():
        raise FileNotFoundError(
            f"baseline comparison is missing: {comparison_path}"
        )
    if not all(
        "local" in summary["eval_sets"][eval_set] for eval_set in EVAL_SETS
    ):
        print("[ranking] skip until all five folds and three eval sets are complete")
        return
    comparison = pd.read_csv(comparison_path)
    root = _experiment_root(summary["tag"], neg_source)
    local = _ranking_frame(summary, comparison, source="local_release")
    paper = _ranking_frame(summary, comparison, source="paper_reference")
    _atomic_csv(root / f"ranking_local_release_{neg_source}.csv", local)
    _atomic_csv(root / f"ranking_paper_reference_{neg_source}.csv", paper)
    tag_order = comparison["tag"].drop_duplicates().astype(str).tolist()
    tag_order.append(str(summary["tag"]))
    local_wide = _wide_comparison(local, tag_order=tag_order)
    paper_wide = _wide_comparison(paper, tag_order=tag_order)
    local_name = f"comparison_with_ours_local_release_{neg_source}"
    paper_name = f"comparison_with_ours_paper_reference_{neg_source}"
    _atomic_csv(root / f"{local_name}.csv", local_wide)
    _atomic_csv(root / f"{paper_name}.csv", paper_wide)
    _atomic_text(
        root / f"{local_name}.md",
        _comparison_markdown(
            local_wide,
            title="NM2025 retrained five-fold: local release comparison with Ours",
            note=(
                "Primary apples-to-apples table: baseline values come from local "
                "official-checkpoint reruns on the released retrain.zip data; Ours "
                "uses the same released folds with a frozen BioSeq backbone."
            ),
        ),
    )
    _atomic_text(
        root / f"{paper_name}.md",
        _comparison_markdown(
            paper_wide,
            title="NM2025 retrained five-fold: paper-reference comparison with Ours",
            note=(
                "Mixed-source reference only: baseline values are paper-reported, "
                "whereas Ours is the local released-fold result."
            ),
        ),
    )
    for eval_set in ("seen_test", "seen_independent"):
        split = _seen_split_comparison(
            local_wide, eval_set=eval_set, ours_tag=str(summary["tag"]),
        )
        split_name = f"comparison_with_ours_{eval_set}_local_release_{neg_source}"
        _atomic_csv(root / f"{split_name}.csv", split)
        _atomic_text(
            root / f"{split_name}.md",
            _seen_split_markdown(
                split,
                title=f"NM2025 retrained five-fold: {eval_set} with Ours",
            ),
        )
    print(f"[ranking] primary -> {root / f'ranking_local_release_{neg_source}.csv'}")
    print(f"[comparison] combined -> {root / f'{local_name}.csv'}")
    ours = local[local["tag"] == summary["tag"]]
    print(ours[["eval_set", "metric", "rank", "n_methods", "mean"]].to_string(index=False))


def _safe_tag(tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError("--tag may contain only letters, digits, underscore, dot, dash")
    return tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tag", type=_safe_tag, required=True)
    parser.add_argument("--neg-source", choices=["AS", "PS", "HS"], default="AS")
    parser.add_argument("--track", choices=["cdr3b", "cdr3ab", "others_cdr3b", "others_longab"], default="cdr3b",
                        help="cdr3b: beta dataset; cdr3ab: others with alpha/beta; "
                             "others_cdr3b: same others rows with beta-only inputs; "
                             "others_longab: same others rows with provided LongB/LongA")
    parser.add_argument("--fold", choices=["all", "1", "2", "3", "4", "5"], default="all")
    parser.add_argument("--eval-set", choices=["all", *EVAL_SETS], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--completion-manifest", type=Path,
                        default=DEFAULT_COMPLETION_MANIFEST,
                        help="Frozen training region-length profile; benchmark rows are completed at runtime")
    parser.add_argument("--cdr3-format", choices=["junction", "core"], default="junction",
                        help="Explicit dataset CDR3 convention; never inferred per sequence")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--feature-chunk-size", type=int, default=4096)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--head-batch-size", type=int, default=512)
    parser.add_argument("--predict-batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-heads", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--feature-only", action="store_true")
    args = parser.parse_args()

    if args.embedding_batch_size <= 0 or args.feature_chunk_size <= 0:
        parser.error("embedding/feature batch sizes must be positive")
    if args.head_hidden < 2 or args.head_batch_size <= 0:
        parser.error("head dimensions/batch size must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.max_epochs <= 0 or args.patience <= 0:
        parser.error("--max-epochs and --patience must be positive")
    if not 0.0 < args.val_fraction < 0.5:
        parser.error("--val-fraction must be in (0, 0.5)")
    if data_track_for_track(args.track) == "cdr3ab" and args.neg_source != "AS":
        parser.error("the official others/cdr3ab track supports AS negatives only")
    query_protocol = TCRBindingQueryProtocol(
        args.completion_manifest, cdr3_format=args.cdr3_format,
        max_length=args.max_length,
        input_mode=input_mode_for_track(args.track),
    )

    archive = _verify_data_archive()
    print(f"[verified] official retrain.zip: {archive}")
    frames, data_provenance = load_protocol(args.neg_source, args.track)
    if args.track == "others_longab":
        # Fail before model loading or partial output, including --audit-only.
        for pair in build_pair_universe(frames, args.track):
            query_protocol.record(*pair)
    audit = protocol_overlap_audit(frames, args.track)
    root = _experiment_root(args.tag, args.neg_source, args.track)
    _atomic_json(root / "protocol_audit.json", {
        "protocol": "nm2025_official_retrained_five_fold",
        "archive": archive,
        "overlap": audit,
        "data_provenance": data_provenance,
        "input_protocol": query_protocol.provenance,
        "input_protocol_sha256": query_protocol.sha256,
    })
    if args.audit_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    weight_path = args.checkpoint
    if weight_path.is_dir():
        weight_path = weight_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"our checkpoint is missing: {args.checkpoint}")
    checkpoint_sha = sha256_file(weight_path)
    pairs = build_pair_universe(frames, args.track)
    print(
        f"[protocol] folds=5 unique_joint_pairs={len(pairs)} "
        f"checkpoint_sha256={checkpoint_sha}"
    )
    cache_dir = args.cache_dir or (
        OUT_ROOT / "_feature_cache" / args.tag / args.track / args.neg_source
    )
    features, feature_manifest = extract_or_load_features(
        pairs=pairs,
        checkpoint=args.checkpoint,
        checkpoint_sha256=checkpoint_sha,
        cache_dir=cache_dir,
        embedding_batch_size=args.embedding_batch_size,
        feature_chunk_size=args.feature_chunk_size,
        force=args.force_features,
        query_protocol=query_protocol,
        model_assets=_model_assets(args.checkpoint),
    )
    if args.feature_only:
        print(f"[features] complete: {features.shape}")
        return

    pair_lookup = {pair: index for index, pair in enumerate(pairs)}
    folds = list(FOLDS) if args.fold == "all" else [int(args.fold)]
    eval_sets = list(EVAL_SETS) if args.eval_set == "all" else [args.eval_set]
    for fold in folds:
        run_fold(
            fold=fold,
            eval_sets=eval_sets,
            frames=frames,
            data_provenance=data_provenance,
            overlap_audit=audit,
            features=features,
            pair_lookup=pair_lookup,
            feature_manifest=feature_manifest,
            checkpoint_sha256=checkpoint_sha,
            args=args,
        )
    summary = summarize(args.tag, args.neg_source,
                        feature_identity=_feature_identity(feature_manifest), track=args.track)
    _atomic_json(root / "run_config.json", {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "tag": args.tag,
        "neg_source": args.neg_source,
        "track": args.track,
        "data_track": data_track_for_track(args.track),
        "input_mode": input_mode_for_track(args.track),
        "folds_requested": folds,
        "eval_sets_requested": eval_sets,
        "feature_manifest": feature_manifest,
        "head_config": _head_config(args, args.seed + 1),
        "paper_table_source": SUPPLEMENTARY_TABLE_7,
    })
    write_rankings(summary, args.neg_source)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
