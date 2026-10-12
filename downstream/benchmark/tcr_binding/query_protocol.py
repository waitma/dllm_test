"""Runtime-only T1 queries: v5 completion, masked recognition, observed pooling.

Used by ``run_retrained_ours.py --completion-manifest PATH --cdr3-format junction``.
Test without a checkpoint/GPU with
``pytest scripts/tests/immune_llada/test_tcr_binding_query.py`` in the pllm env.
Raw benchmark files and the shared training renderer are never modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dllm.pipelines.immune_llada.data import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.data.profiles import profile_digest
from dllm.pipelines.immune_llada.data.records import REGION_ORDER
from dllm.pipelines.immune_llada.data.sources import _receptor_chains


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPLETION_MANIFEST = (
    ROOT / "data/prepared/immune_v5_receptor_completion/dataset_manifest.json"
)
PROTOCOL_ID = "tcr_t1_runtime_v5_masked_recognition_observed_mean_v2"
POOLING = "global_mean_attention_and_residue_and_not_synthetic"
INPUT_RECORD = "peptide_mask_completed_beta_alpha_runtime"
LONG_INPUT_RECORD = "peptide_mask_provided_long_beta_alpha_runtime"
_CODE_FILES = (
    "downstream/benchmark/tcr_binding/query_protocol.py",
    "downstream/benchmark/tcr_binding/run_retrained_ours.py",
    "downstream/benchmark/tcr_binding/retrained_protocol.py",
    "downstream/benchmark/common/model_api.py",
    "dllm/pipelines/immune_llada/data/sources.py",
    "dllm/pipelines/immune_llada/data/profiles.py",
    "dllm/pipelines/immune_llada/data/records.py",
    "dllm/pipelines/immune_llada/data/grammar.py",
    "dllm/pipelines/immune_llada/data/esm_encoding.py",
    "examples/llada/load_fusion_checkpoint.py",
    "examples/llada/protein_fusion_model.py",
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _sequence(value: str, name: str) -> str:
    # Fail on missing/invalid input; do not turn NaN/None into a sequence or
    # silently substitute A. X is allowed as an observed ambiguity, not as an
    # all-X missing-value placeholder. Synthetic status is always metadata.
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty amino-acid string")
    sequence = value.strip().upper()
    if (not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWYXBZUO")
            or set(sequence) == {"X"}):
        raise ValueError(f"invalid or missing {name}: {value!r}")
    return sequence


def input_columns_for_mode(input_mode: str) -> list[str]:
    if input_mode == "longab":
        return ["Epitope", "LongB", "LongA"]
    if input_mode not in {"cdr3b", "cdr3ab"}:
        raise ValueError(f"unsupported input mode: {input_mode}")
    return ["Epitope", "CDR3B"] + (["CDR3A"] if input_mode == "cdr3ab" else [])


def validate_long_chain(sequence: str, cdr3: str, name: str) -> None:
    """Validate source columns only; CDR3 is not an additional model feature."""
    long = _sequence(sequence, name)
    junction = _sequence(cdr3, "CDR3" + name[-1])
    # find/rfind also catches overlapping matches (str.count does not).
    first = long.find(junction)
    if first < 0 or first != long.rfind(junction):
        raise ValueError(f"{name} must contain its CDR3 exactly once")


class TCRBindingQueryProtocol:
    """Label-blind beta-only or paired-CDR3 adapter; completion happens on read.

    ``cdr3_format`` is a dataset contract, never inferred from C/F characters.
    The official NM2025 CDR3B examples use junction form; other sources must
    explicitly select their convention. Completion preserves all observed aa.
    """

    def __init__(
        self, completion_manifest: str | Path = DEFAULT_COMPLETION_MANIFEST,
        *, cdr3_format: str = "junction", max_length: int = 1024,
        input_mode: str = "cdr3b",
    ):
        if cdr3_format not in {"junction", "core"}:
            raise ValueError("cdr3_format must be junction or core")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        input_columns_for_mode(input_mode)
        path = Path(completion_manifest).resolve()
        raw = path.read_bytes()
        config = json.loads(raw)["tcr_region_profile"]
        self.profile = config["profile"]
        digest = profile_digest(self.profile)
        if config.get("digest") != digest:
            raise ValueError("completion manifest profile digest mismatch")
        # Never silently fall back to unrelated region lengths at evaluation.
        for role in ("tcr_beta", "tcr_alpha"):
            regions = self.profile["region_distributions"][role]
            for region in REGION_ORDER:
                distribution = regions[region]
                if not distribution or any(
                    int(length) <= 0 or int(count) <= 0
                    for length, count in distribution.items()
                ):
                    raise ValueError(f"invalid frozen profile: {role}/{region}")
        self.cdr3_format = cdr3_format
        self.max_length = int(max_length)
        self.input_mode = input_mode
        self.provenance = {
            "protocol_id": PROTOCOL_ID,
            "input_mode": input_mode,
            "input_columns": input_columns_for_mode(input_mode),
            "ignored_others_columns": (["CDR3A"] if input_mode == "cdr3b" else [])
            + ["MHC", "TRAJ", "TRBJ", "TRBV", "TRAV", "LongB", "LongA"],
            "input_record": INPUT_RECORD,
            "pooling": POOLING,
            "recognition": "decoder_mask_only_no_gold",
            "completion": "runtime_no_prepared_benchmark_copy",
            "completion_manifest": str(path),
            "completion_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "completion_profile_sha256": digest,
            "completion_profile_seed": int(self.profile["seed"]),
            "completion_key": (
                "sha256(normalized_cdr3b)_integer" if input_mode == "cdr3b"
                else "sha256(json[normalized_cdr3b,normalized_cdr3a])_integer"
            ) + "; no_label_epitope_fold_or_row_order",
            "cdr3_format": cdr3_format,
            "max_length": self.max_length,
            "truncation": "reject_never_crop",
            "code_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in _CODE_FILES
            },
        }
        if input_mode == "longab":
            self.provenance.update({
                "protocol_id": "tcr_t1_provided_longab_masked_recognition_observed_mean_v1",
                "input_record": LONG_INPUT_RECORD,
                "ignored_others_columns": ["MHC", "TRAJ", "TRBJ", "TRBV", "TRAV"],
                "audit_only_columns": ["CDR3B", "CDR3A"],
                "source_validation": "each_cdr3_occurs_exactly_once_in_corresponding_long_chain",
                "completion": "runtime_training_fv_path_verbatim_no_region_inference_or_synthetic_residues",
                "completion_key": "unused_for_provided_full_sequences",
                "cdr3_format": "audit_only_no_cdr3_rendering",
                "sequence_scope": "dataset_provided_long_variable_region_not_verified_complete_protein",
            })
        self.sha256 = _json_hash(self.provenance)

    def record(self, peptide: str, beta: str, alpha: str | None = None) -> BioSeqRecord:
        peptide = _sequence(peptide, "Epitope")
        is_long = self.input_mode == "longab"
        beta = _sequence(beta, "LongB" if is_long else "CDR3B")
        if self.input_mode in {"cdr3ab", "longab"}:
            alpha = _sequence(alpha, "LongA" if is_long else "CDR3A")
        elif alpha is not None:
            raise ValueError("CDR3A supplied to beta-only protocol")
        # Identical receptors get identical scaffolds even with different
        # epitopes, labels, folds, batching, resume positions or input ordering.
        identity = beta if alpha is None else json.dumps([beta, alpha], separators=(",", ":"))
        key = int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big")
        chains = _receptor_chains(
            alpha_fv=alpha if is_long else "", beta_fv=beta if is_long else "",
            alpha_cdr3="" if is_long else alpha or "", beta_cdr3="" if is_long else beta,
            is_junction=self.cdr3_format == "junction", profile=self.profile,
            source="tcr_t1_runtime", row_index=key,
        )
        if is_long and [chain.sequence for chain in chains] != [beta, alpha]:
            raise ValueError("provided long chains must be retained verbatim; no completion fallback")
        # Eight grammar tokens: peptide block (3), relation (1), receptor
        # block/type/separator (4). No invisible truncation of observed residues.
        length = len(peptide) + sum(len(chain.sequence) for chain in chains) + 8
        if length > self.max_length:
            raise ValueError(f"completed T1 input length {length} > {self.max_length}")
        return BioSeqRecord(
            [BioSeqChain(peptide, "peptide"), *chains],
            task_type="tcr", source="tcr_t1_runtime",
        )

    def collate(self, records: list[BioSeqRecord], collator: Any,
                *, decoder_mask_token_id: int) -> dict[str, Any]:
        if decoder_mask_token_id is None or int(decoder_mask_token_id) < 0:
            raise ValueError("a valid decoder-space mask token is required")
        # Deliberately constrain this first adapter to peptide + beta/alpha.
        # MHC presentation edges and null contexts need separate query selection.
        for record in records:
            if record.chain_roles != ["peptide", "tcr_beta", "tcr_alpha"]:
                raise ValueError("T1 query requires peptide,beta,alpha layout")
            if record.labels or record.metadata.get("relation_supervised"):
                raise ValueError("T1 query records must not carry gold relation labels")
        batch = collator(records)
        if any(name != "tcr_peptide" for name in batch["grammar_names"]):
            raise ValueError("unexpected grammar in T1 peptide recognition query")
        query = batch["relation_token_mask"].bool() & batch["attention_mask"].bool()
        # This is the sole peptide->TCR edge only because the layout above was
        # checked. Never apply this to every relation in a pMHC record.
        if not bool(query.sum(-1).eq(1).all()):
            raise ValueError("expected exactly one recognition query per T1 row")
        batch["input_ids"] = batch["input_ids"].clone()
        batch["input_ids"][query] = int(decoder_mask_token_id)
        batch.pop("labels", None)  # collator's clean reconstruction targets are not inputs
        batch["relation_target_mask"] = query
        batch["fixed_context_mask"][query] = False
        batch["diffusion_eligible_mask"][query] = True
        batch["diffusion_loss_mask"][query] = True
        return batch


def observed_residue_pool(hidden: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    """Exclude special/pad and synthetic positions, even when synthetic X attends."""
    mask = (batch["attention_mask"].bool() & batch["residue_mask"].bool()
            & ~batch["synthetic_residue_mask"].bool())
    if not bool(mask.sum(-1).gt(0).all()):
        raise ValueError("T1 query has no observed residue to pool")
    values = hidden.float().masked_fill(~mask.unsqueeze(-1), 0.0)
    return values.sum(1) / mask.sum(1, keepdim=True)


class TCRBindingEmbedder:
    """T1-only adapter around a loaded frozen embedder; other tasks are untouched."""

    def __init__(self, backbone: Any, protocol: TCRBindingQueryProtocol):
        self.backbone = backbone
        self.protocol = protocol
        self.model = backbone.model
        self.hidden = backbone.hidden
        self.feature_source = backbone.feature_source
        self.pool_mode = backbone.pool_mode
        self.batch_size = backbone.batch_size
        self._cache: dict[tuple[str, ...], np.ndarray] = {}
        mask_id = getattr(self.model, "_decoder_mask_token_id", None)
        if mask_id is None:
            # Only legacy non-remapped grammar checkpoints use grammar IDs.
            if hasattr(backbone.collator, "lookup"):
                raise ValueError("remapped collator requires the model's decoder mask ID")
            mask_id = backbone.tokenizer.mask_token_id
        self.mask_id = int(mask_id)

    def embed_pairs(self, beta, alpha=None, peptide=None, order=None) -> np.ndarray:
        if peptide is None or len(beta) != len(peptide):
            raise ValueError("T1 requires equally sized beta and peptide inputs")
        if self.protocol.input_mode in {"cdr3ab", "longab"}:
            if alpha is None or len(alpha) != len(beta):
                raise ValueError("paired-chain track requires an alpha for every beta")
            pairs = list(zip(peptide, beta, alpha))
        else:
            if alpha is not None:
                raise ValueError("CDR3A supplied to beta-only protocol")
            pairs = list(zip(peptide, beta))
        pending = list(dict.fromkeys(pair for pair in pairs if pair not in self._cache))
        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start:start + self.batch_size]
            records = [self.protocol.record(*pair) for pair in chunk]
            batch = self.protocol.collate(
                records, self.backbone.collator, decoder_mask_token_id=self.mask_id,
            )
            batch = {key: value.to(self.backbone.device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            with torch.no_grad():
                hidden, _ = self.backbone._final_hidden(batch)
                vectors = observed_residue_pool(hidden, batch).cpu().numpy()
            self._cache.update(zip(chunk, vectors))
        return np.stack([self._cache[pair] for pair in pairs]).astype(np.float32)
