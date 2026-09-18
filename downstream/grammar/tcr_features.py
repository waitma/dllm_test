"""Runtime v5 receptor-only features for T2/T3; no epitope or binding labels.

Use ``build_embedder('tcr-v5:/absolute/fusion_checkpoint')``. CPU regression:
``python -m pytest scripts/tests/immune_llada/test_tcr_native_features.py -q``.
Original data and the shared training renderer are never rewritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dllm.pipelines.immune_llada.data import BioSeqRecord
from dllm.pipelines.immune_llada.data.sources import _receptor_chains
from downstream.benchmark.tcr_binding.query_protocol import (
    DEFAULT_COMPLETION_MANIFEST, TCRBindingQueryProtocol, _json_hash,
    _sequence, observed_residue_pool,
)


class TCRRepresentationProtocol(TCRBindingQueryProtocol):
    """Use the frozen training profile with a label-blind receptor identity."""

    def __init__(self, completion_manifest=DEFAULT_COMPLETION_MANIFEST, *,
                 cdr3_format="junction", max_length=1024, input_mode="cdr3b"):
        if input_mode not in {"cdr3b", "cdr3ab"}:
            raise ValueError("T2/T3 accept CDR3b or CDR3ab, not binding/long-chain inputs")
        super().__init__(completion_manifest, cdr3_format=cdr3_format,
                         max_length=max_length, input_mode=input_mode)
        self.provenance.update(
            protocol_id="tcr_repr_runtime_v5_null_unknown_observed_mean_v1",
            input_columns=["cdr3b"] + (["cdr3a"] if input_mode == "cdr3ab" else []),
            input_record="null_unknown_completed_beta_alpha_no_peptide",
            recognition="fixed_training_null_unknown_no_relation_target",
            missing_alpha="explicit_empty_string_becomes_fully_unknown_X_partner",
            ignored_others_columns=["peptide", "epitope", "Affinity", "MHC", "V/J"],
        )
        self.provenance["code_sha256"]["downstream/grammar/tcr_features.py"] = (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.sha256 = _json_hash(self.provenance)

    def record(self, beta: str, alpha: str | None = None) -> BioSeqRecord:
        beta = _sequence(beta, "CDR3B")
        if self.input_mode == "cdr3ab":
            alpha = "" if alpha == "" else _sequence(alpha, "CDR3A")
        elif alpha is not None:
            raise ValueError("alpha supplied to beta-only representation")
        identity = beta if alpha is None else json.dumps([beta, alpha], separators=(",", ":"))
        key = int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big")
        chains = _receptor_chains(
            alpha_fv="", beta_fv="", alpha_cdr3=alpha or "", beta_cdr3=beta,
            is_junction=self.cdr3_format == "junction", profile=self.profile,
            source="tcr_repr_runtime", row_index=key,
        )
        if sum(len(c.sequence) for c in chains) + 8 > self.max_length:
            raise ValueError("completed receptor exceeds max_length; truncation forbidden")
        return BioSeqRecord(chains, task_type="tcr", source="tcr_repr_runtime")

    def collate(self, records, collator):
        for record in records:
            if record.chain_roles != ["tcr_beta", "tcr_alpha"] or record.labels:
                raise ValueError("receptor-only paired layout required; no labels or peptide")
        batch = collator(records)
        if any(name != "tcr_pair" for name in batch["grammar_names"]):
            raise ValueError("unexpected receptor representation grammar")
        if bool(batch["relation_target_mask"].any()):
            raise ValueError("T2/T3 must not carry a relation target")
        batch.pop("labels", None)
        return batch


class TCRRuntimeEmbedder:
    """Frozen decoder global observed-residue mean; no persistent legacy cache."""

    def __init__(self, checkpoint_dir=None, *, backbone=None, device=None,
                 batch_size=8, max_length=1024,
                 completion_manifest=DEFAULT_COMPLETION_MANIFEST, cdr3_format="junction"):
        if backbone is None:
            from downstream.benchmark.common.model_api import FusionGrammarEmbedder
            backbone = FusionGrammarEmbedder(str(checkpoint_dir), device=device,
                                             batch_size=batch_size, max_length=max_length)
        self.backbone = backbone
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.feature_source, self.pool_mode = "decoder", "global"
        self.dim = self.hidden = int(backbone.hidden)
        self.name = f"{backbone.name}-tcr-runtime-v5-observed"
        self.protocols = {mode: TCRRepresentationProtocol(
            completion_manifest, cdr3_format=cdr3_format,
            max_length=max_length, input_mode=mode) for mode in ("cdr3b", "cdr3ab")}
        self._load_info = {"input_protocols": {
            mode: {"sha256": p.sha256, **p.provenance} for mode, p in self.protocols.items()}}
        self._cache = {}

    def embed_pairs(self, beta, alpha=None, peptide=None, order=None):
        if peptide is not None:
            raise ValueError("T2/T3 embedder rejects epitope inputs")
        if order not in (None, ("beta",), ("beta", "alpha")):
            raise ValueError("native TCR order is beta then alpha")
        beta = list(beta)
        if alpha is not None and len(alpha) != len(beta):
            raise ValueError("alpha/beta lengths differ")
        mode = "cdr3b" if alpha is None else "cdr3ab"
        protocol = self.protocols[mode]
        keys = [(_sequence(b, "CDR3B"), None) for b in beta] if alpha is None else [
            (_sequence(b, "CDR3B"), "" if a == "" else _sequence(a, "CDR3A"))
            for b, a in zip(beta, alpha)]
        pending = list(dict.fromkeys(key for key in keys if key not in self._cache))
        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start:start + self.batch_size]
            records = [protocol.record(*key) for key in chunk]
            batch = protocol.collate(records, self.backbone.collator)
            batch = {k: v.to(self.backbone.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.no_grad():
                hidden, _ = self.backbone._final_hidden(batch)
                vectors = observed_residue_pool(hidden, batch).cpu().numpy()
            if not np.isfinite(vectors).all():
                raise ValueError("non-finite TCR features")
            self._cache.update(zip(chunk, vectors))
        return np.stack([self._cache[key] for key in keys]).astype(np.float32) if keys else (
            np.empty((0, self.dim), dtype=np.float32))

    def embed(self, seqs, batch_size=None):
        if batch_size is not None and int(batch_size) != self.batch_size:
            raise ValueError("set batch_size at construction for reproducible protocol")
        return self.embed_pairs(seqs)
