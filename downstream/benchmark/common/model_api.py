"""Model wrapper interface for IRBench.

Every model that wants a leaderboard number implements one tiny interface:

    class SequenceEmbedder:
        name: str
        dim: int
        def embed(self, seqs: list[str]) -> np.ndarray   # [N, dim], L2-agnostic

Tasks build features by embedding whichever columns they need (CDR3b, CDR3a,
peptide, full chains, ...) and concatenating. This decouples the benchmark from
any particular backbone.

Adapters provided:
- ``ESM2Embedder``       local ESM2 snapshot via HuggingFace transformers (the
                         always-available reference protein LM).
- ``OphiuchusEmbedder``  the migrated Ophiuchus-Ab backbone (antibody weights;
                         a placeholder until the BioSeq immune model is trained).
- ``BioSeqEmbedder``     the trained diffusion immune-receptor foundation model;
                         loads a backbone state-dict produced by the bioseq DDP
                         trainer. **This is the single class to wire up once the
                         foundation model finishes training.**

Heavy deps (torch / transformers / the bioseq package) are imported lazily so
importing this module never forces a GPU stack to load.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

ESM2_PATHS = {
    "esm2_8m": "/c20250601/mj/model_weights/esm2/esm2_t6_8M_UR50D",
    "esm2_35m": "/c20250601/mj/model_weights/esm2/esm2_t12_35M_UR50D",
    "esm2_150m": "/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D",
    "esm2_650m": "/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D",
    "esm2_3b": "/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D",
}


class SequenceEmbedder:
    """Abstract embedder. Subclasses implement ``embed``."""

    name: str = "base"
    dim: int = 0

    def embed(self, seqs: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class ESM2Embedder(SequenceEmbedder):
    def __init__(self, model_key: str = "esm2_150m", device: str | None = None,
                 batch_size: int = 64, max_length: int = 512, layer: int = -1):
        import torch
        from transformers import AutoTokenizer, AutoModel

        path = ESM2_PATHS.get(model_key, model_key)
        self.name = f"ESM2-{model_key}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.layer = layer
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)

    @property
    def _cache_key(self) -> str:
        return self.name

    def embed(self, seqs: Sequence[str]) -> np.ndarray:
        torch = self._torch
        seqs = ["" if s is None else str(s) for s in seqs]
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size):
                chunk = seqs[i:i + self.batch_size]
                # ESM2 tokenizer rejects empty strings; substitute a single 'A'.
                chunk = [s if len(s) > 0 else "A" for s in chunk]
                enc = self.tokenizer(
                    chunk, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hs = self.model(**enc).last_hidden_state          # [B, L, D]
                mask = enc["attention_mask"].unsqueeze(-1).float()
                # mean-pool over real residues (exclude padding; keep cls/eos
                # contribution negligible relative to sequence length).
                pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1.0)
                out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim))


class OphiuchusEmbedder(SequenceEmbedder):
    """Two-chain Ophiuchus backbone embedder (antibody weights, placeholder).

    ``embed`` treats each sequence as a single chain. Use ``embed_pairs`` for
    true paired alpha/beta embedding.
    """

    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        import sys
        from pathlib import Path as _P
        # ``embeddings`` lives in downstream/, the dllm package in dllm_test/.
        downstream_dir = _P(__file__).resolve().parents[2]      # .../downstream
        dllm_test_dir = downstream_dir.parent                   # .../dllm_test
        for p in (str(downstream_dir), str(dllm_test_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import torch
        from embeddings import OphiuchusEmbeddingModel, OphiuchusEmbeddingConfig  # noqa
        from dllm.pipelines.bioseq import Esm2ProteinTokenizer, OPHIUCHUS_AB_CHAIN_LENGTHS

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "Ophiuchus-Ab"
        self.tokenizer = Esm2ProteinTokenizer()
        self.chain_lengths = OPHIUCHUS_AB_CHAIN_LENGTHS
        self.model = OphiuchusEmbeddingModel(
            checkpoint_path=checkpoint_path,
            config=OphiuchusEmbeddingConfig(sep_chains=False),
            device=self.device,
        ).eval()
        self.dim = self.model.config.hidden_size

    def _encode_chain(self, seq: str, chain_index: int):
        torch = self._torch
        max_len = self.chain_lengths[chain_index]
        encoded = self.tokenizer.encode((seq or "A").replace("J", "L"))
        tok = torch.full((max_len,), self.tokenizer.eos_token_id, dtype=torch.long)
        n = min(len(encoded), max_len)
        tok[:n] = torch.tensor(encoded[:n], dtype=torch.long)
        return tok

    def embed_pairs(self, chain1: Sequence[str], chain2: Sequence[str]) -> np.ndarray:
        torch = self._torch
        out = []
        bs = 32
        with torch.no_grad():
            for i in range(0, len(chain1), bs):
                c1 = chain1[i:i + bs]
                c2 = chain2[i:i + bs]
                h = torch.stack([self._encode_chain(s, 0) for s in c1])
                l = torch.stack([self._encode_chain(s, 1) for s in c2])
                chains = torch.cat([h, l], dim=-1).to(self.device)
                chain_ids = torch.cat([
                    torch.zeros_like(h), torch.ones_like(l)
                ], dim=-1).to(self.device)
                emb = self.model(chains, chain_ids)
                out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim))

    def embed(self, seqs: Sequence[str]) -> np.ndarray:
        placeholder = ["" for _ in seqs]
        return self.embed_pairs(list(seqs), placeholder)


class BioSeqEmbedder(OphiuchusEmbedder):
    """Trained diffusion immune-receptor foundation model embedder.

    Loads a bioseq DDP checkpoint (``backbone_state_dict``), a generic
    ``state_dict`` checkpoint, or a plain backbone state-dict into the Ophiuchus
    backbone architecture, then mean-pools the final representation layer.

    This is the wiring point for the foundation model: once training finishes,
    point ``state_dict_path`` at the checkpoint and register it in ``build``.
    """

    def __init__(self, state_dict_path: str, device: str | None = None):
        import torch
        super().__init__(checkpoint_path=None, device=device)
        self.name = "BioSeq-Immune"
        payload = torch.load(state_dict_path, map_location=self.device)
        if isinstance(payload, dict) and "backbone_state_dict" in payload:
            sd = payload["backbone_state_dict"]
            checkpoint_format = "backbone_state_dict"
        elif isinstance(payload, dict) and "state_dict" in payload:
            sd = payload["state_dict"]
            checkpoint_format = "state_dict"
        else:
            sd = payload
            checkpoint_format = "plain_state_dict"
        # The trainer saves the mint ESM2 backbone under ``model.net``; the
        # embedding model exposes the same module as ``self.model.model``.
        missing, unexpected = self.model.model.load_state_dict(sd, strict=False)
        self._load_info = {
            "checkpoint_format": checkpoint_format,
            "missing": len(missing),
            "unexpected": len(unexpected),
        }


def build_embedder(spec: str, **kwargs) -> SequenceEmbedder:
    """Factory: ``esm2_150m`` / ``ophiuchus`` / ``bioseq:/path/to/final.pt``."""
    if spec.startswith("bioseq:"):
        return BioSeqEmbedder(state_dict_path=spec.split(":", 1)[1], **kwargs)
    if spec == "ophiuchus":
        return OphiuchusEmbedder(**kwargs)
    if spec.startswith("esm2"):
        return ESM2Embedder(model_key=spec, **kwargs)
    raise ValueError(f"unknown embedder spec: {spec}")


__all__ = [
    "SequenceEmbedder", "ESM2Embedder", "OphiuchusEmbedder",
    "BioSeqEmbedder", "build_embedder", "ESM2_PATHS",
]
