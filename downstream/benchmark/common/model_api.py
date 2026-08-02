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

# Residues the grammar single-chain renderer accepts (standard 20 + X + the
# BJOUZ ambiguity codes). We drop the "." chain separator and "-" gap so a
# single-chain post-LLaDA record never accidentally splits into extra chains.
_GRAMMAR_VALID_CHARS = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


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

    def embed_residues(self, seqs: Sequence[str]) -> list[np.ndarray]:
        """Per-residue contextual representations.

        Returns one ``[L_i, dim]`` array per input sequence, aligned to the
        original residues (``<cls>``/``<eos>`` stripped). ``L_i`` reflects the
        post-truncation residue count (``<= max_length - 2``), so callers must
        align position-level labels with ``min(len(seq), L_i)``.
        """
        torch = self._torch
        seqs = ["" if s is None else str(s) for s in seqs]
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size):
                chunk = [s if len(s) > 0 else "A" for s in seqs[i:i + self.batch_size]]
                enc = self.tokenizer(
                    chunk, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hs = self.model(**enc).last_hidden_state.float().cpu().numpy()
                mask = enc["attention_mask"].cpu().numpy()
                for b in range(len(chunk)):
                    n_real = int(mask[b].sum())          # cls + residues + eos
                    out.append(hs[b, 1:max(1, n_real - 1)])
        return out


# ---------------------------------------------------------------------------
# Antibody / nanobody-specific language models (NbBench baselines).
#
# Every entry below is a *public* HuggingFace masked-LM checkpoint. We load the
# bare encoder via ``AutoModel`` and mean-pool its final hidden state, exactly
# mirroring ``ESM2Embedder`` so the leaderboard compares like-for-like (frozen
# backbone, head-only probe). Weights are fetched on first use (set
# ``HF_ENDPOINT=https://hf-mirror.com`` behind the GFW); nothing is bundled.
#
# ``space``: these tokenizers expect **space-separated residues** ("E V Q L ...")
# so each amino acid maps to one token. NanoBERT ships a char-level tokenizer and
# works on the raw string, so it is the sole ``space=False`` entry.
# ---------------------------------------------------------------------------
ANTIBODY_LM_SPECS: dict[str, dict] = {
    "protbert": {"repo": "Rostlab/prot_bert", "name": "ProtBERT", "space": True},
    "igbert": {"repo": "Exscientia/IgBert", "name": "IgBERT", "space": True},
    "ablang_h": {"repo": "qilowoq/AbLang_heavy", "name": "AbLang-H", "space": True},
    "antiberta2": {"repo": "alchemab/antiberta2", "name": "AntiBERTa2", "space": True},
    "antiberta2_cssp": {"repo": "alchemab/antiberta2-cssp",
                        "name": "AntiBERTa2-CSSP", "space": True},
    "nanobert": {"repo": "NaturalAntibody/nanoBERT", "name": "NanoBERT", "space": False},
    "vhhbert": {"repo": "COGNANO/VHHBERT", "name": "VHHBERT", "space": True},
}


class HFProteinLMEmbedder(SequenceEmbedder):
    """Generic frozen HuggingFace masked-LM embedder (mean-pool of final layer).

    Handles the antibody/nanobody PALMs used by NbBench (VHHBERT, NanoBERT,
    AbLang-H, AntiBERTa2(-CSSP), IgBERT, ProtBERT). Position-limited RoBERTa/
    RoFormer backbones are truncated to their ``max_position_embeddings`` (minus
    special-token margin), so long antigens (e.g. SARS spike ~1.2k aa) are
    clipped -- a real constraint of these small-context PALMs, matching the
    official NbBench setup.
    """

    def __init__(self, repo: str, name: str | None = None, space_residues: bool = True,
                 trust_remote_code: bool = True, device: str | None = None,
                 batch_size: int = 32, max_length: int | None = None):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self._torch = torch
        self.repo = repo
        self.name = name or repo.split("/")[-1]
        self.space = bool(space_residues)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo, trust_remote_code=trust_remote_code, do_lower_case=False)
        self.model = AutoModel.from_pretrained(
            repo, trust_remote_code=trust_remote_code).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)
        cfg = self.model.config
        mpe = int(getattr(cfg, "max_position_embeddings", 1024) or 1024)
        # RoBERTa builds position ids as ``padding_idx + 1 + cumsum(mask)`` so the
        # largest position id is ``padding_idx + n_tokens``; the usable token count
        # is therefore ``max_position_embeddings - padding_idx - 1`` (AbLang-H uses
        # padding_idx=21 with mpe=160 -> only ~137 tokens, else the position
        # embedding gather asserts out-of-bounds). BERT/RoFormer have no such offset.
        model_type = str(getattr(cfg, "model_type", "")).lower()
        pad_idx = int(getattr(cfg, "pad_token_id", 0) or 0)
        if model_type == "roberta":
            cap = mpe - pad_idx - 2
        else:
            cap = mpe - 2
        cap = max(16, cap)
        self.max_length = min(max_length if max_length is not None else 1024, cap)
        # Some checkpoints (e.g. AbLang-H) ship a tokenizer whose vocab (incl.
        # ``[UNK]``) is larger than the model's embedding table, so a non-standard
        # residue tokenised to ``[UNK]`` would index out of bounds on the GPU. We
        # (a) fold BJOUZ ambiguity codes to canonical residues before tokenising,
        # and (b) clamp any surviving out-of-range id to a safe in-vocab token.
        self._vocab_limit = int(self.model.get_input_embeddings().weight.shape[0])
        unk = getattr(self.tokenizer, "unk_token_id", None)
        self._safe_id = int(unk) if (unk is not None and unk < self._vocab_limit) else 0

    # Standard ambiguity-code folding (Asx->N, Glx->Q, Xle->L, Sec->C, Pyl->K).
    _AMBIG = str.maketrans({"B": "N", "Z": "Q", "J": "L", "U": "C", "O": "K"})
    _STD = set("ACDEFGHIKLMNPQRSTVWY")

    def _prep(self, seqs: Sequence[str]) -> list[str]:
        # Fold ambiguity codes, then map any *other* non-standard residue -- most
        # importantly the CDR-infilling gap ``-`` and ``X`` -- to a canonical 'A'
        # placeholder. This is REQUIRED for residue-level alignment: NanoBERT's
        # char-level tokenizer silently *drops* characters outside its residue
        # vocab (so ``EE-EE`` -> 4 tokens, breaking per-position label mapping),
        # whereas keeping every residue as one standard-AA token guarantees
        # one-token-per-residue for every tokenizer here.
        out = []
        for s in seqs:
            s = "" if s is None else str(s).upper().translate(self._AMBIG)
            s = "".join(c if c in self._STD else "A" for c in s)
            if len(s) == 0:
                s = "A"
            out.append(" ".join(list(s)) if self.space else s)
        return out

    def _tokenize(self, chunk: list[str]):
        enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                             truncation=True, max_length=self.max_length)
        ids = enc["input_ids"]
        if self._vocab_limit:
            ids = ids.masked_fill(ids >= self._vocab_limit, self._safe_id)
            enc["input_ids"] = ids
        return enc

    def embed(self, seqs: Sequence[str]) -> np.ndarray:
        torch = self._torch
        prepped = self._prep(seqs)
        out = []
        with torch.no_grad():
            for i in range(0, len(prepped), self.batch_size):
                chunk = prepped[i:i + self.batch_size]
                enc = self._tokenize(chunk)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hs = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1.0)
                out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim))

    def embed_residues(self, seqs: Sequence[str]) -> list[np.ndarray]:
        torch = self._torch
        prepped = self._prep(seqs)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(prepped), self.batch_size):
                chunk = prepped[i:i + self.batch_size]
                enc = self._tokenize(chunk)
                enc_d = {k: v.to(self.device) for k, v in enc.items()}
                hs = self.model(**enc_d).last_hidden_state.float().cpu().numpy()
                mask = enc["attention_mask"].cpu().numpy()
                for b in range(len(chunk)):
                    n_real = int(mask[b].sum())          # [CLS] + residues + [SEP]
                    out.append(hs[b, 1:max(1, n_real - 1)])
        return out


class ProtBertEmbedder(HFProteinLMEmbedder):
    """ProtBert (Elnaggar et al. 2021) via official HuggingFace weights.

    ProtBert is a BERT pretrained by masked-LM on UniRef100/BFD
    (`Rostlab/prot_bert <https://huggingface.co/Rostlab/prot_bert>`_). It is the
    "generic protein PLM" baseline in the SCEPTR (Cell Systems 2024) Table SI.
    Inputs are space-separated amino acids ("M K T F ...", one token per
    residue); we mean-pool the final hidden state over real tokens, mirroring
    :class:`ESM2Embedder`. Equivalent to ``build_embedder("protbert")`` via
    :data:`ANTIBODY_LM_SPECS`; exposed as a named class for the T3 TCR-representation
    baseline (encodes CDR3b+CDR3a like the other embedders).
    """

    def __init__(self, repo: str = "Rostlab/prot_bert", name: str = "ProtBERT",
                 **kwargs):
        super().__init__(repo=repo, name=name, space_residues=True, **kwargs)


class TcrBertEmbedder(HFProteinLMEmbedder):
    """TCR-BERT (Wu et al. 2021) via official HuggingFace weights.

    TCR-BERT is a BERT pretrained by masked-amino-acid modelling on ~90k human
    TRB/TRA CDR3 sequences (`wukevin/tcr-bert <https://huggingface.co/wukevin/tcr-bert>`_).
    We load the **MLM-only** checkpoint ``wukevin/tcr-bert-mlm-only`` -- the
    purely self-supervised pretrained encoder, which is the fair analogue of
    ESM2/ProtBert here. The sibling ``wukevin/tcr-bert`` is *additionally*
    fine-tuned to classify PIRD antigen labels, so using its encoder on this
    epitope-specificity task would leak epitope supervision (circular); the
    MLM-only weights keep the baseline honestly self-supervised.

    Inputs are space-separated amino acids ("C A S S ...", one token per
    residue, matching the official tokenizer); we mean-pool the final hidden
    state over real tokens, exactly mirroring :class:`ESM2Embedder` /
    :class:`HFProteinLMEmbedder`. The BERT context is 64 positions -- ample for
    CDR3 loops (TCR-BERT is applied per-chain to CDR3, so ``--columns cdr3b`` is
    the intended usage).
    """

    def __init__(self, repo: str = "wukevin/tcr-bert-mlm-only",
                 name: str = "TCR-BERT", **kwargs):
        super().__init__(repo=repo, name=name, space_residues=True, **kwargs)


class AntiBERTyEmbedder(SequenceEmbedder):
    """AntiBERTy (Ruffolo et al.) via the official ``antiberty`` package.

    The package bundles the weights, so no HuggingFace download is needed. Its
    ``AntiBERTyRunner.embed`` returns per-residue hidden states (incl. [CLS]/
    [SEP]); we strip the specials and mean-pool. Inputs are truncated to 510
    residues (BERT 512 position limit) so long antigens do not overflow.
    """

    def __init__(self, device: str | None = None, batch_size: int = 32,
                 max_residues: int = 510):
        import torch
        from antiberty import AntiBERTyRunner

        self._torch = torch
        self.runner = AntiBERTyRunner()
        self.name = "AntiBERTy"
        self.batch_size = batch_size
        self.max_residues = int(max_residues)
        self.dim = int(self.runner.model.config.hidden_size)

    def _prep(self, seqs: Sequence[str]) -> list[str]:
        out = []
        for s in seqs:
            s = "" if s is None else str(s).upper()
            if len(s) == 0:
                s = "A"
            out.append(s[:self.max_residues])
        return out

    def _residue_arrays(self, seqs: Sequence[str]) -> list[np.ndarray]:
        prepped = self._prep(seqs)
        res: list[np.ndarray] = []
        for i in range(0, len(prepped), self.batch_size):
            embs = self.runner.embed(prepped[i:i + self.batch_size])
            for e in embs:
                e = e[1:-1] if e.shape[0] > 2 else e     # strip [CLS]/[SEP]
                res.append(e.float().cpu().numpy())
        return res

    def embed(self, seqs: Sequence[str]) -> np.ndarray:
        arrs = self._residue_arrays(seqs)
        if not arrs:
            return np.zeros((0, self.dim))
        return np.stack([a.mean(axis=0) if len(a) else np.zeros(self.dim) for a in arrs])

    def embed_residues(self, seqs: Sequence[str]) -> list[np.ndarray]:
        return self._residue_arrays(seqs)


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


class OphiuchusBioSeqEmbedder(OphiuchusEmbedder):
    """Legacy bioseq embedder: loads a checkpoint into the Ophiuchus backbone.

    Handles the older ``backbone_state_dict`` / ``state_dict`` / plain-state-dict
    checkpoints that share the Ophiuchus two-chain architecture. For the newer
    ESMC-based ``grammar_v2`` diffusion checkpoints use :class:`EsmcBioSeqEmbedder`
    (``build_embedder`` auto-detects and routes).
    """

    def __init__(self, state_dict_path: str, device: str | None = None):
        import torch
        super().__init__(checkpoint_path=None, device=device)
        self.name = "BioSeq-Immune"
        payload = torch.load(state_dict_path, map_location=self.device)
        if isinstance(payload, dict) and "backbone_state_dict" in payload:
            sd, checkpoint_format = payload["backbone_state_dict"], "backbone_state_dict"
        elif isinstance(payload, dict) and "state_dict" in payload:
            sd, checkpoint_format = payload["state_dict"], "state_dict"
        else:
            sd, checkpoint_format = payload, "plain_state_dict"
        missing, unexpected = self.model.model.load_state_dict(sd, strict=False)
        self._load_info = {
            "checkpoint_format": checkpoint_format,
            "missing": len(missing),
            "unexpected": len(unexpected),
        }


class EsmcBioSeqEmbedder(SequenceEmbedder):
    """Our trained ``grammar_v2`` diffusion model, via its ESMC-* encoder.

    The ``grammar_v2_*_llada`` checkpoints are ESMC-300M/600M encoders fine-tuned
    inside a LLaDA masked-diffusion objective (keys ``encoder.esmc.*`` +
    ``decoder.*`` + ``condition_norm``). For representation extraction we load the
    **fine-tuned encoder weights** into a standalone ESMC backbone (0 missing / 0
    unexpected keys -- verified), run each amino-acid sequence through it, and
    mean-pool the final hidden states over real residues (mirroring how ESM2 is
    used elsewhere in the benchmark). The ``decoder`` / ``condition_norm`` modules
    are only needed for generation and are ignored here.
    """

    def __init__(self, state_dict_path: str, device: str | None = None,
                 payload=None, batch_size: int = 64, max_length: int = 512):
        import sys as _sys
        from pathlib import Path as _P
        import torch

        dllm_test_dir = _P(__file__).resolve().parents[3]      # .../dllm_test
        if str(dllm_test_dir) not in _sys.path:
            _sys.path.insert(0, str(dllm_test_dir))
        from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import load_local_esmc_encoder
        from dllm.pipelines.qwen3_vl_arch.data import HuggingFaceEsmTokenizerAdapter

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        payload = payload if payload is not None else torch.load(
            state_dict_path, map_location="cpu", weights_only=False)
        args = payload.get("args", {})
        encoder_path = args.get("encoder_path") or args.get("tokenizer_path")
        if encoder_path is None:
            raise ValueError("checkpoint args lack encoder_path for ESMC backbone")
        run_name = _P(state_dict_path).parent.name
        self.name = f"BioSeq-{run_name}" if run_name else "BioSeq-ESMC"

        self.tokenizer = HuggingFaceEsmTokenizerAdapter.from_pretrained(encoder_path)
        self.pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        encoder = load_local_esmc_encoder(encoder_path)
        enc_sd = {k[len("encoder.esmc."):]: v
                  for k, v in payload["model_state_dict"].items()
                  if k.startswith("encoder.esmc.")}
        missing, unexpected = encoder.esmc.load_state_dict(enc_sd, strict=False)
        self._load_info = {
            "checkpoint_format": "grammar_v2_esmc",
            "encoder_path": encoder_path,
            "n_encoder_keys": len(enc_sd),
            "missing": len(missing),
            "unexpected": len(unexpected),
        }
        self.model = encoder.to(self.device).eval()
        with torch.no_grad():
            probe = self._forward_batch(["A"])
        self.dim = int(probe.shape[1])

    def _forward_hidden(self, seqs):
        """Run one batch through the ESMC encoder.

        Returns ``(hs, resm)`` where ``hs`` is ``[B, L, D]`` float last-hidden-
        states and ``resm`` is a ``[B, L]`` float residue mask (1 at residue
        columns, 0 at ``<cls>``/``<eos>``/pad). The grammar ESMC tokenizer emits
        exactly one token per residue -- including the CDR-infilling gap ``-``
        (mapped to a real residue token) -- so residue columns align 1:1 with the
        input string characters, which per-residue probing relies on.
        """
        torch = self._torch
        toks, res_masks = [], []
        for s in seqs:
            s = s if (s and len(s) > 0) else "A"
            ids, rm = self.tokenizer.encode_chain(s, max_length=self.max_length)
            toks.append(ids)
            res_masks.append(rm)
        L = max(len(t) for t in toks)
        input_ids = torch.full((len(toks), L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(toks), L), dtype=torch.long)
        resm = torch.zeros((len(toks), L), dtype=torch.float32)
        for i, (t, rm) in enumerate(zip(toks, res_masks)):
            input_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            attn[i, :len(t)] = 1
            resm[i, :len(rm)] = torch.tensor(rm, dtype=torch.float32)
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)
        resm = resm.to(self.device)
        out = self.model(input_ids=input_ids, attention_mask=attn)
        return out.last_hidden_state.float(), resm

    def _forward_batch(self, seqs):
        torch = self._torch
        hs, resm = self._forward_hidden(seqs)
        resm3 = resm.unsqueeze(-1)
        pooled = (hs * resm3).sum(1) / resm3.sum(1).clamp_min(1.0)
        return pooled.cpu().numpy()

    def embed(self, seqs: Sequence[str]) -> np.ndarray:
        torch = self._torch
        seqs = ["" if s is None else str(s) for s in seqs]
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size):
                out.append(self._forward_batch(seqs[i:i + self.batch_size]))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim))

    def embed_residues(self, seqs: Sequence[str]) -> list[np.ndarray]:
        """Per-residue ESMC representations, one ``[L_i, dim]`` array per input.

        Aligned to the original residues (``<cls>``/``<eos>`` stripped via the
        residue mask), mirroring :meth:`ESM2Embedder.embed_residues` so the
        position-level ``run_residue.py`` probe works with ``bioseq:`` specs.
        """
        torch = self._torch
        seqs = ["" if s is None else str(s) for s in seqs]
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch_size):
                hs, resm = self._forward_hidden(seqs[i:i + self.batch_size])
                hs_np = hs.cpu().numpy()
                m_np = resm.cpu().numpy().astype(bool)
                for b in range(hs_np.shape[0]):
                    out.append(hs_np[b, m_np[b]])
        return out


class GrammarEmbedder(SequenceEmbedder):
    """Grammar-contextual embedder for the ``grammar_v2`` diffusion model.

    Unlike :class:`EsmcBioSeqEmbedder` (which runs each chain through the ESMC
    encoder in isolation), this embedder renders a *joint* ``task_type=tcr``
    grammar record — peptide + TCR-beta (+TCR-alpha) with the ``<binding>``
    relation — and runs the full pretrained pipeline (ESMC encoder condition ->
    LLaDA denoiser) on a **clean, unmasked** input. The per-residue
    representations are therefore contextualised across chains (the TCR "sees"
    its epitope), which is exactly the binding signal the model was pretrained
    on. Two feature sources are exposed:

    - ``feature_source="decoder"`` (default): final LLaDA hidden states, **one
      global mean-pool** over all residue positions in the joint record (our-model
      downstream headline口径).
    - ``feature_source="encoder"``: the gathered per-residue ESMC condition,
      mean-pooled per segment — a per-chain (contextless) baseline.

    ``embed_pairs(beta, alpha, peptide)`` returns ``[N, dim]``:

    - ``pool_mode="global"`` (default): one mean-pool over **all** residue tokens
      in the joint grammar record (``tcr_peptide`` / ``tcr_pair`` / …) after the
      full post-LLaDA forward → a single ``[hidden_dim]`` vector per sample.
    - ``pool_mode="segment_concat"`` (deprecated ablation only): per-chain segment
      mean-pool then concat ``[beta, (alpha), peptide]`` — never the leaderboard
      default (spec ``grammar:decoder:segment:/abs/best.pt``).

    ``embed(seqs)`` (the generic per-column ``SequenceEmbedder`` interface used by
    the TCR runners) honours ``feature_source``:

    - ``feature_source="decoder"`` (default, **post-LLaDA**): each sequence is
      rendered as a *single-chain* ``task_type="tcr"`` grammar record
      (``<prots><tcr> SEQ <protd>``, grammar_name ``tcr_single``), run through the
      clean/unmasked LLaDA decoder (``_final_hidden``), and its final decoder
      hidden states are mean-pooled over the residue positions. This is the
      representation口径 required for evaluating OUR model on downstream tasks:
      the features have passed through the trained LLaDA decoder, not just the
      frozen ESMC encoder.
    - ``feature_source="encoder"``: each sequence is run through the ESMC encoder
      only (``last_hidden_state`` mean-pool) — the contextless per-chain baseline,
      kept for comparison but NOT the our-model口径.
    """

    _POOL_MODES = ("global", "segment_concat")

    def __init__(self, state_dict_path: str, device: str | None = None,
                 feature_source: str = "decoder", pool_mode: str = "global",
                 pair_pooling: str | None = None,
                 batch_size: int = 32, max_length: int = 512):
        import sys as _sys
        from pathlib import Path as _P
        import torch

        dllm_test_dir = _P(__file__).resolve().parents[3]      # .../dllm_test
        downstream_dir = _P(__file__).resolve().parents[2]     # .../downstream
        for p in (str(dllm_test_dir), str(downstream_dir)):
            if p not in _sys.path:
                _sys.path.insert(0, p)
        from grammar.common import (
            load_grammar_checkpoint, build_grammar_tokenizer, build_grammar_collator,
        )
        from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, BioSeqRecord

        self._torch = torch
        self._Chain = BioSeqChain
        self._Record = BioSeqRecord
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if feature_source not in ("decoder", "encoder"):
            raise ValueError(f"feature_source must be decoder|encoder, got {feature_source}")
        if pair_pooling is not None:
            pool_mode = ("segment_concat" if pair_pooling == "segment"
                         else "global" if pair_pooling == "sequence" else pool_mode)
        if pool_mode not in self._POOL_MODES:
            raise ValueError(f"pool_mode must be global|segment_concat, got {pool_mode}")
        self.feature_source = feature_source
        self.pool_mode = pool_mode
        self.pair_pooling = "sequence" if pool_mode == "global" else "segment"
        self.batch_size = batch_size
        self.max_length = max_length

        self._state_dict_path = state_dict_path
        _payload = torch.load(state_dict_path, map_location="cpu", weights_only=False)
        _args = _payload.get("args", {})
        self._encoder_path = _args.get("encoder_path") or _args.get("tokenizer_path")
        del _payload
        self.model, self.tokenizer = load_grammar_checkpoint(state_dict_path, device=self.device)
        self.model.eval()
        self.collator = build_grammar_collator(self.tokenizer)
        run_name = _P(state_dict_path).parent.name
        pool_tag = "" if pool_mode == "global" else "-segconcat"
        self.name = f"Grammar-{run_name}-{feature_source}{pool_tag}"
        self.hidden = int(self.model.config.hidden_size)
        self.enc_hidden = int(self.model.encoder.config.hidden_size)
        # segment feature dim (per segment); dim is set per embed_pairs call.
        self.dim = self.hidden if feature_source == "decoder" else self.enc_hidden
        self._cache: dict[tuple, dict | np.ndarray] = {}
        self._single_cache: dict[str, np.ndarray] = {}

    # -- record construction -------------------------------------------------
    def _record(self, beta: str, alpha: str | None, peptide: str | None = None):
        """Render a joint ``task_type=tcr`` grammar record for embed_pairs.

        With ``peptide`` set → ``tcr_peptide`` (β [+ α] + epitope, binding task).
        With ``peptide=None`` → ``tcr_pair`` (β+α, no epitope; T3 representation).
        Chains carry explicit roles; the renderer picks peptide / α / β by role.
        """
        beta = self._sanitize_chain(beta or "A")
        alpha = self._sanitize_chain(alpha) if (alpha or "").strip() else ""
        pep = self._sanitize_chain(peptide) if (peptide or "").strip() else ""
        chains = []
        if pep:
            chains.append(self._Chain(pep, "peptide"))
        chains.append(self._Chain(beta, "tcr_beta"))
        if alpha:
            chains.append(self._Chain(alpha, "tcr_alpha"))
        return (self._Record(chains=chains, task_type="tcr", source="grammar_embed"),
                bool(alpha), bool(pep))

    @staticmethod
    def _segment_slots(has_alpha: bool, has_peptide: bool = False) -> dict:
        """position_ids_chain slot -> segment name (legacy ``segment_concat`` only).

        Peptide-free paired TCR: α=slot 0, β=slot 1 (``tcr_pair``); β-only → slot 0.
        With peptide (``tcr_peptide``): slot 0=peptide, receptor slots follow.
        """
        if has_peptide:
            if has_alpha:
                return {0: "peptide", 1: "alpha", 2: "beta"}
            return {0: "peptide", 1: "beta"}
        if has_alpha:
            return {0: "alpha", 1: "beta"}
        return {0: "beta"}

    # -- clean forward -------------------------------------------------------
    def _final_hidden(self, batch):
        """Reproduce the LLaDA wrapper forward (clean, unmasked) and return
        (decoder_hidden [B,S,H], encoder_condition [B,S,E])."""
        torch = self._torch
        m = self.model
        chain_token_condition = m.encode_chain_tokens(
            encoder_input_ids=batch["encoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            encoder_residue_mask=batch["encoder_residue_mask"],
            encoder_chain_mask=batch["encoder_chain_mask"],
        )
        token_condition = m.gather_token_condition(
            chain_token_condition,
            chain_ids=batch["chain_ids"],
            position_ids_inner=batch["position_ids_inner"],
            attention_mask=batch["attention_mask"],
            encoder_residue_mask=batch["encoder_residue_mask"],
        )
        condition_mask = m.build_encoder_condition_mask(
            chain_ids=batch["chain_ids"],
            position_ids_inner=batch["position_ids_inner"],
            attention_mask=batch["attention_mask"],
            encoder_position_ids=None,
            residue_mask=batch["residue_mask"],
        )
        word_embeddings = m.decoder.get_input_embeddings()
        inputs_embeds = word_embeddings(batch["input_ids"])
        condition = token_condition.to(inputs_embeds.dtype)
        if m.condition_norm is not None:
            condition = m.condition_norm(condition)
        replace = condition_mask.to(inputs_embeds.dtype).unsqueeze(-1)
        if m.config.use_condition_projection and m.condition_proj is not None:
            inputs_embeds = inputs_embeds + m.condition_proj(condition) * replace
        else:
            inputs_embeds = inputs_embeds * (1.0 - replace) + condition * replace
        out = m.decoder(inputs_embeds=inputs_embeds,
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True)
        return out.hidden_states[-1], token_condition

    def _pool_batch(self, specs: list[tuple[str, str | None, str]]):
        """Pool features for a batch of (beta, alpha, peptide).

        Returns either:
        - ``pair_pooling="sequence"``: ``list[np.ndarray]`` — one whole-record
          mean-pool per spec (all residue tokens after joint LLaDA forward).
        - ``pair_pooling="segment"``: ``list[dict[str, np.ndarray]]`` — per-segment
          mean-pools keyed by chain name.
        """
        torch = self._torch
        records, meta_flags = [], []
        for beta, alpha, pep in specs:
            rec, has_a, has_p = self._record(beta, alpha, pep)
            records.append(rec)
            meta_flags.append((has_a, has_p))
        batch = self.collator(records)
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            dec_hidden, enc_cond = self._final_hidden(batch)
        feats = dec_hidden if self.feature_source == "decoder" else enc_cond
        feats = feats.float()
        chain_ids = batch["position_ids_chain"]
        residue = batch["residue_mask"].bool()

        if self.pair_pooling == "sequence":
            out_seq: list[np.ndarray] = []
            for b in range(len(specs)):
                mask = residue[b]
                if int(mask.sum()) == 0:
                    out_seq.append(np.zeros(feats.shape[-1], dtype=np.float32))
                else:
                    out_seq.append(
                        feats[b][mask].mean(0).cpu().numpy().astype(np.float32))
            return out_seq

        results = []
        for b in range(len(specs)):
            has_a, has_p = meta_flags[b]
            slotmap = self._segment_slots(has_a, has_p)
            seg = {}
            for slot, name in slotmap.items():
                mask = residue[b] & chain_ids[b].eq(slot)
                if int(mask.sum()) == 0:
                    seg[name] = np.zeros(feats.shape[-1], dtype=np.float32)
                else:
                    seg[name] = feats[b][mask].mean(0).cpu().numpy().astype(np.float32)
            results.append(seg)
        return results

    def embed_pairs(self, beta, alpha=None, peptide=None,
                    order=("beta", "alpha", "peptide")) -> np.ndarray:
        beta = [str(x) for x in beta]
        n = len(beta)
        peptide = [""] * n if peptide is None else [str(x) for x in peptide]
        alpha = [None] * n if alpha is None else [str(x) for x in alpha]
        use_alpha = any(a and a.strip() for a in alpha)
        seg_order = [s for s in order if (s != "alpha" or use_alpha)]

        specs = list(zip(beta, alpha, peptide))
        todo = [s for s in set(specs) if s not in self._cache]
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i:i + self.batch_size]
            pooled = self._pool_batch(list(chunk))
            if self.pair_pooling == "sequence":
                for spec, vec in zip(chunk, pooled):
                    self._cache[spec] = vec
            else:
                for spec, seg in zip(chunk, pooled):
                    self._cache[spec] = seg
        rows = []
        for spec in specs:
            cached = self._cache[spec]
            if self.pair_pooling == "sequence":
                rows.append(np.asarray(cached, dtype=np.float32))
                continue
            seg = cached
            # A dataset may mix paired (β+α) and β-only rows; ``seg_order`` is a
            # single global layout (alpha included iff ANY row has it), so a
            # β-only row's ``seg`` legitimately lacks the "alpha" segment. Fill a
            # missing segment with zeros of the pooled feature dim to keep every
            # row the same length (missing chain -> zero vector).
            dim = next(iter(seg.values())).shape[0]
            rows.append(np.concatenate(
                [seg[s] if s in seg else np.zeros(dim, dtype=np.float32)
                 for s in seg_order]))
        out = np.stack(rows).astype(np.float32)
        self.dim = out.shape[1]
        return out

    def embed(self, seqs: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        """Single-sequence embedding honouring ``feature_source``.

        ``decoder`` (default, post-LLaDA): each sequence becomes a single-chain
        ``tcr`` grammar record run through the LLaDA decoder, final hidden states
        mean-pooled over residues (see :meth:`_embed_single_decoder`).
        ``encoder``: ESMC-encoder mean-pool only (contextless baseline).
        """
        seqs = ["" if s is None else str(s) for s in seqs]
        if self.feature_source == "decoder":
            return self._embed_single_decoder(seqs, batch_size or self.batch_size)
        return self._embed_single_encoder(seqs, batch_size or 256)

    # -- single-chain post-LLaDA (decoder) path ------------------------------
    @staticmethod
    def _sanitize_chain(seq: str) -> str:
        """Coerce an arbitrary sequence into a valid single-chain grammar body.

        Uppercase, fold ``J``→``L`` (matching :func:`normalize_sequence`), keep
        only residue characters the renderer accepts (drop the ``.`` separator
        and ``-`` gap so a lone sequence never splits into multiple chains), and
        fall back to ``"A"`` when nothing is left.
        """
        s = "".join(str(seq or "").split()).upper().replace("J", "L")
        s = "".join(c for c in s if c in _GRAMMAR_VALID_CHARS)
        return s or "A"

    def _pool_single_decoder(self, seqs: list[str]) -> list[np.ndarray]:
        """LLaDA-decoder hidden states, mean-pooled over one single-chain record."""
        torch = self._torch
        records = [
            self._Record(chains=[self._Chain(self._sanitize_chain(s), "tcr_beta")],
                         task_type="tcr", source="grammar_embed_single")
            for s in seqs
        ]
        batch = self.collator(records)
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            dec_hidden, _ = self._final_hidden(batch)
        dec_hidden = dec_hidden.float()
        residue = batch["residue_mask"].bool()
        results = []
        for b in range(len(seqs)):
            mask = residue[b]
            if int(mask.sum()) == 0:
                results.append(np.zeros(dec_hidden.shape[-1], dtype=np.float32))
            else:
                results.append(dec_hidden[b][mask].mean(0).cpu().numpy().astype(np.float32))
        return results

    def _embed_single_decoder(self, seqs, batch_size: int) -> np.ndarray:
        """Post-LLaDA single-sequence embeddings (cached, length-bucketed)."""
        todo = [s for s in dict.fromkeys(seqs) if s not in self._single_cache]
        order = sorted(range(len(todo)), key=lambda i: len(todo[i]))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            chunk = [todo[j] for j in idx]
            for s, vec in zip(chunk, self._pool_single_decoder(chunk)):
                self._single_cache[s] = vec
        out = np.stack([self._single_cache[s] for s in seqs]).astype(np.float32)
        self.dim = out.shape[1]
        return out

    # -- per-residue post-LLaDA (decoder) path -------------------------------
    @staticmethod
    def _sanitize_chain_keeplen(seq: str) -> str:
        """Length-preserving sanitize for per-residue alignment.

        Same char folding as :meth:`_sanitize_chain` (uppercase, ``J``→``L``) but
        instead of *dropping* non-residue characters it *replaces* each with an
        ``'A'`` placeholder, so the returned body has **exactly one residue token
        per input character**. This keeps the CDR-infilling gap ``-`` (and the
        ``.`` separator, ``X`` etc.) from either splitting the chain or shifting
        positions, so per-residue features stay aligned 1:1 with the original
        string indices that ``run_residue.py`` uses to map position labels.
        """
        s = "".join(str(seq or "").split()).upper().replace("J", "L")
        s = "".join(c if c in _GRAMMAR_VALID_CHARS else "A" for c in s)
        return s or "A"

    def _residues_single_decoder(self, seqs: list[str]) -> list[np.ndarray]:
        """LLaDA-decoder final hidden states, per residue, for single-chain records."""
        torch = self._torch
        bodies = [self._sanitize_chain_keeplen(s) for s in seqs]
        records = [
            self._Record(chains=[self._Chain(b, "tcr_beta")],
                         task_type="tcr", source="grammar_embed_residues")
            for b in bodies
        ]
        batch = self.collator(records)
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            dec_hidden, _ = self._final_hidden(batch)
        dec_hidden = dec_hidden.float()
        residue = batch["residue_mask"].bool()
        results = []
        for b, body in enumerate(bodies):
            rows = dec_hidden[b][residue[b]].cpu().numpy().astype(np.float32)
            # Guard 1:1 alignment (single-token-per-char rendering guarantees it,
            # but truncate defensively if a tokenizer ever splits a character).
            if rows.shape[0] != len(body):
                n = min(rows.shape[0], len(body))
                rows = rows[:n]
            results.append(rows)
        return results

    def embed_residues(self, seqs: Sequence[str],
                       batch_size: int | None = None) -> list[np.ndarray]:
        """Per-residue **post-LLaDA decoder** hidden states (our-model口径).

        Returns one ``[L_i, decoder_hidden]`` array per input sequence, aligned
        1:1 to the (length-preserving sanitized) residues, mirroring
        :meth:`ESM2Embedder.embed_residues` / :meth:`EsmcBioSeqEmbedder.embed_residues`
        so the position-level ``run_residue.py`` probe runs on features that have
        passed through the trained LLaDA decoder (not just the frozen ESMC
        encoder). Only the ``decoder`` feature source is meaningful here.
        """
        if self.feature_source != "decoder":
            raise NotImplementedError(
                "embed_residues is the post-LLaDA口径 and requires "
                "feature_source='decoder' (spec grammar:decoder:/abs/best.pt)")
        seqs = ["" if s is None else str(s) for s in seqs]
        bs = batch_size or self.batch_size
        out: list[np.ndarray] = []
        for i in range(0, len(seqs), bs):
            out.extend(self._residues_single_decoder(list(seqs[i:i + bs])))
        return out

    # -- single-chain ESMC-encoder path (contextless baseline) ---------------
    def _embed_single_encoder(self, seqs, batch_size: int = 256) -> np.ndarray:
        """Single-sequence ESMC-encoder pooling (contextless; encoder baseline).

        Sequences are length-bucketed to minimise padding, so a large batch of
        short CDR3/epitope sequences runs in a handful of ESMC forwards.
        """
        torch = self._torch
        if not hasattr(self, "_enc_tok"):
            from dllm.pipelines.qwen3_vl_arch.data import HuggingFaceEsmTokenizerAdapter
            if not self._encoder_path:
                raise ValueError("checkpoint args lack encoder_path for single-seq embed()")
            self._enc_tok = HuggingFaceEsmTokenizerAdapter.from_pretrained(self._encoder_path)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out = np.zeros((len(seqs), self.enc_hidden), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(order), batch_size):
                idx = order[i:i + batch_size]
                vecs = self._encode_single([seqs[j] for j in idx])
                for k, j in enumerate(idx):
                    out[j] = vecs[k]
        self.dim = self.enc_hidden
        return out

    def _encode_single(self, seqs):
        torch = self._torch
        toks, masks = [], []
        for s in seqs:
            s = s if (s and len(s) > 0) else "A"
            ids, rm = self._enc_tok.encode_chain(s, max_length=self.max_length)
            toks.append(ids)
            masks.append(rm)
        L = max(len(t) for t in toks)
        pad = int(self._enc_tok.pad_token_id)
        input_ids = torch.full((len(toks), L), pad, dtype=torch.long)
        attn = torch.zeros((len(toks), L), dtype=torch.long)
        resm = torch.zeros((len(toks), L), dtype=torch.float32)
        for i, (t, rm) in enumerate(zip(toks, masks)):
            input_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            attn[i, :len(t)] = 1
            resm[i, :len(rm)] = torch.tensor(rm, dtype=torch.float32)
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)
        resm = resm.to(self.device).unsqueeze(-1)
        out = self.model.encoder(input_ids=input_ids, attention_mask=attn)
        hs = out.last_hidden_state.float()
        pooled = (hs * resm).sum(1) / resm.sum(1).clamp_min(1.0)
        return pooled.cpu().numpy()


class EsmcLladaBioSeqEmbedder(GrammarEmbedder):
    """**Post-LLaDA** per-sequence embedder for ``grammar_v2_*_llada`` checkpoints.

    This is the "after-LLaDA" feature口径: unlike :class:`EsmcBioSeqEmbedder`
    (which stops at the fine-tuned **ESMC encoder** mean-pool) and unlike
    ``grammar:encoder:`` (encoder condition only), it runs the **full pretrained
    pipeline** -- ESMC encoder condition -> **LLaDA diffusion denoiser** -- and
    reads out the LLaDA backbone's *final* hidden state.

    Feature definition (fixed default, reproducible):

    - **input**: each sequence is rendered as a *clean, fully-observed* single-
      chain grammar record (``task_type="tcr"``, role ``tcr_beta``; **no
      peptide/MHC context**, so a CDR3b is embedded in isolation with no epitope
      leakage for clustering). No diffusion corruption is applied; LLaDA is
      RADD-style with no timestep input.
    - **forward** (see :meth:`GrammarEmbedder._final_hidden`): ESMC encodes the
      chain -> the gathered per-residue condition **replaces** the LLaDA token
      embedding at residue positions -> the LLaDA backbone
      (``self.model.decoder``, a :class:`LLaDAModelLM`) runs bidirectionally.
    - **readout**: ``decoder(..., output_hidden_states=True).hidden_states[-1]``
      -- the post-``ln_f`` final hidden state that the tied read-out head
      consumes (``modeling_llada.py`` ``ln_f`` output) -- **mean-pooled over the
      chain's residue tokens** (:meth:`GrammarEmbedder._pool_single_decoder`).

    The whole model (``encoder.esmc.*`` + LLaDA ``decoder.*`` + optional
    ``condition_norm`` / ``condition_proj``) is loaded **strictly** (0 missing /
    0 unexpected keys) by :func:`load_grammar_checkpoint`; :attr:`_load_info`
    records the exact module/layer read out. Spec: ``bioseq-llada:/abs/best.pt``.
    """

    def __init__(self, state_dict_path: str, device: str | None = None,
                 batch_size: int = 32, max_length: int = 512, **kwargs):
        # feature_source is forced to "decoder" (post-LLaDA); ignore any override.
        kwargs.pop("feature_source", None)
        super().__init__(state_dict_path=state_dict_path, device=device,
                         feature_source="decoder", batch_size=batch_size,
                         max_length=max_length)
        from pathlib import Path as _P
        run_name = _P(state_dict_path).parent.name
        self.name = f"BioSeq-LLaDA-{run_name}" if run_name else "BioSeq-LLaDA"
        self._load_info = {
            "checkpoint_format": "grammar_v2_esmc_llada_full",
            "load": "strict (0 missing / 0 unexpected via load_grammar_checkpoint)",
            "feature": "llada_decoder_final_hidden_state_post_lnf",
            "readout": "decoder.hidden_states[-1] (post ln_f)",
            "pooling": "mean_over_residues",
            "input": "clean_unmasked_single_chain_tcr_beta_no_peptide",
            "decoder_hidden": int(self.hidden),
            "encoder_hidden": int(self.enc_hidden),
        }


def BioSeqEmbedder(state_dict_path: str, device: str | None = None, **kwargs):
    """Trained immune-receptor foundation-model embedder (auto-detects backbone).

    Peeks at the checkpoint and routes ESMC ``grammar_v2`` diffusion checkpoints
    to :class:`EsmcBioSeqEmbedder` (**encoder-only** mean-pool) and legacy
    Ophiuchus checkpoints to :class:`OphiuchusBioSeqEmbedder`.

    For downstream evaluation of **our** grammar_v2 model the headline口径 is
    **post-LLaDA** — use ``build_embedder("grammar:decoder:/abs/best.pt")`` or
    ``build_embedder("bioseq-llada:/abs/best.pt")`` instead. ``bioseq:`` is
    kept as a contextless encoder baseline (e.g. T2 clustering对照).
    """
    import torch
    payload = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    is_esmc = (
        isinstance(payload, dict)
        and "model_state_dict" in payload
        and any(k.startswith("encoder.esmc.") for k in payload["model_state_dict"])
    )
    if is_esmc:
        return EsmcBioSeqEmbedder(state_dict_path, device=device, payload=payload, **kwargs)
    del payload
    return OphiuchusBioSeqEmbedder(state_dict_path, device=device, **kwargs)


def build_embedder(spec: str, **kwargs) -> SequenceEmbedder:
    """Factory for all leaderboard embedders.

    Accepts:
      * ``esm2_150m`` / ``esm2_650m`` / ...  -> local ESM2 snapshot.
      * ``ophiuchus``                        -> Ophiuchus-Ab backbone.
      * ``bioseq:/abs/path.pt``              -> our model, ESMC-encoder mean-pool.
      * ``bioseq-llada:/abs/path.pt``        -> our model, **post-LLaDA** feature
        (full ESMC->LLaDA pipeline, LLaDA final hidden state mean-pool).
      * ``protbert`` / ``igbert`` / ``ablang_h`` / ``antiberta2`` /
        ``antiberta2_cssp`` / ``nanobert`` / ``vhhbert`` -> HuggingFace PALM.
      * ``tcrbert``                          -> TCR-BERT (wukevin/tcr-bert-mlm-only).
      * ``antiberty``                        -> AntiBERTy via the antiberty package.
      * ``hf:<repo_id>``                     -> any HF masked-LM (space-separated).
    """
    if spec.startswith("grammar:"):
        # grammar:/abs/best.pt
        # grammar:decoder:/abs/best.pt
        # grammar:decoder:/abs/best.pt              (global pool, default)
        # grammar:decoder:global:/abs/best.pt       (explicit global pool)
        # grammar:decoder:wholefeat:/abs/best.pt    (alias)
        # grammar:decoder:segment:/abs/best.pt      (deprecated segment concat)
        rest = spec.split(":", 1)[1]
        parts = rest.split(":")
        if parts[0] in ("decoder", "encoder"):
            source = parts[0]
            kwargs.setdefault("feature_source", source)
            if len(parts) >= 3 and parts[1] in (
                    "global", "wholefeat", "whole", "sequence", "seq",
                    "segment", "seg", "segment_concat"):
                tag = parts[1]
                if tag in ("global", "wholefeat", "whole", "sequence", "seq"):
                    kwargs.setdefault("pool_mode", "global")
                else:
                    kwargs.setdefault("pool_mode", "segment_concat")
                path = ":".join(parts[2:])
            else:
                path = ":".join(parts[1:])
            return GrammarEmbedder(state_dict_path=path, **kwargs)
        return GrammarEmbedder(state_dict_path=rest, **kwargs)
    if spec.startswith("bioseq-llada:"):
        # Post-LLaDA口径: full ESMC->LLaDA pipeline, final LLaDA hidden state.
        return EsmcLladaBioSeqEmbedder(state_dict_path=spec.split(":", 1)[1], **kwargs)
    if spec.startswith("bioseq:"):
        return BioSeqEmbedder(state_dict_path=spec.split(":", 1)[1], **kwargs)
    if spec == "ophiuchus":
        return OphiuchusEmbedder(**kwargs)
    if spec.startswith("esm2"):
        return ESM2Embedder(model_key=spec, **kwargs)
    if spec == "antiberty":
        return AntiBERTyEmbedder(**kwargs)
    if spec == "tcrbert":
        return TcrBertEmbedder(**kwargs)
    if spec == "protbert":
        return ProtBertEmbedder(**kwargs)
    if spec in ANTIBODY_LM_SPECS:
        cfg = ANTIBODY_LM_SPECS[spec]
        return HFProteinLMEmbedder(repo=cfg["repo"], name=cfg["name"],
                                   space_residues=cfg["space"], **kwargs)
    if spec.startswith("hf:"):
        return HFProteinLMEmbedder(repo=spec.split(":", 1)[1], **kwargs)
    raise ValueError(f"unknown embedder spec: {spec}")


# ---------------------------------------------------------------------------
# Native distance sources (real sequence-comparison / specialised TCR models).
# These expose ``cdist(query_df, ref_df) -> ndarray[Nq, Nr]`` so the few-shot
# engine can use a true distance matrix instead of an embedding + cosine.
# ---------------------------------------------------------------------------

class SceptrDistance:
    """SCEPTR (Cell Systems 2024) native distance source.

    SCEPTR is a small (153k-param) contrastive TCR representation model. It
    expects a DataFrame with columns ``TRAV, CDR3A, TRBV, CDR3B`` where each
    CDR3 is an IMGT-standardised *junction* (leading conserved ``C`` and
    trailing ``F``/``W``) and each V gene is a functional IMGT symbol. Our
    benchmark stores raw CDR3 loops (no anchors) in ``cdr3a/cdr3b`` and IMGT V
    symbols in ``va/vb``; this wrapper standardises both with ``tidytcells``
    (adding the missing conserved residues) before calling SCEPTR.

    ``cdist(query_df, ref_df)`` returns the SCEPTR L2 distance matrix, used as a
    ``native`` provider by ``common.fewshot``. ``embed`` returns the 64-d SCEPTR
    vector representations (for a probe / kNN protocol).
    """

    def __init__(self, variant: str = "default", device: str | None = None,
                 batch_size: int = 512):
        import sceptr
        from sceptr import variant as sceptr_variant

        self._sceptr = sceptr
        loader = getattr(sceptr_variant, variant, None)
        if loader is None:
            raise ValueError(f"unknown SCEPTR variant: {variant}")
        self.model = loader()
        self.model.set_batch_size(batch_size)
        self.variant = variant
        self.name = f"SCEPTR-{variant}" if variant != "default" else "SCEPTR"
        self.dim = 64

    @staticmethod
    def _to_sceptr_df(df):
        import pandas as pd
        import tidytcells as tt

        def std_junction(x):
            x = "" if x is None else str(x)
            if not x:
                return None
            out = tt.junction.standardize(
                x, fix_missing_conserved=True, log_failures=False
            )
            return out if out is not None else x

        def std_v(x):
            x = "" if x is None else str(x)
            if not x:
                return None
            # enforce_functional drops pseudogenes/ORFs (e.g. TRBV6-7) that SCEPTR
            # cannot look up; SCEPTR then treats the V gene as missing (partial input).
            return tt.tr.standardize(x, enforce_functional=True, log_failures=False)

        va = df["va"] if "va" in df.columns else [None] * len(df)
        vb = df["vb"] if "vb" in df.columns else [None] * len(df)
        return pd.DataFrame({
            "TRAV": [std_v(v) for v in va],
            "CDR3A": [std_junction(x) for x in df["cdr3a"]],
            "TRBV": [std_v(v) for v in vb],
            "CDR3B": [std_junction(x) for x in df["cdr3b"]],
        })

    def cdist(self, query_df, ref_df) -> np.ndarray:
        q = self._to_sceptr_df(query_df)
        r = self._to_sceptr_df(ref_df)
        return np.asarray(self.model.calc_cdist_matrix(q, r), dtype=np.float32)

    def embed(self, df) -> np.ndarray:
        return np.asarray(
            self.model.calc_vector_representations(self._to_sceptr_df(df)),
            dtype=np.float32,
        )


class TcrdistDistance:
    """TCRdist3 native distance source (Dash et al. / Mayer-Blackwell 2021).

    Builds a tcrdist3 ``TCRrep`` and uses its rectangular distance routine to
    return the paired alpha+beta TCRdist matrix between two TCR sets. Requires
    junction-form CDR3 (C...F/W) and IMGT V/J genes; missing J genes are filled
    with a functional default per V family is *not* attempted -- rows without a
    usable V gene fall back gracefully. Distances are the standard weighted
    CDR1/2/2.5/3 substitution distances.
    """

    def __init__(self, chains=("alpha", "beta"), organism: str = "human"):
        self.chains = list(chains)
        self.organism = organism
        self.name = "TCRdist"

    @staticmethod
    def _to_tcrdist_df(df):
        import pandas as pd
        import tidytcells as tt

        def std_junction(x):
            x = "" if x is None else str(x)
            if not x:
                return None
            out = tt.junction.standardize(
                x, fix_missing_conserved=True, log_failures=False
            )
            return out if out is not None else x

        def std_v(x):
            x = "" if x is None else str(x)
            if not x:
                return None
            return tt.tr.standardize(x, log_failures=False)

        def std_j(x):
            x = "" if x is None else str(x)
            if not x:
                return None
            return tt.tr.standardize(x, log_failures=False)

        # Missing/non-functional V genes (a handful of rows) get a benign
        # placeholder so tcrdist3 can still infer CDR1/CDR2 germline loops; the
        # CDR3 (the dominant distance term) is always kept from the real data.
        default_v = {"alpha": "TRAV1-1*01", "beta": "TRBV2*01"}
        va = [std_v(v) or default_v["alpha"] for v in df.get("va", [None] * len(df))]
        vb = [std_v(v) or default_v["beta"] for v in df.get("vb", [None] * len(df))]
        ja = [std_j(v) or "TRAJ1*01" for v in df.get("ja", [None] * len(df))]
        jb = [std_j(v) or "TRBJ2-1*01" for v in df.get("jb", [None] * len(df))]
        out = pd.DataFrame({
            "cdr3_a_aa": [std_junction(x) for x in df["cdr3a"]],
            "v_a_gene": va, "j_a_gene": ja,
            "cdr3_b_aa": [std_junction(x) for x in df["cdr3b"]],
            "v_b_gene": vb, "j_b_gene": jb,
            "count": 1,
        })
        return out

    def cdist(self, query_df, ref_df) -> np.ndarray:
        from tcrdist.repertoire import TCRrep

        q = self._to_tcrdist_df(query_df)
        r = self._to_tcrdist_df(ref_df)
        tr_q = TCRrep(cell_df=q, organism=self.organism, chains=self.chains,
                      compute_distances=False, deduplicate=False)
        tr_r = TCRrep(cell_df=r, organism=self.organism, chains=self.chains,
                      compute_distances=False, deduplicate=False)
        # A few V alleles have no germline CDR1/CDR2/pmhc loop in the reference,
        # leaving None cells that break pwseqdist's np.unique sort. Fill with ''.
        cq = tr_q.clone_df.where(tr_q.clone_df.notnull(), "")
        cr = tr_r.clone_df.where(tr_r.clone_df.notnull(), "")
        # rectangular distances between the two germline-annotated clone frames.
        tr_q.compute_rect_distances(df=cq, df2=cr)
        d = None  # total paired TCRdist = sum over requested chains.
        for ch in self.chains:
            m = np.asarray(getattr(tr_q, f"rw_{ch}"), dtype=np.float32)
            d = m if d is None else d + m
        return d


def build_distance_source(spec: str, **kwargs):
    """Factory for native distance sources: ``sceptr[:variant]`` / ``tcrdist``."""
    if spec.startswith("sceptr"):
        variant = spec.split(":", 1)[1] if ":" in spec else "default"
        return SceptrDistance(variant=variant, **kwargs)
    if spec.startswith("tcrdist"):
        return TcrdistDistance(**kwargs)
    raise ValueError(f"unknown distance source spec: {spec}")


__all__ = [
    "SequenceEmbedder", "ESM2Embedder", "OphiuchusEmbedder",
    "OphiuchusBioSeqEmbedder", "EsmcBioSeqEmbedder", "EsmcLladaBioSeqEmbedder",
    "GrammarEmbedder", "BioSeqEmbedder",
    "HFProteinLMEmbedder", "ProtBertEmbedder", "TcrBertEmbedder", "AntiBERTyEmbedder",
    "build_embedder", "ESM2_PATHS", "ANTIBODY_LM_SPECS",
    "SceptrDistance", "TcrdistDistance", "build_distance_source",
]
