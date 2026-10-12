"""Official-baseline generators that run in the main env (TCRT5 HF, OLGA, SONIA).

Each returns the unified design format:
  * conditional: {pMHC: [CDR3b, ...]}  (+ optional greedy hypothesis map)
  * unconditional: [CDR3b, ...]
"""

from __future__ import annotations

import os
import re
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
_TAG_RE = re.compile(r"\[.*?\]")


# ---------------------------------------------------------------------------
# TCRT5 (HuggingFace) -- Nat Mach Intell 2025
# ---------------------------------------------------------------------------

def _load_tcrt5(model_id: str = "dkarthikeyan1/tcrt5_ft_tcrdb", device: str = "cuda"):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tok = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id).eval()
    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    return tok, model.to(device), device


def _decode_cdr3(tok, sequences) -> list[str]:
    return [_TAG_RE.sub("", x).strip().upper()
            for x in tok.batch_decode(sequences, skip_special_tokens=True)]


def tcrt5_conditional(
    eval_entries: dict[str, dict],
    k: int = 100,
    num_beams_mult: int = 3,
    max_new_tokens: int = 25,
    model_id: str = "dkarthikeyan1/tcrt5_ft_tcrdb",
    device: str = "cuda",
    progress: bool = True,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Beam-search K CDR3b per pMHC conditioned on ``[PMHC]{pep}[SEP]{pseudo}[EOS]``."""
    import torch

    tok, model, device = _load_tcrt5(model_id, device)
    designs: dict[str, list[str]] = {}
    greedy: dict[str, str] = {}
    items = list(eval_entries.items())
    for i, (pmhc, entry) in enumerate(items):
        pep = entry["epitope"]
        pseudo = entry.get("mhc_pseudo", "")
        prompt = f"[PMHC]{pep}[SEP]{pseudo}[EOS]"
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                num_return_sequences=k, num_beams=num_beams_mult * k,
                return_dict_in_generate=True,
            )
            g = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=5,
                               num_return_sequences=1, return_dict_in_generate=True)
        designs[pmhc] = _decode_cdr3(tok, out["sequences"])
        greedy[pmhc] = _decode_cdr3(tok, g["sequences"])[0]
        if progress and (i % 5 == 0 or i == len(items) - 1):
            print(f"  TCRT5 [{i + 1}/{len(items)}] {pmhc}: {len(designs[pmhc])} seqs", flush=True)
    return designs, greedy


def tcrt5_unconditional(
    n: int = 5000,
    max_new_tokens: int = 25,
    batch: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    model_id: str = "dkarthikeyan1/tcrt5_ft_tcrdb",
    device: str = "cuda",
    seed: int = 42,
) -> list[str]:
    """Sample ``n`` CDR3b unconditionally (multinomial sampling for repertoire diversity)."""
    import torch

    tok, model, device = _load_tcrt5(model_id, device)
    torch.manual_seed(seed)
    out: list[str] = []
    empty = tok("", return_tensors="pt").to(device)
    while len(out) < n:
        take = min(batch, n - len(out))
        with torch.no_grad():
            gen = model.generate(
                input_ids=empty["input_ids"], attention_mask=empty["attention_mask"],
                max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                top_k=(top_k or 0) or 50, num_return_sequences=take,
                return_dict_in_generate=True,
            )
        out.extend([s for s in _decode_cdr3(tok, gen["sequences"]) if s])
    return out[:n]


# ---------------------------------------------------------------------------
# OLGA / SONIA -- unconditional statistical baselines (Setting A + B lower bound)
# ---------------------------------------------------------------------------

def olga_unconditional(n: int = 5000, seed: int = 42) -> list[str]:
    """Official OLGA human_T_beta sampler (``olga.sequence_generation``)."""
    from . import olga_beta
    return olga_beta.sample_cdr3b(n, seed=seed)


# Official SONIA (statbiophys/SONIA) default humanTRB selection model.
_SONIA_REPO = BENCH / "baselines" / "SONIA"
_SONIA_TRB = _SONIA_REPO / "sonia" / "default_models" / "human_T_beta"


def sonia_unconditional(n: int = 5000, seed: int = 42, rejection_bound: int = 10,
                        processes: int = 8) -> list[str]:
    """soNNia/SONIA post-selected CDR3b via the OFFICIAL SONIA package.

    Pipeline (exactly ``sonia/generate.py --post``): OLGA human_T_beta pre-sample
    -> SONIA left/right-pos selection model rejection sampling
    (``SequenceGeneration.generate_sequences_post``). Uses the bundled official
    default TCR-beta selection weights (``default_models/human_T_beta/model.h5``).
    """
    import sys as _sys
    import numpy as np

    if str(_SONIA_REPO) not in _sys.path:
        _sys.path.insert(0, str(_SONIA_REPO))
    import olga.load_model as olga_load_model
    import olga.sequence_generation as olga_seq_gen
    from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
    from sonia.sequence_generation import SequenceGeneration

    np.random.seed(seed)
    params = str(_SONIA_TRB / "model_params.txt")
    marginals = str(_SONIA_TRB / "model_marginals.txt")
    v_anchor = str(_SONIA_TRB / "V_gene_CDR3_anchors.csv")
    j_anchor = str(_SONIA_TRB / "J_gene_CDR3_anchors.csv")

    genomic_data = olga_load_model.GenomicDataVDJ()
    genomic_data.load_igor_genomic_data(params, v_anchor, j_anchor)
    generative_model = olga_load_model.GenerativeModelVDJ()
    generative_model.load_and_process_igor_model(marginals)
    seqgen_model = olga_seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)

    sonia_model = SoniaLeftposRightpos(
        feature_file=str(_SONIA_TRB / "features.tsv"),
        log_file=str(_SONIA_TRB / "log.txt"),
        vj=False,
    )
    seq_gen = SequenceGeneration(sonia_model, custom_olga_model=seqgen_model,
                                 custom_genomic_data=genomic_data, processes=processes)
    seqs = seq_gen.generate_sequences_post(num_seqs=int(n), upper_bound=rejection_bound, nucleotide=True)
    # rows are [aa_CDR3, V, J, nt]; keep the amino-acid CDR3
    return [str(row[0]) for row in seqs if len(row) and row[0]]


def replicate_pool_per_pmhc(pool: list[str], eval_entries: dict[str, dict]) -> dict[str, list[str]]:
    """Use one unconditional pool as an epitope-agnostic lower bound for every pMHC."""
    return {pmhc: list(pool) for pmhc in eval_entries}
