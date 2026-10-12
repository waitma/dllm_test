"""Scoring for the three TCR generation settings (shared metric suite).

Setting A (unconditional): distribution match vs held-out repertoire + OLGA pgen.
Setting B (epitope-conditional): TCRT5 metric suite per pMHC, macro-averaged and
    stratified by category (held20 / benchmark14 / all) and BioSeq-seen/unseen.
Setting C (full-length): full-length validity + CDR3b extraction -> Setting B.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from common import metrics as M  # noqa: E402

DATA = BENCH / "data" / "tcr_generation_bench"
GEN_DATA = BENCH / "data" / "tcr_generation"
GEN_DATA_FULLREF = BENCH / "data" / "tcr_generation_fullref"
AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

# The anchor paper (TCRT5) reserves this pMHC for a separate simulation and
# excludes it from its headline sparse-13 aggregate.
RESERVED_SIMULATION_PMHC = "RVRAYTYSK_HLA-A*03:01"


# ---------------------------------------------------------------------------
# Eval set loaders
# ---------------------------------------------------------------------------

def load_eval_conditional() -> dict[str, dict]:
    return json.loads((DATA / "eval_conditional.json").read_text())


def load_benchmark_preds() -> dict[str, dict]:
    return json.loads((DATA / "benchmark_preds.json").read_text())


def load_bioseq_unseen() -> set[str]:
    p = DATA / "bioseq_unseen_pmhc.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


@lru_cache(maxsize=1)
def load_common_unseen() -> frozenset[str]:
    """The BioSeq-unseen pMHC that every benchmark14-derived eval set covers.

    ``bioseq_unseen`` is not comparable across runs, because its membership is
    whatever the run happened to be scored on: a held20-only run covers 2 unseen
    pMHC, a benchmark14 run covers 6, and a full 34-pMHC run covers 8.  Putting
    two such rows side by side compares different pMHC.  This fixed set is the
    common denominator -- benchmark14, BioSeq-unseen, reserved pMHC excluded --
    and is the only unseen view valid for cross-model rows.
    """
    try:
        eval_all = load_eval_conditional()
    except FileNotFoundError:
        return frozenset()
    unseen = load_bioseq_unseen()
    return frozenset(
        p for p, e in eval_all.items()
        if e.get("category") == "benchmark14"
        and p in unseen
        and p != RESERVED_SIMULATION_PMHC)


@lru_cache(maxsize=1)
def load_conditioning_by_epitope() -> dict[str, list[str]]:
    df = pd.read_csv(DATA / "train_binders.csv").fillna("")
    by_ep: dict[str, list[str]] = defaultdict(list)
    for ep, cb in zip(df["epitope"].astype(str), df["cdr3b"].astype(str)):
        if cb:
            by_ep[ep].append(cb)
    return dict(by_ep)


# ---------------------------------------------------------------------------
# Setting B / C: epitope-conditional scoring
# ---------------------------------------------------------------------------

def score_conditional(
    designs: dict[str, list[str]],
    eval_entries: dict[str, dict],
    k: int = 100,
    greedy_map: dict[str, str] | None = None,
    use_conditioning_for_novelty: bool = True,
) -> dict:
    """Score epitope-conditioned designs against held-out real binders.

    ``designs``      : {pMHC: [generated CDR3b, ...]} (pMHC key = "EPITOPE_ALLELE").
    ``eval_entries`` : from ``eval_conditional.json``.

    Returns per-pMHC TCRT5 metrics (+ novelty/diversity extras) and macro
    aggregates over: overall, held20, benchmark14/paper_sparse13,
    bioseq_seen, bioseq_unseen.
    """
    greedy_map = greedy_map or {}
    cond_by_ep = load_conditioning_by_epitope() if use_conditioning_for_novelty else {}
    unseen = load_bioseq_unseen()

    per_pmhc: dict[str, dict] = {}
    for pmhc, entry in eval_entries.items():
        gen = [str(s).upper() for s in designs.get(pmhc, []) if s]
        refs = entry["ref_binders"]
        m = M.tcrt5_design_metrics(gen, refs, k=k, greedy=greedy_map.get(pmhc))
        # design-quality extras aligned with the leakage-controlled view
        cond = cond_by_ep.get(entry["epitope"], [])
        m["novelty_vs_conditioning"] = (
            M.novelty(gen, cond) if (gen and cond) else float("nan"))
        m["diversity_edit"] = M._mean_pairwise_edit(gen) if len(gen) > 1 else float("nan")
        m["mean_nn_to_ref"] = M.mean_nn_distance(gen, refs) if (gen and refs) else float("nan")
        m["category"] = entry["category"]
        m["epitope"] = entry["epitope"]
        m["bioseq_unseen"] = pmhc in unseen
        per_pmhc[pmhc] = m

    agg = _aggregate_conditional(per_pmhc)
    return {"k": k, "n_pmhc": len(per_pmhc),
            "reserved_simulation_in_eval_set": RESERVED_SIMULATION_PMHC in per_pmhc,
            "summary": agg, "per_pmhc": per_pmhc}


_AGG_KEYS = ["precision", "recall", "f1", "d_edit", "seq_recovery", "char_bleu",
             "diversity", "novelty_vs_conditioning", "diversity_edit", "mean_nn_to_ref"]


def _macro(per_pmhc: dict, pmhcs: list[str]) -> dict:
    res = {}
    scored = {}
    for key in _AGG_KEYS:
        vals = []
        for p in pmhcs:
            v = per_pmhc[p].get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(v)
        res[key] = float(np.mean(vals)) if vals else float("nan")
        scored[key] = len(vals)
    # ``n_pmhc`` is category *membership*, which is not the number of pMHC that
    # actually entered the average: NaN entries are dropped above, so a category
    # can be fully populated yet contribute nothing.  A run conditioned on one
    # view emits designs only for that view's pMHC, which makes every other
    # category's macro average cover far fewer pMHC than its membership suggests
    # (e.g. a t4_unseen run reports overall n_pmhc=34 while averaging only the 9
    # unseen pMHC).  Report both counts so the gap can never be read away.
    res["n_pmhc"] = len(pmhcs)
    res["n_pmhc_scored"] = scored["d_edit"]
    res["n_scored_by_metric"] = scored
    return res


def _aggregate_conditional(per_pmhc: dict) -> dict:
    all_p = list(per_pmhc.keys())
    held = [p for p in all_p if per_pmhc[p]["category"] == "held20"]
    bench = [p for p in all_p if per_pmhc[p]["category"] == "benchmark14"]
    paper_sparse = [p for p in all_p if per_pmhc[p]["category"] == "paper_sparse13"]
    # ``bioseq_unseen_pmhc.json`` records a leakage *fact* -- the reserved
    # simulation epitope genuinely is absent from the BioSeq training splits -- so
    # that file stays intact.  But bioseq_seen / bioseq_unseen are cross-cutting
    # views used for headline claims, and the anchor paper excludes the reserved
    # pMHC from its headline aggregate, so it is dropped here.  It remains visible
    # under ``benchmark14``, which is explicitly labelled a 14-pMHC control.
    cross = [p for p in all_p if p != RESERVED_SIMULATION_PMHC]
    seen = [p for p in cross if not per_pmhc[p]["bioseq_unseen"]]
    unseen = [p for p in cross if per_pmhc[p]["bioseq_unseen"]]
    out = {"overall": _macro(per_pmhc, all_p)}
    if held:
        out["held20"] = _macro(per_pmhc, held)
    if bench:
        out["benchmark14"] = _macro(per_pmhc, bench)
    if paper_sparse:
        out["paper_sparse13"] = _macro(per_pmhc, paper_sparse)
    if seen:
        out["bioseq_seen"] = _macro(per_pmhc, seen)
    if unseen:
        out["bioseq_unseen"] = _macro(per_pmhc, unseen)
    # Only emitted when the run actually covers the whole common set, so its
    # presence is itself the guarantee that the row is cross-model comparable.
    common = load_common_unseen()
    if common and common.issubset(per_pmhc):
        out["bioseq_unseen_common"] = _macro(per_pmhc, sorted(common))
    return out


# ---------------------------------------------------------------------------
# Setting A: unconditional scoring
# ---------------------------------------------------------------------------

def load_unconditional_ref(full_reference: bool = False) -> tuple[list[str], list[str]]:
    """Setting A train / holdout references.

    The default ``data/tcr_generation`` build was capped: its train file holds
    180,918 of the 1,718,935 unique OTS sequences (10.5%), and its holdout was
    de-duplicated only against that capped subset.  1,893 of its 9,767 holdout
    sequences (19.4%) are in the full corpus, so they are memorised rather than
    held out, which flatters both k-mer JSD and novelty.  ``full_reference=True``
    selects the repaired ``data/tcr_generation_fullref`` build, whose holdout is
    exactly the old one minus those 1,893 (7,874 sequences, verified disjoint
    from the full corpus).
    """
    root = GEN_DATA_FULLREF if full_reference else GEN_DATA
    train = [s for s in (root / "train_cdr3b.txt").read_text().splitlines() if s]
    holdout = [s for s in (root / "holdout_cdr3b.txt").read_text().splitlines() if s]
    return train, holdout


def to_trimmed_cdr3(seq: str) -> str:
    """Strip the conserved flanking C (and trailing F/W) to match the OTS
    repertoire reference convention (stored as e.g. ``ASSLYRHSPLH``).

    Generators (OLGA / TCRT5 / BioSeq) emit full CDR3b (``CASSLYRHSPLHF``);
    this normalisation makes the distribution-match metrics convention-agnostic.
    """
    s = "".join(str(seq or "").split()).upper()
    if len(s) >= 3 and s[0] == "C":
        s = s[1:]
        if s and s[-1] in "FW":
            s = s[:-1]
    return s


def to_full_cdr3(seq: str) -> str:
    """Best-effort canonical full CDR3b (``C..F``) for OLGA pgen scoring."""
    s = "".join(str(seq or "").split()).upper()
    if not s:
        return s
    if s[0] != "C":
        s = "C" + s
    if s[-1] not in "FVW":
        s = s + "F"
    return s


def score_unconditional(
    gen: list[str],
    train: list[str] | None = None,
    holdout: list[str] | None = None,
    with_pgen: bool = True,
    train_ref_cap: int = 20000,
    pgen_cap: int = 3000,
    seed: int = 0,
    full_reference: bool = False,
) -> dict:
    if train is None or holdout is None:
        train, holdout = load_unconditional_ref(full_reference=full_reference)
    rng = np.random.default_rng(seed)
    # normalise generator output to the trimmed OTS convention for distribution match
    gen_trim = [to_trimmed_cdr3(s) for s in gen if s]
    # ``train_ref_cap`` exists for the O(n_gen x n_train) nearest-neighbour
    # distance only.  Novelty is exact set membership, so it gets the whole
    # corpus; sharing the capped sample would let a memorised sequence count as
    # novel just because it missed the subsample.
    ref_train = list(rng.choice(train, size=min(train_ref_cap, len(train)), replace=False))
    m = M.generation_metrics(gen_trim, dist_reference=holdout, train_set=ref_train,
                             k=3, novelty_train_set=train)
    result = {"convention": "trimmed (leading C / trailing F stripped)",
              "reference_build": "tcr_generation_fullref" if full_reference else "tcr_generation",
              "n_holdout": len(holdout), "n_train_corpus": len(train), **m}
    if with_pgen:
        from . import olga_beta
        # pgen is a generator-quality signal: are the sequences recombination
        # plausible? Scored on canonical full CDR3b (OLGA needs the C..F anchors).
        gen_full = [to_full_cdr3(s) for s in gen if s]
        result["pgen_generated"] = olga_beta.pgen_stats(gen_full, cap=pgen_cap, seed=seed)
        result["pgen_note"] = ("pgen on canonical full CDR3b; OLGA is self-referential "
                               "(=1.0). OTS holdout is stored trimmed so its pgen is not "
                               "directly comparable and is omitted.")
    return result


# ---------------------------------------------------------------------------
# CDR3b extraction from full-length beta chains (Setting C)
# ---------------------------------------------------------------------------

_CASS_RE = re.compile(r"C[A-Z]{4,26}?[FW]G[A-Z]G")
_CASS_RE_LOOSE = re.compile(r"(C[AS][SV][A-Z]{2,24}?[FW])")


def extract_cdr3b_regex(seq: str) -> str:
    """Fallback CDR3b extraction via the conserved C...FGxG J-motif anchor."""
    seq = "".join(str(seq or "").split()).upper()
    if not seq:
        return ""
    m = _CASS_RE.search(seq)
    if m:
        frag = m.group(0)
        j = frag.rfind("FG")
        return frag[: j + 1] if j > 0 else frag  # keep through the conserved F
    m = _CASS_RE_LOOSE.search(seq)
    if m:
        return m.group(1)
    return ""


def extract_cdr3b_anarci(seqs: list[str]) -> list[str]:
    """IMGT CDR3 (positions 104-118 inclusive: conserved C..F) via ANARCI batch.

    Falls back to the regex extractor per-sequence on ANARCI failure.
    """
    try:
        from anarci import anarci
    except Exception:
        return [extract_cdr3b_regex(s) for s in seqs]

    cleaned = ["".join(str(s or "").split()).upper() for s in seqs]
    inputs = [(f"s{i}", s) for i, s in enumerate(cleaned) if s]
    if not inputs:
        return ["" for _ in seqs]
    try:
        numbering, details, _ = anarci(inputs, scheme="imgt", output=False)
    except Exception:
        return [extract_cdr3b_regex(s) for s in seqs]

    result_by_idx: dict[int, str] = {}
    for (name, _seq), num in zip(inputs, numbering):
        idx = int(name[1:])
        cdr3 = ""
        if num:
            domain = num[0][0]  # list of ((pos, ins), aa)
            chars = [aa for (pos, ins), aa in domain if 104 <= pos <= 118 and aa != "-"]
            cdr3 = "".join(chars)
        result_by_idx[idx] = cdr3
    out = []
    for i, s in enumerate(cleaned):
        cdr3 = result_by_idx.get(i, "")
        if not cdr3:
            cdr3 = extract_cdr3b_regex(s)
        out.append(cdr3)
    return out


def is_valid_aa(seq: str) -> bool:
    seq = "".join(str(seq or "").split()).upper()
    return bool(seq) and all(c in AA_SET for c in seq)


def is_valid_cdr3b(seq: str) -> bool:
    """Syntactic CDR3β check for pipeline-designed sequences (C..F/W anchors)."""
    seq = "".join(str(seq or "").split()).upper()
    return (
        is_valid_aa(seq)
        and 5 <= len(seq) <= 30
        and seq[0] == "C"
        and seq[-1] in "FW"
    )


def _dedupe_preserve_order(seqs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seqs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def score_pipeline_cdr3beta(
    cdr3_designs: dict[str, list[str]],
    eval_entries: dict[str, dict],
    k: int = 100,
) -> dict:
    """Score epitope-conditioned pipeline output where betaCDR3 is already designed.

    Used for TcrDesign ``tcrdesign.py``: official output is structured V/J/CDR3
    rows, not full-length beta chains. Designed betaCDR3 is scored with Setting B
    metrics; pipeline validity reports syntactic CDR3β quality (valid AA + C..F/W).
    """
    cleaned: dict[str, list[str]] = {}
    validity: dict[str, dict] = {}
    for pmhc, seqs in cdr3_designs.items():
        seqs = _dedupe_preserve_order([str(s).upper() for s in seqs if s])
        valid_aa = [s for s in seqs if is_valid_aa(s)]
        valid_cdr3 = [s for s in valid_aa if is_valid_cdr3b(s)]
        cleaned[pmhc] = valid_cdr3[:k]
        validity[pmhc] = {
            "n_generated": len(seqs),
            "n_valid_aa": len(valid_aa),
            "frac_valid_aa": float(len(valid_aa) / len(seqs)) if seqs else float("nan"),
            "n_cdr3_extracted": len(valid_cdr3),
            "frac_cdr3_extracted": float(len(valid_cdr3) / len(valid_aa)) if valid_aa else float("nan"),
        }
    cond = score_conditional(cleaned, eval_entries, k=k)
    all_gen = sum(v["n_generated"] for v in validity.values())
    all_valid = sum(v["n_valid_aa"] for v in validity.values())
    all_cdr3 = sum(v["n_cdr3_extracted"] for v in validity.values())
    cond["fulllength_validity"] = {
        "n_generated": all_gen,
        "frac_valid_aa": float(all_valid / all_gen) if all_gen else float("nan"),
        "frac_cdr3_extracted": float(all_cdr3 / all_valid) if all_valid else float("nan"),
        "per_pmhc": validity,
    }
    cond["scoring_note"] = (
        "TcrDesign pipeline (tcrdesign.py): designed betaCDR3 scored via Setting B; "
        "validity = syntactic CDR3β (valid AA + C..F/W), not ANARCI full-length extraction."
    )
    return cond


def score_fulllength(
    full_designs: dict[str, list[str]],
    eval_entries: dict[str, dict],
    k: int = 100,
    use_anarci: bool = True,
) -> dict:
    """Score full-length beta designs: validity + CDR3b extraction -> Setting B.

    ``full_designs`` : {pMHC: [full-length beta chain, ...]}.
    """
    cdr3_designs: dict[str, list[str]] = {}
    validity: dict[str, dict] = {}
    for pmhc, seqs in full_designs.items():
        seqs = [str(s).upper() for s in seqs if s]
        valid = [s for s in seqs if is_valid_aa(s)]
        cdr3s = extract_cdr3b_anarci(valid) if use_anarci else [extract_cdr3b_regex(s) for s in valid]
        extracted = [c for c in cdr3s if c]
        cdr3_designs[pmhc] = extracted
        validity[pmhc] = {
            "n_generated": len(seqs),
            "n_valid_aa": len(valid),
            "frac_valid_aa": float(len(valid) / len(seqs)) if seqs else float("nan"),
            "n_cdr3_extracted": len(extracted),
            "frac_cdr3_extracted": float(len(extracted) / len(valid)) if valid else float("nan"),
        }
    cond = score_conditional(cdr3_designs, eval_entries, k=k)
    # roll up validity into the summary
    all_gen = sum(v["n_generated"] for v in validity.values())
    all_valid = sum(v["n_valid_aa"] for v in validity.values())
    all_cdr3 = sum(v["n_cdr3_extracted"] for v in validity.values())
    cond["fulllength_validity"] = {
        "n_generated": all_gen,
        "frac_valid_aa": float(all_valid / all_gen) if all_gen else float("nan"),
        "frac_cdr3_extracted": float(all_cdr3 / all_valid) if all_valid else float("nan"),
        "per_pmhc": validity,
    }
    return cond
