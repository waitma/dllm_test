"""Paper-faithful metrics for the complete TCRT5 CDR3beta evaluation suite.

Run the complete reconstruction with::

    /project_c20250601/251105016/conda/envs/pllm/bin/python \
      /project_c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_tcrt5_full_eval.py all

This module keeps the target-rich ``20 x 100`` validation protocol separate
from the sparse ``13 x 1000`` benchmark and the RVR deep-design target.  The
metric definitions follow Karthikeyan et al., Nature Machine Intelligence
(2025), DOI 10.1038/s42256-025-01096-6.  In particular:

* native recovery selects the closest same-length reference by Levenshtein,
  measures positional Hamming identity to that reference, and only falls back
  to reference-length-normalised Levenshtein when no same-length reference
  exists;
* mAP-by-description averages cumulative precision at every ranked prefix;
* positional delta entropy is ``H(generated) - H(reference)`` in nats;
* k-mer spectrum shift is JS divergence in nats for k=2,...,12.

The author release does not include the greedy hypotheses used for dataset
Char-BLEU or the cumulative log-likelihoods used to produce the paper mAP.
Callers must therefore preserve the evidence label attached to every metric:
an ordered-list mAP proxy is not presented as the published mAP.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from rapidfuzz import process as _rf_process
    from rapidfuzz.distance import Levenshtein as _rf_levenshtein
except ImportError:  # pragma: no cover - the pllm environment includes rapidfuzz
    _rf_process = None
    _rf_levenshtein = None


BENCH = Path(__file__).resolve().parents[1]
TCRT5_REPO = BENCH / "baselines" / "TCRT5_tcr_translate"
AUTHOR_ATOMIC_CSV = TCRT5_REPO / "results" / "all_atomic_eval_dedup.csv"
AUTHOR_DATASET_CSV = TCRT5_REPO / "results" / "all_dataset_eval_dedup.csv"
AUTHOR_SPARSE_CSV = TCRT5_REPO / "data" / "benchmark_data_w_preds.csv"
POLYSPECIFIC_CSV = TCRT5_REPO / "data" / "polyspecific_tcrs.csv"
TRACK_A_GENERATIONS = (
    BENCH / "outputs" / "tcr_beta_public_benchmark" / "results" / "generations.csv"
)
TRACK_A_METRICS = (
    BENCH / "outputs" / "tcr_beta_public_benchmark" / "results" / "per_target_metrics.csv"
)
DEFAULT_OUTPUT_ROOT = BENCH / "outputs" / "tcrt5_full_eval"

PAPER_URL = "https://www.nature.com/articles/s42256-025-01096-6"
PAPER_SUPPLEMENT_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2F"
    "s42256-025-01096-6/MediaObjects/42256_2025_1096_MOESM1_ESM.pdf"
)
RVR_TARGET = "RVRAYTYSK_HLA-A*03:01"
SPARSE_EXACT_TARGETS = {
    "FTDALGIDEY_HLA-A*01:01",
    "HPNGYKSLSTL_HLA-B*07:02",
}
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_HIT_CUTOFFS = (1, 5, 10, 20, 50, 100, 250, 500, 1000)
OLGA_124_MODEL_SHA256 = {
    "J_gene_CDR3_anchors.csv": (
        "38e06fb953be5d85d8cd5f0b4f6774e0654fa498d81184767f99d9287718ce49"
    ),
    "V_gene_CDR3_anchors.csv": (
        "b1552be7927e6397e60fb8e5f0212914d3cfa1e3fc300e33204d372202619161"
    ),
    "model_marginals.txt": (
        "2d8ee55020a0a538f7841883927eefa0bdb6653df9c6ca635b12a6e4a4d57460"
    ),
    "model_params.txt": (
        "2ad3a5fd94a0554543c4835c9e91e31fbe973e7ce8c727b69f3125105b95c3e6"
    ),
}
OLGA_124_SAMPLE_PGEN = {
    "CASSLGQETQYF": 2.266748481297098e-06,
    "CASSIRSSYEQYF": 1.8046338411903988e-07,
    "CASSLAPGATNEKLFF": 3.9051886990354146e-08,
}


AUTHOR_MODEL_LABELS = {
    "TCRBART-0_20eps_ddp/checkpoint-18000": "TCRBART-0",
    "TCRBART-0_B_20eps_ddp/checkpoint-44000": "TCRBART-0(B)",
    "TCRBART-0_M_20eps_ddp/checkpoint-6000": "TCRBART-0(M)",
    "TCRT5-0_20eps_ddp/checkpoint-10000": "TCRT5-0",
    "TCRT5-0_B_20eps_ddp/checkpoint-4000": "TCRT5-0(B)",
    "TCRT5-0_M_20eps_ddp/checkpoint-6000": "TCRT5-0(M)",
    "TCRBART-FT_20eps_ddp/checkpoint-32000": "TCRBART-FT",
    "TCRBART-FT_B_20eps_ddp/checkpoint-32000": "TCRBART-FT(B)",
    "TCRBART-FT_M_20eps_ddp/checkpoint-24000": "TCRBART-FT(M)",
    "TCRT5-FT_20eps_ddp/checkpoint-48000": "TCRT5-FT",
    "TCRT5-FT_B_20eps_ddp/checkpoint-36000": "TCRT5-FT(B)",
    "TCRT5-FT_M_20eps_ddp/checkpoint-46000": "TCRT5-FT(M)",
}


# Supplementary Table 2.  Percent-valued Char-BLEU entries are stored as
# fractions here so every comparison uses one numeric scale.
PAPER_MAIN_TABLE = {
    "TCRBART-0": {
        "char_bleu": 0.935,
        "f1_mean": 0.057,
        "f1_median": 0.010,
        "native_recovery_mean": 0.817,
        "native_recovery_median": 0.815,
        "map": 0.163,
        "total_unique": 1276,
    },
    "TCRBART-0(B)": {
        "char_bleu": 0.930,
        "f1_mean": 0.055,
        "f1_median": 0.015,
        "native_recovery_mean": 0.819,
        "native_recovery_median": 0.839,
        "map": 0.196,
        "total_unique": 1302,
    },
    "TCRBART-0(M)": {
        "char_bleu": 0.917,
        "f1_mean": 0.042,
        "f1_median": 0.030,
        "native_recovery_mean": 0.843,
        "native_recovery_median": 0.859,
        "map": 0.140,
        "total_unique": 240,
    },
    "TCRT5-0": {
        "char_bleu": 0.895,
        "f1_mean": 0.040,
        "f1_median": 0.030,
        "native_recovery_mean": 0.832,
        "native_recovery_median": 0.847,
        "map": 0.142,
        "total_unique": 129,
    },
    "TCRT5-0(B)": {
        "char_bleu": 0.348,
        "f1_mean": 0.054,
        "f1_median": 0.035,
        "native_recovery_mean": 0.845,
        "native_recovery_median": 0.859,
        "map": 0.167,
        "total_unique": 139,
    },
    "TCRT5-0(M)": {
        "char_bleu": 0.378,
        "f1_mean": 0.054,
        "f1_median": 0.020,
        "native_recovery_mean": 0.847,
        "native_recovery_median": 0.859,
        "map": 0.129,
        "total_unique": 177,
    },
    "TCRBART-FT": {
        "char_bleu": 0.874,
        "f1_mean": 0.013,
        "f1_median": 0.000,
        "native_recovery_mean": 0.841,
        "native_recovery_median": 0.851,
        "map": 0.049,
        "total_unique": 240,
    },
    "TCRBART-FT(B)": {
        "char_bleu": 0.898,
        "f1_mean": 0.014,
        "f1_median": 0.010,
        "native_recovery_mean": 0.835,
        "native_recovery_median": 0.846,
        "map": 0.062,
        "total_unique": 185,
    },
    "TCRBART-FT(M)": {
        "char_bleu": 0.861,
        "f1_mean": 0.023,
        "f1_median": 0.010,
        "native_recovery_mean": 0.845,
        "native_recovery_median": 0.850,
        "map": 0.048,
        "total_unique": 127,
    },
    "TCRT5-FT": {
        "char_bleu": 0.964,
        "f1_mean": 0.091,
        "f1_median": 0.020,
        "native_recovery_mean": 0.849,
        "native_recovery_median": 0.850,
        "map": 0.246,
        "total_unique": 1300,
    },
    "TCRT5-FT(B)": {
        "char_bleu": 0.935,
        "f1_mean": 0.083,
        "f1_median": 0.015,
        "native_recovery_mean": 0.846,
        "native_recovery_median": 0.851,
        "map": 0.279,
        "total_unique": 933,
    },
    "TCRT5-FT(M)": {
        "char_bleu": 0.944,
        "f1_mean": 0.082,
        "f1_median": 0.020,
        "native_recovery_mean": 0.847,
        "native_recovery_median": 0.846,
        "map": 0.180,
        "total_unique": 833,
    },
}


@dataclass(frozen=True)
class EvaluationBlock:
    """One ordered model/pMHC generation block and its target-specific truth."""

    model: str
    target_id: str
    generated: tuple[str, ...]
    references: tuple[str, ...]
    rank_basis: str
    greedy: str | None = None
    stored_char_bleu: float | None = None
    scores_available: bool = False
    protocol: str = ""


def clean_sequence(value: object) -> str:
    """Normalize presentation only; never insert, delete, or repair residues."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return "".join(str(value).split()).upper()


def parse_sequence_list(value: object) -> list[str]:
    """Parse a Python-list-valued release cell and preserve its order."""

    if isinstance(value, (list, tuple)):
        return [clean_sequence(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"expected list-valued sequence cell, got {type(parsed).__name__}")
    return [clean_sequence(item) for item in parsed]


def normalise_allele(value: object) -> str:
    allele = str(value).strip()
    return allele if allele.startswith("HLA-") else f"HLA-{allele}"


def target_id(peptide: object, allele: object) -> str:
    return f"{clean_sequence(peptide)}_{normalise_allele(allele)}"


def levenshtein_distance(left: str, right: str) -> int:
    """Levenshtein distance with a rapidfuzz accelerator and exact fallback."""

    left = clean_sequence(left)
    right = clean_sequence(right)
    if _rf_levenshtein is not None:
        return int(_rf_levenshtein.distance(left, right))
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index_left, char_left in enumerate(left, start=1):
        current = [index_left]
        for index_right, char_right in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[index_right] + 1,
                previous[index_right - 1] + (char_left != char_right),
            ))
        previous = current
    return previous[-1]


def _nearest(query: str, references: Sequence[str]) -> tuple[str, int]:
    if not references:
        raise ValueError("nearest-reference lookup requires at least one reference")
    if _rf_process is not None and _rf_levenshtein is not None:
        result = _rf_process.extractOne(
            query,
            references,
            scorer=_rf_levenshtein.distance,
        )
        if result is None:  # pragma: no cover - guarded by non-empty references
            raise RuntimeError("rapidfuzz returned no nearest reference")
        return str(result[0]), int(result[1])
    best = min(
        enumerate(references),
        key=lambda item: (levenshtein_distance(query, item[1]), item[0]),
    )
    return str(best[1]), levenshtein_distance(query, best[1])


class ReferenceIndex:
    """Reusable nearest-neighbour index preserving the paper's tie semantics."""

    def __init__(self, references: Sequence[str]):
        self.references = tuple(clean_sequence(item) for item in references)
        if not self.references:
            raise ValueError("a TCRT5 evaluation block must have references")
        by_length: dict[int, list[str]] = defaultdict(list)
        for reference in self.references:
            by_length[len(reference)].append(reference)
        self.by_length = {length: tuple(values) for length, values in by_length.items()}

    def nearest(self, query: str) -> tuple[str, int]:
        return _nearest(clean_sequence(query), self.references)

    def native_recovery(self, query: str) -> float:
        """Paper-native recovery, including its two-stage same-length rule.

        The upstream evaluator first selects the closest same-length reference
        by Levenshtein distance, but then measures recovery as positional
        (Hamming) identity to that selected reference.  Using
        ``1 - Levenshtein / length`` for this branch is subtly different for
        shifted motifs and does not reproduce the author's released metrics.
        """

        query = clean_sequence(query)
        same_length = self.by_length.get(len(query), ())
        if same_length:
            closest, _ = _nearest(query, same_length)
            return (
                sum(left == right for left, right in zip(query, closest)) / len(query)
                if query else float("nan")
            )
        closest, distance = self.nearest(query)
        return 1.0 - distance / len(closest) if closest else float("nan")


def exact_metrics(
    generated: Sequence[str],
    references: Sequence[str],
    k: int,
) -> dict[str, float]:
    """Exact TCRT5 Precision/Recall/F1@K, including official duplicate semantics."""

    candidates = [clean_sequence(item) for item in generated[:k]]
    reference_set = set(clean_sequence(item) for item in references)
    if not candidates:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    correct = [candidate for candidate in candidates if candidate in reference_set]
    precision = len(correct) / len(candidates)
    denominator = min(k, len(reference_set))
    recall = len(set(correct)) / denominator if denominator else float("nan")
    f1 = (
        0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def ranked_exact_metrics(
    generated: Sequence[str],
    references: Sequence[str],
    k: int,
    cutoffs: Sequence[int] = DEFAULT_HIT_CUTOFFS,
) -> dict[str, object]:
    """Exact-hit ranks plus the paper-text mAP reconstruction.

    ``average_precision_prefix`` is exactly the Methods description: take
    precision at top-1, top-2, ..., top-k and average those values.  The
    separately reported ``average_precision_at_hits`` is standard IR AP over
    observed hit ranks and is diagnostic only.
    """

    candidates = [clean_sequence(item) for item in generated[:k]]
    reference_set = set(clean_sequence(item) for item in references)
    hit_vector = np.asarray(
        [candidate in reference_set for candidate in candidates],
        dtype=np.float64,
    )
    ranks = np.arange(1, len(candidates) + 1, dtype=np.float64)
    prefix_precision = np.cumsum(hit_vector) / ranks if len(ranks) else np.asarray([])
    hit_ranks = [index + 1 for index, hit in enumerate(hit_vector) if hit]
    result: dict[str, object] = {
        "exact_hit_count": int(hit_vector.sum()),
        "exact_unique_hit_count": int(len({
            candidate for candidate in candidates if candidate in reference_set
        })),
        "exact_hit_ranks": ";".join(str(rank) for rank in hit_ranks),
        "first_exact_rank": int(hit_ranks[0]) if hit_ranks else math.nan,
        "median_exact_rank": float(np.median(hit_ranks)) if hit_ranks else math.nan,
        "average_precision_prefix": (
            float(prefix_precision.mean()) if len(prefix_precision) else math.nan
        ),
        "average_precision_at_hits": (
            float(prefix_precision[hit_vector.astype(bool)].mean())
            if hit_vector.any()
            else 0.0
        ),
    }
    for cutoff in cutoffs:
        if cutoff > k:
            continue
        prefix = candidates[:cutoff]
        occurrence_count = sum(candidate in reference_set for candidate in prefix)
        unique_count = len({candidate for candidate in prefix if candidate in reference_set})
        result[f"exact_occurrence_count_at_{cutoff}"] = int(occurrence_count)
        result[f"exact_unique_count_at_{cutoff}"] = int(unique_count)
        result[f"hit_at_{cutoff}"] = int(occurrence_count > 0)
    return result


def find_n_closest_matches(
    query: str,
    references: Sequence[str],
    n: int,
) -> list[str]:
    """Stable closest-reference selection used by the official Char-BLEU."""

    distances = [
        (index, reference, levenshtein_distance(query, reference))
        for index, reference in enumerate(references)
    ]
    distances.sort(key=lambda item: (item[2], item[0]))
    return [str(item[1]) for item in distances[:n]]


def _ngram_counts(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index:index + order])
        for index in range(max(0, len(tokens) - order + 1))
    )


def _modified_precision(
    references: Sequence[Sequence[str]],
    hypothesis: Sequence[str],
    order: int,
) -> tuple[int, int]:
    counts = _ngram_counts(hypothesis, order)
    denominator = max(1, sum(counts.values()))
    if not counts:
        return 0, denominator
    maximum_reference_counts: Counter[tuple[str, ...]] = Counter()
    for reference in references:
        reference_counts = _ngram_counts(reference, order)
        for ngram in counts:
            maximum_reference_counts[ngram] = max(
                maximum_reference_counts.get(ngram, 0),
                reference_counts.get(ngram, 0),
            )
    clipped = {
        ngram: min(count, maximum_reference_counts.get(ngram, 0))
        for ngram, count in counts.items()
    }
    return int(sum(clipped.values())), denominator


def _closest_reference_length(
    references: Sequence[Sequence[str]],
    hypothesis_length: int,
) -> int:
    return min(
        (len(reference) for reference in references),
        key=lambda length: (abs(length - hypothesis_length), length),
    )


def _brevity_penalty(reference_length: int, hypothesis_length: int) -> float:
    if hypothesis_length == 0:
        return 0.0
    if hypothesis_length > reference_length:
        return 1.0
    return math.exp(1.0 - reference_length / hypothesis_length)


def sentence_char_bleu(
    translation: str,
    references: Sequence[str],
    *,
    max_references: int = 20,
    max_ngram: int = 4,
) -> float:
    """Dependency-free port of NLTK sentence_bleu with method0 smoothing."""

    closest = find_n_closest_matches(translation, references, max_references)
    reference_tokens = [list(reference) for reference in closest]
    hypothesis = list(translation)
    if not reference_tokens or not hypothesis:
        return 0.0
    numerators: list[int] = []
    denominators: list[int] = []
    for order in range(1, max_ngram + 1):
        numerator, denominator = _modified_precision(
            reference_tokens,
            hypothesis,
            order,
        )
        numerators.append(numerator)
        denominators.append(denominator)
    if numerators[0] == 0:
        return 0.0
    log_precision = 0.0
    for numerator, denominator in zip(numerators, denominators):
        probability = (
            numerator / denominator
            if numerator
            else sys.float_info.min
        )
        log_precision += math.log(probability) / max_ngram
    reference_length = _closest_reference_length(reference_tokens, len(hypothesis))
    return float(
        _brevity_penalty(reference_length, len(hypothesis))
        * math.exp(log_precision)
    )


def corpus_char_bleu(
    translations: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    max_references: int = 20,
    max_ngram: int = 4,
) -> float:
    """Dependency-free official dataset Char-BLEU (NLTK method1 smoothing)."""

    if len(translations) != len(references):
        raise ValueError("corpus Char-BLEU needs one reference set per translation")
    numerators = np.zeros(max_ngram, dtype=np.int64)
    denominators = np.zeros(max_ngram, dtype=np.int64)
    total_hypothesis_length = 0
    total_reference_length = 0
    for translation, target_references in zip(translations, references):
        closest = find_n_closest_matches(
            translation,
            target_references,
            max_references,
        )
        reference_tokens = [list(reference) for reference in closest]
        hypothesis = list(translation)
        if not reference_tokens:
            continue
        for order in range(1, max_ngram + 1):
            numerator, denominator = _modified_precision(
                reference_tokens,
                hypothesis,
                order,
            )
            numerators[order - 1] += numerator
            denominators[order - 1] += denominator
        total_hypothesis_length += len(hypothesis)
        total_reference_length += _closest_reference_length(
            reference_tokens,
            len(hypothesis),
        )
    if numerators[0] == 0 or total_hypothesis_length == 0:
        return 0.0
    log_precision = 0.0
    for numerator, denominator in zip(numerators, denominators):
        probability = (
            numerator / denominator
            if numerator
            else 0.1 / denominator
        )
        log_precision += math.log(probability) / max_ngram
    return float(
        _brevity_penalty(total_reference_length, total_hypothesis_length)
        * math.exp(log_precision)
    )


def shannon_entropy(values: Sequence[str]) -> float:
    """Shannon entropy in nats, matching scipy.stats.entropy default units."""

    counts = np.asarray(list(Counter(values).values()), dtype=np.float64)
    if not len(counts):
        return math.nan
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def positional_entropy_rows(blocks: Sequence[EvaluationBlock]) -> pd.DataFrame:
    """Per-target positional entropy with and without an explicit right-pad."""

    rows: list[dict[str, object]] = []
    for block in blocks:
        generated = list(block.generated)
        references = list(block.references)
        maximum_length = max(map(len, generated + references))
        for index in range(maximum_length):
            generated_padded = [
                sequence[index] if index < len(sequence) else "|"
                for sequence in generated
            ]
            reference_padded = [
                sequence[index] if index < len(sequence) else "|"
                for sequence in references
            ]
            generated_residues = [value for value in generated_padded if value != "|"]
            reference_residues = [value for value in reference_padded if value != "|"]
            generated_gap_entropy = shannon_entropy(generated_padded)
            reference_gap_entropy = shannon_entropy(reference_padded)
            generated_residue_entropy = shannon_entropy(generated_residues)
            reference_residue_entropy = shannon_entropy(reference_residues)
            rows.append({
                "model": block.model,
                "target_id": block.target_id,
                "position_1based": index + 1,
                "n_generated_residues": len(generated_residues),
                "n_reference_residues": len(reference_residues),
                "generated_entropy_gap_inclusive_nats": generated_gap_entropy,
                "reference_entropy_gap_inclusive_nats": reference_gap_entropy,
                "delta_entropy_gap_inclusive_nats": (
                    generated_gap_entropy - reference_gap_entropy
                ),
                "generated_entropy_residue_only_nats": generated_residue_entropy,
                "reference_entropy_residue_only_nats": reference_residue_entropy,
                "delta_entropy_residue_only_nats": (
                    generated_residue_entropy - reference_residue_entropy
                ),
            })
    return pd.DataFrame(rows)


def _kmer_counts(sequences: Sequence[str], k: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sequence in sequences:
        counts.update(
            sequence[index:index + k]
            for index in range(max(0, len(sequence) - k + 1))
        )
    return counts


def js_divergence_nats(
    left_counts: Mapping[str, int],
    right_counts: Mapping[str, int],
) -> float:
    """Jensen-Shannon divergence in nats, with an upper bound of ln(2)."""

    keys = sorted(set(left_counts) | set(right_counts))
    if not keys:
        return math.nan
    left = np.asarray([left_counts.get(key, 0) for key in keys], dtype=np.float64)
    right = np.asarray([right_counts.get(key, 0) for key in keys], dtype=np.float64)
    if left.sum() == 0 or right.sum() == 0:
        return math.nan
    left /= left.sum()
    right /= right.sum()
    mixture = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    left_kl = np.sum(left[left_mask] * np.log(left[left_mask] / mixture[left_mask]))
    right_kl = np.sum(right[right_mask] * np.log(right[right_mask] / mixture[right_mask]))
    return float(0.5 * (left_kl + right_kl))


def kmer_spectrum_rows(
    blocks: Sequence[EvaluationBlock],
    k_values: Iterable[int] = range(2, 13),
) -> pd.DataFrame:
    """Per-target TCRT5 k-mer spectrum shift for k=2,...,12."""

    by_target: dict[str, list[EvaluationBlock]] = defaultdict(list)
    for block in blocks:
        by_target[block.target_id].append(block)
    rows: list[dict[str, object]] = []
    for target_name, target_blocks in sorted(by_target.items()):
        references = target_blocks[0].references
        for k in k_values:
            reference_counts = _kmer_counts(references, int(k))
            for block in target_blocks:
                generated_counts = _kmer_counts(block.generated, int(k))
                rows.append({
                    "model": block.model,
                    "target_id": target_name,
                    "k": int(k),
                    "js_divergence_nats": js_divergence_nats(
                        generated_counts,
                        reference_counts,
                    ),
                    "generated_total_kmers": int(sum(generated_counts.values())),
                    "reference_total_kmers": int(sum(reference_counts.values())),
                })
    return pd.DataFrame(rows)


def pairwise_jaccard_rows(blocks: Sequence[EvaluationBlock]) -> pd.DataFrame:
    """Full pairwise target-set Jaccard table for every model."""

    by_model: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for block in blocks:
        by_model[block.model][block.target_id] = set(block.generated)
    rows: list[dict[str, object]] = []
    for model, target_sets in sorted(by_model.items()):
        names = sorted(target_sets)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1:]:
                left = target_sets[left_name]
                right = target_sets[right_name]
                intersection = left & right
                union = left | right
                similarity = len(intersection) / len(union) if union else math.nan
                rows.append({
                    "model": model,
                    "target_a": left_name,
                    "target_b": right_name,
                    "intersection": len(intersection),
                    "union": len(union),
                    "jaccard_similarity": similarity,
                    "jaccard_dissimilarity": 1.0 - similarity,
                })
    return pd.DataFrame(rows)


def length_distribution_rows(blocks: Sequence[EvaluationBlock]) -> pd.DataFrame:
    """Count-valued generated/reference length distributions."""

    rows: list[dict[str, object]] = []
    reference_seen: set[str] = set()
    for block in blocks:
        for length, count in sorted(Counter(map(len, block.generated)).items()):
            rows.append({
                "model": block.model,
                "target_id": block.target_id,
                "population": "generated",
                "length": int(length),
                "count": int(count),
                "frequency": count / len(block.generated) if block.generated else math.nan,
            })
        # References are identical across model variants for a target.  Store
        # them once to avoid a 12-fold duplicate table.
        if block.target_id not in reference_seen:
            reference_seen.add(block.target_id)
            for length, count in sorted(Counter(map(len, block.references)).items()):
                rows.append({
                    "model": "__reference__",
                    "target_id": block.target_id,
                    "population": "reference",
                    "length": int(length),
                    "count": int(count),
                    "frequency": count / len(block.references),
                })
    return pd.DataFrame(rows)


def _polyspecific_lookup(path: Path = POLYSPECIFIC_CSV) -> dict[str, int]:
    frame = pd.read_csv(path)
    required = {"CDR3b", "N_Epitopes"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks required columns {sorted(required)}")
    frame = frame.drop_duplicates("CDR3b", keep="first")
    return {
        clean_sequence(sequence): int(number)
        for sequence, number in zip(frame["CDR3b"], frame["N_Epitopes"])
    }


def polyspecificity_rows(
    blocks: Sequence[EvaluationBlock],
    path: Path = POLYSPECIFIC_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-target and per-model exact membership in the 915-sequence panel."""

    lookup = _polyspecific_lookup(path)
    per_target_rows: list[dict[str, object]] = []
    by_model: dict[str, list[str]] = defaultdict(list)
    for block in blocks:
        generated = list(block.generated)
        hits = [sequence for sequence in generated if sequence in lookup]
        by_model[block.model].extend(generated)
        per_target_rows.append({
            "model": block.model,
            "target_id": block.target_id,
            "n_generated": len(generated),
            "polyspecific_generation_count": len(hits),
            "polyspecific_generation_fraction": (
                len(hits) / len(generated) if generated else math.nan
            ),
            "unique_polyspecific_sequences": len(set(hits)),
            "mean_known_cognate_epitopes": (
                float(np.mean([lookup[sequence] for sequence in hits]))
                if hits else math.nan
            ),
            "median_known_cognate_epitopes": (
                float(np.median([lookup[sequence] for sequence in hits]))
                if hits else math.nan
            ),
        })
    model_rows: list[dict[str, object]] = []
    for model, generated in by_model.items():
        hits = [sequence for sequence in generated if sequence in lookup]
        model_rows.append({
            "model": model,
            "n_generated": len(generated),
            "total_unique_sequences": len(set(generated)),
            "polyspecific_generation_count": len(hits),
            "polyspecific_generation_fraction": len(hits) / len(generated),
            "unique_polyspecific_sequences": len(set(hits)),
            "mean_known_cognate_epitopes": (
                float(np.mean([lookup[sequence] for sequence in hits]))
                if hits else math.nan
            ),
            "median_known_cognate_epitopes": (
                float(np.median([lookup[sequence] for sequence in hits]))
                if hits else math.nan
            ),
        })
    return pd.DataFrame(per_target_rows), pd.DataFrame(model_rows)


def score_block(
    block: EvaluationBlock,
    k: int,
    *,
    use_first_rank_as_bleu_proxy: bool = False,
) -> dict[str, object]:
    """Score one ordered generation block under the complete paper core."""

    generated = list(block.generated[:k])
    references = list(block.references)
    reference_index = ReferenceIndex(references)
    exact = exact_metrics(generated, references, k)
    ranked = ranked_exact_metrics(generated, references, k)
    nearest_distances: list[int] = []
    recoveries: list[float] = []
    for candidate in generated:
        _, nearest_distance = reference_index.nearest(candidate)
        nearest_distances.append(nearest_distance)
        recoveries.append(reference_index.native_recovery(candidate))

    if block.stored_char_bleu is not None:
        char_bleu = float(block.stored_char_bleu)
        char_bleu_evidence = "author_atomic_metric_greedy_hypothesis_not_released"
    elif block.greedy:
        char_bleu = sentence_char_bleu(block.greedy, references)
        char_bleu_evidence = "recomputed_from_explicit_greedy_hypothesis"
    elif use_first_rank_as_bleu_proxy and generated:
        char_bleu = sentence_char_bleu(generated[0], references)
        char_bleu_evidence = "first_rank_proxy_not_paper_greedy"
    else:
        char_bleu = math.nan
        char_bleu_evidence = "unavailable_missing_greedy_hypothesis"

    lengths = np.asarray([len(sequence) for sequence in generated], dtype=np.float64)
    reference_lengths = np.asarray(
        [len(sequence) for sequence in references],
        dtype=np.float64,
    )
    row: dict[str, object] = {
        "model": block.model,
        "target_id": block.target_id,
        "protocol": block.protocol,
        "k_requested": int(k),
        "n_generated": len(generated),
        "n_unique_generated": len(set(generated)),
        "n_references": len(references),
        "precision_at_k": exact["precision"],
        "recall_at_k": exact["recall"],
        "f1_at_k": exact["f1"],
        "mean_edit_distance": float(np.mean(nearest_distances)),
        "native_recovery_mean": float(np.mean(recoveries)),
        "native_recovery_median_per_candidate": float(np.median(recoveries)),
        "native_recovery_ge_90_count": int(sum(value >= 0.9 for value in recoveries)),
        "native_recovery_ge_90_fraction": float(np.mean(np.asarray(recoveries) >= 0.9)),
        "char_bleu": char_bleu,
        "char_bleu_evidence": char_bleu_evidence,
        "rank_basis": block.rank_basis,
        "scores_available": bool(block.scores_available),
        "generated_length_mean": float(lengths.mean()),
        "generated_length_std_population": float(lengths.std(ddof=0)),
        "generated_length_median": float(np.median(lengths)),
        "generated_length_min": int(lengths.min()),
        "generated_length_max": int(lengths.max()),
        "reference_length_mean": float(reference_lengths.mean()),
        "reference_length_std_population": float(reference_lengths.std(ddof=0)),
    }
    row.update(ranked)
    return row


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else math.nan


def summarise_models(
    blocks: Sequence[EvaluationBlock],
    per_target: pd.DataFrame,
    jaccard: pd.DataFrame,
    kmer: pd.DataFrame,
    polyspecific_models: pd.DataFrame,
    *,
    dataset_char_bleu: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate paper-facing model metrics without losing evidence fields."""

    dataset_char_bleu = dataset_char_bleu or {}
    block_by_model: dict[str, list[EvaluationBlock]] = defaultdict(list)
    for block in blocks:
        block_by_model[block.model].append(block)
    poly_index = (
        polyspecific_models.set_index("model")
        if len(polyspecific_models)
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    for model, model_blocks in block_by_model.items():
        target_metrics = per_target[per_target["model"] == model]
        generated = [
            sequence
            for block in model_blocks
            for sequence in block.generated
        ]
        lengths = np.asarray([len(sequence) for sequence in generated], dtype=np.float64)
        model_jaccard = jaccard[jaccard["model"] == model]
        model_kmer = kmer[kmer["model"] == model]
        row: dict[str, object] = {
            "model": model,
            "n_targets": len(model_blocks),
            "n_generated": len(generated),
            "total_unique_sequences": len(set(generated)),
            "global_diversity_fraction": len(set(generated)) / len(generated),
            "precision_macro": _mean_numeric(target_metrics, "precision_at_k"),
            "recall_macro": _mean_numeric(target_metrics, "recall_at_k"),
            "f1_mean": _mean_numeric(target_metrics, "f1_at_k"),
            "f1_median": float(target_metrics["f1_at_k"].median()),
            "mean_edit_distance_macro": _mean_numeric(
                target_metrics,
                "mean_edit_distance",
            ),
            "native_recovery_mean": _mean_numeric(
                target_metrics,
                "native_recovery_mean",
            ),
            "native_recovery_median": float(
                target_metrics["native_recovery_mean"].median()
            ),
            "native_recovery_ge_90_count": int(
                target_metrics["native_recovery_ge_90_count"].sum()
            ),
            "exact_hit_occurrences": int(target_metrics["exact_hit_count"].sum()),
            "exact_hit_unique_target_sum": int(
                target_metrics["exact_unique_hit_count"].sum()
            ),
            "map_prefix_ordered_proxy": _mean_numeric(
                target_metrics,
                "average_precision_prefix",
            ),
            "map_prefix": _mean_numeric(
                target_metrics,
                "average_precision_prefix",
            ),
            "map_hit_rank_diagnostic": _mean_numeric(
                target_metrics,
                "average_precision_at_hits",
            ),
            "map_rank_is_paper_compatible": all(
                block.scores_available for block in model_blocks
            ),
            "map_evidence": (
                "paper_exact_if_rank_basis_is_cumulative_model_log_likelihood"
                if all(block.scores_available for block in model_blocks)
                else "ordered_list_proxy_missing_paper_cumulative_log_likelihoods"
            ),
            "char_bleu_dataset": dataset_char_bleu.get(model, math.nan),
            "char_bleu_dataset_evidence": (
                "author_dataset_metric_greedy_hypotheses_not_released"
                if model in dataset_char_bleu
                else "unavailable"
            ),
            "char_bleu_atomic_mean": _mean_numeric(target_metrics, "char_bleu"),
            "jaccard_similarity_mean": (
                float(model_jaccard["jaccard_similarity"].mean())
                if len(model_jaccard) else math.nan
            ),
            "jaccard_dissimilarity_mean": (
                float(model_jaccard["jaccard_dissimilarity"].mean())
                if len(model_jaccard) else math.nan
            ),
            "generated_length_mean": float(lengths.mean()),
            "generated_length_std_population": float(lengths.std(ddof=0)),
        }
        for k_value, group in model_kmer.groupby("k"):
            row[f"kmer_jsd_nats_k{int(k_value)}"] = float(
                group["js_divergence_nats"].mean()
            )
        if len(poly_index) and model in poly_index.index:
            poly_row = poly_index.loc[model]
            for column in (
                "polyspecific_generation_count",
                "polyspecific_generation_fraction",
                "unique_polyspecific_sequences",
                "mean_known_cognate_epitopes",
                "median_known_cognate_epitopes",
            ):
                row[column] = poly_row[column]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_blocks(
    blocks: Sequence[EvaluationBlock],
    *,
    k: int,
    dataset_char_bleu: Mapping[str, float] | None = None,
    use_first_rank_as_bleu_proxy: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run every CPU-only TCRT5 metric over a collection of blocks."""

    per_target = pd.DataFrame([
        score_block(
            block,
            k,
            use_first_rank_as_bleu_proxy=use_first_rank_as_bleu_proxy,
        )
        for block in blocks
    ])
    jaccard = pairwise_jaccard_rows(blocks)
    entropy = positional_entropy_rows(blocks)
    kmer = kmer_spectrum_rows(blocks)
    length_distribution = length_distribution_rows(blocks)
    poly_target, poly_model = polyspecificity_rows(blocks)
    summary = summarise_models(
        blocks,
        per_target,
        jaccard,
        kmer,
        poly_model,
        dataset_char_bleu=dataset_char_bleu,
    )
    return {
        "per_target": per_target,
        "summary": summary,
        "jaccard": jaccard,
        "positional_entropy": entropy,
        "kmer_jsd": kmer,
        "length_distribution": length_distribution,
        "polyspecificity_per_target": poly_target,
        "polyspecificity_summary": poly_model,
    }


def load_author_main_blocks(
    atomic_csv: Path = AUTHOR_ATOMIC_CSV,
) -> tuple[list[EvaluationBlock], pd.DataFrame]:
    """Load the 12 x 20 x 100 author-release validation predictions."""

    frame = pd.read_csv(atomic_csv)
    required = {
        "Unnamed: 0",
        "translations",
        "reference_translations",
        "model",
        "char-bleu",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{atomic_csv} lacks required columns: {missing}")
    unknown_models = sorted(set(frame["model"]) - set(AUTHOR_MODEL_LABELS))
    if unknown_models:
        raise ValueError(f"unmapped author model identifiers: {unknown_models}")
    blocks: list[EvaluationBlock] = []
    for _, row in frame.iterrows():
        blocks.append(EvaluationBlock(
            model=AUTHOR_MODEL_LABELS[str(row["model"])],
            target_id=str(row["Unnamed: 0"]),
            generated=tuple(parse_sequence_list(row["translations"])),
            references=tuple(parse_sequence_list(row["reference_translations"])),
            rank_basis="released_beam_order_without_cumulative_log_likelihood",
            stored_char_bleu=float(row["char-bleu"]),
            scores_available=False,
            protocol="paper_main_top20_k100",
        ))
    if len(blocks) != 240:
        raise ValueError(f"expected 240 author main blocks, found {len(blocks)}")
    return blocks, frame


def load_author_dataset_metrics(
    path: Path = AUTHOR_DATASET_CSV,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Load the author's dataset-level numeric companion table."""

    frame = pd.read_csv(path)
    model_column = "Unnamed: 0" if "Unnamed: 0" in frame.columns else frame.columns[0]
    unknown_models = sorted(set(frame[model_column]) - set(AUTHOR_MODEL_LABELS))
    if unknown_models:
        raise ValueError(f"unmapped dataset model identifiers: {unknown_models}")
    frame = frame.copy()
    frame["model_label"] = frame[model_column].map(AUTHOR_MODEL_LABELS)
    char_bleu = {
        str(row["model_label"]): float(row["char-bleu"])
        for _, row in frame.iterrows()
    }
    return frame, char_bleu


def load_sparse_reference_map(
    path: Path = AUTHOR_SPARSE_CSV,
) -> dict[str, tuple[str, ...]]:
    frame = pd.read_csv(path, low_memory=False)
    return {
        target_id(row["Epitope"], row["Allele"]): tuple(
            dict.fromkeys(parse_sequence_list(row["reference_translations"]))
        )
        for _, row in frame.iterrows()
    }


def load_author_sparse_blocks(
    path: Path = AUTHOR_SPARSE_CSV,
) -> list[EvaluationBlock]:
    """Load TCRT5/GRATCR/ER/ER+ predictions for the 14-target sparse artifact."""

    frame = pd.read_csv(path, low_memory=False)
    methods = {
        "TCRT5": ("tcrt5_translations", "tcrt5_greedy", "released_tcrt5_rank"),
        "GRATCR": ("gratcr_translations", "gratcr_greedy", "released_generation_index"),
        "ER-TRANSFORMER": (
            "er_translations",
            "er_greedy",
            "released_generation_index",
        ),
        "ER-TRANSFORMER+": (
            "er_corrected",
            "er_greedy_corrected",
            "released_generation_index_after_anchor_repair",
        ),
    }
    blocks: list[EvaluationBlock] = []
    for _, row in frame.iterrows():
        name = target_id(row["Epitope"], row["Allele"])
        references = tuple(dict.fromkeys(
            parse_sequence_list(row["reference_translations"])
        ))
        for model, (prediction_column, greedy_column, rank_basis) in methods.items():
            generated = tuple(parse_sequence_list(row[prediction_column]))
            greedy_value = row.get(greedy_column)
            greedy = (
                clean_sequence(greedy_value)
                if isinstance(greedy_value, str) and greedy_value.strip()
                else None
            )
            blocks.append(EvaluationBlock(
                model=model,
                target_id=name,
                generated=generated,
                references=references,
                rank_basis=rank_basis,
                greedy=greedy,
                scores_available=False,
                protocol=(
                    "paper_sparse13_k1000"
                    if name != RVR_TARGET
                    else "paper_rvr_k1000"
                ),
            ))
    return blocks


def load_track_a_blocks(
    generations_path: Path = TRACK_A_GENERATIONS,
    sparse_csv: Path = AUTHOR_SPARSE_CSV,
) -> tuple[list[EvaluationBlock], pd.DataFrame]:
    """Load current rebuilt model generations on the public 14-target Track A."""

    generations = pd.read_csv(generations_path)
    required = {"model", "target_id", "rank", "cdr3b", "raw_score"}
    missing = sorted(required - set(generations.columns))
    if missing:
        raise ValueError(f"{generations_path} lacks required columns: {missing}")
    references = load_sparse_reference_map(sparse_csv)
    blocks: list[EvaluationBlock] = []
    for (model, name), group in generations.groupby(["model", "target_id"], sort=True):
        if name not in references:
            raise ValueError(f"generation target {name} has no sparse reference set")
        ordered = group.sort_values("rank", kind="stable")
        raw_scores = pd.to_numeric(ordered["raw_score"], errors="coerce")
        if str(model) == "TCRT5" and raw_scores.notna().all():
            rank_basis = "cumulative_model_log_likelihood"
            scores_available = True
        elif raw_scores.notna().all():
            rank_basis = "model_specific_official_order_with_scores"
            scores_available = True
        else:
            rank_basis = "seeded_emission_or_official_order_without_likelihood"
            scores_available = False
        blocks.append(EvaluationBlock(
            model=str(model),
            target_id=str(name),
            generated=tuple(ordered["cdr3b"].map(clean_sequence)),
            references=references[str(name)],
            rank_basis=rank_basis,
            scores_available=scores_available,
            protocol=(
                "current_sparse13_k1000"
                if name != RVR_TARGET
                else "current_rvr_k1000"
            ),
        ))
    return blocks, generations


def merge_existing_giana(
    per_target: pd.DataFrame,
    path: Path = TRACK_A_METRICS,
) -> pd.DataFrame:
    """Merge already-completed official GIANA outputs without rerunning GIANA."""

    if not path.is_file():
        result = per_target.copy()
        result["giana_status"] = "not_available"
        return result
    source = pd.read_csv(path)
    columns = {
        "Model": "model",
        "target_id": "target_id",
        "GIANAHit": "giana_generated_hit",
        "GIANAReferenceClusterCoverage": "giana_reference_cluster_coverage",
        "GIANAReferenceSequenceCoverage": "giana_reference_sequence_coverage",
        "GIANAEligibleGenerated": "giana_eligible_generated",
        "GIANAStatus": "giana_status",
    }
    available = [column for column in columns if column in source.columns]
    source = source[available].rename(columns=columns)
    return per_target.merge(
        source,
        on=["model", "target_id"],
        how="left",
        validate="one_to_one",
    )


def _olga_pgen_worker(sequence: str) -> tuple[str, float]:
    from tcr_generation_bench.olga_beta import pgen

    return sequence, float(pgen(sequence))


def compute_pgen_cache(
    sequences: Iterable[str],
    cache_path: Path,
    *,
    workers: int = 1,
) -> dict[str, float]:
    """Compute OLGA Pgen once per unique sequence and persist an append-safe cache."""

    unique_sequences = sorted(set(clean_sequence(item) for item in sequences if item))
    cache: dict[str, float] = {}
    if cache_path.is_file():
        cached = pd.read_csv(cache_path)
        if {"cdr3b", "pgen"}.issubset(cached.columns):
            cache = {
                clean_sequence(sequence): float(value)
                for sequence, value in zip(cached["cdr3b"], cached["pgen"])
            }
    missing = [sequence for sequence in unique_sequences if sequence not in cache]
    if missing:
        if workers <= 1:
            computed = map(_olga_pgen_worker, missing)
            for sequence, value in computed:
                cache[sequence] = value
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for sequence, value in executor.map(
                    _olga_pgen_worker,
                    missing,
                    chunksize=25,
                ):
                    cache[sequence] = value
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            sorted(cache.items()),
            columns=["cdr3b", "pgen"],
        ).to_csv(cache_path, index=False)
    return {sequence: cache[sequence] for sequence in unique_sequences}


def _pgen_summary(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    positive = array[array > 0]
    logs = np.log10(positive)
    return {
        "n": int(len(array)),
        "n_pgen_positive": int(len(positive)),
        "fraction_pgen_positive": float(len(positive) / len(array)) if len(array) else math.nan,
        "mean_log10_pgen_positive": float(logs.mean()) if len(logs) else math.nan,
        "std_log10_pgen_positive_population": (
            float(logs.std(ddof=0)) if len(logs) else math.nan
        ),
        "median_log10_pgen_positive": (
            float(np.median(logs)) if len(logs) else math.nan
        ),
    }


def pgen_summary_rows(
    blocks: Sequence[EvaluationBlock],
    cache_path: Path,
    *,
    workers: int = 1,
    include_references: bool = False,
    models: set[str] | None = None,
) -> pd.DataFrame:
    """Occurrence-weighted OLGA summaries by model/target and optional references."""

    selected = [
        block for block in blocks
        if models is None or block.model in models
    ]
    sequences = [
        sequence for block in selected for sequence in block.generated
    ]
    if include_references:
        reference_by_target = {
            block.target_id: block.references for block in selected
        }
        sequences.extend(
            sequence
            for references in reference_by_target.values()
            for sequence in references
        )
    cache = compute_pgen_cache(sequences, cache_path, workers=workers)
    rows: list[dict[str, object]] = []
    for block in selected:
        values = [cache[sequence] for sequence in block.generated]
        rows.append({
            "model": block.model,
            "target_id": block.target_id,
            "population": "generated",
            **_pgen_summary(values),
        })
    for model, model_blocks in _group_blocks_by_model(selected).items():
        values = [
            cache[sequence]
            for block in model_blocks
            for sequence in block.generated
        ]
        rows.append({
            "model": model,
            "target_id": "__all_targets__",
            "population": "generated",
            **_pgen_summary(values),
        })
    if include_references:
        reference_by_target = {
            block.target_id: block.references for block in selected
        }
        for name, references in sorted(reference_by_target.items()):
            rows.append({
                "model": "__reference__",
                "target_id": name,
                "population": "reference",
                **_pgen_summary([cache[sequence] for sequence in references]),
            })
        all_reference_values = [
            cache[sequence]
            for references in reference_by_target.values()
            for sequence in references
        ]
        rows.append({
            "model": "__reference__",
            "target_id": "__all_targets__",
            "population": "reference",
            **_pgen_summary(all_reference_values),
        })
    return pd.DataFrame(rows)


def pgen_aggregation_sensitivity_rows(
    blocks: Sequence[EvaluationBlock],
    cache_path: Path,
    *,
    workers: int = 1,
) -> pd.DataFrame:
    """Expose Pgen pooling/filter choices that are absent from the paper.

    Figure 4 displays log10 Pgen only from -16 to approximately -5.  The paper
    does not state whether its reported moments use all positive values,
    globally unique sequences, or only the visible histogram window.  We
    materialize each defensible choice rather than tuning a hidden cutoff until
    it matches the published moments.
    """

    generated = [sequence for block in blocks for sequence in block.generated]
    reference_by_target = {
        block.target_id: block.references for block in blocks
    }
    references = [
        sequence
        for target_references in reference_by_target.values()
        for sequence in target_references
    ]
    cache = compute_pgen_cache(
        generated + references,
        cache_path,
        workers=workers,
    )

    populations = {
        "generated_occurrence_weighted": generated,
        "generated_global_unique": list(dict.fromkeys(generated)),
        "reference_target_deduplicated_occurrence_weighted": references,
        "reference_global_unique": list(dict.fromkeys(references)),
    }
    rows: list[dict[str, object]] = []
    for population, sequences in populations.items():
        values = np.asarray([cache[sequence] for sequence in sequences], dtype=np.float64)
        positive = values[values > 0]
        logs = np.log10(positive)
        for window_name, minimum, maximum in (
            ("all_positive", -math.inf, math.inf),
            ("figure_visible_minus16_to_minus5", -16.0, -5.0),
        ):
            selected = logs[(logs >= minimum) & (logs <= maximum)]
            rows.append({
                "population": population,
                "log10_window": window_name,
                "n_sequences": len(sequences),
                "n_pgen_positive": len(positive),
                "n_log10_selected": len(selected),
                "log10_min_inclusive": (
                    minimum if math.isfinite(minimum) else math.nan
                ),
                "log10_max_inclusive": (
                    maximum if math.isfinite(maximum) else math.nan
                ),
                "mean_log10_pgen": (
                    float(selected.mean()) if len(selected) else math.nan
                ),
                "std_log10_pgen_population": (
                    float(selected.std(ddof=0)) if len(selected) else math.nan
                ),
                "median_log10_pgen": (
                    float(np.median(selected)) if len(selected) else math.nan
                ),
            })
    return pd.DataFrame(rows)


def olga_protocol_audit_rows() -> pd.DataFrame:
    """Verify the bundled Pgen inputs against the paper's OLGA 1.2.4 setup."""

    from tcr_generation_bench.olga_beta import OLGA_MODEL, pgen

    rows: list[dict[str, object]] = []
    requirement_path = TCRT5_REPO / "requirements.txt"
    requirement_lines = {
        line.strip()
        for line in requirement_path.read_text(encoding="utf-8").splitlines()
    }
    rows.append({
        "audit_type": "paper_requirement",
        "item": str(requirement_path),
        "expected": "olga==1.2.4",
        "observed": "olga==1.2.4" if "olga==1.2.4" in requirement_lines else "",
        "relative_delta": math.nan,
        "agreement": "olga==1.2.4" in requirement_lines,
    })
    for filename, expected_hash in OLGA_124_MODEL_SHA256.items():
        path = OLGA_MODEL / filename
        observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "audit_type": "human_T_beta_model_sha256",
            "item": str(path),
            "expected": expected_hash,
            "observed": observed_hash,
            "relative_delta": math.nan,
            "agreement": observed_hash == expected_hash,
        })
    for sequence, expected_value in OLGA_124_SAMPLE_PGEN.items():
        observed_value = float(pgen(sequence))
        relative_delta = abs(observed_value - expected_value) / expected_value
        rows.append({
            "audit_type": "sample_probability",
            "item": sequence,
            "expected": f"{expected_value:.17g}",
            "observed": f"{observed_value:.17g}",
            "relative_delta": relative_delta,
            "agreement": relative_delta <= 2e-6,
        })
    return pd.DataFrame(rows)


def _group_blocks_by_model(
    blocks: Sequence[EvaluationBlock],
) -> dict[str, list[EvaluationBlock]]:
    grouped: dict[str, list[EvaluationBlock]] = defaultdict(list)
    for block in blocks:
        grouped[block.model].append(block)
    return dict(grouped)


def polyspecificity_paper_correlation(summary: pd.DataFrame) -> dict[str, object]:
    """Reproduce the paper's selected-family r=-0.957 relationship."""

    selected_models = [
        "TCRBART-0",
        "TCRBART-0(B)",
        "TCRBART-0(M)",
        "TCRT5-FT",
        "TCRT5-FT(B)",
        "TCRT5-FT(M)",
    ]
    indexed = summary.set_index("model")
    missing = [model for model in selected_models if model not in indexed.index]
    if missing:
        return {"status": "unavailable", "missing_models": missing}
    diversity = np.asarray(
        [indexed.loc[model, "total_unique_sequences"] for model in selected_models],
        dtype=np.float64,
    )
    poly_count = np.asarray(
        [indexed.loc[model, "polyspecific_generation_count"] for model in selected_models],
        dtype=np.float64,
    )
    return {
        "status": "ok",
        "models": selected_models,
        "pearson_r_unique_vs_polyspecific_generations": float(
            np.corrcoef(diversity, poly_count)[0, 1]
        ),
        "paper_pearson_r": -0.957,
    }


def author_release_crosscheck(
    reconstructed: pd.DataFrame,
    author_atomic: pd.DataFrame,
    author_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Compare independent core recomputation against both official CSV levels."""

    official_atomic = author_atomic.copy()
    official_atomic["model"] = official_atomic["model"].map(AUTHOR_MODEL_LABELS)
    official_atomic = official_atomic.rename(columns={
        "Unnamed: 0": "target_id",
        "precision": "official_precision",
        "recall": "official_recall",
        "f1": "official_f1",
        "d_edit": "official_d_edit",
        "seq_recovery": "official_native_recovery",
        "char-bleu": "official_char_bleu",
    })
    columns = [
        "model",
        "target_id",
        "official_precision",
        "official_recall",
        "official_f1",
        "official_d_edit",
        "official_native_recovery",
        "official_char_bleu",
    ]
    merged = reconstructed.merge(
        official_atomic[columns],
        on=["model", "target_id"],
        validate="one_to_one",
    )
    pairs = {
        "precision": ("precision_at_k", "official_precision"),
        "recall": ("recall_at_k", "official_recall"),
        "f1": ("f1_at_k", "official_f1"),
        "d_edit": ("mean_edit_distance", "official_d_edit"),
        "native_recovery": ("native_recovery_mean", "official_native_recovery"),
        "char_bleu": ("char_bleu", "official_char_bleu"),
    }
    rows: list[dict[str, object]] = []
    for metric, (ours_column, official_column) in pairs.items():
        delta = np.abs(
            pd.to_numeric(merged[ours_column])
            - pd.to_numeric(merged[official_column])
        )
        rows.append({
            "level": "atomic_240_blocks",
            "metric": metric,
            "n": len(delta),
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "all_close_1e_12": bool((delta <= 1e-12).all()),
            "evidence": "same_released_candidate_lists",
        })

    dataset_model_column = (
        "Unnamed: 0"
        if "Unnamed: 0" in author_dataset.columns
        else author_dataset.columns[0]
    )
    official_dataset = author_dataset.copy()
    official_dataset["model"] = official_dataset[dataset_model_column].map(
        AUTHOR_MODEL_LABELS
    )
    reconstructed_summary = (
        reconstructed.groupby("model", sort=False)
        .agg({
            "precision_at_k": "mean",
            "recall_at_k": "mean",
            "f1_at_k": "mean",
            "mean_edit_distance": "mean",
            "native_recovery_mean": "mean",
        })
        .reset_index()
    )
    merged_dataset = reconstructed_summary.merge(
        official_dataset,
        on="model",
        validate="one_to_one",
        suffixes=("_reconstructed", "_official"),
    )
    dataset_pairs = {
        "precision": ("precision_at_k", "precision"),
        "recall": ("recall_at_k", "recall"),
        "f1": ("f1_at_k", "f1"),
        "d_edit": ("mean_edit_distance", "d_edit"),
        "native_recovery": ("native_recovery_mean", "seq_recovery"),
    }
    for metric, (ours_column, official_column) in dataset_pairs.items():
        delta = np.abs(
            pd.to_numeric(merged_dataset[ours_column])
            - pd.to_numeric(merged_dataset[official_column])
        )
        rows.append({
            "level": "dataset_12_models",
            "metric": metric,
            "n": len(delta),
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "all_close_1e_12": bool((delta <= 1e-12).all()),
            "evidence": (
                "author_dataset_companion_without_its_candidate_lists;"
                "may_be_a_separate_generation_pass"
            ),
        })
    return pd.DataFrame(rows)


def paper_main_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Model-by-model Supplementary Table 2 correspondence."""

    indexed = summary.set_index("model")
    rows: list[dict[str, object]] = []
    mapping = {
        "char_bleu": "char_bleu_dataset",
        "f1_mean": "f1_mean",
        "f1_median": "f1_median",
        "native_recovery_mean": "native_recovery_mean",
        "native_recovery_median": "native_recovery_median",
        "map": "map_prefix_ordered_proxy",
        "total_unique": "total_unique_sequences",
    }
    for model, paper_values in PAPER_MAIN_TABLE.items():
        for metric, paper_value in paper_values.items():
            reconstructed_column = mapping[metric]
            reconstructed_value = float(indexed.loc[model, reconstructed_column])
            tolerance = 0.0 if metric == "total_unique" else 0.00051
            evidence = "independently_recomputed_from_released_sequences"
            if metric == "char_bleu":
                evidence = (
                    "author_dataset_numeric_companion_only;"
                    "greedy_hypotheses_not_released"
                )
            elif metric == "map":
                evidence = (
                    "ordered_release_list_proxy_only;"
                    "paper_cumulative_log_likelihoods_not_released"
                )
            absolute_delta = abs(reconstructed_value - float(paper_value))
            rows.append({
                "model": model,
                "metric": metric,
                "paper_value": paper_value,
                "reconstructed_value": reconstructed_value,
                "absolute_delta": absolute_delta,
                "rounding_tolerance": tolerance,
                "agreement": bool(absolute_delta <= tolerance),
                "evidence": evidence,
            })
    return pd.DataFrame(rows)


def add_giana_summary(summary: pd.DataFrame, per_target: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    if "giana_generated_hit" not in per_target.columns:
        return result
    aggregates = (
        per_target.groupby("model", dropna=False)
        .agg(
            giana_generated_hit=("giana_generated_hit", "sum"),
            giana_reference_cluster_coverage=(
                "giana_reference_cluster_coverage",
                "sum",
            ),
            giana_reference_sequence_coverage=(
                "giana_reference_sequence_coverage",
                "sum",
            ),
        )
        .reset_index()
    )
    return result.merge(aggregates, on="model", how="left", validate="one_to_one")


def write_tables(
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    prefix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        filename = f"{prefix}{name}.csv"
        frame.to_csv(output_dir / filename, index=False)


def json_ready(value: object) -> object:
    """Convert numpy/Pandas/NaN values to strict JSON-compatible objects."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
