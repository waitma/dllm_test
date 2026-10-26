"""Track-A public epitope-conditioned CDR3beta benchmark data and metrics.

Prepare and score with::

    /project_c20250601/251105016/conda/envs/pllm/bin/python \
      /project_c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py prepare
    /project_c20250601/251105016/conda/envs/pllm/bin/python \
      /project_c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py score

This module intentionally does not repair generated sequences.  Whitespace and
case are normalised for comparison, but conserved C/F/W residues are never
inserted and invalid raw candidates remain in the ValidRate denominator.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

try:  # Optional compiled accelerator; the fallback below is dependency-free.
    from rapidfuzz import process as _rapidfuzz_process
    from rapidfuzz.distance import Levenshtein as _rapidfuzz_levenshtein
except ImportError:  # pragma: no cover - exercised in minimal model environments
    _rapidfuzz_process = None
    _rapidfuzz_levenshtein = None


BENCH = Path(__file__).resolve().parents[1]
TCRT5_REPO = BENCH / "baselines" / "TCRT5_tcr_translate"
DEFAULT_BENCHMARK_CSV = TCRT5_REPO / "data" / "benchmark_data_w_preds.csv"
DEFAULT_IMMPREP_CSV = TCRT5_REPO / "data" / "IMMPREP_test_set.csv"
DEFAULT_PSEUDO_CSV = TCRT5_REPO / "refs" / "hla_pseudo_seqs.csv"
DEFAULT_DATA_DIR = BENCH / "data" / "tcr_beta_public_benchmark"
DEFAULT_RESULTS_DIR = BENCH / "outputs" / "tcr_beta_public_benchmark" / "results"

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
GENERATION_COLUMNS = [
    "model",
    "target_id",
    "peptide",
    "mhc_allele",
    "rank",
    "cdr3b",
    "raw_score",
]
OPTIONAL_GENERATION_COLUMNS = ["generation_mode", "source", "run_id"]
RVR_TARGET_ID = "RVRAYTYSK_HLA-A*03:01"


@dataclass(frozen=True)
class Target:
    """One public pMHC target and its paired set of known CDR3beta binders."""

    target_id: str
    source_target_id: str
    peptide: str
    mhc_allele: str
    mhc_pseudosequence: str
    references: tuple[str, ...]
    reference_source: str
    dataset: str
    year: str

    @property
    def n_references(self) -> int:
        return len(self.references)


@dataclass(frozen=True)
class GianaMetrics:
    """GIANA hit counts for one model-target block."""

    generated_hit: int
    reference_clusters_covered: int
    reference_sequences_covered: int
    generated_eligible: int
    status: str = "ok"


def clean_sequence(value: object) -> str:
    """Normalise only presentation details; never add or delete residues."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().upper()


def normalise_allele(value: object) -> str:
    allele = str(value).strip()
    if not allele.startswith("HLA-"):
        allele = f"HLA-{allele}"
    return allele


def canonical_target_id(peptide: object, mhc_allele: object) -> str:
    return f"{clean_sequence(peptide)}_{normalise_allele(mhc_allele)}"


def _parse_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [clean_sequence(x) for x in value]
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"expected a list-valued CSV cell, got {type(parsed).__name__}")
    return [clean_sequence(x) for x in parsed]


def _load_pseudosequences(path: Path = DEFAULT_PSEUDO_CSV) -> dict[str, str]:
    frame = pd.read_csv(path, index_col=0)
    if "pseudo-sequence" not in frame.columns:
        raise ValueError(f"missing pseudo-sequence column in {path}")
    return {
        normalise_allele(allele): clean_sequence(sequence)
        for allele, sequence in frame["pseudo-sequence"].items()
    }


def load_public_targets(
    benchmark_csv: Path = DEFAULT_BENCHMARK_CSV,
    pseudo_csv: Path = DEFAULT_PSEUDO_CSV,
) -> dict[str, Target]:
    """Load only the reference column from TCR-TRANSLATE's public 14-pMHC CSV."""

    frame = pd.read_csv(benchmark_csv, low_memory=False)
    required = {"Epitope", "Allele", "reference_translations", "Reference", "Dataset"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{benchmark_csv} is missing required columns: {missing}")

    pseudos = _load_pseudosequences(pseudo_csv)
    targets: dict[str, Target] = {}
    for _, row in frame.iterrows():
        peptide = clean_sequence(row["Epitope"])
        allele = normalise_allele(row["Allele"])
        target_id = canonical_target_id(peptide, allele)
        references = tuple(dict.fromkeys(x for x in _parse_list(row["reference_translations"]) if x))
        if target_id in targets:
            raise ValueError(f"duplicate pMHC target in benchmark: {target_id}")
        if allele not in pseudos:
            raise ValueError(f"no MHC pseudo-sequence for {target_id} ({allele})")
        targets[target_id] = Target(
            target_id=target_id,
            source_target_id=str(row.get("pMHC", "")),
            peptide=peptide,
            mhc_allele=allele,
            mhc_pseudosequence=pseudos[allele],
            references=references,
            reference_source=str(row["Reference"]),
            dataset=str(row["Dataset"]),
            year=str(row.get("Year", "")),
        )
    if len(targets) != 14:
        raise ValueError(f"expected 14 public benchmark pMHC targets, found {len(targets)}")
    if RVR_TARGET_ID not in targets:
        raise ValueError(f"required smoke target {RVR_TARGET_ID} is absent")
    return targets


def inspect_immprep(path: Path = DEFAULT_IMMPREP_CSV) -> dict[str, object]:
    """Return a compact field/count audit without treating any prediction as reference."""

    frame = pd.read_csv(path, low_memory=False)
    required = {"Peptide", "HLA", "CDR3b", "CDR3b_extended", "Label", "Usage"}
    return {
        "path": str(path),
        "n_rows": int(len(frame)),
        "columns": list(frame.columns),
        "required_columns_present": sorted(required & set(frame.columns)),
        "required_columns_missing": sorted(required - set(frame.columns)),
        "n_peptides": int(frame["Peptide"].nunique()) if "Peptide" in frame else None,
        "n_cdr3b": int(frame["CDR3b_extended"].nunique()) if "CDR3b_extended" in frame else None,
        "usage_counts": (
            {str(k): int(v) for k, v in frame["Usage"].value_counts(dropna=False).items()}
            if "Usage" in frame else {}
        ),
    }


def write_benchmark_audit(
    data_dir: Path = DEFAULT_DATA_DIR,
    benchmark_csv: Path = DEFAULT_BENCHMARK_CSV,
    immprep_csv: Path = DEFAULT_IMMPREP_CSV,
    pseudo_csv: Path = DEFAULT_PSEUDO_CSV,
) -> dict[str, object]:
    """Materialise target/reference tables and a machine-readable source audit."""

    targets = load_public_targets(benchmark_csv, pseudo_csv)
    data_dir.mkdir(parents=True, exist_ok=True)

    target_rows = []
    reference_rows = []
    for target in targets.values():
        target_rows.append({
            "target_id": target.target_id,
            "source_target_id": target.source_target_id,
            "peptide": target.peptide,
            "mhc_allele": target.mhc_allele,
            "mhc_pseudosequence": target.mhc_pseudosequence,
            "n_references": target.n_references,
            "reference_source": target.reference_source,
            "dataset": target.dataset,
            "year": target.year,
        })
        for reference in target.references:
            reference_rows.append({
                "target_id": target.target_id,
                "peptide": target.peptide,
                "mhc_allele": target.mhc_allele,
                "cdr3b_reference": reference,
            })

    pd.DataFrame(target_rows).to_csv(data_dir / "targets.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(data_dir / "references.csv", index=False)
    audit = {
        "protocol": "public_epitope_conditioned_cdr3beta_track_a_v1",
        "benchmark_source": str(benchmark_csv),
        "mhc_pseudosequence_source": str(pseudo_csv),
        "reference_column": "reference_translations",
        "prediction_columns_excluded_from_reference": [
            "tcrt5_translations", "gratcr_translations", "er_translations",
            "tcrt5_greedy", "gratcr_greedy", "er_greedy",
        ],
        "n_targets": len(targets),
        "n_references_total": sum(t.n_references for t in targets.values()),
        "reference_counts": {key: target.n_references for key, target in targets.items()},
        "rvr_target": asdict(targets[RVR_TARGET_ID]) | {
            "references": f"{targets[RVR_TARGET_ID].n_references} sequences omitted from audit display"
        },
        "immprep": inspect_immprep(immprep_csv),
    }
    (data_dir / "benchmark_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    return audit


def validity_reason(sequence: object) -> str:
    """Return ``valid`` or the first failed public Track-A CDR3beta rule."""

    seq = clean_sequence(sequence)
    if not seq:
        return "empty"
    if any(amino_acid not in STANDARD_AA for amino_acid in seq):
        return "non_standard_amino_acid"
    if not 7 <= len(seq) <= 24:
        return "length_outside_7_24"
    if seq[0] != "C":
        return "missing_initial_C"
    if seq[-1] not in {"F", "W"}:
        return "missing_terminal_F_or_W"
    return "valid"


def is_valid_cdr3b(sequence: object) -> bool:
    return validity_reason(sequence) == "valid"


def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance with O(min(len(left), len(right))) memory."""

    left = clean_sequence(left)
    right = clean_sequence(right)
    if _rapidfuzz_levenshtein is not None:
        return int(_rapidfuzz_levenshtein.distance(left, right))
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_left != char_right),
            ))
        previous = current
    return previous[-1]


def sequence_recovery(generated: str, reference: str) -> float:
    generated = clean_sequence(generated)
    reference = clean_sequence(reference)
    denominator = max(len(generated), len(reference))
    if denominator == 0:
        return 1.0
    return 1.0 - edit_distance(generated, reference) / denominator


def best_sequence_recovery(generated: str, references: Sequence[str]) -> float:
    if not references:
        return float("nan")
    generated = clean_sequence(generated)
    if _rapidfuzz_process is not None and _rapidfuzz_levenshtein is not None:
        match = _rapidfuzz_process.extractOne(
            generated,
            references,
            scorer=_rapidfuzz_levenshtein.normalized_similarity,
        )
        return float(match[1])
    return max(sequence_recovery(generated, reference) for reference in references)


def _unique_in_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def empty_generation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=GENERATION_COLUMNS + OPTIONAL_GENERATION_COLUMNS)


def coerce_generation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in GENERATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"generation table is missing required columns: {missing}")
    result = frame.copy()
    for column in ("model", "target_id", "peptide", "mhc_allele", "cdr3b"):
        result[column] = result[column].map(clean_sequence if column in {"peptide", "cdr3b"} else str)
    result["rank"] = pd.to_numeric(result["rank"], errors="raise").astype(int)
    if (result["rank"] < 1).any():
        raise ValueError("generation ranks must start at 1")
    duplicate_rank = result.duplicated(["model", "target_id", "rank"], keep=False)
    if duplicate_rank.any():
        examples = result.loc[duplicate_rank, ["model", "target_id", "rank"]].head().to_dict("records")
        raise ValueError(f"duplicate model/target ranks: {examples}")
    for column in OPTIONAL_GENERATION_COLUMNS:
        if column not in result:
            result[column] = ""
    return result[GENERATION_COLUMNS + OPTIONAL_GENERATION_COLUMNS]


def import_official_predictions(
    models: Sequence[str] = ("tcrt5", "gratcr"),
    benchmark_csv: Path = DEFAULT_BENCHMARK_CSV,
) -> pd.DataFrame:
    """Import author-released de-novo candidates, never the reference column."""

    model_columns = {
        "tcrt5": ("TCRT5", "tcrt5_translations"),
        "gratcr": ("GRATCR", "gratcr_translations"),
    }
    unknown = sorted(set(models) - set(model_columns))
    if unknown:
        raise ValueError(f"no official prediction artifact mapping for {unknown}")
    frame = pd.read_csv(benchmark_csv, low_memory=False)
    rows = []
    for _, source_row in frame.iterrows():
        peptide = clean_sequence(source_row["Epitope"])
        allele = normalise_allele(source_row["Allele"])
        target_id = canonical_target_id(peptide, allele)
        for model_key in models:
            model_label, prediction_column = model_columns[model_key]
            predictions = _parse_list(source_row[prediction_column])
            for rank, sequence in enumerate(predictions, start=1):
                rows.append({
                    "model": model_label,
                    "target_id": target_id,
                    "peptide": peptide,
                    "mhc_allele": allele,
                    "rank": rank,
                    "cdr3b": sequence,
                    "raw_score": float("nan"),
                    "generation_mode": "de_novo",
                    "source": "TCR-TRANSLATE benchmark_data_w_preds.csv official artifact",
                    "run_id": "official_artifact",
                })
    return coerce_generation_frame(pd.DataFrame(rows))


def generation_rows(
    model: str,
    target: Target,
    candidates: Sequence[object],
    *,
    generation_mode: str = "de_novo",
    source: str = "official_runtime",
    run_id: str = "",
) -> pd.DataFrame:
    """Convert adapter candidates into the mandatory row-level interchange schema."""

    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        if isinstance(candidate, str):
            sequence, score = candidate, float("nan")
        elif isinstance(candidate, Mapping):
            sequence = candidate.get("cdr3b", "")
            score = candidate.get("raw_score", float("nan"))
        else:
            sequence = getattr(candidate, "cdr3b")
            score = getattr(candidate, "raw_score", float("nan"))
        rows.append({
            "model": model,
            "target_id": target.target_id,
            "peptide": target.peptide,
            "mhc_allele": target.mhc_allele,
            "rank": rank,
            "cdr3b": clean_sequence(sequence),
            "raw_score": score,
            "generation_mode": generation_mode,
            "source": source,
            "run_id": run_id,
        })
    return coerce_generation_frame(pd.DataFrame(rows)) if rows else empty_generation_frame()


def read_generations(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return empty_generation_frame()
    return coerce_generation_frame(pd.read_csv(path))


def write_generations(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coerce_generation_frame(frame).to_csv(path, index=False)


def migrate_legacy_tcrt5_paper_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert legacy length-normalised TCRT5 scores to paper likelihood ranks.

    This is intentionally a guarded, non-idempotent migration for canonical
    rows decoded before cumulative transition scores were exposed by the
    adapter.  T5's generated hypothesis contains the decoded amino-acid tokens
    plus the TCR prefix and EOS token, so cumulative log-likelihood equals the
    legacy length-normalised score multiplied by ``len(cdr3b) + 2``.
    """

    frame = coerce_generation_frame(frame)
    tcrt5 = frame[frame["model"] == "TCRT5"].copy()
    other = frame[frame["model"] != "TCRT5"].copy()
    if tcrt5.empty:
        raise ValueError("generation table has no TCRT5 rows to migrate")
    scores = pd.to_numeric(tcrt5["raw_score"], errors="coerce")
    if scores.isna().any():
        raise ValueError("legacy TCRT5 rows must all have numeric raw_score values")
    if not scores.between(-2.0, 0.0).all():
        raise ValueError(
            "TCRT5 scores do not look length-normalised; refusing a non-idempotent migration"
        )
    for _, block in tcrt5.groupby("target_id", sort=False):
        if not block.sort_values("rank", kind="stable")["raw_score"].is_monotonic_decreasing:
            raise ValueError("legacy TCRT5 raw scores are not monotonic in recorded beam rank")

    tcrt5["raw_score"] = scores * (tcrt5["cdr3b"].str.len() + 2)
    tcrt5 = tcrt5.sort_values(
        ["target_id", "raw_score"], ascending=[True, False], kind="stable"
    )
    tcrt5["rank"] = tcrt5.groupby("target_id", sort=False).cumcount() + 1
    tcrt5["run_id"] = tcrt5["run_id"].map(
        lambda value: f"{value}_paper_loglik_rank" if value else "paper_loglik_rank"
    )
    return coerce_generation_frame(pd.concat([tcrt5, other], ignore_index=True))


def score_target(
    block: pd.DataFrame,
    target: Target,
    *,
    num_candidates: int = 1000,
    giana: GianaMetrics | None = None,
) -> dict[str, object]:
    """Score one model-target block; missing requested rows count as invalid."""

    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    ordered = block.sort_values("rank", kind="stable")
    selected = ordered[ordered["rank"] <= num_candidates]
    raw = [clean_sequence(sequence) for sequence in selected["cdr3b"].tolist()]
    valid_mask = [is_valid_cdr3b(sequence) for sequence in raw]
    valid = [sequence for sequence, keep in zip(raw, valid_mask) if keep]
    unique_valid = _unique_in_order(valid)
    references = set(target.references)

    def exact_at(k: int) -> set[str]:
        # @K is defined by the recorded raw rank, not by the first K rows that
        # happen to be present.  This distinction matters for incomplete files:
        # a missing early rank must not pull rank K+1 into the prefix.
        rank_limit = min(k, num_candidates)
        prefix = [
            clean_sequence(sequence)
            for sequence in selected.loc[selected["rank"] <= rank_limit, "cdr3b"].tolist()
        ]
        valid_prefix = _unique_in_order(seq for seq in prefix if is_valid_cdr3b(seq))
        return set(valid_prefix) & references

    exact100 = exact_at(100)
    exact1000 = exact_at(min(1000, num_candidates))
    recoveries = [best_sequence_recovery(sequence, target.references) for sequence in unique_valid]
    recovery90 = sum(value >= 0.9 for value in recoveries)
    invalid_reasons: dict[str, int] = {}
    for sequence in raw:
        reason = validity_reason(sequence)
        if reason != "valid":
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    if len(raw) < num_candidates:
        invalid_reasons["missing_candidate"] = num_candidates - len(raw)

    result: dict[str, object] = {
        "Model": str(selected["model"].iloc[0]) if len(selected) else str(block["model"].iloc[0]),
        "target_id": target.target_id,
        "peptide": target.peptide,
        "mhc_allele": target.mhc_allele,
        "n_references": target.n_references,
        "n_requested": num_candidates,
        "n_generated": len(raw),
        "n_valid": len(valid),
        "n_unique_valid": len(unique_valid),
        "ValidRate": len(valid) / num_candidates,
        "UniqueRate": len(unique_valid) / len(valid) if valid else 0.0,
        "ExactHit@100": len(exact100),
        "ExactHit@1000": len(exact1000),
        "ReferenceRecall@1000": len(exact1000) / target.n_references if target.n_references else float("nan"),
        "Recovery90Hit@1000": recovery90,
        "MedianBestRecovery": float(pd.Series(recoveries).median()) if recoveries else float("nan"),
        "GIANAHit": giana.generated_hit if giana else float("nan"),
        "GIANAReferenceClusterCoverage": giana.reference_clusters_covered if giana else float("nan"),
        "GIANAReferenceSequenceCoverage": giana.reference_sequences_covered if giana else float("nan"),
        "GIANAEligibleGenerated": giana.generated_eligible if giana else float("nan"),
        "GIANAStatus": giana.status if giana else "not_run",
        "invalid_reasons": json.dumps(invalid_reasons, sort_keys=True),
    }
    return result


def cross_epitope_jaccard(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Jaccard over each model's per-target unique valid CDR3beta sets."""

    frame = coerce_generation_frame(frame)
    rows = []
    for model, model_frame in frame.groupby("model", sort=True):
        sets = {
            target_id: set(
                sequence for sequence in block.sort_values("rank")["cdr3b"].map(clean_sequence)
                if is_valid_cdr3b(sequence)
            )
            for target_id, block in model_frame.groupby("target_id", sort=True)
        }
        target_ids = sorted(sets)
        for index, left in enumerate(target_ids):
            for right in target_ids[index + 1:]:
                intersection = sets[left] & sets[right]
                union = sets[left] | sets[right]
                rows.append({
                    "Model": model,
                    "target_a": left,
                    "target_b": right,
                    "intersection": len(intersection),
                    "union": len(union),
                    "Jaccard": len(intersection) / len(union) if union else 0.0,
                })
    return pd.DataFrame(rows, columns=[
        "Model", "target_a", "target_b", "intersection", "union", "Jaccard"
    ])


def _sum_if_available(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.sum()) if values.notna().any() else float("nan")


def summarise_metrics(per_target: pd.DataFrame, jaccard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, block in per_target.groupby("Model", sort=True):
        n_valid = int(block["n_valid"].sum())
        n_requested = int(block["n_requested"].sum())
        n_unique = int(block["n_unique_valid"].sum())
        n_references = int(block["n_references"].sum())
        exact = int(block["ExactHit@1000"].sum())
        model_jaccard = jaccard[jaccard["Model"] == model] if len(jaccard) else jaccard
        rows.append({
            "Model": model,
            "Targets": int(len(block)),
            "ValidRate": n_valid / n_requested if n_requested else float("nan"),
            "UniqueRate": n_unique / n_valid if n_valid else 0.0,
            "ExactHit@100": int(block["ExactHit@100"].sum()),
            "ExactHit@1000": exact,
            "ReferenceRecall": exact / n_references if n_references else float("nan"),
            "Recovery90Hit": int(block["Recovery90Hit@1000"].sum()),
            "GIANAHit": _sum_if_available(block["GIANAHit"]),
            "GIANAReferenceClusterCoverage": _sum_if_available(block["GIANAReferenceClusterCoverage"]),
            "GIANAReferenceSequenceCoverage": _sum_if_available(block["GIANAReferenceSequenceCoverage"]),
            "CrossEpitopeJaccardMean": (
                float(model_jaccard["Jaccard"].mean()) if len(model_jaccard) else float("nan")
            ),
            "MacroValidRate": float(block["ValidRate"].mean()),
            "MacroUniqueRate": float(block["UniqueRate"].mean()),
            "MacroReferenceRecall": float(block["ReferenceRecall@1000"].mean()),
        })
    return pd.DataFrame(rows)


def score_generations(
    generations: pd.DataFrame,
    targets: Mapping[str, Target],
    *,
    num_candidates: int = 1000,
    giana_metrics: Mapping[tuple[str, str], GianaMetrics] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-target metrics, model summary, and cross-target Jaccards."""

    generations = coerce_generation_frame(generations)
    unknown_targets = sorted(set(generations["target_id"]) - set(targets))
    if unknown_targets:
        raise ValueError(f"generation table contains unknown targets: {unknown_targets}")
    rows = []
    giana_metrics = giana_metrics or {}
    for (model, target_id), block in generations.groupby(["model", "target_id"], sort=True):
        rows.append(score_target(
            block,
            targets[target_id],
            num_candidates=num_candidates,
            giana=giana_metrics.get((str(model), str(target_id))),
        ))
    per_target = pd.DataFrame(rows)
    jaccard = cross_epitope_jaccard(generations)
    summary = summarise_metrics(per_target, jaccard) if len(per_target) else pd.DataFrame()
    return per_target, summary, jaccard


def write_metric_outputs(
    generations: pd.DataFrame,
    per_target: pd.DataFrame,
    summary: pd.DataFrame,
    jaccard: pd.DataFrame,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    write_generations(generations, results_dir / "generations.csv")
    per_target.to_csv(results_dir / "per_target_metrics.csv", index=False)
    summary.to_csv(results_dir / "summary_metrics.csv", index=False)
    jaccard.to_csv(results_dir / "cross_epitope_jaccard.csv", index=False)
