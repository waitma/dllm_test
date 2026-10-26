"""TcrDesign end-to-end generated-beta -> generated-alpha orchestration.

Run the smoke path with::

    /project_c20250601/251105016/conda/envs/pllm/bin/python \
      /project_c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py \
      generate-tcrdesign-alpha --targets rvr --beta-top-k 1 --alpha-per-beta 10

This module deliberately contains no custom alpha-chain metric.  The released
TcrDesign source provides alpha generation, but the paper's alpha BLOSUM62 and
edit-distance plotting/evaluation code is not present in the public repository
or Zenodo weight archive.  Until a runnable official evaluator is released,
the paired rows produced here are qualitative artifacts and never enter the
public beta Track-A leaderboard.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from tcr_generation_bench.public_generators import (
    BASELINES,
    TCRDESIGN_PYTHON,
    TCRDESIGN_REPO,
    _subprocess_environment,
)


ALPHA_GENERATION_COLUMNS = [
    "target_id",
    "epitope",
    "generated_cdr3b",
    "generated_cdr3a",
    "beta_rank",
    "alpha_rank",
    "raw_beta_score",
    "raw_alpha_score",
]
ALPHA_PROVENANCE_COLUMNS = [
    "mhc_allele",
    "generation_mode",
    "source",
    "run_id",
]
ALPHA_EVALUATION_AUDIT = "tcrdesign_alpha_evaluation_audit.json"


def select_tcrdesign_generated_betas(
    generations: pd.DataFrame,
    target_ids: Iterable[str],
    beta_top_k: int,
) -> pd.DataFrame:
    """Select generated TcrDesign betas only; references are never accepted."""

    if beta_top_k <= 0:
        raise ValueError("beta_top_k must be positive")
    required = {
        "model", "target_id", "peptide", "mhc_allele", "rank", "cdr3b", "raw_score",
    }
    missing = sorted(required - set(generations.columns))
    if missing:
        raise ValueError(f"generation table is missing required columns: {missing}")

    requested = set(target_ids)
    selected = generations.loc[
        generations["model"].eq("TcrDesign")
        & generations["target_id"].isin(requested)
    ].copy()
    if selected.empty:
        raise ValueError("no TcrDesign generated-beta rows matched the requested targets")
    if "generation_mode" in selected and not selected["generation_mode"].eq("de_novo").all():
        modes = sorted(selected["generation_mode"].dropna().astype(str).unique().tolist())
        raise ValueError(f"TcrDesign beta inputs must be de_novo generations, found {modes}")

    selected["rank"] = pd.to_numeric(selected["rank"], errors="raise").astype(int)
    selected = selected.loc[selected["rank"].between(1, beta_top_k)].copy()
    duplicated = selected.duplicated(["target_id", "rank"], keep=False)
    if duplicated.any():
        keys = selected.loc[duplicated, ["target_id", "rank"]].drop_duplicates()
        raise ValueError(f"duplicate TcrDesign target/rank beta rows: {keys.to_dict('records')}")

    counts = selected.groupby("target_id").size().to_dict()
    missing_targets = sorted(requested - set(counts))
    incomplete = {
        target_id: int(counts.get(target_id, 0))
        for target_id in sorted(requested)
        if counts.get(target_id, 0) != beta_top_k
    }
    if missing_targets or incomplete:
        raise ValueError(
            f"expected {beta_top_k} generated betas per target; "
            f"missing={missing_targets}, counts={incomplete}"
        )

    selected = selected.sort_values(["target_id", "rank"], kind="stable")
    return pd.DataFrame({
        "condition_index": range(len(selected)),
        "target_id": selected["target_id"].astype(str).tolist(),
        "epitope": selected["peptide"].astype(str).str.upper().tolist(),
        "mhc_allele": selected["mhc_allele"].astype(str).tolist(),
        "generated_cdr3b": selected["cdr3b"].astype(str).str.upper().tolist(),
        "beta_rank": selected["rank"].astype(int).tolist(),
        "raw_beta_score": selected["raw_score"].tolist(),
    })


def validate_alpha_generations(
    frame: pd.DataFrame,
    beta_conditions: pd.DataFrame,
    alpha_per_beta: int,
) -> None:
    """Check structural completeness only, without scoring alpha sequences."""

    required = set(ALPHA_GENERATION_COLUMNS + ALPHA_PROVENANCE_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"alpha generation output is missing columns: {missing}")
    expected = len(beta_conditions) * alpha_per_beta
    if len(frame) != expected:
        raise ValueError(f"expected {expected} alpha rows, found {len(frame)}")
    sizes = frame.groupby(["target_id", "beta_rank"], dropna=False).size()
    if not sizes.eq(alpha_per_beta).all():
        bad = sizes.loc[~sizes.eq(alpha_per_beta)].to_dict()
        raise ValueError(f"alpha candidate counts differ from {alpha_per_beta}: {bad}")
    expected_ranks = list(range(1, alpha_per_beta + 1))
    for _, group in frame.groupby(["target_id", "beta_rank"], sort=False):
        if group["alpha_rank"].astype(int).tolist() != expected_ranks:
            raise ValueError("alpha ranks must be contiguous and preserve official beam order")
    if frame["generated_cdr3a"].isna().any():
        raise ValueError("official alpha worker returned missing sequences")


def official_alpha_evaluation_audit(repo: Path = TCRDESIGN_REPO) -> dict[str, object]:
    """Return the evidence-backed release audit; no substitute metric is added."""

    generation_entry = repo / "tcrdesign_G.py"
    generation_impl = repo / "utils" / "generate_tcr.py"
    full_pipeline = repo / "utils" / "generate_whole.py"
    alpha_weight = repo / "weights" / "rnn" / "beta_alpha" / "RNN_tcr.ep"
    clip_model = repo / "clip" / "clip_model.py"
    clip_trainer = repo / "clip" / "clip_trainer.py"
    clip_dataset = repo / "clip" / "clip_dataset.py"
    released_weight_names = sorted(
        str(path.relative_to(repo)) for path in (repo / "weights").rglob("*") if path.is_file()
    )
    pairing_weights = [
        name for name in released_weight_names
        if "clip" in name.lower() or "pair" in name.lower()
    ]
    return {
        "audit_date": date.today().isoformat(),
        "official_repository": "https://github.com/XSLiuLab/TcrDesign",
        "official_weights_and_data": "https://zenodo.org/records/14545852",
        "paper": "https://www.biorxiv.org/content/10.64898/2026.01.15.699824v1",
        "supplement": "https://files.sciengine.com/gridfs/2049020319368216576/edit_Supplementary_Materials1.pdf",
        "alpha_generation_available": all(
            path.is_file() for path in (generation_entry, generation_impl, full_pipeline, alpha_weight)
        ),
        "official_alpha_evaluation_available": False,
        "included_in_quantitative_benchmark": False,
        "decision": "qualitative_output_only",
        "paper_metrics_reported_but_code_not_released": [
            "alpha-CDR3 BLOSUM62 similarity to natural alpha-CDR3",
            "minimum edit distance from generated alpha-CDR3 to natural alpha-CDR3",
            "TcrDesign-B binding score shown as an auxiliary model-specific score",
        ],
        "official_generation": {
            "cli_path": str(generation_entry.resolve()),
            "implementation_path": str(generation_impl.resolve()),
            "full_pipeline_path": str(full_pipeline.resolve()),
            "checkpoint_path": str(alpha_weight.resolve()),
            "cli_input": "epitope string plus generated beta-CDR3 string",
            "cli_output": "CSV with one TCRs column containing generated alpha-CDR3",
            "official_defaults": {
                "full_pipeline_beta_count": 100,
                "full_pipeline_alpha_per_beta": 10,
                "alpha_temperature": 0.8,
                "alpha_max_length": 20,
            },
        },
        "released_pair_representation_code": {
            "model_path": str(clip_model.resolve()),
            "trainer_path": str(clip_trainer.resolve()),
            "dataset_path": str(clip_dataset.resolve()),
            "input_csv_columns": ["CDR3a", "CDR3b"],
            "implemented_output": "in-batch alpha-by-beta contrastive logits and train/test loss",
            "released_evaluation_cli": False,
            "released_retrieval_or_pairing_metrics": False,
            "released_pairing_checkpoint": False,
            "matching_weight_files": pairing_weights,
        },
        "released_edit_distance_code": {
            "path": str(full_pipeline.resolve()),
            "function": "finder_TCRs / compare",
            "purpose": (
                "cross-reactivity matching between two generated TCR sets; not an "
                "alpha-reference evaluation script and not the paper Figure S15 runner"
            ),
            "reused_as_alpha_benchmark": False,
        },
        "exclusion_reason": (
            "The release contains alpha generation and training components, but no directly "
            "runnable alpha-reference or alpha-beta pairing evaluation script, no evaluation "
            "CLI/parameterization for the paper's alpha figures, and no CLIP pairing checkpoint. "
            "TcrDesign-B is a pMHC-full-TCR binding model and is not used as a common external "
            "alpha-beta pairing evaluator."
        ),
        "zenodo_archive_scope": {
            "files": ["tcrdesign_data.tar.gz", "tcrdesign_weights.tar.gz"],
            "weight_archive_contains_alpha_generator": alpha_weight.is_file(),
            "weight_archive_contains_pairing_evaluator": bool(pairing_weights),
        },
    }


def write_official_alpha_evaluation_audit(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / ALPHA_EVALUATION_AUDIT
    path.write_text(
        json.dumps(official_alpha_evaluation_audit(), indent=2, ensure_ascii=False) + "\n"
    )
    return path


class TcrDesignAlphaGenerator:
    """Batched adapter around the released TcrDesign alpha model and beam search."""

    def __init__(
        self,
        python: Path = TCRDESIGN_PYTHON,
        device: str = "cuda",
        temperature: float = 0.8,
        max_len: int = 20,
        batch_size: int = 32,
    ) -> None:
        self.python = Path(python)
        self.device = device
        self.temperature = temperature
        self.max_len = max_len
        self.batch_size = batch_size
        self.last_command: list[str] | None = None

    @property
    def worker_path(self) -> Path:
        return Path(__file__).with_name("tcrdesign_alpha_worker.py")

    @property
    def weight_path(self) -> Path:
        return TCRDESIGN_REPO / "weights" / "rnn" / "beta_alpha" / "RNN_tcr.ep"

    def check_ready(self) -> tuple[bool, str]:
        required = [
            self.python,
            self.worker_path,
            TCRDESIGN_REPO / "tcrdesign_G.py",
            TCRDESIGN_REPO / "utils" / "generate_tcr.py",
            self.weight_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            return False, f"missing official alpha-generation runtime artifacts: {missing}"
        return True, "official alpha source/checkpoint and isolated runtime are present"

    def generate_frame(
        self,
        beta_conditions: pd.DataFrame,
        alpha_per_beta: int,
        run_id: str,
    ) -> pd.DataFrame:
        ready, reason = self.check_ready()
        if not ready:
            raise RuntimeError(reason)
        if alpha_per_beta <= 0:
            raise ValueError("alpha_per_beta must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        with tempfile.TemporaryDirectory(prefix="tcrdesign_alpha_pairs_") as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "generated_betas.csv"
            output_path = temp / "generated_alphas.csv"
            beta_conditions.to_csv(input_path, index=False)
            command = [
                str(self.python),
                str(self.worker_path),
                "--input", str(input_path),
                "--output", str(output_path),
                "--tcrdesign-repo", str(TCRDESIGN_REPO),
                "--alpha-per-beta", str(alpha_per_beta),
                "--max-len", str(self.max_len),
                "--temperature", str(self.temperature),
                "--batch-size", str(self.batch_size),
                "--device", self.device,
            ]
            self.last_command = command
            process = subprocess.run(
                command,
                cwd=BASELINES,
                env=_subprocess_environment(self.python),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if process.returncode != 0:
                tail = "\n".join(process.stdout.splitlines()[-50:])
                raise RuntimeError(
                    f"TcrDesign alpha worker failed ({process.returncode}):\n{tail}"
                )
            frame = pd.read_csv(output_path, keep_default_na=False)

        frame["raw_alpha_score"] = pd.to_numeric(
            frame["raw_alpha_score"], errors="coerce"
        )
        frame["raw_beta_score"] = pd.to_numeric(
            frame["raw_beta_score"], errors="coerce"
        )
        frame["generation_mode"] = "end_to_end_generated_beta"
        frame["source"] = "official_TcrDesign_alpha_model_and_beam_search_runtime"
        frame["run_id"] = run_id
        frame = frame[ALPHA_GENERATION_COLUMNS + ALPHA_PROVENANCE_COLUMNS]
        validate_alpha_generations(frame, beta_conditions, alpha_per_beta)
        return frame


def write_alpha_generations(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
