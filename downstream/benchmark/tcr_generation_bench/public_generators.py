"""Unified adapters for the four public epitope-conditioned TCR generators.

Runtime smoke example::

    /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py \
      generate --model tcrt5 --targets rvr --num-candidates 10

Every de-novo adapter implements ``generate(peptide, mhc_allele=None,
mhc_pseudosequence=None, num_candidates=1000)``.  The adapters only bridge to
official code and official weights; sequence repair and post-hoc anchor
insertion are deliberately absent.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


BENCH = Path(__file__).resolve().parents[1]
BASELINES = BENCH / "baselines"
TCRT5_MODEL_ID = "dkarthikeyan1/tcrt5_ft_tcrdb"
TCRDESIGN_REPO = BASELINES / "TcrDesign"
TCRDESIGN_PYTHON = Path("/vepfs-mlp2/c20250601/251105016/conda/envs/tcrdesign/bin/python")
GRATCR_REPO = BASELINES / "GRATCR"
GRATCR_PYTHON = Path("/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python")
EPIDIFF_REPO = BASELINES / "TCR-epiDiff"


def _subprocess_environment(python: Path) -> dict[str, str]:
    """Prefer the selected Conda runtime's shared libraries in subprocesses."""

    environment = os.environ.copy()
    runtime_lib = Path(python).resolve().parents[1] / "lib"
    current = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        part for part in (str(runtime_lib), current) if part
    )
    return environment


@dataclass(frozen=True)
class Candidate:
    """One decoded raw candidate; score is absent when upstream does not expose it."""

    cdr3b: str
    raw_score: float | None = None


def _rank_scored_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Return a stable descending likelihood order for fully scored candidates."""

    if any(candidate.raw_score is None for candidate in candidates):
        raise ValueError("all candidates must have a score before likelihood ranking")
    return sorted(candidates, key=lambda candidate: candidate.raw_score, reverse=True)


class ReconstructionOnlyError(RuntimeError):
    """Raised when a released model has no valid de-novo sampling path."""


class PublicGenerator(ABC):
    name: str
    generation_mode: str = "de_novo"

    @abstractmethod
    def generate(
        self,
        peptide: str,
        mhc_allele: str | None = None,
        mhc_pseudosequence: str | None = None,
        num_candidates: int = 1000,
    ) -> list[Candidate]:
        """Generate ranked CDR3beta candidates using an official inference path."""

    @abstractmethod
    def check_ready(self) -> tuple[bool, str]:
        """Check source/runtime/weight presence without generating."""


class TCRT5Generator(PublicGenerator):
    """Official Hugging Face TCRT5 beam-search adapter."""

    name = "TCRT5"

    def __init__(
        self,
        model_id: str = TCRT5_MODEL_ID,
        device: str = "cuda",
        num_beams_multiplier: int = 3,
        max_new_tokens: int = 25,
    ) -> None:
        self.model_id = model_id
        self.device_requested = device
        self.num_beams_multiplier = num_beams_multiplier
        self.max_new_tokens = max_new_tokens
        self._runtime = None

    def check_ready(self) -> tuple[bool, str]:
        try:
            import sentencepiece  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - status must preserve the cause
            return False, f"runtime import failed: {exc}"
        return True, f"runtime ready; model resolves from Hugging Face id {self.model_id}"

    def _load(self):
        if self._runtime is not None:
            return self._runtime
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained(self.model_id)
        model = T5ForConditionalGeneration.from_pretrained(self.model_id).eval()
        device = (
            "cuda" if self.device_requested == "cuda" and torch.cuda.is_available()
            else self.device_requested
        )
        model = model.to(device)
        self._runtime = (tokenizer, model, device)
        return self._runtime

    def generate(
        self,
        peptide: str,
        mhc_allele: str | None = None,
        mhc_pseudosequence: str | None = None,
        num_candidates: int = 1000,
    ) -> list[Candidate]:
        if not mhc_pseudosequence:
            raise ValueError("TCRT5 requires mhc_pseudosequence")
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        import torch

        tokenizer, model, device = self._load()
        prompt = f"[PMHC]{str(peptide).upper()}[SEP]{str(mhc_pseudosequence).upper()}[EOS]"
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=num_candidates,
                num_beams=self.num_beams_multiplier * num_candidates,
                return_dict_in_generate=True,
                output_scores=True,
            )
        decoded = [
            re.sub(r"\[.*?\]", "", sequence).strip().upper()
            for sequence in tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        ]
        # Hugging Face returns beam hypotheses ordered by a length-normalised
        # ``sequences_scores`` value.  TCR-TRANSLATE's paper, released benchmark
        # artifact, and rank-cutoff analysis instead rank by the complete model
        # log-likelihood.  Summing the per-token transition scores reproduces
        # that cumulative likelihood without changing the decoded candidate set.
        transition_scores = model.compute_transition_scores(
            output.sequences,
            output.scores,
            output.beam_indices,
            normalize_logits=False,
        )
        cumulative_scores = transition_scores.sum(dim=1).detach().cpu().tolist()
        candidates = [
            Candidate(sequence, float(score))
            for sequence, score in zip(decoded, cumulative_scores)
        ]
        return _rank_scored_candidates(candidates)


class GRATCRGenerator(PublicGenerator):
    """Subprocess adapter around the official ``GRA.py --mode generate`` CLI."""

    name = "GRATCR"

    def __init__(self, python: Path = GRATCR_PYTHON, max_len: int = 32) -> None:
        self.python = Path(python)
        self.max_len = max_len

    @property
    def weight_paths(self) -> tuple[Path, ...]:
        model_dir = GRATCR_REPO / "model"
        return (
            model_dir / "gra.pth",
            model_dir / "bert_pretrain.pth",
            model_dir / "gpt_pretrain.pth",
        )

    def check_ready(self) -> tuple[bool, str]:
        if not (GRATCR_REPO / "GRA.py").is_file():
            return False, f"missing official source {GRATCR_REPO / 'GRA.py'}"
        if not self.python.is_file():
            return False, f"missing runtime {self.python}"
        missing = [str(path) for path in self.weight_paths if not path.is_file()]
        if missing:
            return False, f"missing official weights: {', '.join(missing)}"
        return True, "official source, three checkpoints, and isolated runtime are present"

    def generate(
        self,
        peptide: str,
        mhc_allele: str | None = None,
        mhc_pseudosequence: str | None = None,
        num_candidates: int = 1000,
    ) -> list[Candidate]:
        ready, reason = self.check_ready()
        if not ready:
            raise RuntimeError(reason)
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        with tempfile.TemporaryDirectory(prefix="gratcr_generate_") as temp_dir:
            temp = Path(temp_dir)
            input_csv = temp / "target.csv"
            output_csv = temp / "result.csv"
            # Official generate-mode preprocessing first performs an 80/20
            # train/validation split even though it later returns only the set
            # of unique epitope tensors.  A one-row file therefore crashes at
            # ``zip(*train_batch)``.  Duplicate the same target row so both
            # temporary splits are non-empty; ``set(pep_seq_list)`` still emits
            # exactly one conditioning epitope.  The fixed beta placeholder is
            # not a test reference and never reaches the generation model.
            with input_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epitope", "beta"])
                writer.writeheader()
                writer.writerow({"epitope": str(peptide).upper(), "beta": "C"})
                writer.writerow({"epitope": str(peptide).upper(), "beta": "C"})
            command = [
                str(self.python), str(GRATCR_REPO / "GRA.py"),
                "--data_path", str(input_csv),
                "--tcr_vocab_path", str(GRATCR_REPO / "Data" / "vocab" / "total-beta.csv"),
                "--pep_vocab_path", str(GRATCR_REPO / "Data" / "vocab" / "total-epitope.csv"),
                "--model_path", str(self.weight_paths[0]),
                "--bert_path", str(self.weight_paths[1]),
                "--gpt_path", str(self.weight_paths[2]),
                "--mode", "generate",
                "--result_path", str(output_csv),
                "--beam", str(num_candidates),
                "--maxlen", str(self.max_len),
                "--batch_size", "1",
                "--random_seed", "0",
            ]
            environment = _subprocess_environment(self.python)
            environment.setdefault("PYTHONHASHSEED", "0")
            process = subprocess.run(
                command,
                cwd=GRATCR_REPO,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if process.returncode != 0:
                tail = "\n".join(process.stdout.splitlines()[-30:])
                raise RuntimeError(f"official GRATCR CLI failed ({process.returncode}):\n{tail}")
            with output_csv.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            candidates = [Candidate(str(row.get("beta", "")).strip().upper()) for row in rows]
            if len(candidates) != num_candidates:
                raise RuntimeError(
                    f"official GRATCR returned {len(candidates)} candidates, requested {num_candidates}"
                )
            return candidates


class TcrDesignBetaGenerator(PublicGenerator):
    """Official TcrDesign-G beta-only CLI adapter (no alpha/VJ/binding pipeline)."""

    name = "TcrDesign"

    def __init__(
        self,
        python: Path = TCRDESIGN_PYTHON,
        device: str = "cuda",
        temperature: float = 1.0,
        max_len: int = 20,
    ) -> None:
        self.python = Path(python)
        self.device = device
        self.temperature = temperature
        self.max_len = max_len

    @property
    def weight_path(self) -> Path:
        return TCRDESIGN_REPO / "weights" / "rnn" / "beta" / "RNN_tcr.ep"

    def check_ready(self) -> tuple[bool, str]:
        if not (TCRDESIGN_REPO / "tcrdesign_G.py").is_file():
            return False, f"missing official source {TCRDESIGN_REPO / 'tcrdesign_G.py'}"
        if not self.python.is_file():
            return False, f"missing runtime {self.python}"
        if not self.weight_path.is_file():
            return False, f"missing official beta checkpoint {self.weight_path}"
        return True, "official beta-generation source, checkpoint, and isolated runtime are present"

    def generate(
        self,
        peptide: str,
        mhc_allele: str | None = None,
        mhc_pseudosequence: str | None = None,
        num_candidates: int = 1000,
    ) -> list[Candidate]:
        ready, reason = self.check_ready()
        if not ready:
            raise RuntimeError(reason)
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        with tempfile.TemporaryDirectory(prefix="tcrdesign_beta_") as temp_dir:
            output_csv = Path(temp_dir) / "generation.csv"
            command = [
                str(self.python), str(TCRDESIGN_REPO / "tcrdesign_G.py"),
                "-mode", "beta",
                "-epitope", str(peptide).upper(),
                "-num", str(num_candidates),
                "-maxLen", str(self.max_len),
                "-t", str(self.temperature),
                "-cuda", "True" if self.device == "cuda" else "False",
                "-o", str(output_csv),
            ]
            process = subprocess.run(
                command,
                cwd=BASELINES,
                env=_subprocess_environment(self.python),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if process.returncode != 0:
                tail = "\n".join(process.stdout.splitlines()[-30:])
                raise RuntimeError(f"official TcrDesign-G CLI failed ({process.returncode}):\n{tail}")
            with output_csv.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            candidates = [Candidate(str(row.get("TCRs", "")).strip().upper()) for row in rows]
            if len(candidates) != num_candidates:
                raise RuntimeError(
                    f"official TcrDesign-G returned {len(candidates)} candidates, requested {num_candidates}"
                )
            return candidates


class TCREpiDiffGenerator(PublicGenerator):
    """Protocol guard for the released TCR-epiDiff reconstruction code."""

    name = "TCR-epiDiff"
    generation_mode = "reconstruction_only"
    reason = (
        "the released inference path assigns TCR = original_TCR[0], applies add_noise(TCR, ...), "
        "and denoises it; no pure-random x_T reverse sampler or de-novo decoding CLI is released"
    )

    @property
    def weight_path(self) -> Path:
        return EPIDIFF_REPO / "Model" / "TCR-epiDiff_Best_model.pth"

    @property
    def inference_path(self) -> Path:
        return EPIDIFF_REPO / "Tutorial_Code" / "TCR-epiDiff" / "TCR-epiDiff_main.py"

    def check_ready(self) -> tuple[bool, str]:
        missing = [
            str(path) for path in (self.weight_path, self.inference_path) if not path.is_file()
        ]
        if missing:
            return False, f"missing released artifacts: {', '.join(missing)}"
        source = self.inference_path.read_text()
        evidence = [
            "TCR = original_TCR[0].to(device)",
            "noisy_data, _ = add_noise(TCR",
        ]
        if not all(fragment in source for fragment in evidence):
            return False, "released source changed; reconstruction-only audit must be repeated"
        return False, f"reconstruction_only: {self.reason}"

    def generate(
        self,
        peptide: str,
        mhc_allele: str | None = None,
        mhc_pseudosequence: str | None = None,
        num_candidates: int = 1000,
    ) -> list[Candidate]:
        raise ReconstructionOnlyError(
            f"TCR-epiDiff is excluded from de-novo generation: {self.reason}"
        )


def generator_for(name: str, *, device: str = "cuda") -> PublicGenerator:
    key = name.strip().lower().replace("_", "-")
    if key == "tcrt5":
        return TCRT5Generator(device=device)
    if key == "gratcr":
        return GRATCRGenerator()
    if key in {"tcrdesign", "tcrdesign-beta"}:
        return TcrDesignBetaGenerator(device=device)
    if key in {"tcr-epidiff", "tcrepidiff"}:
        return TCREpiDiffGenerator()
    raise ValueError(f"unknown public generator: {name}")


def readiness_table(names: Sequence[str] = ("tcrt5", "gratcr", "tcrdesign", "tcr-epidiff")):
    rows = []
    for name in names:
        generator = generator_for(name)
        ready, reason = generator.check_ready()
        rows.append({
            "model": generator.name,
            "generation_mode": generator.generation_mode,
            "ready_for_de_novo": ready and generator.generation_mode == "de_novo",
            "status": reason,
        })
    return rows
