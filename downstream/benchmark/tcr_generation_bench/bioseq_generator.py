"""Track-A adapter for retained BioSeq 7L checkpoints.

The public benchmark interface is intentionally the same as the baseline
adapters, although the current downstream sampling path uses
only the epitope string and ignores the optional MHC fields.  Candidate order
is deterministic emission order under the recorded seed; masked diffusion does
not expose a comparable per-sequence likelihood, so ``raw_score`` is empty.

Run through ``run_public_beta.py generate-bioseq --checkpoint /abs/best.pt``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .public_generators import Candidate


PROJECT_ROOT = Path("/project_c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BIOSEQ_7L_STEP117000 = (
    PROJECT_ROOT / "output" / "grammar_v2_esmc300m_integrated_llada_7l_step117000" / "best.pt"
)
DEFAULT_BIOSEQ_7L_STEP189000 = (
    PROJECT_ROOT / "output" / "grammar_v2_esmc300m_integrated_llada_7l_step189000" / "best.pt"
)
DEFAULT_BIOSEQ_7L_STEP117000_MANIFEST = DEFAULT_BIOSEQ_7L_STEP117000.with_name(
    "checkpoint_manifest.json"
)
BIOSEQ_7L_STEP117000_LABEL = "BioSeq-7L-step117000"
BIOSEQ_7L_STEP189000_LABEL = "BioSeq-7L-step189000"


class BioSeqTrackAGenerator:
    """Epitope-conditioned CDR3beta sampler backed by a local BioSeq checkpoint."""

    name = BIOSEQ_7L_STEP117000_LABEL
    generation_mode = "de_novo_epitope_conditioned"
    conditioning = "epitope_only"
    score_semantics = "unscored_seeded_emission_order"

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_BIOSEQ_7L_STEP117000,
        *,
        device: str = "cuda",
        seed: int = 42,
        batch_size: int = 100,
        max_iter: int = 32,
        temperature: float = 1.0,
        sampling_strategy: str = "gumbel_argmax",
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.manifest_path = self.checkpoint.with_name("checkpoint_manifest.json")
        manifest = self.checkpoint_manifest()
        raw_step = manifest.get("step")
        self.checkpoint_step = int(raw_step) if raw_step is not None else None
        self.name = (
            f"BioSeq-7L-step{self.checkpoint_step}"
            if self.checkpoint_step is not None
            else "BioSeq-7L-unknown-step"
        )
        self.device = device
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.temperature = float(temperature)
        self.sampling_strategy = str(sampling_strategy)
        self._sampler = None

    def checkpoint_manifest(self) -> dict[str, object]:
        if not self.manifest_path.is_file():
            return {}
        return json.loads(self.manifest_path.read_text())

    def check_ready(self) -> tuple[bool, str]:
        if not self.checkpoint.is_file():
            return False, f"missing BioSeq checkpoint {self.checkpoint}"
        if not self.manifest_path.is_file():
            return False, f"missing checkpoint manifest {self.manifest_path}"
        manifest = self.checkpoint_manifest()
        step = int(manifest.get("step", -1))
        if step <= 0:
            return False, f"checkpoint manifest has invalid step={manifest.get('step')}"
        if self.checkpoint_step != step or self.name != f"BioSeq-7L-step{step}":
            return False, "checkpoint label no longer matches its manifest step"
        directory_match = re.search(r"_step(\d+)$", self.checkpoint.parent.name)
        if directory_match and int(directory_match.group(1)) != step:
            return False, (
                f"checkpoint directory implies step{directory_match.group(1)}, "
                f"but manifest records step{step}"
            )
        snapshot = Path(str(manifest.get("snapshot", ""))).resolve()
        if snapshot != self.checkpoint:
            return False, f"manifest snapshot {snapshot} does not match {self.checkpoint}"
        try:
            import torch
        except Exception as exc:  # noqa: BLE001 - readiness must retain the runtime cause
            return False, f"BioSeq runtime import failed: {exc}"
        if self.device == "cuda" and not torch.cuda.is_available():
            return False, "CUDA was requested but is unavailable"
        return True, (
            f"BioSeq 7L step{step} checkpoint/manifest and runtime are ready; "
            f"sha256={manifest.get('sha256', 'unknown')}"
        )

    def _load(self):
        if self._sampler is None:
            from downstream.grammar.tcr_generation import BioSeqTcrSampler

            self._sampler = BioSeqTcrSampler(
                self.checkpoint,
                device=self.device,
                seed=self.seed,
            )
        return self._sampler

    def generate(
        self,
        peptide: str,
        mhc_allele: str | None = None,
        mhc_pseudosequence: str | None = None,
        num_candidates: int = 1000,
    ) -> list[Candidate]:
        """Generate raw candidates; MHC arguments are deliberately not consumed."""

        del mhc_allele, mhc_pseudosequence
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        sequences = self._load().conditional_cdr3b(
            str(peptide).upper(),
            int(num_candidates),
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            temperature=self.temperature,
            sampling_strategy=self.sampling_strategy,
        )
        if len(sequences) != num_candidates:
            raise RuntimeError(
                f"BioSeq returned {len(sequences)} raw candidates, requested {num_candidates}"
            )
        return [Candidate(sequence, None) for sequence in sequences]
