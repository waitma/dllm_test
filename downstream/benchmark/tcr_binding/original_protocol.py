"""Canonical T1 original-model catalog and source-selection rules.

Run the selector through the benchmark summarizer::

    /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/summarize_nm2025.py

An external T1 baseline is eligible as a local result only when it uses the
author's original checkpoint, the author's inference implementation, and the
test sets extracted from ``original.zip``.  No training CSV or fitted probe is
part of this protocol. Paper values always represent the baseline; local reruns
are separate diagnostics and never fill missing paper cells.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parents[1]
OFFICIAL_REPO = BENCH / "baselines" / "TCREpitopeBenchmark"
TEIM_RUNTIME = OFFICIAL_REPO / "TEIM" / "Original_model_prediciton" / "ckpt"
PANPEP_RUNTIME = BENCH / "baselines" / "PanPep" / "Requirements"
ERGO_RUNTIME = (
    OFFICIAL_REPO
    / "ERGO(ERGO-AE-vdj_ERGO-AE-mc_ERGO-lstm-vdj_ERGO-lstm-mc)"
)
ORIGINAL_ZIP = REPO / "data" / "ntmethod_binding" / "original.zip"
ORIGINAL_ZIP_SHA256 = "906e84ebd4b071d7cb9eec04294f6bfaeb7f967dad0008fd3a758910afef13d9"
ORIGINAL_TEST_MANIFEST = BENCH / "data" / "tcr_binding_nm2025" / "original_test_manifest.json"
T1_CITATION = "doi:10.1038/s41592-025-02910-0"

LOCAL_RESULT_LABEL = "本地复现：官方 checkpoint + 官方推理代码 + original.zip"
PAPER_RESULT_LABEL = "[P] 论文报告值（本地复现另列）"


@dataclass(frozen=True)
class OriginalBaselineSpec:
    """One paper original-model row and its local official wrapper, if any."""

    tag: str
    display_name: str
    paper_method: str
    wrapper: str | None
    checkpoint_paths: tuple[Path, ...]
    expected_model_names: tuple[str, ...]
    artifact_identity_pairs: tuple[tuple[Path, Path], ...] = ()

    @property
    def variant_tag(self) -> str | None:
        """Return the wrapper variant selected through ``TCRBENCH_BASELINE_TAG``."""

        return self.tag if self.wrapper in {"teinet", "ergo", "tpbte"} else None

    @property
    def checkpoint_available(self) -> bool:
        return bool(self.checkpoint_paths) and all(path.exists() for path in self.checkpoint_paths)

    @property
    def artifact_identity_verified(self) -> bool:
        """Verify runtime checkpoint copies against bundled aliases when present."""

        for runtime_path, bundle_path in self.artifact_identity_pairs:
            if not runtime_path.is_file() or not bundle_path.is_file():
                return False
            runtime_hash = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
            bundle_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
            if runtime_hash != bundle_hash:
                return False
        return True


ORIGINAL_BASELINES = (
    OriginalBaselineSpec(
        "atmtcr", "ATM-TCR", "ATM-TCR", "atmtcr",
        (OFFICIAL_REPO / "Original_model" / "ATM_TCR.ckpt",), ("ATM-TCR",),
    ),
    OriginalBaselineSpec(
        "teim", "TEIM", "TEIM", "teim",
        (
            TEIM_RUNTIME / "teim_seq.ckpt",
            TEIM_RUNTIME / "epi_ae.ckpt",
        ),
        ("TEIM (seq)",),
        artifact_identity_pairs=((
            TEIM_RUNTIME / "teim_seq.ckpt",
            OFFICIAL_REPO / "Original_model" / "TEIM.ckpt",
        ),),
    ),
    OriginalBaselineSpec(
        "teinet_large", "TEINet-large", "TEINet-large", "teinet",
        (OFFICIAL_REPO / "Original_model" / "TEINet_large.pth",), ("TEINet-large",),
    ),
    OriginalBaselineSpec(
        "teinet_small", "TEINet-small", "TEINet-small", "teinet",
        (OFFICIAL_REPO / "Original_model" / "TEINet_small.pth",), ("TEINet-small",),
    ),
    OriginalBaselineSpec(
        "nettcr", "NetTCR", "NetTCR", "nettcr",
        (OFFICIAL_REPO / "Original_model" / "NetTCR.h5",), ("NetTCR-2.2 (b)",),
    ),
    OriginalBaselineSpec(
        "epitcr_nomhc", "epiTCR", "epiTCR", "epitcr",
        (OFFICIAL_REPO / "Original_model" / "epiTCR.pickle",),
        ("epiTCR", "epiTCR-noMHC"),
    ),
    OriginalBaselineSpec(
        "ergo_lstm_vdj", "ERGO-lstm (vdj)", "ERGO-lstm-vdj", "ergo",
        (OFFICIAL_REPO / "Original_model" / "ERGO_lstm_vdj.pt",),
        ("ERGO-lstm (vdjdb)",),
    ),
    OriginalBaselineSpec(
        "ergo_lstm_mc", "ERGO-lstm (mc)", "ERGO-lstm-mc", "ergo",
        (OFFICIAL_REPO / "Original_model" / "ERGO_lstm_mc.pt",),
        ("ERGO-lstm (McPAS)",),
    ),
    OriginalBaselineSpec(
        "ergo_ae_vdj", "ERGO-AE (vdj)", "ERGO-AE-vdj", "ergo",
        (
            OFFICIAL_REPO / "Original_model" / "ERGO_AE_vdj.pt",
            ERGO_RUNTIME / "TCR_Autoencoder" / "tcr_ae_dim_100.pt",
        ),
        ("ERGO-AE (vdjdb)",),
    ),
    OriginalBaselineSpec(
        "ergo_ae_mc", "ERGO-AE (mc)", "ERGO-AE-mc", "ergo",
        (
            OFFICIAL_REPO / "Original_model" / "ERGO_AE_mc.pt",
            ERGO_RUNTIME / "TCR_Autoencoder" / "tcr_ae_dim_100.pt",
        ),
        ("ERGO-AE (McPAS)",),
    ),
    OriginalBaselineSpec(
        "tpbte_mc", "TPBTE (mc)", "TPBTE-mc", "tpbte",
        (OFFICIAL_REPO / "Original_model" / "TPBTE_mc.pth",),
        ("TPBTE (McPAS)",),
    ),
    OriginalBaselineSpec(
        "tpbte_vdj", "TPBTE (vdj)", "TPBTE-vdj", "tpbte",
        (OFFICIAL_REPO / "Original_model" / "TPBTE_vdj.pth",),
        ("TPBTE (VDJdb)",),
    ),
    OriginalBaselineSpec(
        "panpep", "PanPep", "PanPep", "panpep",
        (
            PANPEP_RUNTIME / "model.pt",
            PANPEP_RUNTIME / "Content_memory.pkl",
            PANPEP_RUNTIME / "Query.pkl",
        ),
        ("PanPep (zero-shot)",),
        artifact_identity_pairs=((
            PANPEP_RUNTIME / "model.pt",
            OFFICIAL_REPO / "Original_model" / "PanPep.pt",
        ),),
    ),
    OriginalBaselineSpec(
        "sete", "SETE", "SETE", "sete",
        (OFFICIAL_REPO / "Original_model" / "SETE",), ("SETE",),
    ),
    OriginalBaselineSpec(
        "tepcam", "TEPCAM", "TEPCAM", None,
        (OFFICIAL_REPO / "Original_model" / "TEPCAM.pt",), ("TEPCAM",),
    ),
)

ORIGINAL_BASELINE_BY_TAG = {spec.tag: spec for spec in ORIGINAL_BASELINES}


def get_original_spec(tag: str) -> OriginalBaselineSpec | None:
    """Return the canonical original-model spec for an output tag."""

    return ORIGINAL_BASELINE_BY_TAG.get(tag)


def choose_result_source(*, local_complete: bool, paper_available: bool) -> tuple[str, str]:
    """Select the primary source. ``local_complete`` only affects diagnostic columns."""

    if paper_available:
        return "paper_reported", PAPER_RESULT_LABEL
    return "unavailable", "[P] 未获取论文值；本地复现不替代论文值"
