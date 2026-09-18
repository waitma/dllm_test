"""Pinned TCR eval identity. Defaults stay 49000; 92000 jobs export TCR_EVAL_*.

Import-time values are fixed for the process. Historical 49000 audit/collect
scripts keep working when those variables are unset.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAG = "tcr_v5_49000_20260913"
DEFAULT_CKPT = ROOT / "output/ab_eval_checkpoints/ab_v5_49000_llada_20260913"
DEFAULT_STEP = 49000
DEFAULT_SHA = "d5f892bce7d18106a7c417a3e4605f2bf1feac2bfa80709ca74fb149e5c07d92"


class Pin:
    def __init__(self) -> None:
        self.tag = os.environ.get("TCR_EVAL_TAG", DEFAULT_TAG)
        self.ckpt = Path(os.environ.get("TCR_EVAL_CKPT", str(DEFAULT_CKPT)))
        self.step = int(os.environ.get("TCR_EVAL_STEP", str(DEFAULT_STEP)))
        self.sha = os.environ.get("TCR_EVAL_SHA", DEFAULT_SHA)
        self.out = ROOT / "output/downstream_generation" / self.tag


PIN = Pin()
