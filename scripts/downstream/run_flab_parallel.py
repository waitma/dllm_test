#!/usr/bin/env python3
"""Run the existing FLAb evaluator with parallel GridSearchCV workers.

This wrapper changes execution scheduling only.  The imported evaluator still
defines the dataset split, preprocessing, Ridge alpha grid, metrics, and output
schema.  GridSearchCV's ``n_jobs=None`` inherits this joblib context.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from joblib import parallel_config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from downstream.flab.run_flab_baselines import main


if __name__ == "__main__":
    n_jobs = int(os.environ.get("FLAB_N_JOBS", "8"))
    if n_jobs < 1:
        raise ValueError("FLAB_N_JOBS must be >= 1")
    with parallel_config(
        backend="loky",
        n_jobs=n_jobs,
        inner_max_num_threads=1,
    ):
        main()
