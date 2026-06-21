#!/usr/bin/env python3
"""Build the unified interaction-task CSV (step 1 of PPI pipeline).

This is the canonical entry point. It wraps ``scripts/build_ppi_interaction_csv.py``
and writes ``grammar_relation`` / ``string_channel`` columns.

Example::

    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_ppi_unified_csv.py
    python .../build_ppi_unified_csv.py --sources covabdab_neutralization,figshare_gold_standard
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
SCRIPT = PROJECT_ROOT / "scripts/build_ppi_interaction_csv.py"


def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
