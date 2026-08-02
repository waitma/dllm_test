"""DEPRECATED — use gen_fig1_mpl.py (matplotlib publication style).

This wrapper kept for backward compatibility.
"""
from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "scripts/gen_fig1_mpl.py"])
