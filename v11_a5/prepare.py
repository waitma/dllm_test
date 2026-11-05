"""Prepare immutable V11 files; invoke launch_v11_a5_75m.sh --dry-run to inspect."""
from pathlib import Path
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if __name__=='__main__':runpy.run_module('v11_a5.release',run_name='__main__')
