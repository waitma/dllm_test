"""Activate pllm/protenix_abtcr; python -B SCRIPT --run ABS --device cpu|cuda.

Reuse the fixed-chain task checks with the source-bound post-training loader.
GPU checks run from the prepared eval_jobs YAML before benchmark workers.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from downstream import fixed_chain_eval
    from downstream.post_training_eval import load, PostTrainingBackbone
    from scripts.downstream.check_fixed_chain_eval import main as check
    fixed_chain_eval.load = load
    fixed_chain_eval.FixedChainBackbone = PostTrainingBackbone
    check()


if __name__ == '__main__':
    main()
