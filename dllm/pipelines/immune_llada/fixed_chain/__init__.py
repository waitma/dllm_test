"""Fixed-chain model; activate pllm and run pytest scripts/tests/immune_llada/test_fixed_chain.py.

Read model.py for the architecture, layout.py for inputs, generation.py for decoding.
"""

from .model import FixedChainConfig, FixedChainModel
from .layout import FixedChainCollator
