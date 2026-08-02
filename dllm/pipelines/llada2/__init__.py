from __future__ import annotations

from typing import Any

from .models.configuration_llada2_moe import LLaDA2MoeConfig
from .models.modeling_llada2_moe import LLaDA2MoeModelLM

__all__ = [
    "LLaDA2MoeConfig",
    "LLaDA2MoeModelLM",
    "LLaDA2Sampler",
    "LLaDA2SamplerConfig",
]

# The sampler pulls in ``dllm.core`` (and thus the optional ``lm_eval`` eval stack),
# which is not needed to build/train the LLaDA2 backbone. Import it lazily so that
# ``from dllm.pipelines.llada2.models... import ...`` works in training-only
# environments while ``dllm.pipelines.llada2.LLaDA2Sampler`` still works for inference.
_LAZY_SAMPLER_EXPORTS = {"LLaDA2Sampler", "LLaDA2SamplerConfig"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SAMPLER_EXPORTS:
        from .sampler import LLaDA2Sampler, LLaDA2SamplerConfig

        return {"LLaDA2Sampler": LLaDA2Sampler, "LLaDA2SamplerConfig": LLaDA2SamplerConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
