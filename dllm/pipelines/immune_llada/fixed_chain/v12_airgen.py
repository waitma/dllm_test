"""Canonical V12 architecture identity for the changes made on 2026-11-09.

Activate pllm; USE_TF=0 python -B -m pytest scripts/tests/immune_llada/test_v12_versions.py -q
Reuses the verified implementation without changing its numerical computation.
Historical V10 schemas and frozen source bundles keep their original identities.
Current data is independently V10; see immune_llada.release_versions for its
immutable index and the remaining VNAR/training integration requirements.
"""
from ..release_versions import MODEL_VERSION, MODEL_ARCHITECTURE, MODEL_CHECKPOINT_SCHEMA
from .v10_airgen_qk_swiglu_bias import (
    INPUT_FEATURES, OBJECTIVE, RECIPE, OPTIONS as IMPLEMENTATION_OPTIONS,
    V10Collator, V10QKSwiGLUBiasModel, corrupt, parameter_count, swiglu_hidden_size,
)

ARCHITECTURE = MODEL_ARCHITECTURE
SCHEMA = MODEL_CHECKPOINT_SCHEMA
OPTIONS = dict(IMPLEMENTATION_OPTIONS, architecture=ARCHITECTURE, model_version=MODEL_VERSION)
V12Collator = V10Collator


class V12Model(V10QKSwiGLUBiasModel):
    """Same verified MINT computation, explicitly saved and loaded as model V12."""
    architecture = ARCHITECTURE
    schema = SCHEMA
    architecture_options = OPTIONS


def components(family_spec=None):
    if family_spec is not None:
        raise ValueError('V12 has no family prediction')
    return V12Collator, V12Model, corrupt, RECIPE
