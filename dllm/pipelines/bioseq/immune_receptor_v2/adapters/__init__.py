"""Source adapters for immune-receptor v2.

Use these through ``scripts/data/build_immune_receptor_v2.py`` rather than calling
individual adapters for production builds.
"""

from .antibody import (
    iter_abrank,
    iter_cov_abdab,
    iter_flab_properties,
    iter_kothiwal,
)
from .tcr import (
    iter_fullchain_derived,
    iter_mcpas,
    iter_mira,
    iter_piste,
    iter_vdjdb,
)

__all__ = [
    "iter_abrank",
    "iter_cov_abdab",
    "iter_flab_properties",
    "iter_fullchain_derived",
    "iter_kothiwal",
    "iter_mcpas",
    "iter_mira",
    "iter_piste",
    "iter_vdjdb",
]
