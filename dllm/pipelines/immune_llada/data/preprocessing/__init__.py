"""Offline preparation for the immune-LLaDA semantic dataset.

Typical command::

    python scripts/data/preprocess_immune_dataset.py --config configs/data/immune_v3.yaml

Only semantic records are written. Grammar tokenization, padding, encoder input
reconstruction, and diffusion corruption remain in the training collator.
"""

from .filters import BLOCKLIST_NAMES, RecordFilter, build_filters, filter_reason, load_blocklists
from .pipeline import PreprocessConfig, preprocess_dataset
from .validators import SCHEMA_VERSION

__all__ = [
    "BLOCKLIST_NAMES", "PreprocessConfig", "RecordFilter", "SCHEMA_VERSION",
    "build_filters", "filter_reason", "load_blocklists", "preprocess_dataset",
]
