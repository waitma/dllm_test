"""IRBench common framework.

Shared, dependency-light building blocks for the immune-receptor benchmark:

- ``schema``      canonical TCR / pair / prediction records + dataset loaders
- ``leakage``     leakage-controlled split engine, clonotype dedup, edit distance
- ``negatives``   reference-TCR negative generation for binding tasks
- ``metrics``     binding / clustering / representation / generation metrics
- ``featurizers`` non-DL baselines (k-mer, CDR3 edit-distance kNN scorer)
- ``model_api``   model wrapper interface + ESM2 / Ophiuchus / BioSeq adapters

Heavy deps (torch, transformers) are imported lazily inside ``model_api`` so the
rest of the framework stays importable on a CPU-only / minimal environment.
"""

BENCHMARK_VERSION = "irbench.v1"

__all__ = ["BENCHMARK_VERSION"]
