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

#: Default checkpoint for smoke tests and ad-hoc probes.
#:
#: An ``examples/llada`` immune-fusion Trainer directory, so it works with the
#: ``grammar:`` / ``fusion:`` specs (which accept a directory) but **not** with
#: ``bioseq:`` / ``bioseq-llada:`` (which ``torch.load`` a single ``.pt``).
#:
#: This is deliberately a small 270m snapshot: smoke tests want fast load, not
#: the headline number. For real results use the checkpoint named by the task's
#: eval job. The retired ``grammar_v2_*`` ``.pt`` checkpoints that these scripts
#: previously defaulted to were deleted on 2026-09-08; see
#: ``docs/archive/grammar_v2_retired_20260908``.
SMOKE_FUSION_CHECKPOINT = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/"
    "protein_esmc_llada270m_diffusion_immune_v3_8gpu_2m/eval_snapshot_44000"
)

__all__ = ["BENCHMARK_VERSION", "SMOKE_FUSION_CHECKPOINT"]
