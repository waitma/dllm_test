"""Canonical immune-receptor data schema and preprocessing utilities.

Run the end-to-end builder with::

    conda run -n pllm python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py all
"""

from .audit import audit_records, merge_exact_measurements
from .io import iter_jsonl, sha256_file, write_json, write_jsonl
from .schema import (
    BIOSEQ_V2_SCHEMA_VERSION,
    CanonicalRecord,
    Eligibility,
    Evidence,
    Measurement,
    SequenceEntity,
    SourceReference,
    TargetMapping,
    normalize_sequence,
    stable_digest,
    validate_domain_record,
)
from .splits import SplitProtocol, build_split_manifest, verify_split_manifest

__all__ = [
    "BIOSEQ_V2_SCHEMA_VERSION",
    "CanonicalRecord",
    "Eligibility",
    "Evidence",
    "Measurement",
    "SequenceEntity",
    "SourceReference",
    "SplitProtocol",
    "TargetMapping",
    "audit_records",
    "build_split_manifest",
    "iter_jsonl",
    "merge_exact_measurements",
    "normalize_sequence",
    "sha256_file",
    "stable_digest",
    "validate_domain_record",
    "verify_split_manifest",
    "write_json",
    "write_jsonl",
]
