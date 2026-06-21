"""Canonical PPI / interaction split policies (follow published benchmarks).

Do not invent random train/valid/test partitions for assets that already ship
with author-defined splits. Use the policies below and the builders in
``scripts/data/build_mint_string_splits.py`` / ``build_bioseq_grammar_v1.py``.

Audit inventory::

    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_ppi_sources.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SplitRole = Literal["pretrain_train", "pretrain_valid", "supervised_train", "supervised_valid", "supervised_test", "eval_only", "cv_fold"]


@dataclass(frozen=True)
class SplitPolicy:
    policy_id: str
    reference: str
    source_id: str
    allowed_split_names: tuple[str, ...]
    role_by_split: dict[str, SplitRole]
    notes: str
    forbid_merging_test_into_train: bool = True


# MINT Nat Commun 2026 — STRING physical-link pretraining.
# Cluster proteins at 50% identity (MMseqs2), keep one edge per cluster pair,
# shuffle (seed 731), hold out 250k pairs for validation, then remove validation
# clusters from training. See VarunUllanat/mint ``stringdb.py``.
MINT_STRING_PRETRAIN = SplitPolicy(
    policy_id="mint_string_pretrain_v1",
    reference="Ullanat et al., Nat Commun 2026; github.com/VarunUllanat/mint stringdb.py",
    source_id="stringdb_mint",
    allowed_split_names=("train", "valid"),
    role_by_split={
        "train": "pretrain_train",
        "valid": "pretrain_valid",
    },
    notes=(
        "Built from STRING v12 physical links + MMseqs2 50% clusters. "
        "Output files: training_filtered.links.txt.gz, validation.links.txt.gz. "
        "Do not merge Bernett gold-standard or HumanPPI/YeastPPI test rows here."
    ),
)

# Bernett et al. 2024 gold-standard (Figshare) — used by MINT / SaProt / IRBench P1.
# Intra0/1/2 are increasing sequence-identity regimes within species.
BERNETT_GOLD_STANDARD = SplitPolicy(
    policy_id="bernett_gold_standard_figshare",
    reference="Bernett et al., Brief Bioinform 2024, Figshare 21591618",
    source_id="figshare_gold_standard",
    allowed_split_names=("intra0", "intra1", "intra2"),
    role_by_split={
        "intra0": "eval_only",
        "intra1": "eval_only",
        "intra2": "eval_only",
    },
    notes=(
        "Pos/neg pair lists per Intra* regime. Use for supervised eval / MLP probing, "
        "not for large-scale pretraining unless explicitly benchmarking Bernett splits."
    ),
)

# HuggingFace Bernett 90/90 multi-organism split (IRBench P1 positives source).
BERNETT_STRING_90_90 = SplitPolicy(
    policy_id="bernett_string_90_90_hf",
    reference="Bernett-style STRING model_org 90/90 partition (local HF arrow)",
    source_id="string_model_org_90_90_split",
    allowed_split_names=("train", "valid", "test"),
    role_by_split={
        "train": "supervised_train",
        "valid": "supervised_valid",
        "test": "supervised_test",
    },
    notes=(
        "Published 90/90 protein-level partition: no test protein >90% similar to train. "
        "Current grammar_v1 PPI cache is derived from train+valid only; keep test for IRBench."
    ),
)

# SaProt / PEER HumanPPI LMDB splits (MINT Table 1 HumanPPI).
HUMAN_PPI_SAPROT = SplitPolicy(
    policy_id="peer_humanppi_lmdb",
    reference="Xu et al., NeurIPS 2022 PEER; SaProt human_ppi LMDB layout",
    source_id="saprot_humanppi",
    allowed_split_names=("train", "valid", "test", "cross_species_test"),
    role_by_split={
        "train": "supervised_train",
        "valid": "supervised_valid",
        "test": "supervised_test",
        "cross_species_test": "eval_only",
    },
    notes="Split names come from LMDB directory names (human_ppi_{split}.lmdb).",
)

# PEER YeastPPI LMDB splits (MINT Table 1 YeastPPI).
YEAST_PPI_PEER = SplitPolicy(
    policy_id="peer_yeastppi_lmdb",
    reference="Xu et al., NeurIPS 2022 PEER; yeast_ppi LMDB layout",
    source_id="peer_yeastppi",
    allowed_split_names=("train", "valid", "test", "cross_species_test"),
    role_by_split={
        "train": "supervised_train",
        "valid": "supervised_valid",
        "test": "supervised_test",
        "cross_species_test": "eval_only",
    },
    notes="8:1:1 with 40% identity between splits after initial 90% dedup (PEER protocol).",
)

SKEMPI_CV = SplitPolicy(
    policy_id="skempi_complex_3fold",
    reference="Jankauskaite et al., Bioinformatics 2019; Luo et al. 3-fold by complex",
    source_id="skempi",
    allowed_split_names=("fold0", "fold1", "fold2"),
    role_by_split={
        "fold0": "cv_fold",
        "fold1": "cv_fold",
        "fold2": "cv_fold",
    },
    notes="Complex-held-out cross-validation; eval only, not pretraining pool.",
    forbid_merging_test_into_train=True,
)

# PISTE TCR-epitope-HLA random split (Armilius/PISTE data/random/).
PISTE_TCR_RANDOM = SplitPolicy(
    policy_id="piste_tcr_random",
    reference="PISTE github.com/Armilius/PISTE data/random/{train,val,test}_data.csv",
    source_id="piste_tcr_epitope_hla",
    allowed_split_names=("train", "valid", "test"),
    role_by_split={
        "train": "supervised_train",
        "valid": "supervised_valid",
        "test": "supervised_test",
    },
    notes="Author-provided random split; prefer over processed_v2 ad-hoc partition for grammar TCR.",
)

# Nat Methods 2025 TCR-epitope benchmark processed splits.
NAT_METHODS_TCR_BENCHMARK = SplitPolicy(
    policy_id="nat_methods_2025_tcr_benchmark",
    reference="Nat Methods 2025 s41592-025-02910-0; figshare 10.6084/m9.figshare.27020455",
    source_id="nat_methods_tcr_benchmark",
    allowed_split_names=("train", "valid", "test", "seen_epitope_test", "unseen_epitope_test"),
    role_by_split={
        "train": "supervised_train",
        "valid": "supervised_valid",
        "test": "supervised_test",
        "seen_epitope_test": "eval_only",
        "unseen_epitope_test": "eval_only",
    },
    notes="19-source merged benchmark with seen/unseen epitope held-out tests.",
)

SPLIT_POLICIES: dict[str, SplitPolicy] = {
    policy.policy_id: policy
    for policy in (
        MINT_STRING_PRETRAIN,
        BERNETT_GOLD_STANDARD,
        BERNETT_STRING_90_90,
        HUMAN_PPI_SAPROT,
        YEAST_PPI_PEER,
        SKEMPI_CV,
        PISTE_TCR_RANDOM,
        NAT_METHODS_TCR_BENCHMARK,
    )
}

SOURCE_TO_DEFAULT_POLICY: dict[str, str] = {
    "stringdb_mint": MINT_STRING_PRETRAIN.policy_id,
    "figshare_gold_standard": BERNETT_GOLD_STANDARD.policy_id,
    "string_model_org_90_90_split": BERNETT_STRING_90_90.policy_id,
    "saprot_humanppi": HUMAN_PPI_SAPROT.policy_id,
    "peer_yeastppi": YEAST_PPI_PEER.policy_id,
    "skempi": SKEMPI_CV.policy_id,
    "piste_tcr_epitope_hla": PISTE_TCR_RANDOM.policy_id,
    "nat_methods_tcr_benchmark": NAT_METHODS_TCR_BENCHMARK.policy_id,
}


def normalize_split_name(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if text in {"training", "tr"}:
        return "train"
    if text in {"validation", "val"}:
        return "valid"
    if text in {"testing", "te"}:
        return "test"
    return text


def validate_split(source_id: str, split: str, *, policy_id: str | None = None) -> SplitRole:
    """Return the split role or raise if split is not allowed for the source."""
    policy_key = policy_id or SOURCE_TO_DEFAULT_POLICY.get(source_id)
    if policy_key is None:
        raise ValueError(f"No canonical split policy registered for source_id={source_id!r}")
    policy = SPLIT_POLICIES[policy_key]
    normalized = normalize_split_name(split)
    if normalized not in policy.allowed_split_names:
        allowed = ", ".join(policy.allowed_split_names)
        raise ValueError(
            f"Split {split!r} is not allowed for {source_id} under {policy.policy_id}. "
            f"Allowed: {allowed}. Reference: {policy.reference}"
        )
    return policy.role_by_split[normalized]


def splits_allowed_for_pretraining(source_id: str) -> tuple[str, ...]:
    policy_id = SOURCE_TO_DEFAULT_POLICY.get(source_id)
    if policy_id is None:
        return ()
    policy = SPLIT_POLICIES[policy_id]
    return tuple(
        name for name, role in policy.role_by_split.items() if role in {"pretrain_train", "pretrain_valid"}
    )


def splits_allowed_for_grammar_mix(source_id: str) -> tuple[str, ...]:
    """Splits safe to mix into grammar_v1 training (never include eval-only/test)."""
    policy_id = SOURCE_TO_DEFAULT_POLICY.get(source_id)
    if policy_id is None:
        return ("train", "valid")
    policy = SPLIT_POLICIES[policy_id]
    blocked = {"supervised_test", "eval_only", "cv_fold"}
    return tuple(name for name, role in policy.role_by_split.items() if role not in blocked)
