"""Map interaction-task metadata to grammar-v1 relation tokens.

Run the inventory audit with::

    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_ppi_sources.py

When the relationship between two chains is not established by the source metadata,
use the grammar token ``<unknown>`` (stem ``unknown``). Only assign ``binding`` or
other typed relations when the task/source explicitly supports that label.
"""

from __future__ import annotations

from typing import Mapping

GRAMMAR_RELATIONS = (
    "binding",
    "activation",
    "inhibition",
    "catalysis",
    "reaction",
    "expression",
    "ptmod",
    "neutralization",
    "nonbinding",
    "unknown",
)

RELATION_ALIASES = {
    "unknown_relation": "unknown",
    "unk": "unknown",
    "unknown": "unknown",
}

# STRING-DB v12 functional / physical link channels that can be mapped to grammar
# relations. Physical links are the MINT pretraining default (`binding`).
STRING_LINK_CHANNELS: dict[str, str] = {
    "physical": "binding",
    "binding": "binding",
    "activation": "activation",
    "inhibition": "inhibition",
    "catalysis": "catalysis",
    "reaction": "reaction",
    "expression": "expression",
    "ptmod": "ptmod",
    "post_translational": "ptmod",
}

# task_family values from `scripts/build_ppi_interaction_csv.py`.
# Only list families with a well-defined interaction semantics.
TASK_FAMILY_TO_RELATION: dict[str, str] = {
    "ppi_pretraining": "binding",
    "ppi_binary": "binding",
    "tcr_epitope_binding": "binding",
    "antibody_neutralization": "neutralization",
}

# Families that ship two (or more) chains but do not specify the edge type.
TASK_FAMILY_UNKNOWN_RELATION: frozenset[str] = frozenset(
    {
        "ppi_mutation_affinity",
        "mutational_ppi",
        "oncogenic_ppi",
        "protein_ligand_binding",
        "antibody_binding",
        "antibody_sarscov2_binding",
        "tcr_epitope_hla",
        "tcr_epitope_interface",
    }
)

# Optional per-source overrides when task_family alone is too coarse.
SOURCE_ID_TO_RELATION: dict[str, str] = {
    "stringdb_mint": "binding",
    "string_model_org_90_90_split": "binding",
    "figshare_gold_standard": "binding",
    "saprot_humanppi": "binding",
    "peer_yeastppi": "binding",
    "covabdab_neutralization": "neutralization",
}

# How each local asset relates to MINT / downstream usage tiers.
DATA_TIER: dict[str, str] = {
    "stringdb_mint": "pretraining_candidate",
    "string_model_org_90_90_split": "grammar_v1_current",
    "figshare_gold_standard": "supervised_eval",
    "saprot_humanppi": "supervised_eval",
    "peer_yeastppi": "supervised_eval",
    "skempi": "supervised_eval",
    "swing_mutint": "supervised_eval",
    "flab": "supervised_finetune",
    "covabdab_neutralization": "supervised_eval",
    "oncoppi": "case_study",
    "tdc_tcr_epitope": "supervised_eval",
    "piste_tcr_epitope_hla": "supervised_eval",
    "teim_interface": "supervised_eval",
}


def normalize_relation(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not normalized:
        return "unknown"
    if normalized in RELATION_ALIASES:
        return RELATION_ALIASES[normalized]
    if normalized in GRAMMAR_RELATIONS:
        return normalized
    if normalized in STRING_LINK_CHANNELS:
        return STRING_LINK_CHANNELS[normalized]
    return "unknown"


def infer_grammar_relation(
    *,
    task_family: str | None = None,
    source_id: str | None = None,
    string_channel: str | None = None,
    label: str | None = None,
    explicit_relation: str | None = None,
) -> str:
    """Infer the grammar-v1 relation token stem for an interaction record."""
    if explicit_relation:
        return normalize_relation(explicit_relation)
    if string_channel:
        return normalize_relation(STRING_LINK_CHANNELS.get(string_channel.strip().lower(), string_channel))
    if source_id and source_id in SOURCE_ID_TO_RELATION:
        relation = SOURCE_ID_TO_RELATION[source_id]
    elif task_family and task_family in TASK_FAMILY_TO_RELATION:
        relation = TASK_FAMILY_TO_RELATION[task_family]
    elif task_family and task_family in TASK_FAMILY_UNKNOWN_RELATION:
        relation = "unknown"
    else:
        relation = "unknown"
    relation = normalize_relation(relation)
    if label in {"0", "false", "False", "negative", "nonbinding", "non_binding"}:
        return "nonbinding"
    return relation


def infer_relation_from_unified_row(row: Mapping[str, str]) -> str:
    return infer_grammar_relation(
        task_family=row.get("task_family"),
        source_id=row.get("source_id"),
        string_channel=row.get("string_channel"),
        label=row.get("label"),
        explicit_relation=row.get("grammar_relation") or row.get("relation_type"),
    )
