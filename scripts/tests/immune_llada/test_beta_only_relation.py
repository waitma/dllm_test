from __future__ import annotations

import torch

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer
from dllm.pipelines.immune_llada.data.profiles import default_region_profile
from dllm.pipelines.immune_llada.data.sources import row_to_record
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import resolve_partial_mask


def _relation_positions(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["relation_token_mask"][0].bool()


# junc80 rows store the full IMGT junction and are tagged as such; the adapter
# decides anchor handling from this provenance, not from the sequence characters.
_JUNC_ROW = {"cdr3b": "CASSQF", "fv_source": "tcrdesign2026_cdr3_junc80"}


def test_beta_only_places_junction_anchors_in_framework_regions() -> None:
    record = row_to_record(
        "tcr_repertoire",
        dict(_JUNC_ROW),
        profile=default_region_profile(),
        row_index=3,
    )
    assert record is not None
    beta = record.chains[0]
    assert beta.regions["FR3"] == "C"
    assert beta.regions["CDR3"] == "ASSQ"
    assert beta.regions["FR4"] == "F"
    assert beta.metadata["observed_anchor_regions"] == {"FR3": "C", "FR4": "F"}
    assert beta.metadata["synthetic_residue_mask"][-3:] == [0, 0, 0]


def test_anchor_free_core_row_keeps_every_residue_in_cdr3() -> None:
    # An anchor-free core may legitimately start with C or end with F/W. Without a
    # junction marker the adapter must not peel anything off, otherwise a real
    # residue is silently deleted.
    record = row_to_record(
        "tcr_repertoire",
        {"cdr3b": "CASSQF", "fv_source": "tcrdesign2026_cdr3"},
        profile=default_region_profile(),
        row_index=3,
    )
    assert record is not None
    beta = record.chains[0]
    assert beta.regions["CDR3"] == "CASSQF"
    assert beta.metadata["observed_anchor_regions"] == {}
    assert record.identifiers["cdr3b_core"] == "CASSQF"


def test_beta_only_is_completed_to_two_chains_and_preserves_core() -> None:
    record = row_to_record(
        "tcr_repertoire",
        dict(_JUNC_ROW),
        profile=default_region_profile(),
        row_index=3,
    )
    assert record is not None
    assert record.chain_roles == ["tcr_beta", "tcr_alpha"]
    assert record.identifiers["cdr3b_core"] == "ASSQ"
    assert record.chains[0].regions["CDR3"] == "ASSQ"
    assert set(record.chains[1].sequence) == {"X"}
    beta = record.chains[0]
    assert beta.sequence.count("X") == len(beta.sequence) - len("CASSQF")
    assert sum(beta.metadata["synthetic_residue_mask"]) == len(beta.sequence) - len("CASSQF")


def test_synthetic_x_is_not_a_diffusion_target() -> None:
    record = row_to_record(
        "tcr_repertoire",
        dict(_JUNC_ROW),
        profile=default_region_profile(),
        row_index=0,
    )
    assert record is not None
    batch = GrammarBioSeqCollator(GrammarTokenizer(), max_sequence_length=256)([record])
    synthetic = batch["synthetic_residue_mask"][0]
    loss = batch["diffusion_loss_mask"][0]
    eligible = batch["diffusion_eligible_mask"][0]
    assert torch.any(synthetic)
    assert not torch.any(loss & synthetic)
    assert not torch.any(eligible & synthetic)
    assert torch.any(loss & ~synthetic)


def test_supervised_binding_relation_is_a_diffusion_target() -> None:
    record = row_to_record(
        "trait",
        {"epitope_seq": "AAAA", "cdr3b": "CASSQF", "relation": "negative"},
    )
    assert record is not None
    batch = GrammarBioSeqCollator(GrammarTokenizer(), max_sequence_length=256)([record])
    relation = batch["relation_token_mask"][0]
    target = batch["relation_target_mask"][0]
    loss = batch["diffusion_loss_mask"][0]
    eligible = batch["diffusion_eligible_mask"][0]
    assert torch.any(relation)
    assert torch.all(target[relation])
    assert torch.all(loss[target])
    assert torch.all(eligible[target])
    visible = resolve_partial_mask(batch)[0]
    assert not torch.any(visible[target])


def test_presentation_binding_stays_fixed_while_recognition_relation_is_target() -> None:
    record = row_to_record(
        "trait",
        {
            "mhc_seq": "M" * 20,
            "epitope_seq": "AAAA",
            "cdr3b": "CASSQF",
            "relation": "binding",
        },
    )
    assert record is not None
    batch = GrammarBioSeqCollator(GrammarTokenizer(), max_sequence_length=256)([record])
    relation = _relation_positions(batch)
    target = batch["relation_target_mask"][0].bool()
    assert torch.sum(relation) == 2
    assert torch.sum(target) == 1
    assert torch.all(batch["fixed_context_mask"][0][relation & ~target])
    assert torch.all(batch["diffusion_loss_mask"][0][target])
    visible = resolve_partial_mask(batch)[0]
    assert torch.all(visible[relation & ~target])
    assert not torch.any(visible[target])


def test_unknown_relation_is_unsupervised() -> None:
    record = row_to_record(
        "trait",
        {"epitope_seq": "AAAA", "cdr3b": "CASSQF", "relation": "unknown"},
    )
    assert record is not None
    assert record.metadata["relation_supervised"] is False
    batch = GrammarBioSeqCollator(GrammarTokenizer(), max_sequence_length=256)([record])
    relation = _relation_positions(batch)
    assert torch.any(relation)
    assert not torch.any(batch["relation_target_mask"][0][relation])
    assert torch.all(batch["fixed_context_mask"][0][relation])


def test_unsupervised_relation_stays_visible() -> None:
    record = row_to_record(
        "trait",
        {"epitope_seq": "AAAA", "cdr3b": "CASSQF"},
    )
    assert record is not None
    batch = GrammarBioSeqCollator(GrammarTokenizer(), max_sequence_length=256)([record])
    relation = batch["relation_token_mask"][0]
    target = batch["relation_target_mask"][0]
    visible = resolve_partial_mask(batch)[0]
    assert torch.any(relation)
    assert not torch.any(target[relation])
    assert torch.all(visible[relation])
