"""Receptor completion for the epitope-conditioned sources.

Run with ``pytest scripts/tests/immune_llada/test_receptor_completion.py``.

Covers what beta-only repertoire completion did not have to handle: junction-form
CDR3s, real full-length Fv chains that must survive untouched, alpha/beta
disambiguation, all-X upstream placeholders, and the profile feedback loop.
"""

from __future__ import annotations

import json

import pytest

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer
from dllm.pipelines.immune_llada.data.preprocessing.filters import build_filters, filter_reason
from dllm.pipelines.immune_llada.data.preprocessing.pipeline import PreprocessConfig
from dllm.pipelines.immune_llada.data.preprocessing.filters import BLOCKLIST_NAMES
from dllm.pipelines.immune_llada.data.profiles import (
    add_tcr_region_lengths,
    empty_tcr_region_lengths,
    fallback_region_profile,
    sample_region_lengths,
)
from dllm.pipelines.immune_llada.data.sources import (
    is_downgraded_mhc_layout,
    partner_completion_flags,
    row_stripped_placeholder_mhc,
    row_to_record,
)

_PROFILE = fallback_region_profile()
_REGIONS = ("FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4")


def _chain(record, role):
    return next(chain for chain in record.chains if chain.role == role)


def _record(source, row, **kwargs):
    record = row_to_record(source, dict(row), split="train", weight=1.0,
                           profile=_PROFILE, row_index=7, **kwargs)
    assert record is not None
    return record


# --------------------------------------------------------------- trait ------

def test_trait_junction_does_not_duplicate_the_conserved_cysteine() -> None:
    """trait stores C...F/W, so the anchors must be peeled, not doubled.

    Feeding a junction through the core-form path would leave the C both at the
    end of FR3 and at the head of CDR3.
    """
    record = _record("trait", {"epitope_seq": "GILGFVFTL", "cdr3b": "CASSQETQYF", "relation": "binding"})
    beta = _chain(record, "tcr_beta")
    assert beta.regions["FR3"].endswith("C")
    assert beta.regions["CDR3"] == "ASSQETQY"
    assert not beta.regions["CDR3"].startswith("C")
    assert beta.regions["FR4"].startswith("F")
    # exactly one C at the FR3/CDR3 seam
    seam = beta.regions["FR3"][-1] + beta.regions["CDR3"][:1]
    assert seam == "CA"


def test_trait_anchor_residues_are_not_counted_as_synthetic() -> None:
    record = _record("trait", {"epitope_seq": "GILGFVFTL", "cdr3b": "CASSQETQYF", "relation": "binding"})
    beta = _chain(record, "tcr_beta")
    mask = beta.metadata["synthetic_residue_mask"]
    assert len(mask) == len(beta.sequence)
    observed = len(beta.regions["CDR3"]) + len(beta.metadata["observed_anchor_regions"])
    assert len(mask) - sum(mask) == observed
    assert beta.metadata["observed_anchor_regions"] == {"FR3": "C", "FR4": "F"}


# ----------------------------------------------------- native / papers -----

def test_core_form_cdr3_is_passed_through_without_losing_a_residue() -> None:
    """tcr_native/tcr_papers store anchor-free cores; nothing may be stripped."""
    record = _record("tcr_papers", {"epitope_seq": "GILGFVFTL", "cdr3b": "ASSQETQY", "relation": "binding"})
    beta = _chain(record, "tcr_beta")
    assert beta.regions["CDR3"] == "ASSQETQY"
    assert beta.metadata["observed_anchor_regions"] == {}


def test_core_form_cdr3_starting_with_c_keeps_that_c() -> None:
    """A core may legitimately begin with C; a character heuristic would eat it."""
    record = _record("tcr_papers", {"epitope_seq": "GILGFVFTL", "cdr3b": "CASSQETQY", "relation": "binding"})
    assert _chain(record, "tcr_beta").regions["CDR3"] == "CASSQETQY"


def test_real_full_length_fv_is_kept_verbatim_and_never_synthesized() -> None:
    """A populated *_fv column is a measured sequence; completion must not touch it."""
    beta_fv = "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSLGQAYF"
    record = _record("tcr_native", {
        "epitope_seq": "SLLMWITQV", "mhc_seq": "Y" * 34,
        "beta_fv": beta_fv, "cdr3b": "ASSLGQAY", "relation": "binding",
    })
    beta = _chain(record, "tcr_beta")
    assert beta.sequence == beta_fv
    assert beta.regions == {}
    assert "synthetic_residue_mask" not in beta.metadata
    # the absent alpha is still synthesized so the layout stays two-chain
    alpha = _chain(record, "tcr_alpha")
    assert set(alpha.sequence) == {"X"}
    assert sum(alpha.metadata["synthetic_residue_mask"]) == len(alpha.sequence)


def test_beta_precedes_alpha_so_the_smallest_chain_id_is_beta() -> None:
    record = _record("tcr_papers", {"epitope_seq": "GILGFVFTL", "cdr3a": "AVGMNY", "cdr3b": "ASSQETQY", "relation": "binding"})
    receptor = [chain.role for chain in record.chains if chain.role.startswith("tcr_")]
    assert receptor == ["tcr_beta", "tcr_alpha"]


# ------------------------------------------------ alpha/beta ambiguity -----

def test_alpha_only_row_no_longer_renders_identically_to_a_beta_only_row() -> None:
    """There is no <tcra>/<tcrb> marker, so a lone chain is positionally ambiguous.

    Completion gives both rows a two-chain block, after which the real residues
    sit at different chain indices and the streams differ.
    """
    shared = {"epitope_seq": "GILGFVFTL", "relation": "binding"}
    alpha_only = _record("tcr_papers", {**shared, "cdr3a": "AVGMNYGGSQ"})
    beta_only = _record("tcr_papers", {**shared, "cdr3b": "AVGMNYGGSQ"})

    assert _chain(alpha_only, "tcr_alpha").regions["CDR3"] == "AVGMNYGGSQ"
    assert set(_chain(alpha_only, "tcr_beta").sequence) == {"X"}
    assert _chain(beta_only, "tcr_beta").regions["CDR3"] == "AVGMNYGGSQ"
    assert set(_chain(beta_only, "tcr_alpha").sequence) == {"X"}

    tokenizer = GrammarTokenizer()
    collator = GrammarBioSeqCollator(tokenizer, max_sequence_length=512)
    batch = collator([alpha_only, beta_only])
    assert not batch["input_ids"][0].equal(batch["input_ids"][1])


def test_completed_synthetic_positions_carry_no_loss() -> None:
    record = _record("tcr_papers", {"epitope_seq": "GILGFVFTL", "cdr3b": "ASSQETQY", "relation": "binding"})
    batch = GrammarBioSeqCollator(GrammarTokenizer(), max_sequence_length=512)([record])
    synthetic = batch["synthetic_residue_mask"][0].bool()
    assert synthetic.any()
    assert not batch["diffusion_loss_mask"][0].bool()[synthetic].any()
    assert not batch["diffusion_eligible_mask"][0].bool()[synthetic].any()


# ----------------------------------------------- all-X upstream holes ------

def test_all_x_mhc_is_dropped_and_the_row_becomes_epitope_conditioned() -> None:
    row = {"epitope_seq": "GILGFVFTL", "mhc_seq": "X" * 34, "cdr3b": "ASSQETQY", "relation": "binding"}
    record = _record("tcr_papers", row)
    assert [chain.role for chain in record.chains if chain.role == "mhc"] == []
    assert record.task_type == "tcr_epitope"
    assert _chain(record, "peptide").sequence == "GILGFVFTL"


def test_real_mhc_is_retained() -> None:
    record = _record("tcr_papers", {
        "epitope_seq": "GILGFVFTL", "mhc_seq": "Y" * 34, "cdr3b": "ASSQETQY", "relation": "binding",
    })
    assert _chain(record, "mhc").sequence == "Y" * 34
    assert record.task_type == "tcr_pmhc"


@pytest.mark.parametrize("source", ["trait", "tcr_native", "tcr_papers"])
def test_all_x_epitope_is_dropped_by_an_attributed_filter(source: str) -> None:
    blocklists = {name: set() for name in BLOCKLIST_NAMES}
    filters = build_filters(source, [source], blocklists, 1024, 1024)
    blank = _record(source, {"epitope_seq": "X" * 21, "mhc_seq": "Y" * 34, "cdr3b": "ASSQETQY", "relation": "nonbinding"})
    real = _record(source, {"epitope_seq": "GILGFVFTL", "mhc_seq": "Y" * 34, "cdr3b": "ASSQETQY", "relation": "nonbinding"})
    assert filter_reason(blank, filters) == "quality.blank_epitope"
    assert filter_reason(real, filters) is None


def test_partner_completion_and_mhc_strip_are_detectable_on_the_record() -> None:
    beta_only = _record("tcr_papers", {"epitope_seq": "GILGFVFTL", "cdr3b": "ASSQETQY", "relation": "binding"})
    alpha_only = _record("tcr_papers", {"epitope_seq": "GILGFVFTL", "cdr3a": "AVGMNYGGSQ", "relation": "binding"})
    both = _record("tcr_papers", {
        "epitope_seq": "GILGFVFTL", "cdr3a": "AVGMNY", "cdr3b": "ASSQETQY", "relation": "binding",
    })
    assert partner_completion_flags(beta_only) == (True, False)
    assert partner_completion_flags(alpha_only) == (False, True)
    assert partner_completion_flags(both) == (False, False)

    stripped_row = {"epitope_seq": "GILGFVFTL", "mhc_seq": "X" * 34, "cdr3b": "ASSQETQY", "relation": "binding"}
    stripped = _record("tcr_papers", stripped_row)
    assert row_stripped_placeholder_mhc(stripped_row, stripped)
    assert is_downgraded_mhc_layout(stripped)

    empty_mhc_row = {"epitope_seq": "GILGFVFTL", "cdr3b": "ASSQETQY", "relation": "binding"}
    empty_mhc = _record("tcr_papers", empty_mhc_row)
    assert not row_stripped_placeholder_mhc(empty_mhc_row, empty_mhc)
    # Prepared JSONL cannot see the raw MHC, so the layout heuristic over-counts
    # never-had-MHC rows relative to the live strip counter.
    assert is_downgraded_mhc_layout(empty_mhc)


def test_blank_epitope_filter_does_not_touch_unrelated_sources() -> None:
    blocklists = {name: set() for name in BLOCKLIST_NAMES}
    assert not any(f.name == "quality.blank_epitope" for f in build_filters("oas", ["oas"], blocklists, 1024, 1024))


# ------------------------------------------------- profile feedback loop ---

def test_synthetic_regions_never_feed_back_into_the_profile() -> None:
    """Learning from completed chains would collapse the profile onto itself."""
    record = _record("tcr_papers", {"epitope_seq": "GILGFVFTL", "cdr3b": "ASSQETQY", "relation": "binding"})
    counts = empty_tcr_region_lengths()
    add_tcr_region_lengths(counts, record)
    # only the genuinely observed beta CDR3 is counted
    assert counts["tcr_beta"]["CDR3"] == {len("ASSQETQY"): 1}
    for region in ("FR1", "CDR1", "FR2", "CDR2", "FR3", "FR4"):
        assert counts["tcr_beta"][region] == {}
    for region in _REGIONS:
        assert counts["tcr_alpha"][region] == {}


def test_withholding_the_profile_disables_completion() -> None:
    """This is how the profile-learning pass reads raw chains."""
    row = {"epitope_seq": "GILGFVFTL", "cdr3b": "ASSQETQY", "relation": "binding"}
    raw = row_to_record("tcr_papers", dict(row), split="train", weight=1.0, profile=None, row_index=7)
    assert [chain.role for chain in raw.chains] == ["peptide", "tcr_beta"]
    assert _chain(raw, "tcr_beta").sequence == "ASSQETQY"
    assert "receptor_completion" not in raw.metadata


def test_region_sampling_survives_a_manifest_round_trip() -> None:
    """atomic_json_dump writes sort_keys=True, so '10' precedes '2' on reload.

    Sampling must not depend on that ordering or a published dataset cannot be
    regenerated from its own manifest.
    """
    profile = {
        "schema_version": "immune_llada.tcr_region_profile.v1",
        "seed": 42,
        "region_distributions": {
            role: {region: {str(length): 1 for length in range(2, 25)} for region in _REGIONS}
            for role in ("tcr_alpha", "tcr_beta")
        },
    }
    reloaded = json.loads(json.dumps(profile, sort_keys=True))
    assert list(reloaded["region_distributions"]["tcr_beta"]["CDR3"])[:2] == ["10", "11"]
    for row_index in range(16):
        direct = sample_region_lengths(profile, "tcr_beta", seed=42, source="s", row_index=row_index)
        after = sample_region_lengths(reloaded, "tcr_beta", seed=42, source="s", row_index=row_index)
        assert direct == after


# -------------------------------------------------------------- config -----

def test_completion_sources_defaults_to_repertoire_only() -> None:
    """An existing config must keep producing the dataset it produced before."""
    config = PreprocessConfig(sources=[{"name": "oas", "path": "/x"}],
                              blocklists={name: "off" for name in BLOCKLIST_NAMES})
    assert config.completion_sources == ("tcr_repertoire",)


def test_completion_sources_rejects_a_source_that_cannot_be_completed() -> None:
    base = {
        "sources": [{"name": "oas", "path": "/x"}],
        "blocklists": {name: "off" for name in BLOCKLIST_NAMES},
    }
    with pytest.raises(ValueError, match="non-completable"):
        PreprocessConfig.from_mapping({**base, "tcr_region_profile": {"completion_sources": ["oas"]}})
    parsed = PreprocessConfig.from_mapping(
        {**base, "tcr_region_profile": {"completion_sources": ["tcr_papers", "tcr_repertoire"]}}
    )
    assert parsed.completion_sources == ("tcr_papers", "tcr_repertoire")
