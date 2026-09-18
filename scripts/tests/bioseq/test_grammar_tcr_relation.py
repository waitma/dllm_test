"""TCR-pMHC recognition-relation rendering.

Run:
    conda run -n pllm python -m pytest \
        scripts/tests/bioseq/test_grammar_tcr_relation.py -q

Regression guard for the B1 fix: the pMHC<->TCR recognition token must carry the
record's binding/nonbinding label (PISTE negatives), not a hardcoded <binding>.
MHC->peptide stays presentation (<binding>); unlabeled pretraining pMHC-TCR pairs
default to <binding> (byte-identical to the old behaviour when unset).
"""

from __future__ import annotations

from collections import Counter

from dllm.pipelines.immune_llada.data.grammar import GrammarRenderer, GrammarTokenizer
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord

TOK = GrammarTokenizer()
RENDER = GrammarRenderer(TOK)


def _tokens(record: BioSeqRecord) -> list[str]:
    return TOK.decode_tokens(RENDER.encode(record)["input_ids"])


def _pmhc_record(relation: str | None) -> BioSeqRecord:
    labels = {"relation": relation} if relation is not None else {}
    return BioSeqRecord(
        chains=[
            BioSeqChain("YQEVTSVEG", "mhc"),
            BioSeqChain("GILGFVFTL", "peptide"),
            BioSeqChain("CAVRDSNYQLIW", "tcr_alpha"),
            BioSeqChain("CASSLGQAYEQYF", "tcr_beta"),
        ],
        task_type="tcr_pmhc",
        source="tcr_piste",
        labels=labels,
    )


def _peptide_only_record(relation: str | None) -> BioSeqRecord:
    labels = {"relation": relation} if relation is not None else {}
    return BioSeqRecord(
        chains=[
            BioSeqChain("GILGFVFTL", "peptide"),
            BioSeqChain("CASSLGQAYEQYF", "tcr_beta"),
        ],
        task_type="tcr_pmhc",
        source="tcr_piste",
        labels=labels,
    )


def test_pmhc_nonbinding_uses_nonbinding_token() -> None:
    counts = Counter(_tokens(_pmhc_record("nonbinding")))
    # MHC->peptide presentation stays <binding>; peptide->TCR carries the label.
    assert counts["<nonbinding>"] == 1
    assert counts["<binding>"] == 1


def test_pmhc_binding_uses_binding_for_both_links() -> None:
    counts = Counter(_tokens(_pmhc_record("binding")))
    assert counts["<binding>"] == 2
    assert counts["<nonbinding>"] == 0


def test_pmhc_unlabeled_defaults_to_binding() -> None:
    counts = Counter(_tokens(_pmhc_record(None)))
    # Back-compat: unlabeled pMHC-TCR pretraining pairs must NOT become <nonbinding>.
    assert counts["<nonbinding>"] == 0
    assert counts["<binding>"] == 2


def test_peptide_only_nonbinding_propagates() -> None:
    counts = Counter(_tokens(_peptide_only_record("nonbinding")))
    # No MHC block -> only the peptide->TCR link, which carries the label.
    assert counts["<nonbinding>"] == 1
    assert counts["<binding>"] == 0
