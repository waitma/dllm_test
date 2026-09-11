from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dllm.pipelines.immune_llada.data import (
    GRAMMAR_NULL_CONTEXT_TOKEN,
    GrammarTokenizer,
)
from dllm.pipelines.immune_llada.data.grammar import (
    GRAMMAR_TOKENS,
    GrammarRenderer,
)
from dllm.pipelines.immune_llada.data.preprocessing.validators import validate_prepared_row
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord

# The seven v4 sources. Standalone pairs have no antigen/peptide/MHC; the
# conditioned sources always carry at least one of those roles.
_UNCONDITIONAL_SOURCES = ("oas", "ots", "tcr_repertoire")
_CONDITIONED_SOURCES = ("asd_antibody", "trait", "tcr_native", "tcr_papers")
_PREPARED_ROOT = Path(
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v4_beta_relation"
)
_CONTEXT_ROLES = frozenset({"antigen", "peptide", "epitope", "mhc", "pmhc", "hla", "protein_a"})
# Matches preprocessing/filters.py budget.max_length. Duplicated here because
# that module is owned by another agent; the formula is the contract the
# renderer must stay inside of after the 4-token prefix.
_BUDGET_MAX_LENGTH = 1024

# Unconditional records open with a fixed no-context prefix so their layout
# matches conditioned ones slot-for-slot: the only difference is block content,
# which is what allows classifier-free-style two-pass decoding at inference.
_NULL_PREFIX = ["<prots>", GRAMMAR_NULL_CONTEXT_TOKEN, "<protd>", "<unknown>"]


def _tokenizer() -> GrammarTokenizer:
    return GrammarTokenizer()


def _renderer(tokenizer: GrammarTokenizer) -> GrammarRenderer:
    return GrammarRenderer(tokenizer=tokenizer, ppi_max_protein_length=1024)


def _tokens(tokenizer: GrammarTokenizer, encoded: dict) -> list[str]:
    return [tokenizer.id_to_token.get(token_id, "<res>") for token_id in encoded["input_ids"]]


def _antibody_pair() -> BioSeqRecord:
    return BioSeqRecord(
        source="oas",
        task_type="antibody",
        chains=[
            BioSeqChain(sequence="QVQLVQ", role="antibody_heavy"),
            BioSeqChain(sequence="DIQMT", role="antibody_light"),
        ],
    )


def _tcr_pair() -> BioSeqRecord:
    return BioSeqRecord(
        source="ots",
        task_type="tcr",
        chains=[
            BioSeqChain(sequence="CASSQ", role="tcr_beta"),
            BioSeqChain(sequence="CAVRD", role="tcr_alpha"),
        ],
    )


def _antigen_antibody() -> BioSeqRecord:
    return BioSeqRecord(
        source="asd_antibody",
        task_type="antibody_antigen",
        chains=[
            BioSeqChain(sequence="GGGGSA", role="antigen"),
            BioSeqChain(sequence="QVQLVQ", role="antibody_heavy"),
            BioSeqChain(sequence="DIQMT", role="antibody_light"),
        ],
        labels={"relation": "binding"},
    )


def _tcr_peptide() -> BioSeqRecord:
    return BioSeqRecord(
        source="tcr_papers",
        task_type="tcr_epitope",
        chains=[
            BioSeqChain(sequence="GILGFVFTL", role="peptide"),
            BioSeqChain(sequence="CASSQ", role="tcr_beta"),
            BioSeqChain(sequence="CAVRD", role="tcr_alpha"),
        ],
        labels={"relation": "binding"},
    )


def _tcr_pmhc() -> BioSeqRecord:
    return BioSeqRecord(
        source="tcr_native",
        task_type="tcr_pmhc",
        chains=[
            BioSeqChain(sequence="GSHSMRY", role="mhc"),
            BioSeqChain(sequence="NLVPMVATV", role="peptide"),
            BioSeqChain(sequence="CASSQ", role="tcr_beta"),
            BioSeqChain(sequence="CAVRD", role="tcr_alpha"),
        ],
        labels={"relation": "binding"},
    )


def _tcr_repertoire_pair() -> BioSeqRecord:
    # Completed beta-only row: real CDR3β core, synthetic partner. Still a
    # standalone pair — no peptide/MHC — so it must take the null prefix.
    beta = "X" * 8 + "CASSQ" + "X" * 4
    alpha = "X" * 12
    return BioSeqRecord(
        source="tcr_repertoire",
        task_type="tcr",
        chains=[
            BioSeqChain(
                sequence=beta,
                role="tcr_beta",
                metadata={"synthetic_residue_mask": [1] * 8 + [0] * 5 + [1] * 4},
            ),
            BioSeqChain(
                sequence=alpha,
                role="tcr_alpha",
                metadata={"synthetic_residue_mask": [1] * 12},
            ),
        ],
    )


def _decode(tokenizer: GrammarTokenizer, encoded: dict) -> list[str]:
    return tokenizer.decode_tokens(encoded["input_ids"])


def _receptor_block(tokens: list[str]) -> list[str]:
    """The last ``<prots>…<protd>`` block — the generated receptor."""

    start = max(index for index, token in enumerate(tokens) if token == "<prots>")
    return tokens[start:]


def _budget_would_keep(record: BioSeqRecord) -> bool:
    residues = sum(len(chain.sequence) for chain in record.chains)
    return residues + 3 * len(record.chains) + 8 <= _BUDGET_MAX_LENGTH


def test_null_token_is_appended_so_relation_ids_do_not_shift() -> None:
    # Grammar ids are base_vocab_size + enumerate(GRAMMAR_TOKENS). Inserting the
    # null marker into GRAMMAR_STRUCTURE_TOKENS would move <binding> from 39 to 40
    # and drag every relation id with it, invalidating persisted ids and older
    # checkpoints. It must be appended last instead.
    tokenizer = _tokenizer()
    assert tokenizer.special_id("<prots>") == 37
    assert tokenizer.special_id("<protd>") == 38
    assert tokenizer.special_id("<binding>") == 39
    assert tokenizer.special_id("<nonbinding>") == 47
    assert tokenizer.special_id("<unknown>") == 48
    assert tokenizer.special_id(GRAMMAR_NULL_CONTEXT_TOKEN) == 49
    assert GRAMMAR_TOKENS[-1] == GRAMMAR_NULL_CONTEXT_TOKEN
    assert tokenizer.vocab_size == tokenizer.base_vocab_size + len(GRAMMAR_TOKENS)


def test_unconditional_layouts_open_with_the_null_context_prefix() -> None:
    tokenizer = _tokenizer()
    renderer = _renderer(tokenizer)
    for record, expected_name, type_marker in (
        (_antibody_pair(), "antibody_pair", "<ab>"),
        (_tcr_pair(), "tcr_pair", "<tcr>"),
        (
            BioSeqRecord(
                source="nb",
                task_type="nanobody",
                chains=[BioSeqChain(sequence="QVQLV", role="nanobody_vhh")],
            ),
            "nanobody",
            "<nb>",
        ),
    ):
        encoded = renderer.encode(record)
        tokens = _tokens(tokenizer, encoded)
        assert encoded["grammar_name"] == expected_name
        assert tokens[:4] == _NULL_PREFIX
        # The record's own block still follows immediately after the prefix.
        assert tokens[4] == "<prots>"
        assert tokens[5] == type_marker


def test_null_prefix_is_fixed_and_never_supervised() -> None:
    # The prefix costs attention only: it must not be corrupted, must not
    # contribute diffusion loss, and its <unknown> relation must never become a
    # supervised relation target.
    tokenizer = _tokenizer()
    encoded = _renderer(tokenizer).encode(_antibody_pair())
    assert encoded["fixed_context_mask"][:4] == [1, 1, 1, 1]
    assert encoded["diffusion_loss_mask"][:4] == [0, 0, 0, 0]
    assert encoded["diffusion_eligible_mask"][:4] == [0, 0, 0, 0]
    assert encoded["relation_target_mask"][:4] == [0, 0, 0, 0]
    assert encoded["synthetic_residue_mask"][:4] == [0, 0, 0, 0]
    assert sum(encoded["relation_target_mask"]) == 0


def test_null_prefix_does_not_consume_a_chain_index() -> None:
    # Downstream eval addresses chains positionally -- residue_positions_by_chain,
    # tcr_generation_partial_mask(target_chain_indices=0), nbbench chain0. If the
    # residue-free prefix block incremented chain_index, every unconditional record
    # would shift to [1, 2] and those callers would silently target the wrong
    # chain instead of failing loudly.
    tokenizer = _tokenizer()
    renderer = _renderer(tokenizer)
    for record in (_antibody_pair(), _tcr_pair()):
        encoded = renderer.encode(record)
        chain_ids = sorted({c for c in encoded["position_ids_chain"] if c >= 0})
        assert chain_ids == [0, 1]
        # Structure tokens, including the whole prefix, stay unassigned.
        assert encoded["position_ids_chain"][:4] == [-1, -1, -1, -1]


def test_conditioned_layouts_are_unchanged_by_the_prefix() -> None:
    tokenizer = _tokenizer()
    renderer = _renderer(tokenizer)
    encoded = renderer.encode(_antigen_antibody())
    tokens = _tokens(tokenizer, encoded)
    assert encoded["grammar_name"] == "antigen_antibody"
    assert GRAMMAR_NULL_CONTEXT_TOKEN not in tokens
    # Real context occupies the slot the null block would have taken.
    assert tokens[0] == "<prots>"
    assert tokens[1] == "<res>"
    assert sorted({c for c in encoded["position_ids_chain"] if c >= 0}) == [0, 1, 2]


def test_pmhc_recognition_relation_still_supervised_without_prefix() -> None:
    tokenizer = _tokenizer()
    renderer = _renderer(tokenizer)
    encoded = renderer.encode(
        BioSeqRecord(
            source="tcr_papers",
            task_type="tcr_pmhc",
            chains=[
                BioSeqChain(sequence="GSHSMRY", role="mhc"),
                BioSeqChain(sequence="NLVPMVATV", role="peptide"),
                BioSeqChain(sequence="CASSQ", role="tcr_beta"),
                BioSeqChain(sequence="CAVRD", role="tcr_alpha"),
            ],
            labels={"relation": "nonbinding"},
            metadata={"relation_supervised": True},
        )
    )
    tokens = _tokens(tokenizer, encoded)
    assert GRAMMAR_NULL_CONTEXT_TOKEN not in tokens
    # MHC->peptide presentation stays fixed <binding>; only peptide->TCR
    # recognition carries the label and is supervised.
    assert tokens.count("<binding>") == 1
    assert tokens.count("<nonbinding>") == 1
    assert sum(encoded["relation_target_mask"]) == 1
    target_index = encoded["relation_target_mask"].index(1)
    assert tokens[target_index] == "<nonbinding>"


def test_downstream_chain_lookup_survives_the_prefix_on_both_paths() -> None:
    # chain_residue_positions has two addressing paths: position_ids_chain, and a
    # token-scan fallback. The fallback used to break on the first <protd>, which
    # the residue-free prefix block would trip -- returning zero positions for
    # every unconditional record.
    from downstream.grammar.masks import chain_residue_positions

    tokenizer = _tokenizer()
    renderer = _renderer(tokenizer)
    for record, expected in ((_antibody_pair(), 6), (_antigen_antibody(), 6)):
        encoded = renderer.encode(record)
        input_ids = torch.tensor([encoded["input_ids"]])
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        residue = torch.tensor([[c == 1 for c in encoded["token_class_ids"]]])
        chain_pos = torch.tensor([encoded["position_ids_chain"]])
        for position_ids_chain in (chain_pos, None):
            found = chain_residue_positions(
                input_ids,
                attention,
                residue,
                tokenizer,
                chain="heavy",
                position_ids_chain=position_ids_chain,
            )[0]
            assert len(found) == expected


def test_five_layouts_get_the_prefix_exactly_when_unconditioned() -> None:
    # The five training layouts. Prefix is a property of "no real context",
    # not of source name: oas/ots/repertoire share it, the three conditioned
    # layouts must not (that would throw away antigen/peptide/MHC).
    tokenizer = _tokenizer()
    renderer = _renderer(tokenizer)
    cases = (
        (_antibody_pair(), "antibody_pair", True),
        (_tcr_pair(), "tcr_pair", True),
        (_tcr_repertoire_pair(), "tcr_pair", True),
        (_antigen_antibody(), "antigen_antibody", False),
        (_tcr_peptide(), "tcr_peptide", False),
        (_tcr_pmhc(), "tcr_pmhc", False),
    )
    for record, expected_name, want_prefix in cases:
        encoded = renderer.encode(record)
        tokens = _decode(tokenizer, encoded)
        assert encoded["grammar_name"] == expected_name
        if want_prefix:
            assert tokens[:4] == _NULL_PREFIX
            assert encoded["fixed_context_mask"][:4] == [1, 1, 1, 1]
            assert encoded["diffusion_loss_mask"][:4] == [0, 0, 0, 0]
            assert encoded["diffusion_eligible_mask"][:4] == [0, 0, 0, 0]
            assert encoded["synthetic_residue_mask"][:4] == [0, 0, 0, 0]
        else:
            assert tokens[:4] != _NULL_PREFIX
            assert GRAMMAR_NULL_CONTEXT_TOKEN not in tokens
            assert tokens[0] == "<prots>"


def test_source_shaped_records_follow_the_prepared_layout_contract() -> None:
    """Each of the 7 prepared sources, reconstructed the way the loader does."""

    tokenizer = _tokenizer()
    renderer = _renderer(tokenizer)
    # (source, task_type, roles/seqs, expected layout, want prefix)
    specs = (
        (
            "oas",
            "antibody",
            (("QVQLVQ", "antibody_heavy"), ("DIQMT", "antibody_light")),
            "antibody_pair",
            True,
        ),
        (
            "ots",
            "tcr",
            (("CASSQ", "tcr_beta"), ("CAVRD", "tcr_alpha")),
            "tcr_pair",
            True,
        ),
        (
            "tcr_repertoire",
            "tcr",
            (("CASSQETQYF", "tcr_beta"), ("XXXXXXXXXXXX", "tcr_alpha")),
            "tcr_pair",
            True,
        ),
        (
            "asd_antibody",
            "antibody_antigen",
            (
                ("GGGGSA", "antigen"),
                ("QVQLVQ", "antibody_heavy"),
                ("DIQMT", "antibody_light"),
            ),
            "antigen_antibody",
            False,
        ),
        (
            "trait",
            "tcr_epitope",
            (
                ("GILGFVFTL", "peptide"),
                ("CASSQ", "tcr_beta"),
                ("CAVRD", "tcr_alpha"),
            ),
            "tcr_peptide",
            False,
        ),
        (
            "tcr_native",
            "tcr_pmhc",
            (
                ("GSHSMRY", "mhc"),
                ("NLVPMVATV", "peptide"),
                ("CASSQ", "tcr_beta"),
                ("CAVRD", "tcr_alpha"),
            ),
            "tcr_pmhc",
            False,
        ),
        (
            "tcr_papers",
            "tcr_epitope",
            (("GILGFVFTL", "peptide"), ("CASSQ", "tcr_beta")),
            "tcr_peptide",
            False,
        ),
    )
    for source, task_type, chains, expected_name, want_prefix in specs:
        record = BioSeqRecord(
            source=source,
            task_type=task_type,
            chains=[BioSeqChain(sequence=seq, role=role) for seq, role in chains],
            labels={"relation": "binding"},
        )
        encoded = renderer.encode(record)
        tokens = _decode(tokenizer, encoded)
        assert encoded["grammar_name"] == expected_name, source
        assert (tokens[:4] == _NULL_PREFIX) is want_prefix, source


def test_cfg_substitution_of_real_context_matches_the_null_prefix_stream() -> None:
    # Two-pass CFG: run once with the real context block, once with the null
    # prefix. The receptor block after the context slot must be byte-identical,
    # otherwise the null branch is a different positional layout.
    tokenizer = _tokenizer()
    renderer = _renderer(tokenizer)
    pairs = (
        (_antigen_antibody(), _antibody_pair()),
        (_tcr_peptide(), _tcr_pair()),
        (_tcr_pmhc(), _tcr_pair()),
    )
    for conditioned, unconditional in pairs:
        cond_tokens = _decode(tokenizer, renderer.encode(conditioned))
        uncond_tokens = _decode(tokenizer, renderer.encode(unconditional))
        assert uncond_tokens[:4] == _NULL_PREFIX
        assert GRAMMAR_NULL_CONTEXT_TOKEN not in cond_tokens
        cond_receptor = _receptor_block(cond_tokens)
        uncond_receptor = _receptor_block(uncond_tokens)
        assert cond_receptor == uncond_receptor
        assert _NULL_PREFIX + cond_receptor == uncond_tokens


def test_context_free_single_tcr_still_gets_the_prefix() -> None:
    # Pre-completion beta-only rows (one chain, no peptide). The prefix is
    # about missing context, not about chain count.
    tokenizer = _tokenizer()
    encoded = _renderer(tokenizer).encode(
        BioSeqRecord(
            source="tcr_repertoire",
            task_type="tcr",
            chains=[BioSeqChain(sequence="CASSQ", role="tcr_beta")],
        )
    )
    tokens = _decode(tokenizer, encoded)
    assert encoded["grammar_name"] == "tcr_single"
    assert tokens[:4] == _NULL_PREFIX
    assert tokens[4:6] == ["<prots>", "<tcr>"]
    assert encoded["position_ids_chain"][:4] == [-1, -1, -1, -1]
    assert sorted({c for c in encoded["position_ids_chain"] if c >= 0}) == [0]


def test_prefix_keeps_budget_boundary_records_at_or_under_1024() -> None:
    tokenizer = _tokenizer()
    renderer = _renderer(tokenizer)
    # Filter keeps a 2-chain row when residues + 3*n + 8 <= 1024 → 1010 aa.
    # antibody_pair render = 1010 residues + 4 prefix + <prots><ab>.<protd>.
    heavy = "Q" * 505
    light = "D" * 505
    record = BioSeqRecord(
        source="oas",
        task_type="antibody",
        chains=[
            BioSeqChain(sequence=heavy, role="antibody_heavy"),
            BioSeqChain(sequence=light, role="antibody_light"),
        ],
    )
    assert _budget_would_keep(record)
    encoded = renderer.encode(record)
    assert len(encoded["input_ids"]) <= _BUDGET_MAX_LENGTH
    assert _decode(tokenizer, encoded)[:4] == _NULL_PREFIX

    # 3-chain conditioned row at the same filter ceiling must also fit, and
    # must not pick up a prefix (that would add 4 tokens on top of real context).
    antigen = "G" * 500
    heavy = "Q" * 300
    light = "D" * 207
    conditioned = BioSeqRecord(
        source="asd_antibody",
        task_type="antibody_antigen",
        chains=[
            BioSeqChain(sequence=antigen, role="antigen"),
            BioSeqChain(sequence=heavy, role="antibody_heavy"),
            BioSeqChain(sequence=light, role="antibody_light"),
        ],
        labels={"relation": "binding"},
    )
    assert _budget_would_keep(conditioned)
    encoded = renderer.encode(conditioned)
    assert len(encoded["input_ids"]) <= _BUDGET_MAX_LENGTH
    assert GRAMMAR_NULL_CONTEXT_TOKEN not in _decode(tokenizer, encoded)


@pytest.mark.skipif(
    not (_PREPARED_ROOT / "train").is_dir(),
    reason="prepared v4 corpus not on disk",
)
@pytest.mark.parametrize("source", _UNCONDITIONAL_SOURCES + _CONDITIONED_SOURCES)
def test_prepared_first_row_obeys_the_prefix_contract(source: str) -> None:
    shard = _PREPARED_ROOT / "train" / f"{source}-00000.jsonl"
    if not shard.is_file():
        pytest.skip(f"no shard for {source}")
    with shard.open(encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    record = validate_prepared_row(row)
    tokenizer = _tokenizer()
    encoded = _renderer(tokenizer).encode(record)
    tokens = _decode(tokenizer, encoded)
    has_context = bool(_CONTEXT_ROLES & {role.lower() for role in record.chain_roles})
    if has_context:
        assert tokens[:4] != _NULL_PREFIX
        assert GRAMMAR_NULL_CONTEXT_TOKEN not in tokens
        assert source in _CONDITIONED_SOURCES
    else:
        assert tokens[:4] == _NULL_PREFIX
        assert encoded["fixed_context_mask"][:4] == [1, 1, 1, 1]
        assert encoded["diffusion_loss_mask"][:4] == [0, 0, 0, 0]
        assert source in _UNCONDITIONAL_SOURCES
    assert len(encoded["input_ids"]) <= _BUDGET_MAX_LENGTH
