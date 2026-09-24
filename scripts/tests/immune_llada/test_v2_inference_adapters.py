"""CPU regressions for fixed-receptor v2 inference adapters.

Run with::

    python -m pytest scripts/tests/immune_llada/test_v2_inference_adapters.py -q

The tests use the in-memory grammar-v2 tokenizer/collator only; no checkpoint or
GPU is required.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import build_generation_mask
from downstream.grammar.common import antibody_pair_record, tcr_pair_record
from downstream.grammar.light_chain_pairing import (
    generate_for_batch,
    resolve_light_length_mode,
)
from downstream.grammar.masks import (
    chain_residue_positions,
    chain_slot_positions,
    chain_slot_positions_by_chain,
    light_chain_generation_partial_mask,
    residue_positions_by_chain,
    tcr_generation_partial_mask,
)
from downstream.grammar.metrics import extract_chain_sequence
from downstream.grammar.tcr_generation import BioSeqTcrSampler
from examples.llada.protein_fusion_model import RemapCollator


def _fixed_collator() -> tuple[GrammarTokenizer, GrammarBioSeqCollator]:
    tokenizer = GrammarTokenizer()
    return tokenizer, GrammarBioSeqCollator(tokenizer, fixed_receptor_lengths=True)


def _assert_fixed_slot_metadata(batch: dict[str, torch.Tensor]) -> None:
    """Every fixed decoder slot carries the metadata needed by inference."""

    slots = batch["chain_slot_mask"]
    assert slots.any()
    assert batch["position_ids_chain"][slots].ge(0).all()
    assert batch["position_ids_inner"][slots].ge(0).all()
    assert batch["chain_ids"][slots].ge(0).all()
    assert batch["encoder_input_ids"].shape[-1] in {168, 136}


def test_generated_eos_truncates_each_chain_independently() -> None:
    tokenizer, collator = _fixed_collator()
    batch = collator([antibody_pair_record("ACDEFG", "HIKLMN")])
    _assert_fixed_slot_metadata(batch)

    generated = batch["input_ids"][0].clone()
    heavy_slots = chain_slot_positions(
        batch["input_ids"],
        batch["attention_mask"],
        batch["chain_slot_mask"],
        tokenizer,
        chain=0,
        position_ids_chain=batch["position_ids_chain"],
        residue_mask=batch["residue_mask"],
    )[0]
    light_slots = chain_slot_positions(
        batch["input_ids"],
        batch["attention_mask"],
        batch["chain_slot_mask"],
        tokenizer,
        chain=1,
        position_ids_chain=batch["position_ids_chain"],
        residue_mask=batch["residue_mask"],
    )[0]
    assert len(heavy_slots) == 167
    assert len(light_slots) == 135

    # The output is grammar-space, as returned by run_grammar_generate. Put
    # different first generated EOS values into the two decoder canvases.
    generated[heavy_slots[2]] = tokenizer.eos_token_id
    generated[light_slots[4]] = tokenizer.eos_token_id
    position_ids = batch["position_ids_chain"][0]
    slots = batch["chain_slot_mask"][0]
    attention = batch["attention_mask"][0]
    residues = batch["residue_mask"][0]

    heavy = extract_chain_sequence(
        generated,
        attention,
        residues,
        tokenizer,
        chain=0,
        chain_eos_mask=batch["chain_eos_mask"][0],
        position_ids_chain=position_ids,
        chain_slot_mask=slots,
    )
    light = extract_chain_sequence(
        generated,
        attention,
        residues,
        tokenizer,
        chain=1,
        chain_eos_mask=batch["chain_eos_mask"][0],
        position_ids_chain=position_ids,
        chain_slot_mask=slots,
    )
    assert heavy == "AC"
    assert light == "HIKL"

    # Clean target EOS metadata must not become an inference boundary.
    assert extract_chain_sequence(
        generated,
        attention,
        residues,
        tokenizer,
        chain=0,
        chain_eos_mask=torch.zeros_like(batch["chain_eos_mask"][0]),
        position_ids_chain=position_ids,
        chain_slot_mask=slots,
    ) == heavy
    assert extract_chain_sequence(
        generated,
        attention,
        residues,
        tokenizer,
        chain=1,
        chain_eos_mask=torch.zeros_like(batch["chain_eos_mask"][0]),
        position_ids_chain=position_ids,
        chain_slot_mask=slots,
    ) == light


@pytest.mark.parametrize("with_position_ids", [True, False])
@pytest.mark.parametrize("clean_masks", ["original", "empty", "inverted"])
def test_generated_chains_can_extend_beyond_clean_eos(
    with_position_ids: bool, clean_masks: str
) -> None:
    tokenizer, collator = _fixed_collator()
    batch = collator([antibody_pair_record("AC", "H")])
    slots_by_chain = chain_slot_positions_by_chain(batch)[0]
    generated = batch["input_ids"][0].clone()
    expected_chains = {0: "ACDEFG", 1: "HIKLMNPQR"}
    for chain, expected in expected_chains.items():
        positions = slots_by_chain[chain]
        generated[positions] = tokenizer.encode_residues("W")[0]
        generated[positions[:len(expected)]] = torch.tensor(tokenizer.encode_residues(expected))
        generated[positions[len(expected)]] = tokenizer.eos_token_id
        generated[positions[len(expected) + 3]] = tokenizer.eos_token_id

    residue_mask = batch["residue_mask"][0]
    eos_mask = batch["chain_eos_mask"][0]
    if clean_masks == "empty":
        residue_mask = torch.zeros_like(residue_mask)
        eos_mask = torch.zeros_like(eos_mask)
    elif clean_masks == "inverted":
        residue_mask = ~residue_mask.bool()
        eos_mask = ~eos_mask.bool()

    for chain, expected in expected_chains.items():
        assert extract_chain_sequence(
            generated,
            batch["attention_mask"][0],
            residue_mask,
            tokenizer,
            chain=chain,
            chain_eos_mask=eos_mask,
            position_ids_chain=batch["position_ids_chain"][0] if with_position_ids else None,
            chain_slot_mask=batch["chain_slot_mask"][0],
        ) == expected


@pytest.mark.parametrize("with_position_ids", [True, False])
def test_no_generated_eos_decodes_entire_canvas_without_forcing_termination(
    with_position_ids: bool,
) -> None:
    tokenizer, collator = _fixed_collator()
    batch = collator([antibody_pair_record("AC", "H")])
    slots_by_chain = chain_slot_positions_by_chain(batch)[0]
    generated = batch["input_ids"][0].clone()
    expected_chains = {0: "A" * 166 + "C", 1: "W" * 134 + "Y"}
    for chain, expected in expected_chains.items():
        positions = slots_by_chain[chain]
        assert len(positions) == len(expected)
        generated[positions] = torch.tensor(tokenizer.encode_residues(expected))
    original_generated = generated.clone()

    for chain, expected in expected_chains.items():
        assert extract_chain_sequence(
            generated,
            batch["attention_mask"][0],
            torch.zeros_like(batch["residue_mask"][0]),
            tokenizer,
            chain=chain,
            chain_eos_mask=batch["chain_eos_mask"][0],
            position_ids_chain=batch["position_ids_chain"][0] if with_position_ids else None,
            chain_slot_mask=batch["chain_slot_mask"][0],
        ) == expected
    assert torch.equal(generated, original_generated)


def test_full_chain_masks_include_eos_and_keep_only_explicit_prompts() -> None:
    tokenizer, collator = _fixed_collator()
    antibody_batch = collator([antibody_pair_record("ACDEFG", "HIKLMN")])
    light_partial = light_chain_generation_partial_mask(
        antibody_batch, tokenizer, prompt_residues=2
    )
    light_generation = build_generation_mask(antibody_batch, light_partial)
    light_slots = chain_slot_positions(
        antibody_batch["input_ids"],
        antibody_batch["attention_mask"],
        antibody_batch["chain_slot_mask"],
        tokenizer,
        chain="light",
        position_ids_chain=antibody_batch["position_ids_chain"],
        residue_mask=antibody_batch["residue_mask"],
    )[0]
    light_residues = chain_residue_positions(
        antibody_batch["input_ids"],
        antibody_batch["attention_mask"],
        antibody_batch["residue_mask"],
        tokenizer,
        chain="light",
        position_ids_chain=antibody_batch["position_ids_chain"],
    )[0]
    assert len(light_slots) == 135
    assert light_slots[-1] not in light_residues
    assert all(not light_partial[0, position] for position in light_slots[2:])
    assert all(light_partial[0, position] for position in light_residues[:2])
    assert all(light_generation[0, position] for position in light_slots[2:])
    assert not light_generation[0, light_residues[0]]
    heavy_slots = chain_slot_positions(
        antibody_batch["input_ids"],
        antibody_batch["attention_mask"],
        antibody_batch["chain_slot_mask"],
        tokenizer,
        chain="heavy",
        position_ids_chain=antibody_batch["position_ids_chain"],
        residue_mask=antibody_batch["residue_mask"],
    )[0]
    assert all(light_partial[0, position] for position in heavy_slots)
    assert light_partial[0, antibody_batch["structure_token_mask"][0]].all()

    tcr_batch = collator([tcr_pair_record("CAVR", "CASS")])
    tcr_partial = tcr_generation_partial_mask(
        tcr_batch, target_chain_indices={0, 1}, prompt_residues=1
    )
    tcr_generation = build_generation_mask(tcr_batch, tcr_partial)
    slots_by_chain = chain_slot_positions_by_chain(tcr_batch)
    residues_by_chain = residue_positions_by_chain(tcr_batch)
    for chain_index, slots in slots_by_chain[0].items():
        if chain_index not in {0, 1}:
            continue
        prompt = set(residues_by_chain[0][chain_index][:1])
        assert all(tcr_partial[0, position] == (position in prompt) for position in slots)
        assert all(tcr_generation[0, position] == (position not in prompt) for position in slots)
    assert tcr_partial[0, tcr_batch["structure_token_mask"][0]].all()


def test_fixed_v2_pairing_ignores_hidden_reference_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    import downstream.grammar.light_chain_pairing as pairing

    tokenizer, collator = _fixed_collator()
    model = SimpleNamespace(config=SimpleNamespace(fixed_receptor_lengths=True))
    captured: list[dict[str, torch.Tensor]] = []

    def record_generation(model, batch, **kwargs):
        del model, kwargs
        captured.append({
            key: value.clone()
            for key, value in batch.items()
            if torch.is_tensor(value)
        })
        return batch["input_ids"].clone(), torch.zeros_like(batch["input_ids"], dtype=torch.float32)

    monkeypatch.setattr(pairing, "run_grammar_generate", record_generation)
    common: Any = {
        "model": model,
        "collator": collator,
        "tokenizer": tokenizer,
        "device": torch.device("cpu"),
        "num_seqs": 1,
        "max_iter": 2,
        "sampling_strategy": "argmax",
        "temperature": 1.0,
        "light_prompt_tokens": 3,
        "light_length_mode": "auto",
        "length_prior": None,
    }
    first = generate_for_batch(
        [("ACDEFG", "DIQMTQSPSS", {"case": "short"})], **common
    )
    second = generate_for_batch(
        [("ACDEFG", "DIQ" + "W" * 80, {"case": "long"})], **common
    )

    assert len(captured) == 2
    assert first[0]["light_length_mode"] == second[0]["light_length_mode"] == "fixed_v2"
    assert first[0]["target_light_length"] == second[0]["target_light_length"] == 134
    assert first[0]["ref_light_length"] != second[0]["ref_light_length"]
    for key in captured[0]:
        assert torch.equal(captured[0][key], captured[1][key]), key


def test_pairing_mode_auto_and_explicit_mismatch_rejection() -> None:
    fixed = SimpleNamespace(config=SimpleNamespace(fixed_receptor_lengths=True))
    legacy = SimpleNamespace(config=SimpleNamespace(fixed_receptor_lengths=False))
    assert resolve_light_length_mode(fixed, "auto") == "fixed_v2"
    assert resolve_light_length_mode(legacy, "auto") == "reference"
    with pytest.raises(ValueError, match="incompatible"):
        resolve_light_length_mode(fixed, "reference")
    with pytest.raises(ValueError, match="incompatible"):
        resolve_light_length_mode(fixed, "prior")
    with pytest.raises(ValueError, match="requires"):
        resolve_light_length_mode(legacy, "fixed_v2")


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("mode", ["auto", "fixed_v2"])
def test_pairing_fixed_model_rejects_legacy_collator_before_collation(
    monkeypatch: pytest.MonkeyPatch, mode: str, wrapped: bool
) -> None:
    import downstream.grammar.light_chain_pairing as pairing

    tokenizer = GrammarTokenizer()
    collator = GrammarBioSeqCollator(tokenizer, fixed_receptor_lengths=False)
    if wrapped:
        collator = RemapCollator(collator, torch.arange(tokenizer.vocab_size) + 100)
    model = SimpleNamespace(config=SimpleNamespace(fixed_receptor_lengths=True))

    def unexpected_call(*args, **kwargs):
        pytest.fail("Model/collator mismatch must fail before collation or generation")

    monkeypatch.setattr(GrammarBioSeqCollator, "__call__", unexpected_call)
    monkeypatch.setattr(pairing, "run_grammar_generate", unexpected_call)
    with pytest.raises(ValueError, match=r"collator\.fixed_receptor_lengths=True"):
        generate_for_batch(
            [("ACDEFG", "DIQMTQSPSS", {})],
            model=model,
            collator=collator,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            num_seqs=1,
            max_iter=2,
            sampling_strategy="argmax",
            temperature=1.0,
            light_prompt_tokens=3,
            light_length_mode=mode,
            length_prior=None,
        )


def _eval_bundle(*, fixed: bool, remapped: bool = True):
    tokenizer = GrammarTokenizer()
    collator = GrammarBioSeqCollator(tokenizer, fixed_receptor_lengths=fixed)
    model = SimpleNamespace(config=SimpleNamespace(fixed_receptor_lengths=fixed))
    if remapped:
        lookup = torch.arange(tokenizer.vocab_size) + 100
        inverse = torch.full((int(lookup.max()) + 1,), -1, dtype=torch.long)
        inverse[lookup] = torch.arange(tokenizer.vocab_size)
        model.llada_to_grammar_ids = inverse
        collator = RemapCollator(collator, lookup)
    model._fusion_eval_collator = collator
    return model, tokenizer, collator


@pytest.fixture
def copy_generation(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, torch.Tensor]]:
    import downstream.grammar.common as common

    captured = []

    def perfect_copy(model, batch, **kwargs):
        del model, kwargs
        captured.append({key: value.clone() for key, value in batch.items() if torch.is_tensor(value)})
        return batch["input_ids"].clone(), torch.zeros_like(batch["input_ids"], dtype=torch.float32)

    # Keep run_grammar_generate's real inverse vocabulary mapping in the path.
    monkeypatch.setattr(common, "generate_bioseq", perfect_copy)
    return captured


@pytest.mark.parametrize("mode", ["auto", "fixed_v2"])
@pytest.mark.parametrize("collator_policy_only", [False, True])
def test_pairing_accepts_real_remap_collator(
    copy_generation, mode: str, collator_policy_only: bool
) -> None:
    from downstream.grammar.common import build_eval_collator

    model, tokenizer, collator = _eval_bundle(fixed=True)
    assert not hasattr(collator, "fixed_receptor_lengths")
    assert collator.base_collator.fixed_receptor_lengths
    if collator_policy_only:
        del model.config.fixed_receptor_lengths
    rows = generate_for_batch(
        [("ACDEFG", "DIQMTQSPSS", {})],
        model=model,
        collator=build_eval_collator(model, tokenizer),
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        num_seqs=1,
        max_iter=2,
        sampling_strategy="argmax",
        temperature=1.0,
        light_prompt_tokens=3,
        light_length_mode=mode,
        length_prior=None,
    )
    assert len(copy_generation) == 1
    _assert_fixed_slot_metadata(copy_generation[0])
    assert copy_generation[0]["input_ids"].ge(100).all()
    assert rows[0]["light_length_mode"] == "fixed_v2"
    assert rows[0]["target_light_length"] == 134
    assert rows[0]["gen_l_sequence"] == "DIQ" + "A" * 131


@pytest.mark.parametrize("remapped", [False, True])
def test_labels_to_grammar_preserves_ignore_index(remapped: bool) -> None:
    from downstream.grammar.common import labels_to_grammar

    labels = torch.tensor([[-100, 0, 1, 2]], dtype=torch.long)
    original = labels.clone()
    model = SimpleNamespace()
    if remapped:
        # Entry zero is mapped too: clamping -100 before lookup must not lose it.
        model.llada_to_grammar_ids = torch.tensor([7, 6, 5])
    converted = labels_to_grammar(model, labels)
    expected = torch.tensor([[-100, 7, 6, 5]]) if remapped else original
    assert torch.equal(converted, expected)
    assert torch.equal(labels, original)
    assert converted.dtype == labels.dtype
    assert converted.device == labels.device


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("remapped", [False, True])
@pytest.mark.parametrize("mode,target", [("cdrh3", "DEF"), ("cdrl3", "KLM")])
@pytest.mark.parametrize("entrypoint", ["record", "dataset"])
def test_cdr_perfect_copy_aar_is_one(
    monkeypatch: pytest.MonkeyPatch,
    copy_generation,
    capsys: pytest.CaptureFixture[str],
    fixed: bool,
    remapped: bool,
    mode: str,
    target: str,
    entrypoint: str,
) -> None:
    import downstream.grammar.cdr_infill as cdr

    model, tokenizer, _ = _eval_bundle(fixed=fixed, remapped=remapped)
    monkeypatch.setattr(cdr, "build_grammar_tokenizer", lambda: tokenizer)
    if entrypoint == "record":
        aar = cdr.evaluate_record(
            model, "ACDEFG", "HIKLMN", mode,
            max_iter=2, sampling_strategy="argmax", temperature=1.0,
            device="cpu", target_subsequence=target,
        )
        assert aar == 1.0
    else:
        monkeypatch.setattr(cdr, "_load_eval_model", lambda args, device: (model, tokenizer))
        monkeypatch.setattr(cdr, "resolve_sabdab_test_json", lambda *args: "in-memory")
        monkeypatch.setattr(
            cdr, "SAbDabDataset",
            lambda *args, **kwargs: [("ACDEFG", "HIKLMN", target, (0, 0), mode)],
        )
        cdr.evaluate(SimpleNamespace(
            device="cpu", num_folds=1, test_set="in-memory", mode=mode,
            max_samples=1, max_iter=2, sampling_strategy="argmax", temperature=1.0,
        ))
        assert "Average AAR: 100.0" in capsys.readouterr().out
    assert len(copy_generation) == 1
    batch = copy_generation[0]
    if remapped:
        assert batch["input_ids"].ge(100).all()
        assert batch["labels"][batch["labels"].ne(-100)].ge(100).all()
    if fixed:
        _assert_fixed_slot_metadata(batch)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("record_type", ["antibody", "tcr"])
def test_remapped_labels_and_generated_tokens_share_grammar_vocabulary(
    copy_generation, fixed: bool, record_type: str
) -> None:
    from downstream.grammar.common import labels_to_grammar, run_grammar_generate
    from downstream.grammar.metrics import masked_token_accuracy

    model, _, collator = _eval_bundle(fixed=fixed)
    record = (
        antibody_pair_record("ACDEFG", "HIKLMN")
        if record_type == "antibody"
        else tcr_pair_record("CAVR", "CASS")
    )
    batch = collator([record])
    original_labels = batch["labels"].clone()
    outputs, _ = run_grammar_generate(model, batch)
    labels = labels_to_grammar(model, batch["labels"])
    scoring_mask = batch["attention_mask"] & batch["residue_mask"]
    assert scoring_mask.any()
    assert masked_token_accuracy(outputs, original_labels, scoring_mask).item() == 0.0
    assert masked_token_accuracy(outputs, labels, scoring_mask).item() == 1.0
    assert torch.equal(labels.eq(-100), original_labels.eq(-100))
    assert torch.equal(batch["labels"], original_labels)


@pytest.mark.parametrize("fixed", [False, True])
def test_tcr_sampler_reads_policy_from_real_remap_collator(
    monkeypatch: pytest.MonkeyPatch, fixed: bool
) -> None:
    import downstream.grammar.tcr_generation as tcr

    model, tokenizer, collator = _eval_bundle(fixed=fixed)
    del model.config.fixed_receptor_lengths
    monkeypatch.setattr(tcr, "load_grammar_checkpoint", lambda *args, **kwargs: (model, tokenizer))
    sampler = BioSeqTcrSampler("in-memory", device="cpu")
    assert sampler.collator is collator
    assert sampler.fixed_v2 is fixed


@pytest.mark.parametrize("remapped", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_tcr_infill_perfect_copy_window_aar_is_one(
    monkeypatch: pytest.MonkeyPatch, copy_generation, remapped: bool, fixed: bool
) -> None:
    import downstream.grammar.tcr_generation as tcr

    model, tokenizer, _ = _eval_bundle(fixed=fixed, remapped=remapped)
    monkeypatch.setattr(tcr, "load_grammar_checkpoint", lambda *args, **kwargs: (model, tokenizer))
    sampler = BioSeqTcrSampler("in-memory", device="cpu")
    rows = sampler.infill_cdr3b(["CASSF", "CAVRDSF"], mask_width=3)
    assert len(rows) == 2
    assert all(row["length_ok"] and row["pred"] == row["truth"] for row in rows)
    hit = sum(sum(a == b for a, b in zip(row["truth_window"], row["pred_window"])) for row in rows)
    total = sum(len(row["truth_window"]) for row in rows)
    assert total > 0
    assert hit / total == 1.0
