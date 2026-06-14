import torch

from downstream.common import CdrInfillCollator, ChainPaddingCollator, Sab23H2Collator
from dllm.pipelines.bioseq import Esm2ProteinTokenizer


def test_chain_padding_collator_masks_light_prompt_region():
    tokenizer = Esm2ProteinTokenizer()
    collator = ChainPaddingCollator(tokenizer=tokenizer)
    input_ids, chain_ids, meta = collator.stack_chains(
        ["EVQLVESGGGLVQPGGSLRLSCAASG"],
        ["DIQMTQSPSSLSASVGDRVTITC"],
        mask_light_from=4,
    )
    assert input_ids.shape == (1, 278)
    assert chain_ids.shape == input_ids.shape
    assert meta["heavy_max_len"] == 150
    assert meta["light_max_len"] == 128
    assert input_ids[0, 150] == tokenizer.cls_token_id
    assert tokenizer.mask_token_id in input_ids[0, 150:].tolist()
    assert input_ids[0, 151:155].ne(tokenizer.mask_token_id).all()
    assert input_ids[0, 155:].eq(tokenizer.mask_token_id).all()


def test_chain_padding_collator_masks_empty_light_chain_for_generation():
    tokenizer = Esm2ProteinTokenizer()
    collator = ChainPaddingCollator(tokenizer=tokenizer)
    input_ids, chain_ids, _ = collator.stack_chains(
        ["EVQLVESGGGLVQPGGSLRLSCAASG"],
        [""],
        mask_light_from=4,
    )
    assert input_ids.shape == (1, 278)
    assert chain_ids.shape == input_ids.shape
    assert input_ids[0, 150] == tokenizer.cls_token_id
    assert input_ids[0, 151:].eq(tokenizer.mask_token_id).all()


def test_cdr_infill_collator_builds_airgen_shapes():
    tokenizer = Esm2ProteinTokenizer()
    collator = CdrInfillCollator(tokenizer=tokenizer)
    chains, chain_ids, labels = collator(
        [
            ("EVQLVESGGGLVQPGGSLRLSCAASG", "DIQMTQSPSSLSASVGDRVTITC", "GFTF", (10, 13)),
        ]
    )
    assert chains.shape == (1, 278)
    assert chain_ids.shape == chains.shape
    assert labels.shape == chains.shape
    assert tokenizer.mask_token_id in chains[0].tolist()


def test_sab23h2_collator_encodes_mask_tokens():
    tokenizer = Esm2ProteinTokenizer()
    collator = Sab23H2Collator(tokenizer=tokenizer)
    chains, chain_ids, labels = collator(
        [
            ("ACDE", "FG<mask>HI", "ACDE", "FGHI"),
        ]
    )
    assert chains.shape == labels.shape == (1, 278)
    assert tokenizer.mask_token_id in chains[0].tolist()
