"""Verify full-token targets, semantic gradients, label isolation and round trips.

Run after activating pllm: USE_TF=0 python -B -m pytest THIS_FILE -q
"""
from dataclasses import replace

import pytest
import torch

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.mlm import BertMLMModel
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import (
    FullTokenMLMCollator, FullTokenMLMModel, corrupt, excluded_ids)
from scripts.research.bert_mlm_eval import fixed_corruption


def config():
    return FixedChainConfig(d_model=32, n_layers=2, n_heads=4, mlp_hidden_size=64,
                            first_chain_capacity=167, second_chain_capacity=135, full_capacity=304)


def record(relation="binding", synthetic=False):
    heavy = BioSeqChain("ACDEFGHIK", "antibody_heavy",
                       metadata={"synthetic_residue_mask": [int(synthetic)]*3+[0]*6})
    return BioSeqRecord([BioSeqChain("LMNPQRST", "antigen"), heavy,
                        BioSeqChain("VWYACD", "antibody_light")],
                       "antibody_antigen", "test", labels={"relation": relation})


def test_every_observed_token_except_delimiters_is_a_target():
    collator = FullTokenMLMCollator(config())
    batch = collator([record(synthetic=True), record()])
    expected = batch["attention_mask"].clone()
    for token in excluded_ids(collator.tokenizer):
        expected &= batch["input_ids"].ne(token)
    assert torch.equal(batch["eligible_mask"], expected)
    assert torch.equal(batch["labels"].ne(-100), expected)
    assert batch["context_residue_mask"].sum(1).tolist() == [8, 8]
    assert batch["receptor_residue_mask"].sum(1).tolist() == [12, 15]
    for token in ("<ab>", "<binding>"):
        positions = batch["input_ids"].eq(collator.tokenizer.special_id(token))
        assert positions.any() and batch["eligible_mask"][positions].all()
    for token_id in (collator.tokenizer.eos_token_id, collator.tokenizer.chain_separator_id()):
        positions = batch["input_ids"].eq(token_id)
        assert positions.any() and batch["eligible_mask"][positions].all()


def test_relation_visible_for_mlm_but_hidden_from_all_probe_features():
    rows = [record(label) for label in ("binding", "nonbinding", "unknown")]
    raw = FullTokenMLMCollator(config())(rows)
    assert not torch.equal(raw["input_ids"][0], raw["input_ids"][1])
    blind = FullTokenMLMCollator(config(), hide_relation=True)(rows)
    for value in FullTokenMLMModel.feature_args(blind).values():
        assert torch.equal(value[0], value[1]) and torch.equal(value[0], value[2])
    assert not (blind["eligible_mask"] & blind["relation_mask"]).any()


def test_noise_and_labels_include_grammar_without_corrupting_reserved_positions():
    collator = FullTokenMLMCollator(config())
    clean = collator([record()]*512)
    noisy = corrupt(clean, collator.tokenizer, torch.Generator().manual_seed(16))
    selected = noisy["labels"].ne(-100)
    assert abs(float(selected.sum()/clean["eligible_mask"].sum())-.15)<.015
    for kind, expected in ((1,.8),(2,.1),(3,.1)):
        assert abs(float(noisy["noise_kind"].eq(kind).sum()/selected.sum())-expected)<.025
    assert torch.equal(noisy["labels"][selected], clean["input_ids"][selected])
    assert torch.equal(noisy["input_ids"][~selected], clean["input_ids"][~selected])
    replaced = noisy["noise_kind"].eq(2)
    assert (noisy["input_ids"][replaced] != clean["input_ids"][replaced]).all()
    for token in excluded_ids(collator.tokenizer):
        assert not noisy["input_ids"][replaced].eq(token).any()
    for mask in (clean["relation_mask"], clean["terminal_mask"], clean["context_residue_mask"]):
        assert (selected & mask).any()


def test_eos_and_grammar_output_rows_receive_supervised_gradients(tmp_path):
    torch.manual_seed(42)
    model = FullTokenMLMModel(config())
    collator = FullTokenMLMCollator(model.config)
    batch = collator([record()])
    # All eligible labels selected deliberately to exercise every target family.
    ids = batch["input_ids"].clone()
    batch["input_ids"] = ids.masked_fill(batch["eligible_mask"], model.tokenizer.mask_token_id)
    out = model(**batch)
    assert torch.isfinite(out["loss_sum"])
    out["loss_sum"].backward()
    targets = [model.tokenizer.eos_token_id, model.tokenizer.chain_separator_id(),
               model.tokenizer.special_id("<ab>"), model.tokenizer.special_id("<binding>")]
    for i in targets:
        assert model.backbone.transformer.ff_out.weight.grad[i].abs().sum()>0
    assert model.backbone.transformer.ff_out.weight.grad[list(excluded_ids(model.tokenizer))].eq(0).all()
    for token in ("<prots>","<protd>"):
        assert model.backbone.transformer.wte.weight.grad[model.tokenizer.special_id(token)].abs().sum()>0
    model.eval().save_pretrained(tmp_path)
    restored = FullTokenMLMModel.from_pretrained(tmp_path).eval()
    assert torch.equal(model(**batch)["logits"],restored(**batch)["logits"])
    with pytest.raises(ValueError):
        BertMLMModel.from_pretrained(tmp_path)


def test_full_token_fixed_noise_is_padding_independent():
    collator = FullTokenMLMCollator(config())
    rows = [record(), replace(record(), chains=[BioSeqChain("A"*50,"antigen"),*record().chains[1:]])]
    both = fixed_corruption(collator(rows),collator.tokenizer,[31,32],corrupt_fn=corrupt)
    for i,row in enumerate(rows):
        one = fixed_corruption(collator([row]),collator.tokenizer,[31+i],corrupt_fn=corrupt)
        for key in ("input_ids","labels","noise_kind"):
            assert torch.equal(both[key][i,:one[key].shape[1]],one[key][0])
