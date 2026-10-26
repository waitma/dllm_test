"""Verify conditional noise, edit recovery, importance sampling and serialization.

Activate pllm; USE_TF=0 python -B -m pytest THIS_FILE -q.
"""
from dataclasses import replace

import pytest
import torch

from scripts.tests.immune_llada.test_full_token_mlm import config, record
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord, CHAIN_ROLE_TO_ID
from dllm.pipelines.immune_llada.data.grammar import GRAMMAR_RELATIONS
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMModel
from scripts.research.diffusion_continuation_objective import (
    DiffusionEditCollator, DiffusionEditModel, default_recipe, corrupt, sample_times)
from scripts.research.diffusion_continuation_eval import fixed_batch


def noise(rows, **kwargs):
    collator = DiffusionEditCollator(config())
    clean = collator(rows)
    return clean, corrupt(clean, collator.tokenizer, torch.Generator().manual_seed(42),
                          default_recipe(), **kwargs), collator.tokenizer


@pytest.mark.parametrize("edit", [False, True])
def test_context_untouched_relation_hidden_and_complementary_targets(edit):
    clean, noisy, tok = noise([record(synthetic=True)]*4, mode="joint", ratio=.3, edit=edit)
    allowed = clean["target_slot_mask"] | clean["relation_mask"]
    assert torch.equal(noisy["input_ids"][~allowed.repeat(2,1)], clean["input_ids"].repeat(2,1)[~allowed.repeat(2,1)])
    assert noisy["input_ids"][noisy["relation_mask"]].eq(tok.mask_token_id).all()
    assert (noisy["labels"][noisy["relation_mask"]] >= 0).all()
    assert not (noisy["labels"].ne(-100) & ~(clean["eligible_mask"] | clean["relation_mask"]).repeat(2,1)).any()
    selected = noisy["noise_kind"].eq(2 if edit else 1)
    assert torch.equal(selected[:4] | selected[4:], clean["eligible_mask"])
    assert not (selected[:4] & selected[4:]).any()
    if edit:
        original = clean["input_ids"].repeat(2,1)
        assert (noisy["input_ids"][selected] != original[selected]).all()
        assert not noisy["input_ids"][selected].eq(tok.mask_token_id).any()


def test_conditional_direction_and_single_chain_fallback():
    clean, noisy, _ = noise([record()], mode="first_to_second", ratio=1., edit=False)
    # Derive the first-chain role from the actual collator rather than assume ids.
    first_id = clean["chain_ids"][0, clean["eligible_mask"][0]].min()
    first = clean["chain_ids"].eq(first_id) & clean["target_slot_mask"]
    assert torch.equal(noisy["input_ids"][:1][first], clean["input_ids"][first])
    assert noisy["labels"][:1][first].eq(-100).all()
    one = replace(record(), chains=record().chains[:2])
    _, result, _ = noise([one], mode="first_to_second", ratio=1., edit=False)
    assert result["noise_kind"].eq(1).any()


def test_fixed_noise_padding_world_partition_invariant():
    collator = DiffusionEditCollator(config())
    rows = [record(), replace(record(), chains=[BioSeqChain("A"*80,"antigen"), *record().chains[1:]])]
    samples = [{"record": r} for r in rows]
    both = fixed_batch(collator, samples, [0,1], default_recipe(), "joint", .5, True)
    for i in (0,1):
        one = fixed_batch(collator, samples, [i], default_recipe(), "joint", .5, True)
        for key in ("input_ids","labels","noise_kind","loss_weights"):
            torch.testing.assert_close(both[key][i,:one[key].shape[1]],one[key][0])


def test_inverse_sampling_corrects_uniform_time_and_complement():
    recipe=default_recipe()
    times, weights=sample_times((100000,),torch.Generator().manual_seed(1),recipe)
    assert times.mean()<.4
    assert float(weights.mean()) == pytest.approx(1.,abs=.01)
    assert float((times*weights).mean()) == pytest.approx(.5,abs=.01)
    assert float(((1-times)*weights).mean()) == pytest.approx(.5,abs=.01)


def test_parent_identity_eos_relation_gradients_and_strict_reload(tmp_path):
    parent=FullTokenMLMModel(config()).eval()
    parent.save_pretrained(tmp_path/'parent')
    model=DiffusionEditModel.from_parent(tmp_path/'parent',default_recipe(1.)).eval()
    for key,value in parent.state_dict().items():
        assert torch.equal(value,model.state_dict()[key])
    _,batch,tok=noise([record(label) for label in ('binding','nonbinding','unknown')],ratio=1.,edit=False)
    output=model(**batch)
    assert torch.isfinite(output['loss_sum'])
    output['loss_sum'].backward()
    for token in [tok.eos_token_id,*[tok.special_id('<'+x+'>') for x in ('binding','nonbinding','unknown')]]:
        assert model.backbone.transformer.ff_out.weight.grad[token].abs().sum()>0
    model.save_pretrained(tmp_path/'child')
    child=DiffusionEditModel.from_pretrained(tmp_path/'child').eval()
    torch.testing.assert_close(model(**batch)['loss_sum'],child(**batch)['loss_sum'],rtol=0,atol=0)
    with pytest.raises(ValueError):
        FullTokenMLMModel.from_pretrained(tmp_path/'child')


def test_corruption_rng_restore_exact():
    collator=DiffusionEditCollator(config())
    batch=collator([record()]*12)
    rng=torch.Generator().manual_seed(52)
    corrupt(batch,collator.tokenizer,rng,default_recipe())
    state=rng.get_state().clone()
    expected=corrupt(batch,collator.tokenizer,rng,default_recipe())
    rng.set_state(state)
    actual=corrupt(batch,collator.tokenizer,rng,default_recipe())
    assert all(torch.equal(value,actual[key]) for key,value in expected.items())


def test_peptide_and_mhc_sequences_are_clean_in_every_task_mode():
    row=BioSeqRecord([BioSeqChain('ACDEFGHIK','peptide'),BioSeqChain('LMNPQRSTVWY','mhc'),
                      BioSeqChain('CASSFG','tcr_beta'),BioSeqChain('CAVRDE','tcr_alpha')],
                     'tcr_pmhc','test',labels={'relation':'binding'})
    for mode in ('joint','independent','first_to_second','second_to_first','single'):
        clean,noisy,_=noise([row]*8,mode=mode)
        context=(clean['role_ids'].eq(CHAIN_ROLE_TO_ID['peptide'])
                 |clean['role_ids'].eq(CHAIN_ROLE_TO_ID['mhc'])) & clean['residue_mask']
        assert context.sum()==8*20
        assert torch.equal(noisy['input_ids'][context.repeat(2,1)],clean['input_ids'].repeat(2,1)[context.repeat(2,1)])
        assert noisy['labels'][context.repeat(2,1)].eq(-100).all()


def test_every_supported_relation_category_is_a_hidden_target():
    _,noisy,tok=noise([record(label) for label in GRAMMAR_RELATIONS],ratio=.5)
    seen=set(noisy['labels'][noisy['relation_mask']].tolist())
    assert seen=={tok.special_id('<'+label+'>') for label in GRAMMAR_RELATIONS}
    assert noisy['input_ids'][noisy['relation_mask']].eq(tok.mask_token_id).all()
    model=DiffusionEditModel(config(),default_recipe())
    loss=model(**noisy)['loss_sum']
    assert torch.isfinite(loss)
    loss.backward()
    assert all(model.backbone.transformer.ff_out.weight.grad[token].abs().sum()>0 for token in seen)
