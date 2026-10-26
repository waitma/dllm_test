"""Activate pllm; USE_TF=0 python -B -m pytest THIS_FILE -q."""
from dataclasses import replace

import pytest
import torch

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from scripts.tests.immune_llada.test_full_token_mlm import config, record
from scripts.research.diffusion_cdr_tasks import CDRS, default_cdr_tasks, cdr_spans
from scripts.research.diffusion_continuation_objective import (
    DiffusionEditCollator, DiffusionEditModel, default_recipe, corrupt)
from scripts.research.diffusion_continuation_losses import weighted_token_loss


def antibody():
    # Repeated substrings deliberately make naive sequence.find() ambiguous.
    regions = dict(FR1="AAA", CDR1="AA", FR2="AAA", CDR2="AA", FR3="AAA", CDR3="AAAA", FR4="AAA")
    sequence = "".join(regions.values())
    return BioSeqRecord([BioSeqChain("ACDEFG", "antigen"),
        BioSeqChain(sequence, "antibody_heavy", regions=regions),
        BioSeqChain(sequence, "antibody_light", regions=regions)],
        "antibody_antigen", "test", labels={"relation": "binding"})


def recipe():
    return {**default_recipe(), "cdr_tasks": default_cdr_tasks()}


def batch(rows=None):
    collator = DiffusionEditCollator(config())
    return collator, collator([antibody()] if rows is None else rows)


@pytest.mark.parametrize("targets", [[name] for name in CDRS] + [["H1", "L3"], list(CDRS)])
def test_exact_requested_loops_and_every_other_residue_stays_clean(targets):
    collator, raw = batch()
    scope = "cdr_single" if len(targets) == 1 else "cdr_multi"
    out = corrupt(raw, collator.tokenizer, torch.Generator().manual_seed(42), recipe(),
                  mask_scope=scope, target_cdrs=targets, mode="single", ratio=1., edit=False)
    expected = torch.zeros_like(raw["eligible_mask"])
    for name in targets:
        expected |= raw["cdr_ids"].eq(CDRS.index(name)+1)
    target, relation = expected.repeat(2, 1), raw["relation_mask"].repeat(2, 1)
    clean = raw["input_ids"].repeat(2, 1)
    assert torch.equal(out["active_population"], target)
    assert out["input_ids"][target].eq(collator.tokenizer.mask_token_id).all()
    assert torch.equal(out["input_ids"][~(target | relation)], clean[~(target | relation)])
    assert torch.equal(out["labels"].ne(-100), target | relation)
    assert out["loss_weights"][~(target | relation)].eq(0).all()
    assert out["loss_weights"][target].sum() == pytest.approx(2/1.05)
    assert out["labels"][raw["terminal_mask"].repeat(2,1)].eq(-100).all()
    assert torch.equal(out["input_ids"][:1], out["input_ids"][1:])


def test_diffusion_complement_is_restricted_to_selected_loops():
    collator, raw = batch()
    out = corrupt(raw, collator.tokenizer, torch.Generator().manual_seed(14), recipe(),
        mask_scope="cdr_multi", target_cdrs=["H3", "L2"], ratio=.4, edit=False)
    masks = out["noise_kind"].eq(1)
    assert torch.equal(masks[:1] | masks[1:], out["active_population"][:1])
    assert not (masks[:1] & masks[1:]).any()
    fixed = ~out["active_population"] & ~out["relation_mask"]
    assert torch.equal(out["input_ids"][fixed], raw["input_ids"].repeat(2,1)[fixed])


def test_missing_synthetic_and_tcr_annotations_fall_back_without_fabricating_cdrs():
    ab = antibody()
    synthetic = replace(ab, chains=[ab.chains[0], replace(ab.chains[1], metadata={
        "synthetic_residue_mask": [1]+[0]*(len(ab.chains[1].sequence)-1)}), ab.chains[2]])
    incomplete = replace(ab, chains=[ab.chains[0], replace(ab.chains[1], regions={}), ab.chains[2]])
    tcr = BioSeqRecord([BioSeqChain("ACDE", "tcr_beta"), BioSeqChain("FGHI", "tcr_alpha")], "tcr", "test")
    collator, raw = batch([record(), synthetic, incomplete, tcr])
    assert not raw["cdr_ids"].any()
    out = corrupt(raw, collator.tokenizer, torch.Generator().manual_seed(42), recipe(), edit=False)
    assert out["mask_scope"].eq(0).all()
    with pytest.raises(ValueError, match="exact complete"):
        corrupt(raw, collator.tokenizer, torch.Generator().manual_seed(42), recipe(),
                mask_scope="cdr_single", edit=False)


def test_edit_branch_and_legacy_fixed_validation_are_numerically_unchanged():
    collator, raw = batch([antibody()]*32)
    for edit in (False, True):
        old = corrupt(raw, collator.tokenizer, torch.Generator().manual_seed(42), default_recipe(), edit=edit)
        new = corrupt(raw, collator.tokenizer, torch.Generator().manual_seed(42), recipe(), edit=edit,
                      mask_scope="chain")
        for name in old:
            assert torch.equal(old[name], new[name]), name
    old = corrupt(raw, collator.tokenizer, torch.Generator().manual_seed(42), default_recipe(), edit=True)
    new = corrupt(raw, collator.tokenizer, torch.Generator().manual_seed(42), recipe(), edit=True)
    assert all(torch.equal(value, new[key]) for key, value in old.items())


def test_task_ratios_endpoint_coverage_and_exact_rng_restore():
    collator, one = batch()
    raw = {k: v.repeat(4000, 1) for k, v in one.items()}
    rng = torch.Generator().manual_seed(42)
    state = rng.get_state().clone()
    out = corrupt(raw, collator.tokenizer, rng, recipe())
    scopes = out["mask_scope"][:4000, 0]
    edits = out["edit_task"][:4000, 0]
    assert float(edits.float().mean()) == pytest.approx(.25, abs=.025)
    assert scopes[edits].eq(0).all()
    for scope, expected in enumerate((.5, .25, .25)):
        assert float(scopes[~edits].eq(scope).float().mean()) == pytest.approx(expected, abs=.035)
    scoped = scopes.ne(0)
    assert float(out["cdr_full_mask"][:4000, 0][scoped].float().mean()) == pytest.approx(.25, abs=.04)
    counts = torch.stack([(out["active_population"][:4000] & raw["cdr_ids"].eq(i)).any(1)
                          for i in range(1,7)],1).sum(1)
    assert counts[scopes.eq(1)].eq(1).all()
    assert counts[scopes.eq(2)].ge(2).all()
    assert counts[scopes.eq(2)].eq(6).any()
    rng.set_state(state)
    replay = corrupt(raw, collator.tokenizer, rng, recipe())
    assert all(torch.equal(value, replay[key]) for key, value in out.items())


def test_context_logits_receive_no_direct_cdr_reconstruction_loss():
    collator, raw = batch()
    out = corrupt(raw, collator.tokenizer, torch.Generator().manual_seed(1), recipe(),
                  mask_scope="cdr_single", target_cdrs=["H3"], ratio=1., edit=False)
    logits = torch.randn(*out["input_ids"].shape, collator.tokenizer.vocab_size, requires_grad=True)
    loss = weighted_token_loss(logits, out["labels"], out["loss_weights"],
                               eos_id=collator.tokenizer.eos_token_id, per_view=True)
    torch.testing.assert_close(loss["view_loss_sums"].sum(), loss["loss_sum"].detach())
    loss["loss_sum"].backward()
    assert logits.grad[out["labels"].eq(-100)].eq(0).all()
    assert logits.grad[out["active_population"]].abs().sum() > 0
    assert "cdr_ids" not in DiffusionEditModel.feature_args(out)


def test_cdr_recipe_checkpoint_round_trip(tmp_path):
    model = DiffusionEditModel(config(), recipe()).eval()
    model.save_pretrained(tmp_path)
    restored = DiffusionEditModel.from_pretrained(tmp_path).eval()
    assert restored.recipe == recipe()
    assert all(torch.equal(p, restored.state_dict()[name]) for name, p in model.state_dict().items())


def test_fixed_cdr_validation_is_padding_invariant_and_counts_targets():
    from scripts.research.diffusion_continuation_eval import fixed_batch
    from scripts.research.diffusion_cdr_eval import evaluate_cdr
    collator, _ = batch()
    short = antibody()
    long = replace(short, chains=[BioSeqChain("A"*70, "antigen"), *short.chains[1:]])
    samples = [{"record": x, "source_group": "test"} for x in (short, long)]
    both = fixed_batch(collator, samples, [0,1], recipe(), "joint", .5, False,
                       mask_scope="cdr_multi")
    for i in (0,1):
        one = fixed_batch(collator, samples, [i], recipe(), "joint", .5, False, mask_scope="cdr_multi")
        for name in ("input_ids", "labels", "loss_weights", "active_population"):
            assert torch.equal(both[name][i, :one[name].shape[1]], one[name][0])
    model = DiffusionEditModel(config(), recipe()).train()
    metrics = evaluate_cdr(model, samples, torch.device("cpu"), batch_size=2)
    assert model.training
    assert metrics["eligible_records"] == 2
    assert metrics["modes"]["H3"]["tokens"] == 8
    assert metrics["modes"]["all_six"]["tokens"] == 32
    assert all(value is not None for value in metrics["summary"].values())
