"""Verify MLM semantics, epoch coverage, global normalization and checkpoint restore.

Run after activating pllm: python -B -m pytest THIS_FILE -q
"""
from dataclasses import replace
import torch
import pytest

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.mlm import (
    BertMLMCollator, BertMLMModel, RECIPE, corrupt, selected_loss)
from scripts.research.bert_mlm_data import CompleteEpochBatchSampler, step_budget
from scripts.research.bert_mlm_eval import fixed_corruption


def config():
    return FixedChainConfig(d_model=32, n_layers=2, n_heads=4, mlp_hidden_size=64)


def record(relation="binding"):
    return BioSeqRecord([BioSeqChain("ACDEFG", "peptide"),
                        BioSeqChain("ACDEFGHIKLMNPQRSTVWY", "tcr_beta")],
                       "tcr_epitope", "test", labels={"relation": relation})


def test_observed_receptor_only_and_label_blind_inputs():
    collator = BertMLMCollator(config())
    original = record()
    observed = len(original.chains[1].sequence) - 5
    template = replace(original.chains[1], metadata={"synthetic_residue_mask": [1]*5+[0]*observed})
    batch = collator([replace(original, chains=[original.chains[0], template])])
    assert batch["eligible_mask"].sum() == observed
    assert not (batch["eligible_mask"] & batch["terminal_mask"]).any()
    assert not (batch["eligible_mask"] & batch["role_ids"].eq(6)).any()
    batches = [collator([record(label)]) for label in ("binding", "nonbinding", "unknown")]
    for key in BertMLMModel.feature_args(batches[0]):
        assert all(torch.equal(batches[0][key], b[key]) for b in batches[1:])


def test_noise_statistics_original_targets_and_no_noise_flag_features():
    collator = BertMLMCollator(config())
    clean = collator([record()]*1024)
    noisy = corrupt(clean, collator.tokenizer, torch.Generator().manual_seed(24))
    chosen = noisy["labels"].ne(-100)
    assert torch.equal(noisy["labels"][chosen], clean["input_ids"][chosen])
    assert abs(float(chosen.sum()/clean["eligible_mask"].sum())-.15) < .01
    for kind, expected in ((1, .8), (2, .1), (3, .1)):
        mask = noisy["noise_kind"].eq(kind)
        assert abs(float(mask.sum()/chosen.sum())-expected) < .025
    random_other = noisy["noise_kind"].eq(2)
    assert (noisy["input_ids"][random_other] != clean["input_ids"][random_other]).all()
    assert torch.equal(noisy["input_ids"][~chosen], clean["input_ids"][~chosen])
    assert "noise_kind" not in BertMLMModel.feature_args(noisy)
    assert "labels" not in BertMLMModel.feature_args(noisy)
    for key in ("attention_mask", "role_ids", "scope_ids", "chain_ids", "position_ids", "target_slot_mask"):
        assert torch.equal(noisy[key], clean[key])


def test_fixed_validation_is_independent_of_batch_padding_and_rng():
    collator = BertMLMCollator(config())
    records = [record(), replace(record(), chains=[BioSeqChain("A"*30, "peptide"), record().chains[1]])]
    state = torch.random.get_rng_state()
    together = fixed_corruption(collator(records), collator.tokenizer, [91, 92])
    for i in range(2):
        alone = fixed_corruption(collator([records[i]]), collator.tokenizer, [91+i])
        length = alone["input_ids"].shape[1]
        for key in ("input_ids", "labels", "noise_kind"):
            assert torch.equal(together[key][i, :length], alone[key][0])
    assert torch.equal(state, torch.random.get_rng_state())


def test_complete_epoch_tail_and_resume_have_no_repetition():
    sizes, world, batch = (133, 512, 1141, 1508), 8, 512
    for size in sizes:
        ranks = [list(CompleteEpochBatchSampler(size, batch, r, world, 42, 0)) for r in range(world)]
        flattened = [i for rank in ranks for step in rank for i in step]
        assert sorted(flattened) == list(range(size))
        assert len(flattened) == len(set(flattened))
        resumed = list(CompleteEpochBatchSampler(size, batch, 2, world, 42, 0, 1))
        assert resumed == ranks[2][1:]
    assert not torch.equal(CompleteEpochBatchSampler(1141, batch, 0, world, 42, 0).order,
                           CompleteEpochBatchSampler(1141, batch, 0, world, 42, 1).order)


def test_exact_steps_include_epoch_tails_and_stop_partway_through_last_epoch():
    # Three steps/epoch, then stop after two full epochs and one partial batch.
    size, global_batch, world, target = 1141, 512, 8, 7
    steps, records, epoch = 0, 0, 0
    while steps < target:
        ranks = [list(CompleteEpochBatchSampler(size, global_batch, r, world, 42, epoch,
                                               max_batches=target-steps)) for r in range(world)]
        for batches in zip(*ranks):
            merged = [record for batch in batches for record in batch]
            assert len(merged) == len(set(merged))
            records += len(merged)
            steps += 1
        epoch += 1
    assert (steps, records, epoch) == (7, 1141*2+512, 3)
    budget = step_budget(size, global_batch, target)
    assert budget["seen_records"] == records
    assert (budget["completed_epochs"], budget["next_batch"]) == (2, 1)
    # A restored cursor consumes only the remainder of the requested budget.
    resumed = CompleteEpochBatchSampler(size, 512, 0, 8, 42, 2, start_batch=1, max_batches=1)
    full = list(CompleteEpochBatchSampler(size, 512, 0, 8, 42, 2))
    assert list(resumed) == full[1:2]
    assert list(CompleteEpochBatchSampler(size, 512, 0, 8, 42, 2, max_batches=0)) == []
    actual = step_budget(6249956, 512, 300000)
    assert (actual["steps_per_epoch"], actual["completed_epochs"], actual["next_batch"],
            actual["seen_records"], actual["tail_records"]) == (12207, 24, 7032, 153599328, 484)


def test_v7_profile_uses_bound_chain_capacities():
    from scripts.research.profile_bert_mlm import synthetic_batch
    short = replace(config(), first_chain_capacity=167, second_chain_capacity=135, full_capacity=304)
    batch, tokenizer = synthetic_batch(short, 2, 2048)
    assert batch["input_ids"].shape == (2, 2048)
    assert batch["eligible_mask"].sum(1).tolist() == [300, 300]
    assert batch["terminal_mask"].sum(1).tolist() == [2, 2]


def test_v7_full_scan_rejects_negative_and_overflow_rows(tmp_path):
    import json
    from scripts.research.bert_mlm_data import scan_shard
    path = tmp_path / "rows.jsonl"
    entry = dict(path=path.name, records=1, source="test", split="train", forbid_nonbinding=True,
                 receptor_capacities={"tcr_beta": 21})
    path.write_text(json.dumps(record("nonbinding").to_dict())+"\n")
    with pytest.raises(ValueError, match="nonbinding"):
        scan_shard((str(tmp_path), entry, str(tmp_path), set()))
    path.write_text(json.dumps(record().to_dict())+"\n")
    bad = dict(entry, receptor_capacities={"tcr_beta": 20})
    with pytest.raises(ValueError, match="capacity"):
        scan_shard((str(tmp_path), bad, str(tmp_path), set()))
    result = scan_shard((str(tmp_path), entry, str(tmp_path), set()))
    assert result["records"] == 1 and not result["excluded"]


def test_global_token_mean_matches_uneven_ddp_and_accumulation():
    torch.manual_seed(4)
    weight = torch.randn(5, 7, requires_grad=True)
    inputs = torch.randn(17, 5)
    labels = torch.arange(17) % 7
    reference = torch.nn.functional.cross_entropy(inputs @ weight, labels)
    expected, = torch.autograd.grad(reference, weight)
    # Two ranks have unequal target counts; each has differently sized microbatches.
    gradients = []
    for chunks in ((slice(0, 1), slice(1, 9)), (slice(9, 15), slice(15, 17))):
        copy = weight.detach().clone().requires_grad_()
        for chunk in chunks:
            loss_sum = torch.nn.functional.cross_entropy(inputs[chunk] @ copy, labels[chunk], reduction="sum")
            (loss_sum * 2/len(labels)).backward()
        gradients.append(copy.grad)
    torch.testing.assert_close(sum(gradients)/2, expected)


def test_checkpoint_roundtrip_and_empty_loss(tmp_path):
    torch.manual_seed(42)
    model = BertMLMModel(config()).eval()
    collator = BertMLMCollator(model.config)
    batch = corrupt(collator([record()]), model.tokenizer, torch.Generator().manual_seed(4))
    out = model(**batch)
    assert torch.isfinite(out["loss_sum"])
    empty = selected_loss(out["logits"], torch.full_like(batch["labels"], -100))
    assert empty.item() == 0
    empty.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.save_pretrained(tmp_path)
    restored = BertMLMModel.from_pretrained(tmp_path).eval()
    assert torch.equal(model(**batch)["logits"], restored(**batch)["logits"])
    assert not restored.allowed_target[restored.tokenizer.eos_token_id]


def test_optimizer_scheduler_and_corruption_rng_resume(tmp_path):
    from scripts.research.run_bert_mlm import capture_rng, restore_rng, make_scheduler
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = make_scheduler(optimizer, 20, 2)
    noise = torch.Generator().manual_seed(400)
    def step():
        optimizer.zero_grad()
        data = torch.rand((4, 3), generator=noise)
        model(data).square().mean().backward()
        optimizer.step()
        scheduler.step()
    step()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    rng=capture_rng(noise)), tmp_path / "state.pt")
    step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    saved = torch.load(tmp_path / "state.pt", weights_only=False)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    restore_rng(saved["rng"], noise)
    step()
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_optimizer_replay_diagnostic_detects_hidden_moment_drift():
    from copy import deepcopy
    from scripts.research.run_bert_mlm import optimizer_update_difference
    original = {0: dict(step=torch.tensor(2.), exp_avg=torch.tensor([.001, .002]),
                       exp_avg_sq=torch.tensor([1e-8, 2e-8]))}
    assert all(v["relative_l2"] == 0 for v in optimizer_update_difference(original, original).values())
    changed = deepcopy(original)
    changed[0]["exp_avg"] *= 1.01
    result = optimizer_update_difference(original, changed)
    assert result["exp_avg"]["relative_l2"] > .009
    assert result["exp_avg_sq"]["relative_l2"] == 0
    changed[0]["step"] += 1
    with pytest.raises(ValueError, match="Adam step"):
        optimizer_update_difference(original, changed)
