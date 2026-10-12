"""Full-coverage sampling, exact source weights and real CPU Trainer tests.

Run: python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_source_balanced_training.py -q
"""

from bisect import bisect_right
from collections import Counter
from contextlib import nullcontext
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler, ConcatDataset, Dataset
from transformers import TrainerCallback

from dllm.pipelines.immune_llada.data.balanced_sampling import FullCoverageSourceSampler
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import compute_masked_cross_entropy
from examples.llada.protein_pretrain_esmc import FusionTrainer, TrainingArguments


@pytest.mark.parametrize("lengths,batch", [([2, 5, 11], 4), ([1, 2, 13, 25, 40, 9, 103], 256)])
def test_full_coverage_no_truncation_and_no_replacement_before_exhaustion(lengths, batch):
    sampler = FullCoverageSourceSampler(lengths, effective_batch_size=batch, seed=19)
    stream = list(sampler)
    assert len(stream) == len(sampler) and len(stream) % batch == 0
    assert set(stream) == set(range(sum(lengths)))
    ends = list(accumulate(lengths))
    for start in range(0, len(stream), batch):
        counts = Counter(bisect_right(ends, index) for index in stream[start:start + batch])
        assert len(counts) == len(lengths)
        assert max(counts.values()) - min(counts.values()) <= 1
    for source, size in enumerate(lengths):
        subsequence = [index for index in stream if bisect_right(ends, index) == source]
        for start in range(0, len(subsequence), size):
            block = subsequence[start:start + size]
            assert len(set(block)) == len(block)
    assert list(sampler) == stream
    sampler.set_epoch(1)
    assert list(sampler) != stream
    replica = FullCoverageSourceSampler(lengths, effective_batch_size=batch, seed=19)
    replica.set_epoch(1)
    assert list(replica) == list(sampler)


def test_accelerate_eight_rank_sharding_preserves_balanced_effective_batches():
    lengths = [13, 21, 50, 1, 7, 9, 103]
    ends = list(accumulate(lengths))
    shards = []
    for rank in range(8):
        sampler = FullCoverageSourceSampler(lengths, effective_batch_size=256)
        shards.append(list(BatchSamplerShard(BatchSampler(sampler, 4, drop_last=False),
                                            num_processes=8, process_index=rank, split_batches=False)))
    observed = set()
    for offset in range(0, len(shards[0]), 8):
        optimizer_batch = [index for shard in shards for batch in shard[offset:offset + 8] for index in batch]
        counts = Counter(bisect_right(ends, index) for index in optimizer_batch)
        assert set(counts.values()) <= {36, 37}
        assert len(optimizer_batch) == 256
        observed.update(optimizer_batch)
    assert observed == set(range(sum(lengths)))


def test_source_passes_continue_without_reset_at_epoch_boundaries():
    lengths = [3, 7, 11]
    ends = list(accumulate(lengths))
    sampler = FullCoverageSourceSampler(lengths, effective_batch_size=8)
    stream = []
    for epoch in range(4):
        sampler.set_epoch(epoch)
        stream.extend(sampler)
    for source, length in enumerate(lengths):
        indices = [index for index in stream if bisect_right(ends, index) == source]
        for start in range(0, len(indices), length):
            chunk = indices[start:start + length]
            assert len(set(chunk)) == len(chunk)


@pytest.mark.parametrize("lengths,batch", [([], 4), ([0, 2], 4), ([1, 2, 3], 2)])
def test_invalid_or_too_small_batch_fails(lengths, batch):
    with pytest.raises(ValueError):
        FullCoverageSourceSampler(lengths, effective_batch_size=batch)


def test_weighted_ce_and_gradient_equal_source_macro_mean_with_unequal_lengths():
    torch.manual_seed(7)
    logits = torch.randn(5, 7, 4, requires_grad=True)
    labels = torch.randint(0, 4, (5, 7))
    labels[0, 1:] = -100
    labels[1, 3:] = -100
    sources = [0, 0, 0, 1, 2]
    weights = torch.tensor([1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 3])
    weighted = compute_masked_cross_entropy(logits, labels, sequence_weights=weights)
    per_row = torch.stack([compute_masked_cross_entropy(logits[i:i + 1], labels[i:i + 1]) for i in range(5)])
    reference = sum(per_row[[i for i, s in enumerate(sources) if s == source]].mean() for source in range(3)) / 3
    torch.testing.assert_close(weighted, reference)
    torch.testing.assert_close(torch.autograd.grad(weighted, logits, retain_graph=True)[0],
                               torch.autograd.grad(reference, logits)[0])


def test_global_ddp_and_accumulation_scaling_matches_reference():
    torch.manual_seed(2)
    logits = torch.randn(8, 3, 4, requires_grad=True)
    labels = torch.randint(0, 4, (8, 3))
    sources = [0, 1, 2, 0, 1, 2, 0, 1]
    counts = Counter(sources)
    # 2 ranks x 2 accumulation microbatches x 2 samples.
    scaled = sum(compute_masked_cross_entropy(
        logits[start:start + 2], labels[start:start + 2],
        sequence_weights=torch.tensor([4 / (3 * counts[s]) for s in sources[start:start + 2]])
    ) / 4 for start in range(0, 8, 2))
    rows = torch.stack([compute_masked_cross_entropy(logits[i:i + 1], labels[i:i + 1]) for i in range(8)])
    reference = sum(rows[[i for i, s in enumerate(sources) if s == source]].mean() for source in range(3)) / 3
    torch.testing.assert_close(scaled, reference)
    torch.testing.assert_close(torch.autograd.grad(scaled, logits, retain_graph=True)[0],
                               torch.autograd.grad(reference, logits)[0])


class SourceDataset(Dataset):
    def __init__(self, source, length):
        self.source, self.length = source, length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"value": self.source + 1., "source": str(self.source), "index": index}


def toy_collate(records):
    return {"input_ids": torch.tensor([[r["value"]] for r in records]),
            "sources": [r["source"] for r in records],
            "source_groups": [r["source"] for r in records],
            "indices": [r["index"] for r in records]}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.observed = []

    def forward(self, input_ids, source_loss_weights=None, sources=None, indices=None, **kwargs):
        self.observed.extend(zip(sources, indices))
        losses = (self.weight * input_ids[:, 0]).square()
        loss = (losses * source_loss_weights).sum() if source_loss_weights is not None else losses.mean()
        return SimpleNamespace(loss=loss)


class CaptureGrad(TrainerCallback):
    def __init__(self):
        self.gradients = []

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        self.gradients.append(float(model.weight.grad))


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("accumulation", [1, 8])
def test_accumulation_normalization_for_both_sampling_modes(tmp_path, balanced, accumulation):
    """One update must match the full-batch mean, including logs and SGD step."""
    model = ToyModel()
    capture = CaptureGrad()
    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to="none", disable_tqdm=True,
        source_balanced_training=balanced, per_device_train_batch_size=16 // accumulation,
        gradient_accumulation_steps=accumulation, max_steps=1, max_grad_norm=0,
        save_strategy="no", logging_steps=1,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = FusionTrainer(
        model=model, args=args, data_collator=toy_collate,
        train_dataset=SourceDataset(0, 16), callbacks=[capture],
        optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)),
    )
    trainer.train_source_names = ["0"]
    trainer.train_source_lengths = [16]
    result = trainer.train()
    # All inputs equal 1: mean(w*x)^2 = 1, gradient = 2, SGD w = 0.8.
    assert capture.gradients == pytest.approx([2.0])
    assert result.training_loss == pytest.approx(1.0)
    assert model.weight.item() == pytest.approx(0.8)


def test_real_cpu_trainer_accumulation_and_full_coverage(tmp_path):
    model = ToyModel()
    capture = CaptureGrad()
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", disable_tqdm=True,
                             source_balanced_training=True, per_device_train_batch_size=2,
                             gradient_accumulation_steps=2, max_steps=4, max_grad_norm=0,
                             save_strategy="no", logging_steps=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    trainer = FusionTrainer(model=model, args=args, data_collator=toy_collate,
                            train_dataset=ConcatDataset([SourceDataset(i, n) for i, n in enumerate([2, 4, 5])]),
                            callbacks=[capture], optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)))
    trainer.train_source_names = ["0", "1", "2"]
    trainer.train_source_lengths = [2, 4, 5]
    trainer.train()
    assert capture.gradients == pytest.approx([28 / 3] * 4)
    assert set(model.observed) == {(str(i), j) for i, n in enumerate([2, 4, 5]) for j in range(n)}


def test_missing_source_after_drops_fails_instead_of_silent_reweighting(monkeypatch):
    trainer = object.__new__(FusionTrainer)
    trainer.args = SimpleNamespace(source_balanced_training=True, world_size=1, gradient_accumulation_steps=1)
    trainer.train_source_names = ["a", "b"]
    trainer.canvas_drop_counts = Counter()
    trainer.canvas_drop_counts_by_source = {}
    trainer.accelerator = SimpleNamespace(device="cpu", reduce=lambda values, reduction: values)
    monkeypatch.setattr("transformers.Trainer.get_batch_samples", lambda *args, **kwargs: (
        [{"input_ids": torch.zeros(2, 1), "sources": ["a", "a"], "source_groups": ["a", "a"]}], None))
    with pytest.raises(ValueError, match="missing sources"):
        trainer.get_batch_samples(iter([]), 1)


class StopAfterTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_save = True
            control.should_training_stop = True


def make_resume_trainer(root, *, stop=False):
    model = ToyModel()
    args = TrainingArguments(output_dir=str(root), use_cpu=True, report_to="none", disable_tqdm=True,
                             source_balanced_training=True, per_device_train_batch_size=2,
                             gradient_accumulation_steps=2, max_steps=4, max_grad_norm=0,
                             save_strategy="steps", save_steps=2, logging_steps=1,
                             dataloader_num_workers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    trainer = FusionTrainer(model=model, args=args, data_collator=toy_collate,
                            train_dataset=ConcatDataset([SourceDataset(i, n) for i, n in enumerate([2, 4, 5])]),
                            callbacks=[StopAfterTwo()] if stop else [],
                            optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)))
    trainer.train_source_names = ["0", "1", "2"]
    trainer.train_source_lengths = [2, 4, 5]
    return trainer


def test_real_resume_preserves_exact_stream_with_workers(tmp_path):
    uninterrupted = make_resume_trainer(tmp_path / "reference")
    uninterrupted.train()
    first = make_resume_trainer(tmp_path / "interrupted", stop=True)
    first.train()
    resumed = make_resume_trainer(tmp_path / "interrupted")
    resumed.train(resume_from_checkpoint=str(tmp_path / "interrupted" / "checkpoint-2"))
    assert first.model.observed + resumed.model.observed == uninterrupted.model.observed
    assert resumed.state.global_step == 4


def test_resume_rejects_changed_source_policy_before_loading_weights(tmp_path):
    import json
    trainer = make_resume_trainer(tmp_path)
    path = tmp_path / "checkpoint-2"
    path.mkdir()
    policy = trainer._source_sampling_policy()
    policy["source_lengths"] = [2, 4, 6]
    (path / "source_sampling.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="sampling policy changed"):
        trainer.train(resume_from_checkpoint=str(path))
    (path / "source_sampling.json").unlink()
    with pytest.raises(ValueError, match="sampling policy changed"):
        trainer.train(resume_from_checkpoint=str(path))


def test_budget_smaller_than_full_coverage_is_rejected(tmp_path):
    trainer = make_resume_trainer(tmp_path)
    trainer.args.max_steps = 3
    with pytest.raises(ValueError, match="cannot cover every source"):
        trainer._get_train_sampler()


def _ddp_source_weight_worker(rank, rendezvous):
    """A real two-rank CPU backward, not just simulated gradient averaging."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model = DistributedDataParallel(ToyModel())
        # Global round-robin stream, two examples/rank/microbatch.
        global_sources = [0, 1, 2, 0, 1, 2, 0, 1]
        local = [global_sources[rank * 2:rank * 2 + 2], global_sources[4 + rank * 2:6 + rank * 2]]
        counts = torch.tensor([sum(source == i for batch in local for source in batch) for i in range(3)])
        dist.all_reduce(counts)
        for step, sources in enumerate(local):
            weights = torch.tensor([4 / (3 * int(counts[source])) for source in sources])
            with model.no_sync() if step == 0 else nullcontext():
                result = model(torch.tensor([[source + 1.] for source in sources]), source_loss_weights=weights,
                               sources=[str(s) for s in sources], indices=[0, 0])
                (result.loss / 2).backward()
        torch.testing.assert_close(model.module.weight.grad, torch.tensor(28 / 3))
    finally:
        dist.destroy_process_group()


def test_real_two_rank_ddp_source_gradient(tmp_path):
    torch.multiprocessing.spawn(_ddp_source_weight_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)
