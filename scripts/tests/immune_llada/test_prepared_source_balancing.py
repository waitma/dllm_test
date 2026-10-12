"""Prepared JSONL -> grammar/remap -> source-balanced CPU Trainer regressions.

Run: python -B -m pytest scripts/tests/immune_llada/test_prepared_source_balancing.py -q
"""

import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import TrainerCallback

from dllm.pipelines.immune_llada.data import (
    BioSeqChain, BioSeqRecord, GrammarBioSeqCollator, GrammarTokenizer, load_prepared_dataset,
)
from dllm.pipelines.immune_llada.data.preprocessing.validators import SCHEMA_VERSION
from examples.llada.protein_fusion_model import RemapCollator
from examples.llada.protein_pretrain_esmc import FusionTrainer, TrainingArguments


SOURCES = ("oas", "ots", "asd_antibody", "trait", "tcr_native", "tcr_papers", "tcr_repertoire")


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    shards, expected = [], []
    for index, group in enumerate(SOURCES):
        # Provenance need not be unique across groups or match any group name.
        provenance = {"oas": "oas_paired", "ots": "ots_paired", "tcr_papers": "tcr_native"}.get(group, group)
        records = [BioSeqRecord(
            [BioSeqChain("ACD" * (index + 1), "antibody_heavy"), BioSeqChain("FGH", "antibody_light")],
            "antibody", provenance, "train", metadata={"original_group": group},
        ) for _ in range(index % 3 + 1)]
        if group == "oas":
            records.append(BioSeqRecord(
                [BioSeqChain("KLM", "antibody_heavy")], "antibody", "oas_other_provenance", "train",
            ))
        # An unrelated filename ensures grouping comes from manifest metadata.
        path = root / f"part-{index}.jsonl"
        path.write_text("".join(json.dumps(dict(row.to_dict(), schema_version=SCHEMA_VERSION)) + "\n"
                                for row in records))
        shards.append({"path": path.name, "source": group, "records": len(records)})
        expected.extend((group, row.to_dict()) for row in records)
    manifest = {"schema_version": SCHEMA_VERSION, "format": "jsonl", "sources": list(SOURCES),
                "splits": {"train": {"records": len(expected), "shards": shards}}}
    (root / "dataset_manifest.json").write_text(json.dumps(manifest))
    return root, expected


def collator():
    base = GrammarBioSeqCollator(GrammarTokenizer(), max_sequence_length=1024,
                                fixed_receptor_lengths=True, drop_overflow_records=True)
    # Exercise the real remapper, including passthrough of source metadata.
    return RemapCollator(base, torch.arange(4096) + 5)


class PreparedLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.observed = []

    def forward(self, input_ids, attention_mask, sources, source_groups, source_loss_weights=None, **kwargs):
        values = (input_ids.float() * attention_mask).sum(1) / attention_mask.sum(1)
        self.observed.append((list(source_groups), list(sources), values.detach().clone(),
                              source_loss_weights.detach().clone() if source_loss_weights is not None else None))
        losses = (self.weight * values).square()
        loss = (losses * source_loss_weights).sum() if source_loss_weights is not None else losses.mean()
        return SimpleNamespace(loss=loss)


class CaptureGradient(TrainerCallback):
    def __init__(self):
        self.values = []

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        self.values.append(model.weight.grad.detach().clone())


def make_trainer(root, *, dataset=None, workers=0, balanced=True):
    model = PreparedLossModel()
    capture = CaptureGradient()
    args = TrainingArguments(
        output_dir=str(root), use_cpu=True, report_to="none", disable_tqdm=True,
        source_balanced_training=balanced, per_device_train_batch_size=4,
        gradient_accumulation_steps=8, max_steps=1, max_grad_norm=0,
        save_strategy="no", logging_strategy="no", dataloader_num_workers=workers,
        remove_unused_columns=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    trainer = FusionTrainer(
        model=model, args=args, train_dataset=dataset, data_collator=collator(), callbacks=[capture],
        optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)),
    )
    trainer.train_source_names = list(SOURCES)
    if dataset is not None:
        trainer.train_source_lengths = [len(part) for part in dataset.datasets]
    return trainer, capture


@pytest.mark.parametrize("workers", [0, 2])
def test_prepared_manifest_groups_drive_real_trainer_gradient(prepared, tmp_path, workers):
    root, expected = prepared
    parts = [load_prepared_dataset(root, source=source) for source in SOURCES]
    trainer, capture = make_trainer(tmp_path / "run", dataset=ConcatDataset(parts), workers=workers)
    trainer.train()
    assert trainer.state.global_step == 1
    observed = trainer.model.observed
    groups = [group for batch in observed for group in batch[0]]
    provenance = [source for batch in observed for source in batch[1]]
    values = torch.cat([batch[2] for batch in observed])
    weights = torch.cat([batch[3] for batch in observed]) / 8  # Trainer accumulation scale.
    assert len(groups) == 32
    assert set(Counter(groups).values()) == {4, 5}
    allowed = {(group, row["source"]) for group, row in expected}
    assert set(zip(groups, provenance)) <= allowed
    assert "oas_paired" in provenance and "oas_other_provenance" in provenance
    assert "ots_paired" in provenance
    assert ("tcr_native", "tcr_native") in set(zip(groups, provenance))
    assert ("tcr_papers", "tcr_native") in set(zip(groups, provenance))
    reference = torch.tensor(0.)
    for group in SOURCES:
        mask = torch.tensor([value == group for value in groups])
        torch.testing.assert_close(weights[mask].sum(), torch.tensor(1 / 7))
        reference += 2 * values[mask].square().mean() / 7
    torch.testing.assert_close(capture.values[0], reference)


def test_prepared_group_is_runtime_only_and_survives_spawn(prepared):
    root, expected = prepared
    before = {path: path.read_bytes() for path in root.iterdir()}
    dataset = load_prepared_dataset(root)  # Mixed-source loader, no source filter.
    records = list(dataset.iter_records())
    assert [record.source_group for record in records] == [group for group, _ in expected]
    assert [record.to_dict() for record in records] == [row for _, row in expected]
    batches = list(DataLoader(dataset, batch_size=3, num_workers=2, multiprocessing_context="spawn",
                              collate_fn=collator()))
    assert [group for batch in batches for group in batch["source_groups"]] == [g for g, _ in expected]
    assert [source for batch in batches for source in batch["sources"]] == [r["source"] for _, r in expected]
    assert {path: path.read_bytes() for path in root.iterdir()} == before


def test_prepared_groups_stay_aligned_and_reweighted_after_overflow(prepared, tmp_path):
    root, expected = prepared
    dataset = load_prepared_dataset(root)
    records = list(dataset.iter_records())
    # Add an overflowing row to the fixture shard; production prepared data is never touched.
    path = root / "part-0.jsonl"
    overflow = BioSeqRecord([BioSeqChain("A" * 167, "antibody_heavy")], "antibody", "dropped_provenance")
    with path.open("a") as handle:
        handle.write(json.dumps(dict(overflow.to_dict(), schema_version=SCHEMA_VERSION)) + "\n")
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["train"]["shards"][0]["records"] += 1
    manifest["splits"]["train"]["records"] += 1
    manifest_path.write_text(json.dumps(manifest))
    bad = load_prepared_dataset(root, source="oas")[-1]
    batch = collator()([records[0], bad, *records[1:]])
    trainer, _ = make_trainer(tmp_path / "run")
    batches, _ = trainer.get_batch_samples(iter([batch]), 1)
    result = batches[0]
    groups = [group for group, _ in expected]
    assert result["source_groups"] == groups
    assert result["sources"] == [row["source"] for _, row in expected]
    counts = Counter(groups)
    torch.testing.assert_close(result["source_loss_weights"],
                               torch.tensor([8 / (7 * counts[group]) for group in groups]))
    assert trainer.canvas_drop_counts_by_source == {"canvas.receptor_overflow": Counter(dropped_provenance=1)}


@pytest.mark.parametrize("invalid", ["missing", "unknown", "misaligned", "unassigned"])
def test_balancing_rejects_missing_or_invalid_manifest_groups(prepared, tmp_path, invalid):
    root, _ = prepared
    batch = collator()(list(load_prepared_dataset(root).iter_records()))
    if invalid == "missing":
        batch.pop("source_groups", None)
    elif invalid == "unknown":
        batch["source_groups"][0] = "not_in_manifest"
    elif invalid == "unassigned":
        batch["source_groups"][0] = None
    else:
        batch["source_groups"].pop()
    trainer, _ = make_trainer(tmp_path / "run")
    with pytest.raises(ValueError, match="source.group"):
        trainer.get_batch_samples(iter([batch]), 1)


@pytest.mark.parametrize("invalid", [None, ""])
def test_prepared_loader_rejects_missing_manifest_source_group(prepared, invalid):
    root, _ = prepared
    path = root / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["train"]["shards"][0]["source"] = invalid
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest source group"):
        load_prepared_dataset(root)


def test_unbalanced_path_preserves_provenance_without_reweighting(prepared, tmp_path):
    root, expected = prepared
    batch = collator()(list(load_prepared_dataset(root).iter_records()))
    trainer, _ = make_trainer(tmp_path / "run", balanced=False)
    batches, _ = trainer.get_batch_samples(iter([batch]), 1)
    assert batches[0]["sources"] == [row["source"] for _, row in expected]
    assert "source_loss_weights" not in batches[0]
