from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from dllm.pipelines.bioseq import Esm2ProteinTokenizer, MultiChainDynamicCollator
from dllm.pipelines.bioseq.datasets import (
    ImmuneCsvDataset,
    ImmuneSourceSpec,
    nanobody_row_to_record,
    oas_paired_row_to_record,
    ots_paired_row_to_record,
)


def test_dynamic_collator_variable_length_paired():
    tokenizer = Esm2ProteinTokenizer()
    collator = MultiChainDynamicCollator(tokenizer=tokenizer)
    batch = collator(
        [
            {"chains": ["QVQLVQSGAE", "DIQMTQSPSS"], "task_type": "antibody"},
            {"chains": ["QVQLVQSGAEVKKPGAS", "DIQMTQ"], "task_type": "antibody"},
        ]
    )
    heavy = batch["heavy_tokens"]["targets"]
    light = batch["light_tokens"]["targets"]
    # chain1 max = len("QVQLVQSGAEVKKPGAS") + 2 special = 19
    assert heavy.shape == (2, 19)
    # chain2 max = len("DIQMTQSPSS") + 2 = 12
    assert light.shape == (2, 12)
    # shorter sequences are pad-filled with the real <pad> id, not <eos>.
    assert (heavy[0, -1] == tokenizer.pad_token_id).item()
    assert batch["heavy_tokens"]["chain_ids"].unique().tolist() == [0]
    assert batch["light_tokens"]["chain_ids"].unique().tolist() == [1]
    assert batch["weights"].shape == (2, 1)


def test_dynamic_collator_single_chain_nanobody():
    tokenizer = Esm2ProteinTokenizer()
    collator = MultiChainDynamicCollator(tokenizer=tokenizer)
    batch = collator([{"chains": ["QVQLVESGGG"], "task_type": "antibody"}])
    heavy = batch["heavy_tokens"]["targets"]
    light = batch["light_tokens"]["targets"]
    assert heavy.shape == (1, 12)
    # empty chain-2 placeholder is exactly [<cls>, <eos>].
    assert light.shape == (1, 2)
    assert light[0, 0].item() == tokenizer.cls_token_id
    assert light[0, 1].item() == tokenizer.eos_token_id


def test_dynamic_collator_max_length_cap():
    tokenizer = Esm2ProteinTokenizer()
    collator = MultiChainDynamicCollator(tokenizer=tokenizer, max_length=8)
    batch = collator([{"chains": ["A" * 50, "C" * 50], "task_type": "antibody"}])
    assert batch["heavy_tokens"]["targets"].shape == (1, 8)
    assert batch["light_tokens"]["targets"].shape == (1, 8)


def test_immune_csv_dataset_from_synthetic_files(tmp_path):
    oas = tmp_path / "train.csv"
    oas.write_text(
        "cleaned_chain1_seq,cleaned_chain2_seq,chain1_anarci_type,chain2_anarci_type\n"
        "DIQMTQSPSS,QVQLVQSGAE,L,H\n"  # light first; should be reordered heavy-first
    )
    record = ImmuneCsvDataset(
        ImmuneSourceSpec("oas", oas, oas_paired_row_to_record, "antibody"), split="train"
    )[0]
    assert record["chains"][0] == "QVQLVQSGAE"  # heavy oriented to slot 0
    assert record["task_type"] == "antibody"

    ots = tmp_path / "ots.csv"
    ots.write_text(
        "cleaned_chain1_seq,cleaned_chain2_seq,chain1_anarci_type,chain2_anarci_type\n"
        "AAAAAA,CCCCCC,A,B\n"  # alpha first; should be reordered beta-first
    )
    ots_record = ImmuneCsvDataset(
        ImmuneSourceSpec("ots", ots, ots_paired_row_to_record, "tcr"), split="ots"
    )[0]
    assert ots_record["chains"][0] == "CCCCCC"  # beta oriented to slot 0
    assert ots_record["task_type"] == "tcr"

    nano = tmp_path / "nano.csv"
    nano.write_text("vhh_seq,cleaned_seq\nXXXX,QVQLVESGGG\n")
    nano_record = ImmuneCsvDataset(
        ImmuneSourceSpec("nanobody", nano, nanobody_row_to_record, "antibody"), split="nano"
    )[0]
    assert nano_record["chains"] == ["QVQLVESGGG"]
    assert len(nano_record["chains"]) == 1


def test_invalid_rows_are_skipped(tmp_path):
    nano = tmp_path / "n.csv"
    nano.write_text("cleaned_seq\nINVALID1\nQVQLVESGGG\n")  # first row has a digit -> invalid
    dataset = ImmuneCsvDataset(
        ImmuneSourceSpec("nanobody", nano, nanobody_row_to_record, "antibody"), split="n"
    )
    assert len(dataset) == 1
    assert dataset[0]["chains"] == ["QVQLVESGGG"]
