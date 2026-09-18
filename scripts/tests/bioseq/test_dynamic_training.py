from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.bioseq import Esm2ProteinTokenizer, MultiChainDynamicCollator
from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from dllm.pipelines.immune_llada.data.sources import (
    nanobody_row_to_record,
    oas_row_to_record,
    ots_row_to_record,
)


def test_dynamic_collator_variable_length_paired():
    tokenizer = Esm2ProteinTokenizer()
    collator = MultiChainDynamicCollator(tokenizer=tokenizer)
    batch: dict[str, Any] = collator(
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
    batch: dict[str, Any] = collator([{"chains": ["QVQLVESGGG"], "task_type": "antibody"}])
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
    batch: dict[str, Any] = collator([{"chains": ["A" * 50, "C" * 50], "task_type": "antibody"}])
    assert batch["heavy_tokens"]["targets"].shape == (1, 8)
    assert batch["light_tokens"]["targets"].shape == (1, 8)


def test_source_adapters_from_synthetic_rows():
    oas_record = oas_row_to_record(
        {
            "cleaned_chain1_seq": "DIQMTQSPSS",
            "cleaned_chain2_seq": "QVQLVQSGAE",
            "chain1_anarci_type": "L",
            "chain2_anarci_type": "H",
        },
        split="train",
    )
    assert isinstance(oas_record, BioSeqRecord)
    assert oas_record.sequences == ["QVQLVQSGAE", "DIQMTQSPSS"]
    assert oas_record.task_type == "antibody"

    ots_record = ots_row_to_record(
        {
            "cleaned_chain1_seq": "AAAAAA",
            "cleaned_chain2_seq": "CCCCCC",
            "chain1_anarci_type": "A",
            "chain2_anarci_type": "B",
        }
    )
    assert isinstance(ots_record, BioSeqRecord)
    assert ots_record.sequences == ["CCCCCC", "AAAAAA"]
    assert ots_record.task_type == "tcr"

    nano_record = nanobody_row_to_record({"vhh_seq": "QVQLVESGGG"})
    assert isinstance(nano_record, BioSeqRecord)
    assert nano_record.sequences == ["QVQLVESGGG"]
    assert nano_record.chain_roles == ["nanobody_vhh"]


def test_source_adapters_skip_invalid_rows():
    assert nanobody_row_to_record({"cleaned_seq": "INVALID1"}) is None
    assert nanobody_row_to_record({"cleaned_seq": "QVQLVESGGG"}) is not None
