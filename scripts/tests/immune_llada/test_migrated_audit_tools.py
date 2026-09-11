"""Small-fixture tests for the migrated immune audit and smoke tools.

Run with::

    .venv_abnativ/bin/python -m pytest scripts/tests/immune_llada/test_migrated_audit_tools.py

These tests use temporary prepared JSONL and never scan project data or require a
GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer
from dllm.pipelines.immune_llada.data.preprocessing.validators import SCHEMA_VERSION
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from scripts.data.check_immune_adapter_parity import legacy_adapters
from scripts.data.dedup import audit_downstream_leakage as leakage
from scripts.data.run_immune_full_parity import BASELINE_COMMIT, load_legacy
from scripts.data.tcr_native import smoke_test


ROOT = Path(__file__).resolve().parents[3]


def _prepared(tmp_path: Path) -> Path:
    output = tmp_path / "prepared"
    split = output / "train"
    split.mkdir(parents=True)
    records = [
        BioSeqRecord(
            [
                BioSeqChain("PEPTIDE", "peptide"),
                BioSeqChain("CASSQF", "tcr_beta"),
            ],
            "tcr_epitope",
            "tcr_native",
            split="train",
            identifiers={"cdr3b_core": "CASSQF", "pair_key": "CASSQF|PEPTIDE"},
        ),
        BioSeqRecord(
            [BioSeqChain("ASSQQ", "tcr_beta")],
            "tcr",
            "tcr_repertoire",
            split="train",
            identifiers={"cdr3b_core": "ASSQQ", "benchmark_key": "ASSQQ"},
        ),
    ]
    shard = split / "tcr_native-00000.jsonl"
    shard.write_text(
        "\n".join(json.dumps({"schema_version": SCHEMA_VERSION, **record.to_dict()}) for record in records[:1])
        + "\n",
        encoding="utf-8",
    )
    repertoire_shard = split / "tcr_repertoire-00000.jsonl"
    repertoire_shard.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, **records[1].to_dict()}) + "\n",
        encoding="utf-8",
    )
    (output / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "format": "jsonl",
                "splits": {
                    "train": {
                        "shards": [
                            {"path": "train/tcr_native-00000.jsonl", "source": "tcr_native", "records": 1},
                            {"path": "train/tcr_repertoire-00000.jsonl", "source": "tcr_repertoire", "records": 1},
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return output


def test_parity_uses_the_pinned_historical_adapter(tmp_path: Path) -> None:
    snapshot = tmp_path / "legacy-snapshot"
    legacy = load_legacy(snapshot, BASELINE_COMMIT)
    adapter = legacy_adapters(legacy)["tcr_repertoire"]
    assert adapter({"cdr3b": "CASSQF"}) == {
        "chains": ["CASSQF"],
        "roles": ["tcr_beta"],
        "task_type": "tcr",
        "source": "tcr_repertoire",
    }
    assert legacy.hashes["dllm/pipelines/bioseq/datasets.py"]


def test_prepared_leakage_reader_is_bounded_and_uses_canonical_identifier(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    records = list(leakage._prepared_rows(prepared, "train", "tcr_native", 1))
    assert len(records) == 1
    assert leakage._prepared_cdr3b(records[0]) == "CASSQF"
    assert leakage.core(leakage._prepared_cdr3b(records[0]), has_anchors=False) == "CASSQF"


def test_smoke_builds_real_grammar_batch_from_prepared_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared = _prepared(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["smoke_test.py", "--prepared-data-dir", str(prepared), "--dataset-args", "tcr_native", "--max-rows", "1"],
    )
    smoke_test.main()
    output = capsys.readouterr().out
    assert "grammar_batch=" in output
    assert "encoder=" in output
    assert "SMOKE_OK" in output


def test_migrated_scripts_do_not_import_deleted_loader_or_trainer_builder() -> None:
    paths = [
        ROOT / "scripts/data/check_immune_adapter_parity.py",
        ROOT / "scripts/data/dedup/audit_downstream_leakage.py",
        ROOT / "scripts/data/tcr_native/smoke_test.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "dllm.pipelines.bioseq.datasets" not in source
        assert "build_immune_specs" not in source


def test_grammar_batch_fixture_has_decoder_and_encoder_masks(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    from dllm.pipelines.immune_llada.data import load_prepared_dataset

    record = load_prepared_dataset(prepared, source="tcr_native")[0]
    batch = GrammarBioSeqCollator(GrammarTokenizer())([record])
    assert batch["input_ids"].shape[0] == 1
    assert batch["encoder_input_ids"].ndim == 3
    assert batch["encoder_chain_mask"].any()
    assert batch["diffusion_eligible_mask"].any()
