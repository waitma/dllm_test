"""Synthetic observation/rescoring checks. Run: python -m pytest scripts/tests/bioseq/test_cdr_result_io.py."""

import copy
import json

import pytest

from downstream.grammar.cdr_result_io import CDRResultWriter, score_artifacts


def identity():
    return {"checkpoint": {"weights": {"sha256": "toy-checkpoint"}},
            "tokenizer": {"tokens": ["a", "b", "c"]}, "code": [{"sha256": "toy-code"}],
            "parameters": {"mode": "synthetic"},
            "datasets": [{"fold": 0, "n_rows": 2, "sha256": "source-0"},
                         {"fold": 1, "n_rows": 1, "sha256": "source-1"}]}


def append(writer, fold=0, row=0, prediction=None):
    prediction = prediction if prediction is not None else [0, 2]
    writer.append(fold=fold, row=row, reference_ids=[0, 1], prediction_ids=prediction,
                  masked_positions=[3, 4], aar=sum(a == b for a, b in zip([0, 1], prediction)) / 2)


def test_rescore_keeps_sequence_and_fold_macro_distinct(tmp_path):
    writer = CDRResultWriter(tmp_path / "run", identity())
    append(writer)
    append(writer, row=1, prediction=[0, 1])
    append(writer, fold=1, prediction=[2, 2])
    result = writer.finish()
    assert result == score_artifacts(writer.directory)
    assert result["average_aar"] == 50
    assert result["average_aar_all_folds"] == 37.5
    assert result["fold_sd_ddof0"] == 37.5
    assert result["status"] == "complete"


def test_missing_or_truncated_rows_fail_unless_explicitly_partial(tmp_path):
    writer = CDRResultWriter(tmp_path / "run", identity())
    append(writer)
    with pytest.raises(ValueError, match="missing"):
        writer.finish()
    result = writer.finish(allow_partial=True)
    assert result["status"] == "partial" and result["n_missing"] == 2
    with pytest.raises(ValueError, match="missing"):
        score_artifacts(writer.directory)


def test_empty_predictions_fail_even_in_partial_mode(tmp_path):
    writer = CDRResultWriter(tmp_path / "run", identity())
    with pytest.raises(ValueError, match="empty"):
        writer.finish(allow_partial=True)


def test_duplicate_rows_fail_at_write_and_rescore(tmp_path):
    writer = CDRResultWriter(tmp_path / "run", identity())
    append(writer)
    with pytest.raises(ValueError, match="duplicate"):
        append(writer)
    path = writer.directory / "predictions.jsonl"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        score_artifacts(writer.directory, allow_partial=True)


@pytest.mark.parametrize("field,value,match", [
    ("identity_sha256", "other-checkpoint", "identity"),
    ("mode", "other-mode", "identity"),
    ("source_sha256", "other-data", "identity"),
    ("correct", 2, "counts"),
    ("target_length", 0, "counts"),
    ("prediction_tokens", ["c", "c"], "text/ID"),
    ("aar", 1.0, "AAR"),
    ("masked_positions", [3, 3], "positions"),
])
def test_corrupt_rows_fail(tmp_path, field, value, match):
    writer = CDRResultWriter(tmp_path / "run", identity())
    append(writer)
    path = writer.directory / "predictions.jsonl"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=match):
        score_artifacts(writer.directory, allow_partial=True)


def test_manifest_parameter_mismatch_fails(tmp_path):
    writer = CDRResultWriter(tmp_path / "run", identity())
    append(writer)
    path = writer.directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["parameters"]["mode"] = "another-mode"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest identity"):
        score_artifacts(writer.directory, allow_partial=True)


def test_sealed_file_checksum_and_directory_are_immutable(tmp_path):
    writer = CDRResultWriter(tmp_path / "run", identity())
    append(writer)
    writer.finish(allow_partial=True)
    with pytest.raises(ValueError, match="sealed"):
        append(writer, row=1)
    with pytest.raises(FileExistsError):
        CDRResultWriter(writer.directory, identity())
    path = writer.directory / "predictions.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        score_artifacts(writer.directory, allow_partial=True)


def test_empty_fold_and_empty_target_fail(tmp_path):
    bad = copy.deepcopy(identity())
    bad["datasets"][0]["n_rows"] = 0
    with pytest.raises(ValueError, match="empty"):
        CDRResultWriter(tmp_path / "bad", bad)
    writer = CDRResultWriter(tmp_path / "run", identity())
    with pytest.raises(ValueError, match="empty"):
        writer.append(fold=0, row=0, reference_ids=[], prediction_ids=[], masked_positions=[], aar=0)
