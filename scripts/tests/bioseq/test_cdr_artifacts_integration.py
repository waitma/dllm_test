"""CPU adapter tests with stub predictions. Run: python -m pytest scripts/tests/bioseq/test_cdr_artifacts_integration.py."""

import json
from types import SimpleNamespace

import pytest
import torch

from downstream.grammar import cdr_infill as adapter
from downstream.grammar.cdr_result_io import score_artifacts


@pytest.fixture
def setup_adapter(tmp_path, monkeypatch):
    dataset = tmp_path / "data"
    for fold, count in [(0, 2), (1, 1)]:
        path = dataset / f"fold_{fold}" / "test.json"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(json.dumps({"heavy_chain_seq": "toy", "light_chain_seq": "toy",
                                              "cdrh3_seq": "toy", "cdrh3_pos": [1, 2]}) for _ in range(count)))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"synthetic checkpoint identity; no weights loaded")
    tok = SimpleNamespace(vocab_size=4, token=lambda i: ["p", "a", "b", "c"][i], mask_token_id=0, pad_token_id=0)
    monkeypatch.setattr(adapter, "_load_eval_model", lambda args, device: (object(), tok))
    monkeypatch.setattr(adapter, "build_eval_collator", lambda model, tok: lambda rows: {
        "labels": torch.tensor([[0, 1, 2, 0]]), "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "residue_mask": torch.tensor([[False, True, True, False]])})
    monkeypatch.setattr(adapter, "build_antibody_record", lambda heavy, light: object())
    monkeypatch.setattr(adapter, "_build_cdr_partial_mask", lambda *args: torch.tensor([[True, False, False, True]]))
    monkeypatch.setattr(adapter, "labels_to_grammar", lambda model, labels: labels)
    calls = []

    def predict(model, batch, **kwargs):
        calls.append((kwargs["max_iter"], kwargs["sampling_strategy"], kwargs["temperature"]))
        return torch.tensor([[0, 1, 3, 0]]), None

    monkeypatch.setattr(adapter, "run_grammar_generate", predict)
    args = SimpleNamespace(test_set=str(dataset), mode="cdrh3", checkpoint_path=str(checkpoint), num_folds=2,
                           max_samples=None, device="cpu", max_iter=2, sampling_strategy="argmax", temperature=1.0,
                           output_dir=None)
    return args, calls


def test_artifacts_preserve_legacy_adapter_scores_and_generation_arguments(tmp_path, setup_adapter, capsys):
    args, calls = setup_adapter
    adapter.evaluate(args)
    legacy = capsys.readouterr().out
    original_calls = list(calls)
    calls.clear()
    args.output_dir = tmp_path / "result"
    adapter.evaluate(args)
    formal = capsys.readouterr().out
    assert all(line in formal for line in legacy.splitlines())
    assert calls == original_calls == [(2, "argmax", 1.0)] * 3
    metrics = score_artifacts(args.output_dir)
    assert metrics["status"] == "complete" and metrics["average_aar_all_folds"] == 50


def test_adapter_max_samples_retains_full_expected_count(tmp_path, setup_adapter):
    args, _ = setup_adapter
    args.output_dir, args.max_samples = tmp_path / "result", 1
    adapter.evaluate(args)
    result = score_artifacts(args.output_dir, allow_partial=True)
    assert (result["status"], result["n_expected"], result["n_sequences"]) == ("partial", 3, 2)


def test_adapter_zero_target_does_not_emit_zero_score(tmp_path, setup_adapter, monkeypatch):
    args, calls = setup_adapter
    args.output_dir = tmp_path / "result"
    monkeypatch.setattr(adapter, "_build_cdr_partial_mask", lambda *args: torch.ones(1, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="Empty CDR target"):
        adapter.evaluate(args)
    assert not calls and not (args.output_dir / "metrics.json").exists()
