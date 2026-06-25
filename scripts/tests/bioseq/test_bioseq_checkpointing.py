"""Unit tests for val-loss top-k checkpoint retention."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    BioSeqDiffusionTransformerConfig,
    BioSeqNoEncoderDiffusionModel,
)
from dllm.pipelines.qwen3_vl_arch.training.checkpointing import (
    BEST_CHECKPOINT_NAME,
    ValLossTopKCheckpointManager,
    adapt_model_state_dict_for_resume,
    checkpoint_filename,
    load_resume_payload,
)


def _save_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_checkpoint_filename_is_stable() -> None:
    assert checkpoint_filename(1000, 3.8421) == "step_0001000_val_3.8421.pt"


def test_topk_manager_keeps_lowest_k_and_refreshes_best(tmp_path: Path) -> None:
    manager = ValLossTopKCheckpointManager(tmp_path, save_top_k=3)

    losses = [5.0, 4.0, 3.0, 6.0, 2.5, 2.0, 7.0]
    saved_steps = []
    for step, val_loss in enumerate(losses, start=1):
        saved = manager.maybe_save(
            step,
            val_loss,
            {"step": step, "val_loss": val_loss},
            save_payload=_save_payload,
        )
        if saved is not None:
            saved_steps.append(step)

    assert saved_steps == [1, 2, 3, 5, 6]
    assert len(manager.entries) == 3
    assert [entry.step for entry in manager.entries] == [6, 5, 3]
    assert [round(entry.val_loss, 1) for entry in manager.entries] == [2.0, 2.5, 3.0]

    manifest = json.loads((tmp_path / "checkpoints" / "topk_manifest.json").read_text(encoding="utf-8"))
    assert manifest["save_top_k"] == 3
    assert len(manifest["checkpoints"]) == 3

    best_payload = torch.load(tmp_path / BEST_CHECKPOINT_NAME, map_location="cpu")
    assert best_payload["step"] == 6
    assert best_payload["val_loss"] == 2.0

    removed = tmp_path / "checkpoints" / checkpoint_filename(2, 4.0)
    assert not removed.is_file()
    kept = tmp_path / "checkpoints" / checkpoint_filename(5, 2.5)
    assert kept.is_file()


def test_topk_manager_disabled_when_k_is_zero(tmp_path: Path) -> None:
    manager = ValLossTopKCheckpointManager(tmp_path, save_top_k=0)
    assert not manager.enabled
    assert manager.maybe_save(1, 1.0, {"step": 1}, save_payload=_save_payload) is None


def test_adapt_model_state_dict_expands_vocab_rows() -> None:
    old_config = BioSeqDiffusionTransformerConfig(vocab_size=51, hidden_size=16, num_hidden_layers=1)
    new_config = BioSeqDiffusionTransformerConfig(vocab_size=53, hidden_size=16, num_hidden_layers=1)
    old_model = BioSeqNoEncoderDiffusionModel(old_config)
    new_model = BioSeqNoEncoderDiffusionModel(new_config)

    old_state = old_model.state_dict()
    adapted, vocab_expanded = adapt_model_state_dict_for_resume(new_model, old_state)
    assert vocab_expanded is True
    assert adapted["decoder.token_embeddings.weight"].shape == (53, 16)
    assert torch.allclose(
        adapted["decoder.token_embeddings.weight"][:51],
        old_state["decoder.token_embeddings.weight"],
    )
    assert torch.allclose(
        adapted["decoder.lm_head.weight"][:51],
        old_state["decoder.lm_head.weight"],
    )

    missing, unexpected = new_model.load_state_dict(adapted, strict=False)
    assert not missing
    assert not unexpected


def test_load_resume_payload_skips_optimizer_when_vocab_grows() -> None:
    old_config = BioSeqDiffusionTransformerConfig(vocab_size=51, hidden_size=16, num_hidden_layers=1)
    new_config = BioSeqDiffusionTransformerConfig(vocab_size=53, hidden_size=16, num_hidden_layers=1)
    old_model = BioSeqNoEncoderDiffusionModel(old_config)
    new_model = BioSeqNoEncoderDiffusionModel(new_config)
    optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
    old_optimizer = torch.optim.AdamW(old_model.parameters(), lr=1e-4)
    payload = {
        "model_state_dict": old_model.state_dict(),
        "optimizer_state_dict": old_optimizer.state_dict(),
        "step": 14500,
    }

    step, vocab_expanded = load_resume_payload(new_model, optimizer, payload)
    assert step == 14500
    assert vocab_expanded is True
    assert not optimizer.state
