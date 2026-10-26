"""CPU regressions for restart-safe retention and worker-to-Trainer drop logs.

Run: python -m pytest /project_c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_training_retention_and_drop_logs.py -q
"""

import json
from collections import Counter
from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader

from downstream.grammar.common import antibody_pair_record
from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer
from examples.llada.protein_pretrain_esmc import (
    CanvasDropLogCallback, FusionTrainer, TopKValLossCheckpointCallback,
)


def checkpoint(root, step, score):
    path = root / f"checkpoint-{step}"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"test-weights")
    (path / "optimizer.bin").write_bytes(b"test-optimizer")
    (path / "trainer_state.json").write_text(json.dumps({
        "log_history": [{"step": step, "eval_loss": score}] if score is not None else []
    }))
    return path


def saved(callback, root, step, score):
    state = SimpleNamespace(is_world_process_zero=True, global_step=step,
                            log_history=[{"step": step, "eval_loss": score}])
    callback.on_save(SimpleNamespace(output_dir=str(root)), state, None)


def test_restart_keeps_historical_best_and_slims_previous_latest(tmp_path):
    first = TopKValLossCheckpointCallback(save_top_k=1)
    best = checkpoint(tmp_path, 10, 0.1)
    saved(first, tmp_path, 10, 0.1)
    old_latest = checkpoint(tmp_path, 20, 0.8)
    saved(first, tmp_path, 20, 0.8)
    assert not (best / "trainer_state.json").exists()  # Requires manifest recovery.
    resumed = TopKValLossCheckpointCallback(save_top_k=1)
    newest = checkpoint(tmp_path, 30, 0.6)
    saved(resumed, tmp_path, 30, 0.6)
    manifest = json.loads((tmp_path / "topk_val_manifest.json").read_text())
    assert [entry["step"] for entry in manifest["checkpoints"]] == [10, 30]
    assert best.is_dir() and newest.is_dir() and not old_latest.exists()
    assert (newest / "optimizer.bin").exists()
    assert not (best / "optimizer.bin").exists()
    assert not list(tmp_path.glob(".topk_val_*.tmp"))


def test_restart_slims_previous_latest_when_still_in_best_k(tmp_path):
    first = TopKValLossCheckpointCallback(save_top_k=3)
    old = checkpoint(tmp_path, 10, 0.2)
    saved(first, tmp_path, 10, 0.2)
    checkpoint(tmp_path, 20, 0.4)
    saved(TopKValLossCheckpointCallback(save_top_k=3), tmp_path, 20, 0.4)
    assert old.is_dir() and not (old / "optimizer.bin").exists()


def test_missing_manifest_recovers_exact_step_history_and_protects_unknown(tmp_path):
    best = checkpoint(tmp_path, 10, 0.1)
    unknown = checkpoint(tmp_path, 15, None)
    checkpoint(tmp_path, 20, 1.0)
    saved(TopKValLossCheckpointCallback(save_top_k=1), tmp_path, 20, 1.0)
    assert best.exists() and unknown.exists()
    entries = json.loads((tmp_path / "topk_val_manifest.json").read_text())["checkpoints"]
    assert next(e for e in entries if e["step"] == 15)["val_loss"] is None


@pytest.mark.parametrize("kind", ["invalid_json", "external_path", "metric_changed", "symlink"])
def test_unsafe_restore_fails_before_deletion(tmp_path, kind):
    root = tmp_path / "run"
    root.mkdir()
    outside = checkpoint(tmp_path, 999, 0.1)
    local = checkpoint(root, 10, 0.8)
    payload = {"metric": "eval_loss", "checkpoints": [{"step": 10, "path": str(local), "val_loss": .8}]}
    if kind == "external_path":
        payload["checkpoints"][0]["path"] = str(outside)
    if kind == "metric_changed":
        payload["metric"] = "another_metric"
    if kind == "symlink":
        (root / "checkpoint-999").symlink_to(outside, target_is_directory=True)
    (root / "topk_val_manifest.json").write_text("{" if kind == "invalid_json" else json.dumps(payload))
    with pytest.raises(ValueError):
        saved(TopKValLossCheckpointCallback(save_top_k=1), root, 10, 0.8)
    assert (local / "optimizer.bin").exists() and (outside / "model.safetensors").exists()


def test_no_stale_loss_reused_for_new_checkpoint(tmp_path):
    checkpoint(tmp_path, 20, None)
    callback = TopKValLossCheckpointCallback(save_top_k=1)
    callback.on_save(SimpleNamespace(output_dir=str(tmp_path)), SimpleNamespace(
        is_world_process_zero=True, global_step=20, log_history=[{"step": 10, "eval_loss": .1}]
    ), None)
    assert json.loads((tmp_path / "topk_val_manifest.json").read_text())["checkpoints"][0]["val_loss"] is None


@pytest.mark.parametrize("workers", [0, 2])
def test_drop_deltas_reach_parent_once_and_do_not_mix_evaluation(workers):
    collator = GrammarBioSeqCollator(GrammarTokenizer(), fixed_receptor_lengths=True, drop_overflow_records=True)
    bad = antibody_pair_record("A" * 167, "C" * 30)
    good = antibody_pair_record("A" * 90, "C" * 30)
    batches = list(DataLoader([bad, good, bad, good], batch_size=2, num_workers=workers, collate_fn=collator))
    trainer = object.__new__(FusionTrainer)
    trainer.canvas_drop_counts = Counter()
    trainer.canvas_drop_counts_by_source = {}
    for batch in batches:
        assert batch["_canvas_drop_counts"] == {"canvas.receptor_overflow": 1}
        trainer._consume_canvas_drops(batch, training=True)
        trainer._consume_canvas_drops(batch, training=True)  # Already consumed, cannot double-count.
    evaluation = collator([bad, good])
    trainer._consume_canvas_drops(evaluation, training=False)
    logs = {"loss": 1.0}
    CanvasDropLogCallback(trainer).on_log(None, None, None, logs=logs)
    assert logs["canvas_drop/total"] == 2
    assert logs["canvas_drop/canvas.receptor_overflow"] == 2
    assert sum(value for key, value in logs.items() if key.startswith("canvas_drop_by_source/")) == 2
    if workers:
        # Parent only saw the explicit evaluation call, never the training drops.
        assert sum(collator.drop_counts.values()) == 1
