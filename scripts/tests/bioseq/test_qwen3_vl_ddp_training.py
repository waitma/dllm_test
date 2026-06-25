from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import examples.bioseq.train_qwen3_vl_bioseq_ddp as train_ddp
from examples.bioseq.train_qwen3_vl_bioseq_ddp import (
    build_loader,
    build_tokenizer,
    build_validation_loader,
    evaluate_validation,
    lr_at,
    parse_args,
    should_find_unused_parameters,
)


PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")


def test_polynomial_lr_warms_up_from_init_lr_then_cosine_decays() -> None:
    assert math.isclose(
        lr_at(0, 1e-4, 2000, max_steps=50_000, scheduler="polynomial", warmup_init_lr=1e-7),
        1e-7,
    )
    assert math.isclose(
        lr_at(1999, 1e-4, 2000, max_steps=50_000, scheduler="polynomial", warmup_init_lr=1e-7),
        1e-4,
        rel_tol=1e-3,
    )
    assert math.isclose(
        lr_at(2000, 1e-4, 2000, max_steps=50_000, scheduler="polynomial", warmup_init_lr=1e-7),
        1e-4,
    )
    assert 1e-5 < lr_at(25_000, 1e-4, 2000, max_steps=50_000, scheduler="polynomial", warmup_init_lr=1e-7) < 1e-4
    assert math.isclose(
        lr_at(49_999, 1e-4, 2000, max_steps=50_000, scheduler="polynomial", warmup_init_lr=1e-7),
        1e-5,
        rel_tol=1e-3,
    )


def test_cosine_lr_warms_up_then_decays() -> None:
    assert math.isclose(lr_at(0, 1e-4, 1000, max_steps=10_000, scheduler="cosine"), 1e-7)
    assert math.isclose(lr_at(999, 1e-4, 1000, max_steps=10_000, scheduler="cosine"), 1e-4)
    assert math.isclose(lr_at(1000, 1e-4, 1000, max_steps=10_000, scheduler="cosine"), 1e-4)
    assert 1e-5 < lr_at(5000, 1e-4, 1000, max_steps=10_000, scheduler="cosine") < 1e-4
    assert math.isclose(lr_at(10_000, 1e-4, 1000, max_steps=10_000, scheduler="cosine"), 1e-5)


def test_encoder_mode_enables_ddp_find_unused_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_qwen3_vl_bioseq_ddp.py", "--model-type", "encoder"])
    args = parse_args()
    assert should_find_unused_parameters(args)


def test_esm2_mode_enables_ddp_find_unused_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_qwen3_vl_bioseq_ddp.py", "--model-type", "esm2"])
    args = parse_args()
    assert should_find_unused_parameters(args)


def test_no_encoder_mode_keeps_ddp_find_unused_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_qwen3_vl_bioseq_ddp.py", "--model-type", "no_encoder"])
    args = parse_args()
    assert not should_find_unused_parameters(args)


def test_num_workers_defaults_to_zero_safe_ddp_path(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_qwen3_vl_bioseq_ddp.py", "--sources", "oas"])
    args = parse_args()
    assert args.num_workers == 0


def test_save_top_k_defaults_to_ten(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_qwen3_vl_bioseq_ddp.py", "--sources", "oas"])
    args = parse_args()
    assert args.save_top_k == 10


def test_training_loader_emits_grammar_batches_with_diffusion_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_qwen3_vl_bioseq_ddp.py",
            "--sources",
            "oas",
            "--limit-per-source",
            "8",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
        ],
    )
    args = parse_args()
    tokenizer = build_tokenizer(args)
    batch = next(iter(build_loader(args, tokenizer)))
    assert set(batch["view_names"]) == {"grammar_v2"}
    assert int(batch["diffusion_loss_mask"].sum().item()) > 0


def test_validation_loader_reads_valid_split(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_qwen3_vl_bioseq_ddp.py",
            "--sources",
            "oas",
            "--limit-per-source",
            "8",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--val-split",
            "valid",
            "--val-batches",
            "1",
        ],
    )
    args = parse_args()
    tokenizer = build_tokenizer(args)
    loader = build_validation_loader(args, tokenizer)
    assert loader is not None
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 2
    assert set(batch["sources"]) == {"oas_paired"}
    assert set(batch["view_names"]) == {"grammar_v2"}
    assert int(batch["diffusion_loss_mask"].sum().item()) > 0


def test_validation_metrics_include_flat_wandb_aliases(monkeypatch) -> None:
    class DummyModel:
        def eval(self) -> None:
            pass

        def train(self) -> None:
            pass

    def fake_compute_training_output(train_model, module, batch):
        return SimpleNamespace(
            loss=torch.tensor(2.0),
            corruption_mask=torch.tensor([[True, False, True]]),
        )

    monkeypatch.setattr(train_ddp, "compute_training_output", fake_compute_training_output)
    monkeypatch.setattr(train_ddp, "loss_logging_denominator", lambda output, batch, loss_norm: torch.tensor(4.0))
    monkeypatch.setattr(train_ddp, "diffusion_eligible_token_count", lambda batch: torch.tensor(8.0))

    args = argparse.Namespace(val_batches=2, bf16=False)
    module = SimpleNamespace(config=SimpleNamespace(loss_norm="token"))
    metrics = evaluate_validation(
        train_model=DummyModel(),
        module=module,
        val_iter=iter([{}, {}]),
        args=args,
        device=torch.device("cpu"),
        distributed=False,
    )

    assert metrics["val/loss"] == metrics["val_loss"] == 2.0
    assert metrics["val/corrupted_tokens"] == metrics["val_corrupted_tokens"] == 2.0
    assert metrics["val/eligible_tokens"] == metrics["val_eligible_tokens"] == 8.0
    assert metrics["val/corruption_rate"] == metrics["val_corruption_rate"] == 0.25
    assert metrics["val/batches"] == metrics["val_batches"] == 2.0


def test_qwen3_vl_bioseq_ddp_script_single_process_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "qwen3_vl_bioseq_smoke"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "examples/bioseq/train_qwen3_vl_bioseq_ddp.py"),
            "--model-type",
            "no_encoder",
            "--sources",
            "oas",
            "--limit-per-source",
            "8",
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--max-sequence-length",
            "256",
            "--hidden-size",
            "32",
            "--num-hidden-layers",
            "1",
            "--num-attention-heads",
            "4",
            "--intermediate-size",
            "64",
            "--dropout",
            "0.0",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--save-interval",
            "0",
            "--resume",
            "none",
            "--wandb-mode",
            "disabled",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert (output_dir / "final.pt").is_file()
