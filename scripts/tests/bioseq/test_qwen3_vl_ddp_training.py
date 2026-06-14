from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")


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
            "--max-chain-length",
            "64",
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
