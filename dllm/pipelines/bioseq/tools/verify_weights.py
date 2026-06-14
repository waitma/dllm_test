from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from dllm.pipelines.bioseq.config import (
    DEFAULT_MODEL_WEIGHTS_ROOT,
    ESM2_REPOS,
    ESM2_SMOKE_TEST_REPOS,
    ESMC_REPOS,
    OPHIUCHUS_AB_CHECKPOINT,
    OPHIUCHUS_AB_CHECKPOINT_SIZE,
    repo_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify BioSeq local model weights.")
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=DEFAULT_MODEL_WEIGHTS_ROOT,
        help="Absolute model weight root. Default: /c20250601/mj/model_weights",
    )
    parser.add_argument("--smoke-esm2", action="store_true", help="Load ESM2 150M/650M/3B.")
    parser.add_argument("--smoke-esmc", action="store_true", help="Load ESMC-300M.")
    return parser.parse_args()


def assert_repo_files(root: Path, group: str, repo_id: str) -> Path:
    target = root / group / repo_name(repo_id)
    if not target.exists():
        raise FileNotFoundError(f"Missing directory: {target}")
    if not (target / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {target}")
    model_files = list(target.glob("*.safetensors")) + list(target.glob("*.bin"))
    shard_files = list(target.glob("*.safetensors.index.json")) + list(target.glob("*.bin.index.json"))
    if not model_files and not shard_files:
        raise FileNotFoundError(f"Missing model weight files in {target}")
    return target


def assert_ophiuchus_ab_checkpoint(root: Path) -> Path:
    checkpoint = root / "ophiuchus_ab" / "Ophiuchus-Ab" / OPHIUCHUS_AB_CHECKPOINT
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing Ophiuchus-Ab checkpoint: {checkpoint}")
    actual_size = checkpoint.stat().st_size
    if actual_size != OPHIUCHUS_AB_CHECKPOINT_SIZE:
        raise ValueError(
            f"Unexpected Ophiuchus-Ab checkpoint size: "
            f"{actual_size} != {OPHIUCHUS_AB_CHECKPOINT_SIZE}"
        )
    return checkpoint


def smoke_load(path: Path, trust_remote_code: bool = False) -> None:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"Smoke loading {path}")
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust_remote_code)
    model = AutoModelForMaskedLM.from_pretrained(
        str(path),
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float32,
    )
    model.eval()
    encoded = tokenizer("ACDEFGHIKLMNPQRSTVWY", return_tensors="pt")
    with torch.no_grad():
        model(**encoded)
    del model
    gc.collect()


def main() -> None:
    args = parse_args()
    root = args.weights_root.expanduser().resolve()
    for repo_id in ESMC_REPOS:
        assert_repo_files(root, "esmc", repo_id)
    for repo_id in ESM2_REPOS:
        assert_repo_files(root, "esm2", repo_id)
    assert_ophiuchus_ab_checkpoint(root)

    if args.smoke_esm2:
        for repo_id in ESM2_SMOKE_TEST_REPOS:
            smoke_load(root / "esm2" / repo_name(repo_id))
    if args.smoke_esmc:
        smoke_load(root / "esmc" / "ESMC-300M", trust_remote_code=True)


if __name__ == "__main__":
    main()
