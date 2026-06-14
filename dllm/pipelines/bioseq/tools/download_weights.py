from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import time
from urllib.parse import quote
from pathlib import Path

from dllm.pipelines.bioseq.config import (
    DEFAULT_MODEL_WEIGHTS_ROOT,
    DEFAULT_PROJECT_ROOT,
    ESM2_REPOS,
    ESMC_REPOS,
    local_weight_dir,
)

ALLOW_PATTERNS = (
    ".gitattributes",
    "*.bin",
    "*.bin.index.json",
    "*.json",
    "*.md",
    "*.model",
    "*.py",
    "*.safetensors",
    "*.safetensors.index.json",
    "*.txt",
)
IGNORE_PATTERNS = (
    "*.h5",
    "*.msgpack",
)

KNOWN_REPO_FILES = {
    "biohub/ESMC-300M": (
        ".gitattributes",
        "README.md",
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    "biohub/ESMC-600M": (
        ".gitattributes",
        "README.md",
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    "biohub/ESMC-6B": (
        ".gitattributes",
        "README.md",
        "config.json",
        "model-00001-of-00006.safetensors",
        "model-00002-of-00006.safetensors",
        "model-00003-of-00006.safetensors",
        "model-00004-of-00006.safetensors",
        "model-00005-of-00006.safetensors",
        "model-00006-of-00006.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    "facebook/esm2_t6_8M_UR50D": (
        ".gitattributes",
        "README.md",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    ),
    "facebook/esm2_t12_35M_UR50D": (
        ".gitattributes",
        "README.md",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    ),
    "facebook/esm2_t30_150M_UR50D": (
        ".gitattributes",
        "README.md",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    ),
    "facebook/esm2_t33_650M_UR50D": (
        ".gitattributes",
        "README.md",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    ),
    "facebook/esm2_t36_3B_UR50D": (
        ".gitattributes",
        "README.md",
        "config.json",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
        "pytorch_model.bin.index.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    ),
    "facebook/esm2_t48_15B_UR50D": (
        ".gitattributes",
        "README.md",
        "config.json",
        "pytorch_model-00001-of-00007.bin",
        "pytorch_model-00002-of-00007.bin",
        "pytorch_model-00003-of-00007.bin",
        "pytorch_model-00004-of-00007.bin",
        "pytorch_model-00005-of-00007.bin",
        "pytorch_model-00006-of-00007.bin",
        "pytorch_model-00007-of-00007.bin",
        "pytorch_model.bin.index.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BioSeq ESMC/ESM2 weights.")
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=DEFAULT_MODEL_WEIGHTS_ROOT,
        help="Absolute model weight root. Default: /c20250601/mj/model_weights",
    )
    parser.add_argument(
        "--group",
        choices=("all", "esmc", "esm2"),
        default="all",
        help="Weight group to download.",
    )
    parser.add_argument(
        "--repo-id",
        action="append",
        default=None,
        help="Optional specific Hugging Face repo id. Can be passed multiple times.",
    )
    return parser.parse_args()


def download_repo(repo_id: str, group: str, weights_root: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download model weights") from exc

    os.environ["BIOSEQ_MODEL_WEIGHTS_ROOT"] = str(weights_root)
    os.environ.setdefault("HF_HOME", str(weights_root / ".hf_cache"))
    target = local_weight_dir(group, repo_id)
    target.mkdir(parents=True, exist_ok=True)
    filenames = KNOWN_REPO_FILES.get(repo_id)
    if filenames is None:
        filenames = [
            filename
            for filename in list_repo_files(repo_id)
            if should_download(filename)
        ]
    for filename in filenames:
        destination = target / filename
        if destination.exists() and destination.stat().st_size > 0:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if is_weight_file(filename):
            direct_download(repo_id=repo_id, filename=filename, destination=destination)
            continue
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target),
                resume_download=True,
            )
        except OSError:
            direct_download(repo_id=repo_id, filename=filename, destination=destination)


def should_download(filename: str) -> bool:
    if any(fnmatch.fnmatch(filename, pattern) for pattern in IGNORE_PATTERNS):
        return False
    return any(fnmatch.fnmatch(filename, pattern) for pattern in ALLOW_PATTERNS)


def is_weight_file(filename: str) -> bool:
    return filename.endswith((".bin", ".safetensors"))


def direct_download(repo_id: str, filename: str, destination: Path) -> None:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{quote(filename, safe='/')}"
    staging = (
        DEFAULT_PROJECT_ROOT.parent
        / ".download_tmp"
        / "bioseq_weights"
        / repo_id.replace("/", "__")
        / filename
    )
    partial = staging.with_suffix(staging.suffix + ".part")
    legacy_partial = destination.with_suffix(destination.suffix + ".part")
    if legacy_partial.exists():
        legacy_partial.unlink()
    partial.parent.mkdir(parents=True, exist_ok=True)
    remote_size = get_remote_size(url)
    for attempt in range(1, 21):
        if remote_size is not None and partial.exists() and partial.stat().st_size > remote_size:
            partial.unlink()
        if remote_size is not None and partial.exists() and partial.stat().st_size == remote_size:
            break

        command = [
            "curl",
            "-L",
            "--silent",
            "--show-error",
            "--fail",
            "--connect-timeout",
            "60",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ]
        result = subprocess.run(command, check=False)
        if remote_size is None:
            if result.returncode == 0 and partial.exists() and partial.stat().st_size > 0:
                break
        elif partial.exists() and partial.stat().st_size == remote_size:
            break
        if attempt == 20:
            raise RuntimeError(f"Failed to download {repo_id}:{filename}")
        time.sleep(5)

    if remote_size is not None and partial.stat().st_size != remote_size:
        raise RuntimeError(
            f"Incomplete download for {repo_id}:{filename}: "
            f"{partial.stat().st_size} != {remote_size}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(partial, destination)
    partial.unlink()


def get_remote_size(url: str) -> int | None:
    command = [
        "curl",
        "-L",
        "-I",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "60",
        url,
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        return None
    sizes: list[int] = []
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.lower() in {"content-length", "x-linked-size"}:
            value = value.strip()
            if value.isdigit():
                sizes.append(int(value))
    return max(sizes) if sizes else None


def main() -> None:
    args = parse_args()
    weights_root = args.weights_root.expanduser().resolve()
    if not weights_root.is_absolute():
        raise ValueError("weights_root must be an absolute path")
    weights_root.mkdir(parents=True, exist_ok=True)

    if args.repo_id:
        repos = tuple(args.repo_id)
        for repo_id in repos:
            if "ESMC" in repo_id:
                download_repo(repo_id, "esmc", weights_root)
            elif "esm2" in repo_id:
                download_repo(repo_id, "esm2", weights_root)
            else:
                raise ValueError(f"Cannot infer weight group for repo id: {repo_id}")
        return

    if args.group in ("all", "esmc"):
        for repo_id in ESMC_REPOS:
            download_repo(repo_id, "esmc", weights_root)
    if args.group in ("all", "esm2"):
        for repo_id in ESM2_REPOS:
            download_repo(repo_id, "esm2", weights_root)


if __name__ == "__main__":
    main()
