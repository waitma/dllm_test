from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test"
)
DEFAULT_MODEL_WEIGHTS_ROOT = Path("/c20250601/mj/model_weights")
BIOSEQ_MODEL_WEIGHTS_ROOT_ENV = "BIOSEQ_MODEL_WEIGHTS_ROOT"

ESMC_REPOS: tuple[str, ...] = (
    "biohub/ESMC-300M",
    "biohub/ESMC-600M",
    "biohub/ESMC-6B",
)

ESM2_REPOS: tuple[str, ...] = (
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
)

ESM2_OPTIONAL_REPOS: tuple[str, ...] = (
    "facebook/esm2_t48_15B_UR50D",
)

ESM2_SMOKE_TEST_REPOS: tuple[str, ...] = (
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
)

OPHIUCHUS_AB_BASE_REPO = "facebook/esm2_t33_650M_UR50D"
OPHIUCHUS_AB_CHAIN_LENGTHS: tuple[int, int] = (150, 128)
OPHIUCHUS_AB_SEQUENCE_LENGTH = sum(OPHIUCHUS_AB_CHAIN_LENGTHS)
OPHIUCHUS_AB_VOCAB_SIZE = 33
OPHIUCHUS_AB_HIDDEN_SIZE = 1280
OPHIUCHUS_AB_NUM_HIDDEN_LAYERS = 33
OPHIUCHUS_AB_NUM_ATTENTION_HEADS = 20
OPHIUCHUS_AB_INTERMEDIATE_SIZE = 5120

OPHIUCHUS_AB_CHECKPOINT = "Ophiuchus-Ab.ckpt"
OPHIUCHUS_AB_CHECKPOINT_SIZE = 3_253_751_875
OPHIUCHUS_AB_CHECKPOINT_MD5 = "9baa0d3fbe908930d9a7d4f8d8b6144c"
OPHIUCHUS_AB_ZENODO_RECORD = "https://zenodo.org/records/18478480"


def get_model_weights_root() -> Path:
    configured = os.environ.get(BIOSEQ_MODEL_WEIGHTS_ROOT_ENV)
    return Path(configured).expanduser().resolve() if configured else DEFAULT_MODEL_WEIGHTS_ROOT


def repo_name(repo_id: str) -> str:
    return repo_id.rsplit("/", maxsplit=1)[-1]


def local_weight_dir(group: str, repo_id: str) -> Path:
    return get_model_weights_root() / group / repo_name(repo_id)


def ophiuchus_ab_checkpoint_path(weights_root: Path | str | None = None) -> Path:
    root = Path(weights_root).expanduser().resolve() if weights_root is not None else get_model_weights_root()
    return root / "ophiuchus_ab" / "Ophiuchus-Ab" / OPHIUCHUS_AB_CHECKPOINT


def ophiuchus_ab_esm2_weight_dir(weights_root: Path | str | None = None) -> Path:
    root = Path(weights_root).expanduser().resolve() if weights_root is not None else get_model_weights_root()
    return root / "esm2" / repo_name(OPHIUCHUS_AB_BASE_REPO)


@dataclass
class BioSeqModelConfig:
    vocab_size: int
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    pad_token_id: int = 0
    mask_token_id: int = 3
    use_multimer_attention: bool = True
    token_dropout: bool = False
    use_position_embeddings: bool = True
    condition_hidden_size: int | None = None


def ophiuchus_ab_model_config() -> BioSeqModelConfig:
    return BioSeqModelConfig(
        vocab_size=OPHIUCHUS_AB_VOCAB_SIZE,
        hidden_size=OPHIUCHUS_AB_HIDDEN_SIZE,
        num_hidden_layers=OPHIUCHUS_AB_NUM_HIDDEN_LAYERS,
        num_attention_heads=OPHIUCHUS_AB_NUM_ATTENTION_HEADS,
        intermediate_size=OPHIUCHUS_AB_INTERMEDIATE_SIZE,
        max_position_embeddings=OPHIUCHUS_AB_SEQUENCE_LENGTH,
        dropout=0.0,
        pad_token_id=1,
        mask_token_id=32,
        use_multimer_attention=True,
        token_dropout=True,
        use_position_embeddings=False,
    )
