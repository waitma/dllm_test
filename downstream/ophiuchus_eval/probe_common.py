"""Shared frozen-embedding helpers for official Ophiuchus-Ab probe tasks.

Matches the official extract.py protocol: ESM-2 650M tokenizer, pad VH/VL to
(150, 128) with <eos>, ``use_multimer=True``, ``sep_chains=True`` mean-pool.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from downstream.common import ChainPaddingCollator, PROJECT_ROOT
from downstream.embeddings import OphiuchusEmbeddingConfig, OphiuchusEmbeddingModel
from dllm.pipelines.bioseq import Esm2ProteinTokenizer, ophiuchus_ab_checkpoint_path

DEFAULT_CKPT = Path(
    "/project_c20250601/251105016/project/dllm_test/model_weights/"
    "ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt"
)


def resolve_checkpoint(path: str | Path | None) -> Path:
    if path:
        ckpt = Path(path)
        if ckpt.is_file():
            return ckpt
    for candidate in (
        DEFAULT_CKPT,
        Path("/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt"),
        ophiuchus_ab_checkpoint_path(),
        PROJECT_ROOT / "model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt",
    ):
        if Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Ophiuchus-Ab.ckpt not found")


def load_embedder(
    checkpoint_path: str | Path | None,
    device: torch.device,
    sep_chains: bool = True,
) -> OphiuchusEmbeddingModel:
    model = OphiuchusEmbeddingModel(
        checkpoint_path=resolve_checkpoint(checkpoint_path),
        config=OphiuchusEmbeddingConfig(sep_chains=sep_chains, token_dropout=True),
        device=device,
    )
    model.eval()
    return model


class PairSeqDataset(Dataset):
    def __init__(self, heavy: list[str], light: list[str]):
        if len(heavy) != len(light):
            raise ValueError("heavy/light length mismatch")
        self.heavy = [str(x).replace("J", "L") for x in heavy]
        self.light = [str(x).replace("J", "L") for x in light]

    def __len__(self) -> int:
        return len(self.heavy)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.heavy[idx], self.light[idx]


@torch.no_grad()
def embed_pairs(
    model: OphiuchusEmbeddingModel,
    heavy: list[str],
    light: list[str],
    *,
    batch_size: int = 32,
    device: torch.device,
    desc: str = "embed",
) -> torch.Tensor:
    collator = ChainPaddingCollator(tokenizer=Esm2ProteinTokenizer())
    loader = DataLoader(
        PairSeqDataset(heavy, light),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    chunks: list[torch.Tensor] = []
    model.eval()
    for h_batch, l_batch in tqdm(loader, desc=desc):
        input_ids, chain_ids, _ = collator.stack_chains(list(h_batch), list(l_batch))
        emb = model(input_ids.to(device), chain_ids.to(device))
        chunks.append(emb.detach().cpu())
    return torch.cat(chunks, dim=0)
