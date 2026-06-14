from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from dllm.pipelines.bioseq import (
    Esm2ProteinTokenizer,
    MultiChainOphiuchusAbModel,
    OPHIUCHUS_AB_CHAIN_LENGTHS,
    load_ophiuchus_checkpoint,
    ophiuchus_ab_checkpoint_path,
)

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "downstream"


def load_model(checkpoint_path: str | Path | None = None, device: str | torch.device = "cpu"):
    checkpoint_path = Path(checkpoint_path or ophiuchus_ab_checkpoint_path())
    model = MultiChainOphiuchusAbModel()
    load_ophiuchus_checkpoint(model, checkpoint_path, device=device)
    model.eval()
    return model.to(device)


def build_partial_mask(input_ids: torch.Tensor, tokenizer: Esm2ProteinTokenizer) -> torch.Tensor:
    return input_ids.ne(tokenizer.mask_token_id) & input_ids.ne(tokenizer.cls_token_id)


@dataclass
class ChainPaddingCollator:
    tokenizer: Esm2ProteinTokenizer
    chain_lengths: tuple[int, int] = OPHIUCHUS_AB_CHAIN_LENGTHS

    def encode_chain(self, sequence: str, chain_index: int) -> tuple[torch.Tensor, int]:
        max_len = self.chain_lengths[chain_index]
        encoded, _ = self.tokenizer.encode_chain(sequence.replace("J", "L"), max_length=max_len)
        tokens = torch.full((max_len,), self.tokenizer.eos_token_id, dtype=torch.long)
        true_len = min(len(encoded), max_len)
        tokens[:true_len] = torch.tensor(encoded[:true_len], dtype=torch.long)
        return tokens, true_len

    def stack_chains(
        self,
        heavy_sequences: Sequence[str],
        light_sequences: Sequence[str],
        mask_light_from: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        heavy_tokens = []
        light_tokens = []
        light_lens = []
        labels_light = []

        for heavy, light in zip(heavy_sequences, light_sequences):
            heavy_tensor, _ = self.encode_chain(heavy, 0)
            light_tensor, true_len = self.encode_chain(light, 1)
            labels_light.append(light_tensor.clone())
            if mask_light_from is not None:
                observed_aa_count = max(true_len - 2, 0)
                prompt_aa_count = min(max(mask_light_from, 0), observed_aa_count)
                start = 1 + prompt_aa_count
                light_tensor[start:] = self.tokenizer.mask_token_id
            heavy_tokens.append(heavy_tensor)
            light_tokens.append(light_tensor)
            light_lens.append(true_len)

        heavy_batch = torch.stack(heavy_tokens, dim=0)
        light_batch = torch.stack(light_tokens, dim=0)
        input_ids = torch.cat([heavy_batch, light_batch], dim=-1)
        chain_ids = torch.cat(
            [
                torch.zeros_like(heavy_batch, dtype=torch.long),
                torch.ones_like(light_batch, dtype=torch.long),
            ],
            dim=-1,
        )
        meta = {
            "heavy_max_len": heavy_batch.shape[-1],
            "light_max_len": light_batch.shape[-1],
            "light_true_lens": light_lens,
            "labels_light": torch.stack(labels_light, dim=0),
        }
        return input_ids, chain_ids, meta


@dataclass
class CdrInfillCollator:
    tokenizer: Esm2ProteinTokenizer
    chain_lengths: tuple[int, int] = OPHIUCHUS_AB_CHAIN_LENGTHS

    def __call__(self, batches):
        heavy_chains, light_chains, _, pos_idx = zip(*batches)
        heavy = self._convert(list(heavy_chains), 0)
        light = self._convert(list(light_chains), 1)
        label_heavy = heavy.clone()
        label_light = light.clone()
        labels_tensor = torch.cat([label_heavy, label_light], dim=-1)

        heavy_masked = heavy.clone()
        for i, (start_idx, end_idx) in enumerate(pos_idx):
            heavy_masked[i, start_idx + 1 : end_idx + 2] = self.tokenizer.mask_token_id

        input_ids = torch.cat([heavy_masked, light], dim=-1)
        chain_ids = torch.cat(
            [
                torch.zeros_like(heavy_masked, dtype=torch.long),
                torch.ones_like(light, dtype=torch.long),
            ],
            dim=-1,
        )
        return input_ids, chain_ids, labels_tensor

    def _convert(self, sequences: list[str], chain_index: int) -> torch.Tensor:
        rows = []
        max_len = self.chain_lengths[chain_index]
        for sequence in sequences:
            encoded = self.tokenizer.encode(sequence.replace("J", "L"))
            tokens = torch.full((max_len,), self.tokenizer.eos_token_id, dtype=torch.long)
            true_len = min(len(encoded), max_len)
            tokens[:true_len] = torch.tensor(encoded[:true_len], dtype=torch.long)
            rows.append(tokens)
        return torch.stack(rows, dim=0)

    def encode(self, sequence: str) -> list[int]:
        return self.tokenizer.encode(sequence.replace("J", "L"))


@dataclass
class Sab23H2Collator:
    tokenizer: Esm2ProteinTokenizer
    chain_lengths: tuple[int, int] = OPHIUCHUS_AB_CHAIN_LENGTHS

    def __call__(self, batches):
        heavy_chains, light_chains, heavy_labels, light_labels = zip(*batches)
        chains = torch.cat(
            [
                self._convert(list(heavy_chains), 0),
                self._convert(list(light_chains), 1),
            ],
            dim=-1,
        )
        labels = torch.cat(
            [
                self._convert(list(heavy_labels), 0),
                self._convert(list(light_labels), 1),
            ],
            dim=-1,
        )
        chain_ids = torch.cat(
            [
                torch.zeros((chains.size(0), self.chain_lengths[0]), dtype=torch.long),
                torch.ones((chains.size(0), self.chain_lengths[1]), dtype=torch.long),
            ],
            dim=-1,
        )
        return chains, chain_ids, labels

    def _convert(self, sequences: list[str], chain_index: int) -> torch.Tensor:
        rows = []
        max_len = self.chain_lengths[chain_index]
        for sequence in sequences:
            normalized = sequence.replace("J", "L").replace("X", "<mask>")
            encoded = self.tokenizer.encode(normalized)
            tokens = torch.full((max_len,), self.tokenizer.eos_token_id, dtype=torch.long)
            true_len = min(len(encoded), max_len)
            tokens[:true_len] = torch.tensor(encoded[:true_len], dtype=torch.long)
            rows.append(tokens)
        return torch.stack(rows, dim=0)

    def encode(self, sequence: str) -> list[int]:
        return self.tokenizer.encode(sequence.replace("J", "L").replace("X", "<mask>"))


def run_generate(
    model: MultiChainOphiuchusAbModel,
    input_ids: torch.Tensor,
    chain_ids: torch.Tensor,
    tokenizer: Esm2ProteinTokenizer,
    *,
    max_iter: int,
    sampling_strategy: str,
    temperature: float,
    cfg_scale: float,
    partial_masks: torch.Tensor | None = None,
):
    batch = {"input_ids": input_ids, "chain_ids": chain_ids}
    if partial_masks is None:
        partial_masks = build_partial_mask(input_ids, tokenizer)
    with torch.no_grad():
        return model.generate(
            batch,
            max_iter=max_iter,
            temperature=temperature,
            sampling_strategy=sampling_strategy,
            partial_masks=partial_masks,
            cfg_scale=cfg_scale,
        )
