from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from dllm.pipelines.bioseq.ophiuchus.model import OphiuchusAbBackbone
from dllm.pipelines.bioseq import MultiChainOphiuchusAbModel, load_ophiuchus_checkpoint, ophiuchus_ab_checkpoint_path


@dataclass
class OphiuchusEmbeddingConfig:
    num_hidden_layers: int = 33
    hidden_size: int = 1280
    num_attention_heads: int = 20
    token_dropout: bool = True
    sep_chains: bool = False


class OphiuchusEmbeddingModel(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config: OphiuchusEmbeddingConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or OphiuchusEmbeddingConfig()
        self.sep_chains = self.config.sep_chains
        self.backbone = OphiuchusAbBackbone(
            use_multimer=True,
            token_dropout=self.config.token_dropout,
        )
        wrapper = MultiChainOphiuchusAbModel(net=self.backbone)
        load_ophiuchus_checkpoint(
            wrapper,
            checkpoint_path or ophiuchus_ab_checkpoint_path(),
            device=device,
        )
        self.model = self.backbone.model
        self.to(device)

    def _mask(self, chains: torch.Tensor) -> torch.Tensor:
        return (
            chains.ne(self.model.cls_idx)
            & chains.ne(self.model.eos_idx)
            & chains.ne(self.model.padding_idx)
        )

    def _pool_chain(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1).expand_as(hidden)
        masked = hidden * mask_expanded
        return masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).float()

    def forward(self, chains: torch.Tensor, chain_ids: torch.Tensor, inter_chain_mask=None):
        hidden = self.model(chains, chain_ids, repr_layers=[self.config.num_hidden_layers])[
            "representations"
        ][self.config.num_hidden_layers]
        mask = self._mask(chains)
        if self.sep_chains:
            outputs = []
            for chain_id in range(int(chain_ids.max().item()) + 1):
                chain_mask = (chain_ids == chain_id) & mask
                outputs.append(self._pool_chain(hidden, chain_mask))
            return torch.cat(outputs, dim=-1)
        return self._pool_chain(hidden, mask)
