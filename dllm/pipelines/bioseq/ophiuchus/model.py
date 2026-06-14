from __future__ import annotations

import torch
import torch.nn as nn

from ..data import Esm2ProteinTokenizer
from .mint.model.esm2 import ESM2


class OphiuchusAbBackbone(nn.Module):
    def __init__(self, use_multimer: bool = True, token_dropout: bool = True) -> None:
        super().__init__()
        self.model = ESM2(
            num_layers=33,
            embed_dim=1280,
            attention_heads=20,
            alphabet="ESM-1b",
            token_dropout=token_dropout,
            use_multimer=use_multimer,
        )
        tokenizer = Esm2ProteinTokenizer()
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.unk_id = tokenizer.unk_token_id
        self.x_id = tokenizer.token_to_id["X"]
        self.b_id = tokenizer.token_to_id["B"]
        self.u_id = tokenizer.token_to_id["U"]
        self.z_id = tokenizer.token_to_id["Z"]
        self.o_id = tokenizer.token_to_id["O"]
        self.tokenizer = tokenizer

    def init_multimer_attention(self) -> None:
        for layer in self.model.layers:
            layer.multimer_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)

    def forward(self, input_ids: torch.Tensor, chain_ids: torch.Tensor | None = None, **_: object):
        outputs = self.model(input_ids, chain_ids, repr_layers=[self.model.num_layers])
        return {
            "logits": outputs["logits"],
            "last_hidden_state": outputs["representations"][self.model.num_layers],
        }
