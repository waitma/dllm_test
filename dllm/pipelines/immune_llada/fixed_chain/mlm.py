"""BERT-style residue recovery on the fixed-chain bidirectional backbone.

Run: python -B -m pytest scripts/tests/immune_llada/test_bert_mlm.py -q
The corruption flags are loss annotations, never model input features.
"""
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .layout import FixedChainCollator
from .model import FixedChainConfig, FixedChainModel, RESIDUES, RECEPTOR_ROLES
from ..data.grammar import GRAMMAR_RELATION_TOKENS
from ..data.records import CHAIN_ROLE_TO_ID

SCHEMA = "immune.fixed_chain.bert_mlm.v1"
RECIPE = dict(selection_probability=0.15, mask_fraction=0.8,
              random_other_fraction=0.1, unchanged_fraction=0.1,
              residues=RESIDUES, supervision="observed_receptor_residues_only",
              relation_input="all_relation_tokens_replaced_by_MASK",
              minimum_selected_per_record=0, loss="global_selected_token_mean")


class BertMLMCollator(FixedChainCollator):
    """Render clean inputs, hide relation labels and select eligible real residues."""

    def __call__(self, records):
        batch = super().__call__(records)
        for token in GRAMMAR_RELATION_TOKENS:
            relation = batch["input_ids"].eq(self.tokenizer.special_id(token))
            batch["input_ids"].masked_fill_(relation, self.tokenizer.mask_token_id)
        eligible = (batch["target_slot_mask"] & batch["residue_mask"]
                    & batch["attention_mask"])
        batch["eligible_mask"] = eligible
        batch["labels"] = batch["input_ids"].masked_fill(~eligible, -100)
        return batch


def corrupt(batch, tokenizer, generator):
    """Independently select 15%; apply 80% MASK / 10% other AA / 10% identity.

    Bernoulli selection has no forced minimum. A short record may contribute no
    target in one visit; it still counts as visited in that complete epoch.
    """
    clean = batch["input_ids"]
    eligible = batch["eligible_mask"]
    selected = (torch.rand(clean.shape, generator=generator, device=clean.device)
                < RECIPE["selection_probability"]) & eligible
    draw = torch.rand(clean.shape, generator=generator, device=clean.device)
    mask = selected & (draw < 0.8)
    random_other = selected & (draw >= 0.8) & (draw < 0.9)
    unchanged = selected & (draw >= 0.9)
    noised = clean.clone()
    noised[mask] = tokenizer.mask_token_id
    legal = torch.tensor(tokenizer.encode_residues(RESIDUES), device=clean.device)
    lookup = torch.full((tokenizer.vocab_size,), -1, device=clean.device, dtype=torch.long)
    lookup[legal] = torch.arange(len(legal), device=clean.device)
    original = lookup[clean[random_other]]
    if (original < 0).any():
        raise ValueError("Eligible residue is outside the bound vocabulary")
    # Draw uniformly from the other K-1 residues, including the legal ambiguity tokens.
    replacement = torch.randint(len(legal)-1, original.shape, generator=generator,
                                device=clean.device)
    replacement += (replacement >= original).long()
    noised[random_other] = legal[replacement]
    kind = torch.zeros_like(clean)
    kind[mask], kind[random_other], kind[unchanged] = 1, 2, 3
    return {**batch, "input_ids": noised,
            "labels": clean.masked_fill(~selected, -100), "noise_kind": kind}


def selected_loss(logits, labels):
    """Return a sum: DDP/accumulation normalizes once over the optimizer step."""
    selected = labels.ne(-100)
    if not selected.any():
        # A finite vocabulary column keeps the entire graph connected on empty ranks.
        return logits[..., 0].nan_to_num(neginf=0).sum() * 0
    return F.cross_entropy(logits[selected].float(), labels[selected], reduction="sum")


class BertMLMModel(FixedChainModel):
    """Same 75.6M architecture; AA-only recovery and a separate checkpoint schema."""

    def __init__(self, config):
        super().__init__(config)
        self.allowed_target[self.tokenizer.eos_token_id] = False

    def forward(self, **batch):
        logits, _ = self.denoise(**self.feature_args(batch))
        return {"loss_sum": selected_loss(logits, batch["labels"]), "logits": logits}

    @torch.no_grad()
    def receptor_vectors(self, batch):
        """Concatenate observed residue means in H, L, beta, alpha order.

        Missing chains are zero vectors. Context can affect attention but is not
        mixed into the receptor pooling population.
        """
        _, hidden = self.denoise(**self.feature_args(batch), output_hidden_states=True)
        pooled = []
        for role in RECEPTOR_ROLES:
            mask = (batch["residue_mask"] & batch["attention_mask"]
                    & batch["role_ids"].eq(CHAIN_ROLE_TO_ID[role]))
            pooled.append((hidden.float() * mask.unsqueeze(-1)).sum(1)
                          / mask.sum(1, keepdim=True).clamp_min(1))
        return torch.cat(pooled, dim=-1)

    def save_research_config(self, directory):
        settings = dict(schema=SCHEMA, variant="fixed_chain_bert_mlm",
                        options=self.options, base=asdict(self.config), recipe=RECIPE)
        (Path(directory) / "research_config.json").write_text(json.dumps(settings, indent=2)+"\n")

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory / "research_config.json").read_text())
        if (settings.get("schema") != SCHEMA or settings.get("recipe") != RECIPE
                or settings.get("variant") != "fixed_chain_bert_mlm"
                or settings.get("options") != {"residual_scale": .1, "segment_global": True}
                or settings["base"] != json.loads((directory / "immune_config.json").read_text())):
            raise ValueError("BERT model/recipe contract changed")
        model = cls(FixedChainConfig(**settings["base"]))
        if json.loads((directory / "vocabulary.json").read_text()) != [
                model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError("BERT vocabulary changed")
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        return model
