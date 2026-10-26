"""MLM over observed content and grammar, excluding protein block delimiters.

Run: python -B -m pytest scripts/tests/immune_llada/test_full_token_mlm.py -q
Padding, synthetic MASK positions and unused canvas positions are never targets.
"""
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .layout import FixedChainCollator
from .mlm import BertMLMModel
from .model import FixedChainConfig, RESIDUES
from ..data.grammar import GRAMMAR_RELATION_TOKENS

SCHEMA = "immune.fixed_chain.full_token_bert_mlm.v1"
RECIPE = dict(selection_probability=0.15, mask_fraction=0.8,
              random_other_fraction=0.1, unchanged_fraction=0.1,
              supervision="all_observed_tokens_except_prots_protd",
              excluded_targets=["<prots>", "<protd>", "<pad>", "<mask>"],
              random_replacement="other_token_in_same_residue_or_grammar_family",
              relation_input="observed_for_mlm_hidden_for_representation_probe",
              minimum_selected_per_record=0, loss="global_selected_token_mean")


def excluded_ids(tokenizer):
    return (tokenizer.special_id("<prots>"), tokenizer.special_id("<protd>"),
            tokenizer.pad_token_id, tokenizer.mask_token_id)


class FullTokenMLMCollator(FixedChainCollator):
    """Keep real relation input for MLM; explicitly hide it for labelled probes."""

    def __init__(self, config=None, *, hide_relation=False):
        super().__init__(config)
        self.hide_relation = hide_relation

    def __call__(self, records):
        batch = super().__call__(records)
        ids = batch["input_ids"]
        relation = torch.zeros_like(ids, dtype=torch.bool)
        for token in GRAMMAR_RELATION_TOKENS:
            relation |= ids.eq(self.tokenizer.special_id(token))
        if self.hide_relation:
            ids.masked_fill_(relation, self.tokenizer.mask_token_id)
        eligible = batch["attention_mask"].clone()
        for token_id in excluded_ids(self.tokenizer):
            eligible &= ids.ne(token_id)
        batch["eligible_mask"] = eligible
        batch["labels"] = ids.masked_fill(~eligible, -100)
        # Metric annotations only; feature_args never sends these to the encoder.
        batch["relation_mask"] = relation
        batch["receptor_residue_mask"] = batch["residue_mask"] & batch["target_slot_mask"]
        batch["context_residue_mask"] = batch["residue_mask"] & ~batch["target_slot_mask"]
        batch["grammar_mask"] = eligible & ~batch["residue_mask"] & ~batch["terminal_mask"]
        return batch


def corrupt(batch, tokenizer, generator):
    """Apply 15% / 80-10-10 to every eligible token, retaining original labels."""
    clean = batch["input_ids"]
    selected = (torch.rand(clean.shape, generator=generator, device=clean.device)
                < RECIPE["selection_probability"]) & batch["eligible_mask"]
    draw = torch.rand(clean.shape, generator=generator, device=clean.device)
    masked = selected & (draw < .8)
    replaced = selected & (draw >= .8) & (draw < .9)
    unchanged = selected & (draw >= .9)
    noised = clean.clone()
    noised[masked] = tokenizer.mask_token_id
    aa_ids = tokenizer.encode_residues(RESIDUES)
    aa_set = set(aa_ids)
    forbidden = set(excluded_ids(tokenizer))
    grammar_ids = [i for i in range(tokenizer.vocab_size) if i not in aa_set | forbidden]
    for pool, positions in ((aa_ids, replaced & batch["residue_mask"]),
                            (grammar_ids, replaced & ~batch["residue_mask"])):
        legal = torch.tensor(pool, device=clean.device)
        lookup = torch.full((tokenizer.vocab_size,), -1, device=clean.device, dtype=torch.long)
        lookup[legal] = torch.arange(len(legal), device=clean.device)
        original = lookup[clean[positions]]
        if (original < 0).any():
            raise ValueError("Token family does not match its observation mask")
        replacement = torch.randint(len(legal)-1, original.shape, generator=generator, device=clean.device)
        replacement += (replacement >= original).long()
        noised[positions] = legal[replacement]
    kinds = torch.zeros_like(clean)
    kinds[masked], kinds[replaced], kinds[unchanged] = 1, 2, 3
    return {**batch, "input_ids": noised, "labels": clean.masked_fill(~selected, -100),
            "noise_kind": kinds}


class FullTokenMLMModel(BertMLMModel):
    """Same encoder; EOS and grammar have finite output logits and CE gradients."""

    def __init__(self, config):
        super().__init__(config)
        self.allowed_target.fill_(True)
        self.allowed_target[list(excluded_ids(self.tokenizer))] = False

    def denoise(self, *args, **kwargs):
        logits, hidden = super().denoise(*args, **kwargs)
        return logits.masked_fill(~self.allowed_target[None, None], float("-inf")), hidden

    def save_research_config(self, directory):
        settings = dict(schema=SCHEMA, variant="fixed_chain_full_token_bert_mlm",
                        options=self.options, base=asdict(self.config), recipe=RECIPE)
        (Path(directory)/"research_config.json").write_text(json.dumps(settings, indent=2)+"\n")

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory/"research_config.json").read_text())
        if (settings.get("schema") != SCHEMA or settings.get("recipe") != RECIPE
                or settings.get("variant") != "fixed_chain_full_token_bert_mlm"
                or settings.get("options") != {"residual_scale": .1, "segment_global": True}
                or settings["base"] != json.loads((directory/"immune_config.json").read_text())):
            raise ValueError("Full-token MLM checkpoint/recipe contract changed")
        model = cls(FixedChainConfig(**settings["base"]))
        vocab = [model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]
        if json.loads((directory/"vocabulary.json").read_text()) != vocab:
            raise ValueError("Full-token MLM vocabulary changed")
        model.load_state_dict(load_file(str(directory/"model.safetensors")), strict=True)
        return model
