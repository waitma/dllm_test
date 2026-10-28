"""V8 MLM: negative records supervise only the hidden binding relation.

Run: USE_TF=0 python -B -m pytest scripts/tests/immune_llada/test_unbinding_mlm.py -q
Binding and unlabelled records retain full-token MLM. Explicit binding relations
are always masked and supervised; nonbinding sequences remain clean context.
"""
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .full_token_mlm import FullTokenMLMCollator, FullTokenMLMModel
from .full_token_mlm import corrupt as full_token_corrupt, RECIPE as FULL_RECIPE
from .model import FixedChainConfig

SCHEMA = "immune.fixed_chain.unbinding_bert_mlm.v1"
OBJECTIVE = "binding_mlm_nonbinding_relation_only"
RECIPE = dict(FULL_RECIPE, supervision=OBJECTIVE,
              relation_input="explicit_binding_nonbinding_always_MASK",
              relation_selection_probability=1.0,
              nonbinding="relation_only; all_other_positions_clean_and_loss_ignored",
              unlabelled="unchanged_full_token_mlm")


class UnbindingMLMCollator(FullTokenMLMCollator):
    """Keep label-dependent loss annotations outside encoder features."""

    def __call__(self, records):
        batch = super().__call__(records)
        ids = batch["input_ids"]
        explicit = ids.eq(self.tokenizer.special_id("<binding>")) | ids.eq(
            self.tokenizer.special_id("<nonbinding>"))
        negative = ids.eq(self.tokenizer.special_id("<nonbinding>")).any(1, keepdim=True)
        # Probe collation has already hidden all relations: no label-dependent
        # eligibility/features survive hide_relation=True.
        eligible = batch["eligible_mask"] & (~negative | explicit)
        batch["eligible_mask"] = eligible
        batch["labels"] = ids.masked_fill(~eligible, -100)
        batch["explicit_relation_mask"] = explicit
        batch["grammar_mask"] &= eligible
        return batch


def corrupt(batch, tokenizer, generator):
    """Preserve 15%/80-10-10 MLM; predict every explicit relation from MASK."""
    relation = batch["explicit_relation_mask"] & batch["eligible_mask"]
    result = full_token_corrupt({**batch, "eligible_mask": batch["eligible_mask"] & ~relation},
                                tokenizer, generator)
    result["eligible_mask"] = batch["eligible_mask"]
    result["input_ids"][relation] = tokenizer.mask_token_id
    result["labels"][relation] = batch["input_ids"][relation]
    result["noise_kind"][relation] = 1
    return result


class UnbindingMLMModel(FullTokenMLMModel):
    """Checkpoint-compatible full-vocabulary encoder with a distinct recipe."""

    @classmethod
    def from_parent(cls, directory):
        parent = FullTokenMLMModel.from_pretrained(directory)
        model = cls(parent.config)
        model.load_state_dict(parent.state_dict(), strict=True)
        return model

    def save_research_config(self, directory):
        settings = dict(schema=SCHEMA, variant=OBJECTIVE, options=self.options,
                        base=asdict(self.config), recipe=RECIPE)
        (Path(directory) / "research_config.json").write_text(json.dumps(settings, indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory / "research_config.json").read_text())
        if (settings.get("schema") != SCHEMA or settings.get("recipe") != RECIPE
                or settings.get("variant") != OBJECTIVE
                or settings.get("options") != {"residual_scale": .1, "segment_global": True}
                or settings["base"] != json.loads((directory / "immune_config.json").read_text())):
            raise ValueError("V8 unbinding checkpoint/recipe contract changed")
        model = cls(FixedChainConfig(**settings["base"]))
        if json.loads((directory / "vocabulary.json").read_text()) != [
                model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError("V8 vocabulary changed")
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        return model
