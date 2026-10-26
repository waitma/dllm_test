"""Conditional diffusion/edit adaptation of the full-token BERT checkpoint.

Activate pllm; python -B -m pytest
scripts/tests/immune_llada/test_diffusion_continuation_objective.py -q.
Antigen/peptide/MHC and structural tokens remain clean. Receptor AA/EOS and
masked relation tokens are supervised. Optional CDR tasks condition on known
loop spans and fixed observed frameworks; whole-chain tasks keep their old layout.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator, FullTokenMLMModel
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig, RESIDUES
from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
from scripts.research.diffusion_continuation_losses import weighted_token_loss
from scripts.research.diffusion_cdr_tasks import annotate_cdrs, validate_cdr_tasks, select_cdr_targets

SCHEMA = "immune.fixed_chain.conditional_diffusion_edit.v1"
MODES = ("joint", "independent", "first_to_second", "second_to_first", "single")


def default_recipe(eos_gamma=0.):
    return dict(task_ratios=dict(zip(MODES, (.35, .25, .15, .15, .10))),
                edit_probability=.25, time_epsilon=.001, inverse_time_mix=.5,
                reciprocal_offset=.05, complementary=True, keep_weight=.1,
                relation_weight=.1, eos_gamma=eos_gamma, eos_alpha=1.,
                normalization="mean_per_record_view_receptor_length_plus_relation",
                context="clean_antigen_peptide_mhc", relation="always_masked_and_supervised")


def validate_recipe(recipe):
    if (set(recipe) - {"cdr_tasks"} != set(default_recipe())
            or set(recipe["task_ratios"]) != set(MODES)):
        raise ValueError("Unknown/missing diffusion recipe fields")
    if "cdr_tasks" in recipe:
        validate_cdr_tasks(recipe["cdr_tasks"])
    ratios = list(recipe["task_ratios"].values())
    if any(not math.isfinite(x) or x < 0 for x in ratios) or not math.isclose(sum(ratios), 1.):
        raise ValueError("Task ratios must be probabilities summing to one")
    for name in ("edit_probability", "inverse_time_mix"):
        if not 0 <= recipe[name] <= 1:
            raise ValueError("Invalid probability: " + name)
    if not 0 < recipe["time_epsilon"] < .5 or recipe["reciprocal_offset"] < 0:
        raise ValueError("Invalid noise interval/reciprocal offset")
    for name in ("keep_weight", "relation_weight", "eos_gamma", "eos_alpha"):
        if not math.isfinite(recipe[name]) or recipe[name] < 0:
            raise ValueError("Invalid loss coefficient: " + name)
    if recipe["eos_alpha"] <= 0 or recipe["relation_weight"] <= 0:
        raise ValueError("EOS and relation supervision must be enabled")
    for name in ("normalization", "context", "relation"):
        if recipe[name] != default_recipe()[name]:
            raise ValueError("Unsupported training semantics: " + name)
    if recipe["complementary"] is not True:
        raise ValueError("This recipe requires complementary views")


class DiffusionEditCollator(FullTokenMLMCollator):
    def __call__(self, records):
        batch = super().__call__(records)
        receptor = (batch["residue_mask"] | batch["terminal_mask"]) & batch["target_slot_mask"]
        batch["eligible_mask"] = receptor
        batch["labels"] = batch["input_ids"].masked_fill(~(receptor | batch["relation_mask"]), -100)
        batch["cdr_ids"] = annotate_cdrs(batch, records)
        return batch


def sample_times(shape, generator, recipe):
    """Uniform/log-uniform mixture, with density correction to uniform time.

    Inverse time sampling and reciprocal diffusion loss are separate choices.
    Both complementary views use the ORIGINAL draw's importance correction.
    """
    lo, hi = recipe["time_epsilon"], 1 - recipe["time_epsilon"]
    mix = recipe["inverse_time_mix"]
    choose = torch.rand(shape, generator=generator) < mix
    u = torch.rand(shape, generator=generator)
    time = torch.where(choose, lo * torch.exp(u * math.log(hi / lo)), lo + u * (hi - lo))
    density = (1 - mix) / (hi - lo) + mix / (time * math.log(hi / lo))
    return time, (1 / (hi - lo)) / density


def corrupt(batch, tokenizer, generator, recipe, *, mode=None, ratio=None, edit=None,
            paired=True, mask_scope=None, target_cdrs=None):
    """CPU corruption before sharding into microbatches; annotations are not features."""
    validate_recipe(recipe)
    clean, eligible = batch["input_ids"], batch["eligible_mask"]
    if clean.device.type != "cpu":
        raise ValueError("Use CPU corruption with the checkpointed CPU generator")
    bsz, length = clean.shape
    first = (batch["role_ids"].eq(CHAIN_ROLE_TO_ID["antibody_heavy"])
             | batch["role_ids"].eq(CHAIN_ROLE_TO_ID["tcr_beta"])) & batch["target_slot_mask"]
    second = batch["target_slot_mask"] & ~first
    present = torch.stack([(first & eligible).any(1), (second & eligible).any(1)], 1)
    if not present.any(1).all():
        raise ValueError("Every record must contain observed receptor supervision")
    probabilities = torch.tensor([recipe["task_ratios"][key] for key in MODES])
    modes = torch.multinomial(probabilities, bsz, replacement=True, generator=generator)
    if mode is not None:
        modes.fill_(MODES.index(mode))
    modes[~present.all(1)] = MODES.index("joint")
    times, corrections = sample_times((bsz, 2), generator, recipe)
    if ratio is not None:
        if not 0 <= ratio <= 1:
            raise ValueError("Fixed validation ratio outside [0, 1]")
        times.fill_(ratio)
        corrections.fill_(1.)
    original_times, original_corrections = times.clone(), corrections.clone()
    active = torch.ones((bsz, 2), dtype=torch.bool) & present
    hide = torch.zeros_like(active)
    importance = torch.ones(bsz)
    single = torch.randint(2, (bsz,), generator=generator)
    for row in range(bsz):
        kind = MODES[int(modes[row])]
        if kind == "joint":
            times[row, 1] = times[row, 0]
            importance[row] = corrections[row, 0]
        elif kind == "independent":
            importance[row] = corrections[row].prod()
        else:
            target = 1 if kind == "first_to_second" else 0
            if kind == "single":
                target = int(single[row])
                hide[row, 1-target] = True
            active[row, 1-target] = False
            importance[row] = corrections[row, target]
    times[~active] = 0.
    is_edit = torch.rand((bsz, 1), generator=generator) < recipe["edit_probability"]
    if edit is not None:
        is_edit.fill_(edit)
    population = eligible & ((first & active[:, :1]) | (second & active[:, 1:]))
    hidden_context_chain = (first & hide[:, :1]) | (second & hide[:, 1:])
    cdr_metadata = {}
    if "cdr_tasks" in recipe:
        target, scopes, cdr_eligible = select_cdr_targets(
            batch, is_edit, generator, recipe["cdr_tasks"], scope=mask_scope, target_cdrs=target_cdrs)
        scoped = scopes.ne(0)
        population[scoped] = target[scoped]
        hidden_context_chain[scoped] = False
        modes[scoped] = MODES.index("joint")
        # The two selected-chain times from the old task cannot govern a new CDR
        # task. Use one fresh joint draw and its corresponding density correction.
        times[scoped] = original_times[scoped, :1].expand(-1, 2)
        importance[scoped] = original_corrections[scoped, 0]
        full = torch.zeros(bsz, dtype=torch.bool)
        if ratio is None:
            positions = scoped.nonzero().flatten()
            full[positions] = torch.rand(len(positions), generator=generator) < recipe["cdr_tasks"]["full_mask_probability"]
            times[full] = 1.
            importance[full] = 1.  # explicit endpoint objective, not a continuous-time draw
        else:
            full[scoped] = ratio == 1.
        cdr_metadata = dict(mask_scope=scopes[:, None], cdr_full_mask=full[:, None],
                            cdr_eligible=cdr_eligible[:, None], edit_task=is_edit)
    elif mask_scope not in (None, "chain") or target_cdrs is not None:
        raise ValueError("This recipe has no CDR task branch")
    p = torch.where(first, times[:, :1], times[:, 1:])
    u = torch.rand(clean.shape, generator=generator)
    draw = (u < p) & population
    inverse = (~draw) & population
    endpoints = p.eq(0) | p.eq(1)
    inverse = torch.where(endpoints, draw, inverse)
    reciprocal_offset = recipe["reciprocal_offset"]
    record_length = population.sum(1, keepdim=True).clamp_min(1)
    relation = batch["relation_mask"]
    relation_count = relation.sum(1, keepdim=True).clamp_min(1)
    legal = torch.tensor([*tokenizer.encode_residues(RESIDUES), tokenizer.eos_token_id])
    lookup = torch.full((tokenizer.vocab_size,), -1, dtype=torch.long)
    lookup[legal] = torch.arange(len(legal))
    views = []
    for view in range(2 if paired else 1):
        selected = inverse if view else draw
        probability = torch.where(endpoints, p, 1-p) if view else p
        masked = selected & ~is_edit
        replaced = selected & is_edit
        kept = population & ~selected & is_edit
        ids = clean.masked_fill(hidden_context_chain | masked | relation, tokenizer.mask_token_id)
        original = lookup[clean[replaced]]
        if (original < 0).any():
            raise ValueError("Replacement target is outside the AA/EOS family")
        replacement = torch.randint(len(legal)-1, original.shape, generator=generator)
        replacement += replacement.ge(original).long()
        ids[replaced] = legal[replacement]
        labels = clean.masked_fill(~(selected | kept | relation), -100)
        weights = selected.float() / (probability + reciprocal_offset).clamp_min(1e-8)
        weights += kept.float() * recipe["keep_weight"] / (1-probability+reciprocal_offset).clamp_min(1e-8)
        weights /= record_length
        weights += relation.float() * recipe["relation_weight"] / relation_count
        weights *= importance[:, None]
        kind = torch.zeros_like(clean)
        kind[masked], kind[replaced], kind[kept], kind[relation] = 1, 2, 3, 4
        views.append({**batch, **cdr_metadata, "input_ids": ids, "labels": labels, "loss_weights": weights,
                      "noise_kind": kind, "task_mode": modes[:, None],
                      "sampled_times": times, "time_importance": importance[:, None],
                      "active_population": population})
    return {key: torch.cat([item[key] for item in views], 0) for key in views[0]}


class DiffusionEditModel(FullTokenMLMModel):
    """No extra parameters; preserve pretrained parameter keys and input layout."""
    def __init__(self, config, recipe):
        validate_recipe(recipe)
        super().__init__(config)
        self.recipe = json.loads(json.dumps(recipe))

    def forward(self, **batch):
        logits, _ = self.denoise(**self.feature_args(batch))
        return weighted_token_loss(logits, batch["labels"], batch["loss_weights"],
                                   eos_id=self.tokenizer.eos_token_id,
                                   eos_gamma=self.recipe["eos_gamma"], eos_alpha=self.recipe["eos_alpha"],
                                   per_view="mask_scope" in batch)

    def save_research_config(self, directory):
        (Path(directory) / "research_config.json").write_text(json.dumps(dict(
            schema=SCHEMA, variant="conditional_diffusion_edit", base=asdict(self.config),
            options=self.options, recipe=self.recipe), indent=2)+"\n")

    @classmethod
    def from_parent(cls, directory, recipe):
        parent = FullTokenMLMModel.from_pretrained(directory)
        model = cls(parent.config, recipe)
        model.load_state_dict(parent.state_dict(), strict=True)
        return model

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory / "research_config.json").read_text())
        if (settings.get("schema") != SCHEMA or settings.get("variant") != "conditional_diffusion_edit"
                or settings.get("options") != {"residual_scale": .1, "segment_global": True}
                or settings["base"] != json.loads((directory / "immune_config.json").read_text())):
            raise ValueError("Diffusion/edit checkpoint contract differs")
        model = cls(FixedChainConfig(**settings["base"]), settings["recipe"])
        if json.loads((directory / "vocabulary.json").read_text()) != [
                model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError("Diffusion/edit vocabulary differs")
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        return model
