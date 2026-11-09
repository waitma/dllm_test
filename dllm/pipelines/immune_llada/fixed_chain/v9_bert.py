"""Label-independent input noise for fresh V9 BERT pretraining.

Activate pllm; run python -m pytest scripts/tests/immune_llada/test_v9_bert.py.
Negative sequence targets stay ignored; family and relation supervision remain.
"""
from dataclasses import asdict
import json
from pathlib import Path

from .full_token_mlm import corrupt as full_corrupt, excluded_ids
from .model import FixedChainConfig
from .v_family import VFamilyCollator, VFamilyModel, RECIPE as PREVIOUS_RECIPE

OBJECTIVE = 'binding_mlm_nonbinding_relation_only_vhh_v_family_label_independent'
SCHEMA = 'immune.fixed_chain.v9_bert_label_independent.v1'
RECIPE = dict(PREVIOUS_RECIPE, supervision=OBJECTIVE,
    nonbinding='token_loss_relation_only; label_independent_sequence_corruption; family_auxiliary_enabled',
    input_corruption='all eligible observed tokens; 15pct 80-10-10; identical rule for binding/nonbinding',
    target_selection='original label-dependent loss mask; explicit relation always MASK and supervised')
V9Collator = VFamilyCollator


def corrupt(batch, tokenizer, generator):
    eligible = batch['attention_mask'].clone()
    for token in excluded_ids(tokenizer):
        eligible &= batch['input_ids'].ne(token)
    relation = batch['explicit_relation_mask']
    noisy = full_corrupt({**batch, 'eligible_mask':eligible & ~relation}, tokenizer, generator)
    noisy['eligible_mask'] = batch['eligible_mask']
    noisy['labels'].masked_fill_(~batch['eligible_mask'], -100)
    noisy['input_ids'][relation] = tokenizer.mask_token_id
    noisy['labels'][relation] = batch['input_ids'][relation]
    noisy['noise_kind'][relation] = 1
    return noisy


class V9Model(VFamilyModel):
    def save_research_config(self, directory):
        settings = dict(schema=SCHEMA, variant=OBJECTIVE, options=self.options,
            base=asdict(self.config), recipe=RECIPE, family_spec=self.family_spec)
        (Path(directory) / 'research_config.json').write_text(json.dumps(settings, indent=2)+'\n')

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory / 'research_config.json').read_text())
        if (settings.get('schema') != SCHEMA or settings.get('variant') != OBJECTIVE
                or settings.get('recipe') != RECIPE
                or settings.get('options') != {'residual_scale':.1,'segment_global':True}
                or settings['base'] != json.loads((directory / 'immune_config.json').read_text())):
            raise ValueError('V9 BERT checkpoint contract changed')
        model = cls(FixedChainConfig(**settings['base']), family_spec=settings['family_spec'])
        if json.loads((directory / 'vocabulary.json').read_text()) != [
                model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError('V9 BERT vocabulary changed')
        model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
        return model
