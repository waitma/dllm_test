"""V11 all-global A5 encoder; load with V11A5Model.from_pretrained(ABS_CHECKPOINT).

Run CPU regressions: USE_TF=0 python -B -m unittest v11_a5.tests.
The V8 collator, corruption and loss are inherited without changes.
"""
from dataclasses import asdict
import json
from pathlib import Path
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyModel, RECIPE, OBJECTIVE

SCHEMA = 'immune.v11.a5.v1'
VARIANT = 'v11_a5_all_global'
OPTIONS = {'residual_scale': .1, 'segment_global': True, 'attention_schedule': 'all_global'}


class V11A5Model(VFamilyModel):
    def __init__(self, config, *, family_spec):
        super().__init__(config, family_spec=family_spec)
        self.variant = VARIANT
        self.options = dict(OPTIONS)

    @staticmethod
    def mask_key(layer):
        return ('global',)

    def layer_allowed(self, layer, chain_ids, position_ids, attention_mask):
        return attention_mask.bool()[:, None, :].expand(-1, chain_ids.shape[1], -1)

    def save_research_config(self, directory):
        settings = dict(schema=SCHEMA, version='V11', variant=VARIANT, objective=OBJECTIVE,
            options=self.options, base=asdict(self.config), recipe=RECIPE, family_spec=self.family_spec)
        (Path(directory)/'research_config.json').write_text(json.dumps(settings,indent=2)+'\n')

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory=Path(directory)
        spec=json.loads((directory/'research_config.json').read_text())
        if (spec.get('schema')!=SCHEMA or spec.get('version')!='V11'
                or spec.get('variant')!=VARIANT or spec.get('objective')!=OBJECTIVE
                or spec.get('options')!=OPTIONS or spec.get('recipe')!=RECIPE
                or spec.get('base')!=json.loads((directory/'immune_config.json').read_text())):
            raise ValueError('V11 A5 checkpoint architecture/objective contract differs')
        model=cls(FixedChainConfig(**spec['base']),family_spec=spec['family_spec'])
        if json.loads((directory/'vocabulary.json').read_text())!=[
                model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]:
            raise ValueError('V11 vocabulary differs')
        model.load_state_dict(load_file(str(directory/'model.safetensors')),strict=True)
        return model


def install_trainer_adapter():
    """Bind V11 only in this process; the frozen trainer itself stays byte-identical."""
    from functools import partial
    from scripts.research import run_bert_mlm as trainer
    from scripts.research.bert_mlm_objectives import components
    def v11_components(objective='receptor_aa',family_spec=None):
        if objective!=OBJECTIVE or family_spec is None:
            raise ValueError('V11 requires the original accepted V8 family objective')
        collator,_,corrupt,recipe=components(objective,family_spec)
        return collator,partial(V11A5Model,family_spec=family_spec),corrupt,recipe
    trainer.components=v11_components
    return trainer
