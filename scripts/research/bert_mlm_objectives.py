"""Select explicitly versioned BERT objectives; run test_full_token_mlm.py.

Legacy plans default to AA-only. Full-token plans must name their objective.
"""


def components(objective="receptor_aa", family_spec=None):
    if objective == 'binding_mlm_nonbinding_relation_only_vhh_v_family':
        from functools import partial
        from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator, VFamilyModel, corrupt, RECIPE
        if family_spec is None:
            return VFamilyCollator, VFamilyModel, corrupt, RECIPE
        return partial(VFamilyCollator, family_spec=family_spec), partial(VFamilyModel, family_spec=family_spec), corrupt, RECIPE
    if objective == "binding_mlm_nonbinding_relation_only_vhh":
        from dllm.pipelines.immune_llada.fixed_chain.unbinding_vhh import (
            VHHUnbindingCollator, VHHUnbindingModel, corrupt, RECIPE)
        return VHHUnbindingCollator, VHHUnbindingModel, corrupt, RECIPE
    if objective == "binding_mlm_nonbinding_relation_only":
        from dllm.pipelines.immune_llada.fixed_chain.unbinding_mlm import (
            UnbindingMLMCollator, UnbindingMLMModel, corrupt, RECIPE)
        return UnbindingMLMCollator, UnbindingMLMModel, corrupt, RECIPE
    if objective == "receptor_aa":
        from dllm.pipelines.immune_llada.fixed_chain.mlm import (
            BertMLMCollator, BertMLMModel, corrupt, RECIPE)
        return BertMLMCollator, BertMLMModel, corrupt, RECIPE
    if objective == "all_observed_tokens":
        from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import (
            FullTokenMLMCollator, FullTokenMLMModel, corrupt, RECIPE)
        return FullTokenMLMCollator, FullTokenMLMModel, corrupt, RECIPE
    raise ValueError("Unknown BERT objective: " + objective)


def for_model(model):
    from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyModel, OBJECTIVE as FAMILY_OBJECTIVE
    if isinstance(model, VFamilyModel):
        return components(FAMILY_OBJECTIVE, model.family_spec)
    from dllm.pipelines.immune_llada.fixed_chain.unbinding_vhh import VHHUnbindingModel, OBJECTIVE as VHH_OBJECTIVE
    if isinstance(model, VHHUnbindingModel):
        return components(VHH_OBJECTIVE)
    from dllm.pipelines.immune_llada.fixed_chain.unbinding_mlm import UnbindingMLMModel, OBJECTIVE
    if isinstance(model, UnbindingMLMModel):
        return components(OBJECTIVE)
    from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMModel
    return components("all_observed_tokens" if isinstance(model, FullTokenMLMModel) else "receptor_aa")
