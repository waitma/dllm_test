"""Activate pllm/protenix_abtcr; python -m pytest this_file.py -q.

Check preference direction, full conditioning equality and checkpointed gradients.
"""
import math
import pytest
import torch
from dllm.pipelines.immune_llada.fixed_chain.post_training_data import PostTrainingCollator
from dllm.pipelines.immune_llada.fixed_chain.preference import pair_features, preference_objective
from scripts.tests.immune_llada.test_fixed_chain_post_training import fixture


def pairs(model):
    rows = [dict(chains=['ACD', seq, 'CAVF'], chain_roles=['peptide', 'tcr_beta', 'tcr_alpha'],
                 task_type='tcr_epitope', source='test', split='train',
                 labels={'relation': 'binding'}, metadata={'relation_supervised': True})
            for seq in ['CASSF', 'CASTF']]
    return PostTrainingCollator(model.config)(rows)


@pytest.mark.parametrize('recipe', ['weak_preference', 'weak_preference_strong',
                                   'weak_preference_shuffle', 'positive_sft'])
def test_preference_recipes_checkpointed_backward_and_frozen_reference(recipe):
    from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
    parent, model, batch = fixture(recipe)
    parent.eval().requires_grad_(False)
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    original = batch['input_ids'].clone()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model(teacher=parent, preference_batch=pairs(model),
                     preference_directions=torch.tensor([-1]), **batch)['loss']
    loss.backward()
    grad = model.backbone.transformer.blocks[0].base.q_proj.weight.grad
    assert torch.isfinite(loss) and grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(p.grad is None for p in parent.parameters())
    assert torch.equal(original, batch['input_ids'])


def test_preference_gradient_improves_chosen_relative_to_rejected():
    parent, model, _ = fixture('weak_preference')
    parent.eval().requires_grad_(False)
    batch = pairs(model)
    features, _ = pair_features(model, batch)
    for value in features.values():
        assert torch.equal(value[0], value[1])
    loss = preference_objective(model, parent, batch)
    torch.testing.assert_close(loss, torch.tensor(math.log(2)), atol=1e-6, rtol=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    loss.backward()
    optimizer.step()
    assert preference_objective(model, parent, batch) < loss.detach()
    assert preference_objective(model, parent, batch, directions=torch.tensor([-1])) > loss.detach()


def test_preference_rejects_condition_changes_and_trainable_reference():
    parent, model, _ = fixture('weak_preference')
    batch = pairs(model)
    with pytest.raises(ValueError, match='frozen'):
        preference_objective(model, parent, batch)
    condition = (~batch['target_slot_mask'][0]).nonzero()[0, 0]
    batch['input_ids'][1, condition] += 1
    with pytest.raises(ValueError, match='condition differs'):
        pair_features(model, batch)


def test_positive_only_does_not_require_reference():
    _, model, _ = fixture('positive_sft')
    loss = preference_objective(model, None, pairs(model), positive_only=True)
    assert torch.isfinite(loss) and loss > 0


def test_multiseed_launcher_rejects_modified_child_plan(tmp_path):
    import hashlib
    import json
    from scripts.research.launch_fixed_chain_exploration import trial_specs
    child = tmp_path / 'seed42.json'
    child.write_text(json.dumps({'recipes': {'task_mix': {}}, 'seed': 42}))
    entry = {'id': 'task_mix_seed42', 'recipe': 'task_mix', 'plan': str(child),
             'sha256': hashlib.sha256(child.read_bytes()).hexdigest()}
    plan = {'trial_plans': [entry]}
    assert trial_specs(plan, tmp_path / 'parent.json') == [entry]
    child.write_text(child.read_text().replace('42', '43'))
    with pytest.raises(ValueError, match='binding changed'):
        trial_specs(plan, tmp_path / 'parent.json')
