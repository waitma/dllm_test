"""Run: USE_TF=0 python -B -m pytest scripts/tests/immune_llada/test_unbinding_mlm.py -q."""
import json

import pytest
import torch

from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator, FullTokenMLMModel
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import corrupt as old_corrupt
from dllm.pipelines.immune_llada.fixed_chain.unbinding_mlm import (
    UnbindingMLMCollator, UnbindingMLMModel, corrupt, OBJECTIVE)
from scripts.tests.immune_llada.test_full_token_mlm import config, record
from scripts.research.bert_mlm_objectives import components, for_model
from scripts.research.bert_mlm_eval import fixed_corruption


@pytest.mark.parametrize('synthetic', [False, True])
def test_negative_only_relation_with_all_sequences_and_grammar_clean(synthetic):
    collator = UnbindingMLMCollator(config())
    clean = collator([record('nonbinding', synthetic)] * 3)
    for seed in range(5):
        noisy = corrupt(clean, collator.tokenizer, torch.Generator().manual_seed(seed))
        selected = noisy['labels'].ne(-100)
        assert torch.equal(selected, clean['explicit_relation_mask'])
        assert selected.sum(1).tolist() == [1, 1, 1]
        assert not (selected & (clean['residue_mask'] | clean['terminal_mask'])).any()
        assert torch.equal(noisy['input_ids'][~selected], clean['input_ids'][~selected])
        assert noisy['input_ids'][selected].eq(collator.tokenizer.mask_token_id).all()
        assert noisy['labels'][selected].eq(collator.tokenizer.special_id('<nonbinding>')).all()


def test_binding_has_sequence_loss_and_always_hidden_relation():
    collator = UnbindingMLMCollator(config())
    clean = collator([record('binding')] * 64)
    noisy = corrupt(clean, collator.tokenizer, torch.Generator().manual_seed(1))
    relation = clean['explicit_relation_mask']
    assert (noisy['labels'].ne(-100) & clean['residue_mask']).any()
    assert noisy['input_ids'][relation].eq(collator.tokenizer.mask_token_id).all()
    assert noisy['labels'][relation].eq(collator.tokenizer.special_id('<binding>')).all()
    for key, value in UnbindingMLMModel.feature_args(noisy).items():
        assert key not in ('labels', 'eligible_mask', 'explicit_relation_mask')


def test_unknown_preserves_original_corruption_and_probe_hides_labels():
    rows = [record('unknown')] * 3
    a, b = UnbindingMLMCollator(config()), FullTokenMLMCollator(config())
    new = corrupt(a(rows), a.tokenizer, torch.Generator().manual_seed(8))
    old = old_corrupt(b(rows), b.tokenizer, torch.Generator().manual_seed(8))
    assert all(torch.equal(new[k], v) for k, v in old.items())
    blind = UnbindingMLMCollator(config(), hide_relation=True)([record(k) for k in ('binding', 'nonbinding', 'unknown')])
    for value in UnbindingMLMModel.feature_args(blind).values():
        assert torch.equal(value[0], value[1]) and torch.equal(value[0], value[2])
    assert not (blind['eligible_mask'] & blind['relation_mask']).any()


def test_mixed_batch_gradients_only_selected_logits_and_roundtrip(tmp_path):
    torch.manual_seed(42)
    model = UnbindingMLMModel(config())
    collator = UnbindingMLMCollator(model.config)
    clean = collator([record('nonbinding'), record('binding'), record('unknown')])
    batch = corrupt(clean, model.tokenizer, torch.Generator().manual_seed(12))
    result = model(**batch)
    result['logits'].retain_grad()
    result['loss_sum'].backward()
    grad = result['logits'].grad
    assert grad[0, clean['explicit_relation_mask'][0]].abs().sum() > 0
    assert not grad[0, ~clean['explicit_relation_mask'][0]].any()
    assert torch.isfinite(result['loss_sum'])
    model.save_pretrained(tmp_path)
    restored = UnbindingMLMModel.from_pretrained(tmp_path)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert for_model(restored) == components(OBJECTIVE)
    with pytest.raises(ValueError):
        FullTokenMLMModel.from_pretrained(tmp_path)
    settings = json.loads((tmp_path / 'research_config.json').read_text())
    settings['recipe']['relation_selection_probability'] = .15
    (tmp_path / 'research_config.json').write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        UnbindingMLMModel.from_pretrained(tmp_path)


def test_parent_weights_exact_and_fixed_validation_noise(tmp_path):
    parent = FullTokenMLMModel(config())
    parent.save_pretrained(tmp_path)
    model = UnbindingMLMModel.from_parent(tmp_path)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in parent.state_dict().items())
    collator = UnbindingMLMCollator(config())
    clean = collator([record('nonbinding'), record('binding')])
    batch = fixed_corruption(clean, model.tokenizer, [21, 22], corrupt_fn=corrupt)
    single = fixed_corruption(collator([record('nonbinding')]), model.tokenizer, [21], corrupt_fn=corrupt)
    assert torch.equal(batch['labels'][0], single['labels'][0])
    assert torch.equal(batch['input_ids'][0], single['input_ids'][0])


def test_real_representation_evaluator_hides_relation(monkeypatch):
    from scripts.research import bert_mlm_eval as evaluation
    model = UnbindingMLMModel(config())
    seen = []
    def vectors(batch):
        features = model.feature_args(batch)
        for value in features.values():
            assert torch.equal(value[0], value[1]) and torch.equal(value[0], value[2])
        seen.append(True)
        return torch.zeros(3, 4)
    monkeypatch.setattr(model, 'receptor_vectors', vectors)
    monkeypatch.setattr(evaluation, 'read_samples', lambda _: [dict(record=record(k))
                        for k in ('binding', 'nonbinding', 'unknown')])
    monkeypatch.setattr(evaluation, 'fit_relation_probes', lambda *args: {})
    evaluation.representation_validation(model, dict(files=dict(probe_train={}, probe_valid={})), torch.device('cpu'))
    assert len(seen) == 2


def test_smoke_indices_exercise_both_labels_on_every_rank():
    from scripts.research.prepare_bert_unbinding_v8 import smoke_indices
    index = dict(shards=[dict(split='train', path='/data/train/base.jsonl', records=5000),
                         dict(split='train', path='/data/train/unbind_x.jsonl', records=2000)])
    selected = smoke_indices(index, 1536)
    assert len(selected) == len(set(selected)) == 1536
    for rank in range(8):
        for start in (0, 512, 1024):
            records = selected[start:start+512][rank::8]
            assert sum(i >= 5000 for i in records) == 32
