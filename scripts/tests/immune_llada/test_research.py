"""Research contracts: run python -m pytest scripts/tests/immune_llada/test_research.py -q."""
import json
import torch
import pytest

from dllm.pipelines.immune_llada.research.model import ResearchModel, VARIANTS, vicreg_loss
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig, ImmuneModel
from dllm.pipelines.immune_llada.v3.layout import GenerationRequest, ImmuneCollator
from scripts.run_immune_research import toy_rows, collator_for


def tiny():
    return ImmuneConfig(ablation='A4', d_model=32, n_layers=1, n_heads=4, mlp_hidden_size=64,
                        max_length=512, full_capacity=24, fragment_capacity=16)


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_finite_training_and_serialization(variant, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = ResearchModel(tiny(), variant)
    batch = collator_for(tiny())(toy_rows()[:2])
    result = model(**batch)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if model.options.get('qk_norm'):
        assert model.backbone.transformer.blocks[0].q_norm.weight.grad.abs().sum() > 0
    for module in (model.registers, model.conv_weight, model.role_bias):
        if module is not None:
            assert module.grad is not None and module.grad.abs().sum() > 0
    if model.mask_film is not None:
        assert model.mask_film[-1].weight.grad.abs().sum() > 0
    if model.projector is not None:
        assert result['auxiliary_loss'] > 0 and model.projector[-1].weight.grad.abs().sum() > 0
    model.eval()
    expected = model.denoise(**model.feature_args(batch))[0]
    model.save_pretrained(tmp_path)
    restored = ResearchModel.from_pretrained(tmp_path).eval()
    assert torch.equal(expected, restored.denoise(**restored.feature_args(batch))[0])
    assert torch.equal(model(**batch)['loss'], model(**batch)['loss'])


def test_control_matches_existing_a4():
    torch.manual_seed(42)
    base = ImmuneModel(tiny())
    research = ResearchModel(tiny())
    research.load_state_dict(base.state_dict())
    batch = collator_for(tiny())(toy_rows()[:2])
    torch.manual_seed(3)
    expected = base(**batch)['loss']
    torch.manual_seed(3)
    actual = research(**batch)['loss']
    assert torch.allclose(expected, actual, atol=1e-6)


def test_local_conv_cannot_cross_chain_boundary():
    model = ResearchModel(tiny(), 'r06_local_conv7')
    with torch.no_grad():
        model.conv_weight.fill_(1)
    x = torch.randn(1, 8, 32)
    ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    original = model.local_mix(x, ids, torch.ones_like(ids), torch.ones_like(ids, dtype=torch.bool))
    changed = x.clone(); changed[:, 4:] += 100
    new = model.local_mix(changed, ids, torch.ones_like(ids), torch.ones_like(ids, dtype=torch.bool))
    assert torch.equal(original[:, :4], new[:, :4])


@pytest.mark.parametrize('variant', ['r01_span_mask', 'r02_chain_completion'])
def test_corruption_never_changes_condition_or_capacity_tail(variant):
    model = ResearchModel(tiny(), variant)
    batch = collator_for(tiny())(toy_rows()[4:])
    for seed in range(12):
        torch.manual_seed(seed)
        ids, labels, selected = model.corrupt(batch)
        assert not (selected & ~batch['eligible_mask']).any()
        assert torch.equal(ids[~selected], batch['input_ids'][~selected])
        assert torch.equal(labels[selected], batch['labels'][selected])
        assert selected.any(1).all()


@pytest.mark.parametrize('variant', ['r05_register4', 'r06_local_conv7', 'r07_mask_ratio_film', 'r10_role_pair_bias'])
def test_reference_free_generation(variant):
    model = ResearchModel(tiny(), variant).eval()
    collator = ImmuneCollator('A4', mode='generation', max_length=512, full_capacity=24, fragment_capacity=16)
    batch = collator([GenerationRequest(conditions=(), targets=(('tcr_beta', 'junction'),))])
    features = model.feature_args(batch)
    assert not {'labels', 'residue_mask', 'terminal_mask'} & set(features)
    assert model.denoise(**features)[0].shape[:2] == batch['input_ids'].shape
    generated = model.generate(batch, steps=2)
    assert len(generated['chains']) == 1


def test_vicreg_penalizes_collapsed_batch():
    zero = torch.zeros(4, 8)
    assert vicreg_loss(zero, zero) > .9
    with pytest.raises(ValueError):
        vicreg_loss(zero[:1], zero[:1])


def test_profile_rejects_changed_sources_and_memory_overflow(tmp_path, monkeypatch):
    from dataclasses import asdict
    import scripts.run_immune_research as runner
    monkeypatch.setattr(runner, 'fingerprints', lambda: {'test_source': 'current'})
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'fake-test-gpu')
    config = asdict(ImmuneConfig(ablation='A4'))
    measurement = dict(status='passed', batch_size=2, max_allocated_gib=40, total_gib=80,
                       precision='fp32_parameters_adam_bf16_autocast', config=config)
    summary = dict(config=config, gpu='fake-test-gpu', recommended_micro_batch=2, measurements=[measurement])
    def write_proof(source='current'):
        runner.write_json(tmp_path/'summary.json', summary)
        runner.write_json(tmp_path/'COMPLETE.json', dict(status='passed', variant='r00_a4_control',
            source_sha256={'test_source': source}, summary_sha256=runner.sha(tmp_path/'summary.json')))
    write_proof()
    assert runner.verify_profile(tmp_path, 'r00_a4_control') == 2
    write_proof('outdated')
    with pytest.raises(ValueError, match='Code changed'):
        runner.verify_profile(tmp_path, 'r00_a4_control')
    measurement['max_allocated_gib'] = 79
    write_proof()
    with pytest.raises(ValueError, match='Invalid measured'):
        runner.verify_profile(tmp_path, 'r00_a4_control')


def test_validation_masks_independent_of_batch_padding_and_rng():
    from scripts.run_immune_research import fixed_validation_mask
    collator = collator_for(tiny())
    rows = toy_rows()
    single = collator(rows[:1])
    mixed = collator([rows[4], rows[0]])
    previous = torch.zeros_like(single['eligible_mask'])
    for ratio in (.25, .5, .75, 1.):
        state = torch.get_rng_state().clone()
        mask = fixed_validation_mask(single, ratio)
        assert torch.equal(state, torch.get_rng_state())
        other = fixed_validation_mask(mixed, ratio)
        assert torch.equal(mask[0], other[1, :mask.shape[1]])
        assert not (mask.bool() & ~single['eligible_mask']).any()
        assert not (previous & ~mask.bool()).any()
        previous = mask.bool()
    assert torch.equal(previous, single['eligible_mask'])


def test_actual_trainer_checkpoint_resume_matches_uninterrupted(tmp_path):
    from transformers import TrainingArguments
    from examples.llada.immune_pretrain_v3 import PilotStop
    from scripts.run_immune_research import ResearchTrainer, validate_research_resume
    rows = toy_rows()[:4]
    def trainer_at(path, stop=0):
        torch.manual_seed(42)
        model = ResearchModel(tiny())
        arguments = TrainingArguments(output_dir=str(path), use_cpu=True, max_steps=4,
            per_device_train_batch_size=2, per_device_eval_batch_size=2, learning_rate=1e-4,
            warmup_steps=1, seed=42, data_seed=42, save_steps=2, eval_strategy='steps', eval_steps=2,
            report_to='none', disable_tqdm=True, remove_unused_columns=False, label_names=['labels'])
        trainer = ResearchTrainer(model=model, args=arguments, train_dataset=rows,
            eval_dataset={'toy': rows[:2]}, data_collator=collator_for(tiny()))
        trainer.run_provenance = dict(max_steps=4, pilot_stop_step=stop or 4, data='fixture', schedule='fixed')
        if stop:
            trainer.add_callback(PilotStop(stop))
        return trainer
    complete = trainer_at(tmp_path/'complete')
    complete.train()
    first = trainer_at(tmp_path/'first', stop=2)
    first.train()
    checkpoint = tmp_path/'first/checkpoint-2'
    resumed = trainer_at(tmp_path/'resumed')
    assert validate_research_resume(checkpoint, resumed.run_provenance, 4) == 2
    with pytest.raises(ValueError, match='provenance differs'):
        validate_research_resume(checkpoint, {**resumed.run_provenance, 'schedule': 'changed'}, 4)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 4
    for name, expected in complete.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], expected, atol=2e-6, rtol=0)
