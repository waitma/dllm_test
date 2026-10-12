"""Run CPU runner contracts after activating pllm.

USE_TF=0 python -m pytest scripts/tests/immune_llada/test_research_next_runner.py -q
Only artificial strings and temporary mock profile records are used; no jobs are submitted.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import run_immune_architecture as runner
from scripts import run_immune_research as base
from scripts.smoke_immune_v3_cpu import measure as legacy_measure
from dllm.pipelines.immune_llada.research_next.model import ArchitectureModel, VARIANTS
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def configured(monkeypatch):
    # configure is process-local in production; undo its globals between unit tests.
    for name in ('ResearchModel', 'VARIANTS', 'make_model', 'fingerprints', 'profile_unlocked', 'measure'):
        monkeypatch.setattr(base, name, getattr(base, name))
    runner.configure()
    return base


def tiny():
    return ImmuneConfig(ablation='A4', d_model=32, n_layers=2, n_heads=4,
                        mlp_hidden_size=64, max_length=512, full_capacity=24, fragment_capacity=16)


def test_fingerprints_include_original_and_every_new_source():
    original = runner._BASE_FINGERPRINTS()
    actual = runner.fingerprints()
    assert original.items() <= actual.items()
    files = [Path(runner.__file__).resolve()]
    files += list((runner.ROOT/'dllm/pipelines/immune_llada/research_next').rglob('*.py'))
    assert len(files) >= 3
    for path in files:
        assert actual[str(path.relative_to(runner.ROOT))] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_configure_is_idempotent_and_keeps_existing_guards(configured):
    guards = (base.main, base.profile, base.verify_profile, base.train, base.validate_research_resume)
    expected = runner.fingerprints()
    runner.configure()
    assert base.ResearchModel is ArchitectureModel and base.VARIANTS is VARIANTS
    assert base.make_model is runner.make_model
    assert base.measure is runner.measure and base.profile_unlocked is runner.profile_unlocked
    assert base.fingerprints() == expected
    assert guards == (base.main, base.profile, base.verify_profile, base.train, base.validate_research_resume)


def test_post_initialization_seed_aligns_noise_without_changing_shared_weights(configured):
    control = base.make_model(SimpleNamespace(cpu_test=True, variant='r12_architecture_control'))
    after_control = torch.rand(19)
    gated = base.make_model(SimpleNamespace(cpu_test=True, variant='r13_head_output_gate'))
    after_gated = torch.rand(19)
    expected = torch.rand(19, generator=torch.Generator().manual_seed(42))
    assert torch.equal(after_control, expected) and torch.equal(after_gated, expected)
    control_weights, gated_weights = control.state_dict(), gated.state_dict()
    assert control_weights.keys() < gated_weights.keys()
    extra = gated_weights.keys()-control_weights.keys()
    assert extra == {f'backbone.transformer.blocks.{i}.gate.weight' for i in range(control.config.n_layers)}
    for name, value in control_weights.items():
        assert torch.equal(value, gated_weights[name]), name
    for name in extra:
        assert gated_weights[name].shape == (gated.config.n_heads, gated.config.d_model)
        assert gated_weights[name].abs().sum() > 0


def test_profile_child_uses_new_runner_then_reuses_verified_proof(configured, monkeypatch, tmp_path):
    monkeypatch.setattr(base, 'fingerprints', lambda: {'source': 'fixed'})
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'mock-gpu')
    commands = []

    def subprocess_run(command, *, stdout, stderr, timeout):
        commands.append(command)
        assert Path(command[1]).resolve() == Path(runner.__file__).resolve()
        assert command[2:4] == ['--mode', 'profile-candidate']
        assert stderr == runner.subprocess.STDOUT and timeout == 900
        batch = int(command[command.index('--batch-size')+1])
        out = Path(command[command.index('--output-dir')+1])
        if batch == 2:
            stdout.write('CUDA out of memory: expected sweep stopping point\n')
            stdout.flush()
            return SimpleNamespace(returncode=1)
        assert batch == 1
        base.write_json(out/'measurement.json', dict(status='passed', batch_size=batch,
            max_allocated_gib=40, total_gib=80, gpu='mock-gpu',
            precision='fp32_parameters_adam_bf16_autocast', config=asdict(ImmuneConfig(ablation='A4'))))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, 'run', subprocess_run)
    args = SimpleNamespace(profile_dir=tmp_path/'profile', variant='r12_architecture_control')
    assert base.profile(args) == 1
    summary = json.loads((args.profile_dir/'summary.json').read_text())
    assert summary['global_batch'] == 256
    assert summary['measurements'][-1] == {'status': 'oom', 'batch_size': 2}
    assert (args.profile_dir/'profile.lock').is_file()
    assert base.verify_profile(args.profile_dir, args.variant) == 1
    assert base.profile(args) == 1 and len(commands) == 2
    monkeypatch.setattr(base, 'fingerprints', lambda: {'source': 'changed'})
    with pytest.raises(ValueError, match='Code changed'):
        base.profile(args)


def test_non_oom_profile_failure_is_not_certified(configured, monkeypatch, tmp_path):
    def subprocess_run(command, *, stdout, stderr, timeout):
        stdout.write('TypeError: unsupported architecture argument\n')
        stdout.flush()
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, 'run', subprocess_run)
    args = SimpleNamespace(profile_dir=tmp_path/'profile', variant='r13_head_output_gate')
    with pytest.raises(RuntimeError, match='Profile failed'):
        base.profile(args)
    assert not (args.profile_dir/'COMPLETE.json').exists()


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_four_mask_levels_and_full_mask_match_legacy_metric(variant):
    torch.manual_seed(11)
    model = ArchitectureModel(tiny(), variant).eval()
    batch = base.collator_for(tiny())(base.toy_rows())
    expected = legacy_measure(model, batch)
    actual = runner.measure(model, batch)
    for key in runner.METRIC_KEYS:
        assert actual[key] == pytest.approx(expected[key], abs=2e-6)
    assert set(actual['mask_levels']) == {'mask25', 'mask50', 'mask75', 'mask100'}
    previous_count = 0
    for name, ratio in runner.MASK_LEVELS:
        item = actual['mask_levels'][name]
        selected = base.fixed_validation_mask(batch, ratio).bool()
        assert item['selected_count'] == int(selected.sum())
        assert item['selected_count'] == item['aa_count']+item['terminal_count']
        assert previous_count <= item['selected_count']
        previous_count = item['selected_count']
        for key in runner.METRIC_KEYS:
            assert item[key] is None or torch.isfinite(torch.tensor(item[key]))
    full = actual['mask_levels']['mask100']
    assert full['aa_count'] == 82 and full['terminal_count'] == 11
    for key in runner.METRIC_KEYS:
        available = [item[key] for item in actual['mask_levels'].values() if item[key] is not None]
        assert actual['mask_mean'][key] == pytest.approx(sum(available)/len(available))
    assert model.training is False
    assert all(parameter.grad is None for parameter in model.parameters())


class VisibleCopyModel(nn.Module):
    """A fixture that copies visible tokens, fails masked ones, and consumes RNG."""
    feature_args = staticmethod(ArchitectureModel.feature_args)

    def __init__(self):
        super().__init__()
        reference = ArchitectureModel(tiny())
        self.tokenizer = reference.tokenizer
        self.vocab_size = reference.tokenizer.vocab_size
        self.calls = []

    def denoise(self, **features):
        assert set(features) == {'input_ids', 'attention_mask', 'role_ids', 'scope_ids',
                                 'chain_ids', 'position_ids', 'target_slot_mask'}
        self.calls.append({key: value.clone() for key, value in features.items()})
        torch.rand(17)  # The evaluation wrapper must preserve training RNG despite this.
        ids = features['input_ids']
        logits = torch.full((*ids.shape, self.vocab_size), -4., device=ids.device)
        logits.scatter_(-1, ids[..., None], 4.)
        return logits, None


def test_accuracy_excludes_visible_tokens_and_preserves_features_rng_and_mode():
    model = VisibleCopyModel().train()
    batch = base.collator_for(tiny())(base.toy_rows())
    original = {key: value.clone() for key, value in batch.items()}
    before = torch.get_rng_state().clone()
    result = runner.measure(model, batch)
    assert torch.equal(before, torch.get_rng_state()) and model.training
    assert len(model.calls) == 4
    previous = torch.zeros_like(batch['eligible_mask'], dtype=torch.bool)
    for (name, ratio), features in zip(runner.MASK_LEVELS, model.calls):
        selected = base.fixed_validation_mask(batch, ratio).bool()
        assert not (previous & ~selected).any()
        previous = selected
        assert not (selected & ~batch['eligible_mask']).any()
        assert torch.equal(features['input_ids'][~selected], batch['input_ids'][~selected])
        assert features['input_ids'][selected].eq(model.tokenizer.mask_token_id).all()
        for key in features.keys()-{'input_ids'}:
            assert torch.equal(features[key], batch[key])
        assert result['mask_levels'][name]['aa_accuracy'] == 0.
        assert result['mask_levels'][name]['terminal_accuracy'] in (None, 0.)
    for key in batch:
        assert torch.equal(batch[key], original[key])
    # Repeating evaluation is exact and leaves no hidden mask or RNG state behind.
    assert runner.measure(model, batch) == result


def test_empty_terminal_subset_is_reported_as_null_not_nan():
    model = VisibleCopyModel().eval()
    batch = base.collator_for(tiny())(base.toy_rows())
    batch['eligible_mask'] &= batch['residue_mask']
    result = runner.measure(model, batch)
    assert result['terminal_loss'] is None and result['terminal_accuracy'] is None
    assert result['mask_mean']['terminal_accuracy'] is None
    json.dumps(result, allow_nan=False)


def test_failed_evaluation_restores_rng_and_training_mode(monkeypatch):
    model = VisibleCopyModel().train()
    batch = base.collator_for(tiny())(base.toy_rows())

    def fail(**features):
        torch.rand(19)
        raise RuntimeError('fixture failure')

    monkeypatch.setattr(model, 'denoise', fail)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='fixture failure'):
        runner.measure(model, batch)
    assert torch.equal(before, torch.get_rng_state()) and model.training


def test_selected_inconsistent_labels_are_rejected():
    model = VisibleCopyModel().eval()
    batch = base.collator_for(tiny())(base.toy_rows())
    first = base.fixed_validation_mask(batch, .25).bool().nonzero()[0]
    batch['labels'][first[0], first[1]] = model.tokenizer.mask_token_id
    with pytest.raises(ValueError, match='Selected labels'):
        runner.measure(model, batch)
    assert not model.calls
