"""Validate expanded diagnostic accounting and routing after activating pllm.

USE_TF=0 python -m pytest scripts/tests/immune_llada/test_refined_runner.py -q
Uses artificial sequences and temporary manifests; never enqueues GPU jobs.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import json

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts import run_immune_refined as runner
from scripts import run_immune_research as base
from scripts.architecture_prototypes.expanded_fixtures import write_fixture_manifest
from dllm.pipelines.immune_llada.research.model import ResearchModel
from dllm.pipelines.immune_llada.research_refined.model import RefinedModel, VARIANTS


@pytest.fixture(autouse=True)
def globals_restored():
    names = ('ResearchModel', 'VARIANTS', 'make_model', 'config_for', 'fingerprints',
             'profile_unlocked', 'measure', 'synthetic', 'train')
    saved = {name: getattr(base, name) for name in names}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    for name, value in saved.items():
        setattr(base, name, value)
    torch.set_num_threads(threads)


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_public_factory_builds_refined_model(variant):
    config = runner.config_for(SimpleNamespace(cpu_test=True))
    model = ResearchModel(config, variant)
    assert isinstance(model, RefinedModel)
    assert model.variant == variant
    assert model.options == VARIANTS[variant]


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_metrics_are_invariant_to_evaluation_batching(dtype):
    config = runner.config_for(SimpleNamespace(cpu_test=True))
    fixture = runner.build_fixtures(config, True)
    rows = [fixture.valid_records[i] for i in (0, 6, 12)]
    model = RefinedModel(config).eval()
    collate = base.collator_for(config)
    context = nullcontext() if dtype == torch.float32 else torch.autocast('cpu', dtype=dtype)
    with sdpa_kernel(SDPBackend.MATH), context:
        joint = runner.previous.measure(model, collate(rows))
        parts = [runner.previous.measure(model, collate([row])) for row in rows]
    combined = runner.combine_metrics(parts)
    for name, _ in runner.previous.MASK_LEVELS:
        for count in ('selected_count', 'aa_count', 'terminal_count'):
            assert joint['mask_levels'][name][count] == combined['mask_levels'][name][count]
        for key in runner.previous.METRIC_KEYS:
            assert joint['mask_levels'][name][key] == pytest.approx(combined['mask_levels'][name][key], abs=2e-6)
    assert joint['mask_mean'] == pytest.approx(combined['mask_mean'], abs=2e-6)


def test_configure_is_idempotent_and_keeps_real_route():
    runner.configure()
    runner.configure()
    assert base.ResearchModel is RefinedModel
    assert base.synthetic is runner.expanded_diagnostic
    assert base.train is runner.train
    assert base.config_for(SimpleNamespace(cpu_test=False)).d_model == 768
    assert base.config_for(SimpleNamespace(cpu_test=False)).n_layers == 8
    assert runner._REAL_TRAIN is not runner.train


def test_missing_artificial_manifest_uses_original_certified_route(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, '_REAL_TRAIN', lambda args: seen.append(args.data_dir))
    runner.train(SimpleNamespace(data_dir=tmp_path))
    assert seen == [tmp_path]


def test_artificial_manifest_tampering_rejected_before_gpu(tmp_path, monkeypatch):
    write_fixture_manifest(tmp_path, seed=42, train_size=576, valid_size=72,
        full_capacity=256, fragment_capacity=64, max_length=2048)
    path = tmp_path/'synthetic_fixture_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['unexpected_external_rows'] = ['forbidden']
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(base, 'verify_profile', lambda *args: pytest.fail('GPU/profile path must not be reached'))
    with pytest.raises(ValueError):
        runner.train(SimpleNamespace(data_dir=tmp_path, steps=500, resume_from_checkpoint=None))


@pytest.mark.parametrize('steps,resume', [(1001, None), (500, 'some-checkpoint')])
def test_artificial_budget_and_resume_are_explicit(tmp_path, steps, resume):
    (tmp_path/'synthetic_fixture_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='Artificial diagnostic'):
        runner.train(SimpleNamespace(data_dir=tmp_path, steps=steps, resume_from_checkpoint=resume))


def test_empty_evaluation_rejected():
    with pytest.raises(ValueError, match='Empty'):
        runner.combine_metrics([])
