"""Refined enqueue gates with temporary mock evidence; no real data or jobs.

After activating pllm:
USE_TF=0 python -m pytest scripts/tests/immune_llada/test_refined_enqueue.py -q
All passing-looking JSON is explicitly mock-only and lives under pytest tmp_path.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from scripts import enqueue_immune_refined as helper
from scripts import immune_research_pool as pool_module
from scripts.architecture_prototypes import expanded_fixtures as fixtures


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(loss=3., aa=.05, empty_terminal=False):
    levels = {}
    for index, mask in enumerate(('mask25', 'mask50', 'mask75', 'mask100'), 1):
        terminal = 0 if empty_terminal and index == 1 else index
        levels[mask] = dict(mask_ratio=index/4, aa_count=20*index,
            terminal_count=terminal, selected_count=20*index+terminal,
            loss=loss, aa_loss=loss, aa_accuracy=aa,
            terminal_loss=loss if terminal else None, terminal_accuracy=.2 if terminal else None)
    names = ('loss', 'aa_loss', 'aa_accuracy', 'terminal_loss', 'terminal_accuracy')
    mean = {name: sum(v[name] for v in levels.values() if v[name] is not None)/
            sum(v[name] is not None for v in levels.values()) for name in names}
    return dict(mask_levels=levels, mask_mean=mean,
                mask_protocol='fixed_per_record_selected_tokens_v1')


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    pool, variant = tmp_path/'pool', 'r20_refined_control'
    for folder in ('profiles', 'runs', 'results', 'pending', 'running', 'finished'):
        (pool/folder).mkdir(parents=True)
    sources = {'mock_only_source.py': 'a'*64}
    protocol = dict(schema='mock_only_fixture', mock_only=True, splits={'valid': dict(records=72,
        group_length_counts={f'{group}/{length}': 4 for group in fixtures.GROUPS for length in fixtures.LENGTH_BINS})})
    monkeypatch.setattr(helper, 'fingerprints', lambda: sources)
    monkeypatch.setattr(helper, 'build_fixtures', lambda *_: SimpleNamespace(manifest=protocol))
    registered = dict(pool_module.variants())
    registered.update(helper.VARIANTS)
    monkeypatch.setattr(pool_module, 'variants', lambda: registered)
    synthetic = tmp_path/'mock_artificial'
    write(synthetic/fixtures.MANIFEST_NAME, protocol)
    manifest_reads = []
    def mock_load(directory):
        manifest_reads.append(Path(directory))
        actual = helper.read_json(Path(directory)/fixtures.MANIFEST_NAME)
        if actual != protocol:
            raise ValueError('Mock fixture mismatch')
        return SimpleNamespace(manifest=actual)
    monkeypatch.setattr(fixtures, 'load_verified_fixture_manifest', mock_load)
    data_checks = []
    monkeypatch.setattr(helper, 'assert_data_ready', lambda path: data_checks.append(path))
    config = asdict(helper.ImmuneConfig())
    profile = dict(mock_only=True, variant=variant, config=config, global_batch=256,
        gpu='mock_A100', recommended_micro_batch=16,
        measurements=[dict(status='passed', batch_size=16, config=config, gpu='mock_A100',
            precision='fp32_parameters_adam_bf16_autocast', max_allocated_gib=40., total_gib=80.)])
    profile_dir = pool/'profiles'/variant
    write(profile_dir/'summary.json', profile)
    profile_proof = dict(mock_only=True, status='passed', variant=variant,
                         source_sha256=sources, summary_sha256=sha(profile_dir/'summary.json'))
    write(profile_dir/'COMPLETE.json', profile_proof)
    case = SimpleNamespace(pool=pool, variant=variant, sources=sources, protocol=protocol,
        synthetic=synthetic, real=tmp_path/'mock_real', config=config, profile=profile,
        profile_proof=profile_proof, profile_dir=profile_dir,
        data_checks=data_checks, manifest_reads=manifest_reads)
    return case


def evaluation(case, loss=3., aa=.05, empty_terminal=False):
    part = metrics(loss, aa, empty_terminal)
    strata = {key: deepcopy(part) for key in case.protocol['splits']['valid']['group_length_counts']}
    point = deepcopy(part)
    for level in point['mask_levels'].values():
        for key in ('selected_count', 'aa_count', 'terminal_count'):
            level[key] *= len(strata)
    return dict(point, strata=strata, records=72)


def diagnostic(case, steps=25, learning=True, run_id=None, empty_terminal=False):
    run_id = run_id or f'mock_diagnostic_{steps}'
    job = dict(variant=case.variant, mode='profile-synthetic' if steps == 25 else 'train', steps=steps)
    if steps == 500:
        job['data_dir'] = str(case.synthetic)
    result = dict(mock_only=True, job_id=run_id, job=job, state='succeeded', returncode=0)
    initial = evaluation(case, empty_terminal=empty_terminal)
    final = evaluation(case, 2. if learning else 3.1, .15 if learning else .05, empty_terminal)
    report = dict(mock_only=True, status='completed', variant=case.variant,
        kind='synthetic_generalization_diagnostic', dataset_kind='deterministic_artificial_sequences',
        biological_quality_claim=False, validation_uses_test=False, source_sha256=case.sources,
        config=case.config, fixture_manifest=case.protocol, global_batch=256, micro_batch=16,
        grad_accum=16, steps=steps, initial=initial, final=final,
        trace=[dict(step=step, gradient_norm=.2, auxiliary_loss=0., **deepcopy(final))
               for step in range(25, steps+1, 25)])
    proof = dict(mock_only=True, status='completed', global_step=steps,
        kind=report['kind'], biological_quality_claim=False)
    item = SimpleNamespace(run=case.pool/'runs'/run_id, result_path=case.pool/'results'/f'{run_id}.json',
                           report=report, proof=proof, result=result)
    save_diagnostic(item)
    return item


def save_diagnostic(item):
    write(item.run/'summary.json', item.report)
    item.proof['summary_sha256'] = sha(item.run/'summary.json')
    write(item.run/'COMPLETE.json', item.proof)
    write(item.result_path, item.result)


def job(case, stage, variant=None):
    result = dict(variant=variant or case.variant,
                  mode='profile-synthetic' if stage == 'profile_diagnostic25' else 'train',
                  steps={'profile_diagnostic25': 25, 'synthetic500': 500, 'real1000': 1000}[stage])
    if stage != 'profile_diagnostic25':
        result['data_dir'] = str(case.synthetic if stage == 'synthetic500' else case.real)
    return result


@pytest.mark.parametrize('variant', sorted(helper.VARIANTS))
def test_only_refined_family_can_enter_first_stage_without_prior_proof(evidence, variant):
    result = helper.enqueue_jobs(evidence.pool, [job(evidence, 'profile_diagnostic25', variant)], 'profile_diagnostic25')
    assert result[0]['decision'] == 'validated_not_enqueued'
    assert not evidence.data_checks and not evidence.manifest_reads
    assert not (evidence.pool/'refined_enqueue.lock').exists()


@pytest.mark.parametrize('variant', ['r00_a4_control', 'r12_architecture_control', 'unregistered'])
def test_other_families_rejected(evidence, variant):
    with pytest.raises(ValueError):
        helper.validate_job(evidence.pool, job(evidence, 'profile_diagnostic25', variant), 'profile_diagnostic25')


def test_25_step_execution_can_authorize_500_without_claiming_learning(evidence):
    diagnostic(evidence, learning=False)
    helper.validate_job(evidence.pool, job(evidence, 'synthetic500'), 'synthetic500')
    assert evidence.manifest_reads == [evidence.synthetic]
    assert not evidence.data_checks


def test_verified_500_learning_and_real_data_gate_both_precede_real_pilot(evidence):
    diagnostic(evidence, 500, empty_terminal=True)
    helper.validate_job(evidence.pool, job(evidence, 'real1000'), 'real1000')
    assert evidence.data_checks == [evidence.real]
    assert evidence.manifest_reads == [evidence.synthetic]


@pytest.mark.parametrize('message', ['missing audit certificate', 'missing downstream scope certificate'])
def test_either_real_data_certificate_failure_stops_before_diagnostic(evidence, monkeypatch, message):
    def fail_data(_):
        raise ValueError(message)
    monkeypatch.setattr(helper, 'assert_data_ready', fail_data)
    monkeypatch.setattr(helper, 'verify_diagnostic', lambda *_args, **_kwargs: pytest.fail('Read diagnostic before data certificates'))
    with pytest.raises(ValueError, match=message):
        helper.validate_job(evidence.pool, job(evidence, 'real1000'), 'real1000')


def test_artificial_data_cannot_be_called_real_even_if_500_learning_exists(evidence):
    diagnostic(evidence, 500)
    wrong = dict(job(evidence, 'real1000'), data_dir=str(evidence.synthetic))
    with pytest.raises(ValueError, match='Artificial data'):
        helper.validate_job(evidence.pool, wrong, 'real1000')
    assert not evidence.data_checks


@pytest.mark.parametrize('stage,change', [('profile_diagnostic25', {'steps': 100}),
    ('profile_diagnostic25', {'mode': 'train', 'data_dir': '/mock'}),
    ('synthetic500', {'steps': 25}), ('synthetic500', {'resume_from_checkpoint': '/mock/checkpoint'}),
    ('real1000', {'steps': 500}), ('real1000', {'resume_from_checkpoint': '/mock/checkpoint'})])
def test_stage_modes_and_budgets_cannot_be_substituted(evidence, stage, change):
    with pytest.raises(ValueError):
        helper.validate_job(evidence.pool, dict(job(evidence, stage), **change), stage)


@pytest.mark.parametrize('defect', ['missing_proof', 'sha', 'source', 'missing_source', 'variant',
    'config', 'gpu', 'precision', 'memory', 'zero_total', 'missing_measurement', 'duplicate', 'nan'])
def test_invalid_profile_proofs_rejected(evidence, defect):
    c = evidence
    if defect == 'missing_proof':
        (c.profile_dir/'COMPLETE.json').unlink()
    else:
        if defect == 'source': c.profile_proof['source_sha256'] = {'stale.py': 'b'*64}
        elif defect == 'missing_source': c.profile_proof['source_sha256'] = {}
        elif defect == 'variant': c.profile_proof['variant'] = 'r21_refined_head_gate'
        elif defect == 'config': c.profile['config'] = dict(c.config, d_model=32)
        elif defect == 'gpu': c.profile['measurements'][0]['gpu'] = 'different'
        elif defect == 'precision': c.profile['measurements'][0]['precision'] = 'different'
        elif defect == 'memory': c.profile['measurements'][0]['max_allocated_gib'] = 79.
        elif defect == 'zero_total': c.profile['measurements'][0]['total_gib'] = 0.
        elif defect == 'missing_measurement': c.profile['measurements'] = []
        elif defect == 'duplicate': c.profile['measurements'] *= 2
        elif defect == 'nan': c.profile['measurements'][0]['max_allocated_gib'] = float('nan')
        write(c.profile_dir/'summary.json', c.profile)
        c.profile_proof['summary_sha256'] = 'f'*64 if defect == 'sha' else sha(c.profile_dir/'summary.json')
        write(c.profile_dir/'COMPLETE.json', c.profile_proof)
    with pytest.raises(ValueError):
        helper.verify_profile_evidence(c.pool, c.variant)


@pytest.mark.parametrize('defect', ['missing_complete', 'missing_summary', 'summary_sha', 'source',
    'variant', 'config', 'kind', 'biological', 'test_used', 'fixture', 'batch', 'accumulation',
    'mode', 'job_id', 'resume', 'wrong_directory', 'short_trace', 'trace_hole', 'gradient_missing',
    'gradient_inf', 'stratum_nan', 'heldout_count', 'missing_stratum', 'mask_population',
    'aggregate', 'final_mismatch', 'terminal_accuracy_range'])
def test_500_proof_is_bound_to_mode_source_complete_trace_and_heldout(evidence, defect):
    item = diagnostic(evidence, 500)
    r = item.report
    if defect == 'source': r['source_sha256'] = {'stale.py': 'b'*64}
    elif defect == 'variant': r['variant'] = 'r21_refined_head_gate'
    elif defect == 'config': r['config'] = dict(evidence.config, n_layers=1)
    elif defect == 'kind': r['kind'] = 'real_training'
    elif defect == 'biological': r['biological_quality_claim'] = True
    elif defect == 'test_used': r['validation_uses_test'] = True
    elif defect == 'fixture': r['fixture_manifest'] = {'external_records': 'untrusted'}
    elif defect == 'batch': r['global_batch'] = 16
    elif defect == 'accumulation': r['grad_accum'] = 4
    elif defect == 'mode': item.result['job']['mode'] = 'profile-synthetic'
    elif defect == 'job_id': item.result['job_id'] = '../unrelated'
    elif defect == 'resume': item.result['job']['resume_from_checkpoint'] = '/mock/checkpoint'
    elif defect == 'wrong_directory': item.result['job']['data_dir'] = str(evidence.real)
    elif defect == 'short_trace': r['trace'].pop()
    elif defect == 'trace_hole': del r['trace'][4]
    elif defect == 'gradient_missing': del r['trace'][3]['gradient_norm']
    elif defect == 'gradient_inf': r['trace'][3]['gradient_norm'] = float('inf')
    elif defect == 'stratum_nan': r['trace'][4]['strata']['ab_antigen/short']['mask_mean']['aa_loss'] = float('nan')
    elif defect == 'heldout_count': r['final']['records'] = 576
    elif defect == 'missing_stratum': del r['trace'][3]['strata']['unconditional_ab/long']
    elif defect == 'mask_population':
        level = r['trace'][3]['strata']['tcr_pmhc/long']['mask_levels']['mask50']
        level['aa_count'] += 1; level['selected_count'] += 1
    elif defect == 'aggregate':
        for level in r['initial']['mask_levels'].values(): level['loss'] = 4.
        r['initial']['mask_mean']['loss'] = 4.
    elif defect == 'final_mismatch':
        r['final'] = evaluation(evidence, 1., .3)
    elif defect == 'terminal_accuracy_range':
        r['initial']['strata']['ab_antigen/short']['mask_levels']['mask25']['terminal_accuracy'] = 1.1
    save_diagnostic(item)
    if defect == 'missing_complete': (item.run/'COMPLETE.json').unlink()
    elif defect == 'missing_summary': (item.run/'summary.json').unlink()
    elif defect == 'summary_sha':
        item.proof['summary_sha256'] = 'f'*64
        write(item.run/'COMPLETE.json', item.proof)
    with pytest.raises(ValueError):
        helper.verify_diagnostic(evidence.pool, evidence.variant, 500, require_learning=True)


@pytest.mark.parametrize('loss,aa', [(3., .2), (4., .2), (2., .05)])
def test_real_stage_requires_both_loss_and_aa_improvement(evidence, loss, aa):
    item = diagnostic(evidence, 500)
    item.report['final'] = evaluation(evidence, loss, aa)
    item.report['trace'] = [dict(step=step, gradient_norm=.2, **deepcopy(item.report['final'])) for step in range(25, 501, 25)]
    save_diagnostic(item)
    with pytest.raises(ValueError, match='shown learning'):
        helper.verify_diagnostic(evidence.pool, evidence.variant, 500, require_learning=True)


def test_new_invalid_attempt_does_not_hide_an_older_valid_rerun(evidence):
    diagnostic(evidence, run_id='aaa_good')
    bad = diagnostic(evidence, run_id='zzz_bad')
    (bad.run/'COMPLETE.json').unlink()
    assert helper.verify_diagnostic(evidence.pool, evidence.variant, 25)['steps'] == 25


def test_missing_25_proof_blocks_500_even_with_verified_fixture(evidence):
    with pytest.raises(ValueError, match='No completed 25'):
        helper.validate_job(evidence.pool, job(evidence, 'synthetic500'), 'synthetic500')


def test_entire_stage_checked_before_mock_publication(evidence, monkeypatch):
    first = job(evidence, 'profile_diagnostic25')
    invalid = dict(first, steps=20)
    monkeypatch.setattr(helper, 'enqueue', lambda *_: pytest.fail('Partially published stage'))
    with pytest.raises(ValueError):
        helper.enqueue_jobs(evidence.pool, [first, invalid], 'profile_diagnostic25', apply=True)


@pytest.mark.parametrize('folder', ['pending', 'running', 'finished'])
def test_registered_job_is_not_republished(evidence, monkeypatch, folder):
    spec = job(evidence, 'profile_diagnostic25')
    write(evidence.pool/folder/'mock_existing.json', spec)
    monkeypatch.setattr(helper, 'enqueue', lambda *_: pytest.fail('Duplicate publication'))
    result = helper.enqueue_jobs(evidence.pool, [spec], 'profile_diagnostic25', apply=True)
    assert result[0]['decision'] == 'already_registered'


def test_dry_run_duplicate_plan_has_no_lock_or_queue_side_effect(evidence):
    spec = job(evidence, 'profile_diagnostic25')
    result = helper.enqueue_jobs(evidence.pool, [spec, deepcopy(spec)], 'profile_diagnostic25')
    assert [r['decision'] for r in result] == ['validated_not_enqueued', 'duplicate_in_plan']
    assert not (evidence.pool/'refined_enqueue.lock').exists()
    assert not list((evidence.pool/'pending').iterdir())


def test_worker_lifecycle_rename_is_still_detected_as_duplicate(evidence, monkeypatch):
    spec = job(evidence, 'profile_diagnostic25')
    source = evidence.pool/'pending'/'mock_existing.json'
    write(source, spec)
    original = helper.read_json
    def moving_read(path):
        if path == source and path.exists():
            path.rename(evidence.pool/'running'/path.name)
        return original(path)
    monkeypatch.setattr(helper, 'read_json', moving_read)
    result = helper.enqueue_jobs(evidence.pool, [spec], 'profile_diagnostic25')
    assert result[0]['decision'] == 'already_registered'


def test_two_helpers_mock_publish_once_under_shared_lock(evidence, monkeypatch):
    published = []
    def mock_enqueue(pool, spec):
        time.sleep(.02)
        published.append(spec)
        write(pool/'pending'/'mock_published.json', spec)
        return 'mock_published'
    monkeypatch.setattr(helper, 'enqueue', mock_enqueue)
    spec = job(evidence, 'profile_diagnostic25')
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: helper.enqueue_jobs(evidence.pool,
            [deepcopy(spec)], 'profile_diagnostic25', apply=True), range(2)))
    assert len(published) == 1
    assert sorted(r[0]['decision'] for r in results) == ['already_registered', 'enqueued']
