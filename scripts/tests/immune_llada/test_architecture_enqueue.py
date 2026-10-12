"""CPU proof-gate tests; no GPU jobs or real dataset reads.

After activating pllm:
USE_TF=0 python -m pytest scripts/tests/immune_llada/test_architecture_enqueue.py -q
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from scripts import enqueue_immune_architecture as helper


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def metrics(loss, accuracy):
    return dict(mask_levels={name: dict(loss=loss, aa_accuracy=accuracy,
                terminal_accuracy=.2, aa_count=10, terminal_count=2,
                selected_count=12, mask_ratio=index/4)
                for index, name in enumerate(helper.MASKS, 1)}, mask_mean=dict(loss=loss),
                mask_protocol='fixed_per_record_selected_tokens_v1')


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    pool, variant, run_id = tmp_path/'pool', 'r13_head_output_gate', 'diagnostic100'
    for folder in ('pending', 'running', 'finished', 'results'):
        (pool/folder).mkdir(parents=True)
    profile, run = pool/'profiles'/variant, pool/'runs'/run_id
    sources = {'model.py': 'a'*64, 'runner.py': 'b'*64}
    measured = dict(variant=variant, config={'d_model': 768, 'n_layers': 8},
                    recommended_micro_batch=16, global_batch=256)
    measured['measurements'] = [dict(status='passed', batch_size=16,
        config=measured['config'], precision='fp32_parameters_adam_bf16_autocast',
        max_allocated_gib=50., total_gib=80.)]
    write(profile/'summary.json', measured)
    certificate = dict(status='passed', variant=variant, source_sha256=sources,
        summary_sha256=hashlib.sha256((profile/'summary.json').read_bytes()).hexdigest())
    write(profile/'COMPLETE.json', certificate)
    initial, final = metrics(3., .05), metrics(2., .05)
    # Only one noise level improves AA; 100% endpoint accuracy is not required.
    final['mask_levels']['mask25']['aa_accuracy'] = .15
    report = dict(status='completed', variant=variant, steps=100,
        kind='synthetic_functional_diagnostic', biological_quality_claim=False,
        source_sha256=sources, config=measured['config'], micro_batch=16,
        grad_accum=16, global_batch=256, initial=initial, final=final,
        trace=[dict(step=step, gradient_norm=.3, **deepcopy(final)) for step in range(5, 101, 5)])
    write(run/'summary.json', report)
    write(run/'COMPLETE.json', dict(status='completed', biological_quality_claim=False))
    result = dict(job_id=run_id, state='succeeded', returncode=0,
                  job=dict(variant=variant, mode='profile-synthetic', steps=100))
    write(pool/'results'/f'{run_id}.json', result)
    monkeypatch.setattr(helper, 'current_fingerprints', lambda: sources)
    expected_config = deepcopy(measured['config'])
    monkeypatch.setattr(helper, 'production_config', lambda: expected_config)
    data_checks = []
    monkeypatch.setattr(helper, 'assert_data_ready', lambda path: data_checks.append(path))
    job = dict(variant=variant, mode='train', steps=1000, data_dir='/mock/certified/data')
    return SimpleNamespace(pool=pool, variant=variant, run_id=run_id, profile=profile,
        run=run, report=report, result=result, certificate=certificate, measured=measured,
        sources=sources, job=job, data_checks=data_checks)


def save_report(case):
    write(case.run/'summary.json', case.report)


def test_valid_partial_recovery_passes_without_full_accuracy_or_fullmask_loss_rule(evidence):
    case = evidence
    # Full-mask loss can worsen while the four-mask mean improves.
    for point in [case.report['final'], *case.report['trace']]:
        point['mask_levels']['mask100']['loss'] = 4.
        point['mask_mean']['loss'] = 2.5
    case.report['loss_decreased'] = False
    save_report(case)
    assert helper.verify_diagnostic(case.pool, case.variant) == case.run_id
    decisions = helper.enqueue_jobs(case.pool, [case.job])
    assert decisions[0]['decision'] == 'validated_not_enqueued'
    assert case.data_checks == [Path(case.job['data_dir'])]
    assert not list((case.pool/'pending').iterdir())
    assert not (case.pool/'architecture_enqueue.lock').exists()


@pytest.mark.parametrize('missing', ['profile_COMPLETE', 'profile_summary', 'run_COMPLETE', 'run_summary', 'result'])
def test_missing_evidence_rejects(evidence, missing):
    paths = {'profile_COMPLETE': evidence.profile/'COMPLETE.json',
             'profile_summary': evidence.profile/'summary.json',
             'run_COMPLETE': evidence.run/'COMPLETE.json',
             'run_summary': evidence.run/'summary.json',
             'result': evidence.pool/'results'/f'{evidence.run_id}.json'}
    paths[missing].unlink()
    with pytest.raises(ValueError):
        helper.verify_diagnostic(evidence.pool, evidence.variant)


@pytest.mark.parametrize('defect', ['profile_sha', 'profile_source_stale', 'profile_source_incomplete',
    'profile_variant', 'profile_failed', 'run_source', 'run_variant', 'run_incomplete',
    'complete_incomplete', 'result_variant', 'result_failed', 'result_id', 'result_exit',
    'declared_short', 'trace_short', 'trace_hole', 'tiny_config', 'wrong_microbatch',
    'wrong_accumulation', 'small_global_batch', 'missing_mask', 'forged_mean',
    'flat_loss', 'worse_loss', 'no_aa_improvement', 'inconsistent_final'])
def test_incomplete_stale_or_unimproved_proofs_cannot_promote(evidence, defect):
    c = evidence
    if defect == 'profile_sha':
        c.certificate['summary_sha256'] = 'f'*64
    elif defect == 'profile_source_stale':
        c.certificate['source_sha256'] = dict(c.sources, **{'model.py': 'f'*64})
    elif defect == 'profile_source_incomplete':
        c.certificate['source_sha256'] = {'model.py': c.sources['model.py']}
    elif defect == 'profile_variant':
        c.certificate['variant'] = 'r12_architecture_control'
    elif defect == 'profile_failed':
        c.certificate['status'] = 'failed'
    elif defect == 'run_source':
        c.report['source_sha256'] = {'old.py': 'a'*64}
    elif defect == 'run_variant':
        c.report['variant'] = 'r12_architecture_control'
    elif defect == 'run_incomplete':
        c.report.pop('status')
    elif defect == 'complete_incomplete':
        write(c.run/'COMPLETE.json', {'status': 'running'})
    elif defect == 'result_variant':
        c.result['job']['variant'] = 'r12_architecture_control'
    elif defect == 'result_failed':
        c.result['state'] = 'failed'
    elif defect == 'result_id':
        c.result['job_id'] = '../another_run'
    elif defect == 'result_exit':
        c.result['returncode'] = 1
    elif defect == 'declared_short':
        c.report['steps'] = 20
    elif defect == 'trace_short':
        c.report['trace'].pop()
    elif defect == 'trace_hole':
        del c.report['trace'][3]
    elif defect == 'tiny_config':
        c.report['config'] = {'d_model': 32, 'n_layers': 2}
    elif defect == 'wrong_microbatch':
        c.report['micro_batch'] = 8
        c.report['grad_accum'] = 32
    elif defect == 'wrong_accumulation':
        c.report['grad_accum'] = 8
    elif defect == 'small_global_batch':
        c.report['global_batch'] = 16
    elif defect == 'missing_mask':
        del c.report['initial']['mask_levels']['mask75']
    elif defect == 'forged_mean':
        c.report['initial']['mask_mean']['loss'] = 30.
    elif defect in ('flat_loss', 'worse_loss'):
        loss = 3. if defect == 'flat_loss' else 4.
        for point in [c.report['final'], *c.report['trace']]:
            point['mask_mean']['loss'] = loss
            for level in point['mask_levels'].values():
                level['loss'] = loss
    elif defect == 'no_aa_improvement':
        for point in [c.report['final'], *c.report['trace']]:
            point['mask_levels']['mask25']['aa_accuracy'] = .05
    elif defect == 'inconsistent_final':
        c.report['final']['mask_levels']['mask25']['aa_accuracy'] = .9
    write(c.profile/'COMPLETE.json', c.certificate)
    write(c.pool/'results'/f'{c.run_id}.json', c.result)
    save_report(c)
    with pytest.raises(ValueError):
        helper.verify_diagnostic(c.pool, c.variant)


@pytest.mark.parametrize('where,value', [('initial_loss', float('nan')),
    ('trace_gradient', float('inf')), ('trace_auxiliary', float('-inf')),
    ('final_accuracy', float('nan')), ('profile_measurement', float('inf'))])
def test_nonfinite_values_anywhere_in_proof_reject(evidence, where, value):
    c = evidence
    if where == 'initial_loss':
        c.report['initial']['mask_levels']['mask25']['loss'] = value
    elif where == 'trace_gradient':
        c.report['trace'][7]['gradient_norm'] = value
    elif where == 'trace_auxiliary':
        c.report['trace'][9]['auxiliary_loss'] = value
    elif where == 'final_accuracy':
        c.report['final']['mask_levels']['mask100']['aa_accuracy'] = value
    else:
        c.measured['measurements'] = [{'max_allocated_gib': value}]
        write(c.profile/'summary.json', c.measured)
    save_report(c)
    with pytest.raises(ValueError, match='Nonfinite'):
        helper.verify_diagnostic(c.pool, c.variant)


def test_json_numeric_overflow_is_nonfinite(evidence):
    path = evidence.run/'summary.json'
    path.write_text(path.read_text().replace('"gradient_norm": 0.3', '"gradient_norm": 1e999', 1))
    with pytest.raises(ValueError, match='Nonfinite'):
        helper.verify_diagnostic(evidence.pool, evidence.variant)


@pytest.mark.parametrize('defect', ['production_config', 'measurement_missing', 'measurement_duplicate',
    'precision', 'memory', 'zero_memory_total', 'nondivisible_batch', 'global_batch', 'variant'])
def test_profile_sha_alone_does_not_certify_invalid_measurement(evidence, defect):
    c = evidence
    measured = deepcopy(c.measured)
    if defect == 'production_config':
        measured['config']['d_model'] = 32
        c.report['config'] = measured['config']
        save_report(c)
    elif defect == 'measurement_missing':
        measured['measurements'] = []
    elif defect == 'measurement_duplicate':
        measured['measurements'] *= 2
    elif defect == 'precision':
        measured['measurements'][0]['precision'] = 'different'
    elif defect == 'memory':
        measured['measurements'][0]['max_allocated_gib'] = 79.
    elif defect == 'zero_memory_total':
        measured['measurements'][0]['total_gib'] = 0.
    elif defect == 'nondivisible_batch':
        measured['recommended_micro_batch'] = 3
    elif defect == 'global_batch':
        measured['global_batch'] = 16
    elif defect == 'variant':
        measured['variant'] = 'r12_architecture_control'
    write(c.profile/'summary.json', measured)
    c.certificate['summary_sha256'] = hashlib.sha256((c.profile/'summary.json').read_bytes()).hexdigest()
    write(c.profile/'COMPLETE.json', c.certificate)
    with pytest.raises(ValueError):
        helper.verify_diagnostic(c.pool, c.variant)


@pytest.mark.parametrize('defect', ['protocol', 'ratio', 'missing_count', 'negative_count',
                                  'zero_residues', 'count_decomposition', 'changed_population'])
def test_mask_population_and_protocol_cannot_change_silently(evidence, defect):
    point = evidence.report['trace'][6]
    level = point['mask_levels']['mask25']
    if defect == 'protocol':
        point['mask_protocol'] = 'different'
    elif defect == 'ratio':
        level['mask_ratio'] = .1
    elif defect == 'missing_count':
        del level['aa_count']
    elif defect == 'negative_count':
        level['terminal_count'] = -1
    elif defect == 'zero_residues':
        level.update(aa_count=0, selected_count=2)
    elif defect == 'count_decomposition':
        level['selected_count'] = 20
    elif defect == 'changed_population':
        level.update(aa_count=11, selected_count=13)
    save_report(evidence)
    with pytest.raises(ValueError):
        helper.verify_diagnostic(evidence.pool, evidence.variant)


def test_a_later_valid_rerun_can_replace_an_invalid_diagnostic(evidence):
    invalid = dict(evidence.result, job_id='aaa_missing_artifacts')
    write(evidence.pool/'results'/'aaa_missing_artifacts.json', invalid)
    assert helper.verify_diagnostic(evidence.pool, evidence.variant) == evidence.run_id


def test_data_certificate_failure_precedes_diagnostic_and_never_publishes(evidence, monkeypatch):
    calls = []
    def fail_data(path):
        calls.append(('data', path))
        raise ValueError('Data certificates not ready')
    monkeypatch.setattr(helper, 'assert_data_ready', fail_data)
    monkeypatch.setattr(helper, 'verify_diagnostic', lambda *_: pytest.fail('Diagnostic ran before data gate'))
    monkeypatch.setattr(helper, 'enqueue', lambda *_: pytest.fail('Published despite failed data gate'))
    with pytest.raises(ValueError, match='Data certificates'):
        helper.enqueue_jobs(evidence.pool, [evidence.job], apply=True)
    assert calls == [('data', Path(evidence.job['data_dir']))]


def test_invalid_later_job_prevents_partial_stage_publication(evidence, monkeypatch):
    other = dict(evidence.job, variant='r12_architecture_control')
    monkeypatch.setattr(helper, 'enqueue', lambda *_: pytest.fail('Partially published stage'))
    with pytest.raises(ValueError):
        helper.enqueue_jobs(evidence.pool, [evidence.job, other], apply=True)


@pytest.mark.parametrize('folder', ['pending', 'running', 'finished'])
def test_existing_job_is_not_reenqueued(evidence, monkeypatch, folder):
    write(evidence.pool/folder/'already.json', evidence.job)
    monkeypatch.setattr(helper, 'enqueue', lambda *_: pytest.fail('Duplicate publication'))
    result = helper.enqueue_jobs(evidence.pool, [evidence.job], apply=True)
    assert result[0]['decision'] == 'already_registered'


def test_duplicate_plan_dry_run_has_no_side_effect(evidence):
    result = helper.enqueue_jobs(evidence.pool, [evidence.job, deepcopy(evidence.job)])
    assert [r['decision'] for r in result] == ['validated_not_enqueued', 'duplicate_in_plan']
    assert not list((evidence.pool/'pending').iterdir())
    assert not (evidence.pool/'architecture_enqueue.lock').exists()


def test_concurrent_helper_calls_publish_exact_job_once(evidence, monkeypatch):
    published = []
    def mock_enqueue(pool, job):
        time.sleep(.03)
        published.append(deepcopy(job))
        write(pool/'pending'/'mock_publication.json', job)
        return 'mock_publication'
    monkeypatch.setattr(helper, 'enqueue', mock_enqueue)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: helper.enqueue_jobs(evidence.pool,
                    [deepcopy(evidence.job)], apply=True), range(2)))
    assert len(published) == 1
    assert sorted(r[0]['decision'] for r in results) == ['already_registered', 'enqueued']


def test_engineering_stage_does_not_read_data_or_require_prior_diagnostic(evidence, monkeypatch):
    monkeypatch.setattr(helper, 'assert_data_ready', lambda *_: pytest.fail('Unexpected data gate'))
    monkeypatch.setattr(helper, 'verify_diagnostic', lambda *_: pytest.fail('Unexpected prior diagnostic gate'))
    job = dict(variant=evidence.variant, mode='profile-synthetic', steps=100)
    assert helper.enqueue_jobs(evidence.pool, [job])[0]['decision'] == 'validated_not_enqueued'
