"""Non-invasive supervisor checks: pytest -q scripts/tests/immune_llada/test_research_supervision.py."""
import hashlib
import json

from scripts import immune_research_supervision as supervision


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_pool(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    monkeypatch.setattr(supervision, 'ROOT', root)
    source = root / 'model.py'
    source.write_text('model = 1')
    pool = root / 'pool'
    put(pool / 'running' / 'job.json', dict(variant='r00', mode='profile-synthetic', steps=20))
    profile = dict(recommended_micro_batch=32, measurements=[dict(batch_size=32, status='passed', max_allocated_gib=42), dict(batch_size=64, status='oom')])
    put(pool / 'profiles' / 'r00' / 'summary.json', profile)
    put(pool / 'profiles' / 'r00' / 'COMPLETE.json', dict(status='passed', summary_sha256=hashlib.sha256((pool/'profiles/r00/summary.json').read_bytes()).hexdigest(), source_sha256={'model.py': hashlib.sha256(source.read_bytes()).hexdigest()}))
    return pool, source


def test_expected_profile_oom_is_not_failure_and_scope_is_initial20(tmp_path, monkeypatch):
    pool, source = fixture_pool(tmp_path, monkeypatch)
    put(pool / 'pending' / 'extended.json', dict(variant='r00', mode='profile-synthetic', steps=100))
    put(pool / 'runs' / 'job' / 'summary.json', dict(trace=[dict(step=5, loss=2, aa_accuracy=0, terminal_accuracy=1)]))
    state = supervision.inspect(pool)
    record = state['records']['job']
    assert state['initial_jobs'] == 1
    assert record['source_status'] == 'matched'
    assert record['profile_oom_batches'] == [64]
    assert record['alerts'] == ['endpoint_shortcut_signal']
    assert record['state'] == 'running'


def test_changed_source_and_incomplete_success_detected_without_mutation(tmp_path, monkeypatch):
    pool, source = fixture_pool(tmp_path, monkeypatch)
    source.write_text('model = 2')
    put(pool / 'results' / 'job.json', dict(state='succeeded'))
    state = supervision.update(pool, tmp_path / 'supervision.md')
    record = state['records']['job']
    assert 'source_changed' in record['alerts']
    assert 'inconsistent_completion' in record['alerts']
    assert source.read_text() == 'model = 2'
    assert len(state['events']) == 1
    again = supervision.update(pool, tmp_path / 'supervision.md')
    assert len(again['events']) == 1


def test_completed_and_failed_records_classified(tmp_path, monkeypatch):
    pool, source = fixture_pool(tmp_path, monkeypatch)
    put(pool / 'results' / 'job.json', dict(state='succeeded'))
    put(pool / 'runs' / 'job' / 'summary.json', dict(status='completed', final=dict(loss=1, aa_accuracy=.5, terminal_accuracy=.8), trace=[dict(step=20, loss=1)]))
    put(pool / 'runs' / 'job' / 'COMPLETE.json', dict(status='completed'))
    assert supervision.inspect(pool)['records']['job']['alerts'] == []
    put(pool / 'results' / 'job.json', dict(state='failed', error='runner failed'))
    log = pool / 'logs' / 'job.log'
    log.parent.mkdir()
    log.write_text('FloatingPointError: Nonfinite training loss')
    assert 'nonfinite_runner' in supervision.inspect(pool)['records']['job']['alerts']


def test_release_snapshot_and_independent_step_instances_with_mask_metrics(tmp_path, monkeypatch):
    pool, source = fixture_pool(tmp_path, monkeypatch)
    snapshot = source.parent / 'old_release'
    snapshot.mkdir()
    (snapshot / 'model.py').write_text(source.read_text())
    source.write_text('new architecture code')
    put(pool / 'release_map.json', dict(schema_version=1, variants={'r00': {'source_root': str(snapshot)}}))
    put(pool / 'finished' / 'job100.json', dict(variant='r00', mode='profile-synthetic', steps=100))
    put(pool / 'results' / 'job100.json', dict(state='succeeded'))
    levels = {f'mask{ratio}': dict(loss=ratio/100, aa_accuracy=.5, terminal_accuracy=1, aa_count=82) for ratio in (25, 50, 75, 100)}
    metrics = dict(mask_levels=levels, mask_mean=dict(loss=.625, aa_accuracy=.5, terminal_accuracy=1))
    put(pool / 'runs' / 'job100' / 'summary.json', dict(status='completed', initial=metrics, final=metrics, trace=[dict(step=100, **metrics)]))
    put(pool / 'runs' / 'job100' / 'COMPLETE.json', dict(status='completed'))
    document = tmp_path / 'supervision.md'
    state = supervision.update(pool, document)
    assert state['total_jobs'] == 2 and state['initial_jobs'] == 1
    assert set(state['records']) == {'job', 'job100'}
    assert all(record['source_status'] == 'matched' for record in state['records'].values())
    assert state['records']['job100']['alerts'] == []
    assert 'mask25' in document.read_text() and 'mask\\_mean' in document.read_text()
    assert '100/100' in document.read_text() and 'aa\\_count' in document.read_text()
    metrics['mask_levels']['mask50']['loss'] = float('nan')
    put(pool / 'runs' / 'job100' / 'summary.json', dict(status='completed', final=metrics, trace=[dict(step=100, **metrics)]))
    assert 'nonfinite_metric' in supervision.inspect(pool)['records']['job100']['alerts']


def test_new_fingerprinted_module_is_detected(tmp_path):
    directory = tmp_path / 'dllm/pipelines/immune_llada/research_next'
    directory.mkdir(parents=True)
    (directory / 'model.py').write_text('model = 1')
    proof = {'scripts/run_immune_research.py': 'unused', 'scripts/run_immune_architecture.py': 'unused',
             'dllm/pipelines/immune_llada/research_next/model.py': 'unused'}
    assert supervision.added_fingerprinted_files(tmp_path, proof) == []
    (directory / 'new_architecture.py').write_text('new = 1')
    assert supervision.added_fingerprinted_files(tmp_path, proof) == ['dllm/pipelines/immune_llada/research_next/new_architecture.py']
    # Old protocol does not fingerprint the separate next-architecture module.
    del proof['scripts/run_immune_architecture.py']
    assert supervision.added_fingerprinted_files(tmp_path, proof) == []


def test_refined_source_set_is_checked_only_for_refined_release(tmp_path):
    directory = tmp_path / 'dllm/pipelines/immune_llada/research_refined'
    directory.mkdir(parents=True)
    (directory / 'new_module.py').write_text('cache = {}')
    proof = {'scripts/run_immune_research.py': 'unused', 'scripts/run_immune_architecture.py': 'unused'}
    assert supervision.added_fingerprinted_files(tmp_path, proof) == []
    proof['scripts/run_immune_refined.py'] = 'unused'
    assert supervision.added_fingerprinted_files(tmp_path, proof) == ['dllm/pipelines/immune_llada/research_refined/new_module.py']


def test_expanded_train_kind_25step_trace_strata_and_completion_hash(tmp_path, monkeypatch):
    pool, source = fixture_pool(tmp_path, monkeypatch)
    # This is the concrete run_immune_refined.py summary schema. The train entry
    # point remains artificial when this kind/fixture manifest is present.
    put(pool / 'running' / 'job.json', dict(variant='r00', mode='train', steps=500))
    levels = {f'mask{ratio}': dict(mask_ratio=ratio/100, loss=1, aa_loss=.9,
              aa_accuracy=.5, terminal_loss=1.1, terminal_accuracy=.75,
              selected_count=100, aa_count=90, terminal_count=10) for ratio in (25, 50, 75, 100)}
    aggregate = dict(loss=1, aa_loss=.9, aa_accuracy=.5, terminal_loss=1.1,
                     terminal_accuracy=.75, mask_levels=levels,
                     mask_mean=dict(loss=1, aa_accuracy=.5, terminal_accuracy=.75))
    metrics = dict(**aggregate, strata={'conditional_tcr/long|测试': aggregate}, records=72)
    summary = dict(kind='synthetic_generalization_diagnostic', biological_quality_claim=False,
        dataset_kind='deterministic_artificial_sequences', initial=metrics,
        source_sha256={'model.py': hashlib.sha256(source.read_bytes()).hexdigest()},
        fixture_manifest=dict(manifest_sha256='fixture-sha', exact_target_sequence_overlap=0,
                              exact_record_overlap=0, cross_radius32_coverage={'possible': True}),
        trace=[dict(step=25, gradient_norm=.3, **metrics)])
    run = pool / 'runs' / 'job'
    put(run / 'summary.json', summary)
    document = tmp_path / 'supervision.md'
    state = supervision.update(pool, document)
    record = state['records']['job']
    assert record['step'] == 25 and record['requested_steps'] == 500
    assert record['kind'] == 'synthetic_generalization_diagnostic'
    assert record['source_status'] == 'matched' and record['alerts'] == []
    contents = document.read_text()
    assert '25/500' in contents and 'conditional\\_tcr/long&#124;测试' in contents
    assert 'mask100' in contents and 'trainer latest' not in contents
    assert 'fixture-sha' in contents and 'deterministic\\_artificial\\_sequences' in contents
    summary.update(status='completed', final=metrics, train_probe=metrics)
    summary['trace'].append(dict(step=500, gradient_norm=.1, **metrics))
    put(run / 'summary.json', summary)
    put(run / 'COMPLETE.json', dict(status='completed', global_step=500,
        kind=summary['kind'], biological_quality_claim=False,
        summary_sha256=hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()))
    put(pool / 'results' / 'job.json', dict(state='succeeded'))
    state = supervision.update(pool, document)
    assert state['records']['job']['alerts'] == []
    assert 'train\\_probe' in document.read_text()
    # Tampering is reported; neither the evidence nor the queued job is changed.
    summary['train_probe']['strata']['conditional_tcr/long|测试']['mask_mean']['loss'] = float('nan')
    put(run / 'summary.json', summary)
    record = supervision.inspect(pool)['records']['job']
    assert 'nonfinite_metric' in record['alerts']
    assert 'run_summary_hash_mismatch' in record['alerts']
    assert (run/'COMPLETE.json').exists() and (pool/'running/job.json').exists()


def test_train_entry_does_not_excuse_incomplete_synthetic_summary(tmp_path, monkeypatch):
    pool, source = fixture_pool(tmp_path, monkeypatch)
    put(pool / 'running' / 'job.json', dict(variant='r00', mode='train', steps=500))
    put(pool / 'results' / 'job.json', dict(state='succeeded'))
    put(pool / 'runs' / 'job' / 'COMPLETE.json', dict(status='completed', global_step=500))
    put(pool / 'runs' / 'job' / 'summary.json', dict(kind='synthetic_generalization_diagnostic', biological_quality_claim=True))
    record = supervision.inspect(pool)['records']['job']
    assert 'synthetic_claim_inconsistent' in record['alerts']
    assert 'inconsistent_completion' in record['alerts']


def test_conditional_subsets_pool_actual_counts_and_ignore_unknown_groups():
    def stratum(aa_count, aa_accuracy, end_count, end_accuracy):
        level = dict(selected_count=aa_count+end_count, aa_count=aa_count,
            terminal_count=end_count, loss=2, aa_loss=1, aa_accuracy=aa_accuracy,
            terminal_loss=3 if end_count else None, terminal_accuracy=end_accuracy)
        return dict(mask_levels={f'mask{ratio}': dict(level) for ratio in (25, 50, 75, 100)})
    metrics = dict(strata={'ab_antigen/short': stratum(2, 1., 1, 1.),
                          'tcr_pmhc/long': stratum(8, .25, 3, 0.),
                          'unconditional_single/long': stratum(10, .1, 0, None),
                          'unknown/long': stratum(1000, 1., 100, 1.)})
    grouped = supervision.grouped_metrics(metrics)
    assert set(grouped) == {'conditional', 'unconditional'}
    expected = grouped['conditional']
    assert expected['stratum_count'] == 2
    assert expected['mask_mean']['aa_accuracy'] == .4  # 4/10, not (.25+1)/2.
    assert expected['mask_mean']['terminal_accuracy'] == .25  # 1/4.
    assert expected['mask_levels']['mask100']['selected_count'] == 14
    assert grouped['unconditional']['mask_mean']['terminal_accuracy'] is None
    assert grouped['unconditional']['mask_mean']['aa_accuracy'] == .1
    assert supervision.grouped_metrics({}) == {}


def test_old_and_expanded_oracles_are_explicitly_scoped(tmp_path, monkeypatch):
    pool, _ = fixture_pool(tmp_path, monkeypatch)
    put(pool/'monitor/full_mask_identifiability.json', dict(full_mask_input_groups=[[0, 1]],
        aa_only=dict(maximum_correct=1, target_count=2, accuracy_upper_bound=.5),
        all_targets=dict(maximum_correct=1, target_count=2, accuracy_upper_bound=.5)))
    put(pool/'monitor/expanded_identifiability.json', dict(fixture_manifest_sha256='expanded-fixture',
        groups={'conditional':dict(mask_mean_aa_upper_bound=1, levels={'mask100':dict(input_group_count=36,
            aa={'accuracy_upper_bound':1}, selected={'accuracy_upper_bound':1},
            joint_optimal_terminal_accuracy_range=[1, 1])})}))
    document = tmp_path/'supervision.md'
    supervision.update(pool, document)
    text = document.read_text()
    assert '原六条 toy' in text and '不能用于后续扩展人工 heldout' in text
    assert 'expanded-fixture' in text and '联合argmax最优时end范围' in text
