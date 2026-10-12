"""Run one changed-boundary cold control after activating conda pllm.

CUDA_VISIBLE_DEVICES='' python -B /vepfs-mlp2/c20250601/251105016/project/
dllm_test/scripts/research/test_pool_large_batch_ablation_v2_dispatch.py
Reads actual planned catalog/plan/capacity metadata, never sequence rows. Only
the unpublished registry is replaced with an owned temporary literal fixture.
No model, GPU, exec, queue publication, old suite or capacity rerun is allowed.
"""
import builtins
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
import traceback
from unittest.mock import Mock, patch

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
SOURCE = ROOT/'scripts/research/pool_large_batch_ablation_v2_dispatch.py'
OUT = ROOT/'output/v7_large_batch_v2_dispatch_controls_20261003'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rejected(call):
    try:
        call()
    except (ValueError, KeyError, FileNotFoundError) as error:
        return str(error)
    raise AssertionError('Expected new capacity-v2 boundary rejection')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    OUT.mkdir(exist_ok=False)
    started, status, records, error, before = time.monotonic(), 'failed', [], None, {}
    original_import = builtins.__import__
    def cold_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'numpy', 'transformers', 'accelerate', 'dllm'):
            raise AssertionError('ML import forbidden')
        return original_import(name, *args, **kwargs)
    try:
        with ExitStack() as forbidden:
            forbidden.enter_context(patch('builtins.__import__', side_effect=cold_import))
            forbidden.enter_context(patch('subprocess.Popen', side_effect=AssertionError('No process')))
            forbidden.enter_context(patch('socket.socket', side_effect=AssertionError('No socket')))
            execute = forbidden.enter_context(patch('os.execv', side_effect=AssertionError('No exec')))
            module = runpy.run_path(str(SOURCE), run_name='_v2_ablation_cold_control')
            runtime = module['_runtime']
            check = module['check_capacity']
            scope = check.__globals__
            canonical = runtime['previous']()['canonical']
            paths = (SOURCE, Path(__file__), module['BASE'], module['PREVIOUS'], runtime['PLAN'],
                     runtime['CATALOG'], runtime['FREEZE'], runtime['REGISTRY'], module['ROOT_REVIEW'])
            before = {str(path): sha(path) for path in paths}
            plan = json.loads(runtime['PLAN'].read_text())
            catalog = json.loads(runtime['CATALOG'].read_text())
            assert sha(runtime['PLAN']) == '42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04'
            assert sha(runtime['CATALOG']) == '987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d'
            entries = {alias: {'launcher_only': True, 'model_variant': variant, 'seed': seed,
                'study': runtime['STUDY'], 'jobs_catalog_sha256': sha(runtime['CATALOG']),
                'batch_plan_sha256': sha(runtime['PLAN']), 'source_freeze_sha256': runtime['FREEZE_SHA']}
                for alias, (variant, seed) in runtime['ALIASES'].items()}
            def argv(row):
                return ['--variant', row['alias'], '--mode', 'train', '--steps', '3750',
                    '--data-dir', str(runtime['DATA']), '--profile-dir', str(runtime['POOL']/'profiles'/row['alias']),
                    '--output-dir', str(runtime['POOL']/'runs'/row['job_id'])]
            outputs = [runtime['POOL']/'runs'/row['job_id'] for row in catalog['jobs']]
            assert not any(path.exists() for path in outputs)
            with tempfile.TemporaryDirectory(prefix='synthetic_fakecontrol_', dir=OUT) as temporary:
                registry = Path(temporary)/'registry.py'
                registry.write_text('# Artificial unpublished metadata fixture, no GPU claim.\nVARIANTS = '+repr(entries)+'\n')
                freeze = runtime['check_freeze']
                captures = []
                def observed_freeze(canonical):
                    result = freeze(canonical)
                    captures.append(result)
                    return result
                with patch.dict(runtime, {'REGISTRY': registry, 'check_freeze': observed_freeze}):
                    # Actual v2 capacity originals and unchanged 568-source verifier, once.
                    _, first_command = module['prepare'](argv(catalog['jobs'][0]))
                    assert len(captures) == 1
                    snapshot, sources = captures[0]
                    # Remaining parameter envelopes reuse the already checked metadata.
                    with patch.dict(runtime, {'check_freeze': lambda canonical: (snapshot, sources),
                                              'check_capacity': lambda *args: None}):
                        commands = [first_command]
                        for row in catalog['jobs'][1:]:
                            commands.append(module['prepare'](argv(row))[1])
                    for row, command in zip(catalog['jobs'], commands):
                        assert command[command.index('--variant')+1] == row['variant']
                        assert command[command.index('--seed')+1] == str(row['seed'])
                        assert command[command.index('--micro-batch')+1] == '32'
                        assert command[command.index('--grad-accum')+1] == '1'
                        assert command[command.index('--steps')+1] == '3750'
                        assert command[command.index('--batch-plan-sha256')+1] == sha(runtime['PLAN'])
                    assert len(commands) == 22
                records.append({'control': 'real_v2_capacity_and_22_planned_envelopes', 'commands': commands,
                    'capacity_originals_checked': True, 'full_source_verifications': 1,
                    'temporary_registry_only': True, 'registry_published': False})
                print('PASS actual v2 originals + actual plan/catalog + unchanged frozen runner parser for all 22 envelopes; one source/capacity verification')

                actual_read = scope['read']
                summary_path = runtime['POOL']/'runs'/module['CAPACITY_JOB']/'summary.json'
                actual_summary = actual_read(summary_path, canonical)
                changes = []
                def mutated_check(mutation):
                    summary = copy.deepcopy(actual_summary)
                    mutation(summary)
                    def read_modified(path, canonical, expected=None):
                        value = actual_read(path, canonical, expected)
                        if Path(path) == summary_path:
                            return summary
                        if Path(path).name == 'measurement.json':
                            return next(copy.deepcopy(item) for item in summary['measurements']
                                if Path(path).parent.name == f'{item["variant"]}_micro{item["micro_batch"]}')
                        return value
                    with patch.dict(scope, {'read': read_modified}):
                        return rejected(lambda: check(plan, snapshot, sources, canonical))
                changes.append(mutated_check(lambda summary: summary.update(planned_optimizer_updates=72, candidate_seconds=180)))
                def old_cell(summary):
                    summary['measurements'][0].update(completed_optimizer_steps=8, measure_steps=6, measured_losses=[1.]*6)
                changes.append(mutated_check(old_cell))
                def selected_oom(summary):
                    next(item for item in summary['measurements'] if item['micro_batch'] == 32).update(
                        status='oom', expected_capacity_failure=True, error='synthetic_fakecontrol_only')
                changes.append(mutated_check(selected_oom))
                def unsafe(summary):
                    item = next(item for item in summary['measurements'] if item['micro_batch'] == 32)
                    item['peak_reserved_gib'] = item['total_gib']*.86
                changes.append(mutated_check(unsafe))
                missing = copy.deepcopy(plan)
                missing['profile_evidence'] = [item for item in missing['profile_evidence'] if not item['path'].endswith('/f000_micro4/measurement.json')]
                changes.append(rejected(lambda: check(missing, snapshot, sources, canonical)))
                wrong_intent = copy.deepcopy(plan)
                next(item for item in wrong_intent['profile_evidence'] if item['path'].endswith('/capacity_job_v2.json'))['sha256'] = '0'*64
                changes.append(rejected(lambda: check(wrong_intent, snapshot, sources, canonical)))
                def mismatched_original(path, canonical, expected=None):
                    value = actual_read(path, canonical, expected)
                    if Path(path).parent.name == 'f000_micro4' and Path(path).name == 'measurement.json':
                        value['measured_losses'][0] += 1.
                    return value
                with patch.dict(scope, {'read': mismatched_original}):
                    changes.append(rejected(lambda: check(plan, snapshot, sources, canonical)))
                def wrong_prior(path, canonical, expected=None):
                    value = actual_read(path, canonical, expected)
                    if Path(path).name == 'capacity_job_v2.json':
                        value['previous_failure']['result']['sha256'] = '0'*64
                    return value
                with patch.dict(scope, {'read': wrong_prior}):
                    changes.append(rejected(lambda: check(plan, snapshot, sources, canonical)))
                records.append({'control': 'new_protocol_failure_rejections', 'errors': changes,
                    'mutation_scope': 'in-memory values only; actual original bytes and plan unchanged'})
                print('PASS old72/180 and old8/6 cells, selected OOM/headroom, missing/mismatched raw measurement, wrong intent and prior-failure binding rejected')

                # Missing unpublished registry must still fail closed; no public edits.
                registry.write_text('VARIANTS = {}\n')
                with patch.dict(runtime, {'REGISTRY': registry}):
                    rejected(lambda: module['prepare'](argv(catalog['jobs'][0])))
                records.append({'control': 'unpublished_registry_fail_closed'})
                print('PASS missing literal launcher registry rejected')

            unknown = ['--variant=p02_v7_f000_seed42', '--mode', 'train', '--steps', '3750', '--opaque', 'keep']
            delegated = Mock(return_value='mocked_capacity_v2_chain')
            with patch.object(runtime['runpy'], 'run_path', return_value={'main': delegated}) as load:
                assert module['main'](unknown) == 'mocked_capacity_v2_chain'
                load.assert_called_once_with(str(module['PREVIOUS']))
                delegated.assert_called_once_with(unknown)
            records.append({'control': 'private_capacity_v2_fallback_SHA_and_unchanged_argv', 'actual_main_called': False})
            print('PASS unknown argv delegates to SHA-bound private capacity-v2; downstream main mocked')
            execute.assert_not_called()
            assert not any(path.exists() for path in outputs)
            assert not list(OUT.glob('synthetic_fakecontrol_*')) and 'torch' not in sys.modules
            assert all(sha(Path(path)) == value for path, value in before.items())
            status = 'passed'
    except BaseException:
        error = traceback.format_exc()
        print(error)
    finally:
        (OUT/'receipt.json').write_text(json.dumps({'status': status, 'at_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic()-started, 'execution_count': 1, 'controls': records,
            'source_sha256_before': before, 'source_sha256_after': {path: sha(path) for path in before},
            'python': sys.executable, 'error': error, 'scope': 'new_capacity_v2_integration_cold_controls_only',
            'models_or_GPU_executed': False, 'sequence_rows_read': False, 'processes_or_exec': False,
            'public_registry_or_queue_modified': False,
            'temporary_remaining': [str(path) for path in OUT.glob('synthetic_fakecontrol_*')]}, indent=2)+'\n')
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
