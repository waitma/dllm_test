"""Check the new capacity dispatcher once, without executing a model or job.

  source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
  conda activate pllm
  CUDA_VISIBLE_DEVICES='' python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_pool_large_batch_dispatch.py

Only the real prepare/parse_args software checks run once. Rejection controls
use owned synthetic_fakecontrol files; exec and legacy main are mocked. All
temporary fixture files are deleted. Evidence is written only to the new own
/project_c20250601/251105016/project/dllm_test/output/v7_large_batch_dispatch_controls_20261002 directory.
"""
from contextlib import ExitStack, redirect_stderr
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
from unittest.mock import Mock, patch

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
SOURCE = ROOT / 'scripts/research/pool_large_batch_dispatch.py'
EXPECTED_SHA = '76d07d623f0fe068dcffa55a28c60a38695fe0a256d4a4bfd7751bded50aad5c'
OUT = ROOT / 'output/v7_large_batch_dispatch_controls_20261002'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rejected(call, exceptions=(ValueError, SystemExit)):
    with redirect_stderr(io.StringIO()):
        try:
            call()
        except exceptions as error:
            return type(error).__name__
    raise AssertionError('Expected the new boundary to reject this CPU fixture')


class ExecIntercepted(Exception):
    pass


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or sha(SOURCE) != EXPECTED_SHA:
        raise ValueError('Require CPU-only control environment and the reviewed source')
    OUT.mkdir(parents=True, exist_ok=False)
    started, records, logs, before = time.monotonic(), [], [], {}
    status, failure = 'failed', None
    try:
        module = runpy.run_path(str(SOURCE), run_name='_capacity_dispatch_control')
        g = module['prepare'].__globals__
        before = {str(path): sha(path) for path in (SOURCE, g['FREEZE'], g['PREVIOUS'])}
        alias = g['ALIAS']
        def argv(pool, job='9999999999999999999_p04_v7_capacity_c0decafe'):
            return ['--variant', alias, '--mode', 'profile-synthetic', '--steps', '72',
                    '--profile-dir', str(pool / 'profiles' / alias), '--output-dir', str(pool / 'runs' / job)]
        with ExitStack() as forbidden:
            forbidden.enter_context(patch('subprocess.Popen', side_effect=AssertionError('No process in stdlib controls')))
            forbidden.enter_context(patch('socket.socket', side_effect=AssertionError('No socket in stdlib controls')))
            # This is the only call to prepare against the actual manifest.
            artificial_output = g['POOL'] / 'runs/9999999999999999999_p04_v7_capacity_c0decafe'
            assert not artificial_output.exists()
            args, command = g['prepare'](argv(g['POOL']))
            profile = Path(command[4])
            profile_module = runpy.run_path(str(profile), run_name='_stdlib_profile_parse_only')
            parsed = profile_module['parse_args'](command[5:])
            assert parsed.variants == ['f000', 'f111', 'cv_segment_global']
            assert parsed.micro_batches == [4, 16, 32]
            assert (parsed.global_batch, parsed.warmup_steps, parsed.measure_steps, parsed.candidate_seconds,
                    parsed.max_seconds, parsed.seed) == (32, 2, 6, 180, 1800, 42)
            assert len(parsed.variants)*len(parsed.micro_batches)*(parsed.warmup_steps+parsed.measure_steps) == 72
            assert parsed.source_root == profile.parents[2] and parsed.output_dir == artificial_output
            assert profile_module['CONFIG']['max_length'] == 2048
            assert parsed.candidate_variant is None and parsed.candidate_micro is None
            assert command[:4] == [sys.executable, '-I', '-B', '-u']
            assert not artificial_output.exists() and 'torch' not in sys.modules
            manifest = json.loads(g['FREEZE'].read_text())
            records.append({'control': 'actual_prepare_and_frozen_parse_once', 'command': command,
                'source_files_checked': len(manifest['source_sha256']),
                'evidence_files_checked': len(manifest['evidence_sha256']),
                'profile_source_sha256': sha(profile), 'planned_updates': 72,
                'max_length_declaration': 2048, 'artificial_output_created': False})
            logs.append('PASS actual frozen prepare once + actual stdlib profile parse_args: 72 updates, 3x3 candidates, length2048 declaration')

            # All remaining prepares use explicitly artificial software metadata.
            with tempfile.TemporaryDirectory(prefix='synthetic_fakecontrol_', dir=OUT) as temporary:
                base = Path(temporary).resolve()
                own, pool = base/'own', base/'pool'
                snapshot = own/'source_snapshot_v1'
                (snapshot/'scripts/research').mkdir(parents=True)
                (pool/'runs').mkdir(parents=True)
                (pool/'profiles').mkdir()
                fake_profile = snapshot/'scripts/research/profile_v7_large_batch.py'
                fake_profile.write_bytes(profile.read_bytes())
                evidence = own/'CPU_metadata_only.json'
                write(evidence, {'synthetic_fakecontrol': True, 'GPU_accepted': False})
                freeze = own/'source_snapshot_v1.json'
                write(freeze, {'synthetic_fakecontrol': True, 'snapshot_root': str(snapshot),
                    'source_sha256': {'scripts/research/profile_v7_large_batch.py': sha(fake_profile)},
                    'evidence_sha256': {str(evidence): sha(evidence)}})
                with patch.dict(g, {'OWN': own, 'POOL': pool, 'FREEZE': freeze, 'FREEZE_SHA': sha(freeze)}):
                    valid = argv(pool)
                    wrong = []
                    for name, value in (('--mode', 'train'), ('--steps', '71'), ('--steps', '73')):
                        altered = valid.copy()
                        altered[altered.index(name)+1] = value
                        wrong.append(rejected(lambda: g['prepare'](altered)))
                    for flags in (['--data-dir', '/synthetic_fakecontrol/unread'],
                                  ['--resume-from-checkpoint', '/synthetic_fakecontrol/unread'],
                                  ['--micro-batches', '32']):
                        wrong.append(rejected(lambda: g['prepare'](valid+flags)))
                    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3'}), \
                            patch.object(g['os'], 'execv', side_effect=ExecIntercepted) as execute:
                        pid = os.getpid()
                        assert rejected(lambda: g['main'](valid), (ExecIntercepted,)) == 'ExecIntercepted'
                        executable, launched = execute.call_args.args
                        assert executable == sys.executable and launched[:4] == [sys.executable, '-I', '-B', '-u']
                        assert launched[4] == str(fake_profile) and os.getpid() == pid
                    for mask in ('', '0,1', '8'):
                        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': mask}), patch.object(g['os'], 'execv') as execute:
                            rejected(lambda: g['main'](valid))
                            execute.assert_not_called()
                    records.append({'control': 'exact_envelope_and_exec', 'rejected_envelopes': len(wrong),
                        'rejected_masks': ['', '0,1', '8'], 'exec_mocked': True, 'actual_exec_executed': False})
                    logs.append('PASS strict synthetic/72 envelope; data/resume/custom sweep rejected; single-mask execv contract mocked')

                    original_profile = fake_profile.read_bytes()
                    fake_profile.write_bytes(original_profile+b'\n# synthetic_fakecontrol drift\n')
                    rejected(lambda: g['prepare'](valid))
                    fake_profile.write_bytes(original_profile)
                    original_evidence = evidence.read_bytes()
                    evidence.write_bytes(original_evidence+b'\n')
                    rejected(lambda: g['prepare'](valid))
                    evidence.write_bytes(original_evidence)
                    original_freeze = freeze.read_bytes()
                    freeze.write_bytes(original_freeze+b'\n')
                    rejected(lambda: g['prepare'](valid))
                    freeze.write_bytes(original_freeze)
                    outside = base/'outside'
                    outside.mkdir()
                    run = pool/'runs/9999999999999999999_p04_v7_capacity_c0decafe'
                    run.symlink_to(outside, target_is_directory=True)
                    rejected(lambda: g['prepare'](valid))
                    run.unlink()
                    fake_profile.unlink()
                    external_profile = outside/'profile.py'
                    external_profile.write_bytes(original_profile)
                    fake_profile.symlink_to(external_profile)
                    rejected(lambda: g['prepare'](valid))
                    records.append({'control': 'actual_temporary_hash_and_path_rejections',
                        'rejected': ['source_bytes_drift', 'evidence_bytes_drift', 'manifest_bytes_drift',
                                     'output_symlink_escape', 'frozen_source_symlink']})
                    logs.append('PASS actual temporary source/evidence/manifest drift and output/source symlinks rejected')

                previous_argv = ['--variant=p02_v7_f111_seed43', '--mode', 'train', '--steps', '3750',
                                 '--data-dir', '/synthetic_fakecontrol/not_read', '--output-dir', '/synthetic_fakecontrol/not_created']
                previous_main = Mock(return_value='synthetic_fakecontrol_delegated')
                with patch.object(g['runpy'], 'run_path', return_value={'main': previous_main}) as loader:
                    assert g['main'](previous_argv) == 'synthetic_fakecontrol_delegated'
                    loader.assert_called_once_with(str(g['PREVIOUS']))
                    previous_main.assert_called_once_with(previous_argv)
                with patch.dict(g, {'PREVIOUS_SHA': '0'*64}), patch.object(g['runpy'], 'run_path') as loader:
                    rejected(lambda: g['main'](previous_argv))
                    loader.assert_not_called()
                records.append({'control': 'old_p02_delegation', 'argv': previous_argv,
                    'legacy_main_mocked': True, 'actual_previous_sha256': sha(g['PREVIOUS'])})
                logs.append('PASS previous p02 argv unchanged and SHA checked before mocked delegation')
            assert not list(OUT.glob('synthetic_fakecontrol_*')) and 'torch' not in sys.modules
        assert all(sha(Path(path)) == value for path, value in before.items())
        status = 'passed'
    except BaseException:
        import traceback
        failure = traceback.format_exc()
        logs.append(failure)
    finally:
        stdout = '\n'.join(logs)+'\n'
        print(stdout, end='')
        (OUT/'stdout.log').write_text(stdout)
        write(OUT/'result.json', {'status': status, 'finished_at': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic()-started, 'python_executable': sys.executable,
            'command': [sys.executable, '-B', str(Path(__file__).resolve())],
            'test_source_sha256': sha(Path(__file__)), 'source_sha256_before': before,
            'source_sha256_after': {path: sha(Path(path)) for path in before},
            'stdout_sha256': sha(OUT/'stdout.log'), 'controls': records, 'error': failure,
            'scope': 'stdlib_dispatcher_boundaries_only', 'model_executed': False, 'GPU_executed': False,
            'profile_run_executed': False, 'real_data_read': False, 'queue_changed': False,
            'actual_exec_or_legacy_main_executed': False,
            'temporary_remaining': [str(path) for path in OUT.glob('synthetic_fakecontrol_*')]})
    if status != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
