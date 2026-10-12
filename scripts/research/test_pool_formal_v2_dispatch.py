"""Run three new stdlib cold controls once after activating conda pllm.

CUDA_VISIBLE_DEVICES='' python -B /vepfs-mlp2/c20250601/251105016/project/
dllm_test/scripts/research/test_pool_formal_v2_dispatch.py
No model/data/GPU/group/queue execution. Only fallback dispatch is exercised,
with actual downstream main mocked; temporary source drift fixtures are removed.
"""
import builtins
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
SOURCE = ROOT/'scripts/research/pool_formal_v2_dispatch.py'
OUT = ROOT/'output/v7_formal_v2_dispatch_controls_20261003'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejected(call):
    try:
        call()
    except ValueError as error:
        return str(error)
    raise AssertionError('Expected cold-entry rejection')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    OUT.mkdir(exist_ok=False)
    started, status, records, before, error = time.monotonic(), 'failed', [], {}, None
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'numpy', 'transformers', 'accelerate', 'dllm'):
            raise AssertionError('ML/data import forbidden')
        return original_import(name, *args, **kwargs)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch('builtins.__import__', side_effect=guarded))
            stack.enter_context(patch('subprocess.Popen', side_effect=AssertionError('No child process')))
            stack.enter_context(patch('socket.socket', side_effect=AssertionError('No listener')))
            execute = stack.enter_context(patch('os.execv', side_effect=AssertionError('No exec')))
            module = runpy.run_path(str(SOURCE), run_name='_formal_v2_cold_controls')
            runtime = module['_runtime']
            before = {str(path): sha(path) for path in (SOURCE, Path(__file__), module['BASE'], module['PREVIOUS'], module['MANIFEST'])}
            argv = ['--variant=p04_v7_f000_seed42', '--mode', 'train', '--steps', '3750',
                    '--opaque', 'preserve_this_value']
            delegated = Mock(return_value='downstream_main_mocked')
            with patch.object(runtime['runpy'], 'run_path', return_value={'main': delegated}) as load:
                assert module['main'](argv) == 'downstream_main_mocked'
                load.assert_called_once_with(str(module['PREVIOUS']))
                delegated.assert_called_once_with(argv)
            assert module['prepare'] is runtime['prepare'] and module['main'] is runtime['main']
            records.append({'control': 'real_fallback_SHA_manifest_and_unchanged_argv', 'argv': argv,
                'target': str(module['PREVIOUS']), 'downstream_main_mocked': True})
            print('PASS actual frozen fallback/manifest SHA; original argv delegated unchanged; downstream main mocked')

            with tempfile.TemporaryDirectory(prefix='formal_v2_fakecontrol_', dir=OUT) as temporary:
                changed = Path(temporary)/'dispatcher.py'
                changed.write_bytes(module['PREVIOUS'].read_bytes()+b'\n# artificial cold drift\n')
                with patch.dict(runtime, {'PREVIOUS': changed}), patch.object(runtime['runpy'], 'run_path') as load:
                    failure = rejected(lambda: module['main'](argv))
                    load.assert_not_called()
                manifest = Path(temporary)/'manifest.json'
                manifest.write_bytes(module['MANIFEST'].read_bytes()+b'\n')
                manifest_failure = rejected(lambda: module['bound'](manifest, module['MANIFEST_SHA']))
                records.append({'control': 'source_and_manifest_drift_rejected', 'source_error': failure,
                                'manifest_error': manifest_failure, 'temporary_only': True})
            print('PASS temporary source/manifest SHA drift rejected before delegation')

            assert not runtime['CATALOG'].exists(), 'Formal catalog became published; do not extend this control'
            alias = runtime['ALIASES'][0]
            output = runtime['POOL']/'runs'/f'9999999999999999999_{alias}_c0decafe'
            formal_argv = ['--variant', alias, '--mode', 'train', '--steps', '1',
                '--data-dir', str(runtime['DATA']), '--profile-dir', str(runtime['POOL']/'profiles'/alias),
                '--output-dir', str(output)]
            assert not output.exists()
            failure = rejected(lambda: module['prepare'](formal_argv))
            assert str(runtime['CATALOG']) in failure and not output.exists()
            records.append({'control': 'unpublished_formal_catalog_still_rejected', 'error': failure,
                            'output_created': False, 'formal_plan_created': False})
            print('PASS unpublished formal catalog still rejected by unchanged prepare; no plan/output created')
            execute.assert_not_called()
            assert not list(OUT.glob('formal_v2_fakecontrol_*')) and 'torch' not in sys.modules
            assert all(sha(Path(path)) == value for path, value in before.items())
            status = 'passed'
    except BaseException:
        error = traceback.format_exc()
        print(error)
    finally:
        (OUT/'receipt.json').write_text(json.dumps({'status': status, 'at_utc': datetime.now(timezone.utc).isoformat(),
            'execution_count': 1, 'elapsed_seconds': time.monotonic()-started, 'controls': records,
            'source_sha256_before': before, 'source_sha256_after': {path: sha(Path(path)) for path in before},
            'python': sys.executable, 'error': error, 'scope': 'three_new_formal_fallback_cold_controls_only',
            'models_data_GPU_or_groups_executed': False, 'actual_downstream_main_or_exec': False,
            'published_or_enqueued': False, 'old_suites_repeated': False,
            'temporary_remaining': [str(path) for path in OUT.glob('formal_v2_fakecontrol_*')]}, indent=2)+'\n')
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
