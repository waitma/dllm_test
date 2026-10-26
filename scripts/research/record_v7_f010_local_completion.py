"""Record the owned f010/42 attempt's real exit after its full epoch finishes.

Activate conda pllm, then run once after the existing runner has exited:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/record_v7_f010_local_completion.py
This reads real runner sentinels and native artifacts, creates an explicit exit
receipt and a new report selection, and never launches/retries training. The
old frozen selection and the already prepared pool publisher stay unchanged.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import runpy


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
LOCAL = ROOT/'output/v7_local_ablation_20261005'
SELECTION = LOCAL/'mixed_report_selection_v2.json'
SELECTION_SHA = '9a60ebd56551c71c16faf86dd5efd4c44448e71fe3a44de31c86221a089dc0fc'
BASE = LOCAL/'f010_seed42_terminal_evidence_20261006'
REPORTER = LOCAL/'post_generation_pool_v1/source_v1/plot_v7_mixed_ablation.py'
REPORTER_FILES = {
    'plot_v7_mixed_ablation.py': '87e7e6b86d7400ffd467cb7002d9206416d6974a86e9e75a062bb389eb278a0e',
    'plot_v7_large_batch_ablation.py': '0a7f7d927ac35cf6db312faa53e93a4b3a98be12ff0c24e310c3510c9e355dd8',
    'plot_v7_context_ablation.py': '8bfea63868a0422996eb13a9ffc893b607b488ee183831f720076608675940df',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    require(path.is_absolute() and path == path.resolve() and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)),
            'Canonical real evidence required: '+str(path))
    return dict(path=str(path), sha256=sha(path))


def read(reference):
    path = Path(reference['path'])
    require(ref(path) == reference, 'Bound metadata changed: '+str(path))
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    require(not BASE.exists(), 'Existing or partial terminal collection must be reviewed, never replayed')
    selection = read(dict(path=str(SELECTION), sha256=SELECTION_SHA))
    items = [row for row in selection['cells'] if (row['variant'], row['seed']) == ('f010', 42)]
    require(len(items) == 1, 'Owned original cell must be unique')
    item = items[0]
    require(item['transport'] == 'local' and item['attempt_id'] == 'f010_seed42_attempt1'
            and item['run_dir'] == str(LOCAL/'f010_seed42_attempt1')
            and item['exit_evidence'] is None, 'Different or already recorded local attempt')
    launch = read(item['launch_receipt'])
    require(launch['schema'] == 'v7.local.tmux_launch.v2'
            and launch['task_name'] == 'v7_local_f010_seed42_attempt1', 'Wrong managed launcher')
    execution = read(launch['guard_execution']); identity = execution['process_identity']
    require(identity['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'Do not interpret an old process identity from a different host boot')
    stat = Path('/proc')/str(identity['pid'])/'stat'
    if stat.exists():
        fields = stat.read_text().rsplit(')', 1)[1].split()
        require(fields[0] == 'Z' or int(fields[19]) != identity['start_ticks'],
                'The owned training process is still live; no exit claim')
    log = Path(launch['log_dir'])
    require(log == ROOT/'logs/v7_local_f010_seed42_attempt1', 'Actual runner log directory differs')
    code_ref = ref(log/'exit_code'); raw = (log/'exit_code').read_text()
    require(re.fullmatch(r'(0|[1-9][0-9]{0,2})\n', raw) is not None and int(raw) <= 255,
            'Invalid runner exit code')
    require(int(raw) == 0 and (log/'done').is_file()
            and not (log/'failed').exists() and not (log/'failed').is_symlink(),
            'Runner did not successfully finish; preserve evidence without marking completed')
    done_ref = ref(log/'done')
    for filename, expected in REPORTER_FILES.items():
        require(sha(REPORTER.parent/filename) == expected, 'Frozen report helper changed')
    api = runpy.run_path(str(REPORTER), run_name='_owned_local_completion_reader')
    before = api['collect'](SELECTION, 3750)
    local = next(row for row in before['runs'] if (row['variant'], row['seed']) == ('f010', 42))
    require(local['artifact_complete'] and local['latest_evaluation_step'] == 3750
            and local['provenance']['checkpoint_saving'] is False,
            'Owned attempt lacks a complete native metrics-only epoch')
    require(not list(Path(item['run_dir']).glob('checkpoint-*'))
            and not list(Path(item['run_dir']).glob('*.pt'))
            and not list(Path(item['run_dir']).glob('*.safetensors')), 'Unexpected local checkpoint artifacts')
    keys = ('runner', 'task_name', 'command', 'started', 'admission', 'ledger',
            'guard_source', 'guard_argv', 'guard_execution')
    receipt = dict(schema='v7.local.tmux_exit.v2', at_utc=datetime.now(timezone.utc).isoformat(),
        launch_receipt=item['launch_receipt'], **{key: launch[key] for key in keys},
        exit_code=code_ref, done=done_ref, failed=None)
    # All ownership, real process/exit and native completion checks precede writes.
    BASE.mkdir()
    exit_path = BASE/'actual_exit_receipt_v2.json'; write_new(exit_path, receipt)
    item['exit_evidence'] = ref(exit_path)
    selection_path = BASE/'mixed_report_selection_v3.json'; write_new(selection_path, selection)
    report = api['collect'](selection_path, 3750)
    local = next(row for row in report['runs'] if (row['variant'], row['seed']) == ('f010', 42))
    require(local['terminal_verified'] and local['actual_returncode'] == 0,
            'New report selection did not validate the real owned exit')
    old_unknown = next(row for row in report['runs'] if (row['variant'], row['seed']) == ('f000', 42))
    require(old_unknown['actual_returncode'] is None and not old_unknown['exit_verified'],
            'Do not infer a different historical attempt exit')
    write_new(BASE/'terminal_review.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        status='actual_f010_seed42_full_epoch_and_runner_exit_verified', step=3750, records=120000,
        actual_returncode=0, process_identity=identity, original_selection=ref(SELECTION),
        new_selection=ref(selection_path), actual_exit=ref(exit_path), recorder=ref(Path(__file__).resolve()),
        reporter_files={str(REPORTER.parent/name): digest for name, digest in REPORTER_FILES.items()},
        native_complete=ref(Path(item['run_dir'])/'COMPLETE.json'),
        complete_artifacts=sum(r['artifact_complete'] for r in report['runs']),
        verified_successful_exits=sum(r['terminal_verified'] for r in report['runs']),
        historical_f000_exit='unknown_preserved', remaining_pending=sum(r['transport']=='pending' for r in report['runs']),
        source_homogeneous=report['source_homogeneous'], legacy_final=False, no_local_checkpoint_artifacts=True,
        old_selection_and_frozen_pool_publisher_unchanged=True, new_model_runs=0,
        later_pool_report_action='Copy the actual local exit ref into a new combined selection; never edit an already-frozen source selection.'))
    print(json.dumps(dict(status='actual_local_completion_recorded', evidence=ref(BASE/'terminal_review.json'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
