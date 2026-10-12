"""Read the actual two-epoch and generation evidence before later ablations.

Activate conda pllm; call check(catalog) from the SHA-bound v2 pool adapter.
This is a read-only stdlib helper, not a training launcher or certificate maker.
Artificial boundary tests: python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_post_generation_acceptance.py
The publisher must separately verify live supervisor exit, idle workers and
resource deadlines before publishing. This helper also runs inside GPU workers,
so it never interprets root-host PIDs in a different process namespace.
"""
import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
FORMAL = ROOT/'output/v7_formal_eight_gpu_20261002'
PARENT = FORMAL/'where_continuation_20261006'
EPOCH = FORMAL/'epoch2_continuation_20261006'
GENERATION = EPOCH/'generation_pool_v1'
POOL = ROOT/'output/immune_research_pool_c20250601_20261005'
OLD_SUPERVISOR = ROOT/'output/v7_formal_supervision_20261005'
PACKAGE = GENERATION/'package_v1.json'
PACKAGE_SHA = 'a27178a626d6de698bad6a2fc42bec4ff3eff022a49a71a6504dd9f64a56dc23'
DISPATCHER = GENERATION/'source_v1/pool_v7_generation_dispatch.py'
CELLS = tuple((dataset, mode) for dataset, modes in (
    ('kong', ('cdrh1', 'cdrh2', 'cdrh3')),
    ('sab23', ('cdrl1', 'cdrl2', 'cdrl3', 'cdrh1', 'cdrh2', 'cdrh3'))) for mode in modes)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve()
            and not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file(),
            'Required canonical evidence missing: '+str(path))
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(reference, expected=None):
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'}, 'Explicit evidence reference required')
    path = Path(reference['path'])
    require(expected is None or path == expected, 'Evidence path differs: '+str(path))
    value = read(path)
    require(sha(path) == reference['sha256'], 'Evidence changed: '+str(path))
    return value


def successful_exit(reference, job_id):
    result = bound(reference, POOL/'results'/(job_id+'.json'))
    require(result.get('job_id') == job_id and result.get('state') == 'succeeded'
            and type(result.get('returncode')) is int and result['returncode'] == 0,
            'Actual pool exit is not successful: '+job_id)
    require(not any((POOL/folder/(job_id+'.json')).exists() for folder in ('pending', 'running')),
            'Completed ticket still active: '+job_id)
    return result


def check(catalog):
    """Require real training, eight owned exits and all nine evaluated tasks."""
    # The old supervisor was intentionally stopped for the authorized code
    # migration. Its STOP must remain; it is not the current training owner.
    require((OLD_SUPERVISOR/'STOP').is_file(), 'Original supervisor stop must remain')
    for directory in (PARENT, EPOCH, GENERATION):
        require(not (directory/'STOP').exists() and not (directory/'SUPERVISION_STOPPED.json').exists(),
                'Current stage has stopped: '+str(directory))
    require(not any((POOL/name).exists() for name in ('STOP', 'STOP_NOW')), 'Pool is stopped')
    training = bound(catalog['formal_completion'], EPOCH/'COMPLETE.json')
    require(training.get('status') == 'actual_world8_epoch2_continuation_passed'
            and training.get('step') == 50082 and training.get('prior_step') == 50050
            and training.get('epoch') == 2 and training.get('eight_native_restores_and_owned_exits') is True,
            'Actual cumulative two-epoch endpoint is not complete')
    stage = EPOCH/'launches_v1/segment_50082'
    require(catalog.get('formal_stage') == str(stage), 'Wrong final two-epoch stage')
    state = read(stage/'meeting/coordinator/state.json')
    require(state.get('terminal') == 'succeeded' and state.get('abort') is None, 'Final training group failed')
    jobs = read(stage/'enqueue_receipt.json')['job_ids']
    results = training.get('actual_pool_results')
    require(isinstance(results, list) and len(results) == len(jobs) == len(set(jobs)) == 8
            and {str(POOL/'results'/(job+'.json')) for job in jobs} == {item['path'] for item in results},
            'Actual eight training exits are incomplete or duplicated')
    for item in results:
        successful_exit(item, Path(item['path']).stem)
    for rank in range(8):
        receipt = read(stage/f'meeting/coordinator/rank{rank}_parent_receipt.json')
        cleanup = receipt.get('cleanup', {})
        require(receipt.get('rank') == rank and receipt.get('error') is None
                and type(cleanup.get('returncode')) is int and cleanup['returncode'] == 0
                and all(cleanup.get(k) is True for k in ('wait_observed', 'process_absent',
                    'process_group_gone', 'cleanup_completed')), 'Final training rank cleanup is incomplete')
    release = bound(catalog['generation_release'], GENERATION/'release.json')
    require(release.get('schema') == 'v7.cdr.pool_release.v1' and release.get('work_type') == 'generation_evaluation'
            and release.get('optimizer_updates') == 0 and release.get('automatic_retry') is False
            and release.get('training_acceptance') == catalog['formal_completion'], 'Generation release differs')
    package = bound(catalog['generation_package'], PACKAGE)
    require(catalog['generation_package']['sha256'] == PACKAGE_SHA, 'Original generation package changed')
    for name, expected in package['files'].items():
        require(sha(name) == expected and release['files'].get(name) == expected,
                'Frozen generation dependency changed: '+name)
    # The public entry may later be extended for ablations. Its previous bytes
    # remain explicitly bound by the original generation release and package.
    for public, candidate in ((ROOT/'scripts/run_immune_research.py', GENERATION/'public_after_candidate.py'),
                              (ROOT/'dllm/pipelines/immune_llada/research/model.py', GENERATION/'registry_after_candidate.py')):
        require(release['files'][str(public)] == package['files'][str(candidate)], 'Generation public-source history differs')
    api = runpy.run_path(str(DISPATCHER), run_name='_post_generation_native_evidence')
    api['completion'](release)  # Retain the exact original readiness/audit checks.
    done = bound(catalog['generation_completion'], GENERATION/'COMPLETE.json')
    require(done.get('status') == 'nine_actual_cdr_tasks_passed'
            and done.get('checkpoint') == training['checkpoint']
            and done.get('total_cdr_cases') == 9741 and done.get('all_actual_pool_and_evaluator_exits_zero') is True,
            'Actual generation batch is not complete')
    tasks = done.get('tasks')
    require(isinstance(tasks, list) and len(tasks) == 9
            and {tuple(t['cell']) for t in tasks} == set(CELLS), 'Missing or duplicate generation task')
    published = bound(release['catalog'], GENERATION/'catalog.json')
    rows = published.get('jobs', [])
    require(len(rows) == 9 and {tuple(t['cell']) for t in rows} == set(CELLS)
            and len({t['job_id'] for t in rows}) == 9, 'Generation catalog coverage differs')
    for task in tasks:
        cell = tuple(task['cell'])
        row = next(row for row in rows if tuple(row['cell']) == cell)
        job_id = row['job_id']
        successful_exit(task['actual_pool_result'], job_id)
        complete = bound(task['actual_completion'], POOL/'runs'/job_id/'COMPLETE.json')
        require(complete.get('status') == 'generation_evaluation_complete'
                and complete.get('work_type') == 'generation_evaluation' and complete.get('optimizer_updates') == 0
                and complete.get('checkpoint') == training['checkpoint'], 'Actual evaluator completion differs')
        exited = bound(complete['actual_evaluator_exit'], POOL/'runs'/job_id/'EVALUATION_EXIT.json')
        require(type(exited.get('returncode')) is int and exited['returncode'] == 0, 'Actual evaluator exit failed')
        output = Path(row['evaluation_output'])
        metrics = bound(task['metrics'], output/cell[1]/'metrics.json')
        summary = bound(complete['summary'], output/'summary.json')
        count = 3127 if cell[0] == 'kong' else 60
        require(complete['metrics'] == task['metrics'] and metrics.get('status') == 'complete'
                and metrics.get('n_sequences') == metrics.get('n_expected') == task.get('n_sequences') == count
                and metrics.get('n_missing') == 0, 'Actual task coverage differs')
        require(summary.get('status') == 'complete' and summary.get('step') == 50082
                and summary.get('dataset') == cell[0] and summary.get('coverage') == 'assigned_mode'
                and summary.get('assigned_mode') == cell[1], 'Actual task summary differs')
        audit = output/cell[1]/'input_audit.jsonl'
        require(task['input_audit'] == {'path': str(audit), 'sha256': sha(audit)}, 'Saved input audit changed')
    return {'status': 'actual_training_and_generation_evidence_passed', 'training_step': 50082,
            'generation_tasks': 9, 'cdr_cases': 9741, 'new_model_tests': 0}
