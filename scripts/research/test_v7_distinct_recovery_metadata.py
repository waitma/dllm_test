"""Check the new recovery lineage using artificial metadata, never a launch.

Activate pllm, then run once:
CUDA_VISIBLE_DEVICES='' python -I -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_distinct_recovery_metadata.py
This creates only an owned temporary timing fixture and an evidence JSON.
"""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile
import sys
import time
import traceback

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
EPOCH = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006'
BASE = EPOCH/'recovery_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and sys.dont_write_bytecode, 'Empty CUDA mask and -B required')
    evidence = BASE/'cpu_metadata_controls_v1.json'
    require(not evidence.exists(), 'Do not repeat completed metadata controls')
    sources = [BASE/'source_v1'/name for name in
               ('recovery_common.py', 'launch_v7_recovery_segment.py', 'supervise_v7_recovery.py')]
    refs = {str(p): sha(p) for p in [Path(__file__).resolve(), *sources]}
    common, launcher, supervisor = [runpy.run_path(str(p), run_name='_unbound_metadata_only') for p in sources]
    original = EPOCH/'launcher_v1/launch_v7_epoch2_segment.py'
    started = time.monotonic()
    checks, refusals, failure = [], [], None

    def reject(name, function):
        try:
            function()
        except ValueError as exc:
            refusals.append(dict(name=name, actual_error=str(exc)))
        else:
            raise AssertionError('Expected refusal: ' + name)

    try:
        jobs = ['synthetic_job_' + str(i) for i in range(8)]
        group = dict(phase='terminal', terminal='aborted',
                     abort=dict(reason='Coordinator lock wait expired'),
                     exits={str(i): {} for i in range(8)})
        results = {job: dict(job_id=job, state='failed', returncode=1) for job in jobs}
        reports = [dict(schema='v7.formal.rank_supervision.v1', rank=i, error='synthetic CPU error',
            observation=dict(real_training_started=False), child=dict(pid=100+i, parent_pid=200+i),
            parent=dict(pid=200+i), cleanup=dict(pid=100+i, returncode=-15, cleanup_completed=True,
            wait_observed=True, process_absent=True, process_group_gone=True)) for i in range(8)]
        accepted = common['validate_failed_snapshot'](group, results, reports, jobs)
        require(accepted == dict(failure_preserved=True, failed_ranks_cleaned=8, training_restarted=False),
                'Metadata check unexpectedly granted execution')
        checks.append('complete_artificial_failure_receipts_checked_without_grant')
        for field in ('cleanup_completed', 'wait_observed', 'process_absent', 'process_group_gone'):
            changed = copy.deepcopy(reports);changed[3]['cleanup'][field] = False
            reject(field, lambda rows=changed: common['validate_failed_snapshot'](group, results, rows, jobs))
        changed = copy.deepcopy(reports);changed[4]['observation']['real_training_started'] = True
        reject('unknown_failed_training_exposure', lambda: common['validate_failed_snapshot'](group, results, changed, jobs))
        changed = copy.deepcopy(reports);changed[5]['rank'] = 4
        reject('duplicate_rank_receipt', lambda: common['validate_failed_snapshot'](group, results, changed, jobs))
        changed_results = copy.deepcopy(results);changed_results[jobs[0]]['returncode'] = None
        reject('unknown_exit', lambda: common['validate_failed_snapshot'](group, changed_results, reports, jobs))
        changed_group = copy.deepcopy(group);changed_group['terminal'] = None
        reject('not_terminal', lambda: common['validate_failed_snapshot'](changed_group, results, reports, jobs))
        checks.append('incomplete_cleanup_or_unknown_exposure_refused')
        require([common['validate_edge'](*edge) for edge in common['EDGES']] == [8, 5000, 24], 'Budget differs')
        for edge in ((45050,50050), (45050,50082), (45058,50082), (50082,75000), (45050.,45058)):
            reject('edge_' + str(edge), lambda pair=edge: common['validate_edge'](*pair))
        require(common['previous_stage'](45050) == EPOCH/'launches_v1/segment_45050', 'Wrong accepted anchor')
        require(common['previous_stage'](45058) == BASE/'launches_v1/segment_45058', 'Wrong distinct lineage')
        checks.append('only_new_fixed_edges_with_verified_predecessors')
        def definitions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef)}
        before, after = definitions(original), definitions(sources[1])
        require({name for name in before if before[name] != after[name]} == {'validate_plan','prepare','enqueue'},
                'Unrelated original launcher behavior changed')
        require(set(after)-set(before) == {'recovery'}, 'Unexpected new launcher functions')
        require(launcher['WRAPPER'] == EPOCH/'wrapper_v1/formal_pool_job.py'
                and launcher['SOURCE'] == EPOCH/'source_snapshot_v1' and launcher['RUN'] == EPOCH/'run'
                and launcher['STAGES'] == BASE/'launches_v1', 'Training or stage binding differs')
        checks.append('original_cleanup_and_training_sources_preserved')
        with tempfile.TemporaryDirectory(prefix='recovery_metadata_', dir=BASE) as tmp:
            timing = Path(tmp)/'timing.jsonl'
            timing.write_bytes(b'{"step":45050}\n{"step":4505')
            require(supervisor['latest_complete_step'](timing,45050)==45050, 'Partial tail counted')
            timing.write_bytes(b'{"step":45050}\n{"step":45058}\n')
            require(supervisor['latest_complete_step'](timing,45050)==45058, 'Actual complete update lost')
            timing.write_bytes(b'{"step":"45058"}\n')
            reject('malformed_timing',lambda:supervisor['latest_complete_step'](timing,45050))
        require(not Path(tmp).exists(), 'Owned temporary fixture not removed')
        checks.append('complete_actual_timing_only_and_owned_cleanup')
        require(all(sha(Path(p)) == s for p,s in refs.items()), 'Source changed during checks')
        require(not {'torch','numpy','transformers','accelerate','dllm'} & {n.split('.')[0] for n in sys.modules},
                'Unexpected ML import')
    except BaseException:
        failure = traceback.format_exc()
    result = dict(at_utc=datetime.now(timezone.utc).isoformat(),status='passed' if failure is None else 'failed',
        failure=failure,checks=checks,refusals=refusals,source_sha256=refs,elapsed_seconds=time.monotonic()-started,
        original_launcher=dict(path=str(original),sha256=sha(original)),synthetic_only=True,
        launcher_executed=False,supervisor_started=False,GPU_tested=False,actual_recovery_verified=False)
    with evidence.open('x') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(evidence),sha256=sha(evidence),status=result['status'],failure=failure,
                         checks=len(checks),refusals=len(refusals)),ensure_ascii=False))
    return 0 if failure is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
