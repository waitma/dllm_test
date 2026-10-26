"""Wait for two complete epochs, then supervise all nine CDR tasks in the same pool.

Activate conda pllm; launch the frozen copy once through scripts/run_in_tmux.sh.
This does not restart training/containers/workers, retry failures, or run any
annotation branch. Before actual training completion it only reads status.
BASE/STOP prevents publication. Original worker deadlines remain unchanged.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import uuid

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
EPOCH = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006'
PARENT = EPOCH.parent/'where_continuation_20261006'
BASE = EPOCH/'generation_pool_v1'
POOL = ROOT/'output/immune_research_pool_c20250601_20261005'
PACKAGE = BASE/'package_v1.json'
PUBLISHER = BASE/'source_v1/publish_v7_generation_pool.py'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def live(directory):
    receipt = read(directory/'supervisor_launch.json')
    path = Path('/proc')/str(receipt['pid'])/'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] != 'Z' and int(fields[19]) == receipt['start_ticks']


def fixed(package_sha):
    require(sha(PACKAGE) == package_sha, 'Generation package changed')
    for filename, expected in read(PACKAGE)['files'].items():
        require(sha(Path(filename)) == expected, 'Frozen generation dependency changed: '+filename)


def event(message):
    value = dict(at_utc=now(), message=message)
    with (BASE/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+'\n')
    for path in (ROOT/'PROJECT_PROCESS.md', ROOT/'downstream/benchmark/PROGRESS.md'):
        with path.open('a') as stream:
            stream.write(f'\n\n## {value["at_utc"]} V7两轮后生成执行节点\n\n{message}\n\n原件 {BASE}/state.json；{BASE}/events.jsonl。\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def check_result(row, result, score_only):
    require(result['job_id'] == row['job_id'] and result['state'] == 'succeeded' and result['returncode'] == 0,
            'Actual pool evaluation did not exit successfully')
    run = POOL/'runs'/row['job_id']
    complete = read(run/'COMPLETE.json')
    require(complete['status'] == 'generation_evaluation_complete' and complete['work_type'] == 'generation_evaluation'
            and complete['optimizer_updates'] == 0, 'Missing true evaluation completion')
    exit_path = run/'EVALUATION_EXIT.json'
    require(complete['actual_evaluator_exit'] == ref(exit_path) and read(exit_path)['returncode'] == 0,
            'Evaluator exit was not verified')
    output = Path(row['evaluation_output'])
    dataset, mode = row['cell']
    summary_path = output/'summary.json'
    require(ref(summary_path) == complete['summary'], 'Generation summary changed')
    summary = read(summary_path)
    require(summary['status'] == 'complete' and summary['dataset'] == dataset and summary['step'] == 50082
            and summary['coverage'] == 'assigned_mode' and summary['assigned_mode'] == mode,
            'Generation task coverage differs')
    metric_path = output/mode/'metrics.json'
    require(complete['metrics'] == ref(metric_path), 'Saved generation metric changed')
    score = score_only(output/mode)
    require(score == read(metric_path) and score['status'] == 'complete', 'Prediction rescore mismatch')
    expected = 3127 if dataset == 'kong' else 60
    require(score['n_sequences'] == score['n_expected'] == expected and score['n_missing'] == 0,
            'Assigned task is truncated or has the wrong dataset size')
    audit_path = output/mode/'input_audit.jsonl'
    audit = [json.loads(line) for line in audit_path.read_text().splitlines()]
    require(len(audit) == expected and len({(x['fold'], x['row']) for x in audit}) == expected
            and all(x['hidden_reference_perturbation_equal'] and x['forward_calls'] == 1
                    and x['labels_provided'] is False for x in audit), 'Actual input evidence is incomplete')
    return dict(cell=row['cell'], actual_pool_result=ref(POOL/'results'/(row['job_id']+'.json')),
                actual_completion=ref(run/'COMPLETE.json'), metrics=ref(metric_path), input_audit=ref(audit_path),
                n_sequences=expected, average_aar_all_folds=score['average_aar_all_folds'])


def main():
    package_sha = sha(PACKAGE)
    fixed(package_sha)
    api = runpy.run_path(str(PUBLISHER), run_name='_generation_publisher_metadata')
    with (BASE/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (BASE/'supervisor_launch.json').exists(), 'Do not relaunch this supervisor')
        fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        write(BASE/'supervisor_launch.json', dict(at_utc=now(), pid=os.getpid(), start_ticks=int(fields[19]),
              source=ref(Path(__file__).resolve()), package=ref(PACKAGE)))
        deadline = time.monotonic()+5*24*3600
        event('九个CDR生成任务的监督已启动；只等待两轮真实完成，当前没有发布或入队。')
        try:
            while True:
                api['stops']()
                require(time.monotonic() < deadline, 'Bounded generation wait expired; no restart')
                require(not (EPOCH/'SUPERVISION_STOPPED.json').exists(), 'Epoch-two supervision stopped')
                state = read(EPOCH/'supervision_state.json')
                write(BASE/'state.json', dict(at_utc=now(), phase='waiting_two_epochs',
                    latest_training_step=state.get('latest_timed_step'), required_checkpoint_step=50082,
                    prepared_tasks=9, generation_started=False))
                if (EPOCH/'COMPLETE.json').exists() and not live(EPOCH) and not live(PARENT):
                    if api['released']():
                        break
                else:
                    require(live(EPOCH), 'Epoch-two supervisor disappeared without final acceptance')
                    age = time.time()-datetime.fromisoformat(state['at_utc']).timestamp()
                    require(0 <= age < 900, 'Epoch-two supervision heartbeat is stale')
                time.sleep(10)
            fixed(package_sha)
            result = subprocess.run([sys.executable, '-B', str(PUBLISHER)], capture_output=True, text=True, timeout=300)
            with (BASE/'publication.log').open('x') as stream:
                stream.write(json.dumps(dict(at_utc=now(), returncode=result.returncode))+'\n'+result.stdout+result.stderr)
            require(result.returncode == 0, 'Post-training publication failed; do not replay')
            receipt = read(BASE/'enqueue_receipt.json')
            require(receipt['state'] == 'enqueued' and len(receipt['enqueued_job_ids']) == 9, 'Not all nine tasks were enqueued')
            catalog = read(BASE/'catalog.json')
            require(set(receipt['enqueued_job_ids']) == {r['job_id'] for r in catalog['jobs']}, 'Enqueued task set differs')
            event('两轮训练及所有训练进程退出已确认；九个固定CDR任务已在原八卡单次入队，没有新资源申请。')
            sys.path.insert(0, str(EPOCH/'generation_source_v3'))
            io = runpy.run_path(str(EPOCH/'generation_source_v3/downstream/grammar/cdr_result_io.py'), run_name='_generation_scorer')
            accepted = {}
            while time.monotonic() < deadline:
                api['stops']()
                for row in catalog['jobs']:
                    job = row['job_id']
                    path = POOL/'results'/(job+'.json')
                    if path.exists() and job not in accepted:
                        accepted[job] = check_result(row, read(path), io['score_artifacts'])
                        write(BASE/f'{row["cell"][0]}_{row["cell"][1]}_acceptance.json', accepted[job])
                        event(f'{row["cell"][0]}/{row["cell"][1]}实际生成、退出和逐行重评分通过；完成{len(accepted)}/9。')
                write(BASE/'state.json', dict(at_utc=now(), phase='evaluating' if len(accepted) < 9 else 'complete',
                    checkpoint_step=50082, completed_tasks=len(accepted), expected_tasks=9,
                    actual_results=list(accepted.values()), generation_started=True))
                if len(accepted) == 9:
                    require(sum(r['n_sequences'] for r in accepted.values()) == 9741, 'Full generation coverage differs')
                    write(BASE/'COMPLETE.json', dict(at_utc=now(), status='nine_actual_cdr_tasks_passed',
                        checkpoint=read(BASE/'release.json')['checkpoint'], tasks=list(accepted.values()),
                        total_cdr_cases=9741, all_actual_pool_and_evaluator_exits_zero=True,
                        leaderboard_publication_pending=True))
                    event('固定50082的九个CDR任务全部完成并重评分，共9741条；成绩待根端审核写入唯一RESULTS。')
                    return
                time.sleep(10)
            raise TimeoutError('Bounded generation supervision expired; no automatic retry')
        except BaseException as error:
            write(BASE/'SUPERVISION_STOPPED.json', dict(at_utc=now(), error=repr(error),
                existing_training_and_evaluations_not_restarted_or_killed=True))
            event('生成监督停止后续动作，保留真实证据：'+repr(error))
            raise


if __name__ == '__main__':
    main()
