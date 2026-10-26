"""Supervise the existing where-DDP segment and admit only verified continuations.

Activate conda pllm, then launch the frozen copy once in a new managed tmux job.
Starts by observing the already-enqueued7008->7016 segment. Thereafter continues
bounded5000-update segments to25042 in the same allocation. BASE/STOP prevents
future admissions. No platform submit/cancel, failed-attempt retry, or original
supervisor restart. Checkpoint deletion belongs exclusively to the Trainer policy.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
import uuid

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
BASE = ROOT/'output/v7_formal_eight_gpu_20261002/where_continuation_20261006'
POOL = ROOT/'output/immune_research_pool_c20250601_20261005'
RUN = BASE/'run'
SOURCE = BASE/'source_snapshot_v1'
LAUNCHER = BASE/'launcher_v2/launch_v7_where_segment.py'
LAUNCHER_SHA = '3857fd5bbe9a9fe06e9fa6d508e6ac87bad5656298cf5fe44c3ee6e8df55bc0f'
MANIFEST_SHA = '053f95d2237eed6f4bd45d53ea07126447c8ddaf4efa67f62c571f94361d0c5b'
TRANSITION_SHA = '52cbc3431d5c3fa4bf6e856d39711b5de3c22d5b8894dfbfe5fff05662f8c197'


def require(ok, message):
    if not ok:
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
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def fixed():
    for p, expected in ((LAUNCHER, LAUNCHER_SHA), (BASE/'source_manifest_v1.json', MANIFEST_SHA),
                        (BASE/'transition_v1.json', TRANSITION_SHA)):
        require(sha(p) == expected, 'Continuation supervisor dependency changed')


def event(message, stage):
    value = dict(at_utc=now(), message=message, stage=str(stage))
    with (BASE/'supervision_events.jsonl').open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+'\n')
    path = ROOT/'PROJECT_PROCESS.md'
    with path.open('a') as stream:
        stream.write(f'\n\n## {value["at_utc"]} where正式接续监督节点\n\n{message}\n\n原件 {BASE}/supervision_state.json；{BASE}/supervision_events.jsonl。\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def accept(stage, launcher, native, driver):
    receipt = launcher['cleanup_checked'](stage, success=True)
    prior, step = receipt['prior_step'], receipt['stop_after']
    group = read(stage/'meeting/coordinator/state.json')
    require(group['terminal'] == 'succeeded' and group['abort'] is None, 'New group not successful')
    binding = read(RUN/'run_provenance.json')
    marker_path = RUN/('COMPLETE.json' if step == 25042 else f'SEGMENT_{step}.json')
    marker = read(marker_path)
    resume_path = RUN/f'RESUME_{prior}_TO_{step}.json';resume = read(resume_path)
    require(native['complete_receipt'](resume['ranks']) == resume, 'Native eight-rank restored state mismatch')
    require(resume['prior_step'] == prior and resume['final_step'] == step
            and resume['binding_sha256'] == marker['binding_sha256'] == driver['digest'](binding),
            'Continuation step/binding mismatch')
    cp = RUN/f'checkpoint-{step}';side = read(cp/'formal_checkpoint.json')
    require(side['step'] == step and side['binding_sha256'] == driver['digest'](binding)
            and side['rank_acknowledgments'] == list(range(8)) and len(side['files']) == 17,
            'Native checkpoint incomplete')
    require(side['rank_parameter_digests'] == [dict(rank=n, **side['saved_parameter_digest']) for n in range(8)],
            'Rank parameters differ')
    for name, expected in side['files'].items():
        p = cp/name
        require(p.is_file() and not p.is_symlink() and p.stat().st_size > 0, 'Native file missing')
        if name.endswith('.json'):
            require(sha(p) == expected, 'Native JSON changed')
    retention = read(RUN/'checkpoint_retention.json')
    require(retention['status'] == 'applied' and retention['latest'] == str(cp)
            and len(retention['keep']) <= 4 and retention['metric'] == 'eval_loss'
            and retention['best_k'] == 3 and retention['all_retained_states'] == 'full', 'Retention policy differs')
    require(all(Path(x['path']).is_dir() for x in retention['keep'])
            and all(not Path(x['path']).exists() for x in retention['remove']), 'Retention actual state differs')
    report = dict(at_utc=now(), status='actual_world8_continuation_passed', prior_step=prior, step=step,
        native_resume=ref(resume_path), marker=ref(marker_path), checkpoint=ref(cp/'formal_checkpoint.json'),
        retention=ref(RUN/'checkpoint_retention.json'), plan=ref(stage/'plan.json'),
        actual_pool_results=[ref(POOL/'results'/f'{j}.json') for j in receipt['job_ids']],
        eight_native_restores_and_owned_exits=True, full_checkpoint_bytes_verified_by='Eight native slot completion callbacks')
    target = BASE/f'segment{step}_acceptance.json'
    require(not target.exists(), 'Accepted segment must not be replayed')
    write(target, report)
    return report


def main():
    fixed()
    require((ROOT/'output/v7_formal_supervision_20261005/STOP').exists(), 'Old auto-continuation must stay stopped')
    require(read(BASE/'segment7008_root_review.json')['status'] == 'actual_world8_where_migration_passed',
            'Initial actual migration must already be accepted')
    with (BASE/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (BASE/'supervisor_launch.json').exists(), 'Never silently relaunch this supervisor')
        fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        write(BASE/'supervisor_launch.json', dict(at_utc=now(), pid=os.getpid(), start_ticks=int(fields[19]),
              argv=sys.argv, source=ref(Path(__file__).resolve())))
        launcher = runpy.run_path(str(LAUNCHER), run_name='_where_bound_launcher')
        native = runpy.run_path(str(SOURCE/'scripts/research/formal_resume_observation.py'), run_name='_where_native_receipts')
        driver = runpy.run_path(str(SOURCE/'scripts/research/run_v7_formal.py'), run_name='_where_driver_metadata')
        old = runpy.run_path(str(ROOT/'output/v7_formal_supervision_20261005/source_v1/supervise_v7_formal_20261005.py'), run_name='_existing_readonly_platform')
        stage = BASE/'launches_v2/segment_7016'
        deadline = time.monotonic()+5*24*3600
        accepted = None
        event('where持续监督开始；只观察已入队7016段，未重放启动。', stage)
        try:
            while time.monotonic() < deadline:
                require(not (BASE/'STOP').exists(), 'Requested stop of future continuation')
                receipt = read(stage/'enqueue_receipt.json')
                state_path = stage/'meeting/coordinator/state.json'
                state = read(state_path) if state_path.exists() else {}
                results = {j: read(POOL/'results'/f'{j}.json') for j in receipt['job_ids'] if (POOL/'results'/f'{j}.json').exists()}
                timings = (RUN/'step_timing.jsonl').read_bytes().splitlines()
                try:
                    latest_step = json.loads(timings[-1])['step']
                except (IndexError, json.JSONDecodeError):
                    latest_step = json.loads(timings[-2])['step'] if len(timings)>1 else receipt['prior_step']
                current = dict(at_utc=now(), stage=str(stage), target_step=receipt['stop_after'],
                    prior_step=receipt['prior_step'], latest_timed_step=latest_step, phase=state.get('phase'),
                    terminal=state.get('terminal'), abort=state.get('abort'), completed_slots=len(results),
                    accepted_step=accepted['step'] if accepted else 7008, goal_target_step=25042)
                write(BASE/'supervision_state.json', current)
                require(not state.get('abort') and state.get('terminal') != 'aborted'
                        and all(x['state']=='succeeded' and x['returncode']==0 for x in results.values()),
                        'Actual continuation failed; preserve evidence, do not retry')
                if len(results) == 8 and state.get('terminal') == 'succeeded':
                    fixed()
                    accepted = accept(stage, launcher, native, driver)
                    event(f"实际累计{accepted['step']}步通过8rank恢复/保存/退出与best3+latest；准备下一段。",stage)
                    if accepted['step'] == 25042:
                        write(BASE/'COMPLETE.json',accepted)
                        return
                    # This exact allocation must still be live; no platform retries or replacements.
                    platform=old['platform']();require(platform['task']['Status']=='Running','Original allocation is not Running')
                    write(BASE/'platform_latest.json',platform)
                    release_deadline=time.monotonic()+120
                    while True:
                        workers=[read(p) for p in (POOL/'workers').glob('*.json')]
                        released=(len(workers)==8 and not list((POOL/'running').glob('*.json'))
                            and not list((POOL/'pending').glob('*.json'))
                            and all(w['state']=='standby_load' and 0<=time.time()-w['time']<15 for w in workers))
                        if released:
                            break
                        require(time.monotonic()<release_deadline, 'Prior workers not released; do not duplicate admission')
                        time.sleep(2)
                    next_step=min(accepted['step']+5000,25042)
                    for command in ([sys.executable,'-B',str(LAUNCHER),'prepare','--resume-checkpoint',str(RUN/f"checkpoint-{accepted['step']}"),'--stop-after',str(next_step)],
                                    [sys.executable,'-B',str(LAUNCHER),'enqueue','--stage',str(BASE/'launches_v2'/f'segment_{next_step}')]):
                        require(not (BASE/'STOP').exists(),'STOP before admission')
                        result=subprocess.run(command,capture_output=True,text=True,timeout=300)
                        with (BASE/'launcher_events.log').open('a') as stream:
                            stream.write(json.dumps(dict(at_utc=now(),command=command,returncode=result.returncode))+'\n'+result.stdout+result.stderr)
                        require(result.returncode==0,'Continuation launcher did not succeed; inspect, never replay automatically')
                    stage=BASE/'launches_v2'/f'segment_{next_step}'
                    event(f'同一八卡已单次接续到累计目标{next_step}；总目标25042不变。',stage)
                time.sleep(10)
            raise TimeoutError('Bounded supervision deadline reached; original job is not restarted')
        except BaseException as error:
            write(BASE/'SUPERVISION_STOPPED.json',dict(at_utc=now(),error=repr(error),stage=str(stage),
                  active_training_not_restarted_or_killed=True))
            event('监督停止后续准入，保留当前任务：'+repr(error),stage)
            raise


if __name__ == '__main__':
    main()
