"""Continue the existing allocation to two data epochs, then expose an eval node.

Activate conda pllm; launch the frozen copy once through scripts/run_in_tmux.sh.
Initially this only observes the existing parent. It cannot interrupt/restart it,
resubmit resources, retry a failed segment, or publish an unevaluated result.
BASE/STOP prevents further admissions. Checkpoint pruning stays with the Trainer.
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
PARENT = ROOT/'output/v7_formal_eight_gpu_20261002/where_continuation_20261006'
BASE = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006'
POOL = ROOT/'output/immune_research_pool_c20250601_20261005'
RUN = BASE/'run'
SOURCE = BASE/'source_snapshot_v1'
LAUNCHER = BASE/'launcher_v1/launch_v7_epoch2_segment.py'
SEALER = BASE/'sealer_v1/seal_v7_epoch2_continuation.py'
PACKAGE = BASE/'package_v1.json'
TARGET = 50082
FIRST = 25042


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
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def fixed(package_sha):
    require(sha(PACKAGE) == package_sha, 'Continuation package changed')
    for filename, expected in read(PACKAGE)['files'].items():
        path = Path(filename)
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                'Frozen dependency changed: '+filename)
    for filename, expected in read(BASE/'source_manifest_v1.json')['files'].items():
        require(sha(Path(filename)) == expected, 'Bound training source changed')
    if (BASE/'seal_receipt.json').exists():
        for key in ('transition', 'manifest', 'parent', 'original_parent_data_audit', 'initial_driver_argv', 'sealer'):
            item = read(BASE/'seal_receipt.json')[key]
            require(ref(Path(item['path'])) == item, 'Sealed transition evidence changed')


def process_matches(receipt):
    path = Path('/proc')/str(receipt['pid'])/'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] != 'Z' and int(fields[19]) == receipt['start_ticks']


def event(message, stage):
    value = dict(at_utc=now(), message=message, stage=str(stage))
    with (BASE/'supervision_events.jsonl').open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+'\n')
    with (ROOT/'PROJECT_PROCESS.md').open('a') as stream:
        stream.write(f'\n\n## {value["at_utc"]} 两轮训练接续节点\n\n{message}\n\n原件 {BASE}/supervision_state.json；{BASE}/supervision_events.jsonl。\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def stops():
    require(not any(p.exists() for p in (BASE/'STOP', PARENT/'STOP', POOL/'STOP', POOL/'STOP_NOW')),
            'Explicit stop prevents further admissions')
    require((ROOT/'output/v7_formal_supervision_20261005/STOP').exists(), 'Original stopped supervisor must stay stopped')


def released():
    workers = [read(path) for path in (POOL/'workers').glob('*.json')]
    return (len(workers) == 8 and {str(w['device']) for w in workers} == set(map(str, range(8)))
        and not list((POOL/'running').glob('*.json')) and not list((POOL/'pending').glob('*.json'))
        and all(w['state'] == 'standby_load' and 0 <= time.time()-w['time'] < 15 for w in workers))


def run_command(command):
    stops()
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    with (BASE/'launcher_events.log').open('a') as stream:
        stream.write(json.dumps(dict(at_utc=now(), command=command, returncode=result.returncode))+'\n'
                     +result.stdout+result.stderr)
    require(result.returncode == 0, 'Command failed; preserve evidence and do not replay: '+str(command))


def admit(checkpoint, target, package_sha, platform_api):
    fixed(package_sha)
    platform = platform_api['platform']()
    require(platform['task']['Status'] == 'Running', 'Original allocation is not Running')
    write(BASE/'platform_latest.json', platform)
    deadline = time.monotonic()+120
    while not released():
        stops()
        require(time.monotonic() < deadline, 'Previous eight slots not released; no duplicate admission')
        time.sleep(2)
    stage = BASE/'launches_v1'/f'segment_{target}'
    run_command([sys.executable, '-B', str(LAUNCHER), 'prepare', '--resume-checkpoint', str(checkpoint), '--stop-after', str(target)])
    run_command([sys.executable, '-B', str(LAUNCHER), 'enqueue', '--stage', str(stage)])
    event(f'原八卡单次接续：{checkpoint.name}→{target}；累计两轮目标50082。', stage)
    return stage


def restore_binding(resume, marker, binding_sha, prior, step, migration, transition, old_sha):
    require(resume['prior_step'] == prior and resume['final_step'] == step
            and marker['step'] == step and marker['binding_sha256'] == binding_sha,
            'Native restoration step or new binding differs')
    if prior == FIRST:
        require(migration is not None and migration['schema'] == 'v7.epoch2.native_migration.v1'
                and migration['transition'] == transition and migration['old_binding_sha256'] == old_sha
                and migration['new_binding_sha256'] == binding_sha and migration['native_resume'] == resume
                and resume['binding_sha256'] == old_sha, 'Explicit old-to-new native migration differs')
    else:
        require(resume['binding_sha256'] == binding_sha, 'Same-run native restore binding differs')


def accept(stage, launcher, native, driver):
    receipt = launcher['cleanup_checked'](stage, success=True)
    prior, step = receipt['prior_step'], receipt['stop_after']
    group = read(stage/'meeting/coordinator/state.json')
    require(group['terminal'] == 'succeeded' and group['abort'] is None, 'Group did not succeed')
    binding = read(RUN/'run_provenance.json')
    marker_path = RUN/('COMPLETE.json' if step == TARGET else f'SEGMENT_{step}.json')
    marker = read(marker_path)
    resume_path = RUN/f'RESUME_{prior}_TO_{step}.json'
    resume = read(resume_path)
    require(native['complete_receipt'](resume['ranks']) == resume, 'Native eight-rank restoration mismatch')
    migration = read(RUN/'MIGRATION_RESTORE.json') if prior == FIRST else None
    restore_binding(resume, marker, driver['digest'](binding), prior, step, migration,
                    ref(BASE/'transition_v1.json'), driver['digest'](read(BASE/'parent_provenance.json')))
    cp = RUN/f'checkpoint-{step}'
    side = read(cp/'formal_checkpoint.json')
    require(side['step'] == step and side['world_size'] == 8 and side['binding_sha256'] == driver['digest'](binding)
            and side['rank_acknowledgments'] == list(range(8)) and len(side['files']) == 17,
            'Incomplete native checkpoint')
    require(side['rank_parameter_digests'] == [dict(rank=n, **side['saved_parameter_digest']) for n in range(8)],
            'Saved and actual rank parameters differ')
    for name, expected in side['files'].items():
        path = cp/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0, 'Missing native file')
        if name.endswith('.json'):
            require(sha(path) == expected, 'Changed native checkpoint JSON')
    state = read(cp/'trainer_state.json')
    require(state['global_step'] == step and state['max_steps'] == TARGET
            and abs(state['epoch'] - step/25041) < 1e-9, 'Actual data epoch or horizon differs')
    retention = read(RUN/'checkpoint_retention.json')
    require(retention['status'] == 'applied' and retention['latest'] == str(cp)
            and len(retention['keep']) <= 4 and retention['metric'] == 'eval_loss'
            and retention['best_k'] == 3 and retention['all_retained_states'] == 'full', 'Retention policy differs')
    require(all(Path(x['path']).is_dir() for x in retention['keep'])
            and all(not Path(x['path']).exists() for x in retention['remove']), 'Retention actual state differs')
    report = dict(at_utc=now(), status='actual_world8_epoch2_continuation_passed', prior_step=prior, step=step,
        epoch=state['epoch'], native_resume=ref(resume_path), marker=ref(marker_path),
        checkpoint=ref(cp/'formal_checkpoint.json'), retention=ref(RUN/'checkpoint_retention.json'),
        plan=ref(stage/'plan.json'), actual_pool_results=[ref(POOL/'results'/f'{j}.json') for j in receipt['job_ids']],
        eight_native_restores_and_owned_exits=True,
        full_checkpoint_bytes_verified_by='Eight native slot completion callbacks')
    path = BASE/f'segment{step}_acceptance.json'
    require(not path.exists(), 'Do not repeat an accepted segment')
    write(path, report)
    return report


def main():
    package_sha = sha(PACKAGE)
    fixed(package_sha)
    stops()
    with (BASE/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (BASE/'supervisor_launch.json').exists(), 'Never silently relaunch this supervisor')
        fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        write(BASE/'supervisor_launch.json', dict(at_utc=now(), pid=os.getpid(), start_ticks=int(fields[19]),
              argv=sys.argv, source=ref(Path(__file__).resolve()), package=ref(PACKAGE)))
        stage = PARENT
        deadline = time.monotonic()+5*24*3600
        event('两轮接续监督已启动，只等待现有25042阶段自然完成；没有重启/提交/停止作业。', stage)
        try:
            parent_launch = read(PARENT/'supervisor_launch.json')
            while True:
                stops()
                require(time.monotonic() < deadline, 'Bounded waiting deadline reached')
                require(not (PARENT/'SUPERVISION_STOPPED.json').exists(), 'Parent supervisor stopped; no retry')
                state = read(PARENT/'supervision_state.json')
                write(BASE/'supervision_state.json', dict(at_utc=now(), phase='waiting_parent',
                    latest_timed_step=state.get('latest_timed_step'), parent_phase=state.get('phase'),
                    parent_target_step=FIRST, goal_target_step=TARGET, generation_eval_step=TARGET))
                if (PARENT/'COMPLETE.json').exists():
                    if not process_matches(parent_launch):
                        break
                else:
                    require(process_matches(parent_launch), 'Parent disappeared without final acceptance')
                    require(not state.get('abort'), 'Parent aborted; no retry')
                    age = time.time()-datetime.fromisoformat(state['at_utc']).timestamp()
                    require(0 <= age < 180, 'Parent supervision heartbeat is stale')
                time.sleep(10)
            fixed(package_sha)
            run_command([sys.executable, '-B', str(SEALER)])
            fixed(package_sha)
            launcher = runpy.run_path(str(LAUNCHER), run_name='_epoch2_launcher')
            native = runpy.run_path(str(SOURCE/'scripts/research/formal_resume_observation.py'), run_name='_epoch2_native')
            driver = runpy.run_path(str(SOURCE/'scripts/research/run_v7_formal.py'), run_name='_epoch2_driver')
            platform_api = runpy.run_path(str(ROOT/'output/v7_formal_supervision_20261005/source_v1/supervise_v7_formal_20261005.py'), run_name='_existing_readonly_platform')
            stage = admit(PARENT/'run/checkpoint-25042', 25050, package_sha, platform_api)
            accepted_step = FIRST
            while time.monotonic() < deadline:
                stops()
                receipt = read(stage/'enqueue_receipt.json')
                state_path = stage/'meeting/coordinator/state.json'
                state = read(state_path) if state_path.exists() else {}
                results = {j: read(POOL/'results'/f'{j}.json') for j in receipt['job_ids'] if (POOL/'results'/f'{j}.json').exists()}
                timing_path = RUN/'step_timing.jsonl'
                lines = timing_path.read_bytes().splitlines() if timing_path.exists() else []
                latest = receipt['prior_step']
                for line in reversed(lines[-2:]):
                    try:
                        latest = json.loads(line)['step']
                        break
                    except json.JSONDecodeError:
                        pass
                write(BASE/'supervision_state.json', dict(at_utc=now(), stage=str(stage),
                    target_step=receipt['stop_after'], prior_step=receipt['prior_step'], latest_timed_step=latest,
                    phase=state.get('phase'), terminal=state.get('terminal'), abort=state.get('abort'),
                    completed_slots=len(results), accepted_step=accepted_step,
                    goal_target_step=TARGET, generation_eval_step=TARGET))
                require(not state.get('abort') and state.get('terminal') != 'aborted'
                        and all(x['state'] == 'succeeded' and x['returncode'] == 0 for x in results.values()),
                        'Actual continuation failed; preserve evidence, no retry')
                if len(results) == 8 and state.get('terminal') == 'succeeded':
                    fixed(package_sha)
                    accepted = accept(stage, launcher, native, driver)
                    accepted_step = accepted['step']
                    event(f'实际累计{accepted_step}步、epoch={accepted["epoch"]}通过八rank原生恢复/保存/退出。', stage)
                    if accepted_step == TARGET:
                        write(BASE/'COMPLETE.json', accepted)
                        write(BASE/'EVALUATION_READY.json', dict(at_utc=now(), status='training_complete_evaluation_pending',
                            checkpoint=accepted['checkpoint'], training_acceptance=ref(BASE/'COMPLETE.json'),
                            fixed_evaluation_step=TARGET, native_epoch=accepted['epoch'], source_manifest=ref(BASE/'source_manifest_v1.json'),
                            protocol=ref(ROOT/'docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md'),
                            generation_has_not_been_run=True))
                        write(BASE/'supervision_state.json', dict(at_utc=now(), phase='two_epochs_complete_evaluation_pending',
                            latest_timed_step=TARGET, accepted_step=TARGET, goal_target_step=TARGET, generation_eval_step=TARGET))
                        event('两轮原生训练已完成，固定50082评测节点就绪；生成评测尚未执行，不自动进入第三轮。', stage)
                        return
                    stage = admit(RUN/f'checkpoint-{accepted_step}', min(accepted_step+5000, TARGET), package_sha, platform_api)
                time.sleep(10)
            raise TimeoutError('Bounded supervision deadline reached; no automatic restart')
        except BaseException as error:
            write(BASE/'SUPERVISION_STOPPED.json', dict(at_utc=now(), error=repr(error), stage=str(stage),
                  active_training_not_restarted_or_killed=True))
            event('停止后续准入，保留现有任务和原件：'+repr(error), stage)
            raise


if __name__ == '__main__':
    main()
