"""Observe the authorized formal run and continue only successful bounded segments.

Activate conda pllm, then run this file with --inspect-only for a read-only check.
The frozen copy is launched once without flags for supervision. Creating
/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/STOP
stops this supervisor and future segment admission, not the active training job.
No platform submit/cancel/restart, failed-job retry, or checkpoint deletion exists.
"""
import argparse
import ast
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

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/v7_formal_eight_gpu_20261002'
POOL = ROOT/'output/immune_research_pool_c20250601_20261005'
RUN = ROOT/'output/v7_initial_training_20261005/run'
OUTPUT = ROOT/'output/v7_formal_supervision_20261005'
LAUNCHES = OWN/'launch_20261005'
LAUNCHER = OWN/'supervisor_launcher_v1/launch_v7_formal_segment_20261005.py'
LAUNCHER_SHA = '8e9f2380c478d8e61b226e27c4df7948f21e909178aa92f2cc31981f2c006cec'
TRANSPORT = OWN/'transport_preload_v1/formal_pool_transport.py'
TRANSPORT_SHA = '7fe96ae577ac9ae5b421caf05756d7d58d7daadfb13af35f0f6507d1012cdaec'
MANIFEST = OWN/'source_manifest_v3.json'
MANIFEST_SHA = '992bc3c5718b71c379c7a2ee4f39d8e5866f974266500d2c2ec69d5ab4db700b'
OBSERVER = OWN/'source_snapshot_v3/scripts/research/formal_resume_observation.py'
DRIVER = OWN/'source_snapshot_v3/scripts/research/run_v7_formal.py'
TASK = 't-20261005205851-lqfjg'
TARGETS = (5016, 10016, 15016, 20016, 25016, 25042)
DOCUMENT = ROOT/'docs/V7_FORMAL_SUPERVISION_20261005.md'
BEGIN = '<!-- V7_FORMAL_AUTOMATIC_STATUS_START -->'
END = '<!-- V7_FORMAL_AUTOMATIC_STATUS_END -->'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    if not path.exists():
        raise FileNotFoundError(str(path))
    require(path.is_file() and not path.is_symlink(), 'Regular file required: '+str(path))
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path):
    return dict(path=str(path), sha256=sha(path))


def atomic(path, value):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def write(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def fixed_sources():
    for path, expected in ((LAUNCHER, LAUNCHER_SHA), (TRANSPORT, TRANSPORT_SHA), (MANIFEST, MANIFEST_SHA)):
        require(path.is_file() and not path.is_symlink(), 'Fixed source unavailable: '+str(path))
        require(sha(path) == expected, 'Supervisor dependency drift: '+str(path))
    for path in (OBSERVER, DRIVER):
        require(path.is_file() and not path.is_symlink(), 'Fixed verifier unavailable: '+str(path))
        require(sha(path) == read(MANIFEST)['files'][str(path)], 'Native metadata verifier drift')


def platform():
    result = subprocess.run(['bash', str(ROOT/'scripts/volc-no-proxy.sh'), 'ml_task', 'get',
                             '-i', TASK, '-o', 'json'], capture_output=True, text=True, timeout=20, check=True)
    decoder = json.JSONDecoder()
    for index, char in enumerate(result.stdout):
        if char not in '[{':
            continue
        try:
            value, _ = decoder.raw_decode(result.stdout[index:])
        except ValueError:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, dict) and item.get('JobId') == TASK:
                return dict(at_utc=now(), task=item)
    raise ValueError('Platform response did not identify the exact allocated task')


def snapshot(stage):
    receipt = read(stage/'enqueue_receipt.json')
    require(receipt['state'] == 'enqueued' and len(set(receipt['job_ids'])) == 8,
            'Exactly eight actually enqueued tickets required')
    results = {}
    for job_id in receipt['job_ids']:
        path = POOL/'results'/f'{job_id}.json'
        if path.exists():
            results[job_id] = read(path)
    coordinator = stage/'meeting/coordinator/state.json'
    state = read(coordinator) if coordinator.exists() else {}
    log = stage/'meeting/coordinator/rank0_child.log'
    latest = None
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-128*1024))
            for line in stream.read().decode(errors='replace').splitlines():
                if line.startswith("{'loss':"):
                    try:
                        latest = ast.literal_eval(line)
                    except (ValueError, SyntaxError):
                        pass
    workers = [read(path) for path in (POOL/'workers').glob('*.json')]
    ages = [time.time()-worker.get('heartbeat', worker.get('time', 0)) for worker in workers]
    return dict(at_utc=now(), stage=str(stage), target_step=receipt['stop_after'],
                prior_step=receipt['prior_step'], phase=state.get('phase'),
                terminal=state.get('terminal'), abort=state.get('abort'), results=results,
                latest_training_log=latest, workers=workers,
                worker_age_seconds=ages, worker_observation_fresh=len(workers) == 8
                and all(0 <= age < 30 for age in ages))


def preserve_checkpoint_metadata(step):
    fixed_sources()
    checkpoint = RUN/f'checkpoint-{step}'
    sidecar = checkpoint/'formal_checkpoint.json'
    if not sidecar.exists():
        return None
    evidence = read(sidecar)
    provenance = read(RUN/'run_provenance.json')
    binding = runpy.run_path(str(DRIVER), run_name='_original_binding_digest')['digest'](provenance)
    require(evidence['binding_sha256'] == binding, 'Checkpoint differs from the actual formal run')
    require(evidence['step'] == step and evidence['world_size'] == 8
            and evidence['rank_acknowledgments'] == list(range(8)), 'Incomplete checkpoint acknowledgments')
    require(evidence['rank_parameter_digests'] ==
            [dict(rank=rank, **evidence['saved_parameter_digest']) for rank in range(8)],
            'Checkpoint rank parameters differ')
    require(len(evidence['files']) == 17, 'Incomplete native inventory')
    destination = OUTPUT/'checkpoint_evidence'/f'checkpoint-{step}'
    destination.mkdir(parents=True, exist_ok=True)
    for path in checkpoint.glob('*.json'):
        if path.name != 'formal_checkpoint.json':
            require(sha(path) == evidence['files'][path.name], 'Native JSON inventory drift')
        target = destination/path.name
        if target.exists():
            require(sha(target) == sha(path), 'Archived metadata changed')
        else:
            target.write_bytes(path.read_bytes())
    return dict(step=step, sidecar=reference(destination/'formal_checkpoint.json'),
                native_inventory_verified_by='all eight original Trainer checkpoint callbacks',
                supervisor_rehashed_tensor_files=False)


def accept_segment(stage, status):
    require(status['terminal'] == 'succeeded' and status['abort'] is None, 'Group has not succeeded')
    require(len(status['results']) == 8 and all(item['state'] == 'succeeded' and item['returncode'] == 0
            for item in status['results'].values()), 'Eight successful actual pool results required')
    for rank in range(8):
        item = read(stage/f'meeting/coordinator/rank{rank}_parent_receipt.json')
        cleanup = item['cleanup']
        require(item['error'] is None and cleanup['returncode'] == 0 and cleanup['wait_observed']
                and cleanup['process_absent'] and cleanup['process_group_gone']
                and cleanup['cleanup_completed'], 'Owned rank cleanup incomplete')
    step, prior = status['target_step'], status['prior_step']
    marker = RUN/('COMPLETE.json' if step == 25042 else f'SEGMENT_{step}.json')
    segment = read(marker)
    require(segment['step'] == step and segment['world_size'] == 8, 'Actual completed step differs')
    resume_path = RUN/f'RESUME_{prior}_TO_{step}.json'
    resume = read(resume_path)
    verifier = runpy.run_path(str(OBSERVER), run_name='_native_resume_receipt_validation')
    require(verifier['complete_receipt'](resume['ranks']) == resume, 'Native restored-state evidence differs')
    require(resume['prior_step'] == prior and resume['final_step'] == step
            and resume['binding_sha256'] == segment['binding_sha256'], 'Resume segment binding differs')
    checkpoint = preserve_checkpoint_metadata(step)
    require(checkpoint is not None, 'No complete final checkpoint')
    sidecar = read(Path(checkpoint['sidecar']['path']))
    require(sidecar['binding_sha256'] == resume['binding_sha256'], 'Checkpoint binding differs')
    prior_sidecar = OUTPUT/'checkpoint_evidence'/f'checkpoint-{prior}/formal_checkpoint.json'
    require(sha(prior_sidecar) == resume['sidecar_sha256'], 'Prior checkpoint evidence differs')
    require(read(prior_sidecar)['saved_parameter_digest'] != sidecar['saved_parameter_digest'],
            'No changed model parameters after resumed updates')
    evidence = dict(at_utc=now(), status='actual_segment_passed', step=step, prior_step=prior,
                    checkpoint=checkpoint, resume=reference(resume_path), marker=reference(marker),
                    pool_results={job:reference(POOL/'results'/f'{job}.json') for job in status['results']},
                    trajectory_equivalence_claim=False)
    write(OUTPUT/f'segment_{step}_acceptance.json', evidence)
    for path in (resume_path, marker):
        target = OUTPUT/'checkpoint_evidence'/path.name
        if not target.exists():
            target.write_bytes(path.read_bytes())
    # Trainer overwrites these at the end of each invocation; retain each segment.
    for name in ('train_results.json', 'all_results.json'):
        path = RUN/name
        target = OUTPUT/'checkpoint_evidence'/f'segment_{step}_{name}'
        if target.exists():
            require(sha(target) == sha(path), 'Archived segment metrics differ')
        else:
            target.write_bytes(path.read_bytes())
    return evidence


def report(status, event):
    write(OUTPUT/'state.json', status)
    text = (f'# V7 正式训练持续监督\n\n更新时间：{status["at_utc"]}。\n\n'
            f'{event}\n\n当前训练段：{status.get("stage")}；累计目标 {status.get("target_step")}；'
            f'phase={status.get("phase")}，terminal={status.get("terminal")}。'
            '目标步数不是已完成步数。\n\n'
            f'最新原始训练日志：`{json.dumps(status.get("latest_training_log"), ensure_ascii=False)}`。\n\n'
            f'完整状态和实际结果：{OUTPUT}/state.json；关键事件：{OUTPUT}/events.jsonl。\n\n'
            '每段全部八rank成功且原生恢复证据通过才接续；失败或STOP停止接段。'
            '不重启/重放作业，不改变训练源、审核、batch、目标或清理正式恢复文件。\n')
    atomic(DOCUMENT, text)


def milestone(status, event):
    report(status, event)
    item = dict(at_utc=now(), event=event, stage=status.get('stage'), target=status.get('target_step'))
    with (OUTPUT/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(item, ensure_ascii=False)+'\n')
    project = ROOT/'PROJECT_PROCESS.md'
    current = project.read_text()
    block = f'{BEGIN}\n\n最新自动监督：{item["at_utc"]} — {event}\n\n实际状态：{DOCUMENT}。\n\n{END}'
    if BEGIN in current:
        require(current.count(BEGIN) == current.count(END) == 1, 'Ambiguous status section')
        before, remaining = current.split(BEGIN, 1)
        _, after = remaining.split(END, 1)
        current = before+block+after
    else:
        head, remaining = current.split('\n', 1)
        current = head+'\n\n'+block+'\n'+remaining
    current = re.sub(r'^Last updated:.*$', f'Last updated: {item["at_utc"]} — {event}',
                     current, count=1, flags=re.M)
    atomic(project, current)
    with project.open('a') as stream:
        stream.write(f'\n## {item["at_utc"]} 正式训练监督节点\n\n{event}\n\n原件：{OUTPUT}/events.jsonl、{DOCUMENT}。\n')


def run():
    fixed_sources()
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (OUTPUT/'launch.json').exists(), 'Supervisor already launched; never silently restart')
        write(OUTPUT/'launch.json', dict(at_utc=now(), pid=os.getpid(), source=reference(Path(__file__).resolve())))
        stage = LAUNCHES/'segment_5016'
        seen = set()
        accepted = set()
        status = snapshot(stage)
        require(preserve_checkpoint_metadata(16) is not None,
                'Prior checkpoint-16 evidence must be archived before observing later rotation')
        deadline = time.monotonic()+6*24*3600
        last_platform, last_query = None, 0.
        milestone(status, '持续监督已启动；当前5016段已在运行，未新增训练或平台任务。')
        try:
            while time.monotonic() < deadline:
                if (OUTPUT/'STOP').exists():
                    milestone(status, '收到STOP，监督器停止接续；既有训练任务未被终止。')
                    return
                try:
                    status = snapshot(stage)
                except (OSError, json.JSONDecodeError) as exc:
                    status.update(at_utc=now(), observation_error=repr(exc))
                    report(status, '暂时无法读取观察文件；保持原任务并等待同一任务新证据，不重启或接段。')
                    time.sleep(30)
                    continue
                if time.monotonic()-last_query >= 300:
                    try:
                        last_platform = platform()
                        write(OUTPUT/'platform_latest.json', last_platform)
                        status['platform_observation_error'] = None
                    except (OSError, subprocess.SubprocessError, ValueError) as exc:
                        status['platform_observation_error'] = str(exc)
                    last_query = time.monotonic()
                status['platform'] = last_platform
                if status['abort'] or status['terminal'] == 'aborted' or any(
                        item['state'] != 'succeeded' for item in status['results'].values()):
                    raise RuntimeError('Actual formal group failed; stopping future admission, preserving evidence')
                if last_platform and last_platform['task']['Status'] in ('Failed', 'Success', 'Canceled', 'Cancelled'):
                    raise RuntimeError('Allocated platform job is terminal; never restart or duplicate-submit')
                for path in sorted(RUN.glob('checkpoint-*/formal_checkpoint.json')):
                    step = int(path.parent.name.split('-')[1])
                    if step not in seen:
                        try:
                            evidence = preserve_checkpoint_metadata(step)
                        except (OSError, json.JSONDecodeError) as exc:
                            status['checkpoint_observation_error'] = repr(exc)
                            continue
                        if evidence is not None:
                            seen.add(step)
                            if step > 16:
                                milestone(status, f'原Trainer实际checkpoint-{step}完成，8rank参数一致且原生状态清单齐全；JSON证据已归档。')
                if status['terminal'] == 'succeeded' and len(status['results']) == 8:
                    fixed_sources()
                    step = status['target_step']
                    if step not in accepted:
                        acceptance = accept_segment(stage, status)
                        accepted.add(step)
                        milestone(status, f'正式累计{step}步通过：8rank成功退出、原生恢复与新checkpoint完整。')
                    if step == 25042:
                        write(OUTPUT/'COMPLETE.json', acceptance)
                        return
                    next_step = TARGETS[TARGETS.index(step)+1]
                    workers = [read(path) for path in (POOL/'workers').glob('*.json')]
                    if (list((POOL/'running').glob('*.json')) or list((POOL/'pending').glob('*.json'))
                            or len(workers) != 8 or any(worker['state'] != 'standby_load'
                            or not 0 <= time.time()-worker['time'] < 15 for worker in workers)):
                        report(status, '前段已通过，等待原8worker实际释放并刷新心跳；尚未接段。')
                        time.sleep(30)
                        continue
                    # No admission on stale/unknown platform observations, duplicate stages or STOP.
                    try:
                        observed = platform()
                    except (OSError, subprocess.SubprocessError, ValueError) as exc:
                        status['platform_observation_error'] = repr(exc)
                        report(status, '前段已通过，暂时无法确认原平台任务；等待同一任务，不接段。')
                        time.sleep(30)
                        continue
                    require(observed['task']['Status'] == 'Running', 'Live exact platform resource required')
                    require(not (OUTPUT/'STOP').exists(), 'STOP before next admission')
                    next_stage = LAUNCHES/f'segment_{next_step}'
                    require(not next_stage.exists(), 'Next segment already exists; inspect instead of replay')
                    command = [sys.executable, '-B', str(LAUNCHER), '--stop-after', str(next_step),
                               '--resume-step', str(step), '--previous-stage', str(stage),
                               '--transport', str(TRANSPORT), '--enqueue']
                    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
                    write(OUTPUT/f'enqueue_{next_step}.json', dict(at_utc=now(), command=command,
                          returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
                    require(result.returncode == 0, 'Launcher did not succeed; inspect artifacts, never automatic retry')
                    stage = next_stage
                    status = snapshot(stage)
                    milestone(status, f'成功段之后已接续新段：累计目标{next_step}；未重新申请或重启8卡资源。')
                else:
                    report(status, '正在观察当前段；等待实际checkpoint或八rank终态，未提交重复任务。')
                time.sleep(30)
            milestone(status, '监督器已达6天时间界限，停止后续接段；既有训练未被终止。')
        except BaseException as exc:
            status.update(at_utc=now(), supervisor_error=repr(exc))
            milestone(status, '监督器停止接段：'+repr(exc)+'；原训练、恢复状态及失败证据保留。')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    if args.inspect_only:
        fixed_sources()
        status = snapshot(LAUNCHES/'segment_5016')
        status['platform'] = platform()
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        run()


if __name__ == '__main__':
    main()
