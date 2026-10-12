"""Read-only, persistent monitor for the shared-filesystem research GPU pool.

After activating pllm:
  python scripts/monitor_immune_research_pool.py --pool-dir /absolute/pool --once
  python scripts/monitor_immune_research_pool.py --pool-dir /absolute/pool --start
Stop only this monitor by creating /absolute/pool/monitor/STOP.
The monitor never submits, cancels, executes queued jobs, or deletes artifacts.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import html
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

if __package__:
    from .immune_diagnostic_reporting import is_diagnostic, is_trainer_diagnostic, case_progress, optimizer_progress
else:
    from immune_diagnostic_reporting import is_diagnostic, is_trainer_diagnostic, case_progress, optimizer_progress

ROOT = Path(__file__).resolve().parents[1]


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_json(path, errors=None):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('Expected a JSON object')
        return value
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, ValueError) as exc:
        if errors is not None:
            errors.append(f'{path}: {type(exc).__name__}: {exc}')
        return {}


def cell(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        value = f'{value:.6g}' if math.isfinite(value) else str(value)
    value = html.escape(str(value), quote=False)
    for character in ('\\', '`', '[', ']', '*', '_'):
        value = value.replace(character, '\\' + character)
    return value.replace('|', '&#124;').replace('\r', ' ').replace('\n', '<br>')


def utc(value=None):
    try:
        return datetime.fromtimestamp(time.time() if value is None else float(value), timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    except (ValueError, OverflowError, TypeError, OSError):
        return str(value)


def safe_name(value):
    return isinstance(value, str) and value not in {'', '.', '..'} and Path(value).name == value


def mapping(value):
    return value if isinstance(value, dict) else {}


def items(value):
    return value if isinstance(value, list) else []


def platform_query(pool, interval=90):
    """Query only the explicitly recorded task; never infer commands from files."""
    task_id = read_json(pool / 'task_info.json').get('task_id')
    if not isinstance(task_id, str) or not re.fullmatch(r't-\d{14}-[a-z0-9]+', task_id):
        return
    path = pool / 'platform_status.json'
    previous = read_json(path)
    now = time.time()
    last_query = previous.get('queried_at', 0)
    if previous.get('task_id') == task_id and isinstance(last_query, (int, float)) and now-last_query < interval:
        return
    record = dict(previous if previous.get('task_id') == task_id else {}, task_id=task_id, queried_at=now)
    try:
        completed = subprocess.run(['bash', str(ROOT / 'scripts/volc-no-proxy.sh'), 'ml_task', 'get', '-i', task_id, '-o', 'json'],
                                   capture_output=True, text=True, timeout=20, check=True)
        decoder = json.JSONDecoder()
        found = None
        for index, character in enumerate(completed.stdout):
            if character not in '[{':
                continue
            try:
                value, _ = decoder.raw_decode(completed.stdout[index:])
            except ValueError:
                continue
            values = value if isinstance(value, list) else [value]
            found = next((item for item in values if isinstance(item, dict) and item.get('JobId') == task_id), None)
            if found:
                break
        if not found:
            raise ValueError('No matching task JSON in CLI output')
        record.update(observed_at=time.time(), status=found.get('Status'), task=found, query_error=None)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        record['query_error'] = str(exc)[:1000]
    write_json(path, record)
    if record.get('status') != previous.get('status'):
        with (pool / 'monitor' / 'platform_events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def latest_trainer_state(directory, errors):
    candidates = [directory / 'trainer_state.json']
    candidates.extend(directory.glob('checkpoint-*/trainer_state.json'))
    states = [read_json(path, errors) for path in candidates]
    return max(states, key=lambda value: value.get('global_step', 0) if isinstance(value.get('global_step', 0), (int, float)) else 0, default={})


def snapshot(pool, previous=None):
    """Read artifacts only; preserve previous evidence when a file is removed/broken."""
    errors = []
    previous = previous or {}
    records = {key: dict(value) for key, value in mapping(previous.get('records')).items() if isinstance(value, dict)}
    workers = [read_json(path, errors) | {'worker_file': path.stem} for path in sorted((pool / 'workers').glob('*.json'))]
    for category in ('pending', 'running', 'results'):
        for path in sorted((pool / category).glob('*.json')):
            value = read_json(path, errors)
            if not value:
                continue
            record = dict(records.get(path.stem, {}))
            if category == 'results':
                record.update(value)
            else:
                record.update(job_id=path.stem, job=value, state=category)
                for worker in workers:
                    if worker.get('job_id') == path.stem:
                        record.update(worker)
            records[path.stem] = record
    for job_id, record in records.items():
        if not safe_name(job_id):
            errors.append(f'Ignored unsafe run identifier: {job_id!r}')
            continue
        directory = pool / 'runs' / job_id
        for name in ('summary', 'COMPLETE', 'train_results', 'all_results', 'initial_eval_results'):
            value = read_json(directory / (name + '.json'), errors)
            if value:
                record[name] = value
        trainer = latest_trainer_state(directory, errors)
        if trainer:
            record['trainer_state'] = trainer
        job = mapping(record.get('job'))
        variant = job.get('variant')
        if is_diagnostic(job, mapping(record.get('summary')), mapping(record.get('COMPLETE'))):
            record.pop('profile', None)
        if safe_name(variant) and not is_diagnostic(job, mapping(record.get('summary')), mapping(record.get('COMPLETE'))):
            profile = read_json(pool / 'profiles' / variant / 'summary.json', errors)
            if profile:
                record['profile'] = profile
            elif not record.get('profile') or record['profile'].get('partial'):
                measurements = [read_json(path, errors) for path in (pool / 'profiles' / variant).glob('bs*/measurement.json')]
                measurements = [item for item in measurements if item]
                if measurements:
                    record['profile'] = dict(partial=True, measurements=sorted(measurements, key=lambda item: item.get('batch_size', 0)), gpu=measurements[-1].get('gpu'))
        log_path = pool / 'logs' / (job_id + '.log')
        try:
            with log_path.open('rb') as handle:
                handle.seek(max(0, log_path.stat().st_size - 4000))
                record['log_tail'] = handle.read().decode('utf-8', errors='replace')
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(f'{log_path}: {exc}')
    return dict(time=time.time(), pool=str(pool), records=records, workers=workers,
                telemetry=read_json(pool / 'gpu_status.json', errors),
                task=read_json(pool / 'task_info.json', errors), errors=errors,
                platform=read_json(pool / 'platform_status.json', errors),
                local_validation=read_json(pool / 'monitor' / 'local_validation.json', errors),
                counts={name: len(list((pool / name).glob('*.json'))) for name in ('pending', 'running', 'results')})


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + [
        '| ' + ' | '.join(cell(value) for value in row) + ' |' for row in rows]


def progress(record):
    summary = mapping(record.get('summary'))
    if is_diagnostic(mapping(record.get('job')), summary, mapping(record.get('COMPLETE'))):
        return None
    trace = items(summary.get('trace'))
    steps = [point.get('step', 0) for point in trace if isinstance(point, dict)]
    steps += [mapping(record.get('trainer_state')).get('global_step', 0), mapping(record.get('COMPLETE')).get('global_step', 0)]
    steps = [value for value in steps if isinstance(value, (int, float))]
    return max(steps, default=0)


def metric_rows(value, prefix=''):
    for key, item in value.items():
        label = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            yield from metric_rows(item, label)
        elif isinstance(item, (str, float, int, bool)) or item is None:
            yield [label, item]


def render(state, interval):
    lines = ['# 免疫受体研究 8 GPU 工作池实时记录', '',
             f'最后观测：{utc(state["time"])}；本地监听每 {interval:g} 秒读取共享目录。', '',
             '本文件自动更新。监听只读取状态与实验产物，不提交、取消、重提或执行任务；本地后台进程不等于对话窗口保持在线。', '',
             f'工作目录：{cell(state["pool"])}', '',
             f'平台信息：{cell(json.dumps(state["task"], ensure_ascii=False))}', '',
             '合成测试和显存扫描仅验证工程可用性；真实训练与泛化结论需清理证书及独立验证。', '',
             '## 队列与 worker', '',
             f'待运行 {state["counts"]["pending"]}；运行中 {state["counts"]["running"]}；结果文件 {state["counts"]["results"]}。', '']
    platform = state.get('platform', {})
    if platform:
        lines += [f'平台当前状态：**{cell(platform.get("status", "未知"))}**；有效观测：{utc(platform["observed_at"]) if platform.get("observed_at") else "尚无"}；任务：{cell(platform.get("task_id"))}。', '',
                  '平台状态每 90 秒只读查询一次，20 秒超时；终态不会自动取消、重提或继续申请资源。', '']
        if platform.get('query_error'):
            lines += ['最近查询错误（上次有效状态保留）：' + cell(platform['query_error']), '']
    lines += table(['Worker', 'GPU', '状态', '维护负载', '任务', '心跳 UTC'], [
        [worker.get('worker', worker['worker_file']), worker.get('device'), worker.get('state'), worker.get('maintenance_only', False), worker.get('job_id'),
         utc(worker.get('heartbeat', worker.get('time', worker.get('finished', worker.get('started')))))
         if any(worker.get(key) is not None for key in ('heartbeat', 'time', 'finished', 'started')) else None] for worker in state['workers']])
    lines += ['', 'standby_load / maintenance_only 表示待机维护计算，不是实验进度或模型效果；此时 GPU 利用率不得计为实验利用率。']
    telemetry = state['telemetry']
    lines += ['', '## GPU 遥测', '']
    if telemetry:
        sampled = telemetry.get('time')
        fresh = isinstance(sampled, (int, float)) and state['time'] - sampled < max(90, interval * 3)
        lines += [f'采样时间：{utc(sampled)}；主机：{cell(telemetry.get("hostname"))}；状态：{"有效" if fresh else "过期或无时间戳"}。', '']
        gpu_rows = []
        for gpu in items(telemetry.get('gpus')):
            if not isinstance(gpu, dict):
                continue
            utilization = gpu.get('utilization_gpu', gpu.get('utilization.gpu'))
            low = fresh and isinstance(utilization, (int, float)) and utilization < 20
            matched_workers = [worker for worker in state['workers'] if str(worker.get('device')) in {str(gpu.get('index')), str(gpu.get('uuid'))}]
            activity = ', '.join(str(worker.get('state', '?')) for worker in matched_workers) or '未知'
            gpu_rows.append([gpu.get('index'), gpu.get('uuid'), activity, utilization, gpu.get('memory_used_mib', gpu.get('memory.used')),
                             gpu.get('memory_total_mib', gpu.get('memory.total')), '低于 20%（单次采样）' if low else ''])
        lines += table(['GPU', 'UUID', 'worker 状态', '总利用率 %', '已用 MiB', '总量 MiB', '观测'], gpu_rows)
        if telemetry.get('error'):
            lines += ['', '遥测错误：' + cell(telemetry['error'])]
    else:
        lines += ['尚无容器 GPU 遥测；不能据此判断资源已分配或 GPU 空闲。']
    lines += ['', '单次低利用率可能来自加载、验证或实验切换；这里只提示，不自动取消任务。', '', '## 全部实验结果', '']
    rows = []
    for job_id, record in sorted(state['records'].items()):
        job = mapping(record.get('job'))
        summary = mapping(record.get('summary'))
        initial = mapping(summary.get('initial'))
        trace = [point for point in items(summary.get('trace')) if isinstance(point, dict)]
        latest = mapping(summary.get('final')) or (trace[-1] if trace else {})
        initial = mapping(initial.get('mask_mean')) or initial
        latest = mapping(latest.get('mask_mean')) or latest
        diagnostic = is_diagnostic(job, summary, mapping(record.get('COMPLETE')))
        trainer_diagnostic = is_trainer_diagnostic(job, summary, mapping(record.get('COMPLETE')))
        position = optimizer_progress(summary) if trainer_diagnostic else (case_progress(summary) if diagnostic else f'{progress(record)}/{job.get("steps", "?")}')
        default_kind = 'trainer_gpu_runtime_functional_diagnostic' if trainer_diagnostic else ('cuda_attention_backend_probe' if diagnostic else 'unknown')
        rows.append([job_id, job.get('mode'), summary.get('kind', default_kind), record.get('state'), position,
                     initial.get('loss'), latest.get('loss'),
                     mapping(record.get('profile')).get('recommended_micro_batch'), summary.get('max_allocated_gib'), record.get('error')])
    lines += ['mode=train 不等同于真实语料训练；以 kind 标签和数据证据区分。D01显示实际人工optimizer windows，corpus_training_started=false。下表优先使用四档 mask mean loss，未提供四档指标的旧任务保留原 loss。', '']
    lines += table(['任务', '模式', 'kind', '状态', 'step', '初始 loss', '最新 loss', 'micro-bs', '峰值 GiB', '错误'], rows)
    for job_id, record in sorted(state['records'].items()):
        if not any(key in record for key in ('summary', 'profile', 'trainer_state', 'COMPLETE', 'error')):
            continue
        lines += ['', '### ' + cell(job_id), '', '模式/配置：' + cell(json.dumps(record.get('job', {}), ensure_ascii=False)), '']
        if record.get('argv'):
            lines += ['实际 argv：' + cell(json.dumps(record['argv'], ensure_ascii=False)), '']
        summary = mapping(record.get('summary'))
        metrics = {key: summary[key] for key in ('kind', 'dataset_kind', 'status', 'biological_quality_claim', 'micro_batch', 'grad_accum', 'global_batch', 'elapsed_seconds', 'max_allocated_gib', 'initial', 'final', 'train_probe', 'training_started', 'corpus_training_started', 'execution_completed', 'functional_gpu_passed', 'optimizer_window_count', 'completed_optimizer_window_count', 'all_cases_passed', 'case_count', 'completed_case_count', 'case_statuses') if key in summary}
        for key in ('COMPLETE', 'train_results', 'all_results', 'initial_eval_results'):
            if key in record:
                metrics[key] = record[key]
        if metrics:
            lines += table(['指标', '值'], metric_rows(metrics)) + ['']
        if is_trainer_diagnostic(mapping(record.get('job')), summary, mapping(record.get('COMPLETE'))):
            windows = [window for window in items(summary.get('optimizer_windows')) if isinstance(window, dict)]
            if windows:
                lines += ['实际人工Trainer优化窗口：', '']
                lines += table(['optimizer step', 'microsteps', 'records', 'finite gradients', 'mean micro loss'],
                    [[window.get(key) for key in ('step', 'microsteps', 'records', 'finite_gradients', 'mean_micro_loss')]
                     for window in windows]) + ['']
        profile = mapping(record.get('profile'))
        if profile:
            lines += ['显存扫描：' + cell(profile.get('gpu')) + '；推荐 micro-bs：' + cell(profile.get('recommended_micro_batch')), '']
            measurements = [item for item in items(profile.get('measurements')) if isinstance(item, dict)]
            lines += table(['micro-bs', '状态', '峰值 GiB'], [[item.get('batch_size'), item.get('status'), item.get('max_allocated_gib')] for item in measurements]) + ['']
        events = [event for event in items(summary.get('trace')) + items(mapping(record.get('trainer_state')).get('log_history')) if isinstance(event, dict)]
        if events:
            lines += ['逐次全局记录（保持原始 step 与指标名；分组初始/最新/训练探针见监督文档，各次完整 strata 保留在 summary.json）：', '']
            lines += table(['记录', '指标', '值'], [[index, key, value] for index, event in enumerate(events) for key, value in metric_rows({name: item for name, item in event.items() if name != 'strata'})]) + ['']
        if record.get('state') == 'failed' and record.get('log_tail'):
            lines += ['失败日志尾部：', '', '<pre>' + html.escape(record['log_tail']) + '</pre>', '']
    if state['errors']:
        lines += ['', '## 读取错误（保留上次有效结果）', ''] + ['- ' + cell(error) for error in state['errors']]
    lines += ['', '完整观测快照：' + cell(str(Path(state['pool']) / 'monitor' / 'state.json')), '']
    if state.get('local_validation'):
        lines += ['## 本地监听验证', ''] + table(['项目', '结果'], metric_rows(state['local_validation'])) + ['']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool-dir', required=True, type=Path)
    parser.add_argument('--document', type=Path, default=ROOT / 'docs/IMMUNE_RESEARCH_POOL_LIVE.md')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    pool = args.pool_dir.resolve()
    directory = pool / 'monitor'
    if not (pool / 'pool.json').is_file() or args.interval < .05:
        parser.error('An initialized pool and a positive polling interval are required')
    directory.mkdir(exist_ok=True)
    if args.start:
        if args.once or (directory / 'STOP').exists():
            parser.error('--start requires no --once and no monitor/STOP sentinel')
        argv = [sys.executable, str(Path(__file__).resolve()), '--pool-dir', str(pool), '--document', str(args.document.resolve()), '--interval', str(args.interval)]
        with (directory / 'monitor.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        time.sleep(.4)
        if process.poll() is not None:
            raise SystemExit('Monitor did not remain active; inspect monitor/monitor.log (a shared lock may already be held)')
        print(json.dumps(dict(pid=process.pid, hostname=socket.gethostname(), monitor_dir=str(directory)), ensure_ascii=False))
        return
    stop = threading.Event()
    if __package__:
        from .immune_research_supervision import update as update_supervision
    else:
        from immune_research_supervision import update as update_supervision
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    with (directory / 'monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another monitor holds this pool shared-filesystem lock')
        identity = dict(pid=os.getpid(), hostname=socket.gethostname(), started=time.time(), document=str(args.document.resolve()))
        write_json(directory / 'pid.json', identity)
        atomic_write(directory / 'monitor.pid', str(os.getpid()) + '\n')
        previous = read_json(directory / 'state.json')
        while not stop.is_set() and not (directory / 'STOP').exists():
            try:
                platform_query(pool)
                state = snapshot(pool, previous)
                atomic_write(args.document, render(state, args.interval))
                write_json(directory / 'state.json', state)
                previous = state
                update_supervision(pool, args.document.with_name('IMMUNE_RESEARCH_SUPERVISION.md'))
                write_json(directory / 'heartbeat.json', dict(identity, time=time.time(), state='watching', counts=state['counts']))
            except Exception as exc:
                print(f'{utc()} monitor error: {exc!r}', flush=True)
                write_json(directory / 'heartbeat.json', dict(identity, time=time.time(), state='read_error', error=repr(exc)))
            if args.once:
                break
            stop.wait(args.interval)
        write_json(directory / 'heartbeat.json', dict(identity, time=time.time(), state='stopped'))


if __name__ == '__main__':
    main()
