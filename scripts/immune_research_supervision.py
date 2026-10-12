"""Read-only interpretation of GPU-pool experiment evidence, independent of training.

After activating pllm:
  python scripts/immune_research_supervision.py --pool-dir /absolute/pool
The persistent pool monitor also calls this writer every observation cycle.
Only monitoring JSON/Markdown is written; jobs, code, and checkpoints are untouched.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import time
import uuid

if __package__:
    from .immune_diagnostic_reporting import is_diagnostic, is_trainer_diagnostic, inspect_diagnostic, case_progress
else:
    from immune_diagnostic_reporting import is_diagnostic, is_trainer_diagnostic, inspect_diagnostic, case_progress

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / 'docs/IMMUNE_RESEARCH_SUPERVISION.md'


def read(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def cell(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.6g}'
    return html.escape(str(value)).replace('|', '&#124;').replace('\n', '<br>').replace('_', '\\_').replace('`', '\\`')


def digest(path, cache):
    if path not in cache:
        try:
            cache[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            cache[path] = None
    return cache[path]


def finite_values(value):
    if isinstance(value, dict):
        return all(finite_values(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_values(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def full_mask_metrics(metrics):
    levels = metrics.get('mask_levels', {})
    if isinstance(levels, dict) and isinstance(levels.get('mask100'), dict):
        return levels['mask100']
    return metrics


def grouped_metrics(metrics):
    """Pool per-level token numerators before averaging mask levels by subset."""
    strata = metrics.get('strata', {}) if isinstance(metrics, dict) else {}
    if not isinstance(strata, dict):
        return {}
    members = {'conditional': {'ab_antigen', 'tcr_peptide', 'tcr_pmhc'},
               'unconditional': {'unconditional_ab', 'unconditional_tcr', 'unconditional_single'}}
    counts = {'loss': 'selected_count', 'aa_loss': 'aa_count', 'aa_accuracy': 'aa_count',
              'terminal_loss': 'terminal_count', 'terminal_accuracy': 'terminal_count'}
    result = {}
    for name, groups in members.items():
        portions = [value for key, value in strata.items()
                    if isinstance(key, str) and key.split('/', 1)[0] in groups and isinstance(value, dict)]
        if not portions:
            continue
        merged = {}
        for mask in ('mask25', 'mask50', 'mask75', 'mask100'):
            values = [value.get('mask_levels', {}).get(mask, {}) for value in portions
                      if isinstance(value.get('mask_levels'), dict)]
            values = [value for value in values if isinstance(value, dict) and value]
            if not values:
                continue
            totals = {key: sum(value.get(key, 0) for value in values) for key in set(counts.values())}
            metrics_for_level = dict(totals)
            for metric, count in counts.items():
                valid = [value for value in values if value.get(count, 0)]
                available = all(isinstance(value.get(metric), (float, int)) for value in valid)
                metrics_for_level[metric] = (sum(value[metric]*value[count] for value in valid)/totals[count]
                                             if totals[count] and available else None)
            merged[mask] = metrics_for_level
        means = {}
        for metric in counts:
            defined = [value[metric] for value in merged.values() if value[metric] is not None]
            means[metric] = sum(defined)/len(defined) if defined else None
        result[name] = dict(mask_levels=merged, mask_mean=means, stratum_count=len(portions))
    return result


def added_fingerprinted_files(source_root, recorded):
    """Detect additions to the runner's globbed source sets, not just edits."""
    if 'scripts/run_immune_research.py' not in recorded:
        return []
    patterns = ['dllm/pipelines/immune_llada/v3/*.py',
                'dllm/pipelines/immune_llada/v3/*.json',
                'dllm/pipelines/immune_llada/research/*.py',
                'dllm/pipelines/immune_llada/data/**/*.py']
    if 'scripts/run_immune_architecture.py' in recorded:
        patterns.append('dllm/pipelines/immune_llada/research_next/**/*.py')
    if 'scripts/run_immune_refined.py' in recorded:
        patterns.append('dllm/pipelines/immune_llada/research_refined/**/*.py')
    observed = {str(path.relative_to(source_root)) for pattern in patterns for path in source_root.glob(pattern)}
    return sorted(observed-set(recorded))


def inspect(pool):
    jobs = {}
    hashes = {}
    releases = read(pool / 'release_map.json').get('variants', {})
    if not isinstance(releases, dict):
        releases = {}
    for directory in ('pending', 'running', 'finished'):
        for path in (pool / directory).glob('*.json'):
            job = read(path)
            if isinstance(job.get('variant'), str) and isinstance(job.get('mode'), str):
                jobs[path.stem] = dict(job=job, state=directory)
    for job_id, item in jobs.items():
        result = read(pool / 'results' / (job_id+'.json'))
        item['state'] = result.get('state', item['state'])
        variant = item['job'].get('variant', '')
        if not isinstance(variant, str) or Path(variant).name != variant:
            item['alerts'] = ['invalid_variant']
            continue
        run = pool / 'runs' / job_id
        profile_dir = pool / 'profiles' / variant
        summary = read(run / 'summary.json')
        completion = read(run / 'COMPLETE.json')
        if is_diagnostic(item['job'], summary, completion):
            release = releases.get(variant, {})
            source_root = Path(release.get('source_root', ROOT)).resolve() if isinstance(release, dict) else ROOT
            if is_trainer_diagnostic(item['job'], summary, completion):
                source_root = ROOT
            item.update(inspect_diagnostic(item['job'], item['state'], run, summary, completion, source_root))
            continue
        provenance = read(run / 'run_provenance.json')
        trainer_states = [read(path) for path in [run / 'trainer_state.json', *run.glob('checkpoint-*/trainer_state.json')]]
        trainer = max(trainer_states, key=lambda item: item.get('global_step', 0), default={})
        profile = read(profile_dir / 'summary.json')
        certificate = read(profile_dir / 'COMPLETE.json')
        trace = summary.get('trace', [])
        trace = [point for point in trace if isinstance(point, dict)] if isinstance(trace, list) else []
        latest = summary.get('final') or (trace[-1] if trace else summary.get('initial', {}))
        if not latest and trainer.get('log_history'):
            latest = trainer['log_history'][-1]
        latest = latest if isinstance(latest, dict) else {}
        release = releases.get(variant, {})
        source_root = Path(release.get('source_root', ROOT)).resolve() if isinstance(release, dict) else ROOT
        item.update(variant=variant, step=max([point.get('step', 0) for point in trace] + [completion.get('global_step', 0), trainer.get('global_step', 0)]), metrics=latest,
                    requested_steps=item['job'].get('steps'), mode=item['job'].get('mode'), source_root=str(source_root),
                    kind=summary.get('kind', provenance.get('kind', completion.get('kind', 'unknown'))),
                    dataset_kind=summary.get('dataset_kind'), biological_quality_claim=summary.get('biological_quality_claim'),
                    fixture_manifest=summary.get('fixture_manifest', {}), train_probe=summary.get('train_probe', {}), trace=trace,
                    initial_group_metrics=grouped_metrics(summary.get('initial', {})), latest_group_metrics=grouped_metrics(latest),
                    group_trace=[dict(step=point.get('step'), groups=grouped_metrics(point)) for point in trace if point.get('strata')],
                    initial_metrics=summary.get('initial', {}), final_metrics=summary.get('final', {}),
                    profile_status=certificate.get('status'), micro_batch=profile.get('recommended_micro_batch'),
                    profile_oom_batches=[point.get('batch_size') for point in profile.get('measurements', []) if point.get('status') == 'oom'],
                    profile_peak_gib=next((point.get('max_allocated_gib') for point in profile.get('measurements', []) if point.get('batch_size') == profile.get('recommended_micro_batch')), None),
                    completion=completion, source_mismatches=[], source_files=0, source_status='pending', alerts=[])
        snapshots = [source.get('source_sha256', {}) for source in (certificate, summary, provenance)]
        for source in snapshots:
            if not isinstance(source, dict):
                item['alerts'].append('invalid_source_certificate')
                continue
            item['source_files'] = max(item['source_files'], len(source))
            item['source_mismatches'].extend('added:'+relative for relative in added_fingerprinted_files(source_root, source))
            for relative, expected in source.items():
                source_path = (source_root / relative).resolve()
                if not source_path.is_relative_to(source_root) or digest(source_path, hashes) != expected:
                    item['source_mismatches'].append(relative)
        if item['source_files']:
            item['source_status'] = 'mismatch' if item['source_mismatches'] else 'matched'
        if item['source_mismatches']:
            item['alerts'].append('source_changed')
        if certificate and digest(profile_dir / 'summary.json', hashes) != certificate.get('summary_sha256'):
            item['alerts'].append('profile_summary_hash_mismatch')
        if completion.get('summary_sha256') and digest(run / 'summary.json', hashes) != completion['summary_sha256']:
            item['alerts'].append('run_summary_hash_mismatch')
        if not finite_values(trace + [latest, item['initial_metrics'], item['train_probe']]):
            item['alerts'].append('nonfinite_metric')
        synthetic = item['mode'] in {'synthetic', 'profile-synthetic'} or str(item['kind']).startswith('synthetic_')
        if str(item['kind']).startswith('synthetic_') and item['biological_quality_claim'] is not False:
            item['alerts'].append('synthetic_claim_inconsistent')
        if item['state'] == 'succeeded' and (completion.get('status') != 'completed' or (synthetic and summary.get('status') != 'completed') or item['step'] != item['requested_steps']):
            item['alerts'].append('inconsistent_completion')
        if item['state'] == 'failed':
            try:
                log = (pool / 'logs' / (job_id+'.log')).read_text(errors='replace')[-20000:]
            except OSError:
                log = ''
            item['error'] = result.get('error')
            if 'Source changed' in log or 'fingerprint' in log.lower():
                item['alerts'].append('source_changed_runner')
            elif 'nonfinite' in log.lower():
                item['alerts'].append('nonfinite_runner')
            elif 'out of memory' in log.lower():
                item['alerts'].append('runner_oom_failure')
            else:
                item['alerts'].append('runner_failure')
        full_metrics = full_mask_metrics(latest)
        aa = full_metrics.get('aa_accuracy')
        terminal = full_metrics.get('terminal_accuracy')
        if isinstance(aa, (int, float)) and isinstance(terminal, (int, float)):
            if aa <= .01 and terminal >= .99 and item['step'] >= 5:
                item['alerts'].append('endpoint_shortcut_signal')
            elif terminal == 0 and item['step'] >= 10:
                item['alerts'].append('end_not_learned_yet')
    initial = {}
    for job_id, item in sorted(jobs.items()):
        prefix = item.get('variant', '').split('_', 1)[0]
        if prefix.startswith('r') and prefix[1:].isdigit() and 0 <= int(prefix[1:]) <= 11 and item.get('requested_steps') == 20 and item.get('mode') == 'profile-synthetic':
            initial.setdefault(item['variant'], item)
    return dict(time=time.time(), pool=str(pool), initial_jobs=len(initial), total_jobs=len(jobs), records=jobs,
                counts={state: sum(item['state'] == state for item in jobs.values()) for state in ('pending', 'running', 'succeeded', 'failed')},
                all_initial_done=len(initial) == 12 and all(item['state'] in {'succeeded', 'failed'} for item in initial.values()),
                all_current_done=bool(jobs) and all(item['state'] in {'succeeded', 'failed'} for item in jobs.values()))


def update(pool, document=DOCUMENT):
    state = inspect(pool)
    directory = pool / 'monitor'
    previous = read(directory / 'supervision.json')
    previous_records = previous.get('records', {})
    changed = [(job_id, record) for job_id, record in state['records'].items() if record != previous_records.get(job_id)]
    events = previous.get('events', [])
    if changed:
        events += [dict(time=state['time'], changes=[dict(job_id=job_id, variant=record.get('variant'), state=record['state'], step=record.get('step'), requested_steps=record.get('requested_steps'), alerts=record.get('alerts', [])) for job_id, record in changed])]
    state['events'] = events
    lines = ['# 架构实验工作池监督记录', '', f'最后观测：{stamp(state["time"])}。', '',
             '范围：8 GPU 工作池全部方法与步数。每个 job_id 单独留档，20 步与 100 步等不同实例不互相覆盖。首轮完成标记仅指 R00–R11 各自最早的 20 步诊断。', '',
             '监督只读取证据，不改变活跃训练源码、不取消或重提平台任务、不自动加入新实验。后台监听持续落盘，当前对话中的监督 agent 负责解释与排错。', '',
             '状态计数：'+cell(json.dumps(state['counts'], ensure_ascii=False))+'。全部首轮结束：'+str(state['all_initial_done'])+'；全部当前实验结束：'+str(state['all_current_done'])+'。', '',
             '指纹校验以 release_map.json 中每个 variant 的 source_root 为准；无映射时使用当前仓库。旧源码快照和旧 profile 应一同保留。', '',
             'mode=train 只是入口模式。kind=synthetic_generalization_diagnostic 仍为人工字符串诊断，不代表真实语料训练；kind 未落盘时不推断数据性质。CUDA operator诊断单列cases；D01 Trainer runtime诊断单列实际3个optimizer windows，training_started指人工训练，corpus_training_started=false。', '',
             '| job_id | 方法 | 模式 | kind | 状态 | 完成/目标 step | 指标层级 | micro-bs | 峰值 GiB | loss | AA acc | end acc | source SHA | 关注项 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for job_id, record in sorted(state['records'].items(), key=lambda pair: (pair[1].get('variant', ''), pair[0])):
        metrics = full_mask_metrics(record.get('metrics', {}))
        level = 'mask100' if record.get('metrics', {}).get('mask_levels') or record.get('mode') != 'train' else 'trainer latest'
        diagnostic = is_diagnostic(record.get('job', {}), record)
        progress = case_progress(record) if diagnostic else f'{record.get("step")}/{record.get("requested_steps")}'
        if diagnostic:
            level = 'artificial Trainer windows' if is_trainer_diagnostic(record.get('job', {}), record) else 'operator cases'
        row = [job_id, record.get('variant'), record.get('mode'), record.get('kind'), record['state'], progress, level, record.get('micro_batch'), record.get('profile_peak_gib'), metrics.get('loss'), metrics.get('aa_accuracy'), metrics.get('terminal_accuracy'), f'{record.get("source_status")} ({record.get("source_files", 0)} files)', ', '.join(record.get('alerts', []))]
        lines.append('| '+' | '.join(cell(value) for value in row)+' |')
    for job_id, record in sorted(state['records'].items()):
        if not is_diagnostic(record.get('job', {}), record):
            continue
        if is_trainer_diagnostic(record.get('job', {}), record):
            lines += ['', '### '+cell(job_id)+'：人工 Trainer runtime诊断', '',
                      '执行完成='+cell(record.get('execution_completed'))+'；功能GPU检查='+cell(record.get('functional_gpu_passed'))
                      +'；人工训练开始='+cell(record.get('training_started'))+'；corpus_training_started='
                      +cell(record.get('corpus_training_started'))+'。transport steps=1仅代表一次调用，进度以实际optimizer windows计。', '',
                      '| optimizer step | microsteps | records | finite gradients | mean micro loss |', '| --- | --- | --- | --- | --- |']
            windows = record.get('optimizer_windows')
            if isinstance(windows, list):
                for window in windows:
                    if isinstance(window, dict):
                        lines.append('| '+' | '.join(cell(window.get(key)) for key in
                            ('step', 'microsteps', 'records', 'finite_gradients', 'mean_micro_loss'))+' |')
            continue
        lines += ['', '### '+cell(job_id)+'：CUDA operator诊断', '',
                  '执行完成='+cell(record.get('execution_completed'))+'；全部数值通过='+cell(record.get('all_cases_passed'))+'。unsupported/OOM/数值不符保留原状态，不能算后端通过。', '',
                  '| case | 状态 |', '| --- | --- |']
        statuses = record.get('case_statuses')
        if isinstance(statuses, dict):
            lines += ['| '+cell(key)+' | '+cell(value)+' |' for key, value in sorted(statuses.items())]
    for job_id, record in sorted(state['records'].items()):
        latest = record.get('metrics', {})
        if not latest.get('mask_levels') and not latest.get('mask_mean'):
            continue
        lines += ['', '### '+cell(job_id)+'：分层 mask 指标', '',
                  '源码目录：'+cell(record.get('source_root'))+'；完成/目标 step：'+cell(f'{record.get("step")}/{record.get("requested_steps")}')+'。', '',
                  'kind='+cell(record.get('kind'))+'；dataset_kind='+cell(record.get('dataset_kind'))+'；biological_quality_claim='+cell(record.get('biological_quality_claim'))+'。', '',
                  '| 时点 | mask level | 指标 | 值 |', '| --- | --- | --- | --- |']
        for phase, metric in [('initial', record.get('initial_metrics', {})), ('latest', latest)]:
            if not isinstance(metric, dict):
                continue
            levels = metric.get('mask_levels', {})
            levels = dict(levels) if isinstance(levels, dict) else {}
            if isinstance(metric.get('mask_mean'), dict):
                levels['mask_mean'] = metric['mask_mean']
            for level, values in levels.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        if isinstance(value, (str, float, int, bool)) or value is None:
                            lines.append('| '+' | '.join(cell(value) for value in (phase, level, key, value))+' |')
        if record.get('trace'):
            lines += ['', '评测曲线（原始 step，四档等权均值）：', '',
                      '| step | mean loss | mean AA acc | mean end acc | gradient norm |', '| --- | --- | --- | --- | --- |']
            for point in record['trace']:
                mean = point.get('mask_mean', {})
                lines.append('| '+' | '.join(cell(value) for value in (point.get('step'), mean.get('loss'), mean.get('aa_accuracy'), mean.get('terminal_accuracy'), point.get('gradient_norm')))+' |')
        if record.get('latest_group_metrics'):
            lines += ['', '有条件/无条件曲线（每档按token计数合并，再平均四档；缺失分组不推断数值）：', '',
                      '| step | 子集 | mean loss | mean AA | mean end | Full-mask AA | Full-mask end | Full-mask AA count | Full-mask end count |',
                      '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
            points = [dict(step='initial', groups=record.get('initial_group_metrics', {})), *record.get('group_trace', [])]
            for point in points:
                for name, values in point['groups'].items():
                    mean = values.get('mask_mean', {})
                    full = values.get('mask_levels', {}).get('mask100', {})
                    row = (point.get('step'), name, mean.get('loss'), mean.get('aa_accuracy'), mean.get('terminal_accuracy'), full.get('aa_accuracy'), full.get('terminal_accuracy'), full.get('aa_count'), full.get('terminal_count'))
                    lines.append('| '+' | '.join(cell(value) for value in row)+' |')
        manifest = record.get('fixture_manifest', {})
        if manifest:
            lines += ['', '人工数据 manifest：'+cell(manifest.get('manifest_sha256'))+'；精确 target sequence 重叠='+cell(manifest.get('exact_target_sequence_overlap'))+'；精确 record 重叠='+cell(manifest.get('exact_record_overlap'))+'。', '',
                      '跨 radius32 覆盖：'+cell(json.dumps(manifest.get('cross_radius32_coverage', {}), ensure_ascii=False))+'。生成规则共享是诊断设计的一部分，不能当作真实下游去重证书。', '']
        if latest.get('strata'):
            lines += ['', '按任务/链长分组（全局 level 按 token 计数汇总，下面保留各组计数；不同组不得直接比较未经加权的总 loss）：', '',
                      '| 时点 | group/length | mask level | loss | AA acc | end acc | selected count | AA count | end count |',
                      '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
            for phase, metric in [('initial', record.get('initial_metrics', {})), ('latest', latest), ('train_probe', record.get('train_probe', {}))]:
                strata = metric.get('strata', {}) if isinstance(metric, dict) else {}
                for group, values in sorted(strata.items()) if isinstance(strata, dict) else []:
                    if not isinstance(values, dict):
                        continue
                    levels = dict(values.get('mask_levels', {}))
                    levels['mask_mean'] = values.get('mask_mean', {})
                    for level, metrics in levels.items():
                        if isinstance(metrics, dict):
                            row = (phase, group, level, metrics.get('loss'), metrics.get('aa_accuracy'), metrics.get('terminal_accuracy'), metrics.get('selected_count'), metrics.get('aa_count'), metrics.get('terminal_count'))
                            lines.append('| '+' | '.join(cell(value) for value in row)+' |')
    lines += ['', '## 判读规则与下一阶段', '',
              '- 显存扫描中的预期 OOM 档位属于正常上界探索；只有 runner 失败或 completion 不一致才视为执行失败。',
              '- source_changed / profile_summary_hash_mismatch / run_summary_hash_mismatch：源码或结果与证书不一致，优先报告；不自动修改训练代码或重试。',
              '- endpoint_shortcut_signal：AA acc ≤ 0.01 且 end acc ≥ 0.99；短程总 loss 下降不足以证明学会残基。该项是诊断信号，不能直接据此判定最终架构优劣。',
              '- end_not_learned_yet：≥10 步但 end acc=0，保留后续曲线观察，不能等同程序失败。',
              '- 原六条 toy 的100步诊断已结束；下一阶段使用独立 release/profile 的广长度、条件链和独立人工 heldout 诊断，按25步记录并保留同方法不同预算实例。不能把重新初始化称为续训。',
              '- 人工数据的25/500步等结果只验证实现、优化与人工规则迁移；不能替代真实语料清理证书、真实数据验证、下游测试与多个 seed。', '',
              '## 有效变化历史', '']
    identifiability = read(directory / 'full_mask_identifiability.json')
    if identifiability:
        aa_bound = identifiability.get('aa_only', {})
        joint_bound = identifiability.get('all_targets', {})
        lines[-2:] = ['## 原六条 toy 全 mask 诊断的可识别性限制', '',
                     '以下只适用于原 R00–R19 六条 toy 协议，不能用于后续扩展人工 heldout。CPU 实际比较全部 7 个 denoise 输入特征：4 条 TCR 在全 mask 后输入完全相同，标签不同；AB/Nb 各自可区分。输入分组：'+cell(identifiability.get('full_mask_input_groups'))+'。', '',
                     f'原 6 条 toy 的 AA-only 准确率理论上限 {aa_bound.get("maximum_correct")}/{aa_bound.get("target_count")}={aa_bound.get("accuracy_upper_bound", 0):.2%}；全部目标 token 联合上限 {joint_bound.get("maximum_correct")}/{joint_bound.get("target_count")}={joint_bound.get("accuracy_upper_bound", 0):.2%}。两种上限分别优化对应指标，不声称可同时达到。', '',
                     '原100步不能以AA/end同时100%为通过标准。后续R20–R27采用广长度和条件链人工诊断，应使用对应fixture的独立可识别性审计，不能沿用旧toy上限。', '',
                     '证据：'+cell(str(directory/'full_mask_identifiability.json'))+'。', '', '## 有效变化历史', '']
    expanded = read(directory / 'expanded_identifiability.json')
    if expanded:
        lines[-2:] = ['## 扩展人工 heldout 的独立可识别性限制', '',
                     '仅对应fixture manifest SHA='+cell(expanded.get('fixture_manifest_sha256'))+'。输入确定性argmax上限不等于模型可学习性或生物泛化保证。', '',
                     '| 子集 | full-mask输入组数 | full-mask AA-only上限 | 联合token上限 | 联合argmax最优时end范围 | 四档mean-AA上限 |',
                     '| --- | --- | --- | --- | --- | --- |']
        groups = expanded.get('groups', {})
        for name, values in groups.items() if isinstance(groups, dict) else []:
            if not isinstance(values, dict):
                continue
            full = values.get('levels', {}).get('mask100', {})
            row = (name, full.get('input_group_count'), full.get('aa', {}).get('accuracy_upper_bound'), full.get('selected', {}).get('accuracy_upper_bound'), full.get('joint_optimal_terminal_accuracy_range'), values.get('mask_mean_aa_upper_bound'))
            lines.append('| '+' | '.join(cell(value) for value in row)+' |')
        lines += ['', 'end范围只限定为同时达到联合argmax最优；terminal单独优化可更高，AA-only与end-only最优不可混同。条件组仍可区分，不能把其end未学习全部归因于无条件输入碰撞。', '',
                  '证据：'+cell(str(directory/'expanded_identifiability.json'))+'；详细方法：'+cell(str(ROOT/'docs/IMMUNE_RESEARCH_EXPANDED_IDENTIFIABILITY.md'))+'。', '', '## 有效变化历史', '']
    for event in events:
        changes = '; '.join(f'{change.get("variant")} ({change.get("job_id")}): {change["state"]}, step={change.get("step")}/{change.get("requested_steps", "?")}, alerts={change.get("alerts")}' for change in event['changes'])
        lines.append('- '+stamp(event['time'])+' — '+cell(changes))
    lines += ['', '证据目录：'+cell(str(pool))+ '。完整监督快照：'+cell(str(directory/'supervision.json'))+'。', '']
    atomic(directory / 'supervision.json', json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    atomic(document, '\n'.join(lines))
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool-dir', type=Path, required=True)
    args = parser.parse_args()
    result = update(args.pool_dir.resolve())
    print(json.dumps(dict(counts=result['counts'], all_initial_done=result['all_initial_done'], alerts={record.get('variant'): record.get('alerts') for record in result['records'].values() if record.get('alerts')}), ensure_ascii=False))
