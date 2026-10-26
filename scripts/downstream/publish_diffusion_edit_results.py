"""Publish only completed diffusion/edit task artifacts into the existing tables.

Activate pllm/protenix_abtcr; python -B THIS_FILE --run ABSOLUTE_RUN.
Reads scores, checks their checkpoint identity and generation audit receipts,
and appends Ours rows at the end of each original task table under a shared lock.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
from downstream.bert_full_token_eval import sha
from scripts.downstream.run_bert_representation_suite import report, write


def read(path):
    return json.loads(Path(path).read_text())


def number(value):
    if value is None:
        return '—'
    assert math.isfinite(value), value
    return f'{value:.6f}'


def table_bounds(text, heading):
    marker = '\n| 本地方法 / checkpoint' if heading == '#### T3-deep：' else '\n|'
    start = text.index(marker, text.index(heading)) + 1
    end = text.find('\n\n', start)
    return start, len(text) if end < 0 else end


def collect(run, plan):
    """Produce task-specific rows; every numeric cell retains a concrete artifact."""
    label = f"Ours diffusion/edit V7 / {plan['step']}（BERT287000 起始）"
    rows, completed, artifacts = [], [], {}
    t2 = None

    def add(heading, path, values, suffix=''):
        name = label + suffix
        rows.append((heading, name, Path(path), values))
        artifacts[str(path)] = sha(path)

    for task in plan['tasks']:
        folder = run / task
        if not (folder / 'status.json').exists() or read(folder / 'status.json')['status'] != 'success':
            continue
        identity = read(folder / 'model_identity.json')
        assert identity['step'] == plan['step'] and identity['model_sha256'] == plan['model_sha256']
        completed.append(task)
        if task in plan['generation_tasks']:
            audit = read(folder / 'verification.json')
            assert audit['status'] == 'passed'
            if task == 'cdr':
                for dataset, heading, modes, suffix in [
                    ('kong', '#### SAbDab Kong：H1 / H2 / H3',
                     ['cdrh1', 'cdrh2', 'cdrh3'], '，iter=1 / bf16'),
                    ('sab23', '#### SAb23H2：六个 CDR',
                     ['cdrh1', 'cdrh2', 'cdrh3', 'cdrl1', 'cdrl2', 'cdrl3'], '，iter=2 / fp32'),
                ]:
                    path = folder / dataset / 'metrics.json'
                    assert audit['datasets'][dataset]['predictions_sha256'] == sha(folder / dataset / 'predictions.jsonl')
                    metric = read(path)
                    add(heading, path, [number(metric['modes'][m]['airgen_aar_percent']) for m in modes], suffix)
            elif task == 'pairing':
                assert audit['predictions_sha256'] == sha(folder / 'predictions.jsonl')
                path = folder / 'metrics.json'
                metric = read(path)
                assert metric['total_pairs'] == audit['candidates'] == 4000
                add('AB Light-chain pairing', path,
                    [number(metric[k]) for k in ('gen_immunomatch_mean', 'gen_valid_rate',
                                                  'overall_chain_match_rate', 'diversity_mean')],
                    '，p0 / reference / iter=16')
            else:
                assert audit['designs_sha256'] == sha(folder / 'designs.jsonl')
                path = folder / 'metrics.json'
                metric = read(path)
                views = [('benchmark14', '#### T4-1 benchmark14：'),
                         ('paper_sparse13', '#### 论文主视图 sparse13：'),
                         ('historical_bioseq_unseen_common', '#### common6：')] if task.endswith('benchmark14') else [
                             ('held20', '#### T4-2 held20：')]
                for view, heading in views:
                    if view not in metric['summary']:
                        continue
                    summary = metric['summary'][view]
                    prefix = ([] if view.endswith('_common') else ['≤100'] if view == 'paper_sparse13'
                              else [f"{summary['n_pmhc_scored']}/{summary['n_pmhc']}"])
                    add(heading, path, [*prefix,
                        *[number(summary[k]) for k in ('d_edit', 'seq_recovery', 'f1')]],
                        '，native β EOS / iter=32')
        elif task == 'specificity':
            path = folder / 'metrics.json'
            metric = read(path)
            assert metric['selected_epoch'] == 100
            add('#### Specificity：', path,
                ['100', *[number(metric['mean'][k]) for k in ('accuracy', 'f1_macro', 'mcc')]])
        elif task == 'gdp_a1':
            path = folder / 'metrics.json'
            metric = read(path)
            add('#### GDPa1：', path, [number(metric['properties'][k]['legacy_reference']['cv_mean_spearman'])
                for k in ('AC-SINS_pH7.4', 'HIC', 'PR_CHO', 'Titer', 'Tm2')])
        elif task == 'm396':
            path = folder / 'metrics.json'
            splits = read(path)['splits']
            main = sorted([s for s in splits if s['split_protocol'] == 'row_reference'],
                          key=lambda s: s['train_fraction'])
            assert [s['train_fraction'] for s in main] == [.005, .01, .02, .05, .1, .2]
            add('#### m396：突变体计算能量预测', path, [number(s['spearman_mean']) for s in main])
            grouped = next(s for s in splits if s['split_protocol'] == 'sequence_grouped'
                           and s['train_fraction'] == .2)
            assert grouped['test_rows_sequence_seen_in_train'] == 0
            add('#### m396：按序列分组划分补充结果', path, [number(grouped['spearman_mean'])])
        elif task in ('t3broad', 't3deep'):
            base = folder if task == 't3broad' else Path(read(folder / 'result_location.json')['path'])
            path = base / 'fewshot.json'
            metric = read(path)
            shots = [1, 2, 5, 10, 20, 50, 100] + ([200] if task == 't3deep' else [])
            add('#### T3-' + task[2:] + '：', path,
                [number(metric['auroc_by_shot'][str(s)]['mean']) for s in shots])
        elif task == 't2a':
            path = Path(read(folder / 'result_location.json')['path']) / 'metrics.json'
            metric = read(path)
            t2 = label, path, metric['aligned_points']
            artifacts[str(path)] = sha(path)
        elif task.startswith('t1-'):
            track = task[3:]
            path = folder / 'protocol_outputs' / (run.name + '_' + task) / track / 'AS/summary.json'
            metric = read(path)
            cells = []
            for split in ('seen_test', 'seen_independent', 'unseen_independent'):
                item = metric['eval_sets'][split]
                assert item['folds_complete'] == [1, 2, 3, 4, 5]
                cells.append(number(item['local']['auroc_mean']) + ' / ' +
                             number(item['local']['auprc_precrec_mean']))
            index = {'cdr3b': 1, 'others_cdr3b': 2, 'cdr3ab': 3, 'others_longab': 4}[track]
            add(f'#### T1-{index} ', path, cells)
        elif task == 'uptcr':
            for setting in ('majority', 'few-shot', 'zero-shot'):
                path = folder / 'protocol_outputs' / setting / 'metrics.json'
                metric = read(path)
                assert metric['checkpoint_sha256'] == plan['model_sha256']
                add('#### Ours：作者发布包划分', path,
                    [str(metric['n_test']), number(metric['auroc']),
                     number(metric['auprc_average_precision'])], ' / ' + setting)
    return rows, t2, completed, artifacts


def render_rows(text, rows):
    """Respect the two existing table orientations and optional evidence column."""
    for heading, name, evidence, values in rows:
        start, end = table_bounds(text, heading)
        if name in text[start:end]:
            continue
        lines = text[start:end].splitlines()
        headers = [v.strip() for v in lines[0].split('|')[1:-1]]
        if headers[0] == '属性':
            assert heading == '#### GDPa1：' and len(lines) == len(values) + 2
            lines[0] += f' [{name}]({evidence}) |'
            lines[1] += ' ---: |'
            for index, value in enumerate(values, 2):
                lines[index] += ' ' + value + ' |'
            text = text[:start] + '\n'.join(lines) + text[end:]
            continue
        cells = list(values)
        if headers[-1].startswith('来源') or headers[-1] == '证据':
            cells.append(f'[本地评测产物]({evidence})')
        row = '| ' + f'[{name}]({evidence})' + ' | ' + ' | '.join(cells) + ' |'
        assert len(row.split('|')) == len(lines[0].split('|')), heading
        text = text[:end] + '\n' + row + text[end:]
    return text


def publish(run):
    plan = read(run / 'plan.json')
    rows, t2, completed, artifacts = collect(run, plan)
    previous = read(run / 'publication.json') if (run / 'publication.json').exists() else {}
    additions = set(completed) - set(previous.get('completed', []))
    if not additions:
        return
    bench = ROOT / 'downstream/benchmark'
    lock_path = ROOT / 'output/downstream_generation/v5_native_best_20261008/results_update.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for filename in ('RESULTS.md', 'RESULTS_SUMMARY.md'):
            path = bench / filename
            original = text = path.read_text()
            text = render_rows(text, rows)
            if t2:
                name, evidence, points = t2
                start, end = table_bounds(text, '#### T2A：')
                if name not in text[start:end]:
                    lines = text[start:end].splitlines()
                    lines[0] += f' [{name}]({evidence}) |'
                    lines[1] += ' ---: |'
                    for i in range(2, len(lines)):
                        baseline = lines[i].split('|')[1].strip()
                        point = next(v for k, v in points.items() if k.startswith('purity@ret~' + baseline + '('))
                        value = (number(point['retention']) + ' / ' + number(point['purity'])
                                 if point['comparison_allowed'] else '未对齐')
                        lines[i] += ' ' + value + ' |'
                    text = text[:start] + '\n'.join(lines) + text[end:]
            if text != original:
                temporary = path.with_suffix('.diffedit.tmp')
                temporary.write_text(text)
                temporary.replace(path)
        stamp = datetime.now(timezone.utc).isoformat()
        with (bench / 'PROGRESS.md').open('a') as stream:
            stream.write(f'\n{stamp} diffusion/edit {plan["step"]} 完成并回填原任务表：'
                         + '、'.join(sorted(additions)) + f'。[评测产物]({run})。\n')
    write(run / 'publication.json', dict(completed=completed, artifacts_sha256=artifacts,
                                         updated_utc=stamp, row_count=len(rows)))
    report(run)
    print(json.dumps(dict(published=sorted(additions), complete=len(completed))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    publish(args.run.resolve())
