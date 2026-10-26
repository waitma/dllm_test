"""Run: python -m pytest scripts/tests/test_abbibench_scope.py -q."""
import json
from pathlib import Path
import pytest
from downstream.benchmark.ab_antigen.abbibench_scope import (
    ASSAYS, ASSAY_MAP, EXTRA_ASSAYS, select_rows, select_tasks,
    summarize_scores, summarize_run,
)
from downstream.benchmark.ab_antigen.interfaces import load_dataset, evaluate


def test_scope_matches_pinned_official_header():
    root = Path(__file__).resolve().parents[2]
    path = root / 'downstream/benchmark/baselines/ab_antigen_official/abbibench/README.md'
    if not path.exists():
        pytest.skip('Pinned official source is not downloaded')
    line = next(l for l in path.read_text().splitlines() if l.startswith('| Rank'))
    assert [c.strip() for c in line.strip('|').split('|')][3:-1] == list(ASSAY_MAP)


def test_default_view_preserves_repeated_rows_and_input_order():
    rows = [{'id': str(i), 'assay': assay} for i, assay in enumerate(ASSAYS + EXTRA_ASSAYS)]
    rows.insert(2, dict(rows[0], id='same-sequence-different-source-row'))
    selected = select_rows(rows)
    assert selected == rows[:-1] and len(selected) == 17
    assert select_rows(rows, include_extra_assays=True) == rows
    with pytest.raises(ValueError, match='include-extra-assays'):
        select_rows(rows, assay=EXTRA_ASSAYS[0])
    assert select_rows(rows, assay=EXTRA_ASSAYS[0], include_extra_assays=True) == rows[-1:]


def test_task_plan_does_not_inherit_supplementary_assays_by_default():
    tasks = ['ag-binding'] + ['ag-abbibench:' + a for a in ASSAYS + EXTRA_ASSAYS]
    assert select_tasks(tasks) == tasks[:-1]
    assert select_tasks(tasks, include_extra_assays=True) == tasks
    with pytest.raises(ValueError, match='Unknown'):
        select_tasks(['ag-abbibench:typo'])


def test_missing_assay_cannot_be_replaced_by_extra_assay():
    scores = {a: 0.2 for a in ASSAYS}
    scores[EXTRA_ASSAYS[0]] = 0.9
    result = summarize_scores(scores)
    assert result['completed_assays'] == 16 and result['mean_spearman'] == 0.2
    del scores[ASSAYS[0]]
    result = summarize_scores(scores)
    assert result['status'] == 'partial' and result['completed_assays'] == 15
    assert result['mean_spearman'] is None and result['missing_assays'] == [ASSAYS[0]]
    scores[ASSAYS[0]] = float('nan')
    assert summarize_scores(scores)['mean_spearman'] is None


def test_default_dataset_loader_and_scoring_share_scope(tmp_path):
    rows = [dict(id=f'{assay}:{i}', dataset='abbibench', assay=assay,
                 antigen='AC', heavy='DE', light='FG', wt_heavy='DE', wt_light='FG', label=float(i))
            for assay in ASSAYS + EXTRA_ASSAYS for i in range(3)]
    path = tmp_path / 'data.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    selected = load_dataset(path, 'abbibench')
    assert len(selected) == 48 and len(rows) == 51
    assert load_dataset(path, 'abbibench', include_extra_assays=True) == rows
    predictions = [dict(id=r['id'], score=r['label']) for r in selected]
    report = evaluate('abbibench', selected, predictions)
    assert report['benchmark16']['mean_spearman'] == 1.0
    assert report['benchmark16']['completed_assays'] == 16
    assert path.read_text().count('\n') == 51


def test_run_summary_rejects_wrong_assay_and_unfinished_results(tmp_path):
    folder = tmp_path / ('ag-abbibench:' + ASSAYS[0])
    folder.mkdir()
    data = dict(status='complete', per_assay=dict(groups=[dict(group=ASSAYS[1], spearman=.2)],
                                                macro=dict(spearman=dict(mean=.2))))
    (folder / 'metrics.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Assay/result mismatch'):
        summarize_run(tmp_path)
    data['per_assay']['groups'][0]['group'] = ASSAYS[0]
    (folder / 'metrics.json').write_text(json.dumps(data))
    assert summarize_run(tmp_path)['completed_assays'] == 1
    (folder / 'status.json').write_text(json.dumps(dict(status='running')))
    assert summarize_run(tmp_path)['completed_assays'] == 0
