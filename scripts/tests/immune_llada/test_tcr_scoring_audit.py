"""Run: python -m pytest -q scripts/tests/immune_llada/test_tcr_scoring_audit.py.

Scoring regressions: no weights, no quality evaluation.
"""
import json

import numpy as np
import pandas as pd
import pytest

from downstream.benchmark.tcr_clustering import run as clustering
from downstream.benchmark.common import fewshot
from scripts.downstream.run_tcr_native_repr import receptor_groups, multilabel_fewshot


def tight_similarity():
    sim = np.eye(6, dtype=np.float32)
    for i, j, value in [(0, 1, .999994), (2, 3, .999993), (4, 5, .999992)]:
        sim[i, j] = sim[j, i] = value
    return sim


def test_t2_thresholds_preserve_float32_edges():
    sim = tight_similarity()
    taus = clustering._taus_from_targets(sim, [1/3, 2/3, 1.0])
    assert len(taus) == 3
    assert sorted((clustering._components_at_threshold(sim, t) >= 0).sum()
                  for t in taus) == [2, 4, 6]


def test_t2_paper_anchors_and_high_retention():
    expected = dict(clusTCR=.09, TCRMatch=.09, GLIPH2=.22, TCRdist3=.44,
                    GIANA=.25, DeepTCR=.92, HD=.27, LD=.31, iSMART=.18)
    assert clustering.PAPER_RETENTIONS == expected
    assert set(expected.values()) <= set(clustering.TARGET_RETENTIONS)


def test_t2_curve_serializes_exact_threshold_and_labelblind_selection(tmp_path, monkeypatch):
    # Tiny angular separation must survive CSV/JSON serialization, with no
    # epitope-dependent threshold choice or unjustified retention alignment.
    theta = np.array([0, .001, .004, .01, .02, .03])
    x = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    frame = pd.DataFrame(dict(peptide=['a', 'a', 'b', 'b', 'c', 'c']))
    monkeypatch.setattr(clustering, '_embed_features', lambda *a: (x, 'fake'))
    monkeypatch.setattr(clustering, 'OUT', tmp_path)
    clustering.run_embed_threshold(frame, 'fake', 'a', False, None)
    clustering.run_embed_threshold(frame.iloc[::-1], 'fake', 'b', False, None)
    a = pd.read_csv(tmp_path / 'a/curve.csv', float_precision='round_trip')
    b = pd.read_csv(tmp_path / 'b/curve.csv', float_precision='round_trip')
    np.testing.assert_array_equal(a.tau, b.tau)
    np.testing.assert_array_equal(a.retention, b.retention)
    for row in a.itertuples():
        actual = (clustering._components_at_threshold(x @ x.T, row.tau) >= 0).sum()
        assert row.n_clustered == actual
    result = json.loads((tmp_path / 'a/metrics.json').read_text())
    assert result['taus'] == a.tau.tolist()
    for point in result['aligned_points'].values():
        assert point['comparison_allowed'] == (point['retention_gap'] <= .02)


def test_t3_deep_excludes_self_support(monkeypatch):
    # All off-diagonal distances tie. Including self would inflate AUROC;
    # after removing support, every episode must be exactly chance.
    labels = np.array(['a'] * 4 + ['b'] * 4)
    d = np.ones((8, 8), dtype=np.float32) - np.eye(8, dtype=np.float32)
    calls = []
    original = fewshot.metrics._safe_auroc
    def capture(y, score):
        calls.append(len(y))
        return original(y, score)
    monkeypatch.setattr(fewshot.metrics, '_safe_auroc', capture)
    result = fewshot.few_shot_paper_auroc(labels, ['a', 'b'], shots=[1, 2], seeds=2,
        pairwise_fn=lambda idx: d[:, idx])
    assert set(calls) == {6, 7}
    assert all(v['mean'] == pytest.approx(.5) for v in result['auroc_by_shot'].values())


def test_t3_broad_multilabel_support_identity():
    raw = pd.DataFrame(dict(cdr3b=['b1', 'b1', 'b2'], cdr3a=['a1', 'a1', ''],
                            peptide=['p1', 'p2', 'p1']))
    train = receptor_groups(raw)
    assert len(train) == 2 and train.peptide.iloc[0] == ('p1', 'p2')
    test = pd.DataFrame(dict(peptide=[('p1',), ('p2',)]))
    report = multilabel_fewshot(np.array([[0., 0.], [1., 1.]]), train, test,
                               shots=[1], seeds=2)
    assert all(len(e['support_indices']) == len(set(e['support_indices'])) == 1
               for e in report['episodes'])
    assert report['n_epitopes_by_shot'][1] == 2


@pytest.mark.parametrize('scores,expected', [([.1, .2, .8, .9], 1.),
    ([.9, .8, .2, .1], 0.), ([.5, .5, .5, .5], .5)])
def test_t1_score_direction(scores, expected):
    from downstream.benchmark.tcr_binding.run_retrained_ours import paper_metrics
    result = paper_metrics(np.array([0, 0, 1, 1]), np.array(scores))
    assert result['auroc'] == pytest.approx(expected)


def test_t4_perfect_reference_raw_budget_and_greedy(monkeypatch):
    from downstream.benchmark.tcr_generation_bench import scoring
    ref = 'CASSLGQETQYF'
    control = 'CASSIRSSYEQYF'
    seen = []
    def bleu(hyp, refs, **kwargs):
        seen.append(hyp)
        return .123
    monkeypatch.setattr(scoring.M, 'tcrt5_sequence_bleu', bleu)
    result = scoring.score_conditional({'p': [ref] * 100},
        {'p': dict(epitope='AAAA', category='held20', ref_binders=[ref])},
        k=100, greedy_map={'p': control}, use_conditioning_for_novelty=False)
    item = result['per_pmhc']['p']
    assert item['n_generated'] == 100 and item['n_unique'] == 1
    assert item['d_edit'] == 0 and item['seq_recovery'] == 1
    assert item['diversity'] == .01 and seen == [control]
    assert result['summary']['overall']['n_pmhc_scored'] == 1


def test_t4_missing_designs_not_counted_as_scored():
    from downstream.benchmark.tcr_generation_bench import scoring
    entries = {p: dict(epitope='AAAA', category='held20', ref_binders=['CASSLGQETQYF'])
               for p in ('p', 'q')}
    result = scoring.score_conditional({'p': ['CASSLGQETQYF']}, entries,
        k=1, use_conditioning_for_novelty=False)
    assert result['summary']['overall']['n_pmhc'] == 2
    assert result['summary']['overall']['n_pmhc_scored'] == 1
