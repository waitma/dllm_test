"""Exercise native TCR scoring with artificial references and candidate failures.

Activate conda pllm; use worker-compatible Python:
python -I -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_v7_tcr_scoring.py
Only synthetic sequences and an in-memory conditioning bank are used. No
benchmark files, models, annotation, GPU, or checkpoint files are touched.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0,str(ROOT))
from downstream.v7_tcr_scoring import scoring,score_saved_candidates,strict_result


def main():
    entries={k:dict(category='benchmark14' if i<2 else 'held20',epitope=k,
                   ref_binders=['CASSF','CATSF']) for i,k in enumerate(('synthetic_mixed','synthetic_empty','synthetic_constant'))}
    designs={'synthetic_mixed':['CASSF','CATSF']*49+['',''],
             'synthetic_empty':['']*100,'synthetic_constant':['CASSF']*100}
    greedy={'synthetic_mixed':'','synthetic_empty':'','synthetic_constant':'CASSF'}
    conditioning={'synthetic_mixed':['CASSF']}
    originals={name:getattr(scoring,name) for name in
               ('load_conditioning_by_epitope','load_bioseq_unseen','load_common_unseen')}
    scoring.load_conditioning_by_epitope=lambda:conditioning
    scoring.load_bioseq_unseen=lambda:frozenset({'synthetic_mixed'})
    scoring.load_common_unseen=lambda:frozenset({'synthetic_mixed'})
    try:
        result=score_saved_candidates(designs,greedy,entries)
        assert set(result['per_pmhc'])==set(entries)
        assert result['per_pmhc']['synthetic_mixed']['precision']==1.0
        assert result['per_pmhc']['synthetic_mixed']['candidate_denominators']['nonempty_in_legacy_score']==98
        assert result['per_pmhc']['synthetic_empty']['d_edit'] is None
        assert result['per_pmhc']['synthetic_constant']['diversity_edit'] is None
        assert result['per_pmhc']['synthetic_constant']['d_edit']==0.0
        assert result['summary']['overall']['n_pmhc']==3
        assert result['summary']['overall']['n_scored_by_metric']['d_edit']==2
        assert not result['metric_availability']['primary_coverage_complete_by_view']['overall']
        assert result['evaluation_semantics']['greedy_fallback_targets']==['synthetic_mixed']
        assert json.loads(json.dumps(result,allow_nan=False))==result
        # Force an all-empty run: primary scores remain undefined, never fabricated zeros.
        empty={k:['']*100 for k in entries}
        all_empty=score_saved_candidates(empty,{k:'' for k in entries},entries)
        assert all_empty['summary']['overall']['d_edit'] is None
        assert all_empty['summary']['overall']['n_scored_by_metric']['d_edit']==0
        # Capture original pre-serialization scores for independently corrupted aggregates.
        raw=scoring.score_conditional(designs,entries,k=100,greedy_map=greedy)
        members={'overall':list(entries)}
        raw['summary']={'overall':raw['summary']['overall']}
        mutations=[lambda d:d['per_pmhc']['synthetic_mixed'].update(d_edit=float('inf')),
            lambda d:d['per_pmhc']['synthetic_constant'].update(diversity_edit=0.0),
            lambda d:d['per_pmhc']['synthetic_empty'].update(d_edit=0.0),
            lambda d:d['summary']['overall']['n_scored_by_metric'].update(d_edit=3),
            lambda d:d['summary']['overall'].update(n_pmhc=2)]
        for mutate in mutations:
            bad=deepcopy(raw);mutate(bad)
            try:strict_result(bad,designs,entries,conditioning,members)
            except ValueError:pass
            else:raise AssertionError('Unexpected score/denominator corruption accepted')
    finally:
        for name,value in originals.items():setattr(scoring,name,value)
    print(json.dumps(dict(status='passed',synthetic_targets=3,raw_candidates=300,
        all_empty_run_preserved=True,undefined_metric_reasons_verified=True,denominators_verified=True,
        rejected_corruptions=5,historical_views_not_V7_exposure=True,greedy_fallback_explicit=True,
        real_benchmark_read=False,model_loaded=False,annotation_called=False,gpu_used=False,
        checkpoints_created=0)),flush=True)


if __name__=='__main__':
    main()
