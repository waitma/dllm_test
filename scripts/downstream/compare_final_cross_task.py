"""Compare a completed final-layer AB/TCR probe against the original checkpoint.

Activate pllm/protenix_abtcr; from repository root run:
python -B scripts/downstream/compare_final_cross_task.py --task specificity --run ABS_TASK_DIR
For specificity/m396, first run verify_final_preservation_probe.py. T3 checks
source support sampling and aggregation, not an independent distance recompute.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['specificity','m396','t3broad'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    parent = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013'/args.task
    sources = {}

    def read(path):
        sources[str(path)] = sha(path)
        return json.loads(path.read_text())

    assert read(run/'status.json')['status'] == 'success'
    assert read(run/'readout_verification.json')['saved_gdp_features_bitwise_equal']
    execution = read(run/'execution.json')
    for path, digest in execution['code_sha256'].items():
        assert sha(Path(path)) == digest, path
    import numpy as np
    checks, rows = {}, []
    if args.task in ('specificity','m396'):
        assert read(run/'prediction_verification.json')['status'] == 'passed'
        current, baseline = read(run/'run_manifest.json'), read(parent/'run_manifest.json')
        for key in ['input_audit','data_sha256','sample_ids','runner_sha256']:
            assert current[key] == baseline[key], key
        ignored = {'checkpoint', 'out_dir'}
        assert {k:v for k,v in current['args'].items() if k not in ignored} == {
            k:v for k,v in baseline['args'].items() if k not in ignored}
        for path,digest in current['data_sha256'].items():
            assert sha(Path(path)) == digest, path
        checks.update(original_data_and_inputs_equal=True, head_arguments_equal=True,
                      prediction_rescoring_passed=True)
        new, old = read(run/'metrics.json'), read(parent/'metrics.json')
        if args.task == 'specificity':
            for key in ['accuracy','f1_macro','mcc']:
                rows.append(dict(metric=key, original=old['mean'][key], adapted=new['mean'][key],
                                 delta=new['mean'][key]-old['mean'][key],
                                 fold_deltas=[b[key]-a[key] for a,b in zip(old['folds'],new['folds'])]))
        else:
            assert len(old['splits']) == len(new['splits']) == 12
            for a,b in zip(old['splits'],new['splits']):
                for key in ['split_protocol','train_fraction','n_train','n_test','fraction_unit',
                            'target_scale','train_index_sha256','test_index_sha256','targets']:
                    assert a[key] == b[key], key
                rows.append(dict(protocol=b['split_protocol'],train_fraction=b['train_fraction'],
                                 original=a['spearman_mean'],adapted=b['spearman_mean'],
                                 delta=b['spearman_mean']-a['spearman_mean']))
    else:
        from scripts.downstream.run_tcr_native_repr import inputs, receptor_groups
        frames, paths, _ = inputs('t3broad')
        train,test = [receptor_groups(f) for f in frames]
        identity = read(run/'data_identity.json')
        assert {str(p):sha(p) for p in paths} == identity
        new, old = read(run/'fewshot.json'),read(parent/'fewshot.json')
        for key in ['protocol','shots','seeds','agg','n_train','n_test','multilabel_pairs','n_epitopes_by_shot']:
            assert new[key] == old[key], key
        assert new['n_train'] == len(train) and new['n_test'] == len(test)
        assert not set(zip(train.cdr3b,train.cdr3a)) & set(zip(test.cdr3b,test.cdr3a))
        assert len(new['episodes']) == len(old['episodes']) == 775
        for old_episode, episode in zip(old['episodes'], new['episodes']):
            for key in ['k','epitope','seed','support_indices']:
                assert episode[key] == old_episode[key], key
            k, ep, seed = episode['k'], episode['epitope'], episode['seed']
            pool = np.array([i for i,labels in enumerate(train.peptide) if ep in labels])
            expected = np.random.default_rng(seed+1000*k).choice(pool,k,replace=False).tolist()
            assert episode['support_indices'] == expected
            assert 0 <= episode['auroc'] <= 1
        for k in new['shots']:
            episodes = [e for e in new['episodes'] if e['k']==k]
            per_ep = {ep:float(np.mean([e['auroc'] for e in episodes if e['epitope']==ep]))
                      for ep in sorted({e['epitope'] for e in episodes})}
            assert len(per_ep) == new['n_epitopes_by_shot'][str(k)]
            for ep,value in per_ep.items():
                assert abs(value-new['per_epitope_by_shot'][str(k)][ep]) < 1e-12
            score = new['auroc_by_shot'][str(k)]['mean']
            assert abs(np.mean(list(per_ep.values()))-score) < 1e-12
            old_score = old['auroc_by_shot'][str(k)]['mean']
            rows.append(dict(shot=k,original=old_score,adapted=score,delta=score-old_score))
        checks.update(current_data_hashes_verified=True, support_indices_match_parent=True,
                      original_sampling_recomputed=True, matched_episodes=775,
                      aggregation_recomputed=True, distance_predictions_independently_rescored=False)
    sources[str(Path(__file__))] = sha(Path(__file__))
    report = dict(status='passed',task=args.task,created_utc=datetime.now(timezone.utc).isoformat(),
                  baseline=str(parent),attention_mode=execution['attention_mode'],checks=checks,results=rows,
                  interpretation='Comparison of adapted weights plus explicit readout against original native pipeline; not an isolated training-effect attribution or independent-seed confirmation.',
                  source_sha256=sources)
    write(run/'comparison_verification.json',report)
    print(json.dumps(dict(task=args.task,results=rows),indent=2))


if __name__=='__main__':
    main()
