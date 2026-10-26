"""Verify fixed5000 complementary-teacher endpoints and saved GDP predictions.

Activate pllm/protenix_abtcr; run from repository root:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_final_complementary_teacher.py --out ABS_NEW_DIR
Requires both training arms terminal. Compares actual final features, never
substitutes direct teacher features for the trained model's results.
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
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    root = ROOT/'output/diagnostics'
    run = root/'final_layer_complementary_teacher_pilot_20261018'
    hashes = {}

    def read(path):
        hashes[str(path)] = sha(path)
        return json.loads(path.read_text())

    def array(path):
        hashes[str(path)] = sha(path)
        return np.load(path)

    assert read(run/'status.json')['status'] == 'success'
    plan = read(run/'plan.json')
    for group in ['source_sha256', 'target_sha256', 'initial_artifact_sha256']:
        for path, digest in plan[group].items():
            assert sha(Path(path)) == digest, path
    assert sha(run/'schedule.json') == plan['schedule_sha256']
    assert len(read(run/'schedule.json')) == 3300
    assert sha(Path(plan['initial_checkpoint'])/'optimizer.pt') == plan['initial_optimizer_sha256']
    results = read(run/'results.json')
    assert set(results) == set(plan['arms']) == {'early512', 'early256_global256'}
    checks, validation, checkpoints = [], {}, {}
    for name in plan['arms']:
        folder = run/name
        checkpoint = read(folder/'checkpoint.json')
        checkpoints[name] = checkpoint
        assert checkpoint['steps'] == 5000
        assert sha(Path(checkpoint['parameter_file'])) == checkpoint['sha256']
        assert checkpoint['initial_adaptation_sha256'] == plan['initial_parameter_sha256']
        assert checkpoint['frozen_parameters_bitwise_equal']
        verification = read(folder/'verification.json')
        assert verification['status'] == 'passed'
        assert verification['reloaded_features_bitwise_equal']
        assert verification['frozen_parameters_bitwise_equal']
        assert len(verification['oof_checks']) == 10
        resume = read(folder/'resume_verification.json')
        assert resume['global_step'] == 1700 and resume['optimizer_steps'] == [1400]
        assert resume['optimizer_sha256'] == plan['initial_optimizer_sha256']
        target = read(folder/'target_verification.json')
        assert target['status'] == 'passed' and target['rank'] == 512
        assert not target['labels_used']
        validation[name] = read(folder/'validation.json')[-1]
        assert validation[name]['step'] == 5000
    assert checkpoints['early512']['trainable_names'] == checkpoints['early256_global256']['trainable_names']
    first = [read(run/name/'first_continued_batch.json') for name in plan['arms']]
    assert first[0]['ce'] == first[1]['ce'] and all(x['ce_requires_grad'] for x in first)
    teacher_dir = root/'fixed50000_depth_20261018/native_block2'
    teacher = read(teacher_dir/'metrics.json')
    comparison = {}
    for mode, dirname in [('native', 'gdp_a1'), ('no_mask_keys', 'gdp_no_mask_keys')]:
        comparison[mode] = {}
        for prop, reference in teacher['properties'].items():
            paths = {'block2': teacher_dir/prop/'legacy_oof.npz'}
            paths.update({name: run/name/dirname/prop/'legacy_oof.npz' for name in plan['arms']})
            data = {name: array(path) for name, path in paths.items()}
            scores = {}
            for name, saved in data.items():
                for key in ['row_index', 'fold', 'y_true', 'y_transformed']:
                    np.testing.assert_array_equal(saved[key], data['block2'][key])
                fold = saved['fold']
                values = [float(spearmanr(saved['y_transformed'][fold == k], saved['prediction'][fold == k]).statistic) for k in range(5)]
                expected = (reference if name == 'block2' else results[name][mode]['properties'][prop])['legacy_reference']['cv_mean_spearman']
                assert abs(np.mean(values)-expected) < 1e-12
                scores[name] = dict(mean=float(np.mean(values)), folds=values)
                checks.append(dict(mode=mode, property=prop, model=name, n=len(fold)))
            delta = lambda a,b: (np.array(scores[a]['folds'])-np.array(scores[b]['folds'])).tolist()
            comparison[mode][prop] = dict(scores=scores,
                mixed_minus_early=delta('early256_global256', 'early512'),
                mixed_minus_block2=delta('early256_global256', 'block2'))
    # Keep the teacher's fold probe fixed to distinguish lost target signal
    # from a changed refitted Ridge readout. Diagnostic only, no training use.
    transfer = {}
    for name in plan['arms']:
        folder = run/name
        target_dir = root/'global_teacher_targets_20261018'/name
        target = array(target_dir/'gdp_targets.npy').astype(np.float64)
        target_metrics = read(target_dir/'metrics.json')
        transfer[name] = {}
        for mode, feature_file, dirname in [('native','final_features.npy','gdp_a1'),
                ('no_mask_keys','final_no_mask_features.npy','gdp_no_mask_keys')]:
            student = array(folder/feature_file).astype(np.float64)
            assert student.shape == target.shape
            transfer[name][mode] = {}
            for prop, detail in target_metrics['properties'].items():
                saved = array(target_dir/prop/'legacy_oof.npz')
                actual = array(folder/dirname/prop/'legacy_oof.npz')
                for key in ['row_index','fold','y_true','y_transformed']:
                    np.testing.assert_array_equal(saved[key], actual[key])
                idx, folds, y = saved['row_index'], saved['fold'], saved['y_transformed']
                x, s = target[idx], student[idx]
                values = []
                for k in range(5):
                    train, test = folds != k, folds == k
                    probe = Ridge(alpha=detail['legacy_reference']['alpha']).fit(x[train],y[train])
                    reference = probe.predict(x[test])
                    np.testing.assert_allclose(reference,saved['prediction'][test],atol=1e-9,rtol=1e-9)
                    predicted = probe.predict(s[test])
                    error = predicted-reference
                    np.testing.assert_allclose(error,(s[test]-x[test])@probe.coef_,atol=1e-10,rtol=1e-10)
                    values.append(dict(fold=k,teacher=float(spearmanr(y[test],reference).statistic),
                        fixed_teacher_probe_on_student=float(spearmanr(y[test],predicted).statistic),
                        refit_student=float(spearmanr(y[test],actual['prediction'][test]).statistic),
                        prediction_rank_agreement=float(spearmanr(reference,predicted).statistic),
                        centered_error_over_signal=float(np.var(error)/np.var(reference))))
                transfer[name][mode][prop] = dict(folds=values,
                    mean={key:float(np.mean([v[key] for v in values])) for key in values[0] if key!='fold'})
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out/'results.json', dict(GDP=comparison, validation=validation, target_probe_transfer=transfer,
        limitation='Single seed, exploratory same-CV alpha selection; no cross-task preservation established for these arms. Teacher target and its own metric differ; gradient magnitudes are not matched.'))
    hashes[str(Path(__file__))] = sha(Path(__file__))
    write(out/'verification.json', dict(status='passed', fixed_endpoint=5000,
        actual_checkpoints_verified=True, matched_initial_optimizer_schedule=True,
        source_sha256=hashes, oof_checks=checks))
    write(out/'status.json', dict(status='success', finished_utc=datetime.now(timezone.utc).isoformat()))
    for mode, properties in comparison.items():
        print(mode, {p: {n: round(s['mean'], 6) for n,s in item['scores'].items()} for p,item in properties.items()}, flush=True)


if __name__ == '__main__':
    main()
