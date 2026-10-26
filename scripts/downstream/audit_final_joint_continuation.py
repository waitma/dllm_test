"""Verify and compare the matched CE0/CE1 fixed5000 continuations.

Activate pllm then protenix_abtcr; from the repository root run:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_final_joint_continuation.py \
  --out ABS_NEW_DIRECTORY
Requires both runs terminal-success. Recomputes all saved GDP fold scores,
checks matched initialization/schedule/optimizer and checkpoint identities,
and records endpoint and foldwise differences. This is exploratory analysis
of the fixed reference protocol, not an independent confirmation experiment.
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

    root = ROOT/'output/diagnostics'
    runs = {name: root/f'final_layer_{stem}convergence_pilot_20261018'
            for name, stem in [('ce0', ''), ('ce1', 'joint_')]}
    plans, checkpoints, summaries = {}, {}, {}
    hashes, checks = {}, []

    def read(path):
        hashes[str(path)] = sha(path)
        return json.loads(path.read_text())

    def array(path):
        hashes[str(path)] = sha(path)
        return np.load(path)

    for name, run in runs.items():
        assert read(run/'status.json')['status'] == 'success'
        plans[name] = read(run/'plan.json')
        folder = run/'token_all_direction'
        side = read(folder/'checkpoint.json')
        checkpoints[name] = side
        assert sha(Path(side['parameter_file'])) == side['sha256']
        verification = read(folder/'verification.json')
        assert verification['status'] == 'passed'
        assert verification['reloaded_features_bitwise_equal']
        assert verification['frozen_parameters_bitwise_equal']
        assert len(verification['oof_checks']) == 10
        for source, digest in plans[name]['source_sha256'].items():
            assert sha(Path(source)) == digest, source
        summaries[name] = read(run/'results.json')['token_all_direction']

    matched = ['initial_checkpoint', 'initial_parameter_sha256', 'parent_checkpoint',
               'initial_steps', 'additional_steps', 'total_steps', 'data_sha256',
               'schedule_sha256', 'seed', 'batch_size', 'lr', 'initial_optimizer_sha256',
               'calibration_sha256']
    for key in matched:
        assert plans['ce0'][key] == plans['ce1'][key], key
    assert read(runs['ce0']/'schedule.json') == read(runs['ce1']/'schedule.json')
    assert checkpoints['ce0']['trainable_names'] == checkpoints['ce1']['trainable_names']
    assert checkpoints['ce0']['parent_identity'] == checkpoints['ce1']['parent_identity']
    first = {n: read(p/'token_all_direction/first_continued_batch.json') for n,p in runs.items()}
    for key in ['step', 'ce', 'representation_loss', 'details']:
        assert first['ce0'][key] == first['ce1'][key], key
    assert first['ce1']['ce_requires_grad'] is True
    assert read(runs['ce1']/'token_all_direction/paired_first_batch_verification.json')['status'] == 'passed'

    teacher_dir = root/'fixed50000_depth_20261018/native_block2'
    teacher = read(teacher_dir/'metrics.json')
    results = {}
    for mode, dirname in [('native', 'gdp_a1'), ('no_mask_keys', 'gdp_no_mask_keys')]:
        results[mode] = {}
        for prop, reference in teacher['properties'].items():
            data = {'block2': array(teacher_dir/prop/'legacy_oof.npz')}
            data.update({n: array(p/'token_all_direction'/dirname/prop/'legacy_oof.npz') for n,p in runs.items()})
            scores = {}
            for name, saved in data.items():
                for key in ['row_index', 'fold', 'y_true', 'y_transformed']:
                    np.testing.assert_array_equal(saved[key], data['block2'][key])
                fold = saved['fold']
                values = [float(spearmanr(saved['y_transformed'][fold == k], saved['prediction'][fold == k]).statistic) for k in range(5)]
                recorded = (reference if name == 'block2' else summaries[name][mode]['properties'][prop])['legacy_reference']['cv_mean_spearman']
                assert abs(np.mean(values)-recorded) < 1e-12
                scores[name] = dict(mean=float(np.mean(values)), folds=values)
                checks.append(dict(mode=mode, property=prop, model=name, rows=len(fold)))
            delta = lambda a,b: (np.array(scores[a]['folds'])-np.array(scores[b]['folds'])).tolist()
            results[mode][prop] = dict(scores=scores, ce1_minus_ce0=delta('ce1','ce0'),
                                       ce1_minus_block2=delta('ce1','block2'))
    validation = {name: item['validation'] for name,item in summaries.items()}
    assert validation['ce0']['step'] == validation['ce1']['step'] == 5000
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  GDP=results, validation=validation,
                  validation_ce1_minus_ce0={k:validation['ce1'][k]-validation['ce0'][k] for k in validation['ce0'] if k != 'step'},
                  limitations='Single seed; reference same-CV alpha selection and transformed labels. Masked AA accuracy is not CDR infilling AAR. No cross-task preservation claim.',
                  completion='Not established by this diagnostic alone: requires final-layer property gains and cross-task validation.')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out/'results.json', report)
    hashes[str(Path(__file__))] = sha(Path(__file__))
    write(out/'verification.json', dict(status='passed', matched_plan_fields=matched,
          first_batch_equal=True, actual_checkpoints_verified=True, oof_checks=checks, source_sha256=hashes))
    write(out/'status.json', dict(status='success', finished_utc=datetime.now(timezone.utc).isoformat()))
    for mode, properties in results.items():
        print(mode, {p:{n:round(s['mean'],6) for n,s in v['scores'].items()} for p,v in properties.items()}, flush=True)
    print('validation', validation, flush=True)


if __name__ == '__main__':
    main()
