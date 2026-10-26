"""Verify fresh reproduction and run our Ridge on the identical official features.

Activate pllm then protenix_abtcr; python -B scripts/downstream/verify_ophiuchus_gdpa1.py
--run ABS_FRESH_REPRODUCTION_DIRECTORY. Writes separate verification artifacts;
does not rerun the backbone or modify the original metrics/predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    run = p.parse_args().run
    import numpy as np
    import pandas as pd
    import torch
    from scipy.stats import spearmanr
    from downstream.grammar.ab_probes import run_gdp
    torch.set_num_threads(8)
    metrics = json.loads((run/'metrics.json').read_text())
    plan = json.loads((run/'plan.json').read_text())
    assert metrics['status'] == 'passed'
    assert hashlib.sha256((run/'plan.json').read_bytes()).hexdigest() == metrics['plan_sha256']
    assert hashlib.sha256(Path(plan['data']).read_bytes()).hexdigest() == plan['data_sha256']
    df = pd.read_csv(plan['data'])
    features = np.load(run/'features.npz')
    assert features['features'].shape == (len(df), 2560)
    assert np.array_equal(features['row_index'], np.arange(len(df)))
    assert np.isfinite(features['features']).all()
    first = torch.load(run/'first_batch.pt', map_location='cpu', weights_only=True)
    np.testing.assert_array_equal(first['features'].numpy(), features['features'][:len(first['features'])])
    folder = run/'ours_same_features'
    folder.mkdir(exist_ok=False)
    ours = run_gdp(df, torch.from_numpy(features['features']), SimpleNamespace(out_dir=folder))
    (folder/'metrics.json').write_text(json.dumps(ours, indent=2)+'\n')
    old = json.loads((ROOT/'output/downstream_generation/ophiuchus_ab/developability_gdp_a1_metrics.json').read_text())
    checks = {}
    for prop, item in metrics['properties'].items():
        a = np.load(run/prop/'oof.npz')
        indices = np.flatnonzero(df[prop].notna().to_numpy())
        assert np.array_equal(a['row_index'], indices)
        assert np.array_equal(a['y_true'], df.iloc[indices][prop].to_numpy())
        assert np.array_equal(a['fold'], df.iloc[indices]['hierarchical_cluster_IgG_isotype_stratified_fold'].to_numpy())
        scores = [float(spearmanr(a['y_transformed'][a['fold']==k], a['prediction'][a['fold']==k]).statistic) for k in range(5)]
        assert abs(float(np.mean(scores))-item['cv_mean_spearman']) < 1e-10
        value = ours['properties'][prop]['legacy_reference']
        assert abs(value['cv_mean_spearman']-item['cv_mean_spearman']) < 1e-10
        assert value['alpha'] == item['alpha']
        oof = np.load(folder/prop/'legacy_oof.npz')
        np.testing.assert_allclose(a['prediction'],oof['prediction'],atol=1e-6,rtol=1e-6)
        checks[prop] = dict(n=len(indices), fold_scores=scores, mean_recomputed=float(np.mean(scores)),
                            ours_score=value['cv_mean_spearman'], alpha=value['alpha'],
                            difference_from_historical=item['cv_mean_spearman']-old['properties'][prop]['cv_spearman_transformed'],
                            matches_paper_3dp=(round(item['cv_mean_spearman'],3)==item['paper']) if item['paper'] is not None else None)
    receipt = dict(status='passed', checks=checks, saved_rows_labels_folds_verified=True,
                   actual_ours_runner_on_official_features=True,
                   verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
