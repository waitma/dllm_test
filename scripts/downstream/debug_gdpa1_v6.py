"""Audit cached GDPa1 CV and replay checkpoint features without changing scores.

USE_TF=0 OMP_NUM_THREADS=4 python scripts/downstream/debug_gdpa1_v6.py [--replay]
Writes diagnostics only under output/diagnostics/gdpa1_v6_20260927.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import PowerTransformer

from downstream.grammar.ab_features import embed_native_pairs, file_sha256, write_json
from downstream.grammar.ab_probes import prepare_task
from downstream.ophiuchus_eval.developability import ALL_COLS, FOLD_COL, LAMBDA_GRID

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/diagnostics/gdpa1_v6_20260927"
RUNS = {"V5_49000": "ab_v5_49000_llada_20260913",
        "V5_159000": "ab_v5_159000_full_20260917",
        "V6_ESMC_15000": "ab_v2_esmc_v6_15000_20260927",
        "V6_noESMC_110000": "ab_v2_noesmc_v6_110000_20260927"}


def rho(y, p):
    return float(spearmanr(y, p).statistic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df, pairs, _, _ = prepare_task("gdp_a1")
    report = {"diagnostic_only": True, "models": {}}
    for name, run in RUNS.items():
        folder = ROOT / "output/downstream_generation" / run / "gdp_a1"
        cache = torch.load(folder / "features.pt", map_location="cpu", weights_only=False)
        x = cache["features"].float().numpy()
        scores = json.loads((folder / "metrics.json").read_text())["properties"]
        singular = np.linalg.svd(x - x.mean(0), compute_uv=False)
        energy = singular ** 2
        row = {"shape": list(x.shape), "finite": bool(np.isfinite(x).all()),
               "mean_norm": float(np.linalg.norm(x, axis=1).mean()),
               "top_pc_variance_fraction": float(energy[0] / energy.sum()),
               "pairs_sha256": cache['manifest']['pairs_sha256'], "properties": {}}
        for col in ALL_COLS:
            valid = df[col].notna().to_numpy()
            y = df.loc[valid, col].to_numpy(dtype=float)
            folds = df.loc[valid, FOLD_COL].to_numpy(dtype=int)
            saved = np.load(folder / col / "legacy_oof.npz")
            assert np.array_equal(saved['row_index'], np.flatnonzero(valid))
            assert np.array_equal(saved['fold'], folds)
            assert np.array_equal(saved['y_true'], y)
            yt = PowerTransformer().fit_transform(y[:, None]).ravel()
            search = GridSearchCV(Ridge(), {'alpha': LAMBDA_GRID},
                                  scoring=make_scorer(rho), cv=PredefinedSplit(folds))
            search.fit(x[valid], yt)
            stored = scores[col]['legacy_reference']['cv_mean_spearman']
            row['properties'][col] = {"stored": stored, "recomputed": float(search.best_score_),
                "difference": float(search.best_score_ - stored), "alpha": search.best_params_['alpha'],
                "cv_by_alpha": dict(zip(map(str, LAMBDA_GRID), map(float, search.cv_results_['mean_test_score']))),
                "fold_scores": [float(search.cv_results_[f'split{k}_test_score'][search.best_index_]) for k in range(5)]}
        if args.replay and name != 'V5_49000':
            from examples.llada.load_fusion_checkpoint import load_fusion_for_eval
            ckpt = Path(cache['manifest']['checkpoint'])
            assert file_sha256(ckpt / 'model.safetensors') == cache['manifest']['checkpoint_sha256']
            bundle = load_fusion_for_eval(ckpt, device='cuda')
            fresh = embed_native_pairs(bundle.model, bundle.collator, pairs[:16], device='cuda', batch_size=16)
            previous = cache['features'][:16].float()
            row['replay'] = {"n": 16, "max_abs_difference": float((fresh-previous).abs().max()),
                             "mean_abs_difference": float((fresh-previous).abs().mean()),
                             "min_cosine": float(torch.nn.functional.cosine_similarity(fresh, previous).min())}
            del bundle, fresh
            gc.collect(); torch.cuda.empty_cache()
        report['models'][name] = row
        write_json(OUT / 'audit.json', report)
        print(name, 'replay', row.get('replay'), 'max_cv_difference',
              max(abs(p['difference']) for p in row['properties'].values()), flush=True)


if __name__ == '__main__':
    main()
