"""Check layer-probe precision and training-fold-only feature scaling controls.

Activate pllm then protenix_abtcr; from repository root run:
python -B scripts/downstream/verify_gdpa1_layers.py --audit ABS_LAYER_AUDIT_DIRECTORY
Scaling is a separate diagnostic, not the AirGen reference protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PowerTransformer,StandardScaler
    from downstream.grammar.ab_probes import prepare_task
    from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL,LAMBDA_GRID
    df,_,_,_=prepare_task('gdp_a1')
    features=np.load(a.audit/'features.npz')
    original=json.loads((a.audit/'metrics.json').read_text())
    plan=dict(controls=['float64_reference','float64_train_fold_standard_scaler'],
              features_sha256=hashlib.sha256((a.audit/'features.npz').read_bytes()).hexdigest(),
              readouts=features.files,properties=list(ALL_COLS),selection='none; report all combinations',
              scaling='StandardScaler fit inside each CV training fold; no test feature statistics',
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest=a.audit/'numerical_controls.json'
    assert not dest.exists()
    (a.audit/'numerical_controls_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    results=[]
    for name in features.files:
        for col in ALL_COLS:
            ok=df[col].notna().to_numpy()
            x=features[name][ok].astype('float64')
            y=PowerTransformer().fit_transform(df.loc[ok,col].to_numpy()[:,None]).ravel()
            cv=PredefinedSplit(df.loc[ok,FOLD_COL].to_numpy())
            row=dict(readout=name,property=col,original=original[name]['properties'][col]['legacy_reference'])
            for mode in plan['controls']:
                scaled=mode.endswith('standard_scaler')
                model=make_pipeline(StandardScaler(),Ridge()) if scaled else Ridge()
                search=GridSearchCV(model,{'ridge__alpha' if scaled else 'alpha':LAMBDA_GRID},cv=cv,
                                    scoring=make_scorer(lambda yy,pp:float(spearmanr(yy,pp).statistic)))
                search.fit(x,y)
                row[mode]=dict(score=float(search.best_score_),alpha=float(next(iter(search.best_params_.values()))))
            results.append(row)
            print(name,col,row['float64_reference'],row['float64_train_fold_standard_scaler'],flush=True)
    dest.write_text(json.dumps(dict(plan=plan,results=results,status='complete'),indent=2)+'\n')


if __name__=='__main__':
    main()
