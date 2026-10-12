"""Recompute saved T1 AUROCs: python -B SCRIPT --run ABS. No model or training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    rows=[]
    for task in ['t1-cdr3b','t1-others_cdr3b','t1-cdr3ab','t1-others_longab']:
        for path in sorted((args.run/task/'protocol_outputs').rglob('predictions.csv')):
            df=pd.read_csv(path)
            assert set(df.label.unique())=={0,1}
            assert np.isfinite(df.score).all() and df.score.between(0,1).all()
            metric=json.loads(path.with_name('metrics.json').read_text())
            metric=metric.get('metrics',metric)
            auroc=float(roc_auc_score(df.label,df.score))
            reported=metric.get('auroc',metric.get('AUROC'))
            if reported is None:raise ValueError(str(metric.keys()))
            assert abs(auroc-reported)<1e-6,(path,auroc,reported)
            rows.append(dict(task=task,fold=path.parent.parent.name,split=path.parent.name,
                n=len(df),positive_rate=float(df.label.mean()),score_std=float(df.score.std()),
                auroc=auroc,reported_auroc=reported,path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(rows)==60
    report=dict(status='passed',checks='60 fold/split predictions: labels binary, scores finite, independent AUROC agreement',
        interpretation='No score aggregation error detected; this does not establish out-of-distribution model quality',rows=rows)
    (args.run/'binding_score_audit_20261009.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report['checks'])


if __name__=='__main__':main()
