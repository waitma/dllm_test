"""Rescore a completed Kong retention run and compare its exact protocol.

Activate pllm/protenix_abtcr; run from repository root:
python -B scripts/downstream/compare_final_cdr.py --run ABS_COMPLETED_CDR_DIR
CPU-only; reuses authoritative source rows rather than deleted historic caches.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha,write
from scripts.downstream.evaluate_final_preservation_cdr import verify_existing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args();run=args.run.resolve()
    parent=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013'
    assert json.loads((run/'status.json').read_text())['status']=='success'
    verify_existing(run,parent)
    sources={}
    def read(path):
        sources[str(path)]=sha(path);return json.loads(path.read_text())
    verification=read(run/'verification.json')
    assert verification['status']=='passed' and verification['n_predictions']==9381
    sources[str(run/'kong/predictions.jsonl')]=sha(run/'kong/predictions.jsonl')
    new,old=read(run/'kong/identity.json'),read(parent/'cdr-kong/kong/identity.json')
    assert {k:v for k,v in new.items() if k!='model'}=={k:v for k,v in old.items() if k!='model'}
    assert new['model']['parent_model_sha256']==old['model']['model_sha256']
    folder=Path(new['model']['adaptation_directory'])
    side=read(folder/'checkpoint.json')
    assert sha(Path(side['parameter_file']))==side['sha256']==new['model']['adaptation_sha256']
    for path,digest in side['source_sha256'].items():assert sha(Path(path))==digest,path
    current,baseline=read(run/'kong/metrics.json'),read(parent/'cdr-kong/kong/metrics.json')
    results=[]
    assert set(current['modes'])==set(baseline['modes'])=={'cdrh1','cdrh2','cdrh3'}
    for mode,entry in current['modes'].items():
        original=baseline['modes'][mode]
        assert entry['n_sequences']==original['n_sequences']==3127
        assert len(entry['fold_means'])==len(original['fold_means'])==10
        delta=[a-b for a,b in zip(entry['fold_means'],original['fold_means'])]
        results.append(dict(mode=mode,original=original['airgen_aar_percent'],adapted=entry['airgen_aar_percent'],
            delta_pp=entry['airgen_aar_percent']-original['airgen_aar_percent'],
            fold_deltas_pp=delta,negative_folds=sum(x<0 for x in delta)))
    sources[str(Path(__file__))]=sha(Path(__file__))
    write(run/'comparison_verification.json',dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),
        model_identity=new['model'],protocol_identity_equal=True,source_predictions_rescored=True,
        results=results,source_sha256=sources))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
