"""Check full Kong generation retention with the existing four-step protocol.

Activate pllm/protenix_abtcr, then from repository root:
python -B scripts/downstream/evaluate_final_preservation_cdr.py \
  --checkpoint-dir ABS_PILOT_ARM --out ABS_NEW_DIRECTORY
Reuses the pinned AirGen evaluator, sampler, fixed-chain adapter and all 10 folds.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha


def verify_existing(out,parent):
    """Use authoritative test rows when historic prediction caches were cleaned."""
    import numpy as np
    assert json.loads((out/'kong/status.json').read_text())['status']=='success'
    audit=json.loads((out/'protocol_audit.json').read_text())
    baseline_audit=json.loads((parent/'cdr-kong/protocol_audit.json').read_text())
    assert audit['datasets']['kong']==baseline_audit['datasets']['kong']
    for path,digest in audit['datasets']['kong']['files'].items():assert sha(path)==digest,path
    for path,digest in audit['code_sha256'].items():assert sha(path)==digest,path
    path=parent/'airgen_evaluate.py'
    spec=importlib.util.spec_from_file_location('verify_preservation_airgen',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source_rows,_=module.records('kong')
    expected=[(r['mode'],r['fold'],r['row'],r['key']) for r in source_rows]
    vocabulary=json.loads((parent/'checkpoint/vocabulary.json').read_text())
    token_id={token:i for i,token in enumerate(vocabulary)}
    metrics=json.loads((out/'kong/metrics.json').read_text())
    baseline=json.loads((parent/'cdr-kong/kong/metrics.json').read_text())
    rows=[json.loads(line) for line in (out/'kong/predictions.jsonl').read_text().splitlines()]
    assert [(r['mode'],r['fold'],r['row'],r['key']) for r in rows]==expected
    by_fold=defaultdict(list)
    for row,source in zip(rows,source_rows):
        gold=[token_id[source['gold_h'][j]] for j in source['mask_h']]
        assert row['reference_ids']==gold
        assert row['masked_positions']==[j+1 for j in source['mask_h']]
        assert len(row['reference_ids'])==len(row['prediction_ids'])>0
        correct=sum(x==y for x,y in zip(row['reference_ids'],row['prediction_ids']))
        aar=float(np.float32(correct)/np.float32(len(row['reference_ids'])))
        assert abs(aar-row['aar'])<1e-7
        by_fold[(row['mode'],row['fold'])].append(aar)
    for mode,result in metrics['modes'].items():
        values=[float(np.mean(by_fold[(mode,i)])*100) for i in range(10)]
        np.testing.assert_allclose(values,result['fold_means'],rtol=0,atol=1e-6)
        assert abs(np.mean(values)-result['airgen_aar_percent'])<1e-6
        assert sum(len(by_fold[(mode,i)]) for i in range(10))==result['n_sequences']==baseline['modes'][mode]['n_sequences']
    write(out/'verification.json',dict(status='passed',n_predictions=len(rows),
          original_test_row_order_and_references_verified=True,baseline_source_manifest_identical=True,
          saved_predictions_rescored=True,historic_prediction_cache_required=False,
          historic_predictions_present=(parent/'cdr-kong/kong/predictions.jsonl').exists(),
          verifier_sha256=sha(Path(__file__))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true',help='Rescore existing complete predictions; no model loading or sampling')
    args=parser.parse_args();out=args.out.resolve()
    parent=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013'
    if args.verify_only:
        previous=json.loads((out/'status.json').read_text())
        verify_existing(out,parent)
        write(out/'verification_recovery.json',dict(previous_status=previous,
              action='CPU-only original-source validation and saved-prediction rescoring; no sampling rerun',
              completed_utc=datetime.now(timezone.utc).isoformat()))
        write(out/'status.json',dict(status='success',stage='verified_existing_predictions',
              original_process_failed_after_sampling=True,original_status='verification_recovery.json',
              finished_utc=datetime.now(timezone.utc).isoformat()))
        return
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',state)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        from downstream.final_layer_preservation_eval import load
        from downstream.fixed_chain_eval import airgen_inputs
        model,identity=load(args.checkpoint_dir.resolve())
        parent=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013'
        path=parent/'airgen_evaluate.py'
        spec=importlib.util.spec_from_file_location('preservation_airgen_evaluator',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.OUT=out;module.CKPT=args.checkpoint_dir.resolve();module.report=lambda:None
        module.inputs=airgen_inputs
        from downstream import v5_native_eval
        v5_native_eval.load=lambda *a,**kw:(model,identity)
        write(out/'execution.json',dict(model=identity,baseline=str(parent/'cdr-kong/kong'),
              evaluator_sha256=sha(path),adapter_sha256=sha(ROOT/'downstream/fixed_chain_eval.py'),
              bridge_sha256=sha(ROOT/'downstream/final_layer_preservation_eval.py'),
              protocol='same four-step argmax, batch64 BF16, official position fields, ten-fold means'))
        sys.argv=[str(path),'--audit'];module.main()
        sys.argv=[str(path),'--dataset','kong'];module.main()
        verify_existing(out,parent)
        state.update(status='success',finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:write(out/'status.json',state)


if __name__=='__main__':main()
