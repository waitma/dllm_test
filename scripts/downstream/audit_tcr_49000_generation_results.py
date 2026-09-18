"""Re-score completed 49000 conditional generation from retained raw candidates.

Activate pllm then protenix_abtcr; no GPU is used. Example:
CUDA_VISIBLE_DEVICES='' python scripts/downstream/audit_tcr_49000_generation_results.py \
  --dataset benchmark14 --report /abs/new_report.json
Requires a successful manifest and all pMHCs; does not alter original results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from downstream.benchmark.tcr_generation_bench import scoring
from downstream.benchmark.tcr_generation_bench.result_io import prepare_metrics_for_json
from downstream.grammar.tcr_generation_v5 import BetaGenerationProtocol
from scripts.downstream import run_tcr_native_generation as generation
from scripts.downstream.run_tcr_local_short import now, sha


def same(actual,expected,path='root'):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),path
        for key in expected:
            same(actual[key],expected[key],path+'/'+str(key))
    elif isinstance(expected,(float,np.floating)):
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12,equal_nan=True,err_msg=path)
    else:
        assert actual==expected,path


def validate_raw(dataset, output):
    """Validate complete immutable saved designs before rescore or acceptance."""
    entries={k:v for k,v in json.loads(generation.EVAL.read_text()).items() if v['category']==dataset}
    raw=[json.loads(line) for line in (output/'designs.jsonl').read_text().splitlines()]
    assert len(raw)==len(entries)==(14 if dataset=='benchmark14' else 20)
    assert len({r['pmhc'] for r in raw})==len(raw) and {r['pmhc'] for r in raw}==set(entries)
    protocol=BetaGenerationProtocol()
    designs,greedy={},{}
    for record in raw:
        key=record['pmhc']; entry=entries[key]
        assert record['epitope']==entry['epitope'] and record['mhc_pseudo']==entry['mhc_pseudo']
        assert record['n_requested']==record['n_raw']==record['n_valid']==100
        assert len(record['sequences'])==len(record['candidates'])==100
        assert [c['sequence'] for c in record['candidates']]==record['sequences']
        lengths=protocol.lengths(42,key,100)
        for i,candidate in enumerate(record['candidates']):
            assert candidate['index']==i and candidate['core_length']==lengths[i]
            assert candidate['context_invariant'] and candidate['initial_targets_masked']
            # max_iter is a ceiling: generate_bioseq exits early once every
            # target is committed. Do not mistake its legal 30/31 steps for
            # a changed sampling budget. The manifest still pins max_iter=32.
            assert 1 <= candidate['steps'] <= 32 and len(candidate['core_token_ids'])==lengths[i]
        for candidate in [*record['candidates'],record['greedy_raw']]:
            sequence=candidate['sequence']
            assert candidate['valid'] is True and sequence[0]=='C' and sequence[-1]=='F'
            assert len(sequence)==candidate['core_length']+2
            assert set(sequence)<=set('ACDEFGHIKLMNPQRSTVWY')
        assert record['greedy']==record['greedy_raw']['sequence']
        assert record['v5_overlap']==generation.overlap_flags(key)
        designs[key],greedy[key]=record['sequences'],record['greedy']
    return entries,raw,designs,greedy


def audit(dataset):
    output = generation.output_for(dataset)
    manifest=json.loads((output/'run_manifest.json').read_text())
    result=json.loads((output/'metrics.json').read_text())
    identity=generation.identity()
    assert manifest['status']=='success' and manifest['paired_generation'] is False
    assert manifest['identity']==result['identity']==identity
    assert sha(manifest['gate'])==manifest['gate_sha256']
    gate=json.loads(Path(manifest['gate']).read_text())
    assert gate['status']=='passed' and gate['identity']==identity
    entries,raw,designs,greedy=validate_raw(dataset,output)
    expected=scoring.score_conditional(designs,entries,k=100,greedy_map=greedy)
    for key in list(expected['summary']):
        if key.startswith('bioseq_'):
            expected['summary']['historical_'+key]=expected['summary'].pop(key)
    for key,item in expected['per_pmhc'].items():
        item['historical_bioseq_unseen']=item.pop('bioseq_unseen')
        item['v5_overlap']=generation.overlap_flags(key)
    allowed=[k for k in entries if k!=scoring.RESERVED_SIMULATION_PMHC]
    for seen in (False,True):
        members=[k for k in allowed if generation.overlap_flags(k)['train_epitope_targets_seen']==seen]
        if members:
            expected['summary']['v5_train_epitope_'+('seen' if seen else 'unseen')]=scoring._macro(expected['per_pmhc'],members)
    if dataset=='benchmark14':
        expected['summary']['paper_sparse13']=scoring._macro(expected['per_pmhc'],allowed)
    expected.update(identity=identity,dataset=dataset,candidate_count=100,greedy_count=1,
        input_condition='epitope_and_provided_34mer_mhc_no_reference_tcr',
        interpretation='sequence_similarity_not_experimental_binding_validation')
    conditioning=scoring.load_conditioning_by_epitope()
    missing={k for k,e in entries.items() if not conditioning.get(e['epitope'])}
    expected=prepare_metrics_for_json(expected,missing)
    if 'recovery' in manifest:
        provenance=manifest['recovery']
        generation.verify_recovery_identity(provenance['generation_identity'],identity)
        source=Path(provenance['source'])
        original=json.loads((source/'run_manifest.json').read_text())
        assert original['status']=='failed' and original['identity']==provenance['generation_identity']
        for path,digest in provenance['source_sha256'].items():
            assert sha(path)==digest,'original recovery evidence changed'
        assert sha(output/'designs.jsonl')==sha(source/'designs.jsonl')
        expected['recovery']=provenance
    same(result,expected)
    assert result['summary'][dataset]['n_pmhc_scored']==len(entries)
    return {'status':'passed','dataset':dataset,'output':str(output),'pmhc_complete':len(entries),
        'raw_candidates':len(entries)*100,'independent_greedy':len(entries),
        'actual_denoising_steps':sorted({c['steps'] for r in raw for c in r['candidates']}),
        'artifact_sha256':{str(p):sha(p) for p in output.iterdir() if p.suffix in ('.json','.jsonl')},
        'checks':['complete target membership, no duplicate/missing targets',
                  '100 raw valid candidates and independent greedy per target',
                  'training-profile sampled lengths and fixed-context runtime assertions',
                  'entire scoring recomputed from saved sequences against original references',
                  'per-pMHC and all aggregates, actual denominators, v5 overlap strata']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['benchmark14','held20'],required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():
        raise FileExistsError('choose a new audit report path')
    report={'status':'running','global_step':49000,'start_utc':now(),'auditor_sha256':sha(__file__)}
    try:
        report['result']=audit(args.dataset)
        report['status']='passed'
        print('PASS',args.dataset,flush=True)
    except Exception as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        report['end_utc']=now()
        report['limits']=['no new model generation','beta-only, not paired generation',
            'held20 is not v5-unseen','sequence scores do not validate experimental binding']
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open('x') as handle:
            json.dump(report,handle,indent=2)


if __name__=='__main__':
    main()
