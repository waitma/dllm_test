"""Check TCR budget, order, invalid-candidate retention, and sealing with artificial rows.

Activate conda pllm; run worker-compatible Python:
python -I -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_tcr_artifacts.py
Uses no model, real benchmark, GPU, annotation, or checkpoints. Temporary
artifact fixtures are automatically removed; artificial traces are not scores.
"""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import tempfile

ROOT=Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
API=runpy.run_path(str(ROOT/'scripts/downstream/run_v7_tcr_generation.py'),run_name='_artifact_fixture')


def main():
    entries={'artificial_a':dict(category='benchmark14',epitope='A',mhc_pseudo='A'*34),
             'artificial_b':dict(category='held20',epitope='G',mhc_pseudo='G'*34)}
    identity=dict(tokens=['A','C','S','F','W','<protd>'],terminal_id=5,allowed_target_ids=list(range(6)))
    rows=[]
    for i,(key,entry) in enumerate(entries.items()):
        for j in range(101):
            # Explicit invalid and early-end observations remain members of the budget.
            if j==0:
                ids=[5]+[0]*63;sequence='';valid=False;end=0;failure='empty_candidate'
            elif j==1:
                ids=[0]*64;sequence='A'*64;valid=False;end=64;failure='missing_terminal'
            else:
                ids=[1,0,2,2,3,5]+[0]*58;sequence='CASSF';valid=True;end=5;failure=None
            seed=42+101*i+j
            trace=[dict(iteration=t+1,pending_before=64-2*t,pending_after=62-2*t,
                        actual_forward_input_sha256='0'*64) for t in range(32)]
            rows.append(dict(target_key=key,target_index=i,category=entry['category'],candidate_index=j,
                candidate_kind='independent_argmax' if j==100 else 'stochastic',seed=seed,
                identity_sha256=API['object_sha256'](identity),sequence=sequence,raw_beta_token_ids=ids,
                native_capacity=64,generated_length=end,terminated_by_protd=j!=1,
                syntactically_valid=valid,terminal_count=int(j!=1),repaired=False,failure_reason=failure,
                audit=dict(protocol=API['PROTOCOL_ID'],seed=seed,max_iter=32,
                    sampling_strategy='argmax' if j==100 else 'gumbel_argmax',
                    decoding_strategy='confidence-deterministic-linear',temperature=1.,cfg_scale=0.,
                    labels_provided=False,reference_residues_or_lengths_provided=False,
                    forward_calls=32,trace=trace,original_input_sha256='0'*64)))
    assert API['seed_for'](0,0)==42 and API['seed_for'](0,100)==142 and API['seed_for'](33,100)==3475
    result=API['validate_rows'](rows,identity,entries)
    assert result['stochastic_candidates']==200 and result['independent_argmax_controls']==2
    assert result['empty_stochastic_candidates']==2 and result['syntactically_valid_stochastic_candidates']==196
    mutations=[lambda x:x.pop(),lambda x:x.append(deepcopy(x[-1])),lambda x:x.reverse(),
        lambda x:x[0].update(sequence='CASSF'),lambda x:x[1].update(terminated_by_protd=True),
        lambda x:x[2].update(repaired=True),lambda x:x[100].update(candidate_kind='stochastic'),
        lambda x:x[0].update(seed=99),lambda x:x[0]['audit']['trace'][-1].update(pending_after=1),
        lambda x:x[2]['raw_beta_token_ids'].__setitem__(0,99)]
    for mutate in mutations:
        bad=deepcopy(rows);mutate(bad)
        try:API['validate_rows'](bad,identity,entries)
        except ValueError:pass
        else:raise AssertionError('Corrupt TCR artifact accepted')
    with tempfile.TemporaryDirectory(prefix='v7_tcr_artifacts_') as directory:
        output=Path(directory);API['write'](output/'identity.json',identity)
        (output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        sealed=API['seal_predictions'](output,identity,entries)
        exported=[json.loads(s) for s in (output/'designs.jsonl').read_text().splitlines()]
        assert len(exported)==2 and all(len(r['sequences'])==100 for r in exported)
        assert all(r['sequences'][0]=='' and r['greedy']=='CASSF' for r in exported)
        assert sealed['status']=='full_tcr_predictions_complete_scoring_pending'
        assert not (output/'metrics.json').exists()
        try:API['seal_predictions'](output,identity,entries)
        except FileExistsError:pass
        else:raise AssertionError('Sealed TCR result overwritten')
    assert not output.exists()
    print(json.dumps(dict(status='passed',artificial_candidates=202,retained_empty_candidates=2,
        rejected_corruptions=len(mutations),separate_argmax_budget=True,exact_design_export=True,
        overwrite_rejected=True,temporary_files_removed=True,model_loaded=False,
        real_benchmark_read=False,gpu_used=False,annotation_called=False,checkpoints_created=0)),flush=True)


if __name__=='__main__':
    main()
