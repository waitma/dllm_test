"""Check new pairing artifact integrity using disposable artificial observations.

Activate conda pllm and run python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_pairing_artifacts.py
No model, real sequence, certificate, GPU, job, or checkpoint is created/loaded.
All artificial prediction files are removed when TemporaryDirectory exits.
"""
import copy
import csv
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')


def main():
    api = runpy.run_path(str(ROOT/'scripts/downstream/run_v7_pairing_eval.py'), run_name='_test_pairing_artifact_contract')
    identity = dict(synthetic_fixture=True, tokens=['A', 'C', '<protd>'], terminal_id=2, allowed_target_ids=[0,1,2])
    records = [dict(row_index=0, heavy='ACAC', light='AAAA'), dict(row_index=1, heavy='CACA', light='CCCC')]
    rows = []
    for record in records:
        for variant in range(8):
            terminal = variant == 0
            seed = 42+record['row_index']*8+variant
            audit = dict(labels_provided=False,max_iter=16,seed=seed,sampling_strategy='gumbel_argmax',
                decoding_strategy='confidence-deterministic-linear',temperature=1.0,cfg_scale=0.0,
                forward_calls=1,original_input_sha256='a'*64,
                trace=[dict(iteration=1,pending_before=4,pending_after=0,actual_forward_input_sha256='a'*64)])
            rows.append(dict(row_index=record['row_index'],variant_idx=variant,seed=seed,
                h_sequence=record['heavy'],raw_l_sequence=record['light'],sequence='' if terminal else 'ACAC',
                raw_light_token_ids=[2,2,2,2] if terminal else [0,1,0,1],generated_light_length=0 if terminal else 4,
                known_light_length=4,early_terminal_count=4 if terminal else 0,repaired=False,
                terminal_status='predicted_early_terminal' if terminal else 'fixed_known_length',
                identity_sha256=api['object_sha256'](identity),audit=audit,hidden_reference_input_invariance=True))
    valid = api['validate_rows'](rows, identity, records)
    assert valid['candidates']==16 and valid['empty_candidates']==2 and valid['candidates_dropped']==0
    corruptions = []
    corruptions.append(rows[:-1])
    corruptions.append(rows[:-1]+[rows[0]])
    for key,value in [('seed',999),('raw_l_sequence','GGGG'),('sequence','A'),('raw_light_token_ids',[99]*4),('repaired',True)]:
        changed=copy.deepcopy(rows); changed[0][key]=value; corruptions.append(changed)
    changed=copy.deepcopy(rows); changed[0]['audit']['labels_provided']=True; corruptions.append(changed)
    changed=copy.deepcopy(rows); changed[0]['audit']['trace'][0]['pending_after']=1; corruptions.append(changed)
    for changed in corruptions:
        try:
            api['validate_rows'](changed,identity,records)
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupted pairing artifacts were accepted')
    with TemporaryDirectory(prefix='v7_pairing_artifacts_') as directory:
        output=Path(directory).resolve()
        api['write'](output/'manifest.json',dict(status='running',identity=identity,identity_sha256=api['object_sha256'](identity)))
        (output/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        completed=api['seal_predictions'](output,identity,records,None)
        assert completed['status']=='full_predictions_complete_scoring_pending' and completed['scoring_performed'] is False
        with (output/'holdout500_n8.csv').open(newline='') as stream:
            parsed=list(csv.DictReader(stream))
        assert len(parsed)==16 and sum(row['gen_l_sequence']=='' for row in parsed)==2
        assert not (output/'COMPLETE.json').exists() and not (output/'metrics.json').exists()
        try:
            api['seal_predictions'](output,identity,records,None)
        except ValueError:
            pass
        else:
            raise AssertionError('Sealed artifact was overwritten')
    assert not Path(directory).exists()
    print(json.dumps(dict(status='passed',artificial_candidates=16,empty_candidates_retained=2,
        rejected_corruptions=len(corruptions),csv_roundtrip_exact=True,sealed_artifact_replay_rejected=True,
        predictions_not_misrepresented_as_scores=True,temporary_artifacts_removed=True,
        model_loaded=False,real_data_read=False,certificate_created=False,gpu_used=False)),flush=True)


if __name__=='__main__':
    main()
