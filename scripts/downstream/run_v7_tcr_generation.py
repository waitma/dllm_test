"""Collect fixed two-epoch V7 TCR predictions after a matching device audit.

Activate conda pllm; use the worker-compatible Python in a clean process:
python -I -B /project_c20250601/251105016/project/dllm_test/scripts/downstream/run_v7_tcr_generation.py --device cuda:0 --preflight /absolute/preflight.json --output-root /absolute/new_directory
All original 34 targets receive 100 stochastic candidates and one independent
argmax control. No scoring, annotation, retries, training, or job submission.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import runpy
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
from downstream.v7_native import load_native_checkpoint, require
from downstream.v7_tcr_generation import beta_input, beta_decode, beta_result, PROTOCOL as PROTOCOL_ID
from downstream.benchmark.common.reporting import object_sha256

PREFLIGHT_SOURCE = ROOT/'scripts/downstream/preflight_v7_tcr_generation.py'
API = runpy.run_path(str(PREFLIGHT_SOURCE), run_name='_v7_native_tcr_audit_helpers')
read, reference, write = API['read'], API['reference'], API['write']
PROTOCOL = dict(name=PROTOCOL_ID, stochastic_per_target=100, independent_argmax_per_target=1,
    native_capacity=64, max_iter=32, temperature=1.0, cfg_scale=0.0,
    sampling_strategy='gumbel_argmax', decoding_strategy='confidence-deterministic-linear',
    candidate_batch_size=1, seed_assignment='42 + original_target_index * 101 + candidate_index',
    argmax_candidate_index=100, reference_length_access=False, repair_candidates=False,
    conditions='epitope + provided34_MHC_pseudo + desired_binding; alpha_missing',
    targets='complete_native_beta_junction_canvas_including_anchors_and_terminal')


def seed_for(target_index, candidate_index):
    return 42+target_index*101+candidate_index


def validate_rows(rows, identity, entries):
    """Check every assigned raw candidate, including invalid/empty/control cases."""
    from downstream.benchmark.tcr_generation_bench.scoring import is_valid_cdr3b
    source = {key:(i,e) for i,(key,e) in enumerate(entries.items())}
    expected = {(key,j) for key in entries for j in range(101)}
    require(len(rows)==len(expected), 'Missing or extra TCR candidates')
    require([(r['target_key'],r['candidate_index']) for r in rows]
            == [(key,j) for key in entries for j in range(101)], 'Raw candidate order changed')
    seen=set(); tokens=identity['tokens']; terminal=identity['terminal_id']
    allowed=set(identity['allowed_target_ids']); digest=object_sha256(identity)
    for row in rows:
        key=row['target_key']; j=row['candidate_index']; slot=(key,j)
        require(type(j) is int and slot in expected and slot not in seen, 'Duplicate or unexpected TCR candidate')
        seen.add(slot); i,entry=source[key]
        strategy='argmax' if j==100 else 'gumbel_argmax'
        require(row['target_index']==i and row['category']==entry['category']
                and row['identity_sha256']==digest and row['seed']==seed_for(i,j)
                and row['candidate_kind']==('independent_argmax' if j==100 else 'stochastic'),
                'TCR candidate identity/seed differs')
        ids=row['raw_beta_token_ids']
        require(len(ids)==64 and all(type(t) is int and t in allowed for t in ids),
                'Wrong native raw TCR tokens')
        terminated=terminal in ids; end=ids.index(terminal) if terminated else len(ids)
        sequence=''.join(tokens[t] for t in ids[:end])
        valid=bool(terminated and is_valid_cdr3b(sequence))
        failure=('missing_terminal' if not terminated else 'empty_candidate' if not sequence
                 else 'invalid_junction' if not valid else None)
        require(row['sequence']==sequence and row['native_capacity']==64
                and row['generated_length']==end and row['terminated_by_protd'] is terminated
                and row['syntactically_valid'] is valid and row['terminal_count']==ids.count(terminal)
                and row['repaired'] is False and row['failure_reason']==failure,
                'TCR candidate changed, repaired, or misclassified')
        audit=row['audit']; trace=audit['trace']
        require(audit['protocol']==PROTOCOL_ID and audit['seed']==row['seed']
                and audit['max_iter']==32 and audit['sampling_strategy']==strategy
                and audit['decoding_strategy']=='confidence-deterministic-linear'
                and audit['temperature']==1.0 and audit['cfg_scale']==0.0
                and audit['labels_provided'] is False and audit['reference_residues_or_lengths_provided'] is False
                and 1<=audit['forward_calls']==len(trace)<=32, 'Wrong TCR decoding/input trace')
        pending=64
        for iteration,item in enumerate(trace,1):
            require(item['iteration']==iteration and item['pending_before']==pending
                    and 0<=item['pending_after']<=pending
                    and re.fullmatch('[0-9a-f]{64}',item['actual_forward_input_sha256']),
                    'Incomplete TCR forward trace')
            pending=item['pending_after']
        require(pending==0 and audit['original_input_sha256']==trace[0]['actual_forward_input_sha256'],
                'Unfinished or changed TCR generation input')
    main=[r for r in rows if r['candidate_index']<100]
    return dict(status='all_assigned_raw_tcr_candidates_validated',targets=len(entries),
        stochastic_candidates=len(main),independent_argmax_controls=len(entries),
        empty_stochastic_candidates=sum(not r['sequence'] for r in main),
        terminated_stochastic_candidates=sum(r['terminated_by_protd'] for r in main),
        syntactically_valid_stochastic_candidates=sum(r['syntactically_valid'] for r in main),
        candidates_dropped=0,scoring_performed=False)


def seal_predictions(output, identity, entries):
    require(read(output/'identity.json')==identity, 'Saved TCR identity changed')
    rows=[json.loads(s) for s in (output/'predictions.jsonl').read_text().splitlines()]
    validation=validate_rows(rows,identity,entries)
    indexed={(r['target_key'],r['candidate_index']):r for r in rows}
    designs=[]
    with (output/'designs.jsonl').open('x') as stream:
        for i,(key,entry) in enumerate(entries.items()):
            main=[indexed[key,j] for j in range(100)]; greedy=indexed[key,100]
            row=dict(pmhc=key,target_index=i,category=entry['category'],epitope=entry['epitope'],
                mhc_pseudo=entry['mhc_pseudo'],sequences=[r['sequence'] for r in main],
                greedy=greedy['sequence'],n_requested=100,n_raw=100,
                n_syntactically_valid=sum(r['syntactically_valid'] for r in main),
                n_terminated=sum(r['terminated_by_protd'] for r in main),
                protocol=PROTOCOL_ID,identity_sha256=object_sha256(identity))
            designs.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
    require([json.loads(s) for s in (output/'designs.jsonl').read_text().splitlines()]==designs,
            'Design export changed candidate order or empty values')
    result=dict(validation,status='full_tcr_predictions_complete_scoring_pending',
        at_utc=datetime.now(timezone.utc).isoformat(),identity_sha256=object_sha256(identity),
        predictions=reference(output/'predictions.jsonl'),designs=reference(output/'designs.jsonl'))
    write(output/'PREDICTIONS_COMPLETE.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--device',choices=('cpu','cuda:0'),required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();output=args.output_root
    require(output.is_absolute() and output==output.resolve() and not output.exists()
            and output.is_relative_to(ROOT/'output/downstream_generation')
            and not any(p.is_symlink() for p in (output,*output.parents)), 'New canonical output required')
    API['check_stop'](output);entries,binding=API['bind_inputs']()
    code=API['code_identity']();pref=read(args.preflight);pref_ref=reference(args.preflight)
    require(pref['status']=='actual_native_tcr_input_audit_passed' and pref['binding']==binding
            and pref['code']==code and pref['protocol']==PROTOCOL_ID
            and pref['device_type']==args.device.split(':')[0]
            and pref['checkpoint']==reference(API['CHECKPOINT']/'formal_checkpoint.json'),
            'Matching actual checkpoint/device/source/input audit is required')
    require(len(entries)==34 and sum(e['category']=='benchmark14' for e in entries.values())==14
            and sum(e['category']=='held20' for e in entries.values())==20, 'Full fixed TCR benchmark required')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),
        binding=binding,preflight=pref_ref,entrypoint=reference(Path(__file__).resolve()),protocol=PROTOCOL))
    try:
        model,loaded=load_native_checkpoint(API['CHECKPOINT'],args.device)
        require(model.config.fragment_capacity==64,'Native junction capacity differs')
        identity=dict(binding=binding,code=code,entrypoint=reference(Path(__file__).resolve()),
            checkpoint=reference(API['CHECKPOINT']/'formal_checkpoint.json'),native_model_load=loaded,
            device_type=args.device.split(':')[0],protocol=PROTOCOL,preflight=pref_ref,
            tokens=[model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)],
            terminal_id=model.terminal_id,allowed_target_ids=model.allowed_target.nonzero().flatten().tolist())
        write(output/'identity.json',identity);count=0
        with (output/'predictions.jsonl').open('x') as stream:
            for i,(key,entry) in enumerate(entries.items()):
                for j in range(101):
                    API['check_stop'](output);seed=seed_for(i,j)
                    features,target=beta_input(model,entry['epitope'],entry['mhc_pseudo'])
                    tokens,audit=beta_decode(model,features,target,seed=seed,
                        sampling_strategy='argmax' if j==100 else 'gumbel_argmax')
                    row=dict(target_key=key,target_index=i,category=entry['category'],candidate_index=j,
                        candidate_kind='independent_argmax' if j==100 else 'stochastic',seed=seed,
                        identity_sha256=object_sha256(identity),**beta_result(model,tokens,target),audit=audit)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();count+=1
                    if j%10==9 or j==100:
                        temporary=output/'progress.tmp'
                        write(temporary,dict(target_index=i,candidate_index=j,raw_candidates=count,
                            targets_completed=i+(j==100),targets=34,expected_raw_candidates=3434))
                        temporary.replace(output/'progress.json')
        require(API['code_identity']()==code
                and reference(Path(__file__).resolve())==identity['entrypoint'], 'Sources changed during generation')
        result=seal_predictions(output,identity,entries)
        print(json.dumps(dict(status=result['status'],output=str(output))),flush=True)
    except BaseException as error:
        write(output/'FAILURE.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),error=repr(error)))
        raise


if __name__=='__main__':
    main()
