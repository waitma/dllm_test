"""Run the fixed V7 light-chain input audit or full prediction collection.

Activate conda pllm; use the worker-compatible Python in a clean process:
python -B /project_c20250601/251105016/project/dllm_test/scripts/downstream/run_v7_pairing_eval.py --action preflight --device cpu --output-root NEW_ABSOLUTE_DIRECTORY
Use --action generate for all 500 heavies x 8 candidates, with a matching
--preflight /absolute/path/preflight.json produced on the same device type.
The fixed checkpoint is 50082, p0/reference length/iter16/gumbel/CFG0/seed42.
Prediction completion is separate from scoring; no annotation, resource
submission, automatic retry, checkpoint copy, or training is performed here.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import runpy
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
EVAL_SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_SOURCE))
from downstream.v7_native import load_native_checkpoint, sha, require, input_fingerprint
from downstream.v7_pairing import pairing_input, pairing_decode, light_result
from downstream.benchmark.common.reporting import object_sha256

EPOCH = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006'
RECOVERY = EPOCH/'recovery_v1'
GENERATION = RECOVERY/'generation_pool_v1'
CHECKPOINT = EPOCH/'run/checkpoint-50082'
PREPARATION = ROOT/'output/v7_pairing_adapter_20261006'
INVENTORY = PREPARATION/'benchmark_inventory_v2.json'
INVENTORY_SHA = '7617fe50105ced20daceadcd7003062606690c8b74b32c7f024594d6cb0cef9e'
GATE = ROOT/'output/v7_local_ablation_20261005/post_generation_pool_v2/source_v1/v7_post_generation_acceptance.py'
GATE_SHA = '9a05fb9cee32aa6e60790fb2b1afb07eb3b63aa8dafe127e05d14cb070338f0c'
PROTOCOL = dict(name='v7_native_a4_pairing_known_length_v1', prefix_residues=0,
    max_iter=16, candidates_per_heavy=8, base_seed=42, temperature=1.0, cfg_scale=0.0,
    sampling_strategy='gumbel_argmax', decoding_strategy='confidence-deterministic-linear',
    seed_assignment='42 + original_row_index * 8 + variant_idx', candidate_batch_size=1,
    length_condition='reference', vocabulary='native_A4_residues_plus_protd',
    terminal_handling='preserve raw tokens; interpret first protd; never repair or drop a candidate')


def read(path):
    return json.loads(path.read_text())


def ref(path):
    require(path.is_absolute() and path == path.resolve()
            and not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file(),
            'Canonical existing evidence required: '+str(path))
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def stopped(output):
    for directory in (output, PREPARATION, ROOT/'output/immune_research_pool_c20250601_20261005'):
        require(not any((directory/name).exists() for name in ('STOP', 'STOP_NOW')),
                'Explicit stop prevents continuing: '+str(directory))


def bind_inputs():
    """Retain the existing real completion/readiness gates; never issue certificates."""
    require(sha(GATE) == GATE_SHA and sha(INVENTORY) == INVENTORY_SHA, 'Fixed prerequisites changed')
    api = runpy.run_path(str(GATE), run_name='_pairing_existing_completion_gate')
    fields = dict(formal_completion=ref(RECOVERY/'COMPLETE.json'),
        formal_stage=str(RECOVERY/'launches_v1/segment_50082'),
        generation_completion=ref(GENERATION/'COMPLETE.json'),
        generation_release=ref(GENERATION/'release.json'),
        generation_package=ref(GENERATION/'package_v1.json'))
    acceptance = api['check'](fields)
    inventory = read(INVENTORY)
    require(inventory['status'] == 'existing_pairing_benchmark_inventory_only'
            and inventory['rows'] == 500 and inventory['expected_candidate_rows'] == 4000,
            'Full fixed paired benchmark required')
    for evidence in inventory['formal_metadata'].values():
        require(ref(Path(evidence['path'])) == evidence, 'Original data audit metadata changed')
    benchmark = inventory['benchmark']; path = Path(benchmark['path'])
    require(ref(path) == benchmark, 'Pairing benchmark changed')
    with path.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    require(len(records) == 500, 'Full 500-row benchmark required; do not filter')
    records = [dict(row_index=i, heavy=r['h_sequence'].replace('-', ''),
                    light=r['l_sequence'].replace('-', '')) for i, r in enumerate(records)]
    require(all(0 < len(r[k]) < 256 and set(r[k]) <= set('ACDEFGHIKLMNPQRSTVWY')
                for r in records for k in ('heavy', 'light')), 'Benchmark normalization/capacity differs')
    return records, dict(inventory=ref(INVENTORY), benchmark=benchmark,
        predecessor_evidence=fields, actual_completion=acceptance)


def code_identity():
    return [ref(EVAL_SOURCE/name) for name in ('scripts/downstream/run_v7_pairing_eval.py',
        'downstream/v7_pairing.py', 'downstream/v7_native.py', 'downstream/benchmark/common/reporting.py')]


def predict(model, record, variant):
    seed = 42+record['row_index']*8+variant
    features, target, light = pairing_input(model, record['heavy'], len(record['light']))
    # Alter the hidden suffix before rendering: only length may be visible.
    alternate_reference = 'G'*len(record['light'])
    other, other_target, _ = pairing_input(model, record['heavy'], len(alternate_reference))
    require(input_fingerprint(other) == input_fingerprint(features)
            and target.equal(other_target), 'Hidden reference changed model features')
    output, audit = pairing_decode(model, features, target, max_iter=16, seed=seed)
    return dict(row_index=record['row_index'], variant_idx=variant, seed=seed,
        h_sequence=record['heavy'], raw_l_sequence=record['light'], **light_result(model, output, light),
        audit=audit, hidden_reference_input_invariance=True)


def validate_rows(rows, identity, records):
    """Validate all candidate identities and raw-token interpretation without inference."""
    expected = {(r['row_index'], variant) for r in records for variant in range(8)}
    require(len(rows) == len(expected), 'Missing or extra candidates')
    source = {r['row_index']: r for r in records}
    vocab = identity['tokens']; terminal = identity['terminal_id']; allowed = set(identity['allowed_target_ids'])
    seen = set()
    for row in rows:
        key = row['row_index'], row['variant_idx']
        require(key in expected and key not in seen, 'Duplicate or unexpected candidate identity')
        seen.add(key); original = source[key[0]]
        require(row['identity_sha256'] == object_sha256(identity)
                and row['h_sequence'] == original['heavy'] and row['raw_l_sequence'] == original['light']
                and row['seed'] == 42+8*key[0]+key[1], 'Candidate inputs/seed/provenance differ')
        tokens = row['raw_light_token_ids']
        require(len(tokens) == len(original['light']) and all(type(t) is int and t in allowed for t in tokens),
                'Raw candidate length or vocabulary differs')
        end = tokens.index(terminal) if terminal in tokens else len(tokens)
        require(row['sequence'] == ''.join(vocab[t] for t in tokens[:end])
                and row['generated_light_length'] == end and row['known_light_length'] == len(tokens)
                and row['early_terminal_count'] == tokens.count(terminal) and row['repaired'] is False
                and row['terminal_status'] == ('predicted_early_terminal' if end < len(tokens) else 'fixed_known_length'),
                'Candidate was repaired, truncated inconsistently, or changed')
        audit = row['audit']; trace = audit['trace']
        require(row['hidden_reference_input_invariance'] is True and audit['labels_provided'] is False
                and audit['max_iter'] == 16 and audit['seed'] == row['seed']
                and audit['sampling_strategy'] == 'gumbel_argmax'
                and audit['decoding_strategy'] == 'confidence-deterministic-linear'
                and audit['temperature'] == 1.0 and audit['cfg_scale'] == 0.0
                and 1 <= audit['forward_calls'] == len(trace) <= 16, 'Wrong decoding/input audit')
        pending = len(tokens)
        for step, observation in enumerate(trace, 1):
            require(observation['iteration'] == step and observation['pending_before'] == pending
                    and 0 <= observation['pending_after'] <= pending
                    and re.fullmatch('[0-9a-f]{64}', observation['actual_forward_input_sha256']),
                    'Incomplete or inconsistent actual forward trace')
            pending = observation['pending_after']
        require(pending == 0 and audit['original_input_sha256'] == trace[0]['actual_forward_input_sha256'],
                'Unfinished candidate or initial input identity mismatch')
    return dict(status='all_assigned_candidate_artifacts_validated', heavy_rows=len(records),
                candidates=len(rows), empty_candidates=sum(not r['sequence'] for r in rows),
                predicted_early_terminal_candidates=sum(r['early_terminal_count'] > 0 for r in rows),
                candidates_dropped=0, scoring_performed=False)


def seal_predictions(output, identity, records, preflight_ref):
    """Seal complete raw observations and the scoring CSV; never claim a score."""
    manifest = read(output/'manifest.json')
    require(manifest['status'] == 'running' and manifest['identity'] == identity
            and manifest['identity_sha256'] == object_sha256(identity), 'Wrong running artifact identity')
    saved = [json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
    validation = validate_rows(saved, identity, records)
    csv_path = output/'holdout500_n8.csv'
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['_input_row_idx','variant_idx','h_sequence','raw_l_sequence','gen_l_sequence'])
        writer.writeheader()
        for row in saved:
            writer.writerow(dict(_input_row_idx=row['row_index'],variant_idx=row['variant_idx'],
                h_sequence=row['h_sequence'],raw_l_sequence=row['raw_l_sequence'],gen_l_sequence=row['sequence']))
    with csv_path.open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == len(saved) and all(
        (int(a['_input_row_idx']), int(a['variant_idx']), a['h_sequence'], a['raw_l_sequence'], a['gen_l_sequence'])
        == (b['row_index'], b['variant_idx'], b['h_sequence'], b['raw_l_sequence'], b['sequence'])
        for a, b in zip(csv_rows, saved)), 'Scoring CSV lost or altered a candidate')
    result = dict(validation, status='full_predictions_complete_scoring_pending',
        identity_sha256=manifest['identity_sha256'], predictions=ref(output/'predictions.jsonl'),
        scoring_csv=ref(csv_path), preflight=preflight_ref, at_utc=datetime.now(timezone.utc).isoformat())
    manifest.update(status=result['status'], predictions=result['predictions'], scoring_csv=result['scoring_csv'])
    write(output/'manifest.json', manifest)
    write(output/'PREDICTIONS_COMPLETE.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--action', choices=('preflight', 'generate'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args(argv)
    output = args.output_root
    require(output.is_absolute() and output == output.resolve()
            and output.is_relative_to(ROOT/'output/downstream_generation')
            and not any(p.is_symlink() for p in (output, *output.parents)) and not output.exists(),
            'New canonical output directory required; never overwrite or retry a started attempt')
    stopped(output)
    records, binding = bind_inputs()
    code = code_identity()
    if args.action == 'generate':
        require(args.preflight is not None, 'A real native input preflight is required')
        preflight = read(args.preflight); preflight_ref = ref(args.preflight)
        require(preflight['status'] == 'actual_native_pairing_input_audit_passed'
                and preflight['binding'] == binding and preflight['code'] == code
                and preflight['protocol'] == PROTOCOL and preflight['device_type'] == args.device.split(':')[0]
                and preflight['checkpoint'] == ref(CHECKPOINT/'formal_checkpoint.json'),
                'Preflight does not bind this checkpoint, protocol, sources, data, and device type')
    else:
        require(args.preflight is None, 'Preflight cannot reuse a previous preflight')
        preflight_ref = None
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='initializing', action=args.action, at_utc=datetime.now(timezone.utc).isoformat(),
                 binding=binding, code=code, protocol=PROTOCOL, preflight=preflight_ref)
    write(output/'status.json', state)
    try:
        model, loaded = load_native_checkpoint(CHECKPOINT, args.device)
        identity = dict(binding=binding, checkpoint=ref(CHECKPOINT/'formal_checkpoint.json'),
            native_model_load=loaded, code=code, protocol=PROTOCOL, device_type=args.device.split(':')[0],
            tokens=[model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)],
            terminal_id=model.terminal_id, allowed_target_ids=model.allowed_target.nonzero().flatten().tolist())
        write(output/'identity.json', identity)
        if args.action == 'preflight':
            captures = []
            original = model.denoise
            def capture(**features):
                require(set(features) == {'input_ids','attention_mask','role_ids','scope_ids','chain_ids','position_ids','target_slot_mask'},
                        'Unexpected actual model input')
                captures.append(input_fingerprint(features))
                return original(**features)
            model.denoise = capture
            checks = []
            try:
                for record in records[:2]:
                    for variant in range(2):
                        stopped(output); captures.clear()
                        first = predict(model, record, variant); observed = list(captures)
                        changed = {**record, 'light': 'G'*len(record['light'])}
                        captures.clear(); second = predict(model, changed, variant)
                        require(observed == captures == [t['actual_forward_input_sha256'] for t in first['audit']['trace']]
                                and first['raw_light_token_ids'] == second['raw_light_token_ids'],
                                'Actual hidden-reference perturbation changed input or generation')
                        checks.append(dict(row_index=record['row_index'],variant_idx=variant,
                            forward_calls=first['audit']['forward_calls'], actual_input_hashes=observed,
                            hidden_reference_perturbation_equal=True))
            finally:
                model.denoise = original
            result = dict(status='actual_native_pairing_input_audit_passed', binding=binding, code=code,
                protocol=PROTOCOL, checkpoint=identity['checkpoint'], device_type=identity['device_type'],
                native_model_load=loaded, cases=checks, labels_provided=False, scored=False,
                full_evaluation=False, at_utc=datetime.now(timezone.utc).isoformat())
            write(output/'preflight.json', result)
        else:
            manifest = dict(status='running', identity=identity, identity_sha256=object_sha256(identity))
            write(output/'manifest.json', manifest)
            rows = []
            with (output/'predictions.jsonl').open('x') as stream:
                for record in records:
                    for variant in range(8):
                        stopped(output); row = predict(model, record, variant)
                        row['identity_sha256'] = manifest['identity_sha256']
                        rows.append(row); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                    write(output/'progress.json', dict(heavy_rows=record['row_index']+1,candidates=len(rows)))
            result = seal_predictions(output, identity, records, preflight_ref)
        state.update(status=result['status'], finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(output/'status.json', state)
        print(json.dumps(dict(status=state['status'],output=str(output)),ensure_ascii=False),flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error),finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(output/'status.json', state)
        raise


if __name__ == '__main__':
    main()
