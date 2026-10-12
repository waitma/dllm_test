"""Audit actual native V7 TCR inputs without scoring or full benchmark generation.

Activate conda pllm and run with the worker-compatible Python in a clean process:
python -I -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/preflight_v7_tcr_generation.py --device cpu --output-root /absolute/new_output_directory
The CPU audit does not authorize a GPU run. No annotation, training, retries,
checkpoint copies, resource submission, or certification is performed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import sys

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
from downstream.v7_native import load_native_checkpoint, input_fingerprint, require, sha
from downstream.v7_tcr_generation import beta_input, beta_decode, beta_result, PROTOCOL

EPOCH = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006'
RECOVERY = EPOCH/'recovery_v1'
CDR = RECOVERY/'generation_pool_v1'
CHECKPOINT = EPOCH/'run/checkpoint-50082'
BASE = ROOT/'output/v7_tcr_adapter_20261006'
INVENTORY = BASE/'benchmark_inventory_v1.json'
INVENTORY_SHA = '1199a5c98382309b0c419773d9eb4cb36b2eb33fbff19c2ad88b394a1257866f'
GATE = ROOT/'output/v7_local_ablation_20261005/post_generation_pool_v2/source_v1/v7_post_generation_acceptance.py'
GATE_SHA = '9a05fb9cee32aa6e60790fb2b1afb07eb3b63aa8dafe127e05d14cb070338f0c'


def read(path):
    return json.loads(path.read_text())


def reference(path):
    require(path.is_absolute() and path == path.resolve() and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical evidence required')
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def check_stop(output):
    for directory in (output, BASE, ROOT/'output/immune_research_pool_c20250601_20261005'):
        require(not any((directory/n).exists() for n in ('STOP','STOP_NOW')), 'Explicit stop is present')


def bind_inputs():
    require(sha(INVENTORY) == INVENTORY_SHA and sha(GATE) == GATE_SHA, 'Fixed prerequisites changed')
    inventory = read(INVENTORY)
    for item in inventory['formal_metadata'].values():
        require(reference(Path(item['path'])) == item, 'Formal data metadata changed')
    require(reference(Path(inventory['benchmark']['path'])) == inventory['benchmark'], 'T4 benchmark changed')
    fields = dict(formal_completion=reference(RECOVERY/'COMPLETE.json'),
        formal_stage=str(RECOVERY/'launches_v1/segment_50082'),
        generation_completion=reference(CDR/'COMPLETE.json'),
        generation_release=reference(CDR/'release.json'), generation_package=reference(CDR/'package_v1.json'))
    gate = runpy.run_path(str(GATE), run_name='_existing_two_epoch_completion_gate')
    actual = gate['check'](fields)
    entries = read(Path(inventory['benchmark']['path']))
    observed = [dict(index=i,target_key=k,category=e['category'],epitope_length=len(e['epitope']),
                    mhc_pseudo_length=len(e['mhc_pseudo']),reference_count=len(e['ref_binders']))
                for i,(k,e) in enumerate(entries.items())]
    require(observed == inventory['entries'], 'Benchmark inventory differs')
    return entries, dict(inventory=reference(INVENTORY), existing_gate=reference(GATE),
                         predecessor=fields, actual_completion=actual)


def code_identity():
    names = ['scripts/downstream/preflight_v7_tcr_generation.py', 'downstream/v7_tcr_generation.py',
             'downstream/v7_native.py', 'downstream/v7_pairing.py',
             'downstream/benchmark/tcr_generation_bench/scoring.py', 'downstream/benchmark/common/metrics.py']
    return [reference(ROOT/name) for name in names]


def predict(model, entry, seed):
    features, target = beta_input(model, entry['epitope'], entry['mhc_pseudo'])
    tokens, audit = beta_decode(model, features, target, seed=seed)
    return beta_result(model, tokens, target), audit


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--device', choices=('cpu','cuda:0'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(); output = args.output_root
    require(output.is_absolute() and output == output.resolve() and not output.exists()
            and output.is_relative_to(ROOT/'output/downstream_generation')
            and not any(p.is_symlink() for p in (output, *output.parents)), 'New canonical output required')
    check_stop(output)
    entries, binding = bind_inputs(); code = code_identity()
    output.mkdir(parents=True, exist_ok=False)
    write(output/'started.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        device=args.device, binding=binding, code=code))
    try:
        model, loaded = load_native_checkpoint(CHECKPOINT, args.device)
        require(model.config.fragment_capacity == 64, 'Expected native V7 junction capacity64')
        captures = []; original = model.denoise; checks = []
        def capture(**features):
            require(set(features) == {'input_ids','attention_mask','role_ids','scope_ids',
                'chain_ids','position_ids','target_slot_mask'}, 'Unexpected actual model inputs')
            captures.append(input_fingerprint(features))
            return original(**features)
        model.denoise = capture
        try:
            for category in ('benchmark14','held20'):
                check_stop(output)
                key,entry = next((k,e) for k,e in entries.items() if e['category']==category)
                captures.clear(); first,audit = predict(model,entry,42); observed = list(captures)
                # Change both number and lengths of hidden references before the actual call.
                changed = {**entry, 'ref_binders':['G','G'*200], 'n_ref':2}
                captures.clear(); second,other = predict(model,changed,42)
                require(first == second and audit == other and observed == captures
                        == [t['actual_forward_input_sha256'] for t in audit['trace']],
                        'Hidden reference content/length affected actual inputs or outputs')
                checks.append(dict(category=category,target_key=key,actual_forward_calls=len(observed),
                    repeated_actual_forward_calls=len(captures),native_capacity=64,
                    hidden_reference_content_and_length_invariant=True,actual_input_hashes=observed,
                    candidate=first,audit=audit))
        finally:
            model.denoise=original
        require(code_identity()==code, 'Sources changed during the real input audit')
        result = dict(at_utc=datetime.now(timezone.utc).isoformat(),
            status='actual_native_tcr_input_audit_passed', protocol=PROTOCOL, binding=binding,
            code=code, device_type=args.device.split(':')[0], native_model_load=loaded,
            checkpoint=reference(CHECKPOINT/'formal_checkpoint.json'), cases=checks,
            labels_provided=False, reference_lengths_provided=False, scoring_performed=False,
            full_benchmark_generation=False, annotation_called=False)
        write(output/'preflight.json',result)
        print(json.dumps(dict(status=result['status'],output=str(output))),flush=True)
    except BaseException as error:
        write(output/'FAILURE.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),error=repr(error)))
        raise


if __name__ == '__main__':
    main()
