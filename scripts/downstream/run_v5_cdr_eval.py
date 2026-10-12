"""Evaluate native V5 best validation checkpoint with the existing fixed CDR protocol.

Activate conda pllm. Run this file with --help in the protenix_abtcr environment.
Required: --checkpoint ABS --dataset kong|sab23
--output-root NEW_ABSOLUTE_DIRECTORY [--device cuda:0]. Uses iter=1 argmax only.
No annotations, training, parameter sweep, resource submission, or checkpoint
copy is performed. Every row is saved through the existing CDRResultWriter.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import sys

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
EVAL_SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_SOURCE))
from downstream.v7_native import cdr_argmax, input_fingerprint, require, sha
from downstream.v5_native_eval import load, cdr_input


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', choices=('kong', 'sab23'), required=True)
    parser.add_argument('--mode', choices=('cdrh1', 'cdrh2', 'cdrh3', 'cdrl1', 'cdrl2', 'cdrl3'),
                        help='One independently scored task for parallel GPU execution')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--inventory', type=Path, default=ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_benchmark_inventory_v1.json')
    parser.add_argument('--max-samples', type=int, help='Partial diagnostic only; per fold')
    args = parser.parse_args(argv)
    require(args.output_root.is_absolute() and args.output_root == args.output_root.resolve()
            and args.output_root.is_relative_to(ROOT/'output/downstream_generation'), 'New canonical evaluation output root required')
    require(not args.output_root.exists(), 'Do not overwrite or resume a different evaluation')
    require(args.max_samples is None or args.max_samples > 0, 'Invalid partial limit')
    model, loaded = load(args.checkpoint, args.device)
    # Model imports must occur only after source isolation in the loader.
    import torch
    # The legacy grammar package __init__ eagerly imports fusion. Execute the
    # unchanged pure artifact module by path to keep the model loader isolated.
    result_api = runpy.run_path(str(EVAL_SOURCE/'downstream/grammar/cdr_result_io.py'), run_name='_v7_cdr_artifacts')
    CDRResultWriter, checkpoint_identity = result_api['CDRResultWriter'], result_api['checkpoint_identity']
    from downstream.benchmark.common.reporting import file_identity
    tokenizer = model.tokenizer
    checkpoint = checkpoint_identity(args.checkpoint)
    base = ROOT/'data/downstream/cdr_infilling'/('sabdab_kong' if args.dataset == 'kong' else 'sab23h2_converted')
    modes = ('cdrh1', 'cdrh2', 'cdrh3') if args.dataset == 'kong' else ('cdrl1', 'cdrl2', 'cdrl3', 'cdrh1', 'cdrh2', 'cdrh3')
    if args.mode is not None:
        require(args.mode in modes, 'CDR mode is outside the selected dataset protocol')
        modes = (args.mode,)
    folds = 10 if args.dataset == 'kong' else 1
    inventory = read(args.inventory)
    require(inventory['status'] == 'existing_benchmark_inventory_only', 'Pinned benchmark inventory required')
    expected_sources = {(x['dataset'], x['mode'], x['fold']): x for x in inventory['files']}
    require(len(expected_sources) == len(inventory['files']), 'Duplicate benchmark identity')
    sources = {}
    for mode in modes:
        for fold in range(folds):
            candidates = [base/mode/f'fold_{fold}'/'test.json', base/mode/'test.json']
            path = next((p for p in candidates if p.is_file()), None)
            require(path is not None, f'Missing benchmark fold: {mode}/{fold}')
            require(folds == 1 or 'fold_' in str(path), 'Do not substitute one file for multiple folds')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            require(bool(rows), 'Empty benchmark fold')
            expected = expected_sources[(args.dataset, mode, fold)]
            require(str(path) == expected['path'] and sha(path) == expected['sha256']
                    and len(rows) == expected['records'], 'Benchmark differs from the fixed pre-evaluation inventory')
            sources[(mode, fold)] = (path, rows, file_identity(path))
    code = [file_identity(EVAL_SOURCE/name) for name in ('scripts/downstream/run_v5_cdr_eval.py',
        'downstream/v7_native.py', 'downstream/v5_native_eval.py', 'downstream/v5_layout.py', 'downstream/grammar/cdr_result_io.py',
        'downstream/benchmark/common/reporting.py')]
    args.output_root.mkdir(parents=True, exist_ok=False)
    write(args.output_root/'model_load.json', loaded)
    summary = dict(status='running', dataset=args.dataset, checkpoint=checkpoint, step=loaded['step'],
                   coverage='assigned_mode' if args.mode else 'all_dataset_modes', assigned_mode=args.mode,
                   protocol='v5_native_object302_cdr_iter1_argmax_v1', modes={}, started_at_utc=datetime.now(timezone.utc).isoformat())
    write(args.output_root/'status.json', summary)
    try:
        for mode in modes:
            identity = dict(checkpoint=checkpoint,
                datasets=[{**sources[(mode, fold)][2], 'fold': fold, 'n_rows': len(sources[(mode, fold)][1])} for fold in range(folds)],
                tokenizer=dict(tokens=[tokenizer.token(i) for i in range(tokenizer.vocab_size)],
                               mask_token_id=tokenizer.mask_token_id, pad_token_id=tokenizer.pad_token_id),
                code=code, model_source=loaded, benchmark_inventory=file_identity(args.inventory),
                parameters=dict(mode=mode, num_folds_requested=folds, resolved_folds=folds,
                    max_samples=args.max_samples, max_iter=1, sampling_strategy='argmax', temperature=1.0,
                    device=args.device, torch_version=torch.__version__, normalization='strip whitespace; uppercase; J->L (legacy normalized residue identity)',
                    protocol='v5_native_object302_cdr_iter1_argmax_v1', length_condition='known_cdr_and_visible_chain_lengths',
                    cdr_location='first_exact_subsequence_match_as_legacy', relation='unknown',
                    target_vocabulary='native_V5_residues_split_protd; nonresidue predictions count incorrect',
                    input_layout='trained_V5_object_capacity; visible_framework_other_chain_and_termini'))
            writer = CDRResultWriter(args.output_root/mode, identity)
            audit_path = args.output_root/mode/'input_audit.jsonl'
            role = 'antibody_heavy' if mode.startswith('cdrh') else 'antibody_light'
            with audit_path.open('x') as audit:
                for fold in range(folds):
                    path, rows, source_id = sources[(mode, fold)]
                    for index, row in enumerate(rows):
                        if args.max_samples is not None and index >= args.max_samples:
                            break
                        heavy = ''.join(str(row['heavy_chain_seq']).split()).upper().replace('J', 'L')
                        light = ''.join(str(row['light_chain_seq']).split()).upper().replace('J', 'L')
                        reference = ''.join(str(row[f'{mode}_seq']).split()).upper().replace('J', 'L')
                        sequence = heavy if role == 'antibody_heavy' else light
                        start = sequence.find(reference)
                        require(bool(reference) and start >= 0, f'CDR target absent: {mode}/{fold}/{index}')
                        end = start+len(reference)
                        features, mask = cdr_input(model, heavy, light, role, start, end)
                        # Only hidden letters change; coordinates, known lengths and context stay fixed.
                        altered = sequence[:start]+'G'*(end-start)+sequence[end:]
                        alt_h, alt_l = (altered, light) if role == 'antibody_heavy' else (heavy, altered)
                        alternative, alt_mask = cdr_input(model, alt_h, alt_l, role, start, end)
                        require(all(torch.equal(features[k], alternative[k]) for k in features)
                                and torch.equal(mask, alt_mask), 'Hidden-reference perturbation changed native inputs')
                        expected_fingerprint = input_fingerprint(features)
                        output, observation = cdr_argmax(model, features, mask)
                        require(observation['actual_forward_input_sha256'] == expected_fingerprint,
                                'Actual model input differs from verified masked tensors')
                        references = tokenizer.encode_residues(reference)
                        predictions = output[mask.to(output.device)].tolist()
                        correct = sum(a == b for a, b in zip(references, predictions))
                        aar = float(torch.tensor(correct, dtype=torch.float32)/len(references))
                        writer.append(fold=fold, row=index, reference_ids=references, prediction_ids=predictions,
                                      masked_positions=mask[0].nonzero().flatten().tolist(), aar=aar)
                        audit.write(json.dumps(dict(fold=fold, row=index, source_sha256=source_id['sha256'],
                            start=start, end=end, ambiguous_subsequence=sequence.find(reference, start+1) >= 0,
                            hidden_reference_perturbation_equal=True, **observation))+'\n')
                    require(file_identity(path) == source_id, 'Benchmark file changed during evaluation')
            result = writer.finish(allow_partial=args.max_samples is not None)
            summary['modes'][mode] = dict(metrics=str(args.output_root/mode/'metrics.json'), status=result['status'],
                                         input_audit=file_identity(audit_path))
            write(args.output_root/'status.json', summary)
        summary['status'] = 'complete' if all(x['status'] == 'complete' for x in summary['modes'].values()) else 'partial'
        summary['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(args.output_root/'status.json', summary)
        write(args.output_root/'summary.json', summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), stopped_at_utc=datetime.now(timezone.utc).isoformat())
        write(args.output_root/'status.json', summary)
        raise


if __name__ == '__main__':
    main()
