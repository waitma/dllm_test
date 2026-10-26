"""Activate pllm/protenix_abtcr; python -B THIS_FILE --run ABS --check.

GPU workers: --task cdr|pairing via the prepared non-preemptible eval_jobs YAML.
Reuse the pinned AirGen CDR evaluator and the validated reference pairing v2.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def evaluator(run):
    path = run / 'airgen_evaluate.py'
    spec = importlib.util.spec_from_file_location('_oas_cdr_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from downstream import fixed_chain_eval, v5_native_eval
    v5_native_eval.load = fixed_chain_eval.load
    module.inputs = fixed_chain_eval.airgen_inputs
    module.CKPT = run / 'checkpoint'
    module.OUT = run / 'cdr'
    module.report = lambda: None
    return module


def invoke(module, *argv):
    saved = sys.argv
    try:
        sys.argv = [str(module.__file__), *argv]
        module.main()
    finally:
        sys.argv = saved


def pairing_gate(model, frame, out, device):
    from downstream.fixed_chain_pairing import decode
    from scripts.downstream.run_fixed_chain_pairing_reference import reference_input, check
    from scripts.downstream.debug_pairing_sampler import sample
    if device == 'cpu':
        check(model, frame, out, 'cpu')
    # Compare against the exact implementation used in the successful sampler
    # diagnostic, on this checkpoint and device, without an unrelated baseline.
    row = frame.iloc[0]
    f, target, positions = reference_input(model, row.h_sequence.replace('-', ''),
                                          len(row.l_sequence.replace('-', '')))
    actual = decode(model, f, target, positions, seed=42)
    expected = sample(model, f, target, 'legacy', 42)
    assert actual['raw_light_token_ids'][:int(target.sum())] == expected['raw_light_token_ids']
    assert (actual.get('sequence') or '') == expected['sequence']
    write(out / f'{device}_sampler_alignment.json', dict(status='passed', device=device,
          actual_checkpoint=True, matches_validated_debug_sampler=True,
          protocol=actual['sampler_protocol']))


def audit_cdr(run, module, tokenizer):
    import numpy as np
    receipts = {}
    for dataset in module.MODES:
        folder = run / 'cdr' / dataset
        originals, _ = module.records(dataset)
        predictions = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
        assert len(predictions) == len(originals)
        groups = defaultdict(lambda: defaultdict(list))
        for expected, pred in zip(originals, predictions):
            assert all(expected[key] == pred[key] for key in ('mode', 'fold', 'row', 'key'))
            positions = [j + 1 for j in expected['mask_h']] + [151 + j for j in expected['mask_l']]
            reference = ''.join(expected['gold_h'][j] for j in expected['mask_h'])
            reference += ''.join(expected['gold_l'][j] for j in expected['mask_l'])
            assert pred['masked_positions'] == positions
            assert pred['reference_ids'] == tokenizer.encode_residues(reference)
            assert len(pred['prediction_ids']) == len(positions) > 0
            correct = sum(a == b for a, b in zip(pred['reference_ids'], pred['prediction_ids']))
            value = float(np.float32(correct) / np.float32(len(positions)))
            assert abs(value - pred['aar']) < 1e-7
            groups[pred['mode']][pred['fold']].append(value)
        metrics = json.loads((folder / 'metrics.json').read_text())
        for mode, folds in groups.items():
            values = [v for fold in folds.values() for v in fold]
            fold_means = [float(np.mean(folds[i]) * 100) for i in sorted(folds)]
            score = float(np.mean(fold_means)) if dataset == 'kong' else float(np.round(np.mean(values), 4) * 100)
            assert len(values) == metrics['modes'][mode]['n_sequences']
            assert abs(score - metrics['modes'][mode]['airgen_aar_percent']) < 1e-8
        receipts[dataset] = dict(records=len(predictions), predictions_sha256=sha(folder / 'predictions.jsonl'))
    write(run / 'cdr' / 'verification.json', dict(status='passed', datasets=receipts))


def pairing(run, plan):
    import pandas as pd
    import torch
    from downstream.fixed_chain_eval import load
    from downstream.fixed_chain_pairing import decode
    from scripts.downstream.run_fixed_chain_pairing_reference import reference_input
    from scripts.downstream.debug_pairing_sampler import score
    from scripts.downstream.summarize_pairing_sampler_debug import independent_diversity
    out = run / 'pairing'
    model, identity = load(run / 'checkpoint', 'cuda')
    frame = pd.read_csv(plan['benchmark']['path'], keep_default_na=False)
    assert len(frame) == 500
    pairing_gate(model, frame, out, 'cuda')
    write(out / 'identity.json', dict(model=identity, protocol=plan['protocol'], precision='bf16_autocast'))
    rows = []
    with (out / 'predictions.jsonl').open('x') as stream:
        for i, row in frame.iterrows():
            heavy, reference = row.h_sequence.replace('-', ''), row.l_sequence.replace('-', '')
            features, target, positions = reference_input(model, heavy, len(reference))
            for variant in range(8):
                raw = decode(model, features, target, positions, seed=42 + 8 * i + variant)
                ids = raw['raw_light_token_ids']
                tok = model.tokenizer
                assert len(ids) == model.config.second_chain_capacity
                assert ids[len(reference)] == tok.eos_token_id
                assert all(t == tok.mask_token_id for t in ids[len(reference) + 1:])
                assert tok.mask_token_id not in ids[:len(reference)]
                end = ids.index(tok.eos_token_id)
                sequence = raw.get('sequence') or ''
                if raw['status'] == 'ok':
                    assert sequence == ''.join(tok.token(t) for t in ids[:end])
                item = dict(row_index=int(i), variant_idx=variant, h_sequence=heavy,
                            raw_l_sequence=reference, gen_l_sequence=sequence, raw=raw)
                rows.append(item)
                stream.write(json.dumps(item) + '\n')
            stream.flush()
            write(out / 'progress.json', dict(heavy_completed=int(i) + 1, candidates=len(rows)))
            print('pairing', i + 1, '/ 500', flush=True)
    assert len(rows) == 4000 and len({(r['row_index'], r['variant_idx']) for r in rows}) == 4000
    data = pd.DataFrame([{k: v for k, v in row.items() if k != 'raw'} for row in rows])
    data.to_csv(out / 'generated.csv', index=False)
    del model
    torch.cuda.empty_cache()
    sys.path.insert(0, str(ROOT / 'downstream/comp_chain/eval_scripts'))
    os.chdir(out)
    score(out)
    metrics = json.loads((out / 'metrics.json').read_text())
    metrics.update(native_identity=identity, evaluation_protocol=plan['protocol'],
                   candidate_attempts=4000, length_condition='reference; no light residues supplied')
    assert metrics['total_pairs'] == 4000
    details = metrics['detailed_results']
    assert len(details) == 4000 and [r['index'] for r in details] == list(range(4000))
    for key, field in [('gen_valid_rate', 'gen_valid'), ('overall_chain_match_rate', 'chain_match')]:
        assert abs(sum(bool(row[field]) for row in details) / 4000 - metrics[key]) < 1e-10
    diversity = sum(independent_diversity(g.gen_l_sequence.tolist()) for _, g in data.groupby('h_sequence')) / data.h_sequence.nunique()
    assert abs(diversity - metrics['diversity_mean']) < 1e-10
    write(out / 'metrics.json', metrics)
    write(out / 'verification.json', dict(status='passed', candidates=4000,
          predictions_sha256=sha(out / 'predictions.jsonl'), all_attempts_retained=True,
          independent_metric_aggregation=True, matches_validated_debug_sampler=True))


def publish(run, task):
    label = 'Ours OAS-only fixed-chain 75.6M / 11000'
    entries = []
    if task == 'cdr':
        for dataset, heading, modes, setting in [
            ('kong', 'SAbDab Kong：H1 / H2 / H3', ['cdrh1', 'cdrh2', 'cdrh3'], 'iter=1 / bf16'),
            ('sab23', 'SAb23H2：六个 CDR', ['cdrh1', 'cdrh2', 'cdrh3', 'cdrl1', 'cdrl2', 'cdrl3'], 'iter=2 / fp32')]:
            path = run / 'cdr' / dataset / 'metrics.json'
            metrics = json.loads(path.read_text())
            entries.append((heading, f'{label}, {setting}', path,
                            [metrics['modes'][m]['airgen_aar_percent'] for m in modes]))
    else:
        path = run / 'pairing' / 'metrics.json'
        metrics = json.loads(path.read_text())
        entries.append(('AB Light-chain pairing', f'{label}, p0 / reference / iter=16 / 对齐V7采样器', path,
                        [metrics[k] for k in ('gen_immunomatch_mean', 'gen_valid_rate', 'overall_chain_match_rate', 'diversity_mean')]))
    with (ROOT / 'output/downstream_generation/v5_native_best_20261008/results_update.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for filename in ('RESULTS.md', 'RESULTS_SUMMARY.md'):
            path = ROOT / 'downstream/benchmark' / filename
            text = path.read_text()
            for heading, name, evidence, values in entries:
                if name in text:
                    continue
                start = text.index(heading)
                table = text.index('\n|', start) + 1
                end = text.index('\n\n', table)
                title = f'[{name}]({evidence})' if filename == 'RESULTS.md' else name
                row = '| ' + title + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |'
                if task == 'cdr' and heading.startswith('SAbDab') and filename == 'RESULTS.md':
                    row += ' AirGen掩码对齐；OAS-only；最新完整checkpoint |'
                text = text[:end] + '\n' + row + text[end:]
            path.write_text(text)
        with (ROOT / 'downstream/benchmark/PROGRESS.md').open('a') as stream:
            stream.write(f'\n{datetime.now(timezone.utc).isoformat()} OAS-only 11000 {task} 全量评测及独立聚合核验完成，结果已回填原表。[产物]({run}/{task})。\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--task', choices=('cdr', 'pairing'))
    args = parser.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected, 'Evaluation source changed: ' + path
    assert sha(plan['benchmark']['path']) == plan['benchmark']['sha256']
    from downstream.fixed_chain_pairing import validate_plan
    validate_plan(plan)
    import torch
    torch.set_num_threads(4)
    if args.check:
        module = evaluator(run)
        invoke(module, '--audit')
        invoke(module, '--check')
        import pandas as pd
        from downstream.fixed_chain_eval import load
        model, identity = load(run / 'checkpoint', 'cpu')
        assert identity['step'] == 11000 and identity['model_sha256'] == plan['model_sha256']
        pairing_gate(model, pd.read_csv(plan['benchmark']['path'], keep_default_na=False), run / 'pairing', 'cpu')
        write(run / 'cpu_gate.json', dict(status='passed', model=identity, cdr_modes=9, pairing_records=500))
        return
    if args.task is None:
        parser.error('--check or --task required')
    assert json.loads((run / 'cpu_gate.json').read_text())['status'] == 'passed'
    out = run / args.task
    assert not (out / 'status.json').exists(), 'Task already started; use a new output for retry'
    write(out / 'status.json', dict(status='running', started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        if args.task == 'cdr':
            module = evaluator(run)
            for dataset in ('kong', 'sab23'):
                invoke(module, '--dataset', dataset)
            from downstream.fixed_chain_eval import load
            model, _ = load(run / 'checkpoint', 'cpu')
            audit_cdr(run, module, model.tokenizer)
        else:
            pairing(run, plan)
        publish(run, args.task)
        write(out / 'status.json', dict(status='success', completed_utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as error:
        write(out / 'status.json', dict(status='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
