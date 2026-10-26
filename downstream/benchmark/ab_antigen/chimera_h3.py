"""CHIMERA H3 sequence track on the unfiltered official epitope-group split.

CPU: python -m downstream.benchmark.ab_antigen.chimera_h3 prepare
CPU: python -m downstream.benchmark.ab_antigen.chimera_h3 preflight
CPU: python -m downstream.benchmark.ab_antigen.chimera_h3 score --predictions FILE --out-dir DIR
GPU: python -m downstream.benchmark.ab_antigen.chimera_h3 run --checkpoint CKPT --out-dir NEW_DIR
The CLI loads native V5 object302 checkpoints on Volc; run() also accepts loaded fixed-chain models.
Official structures supply scoring references only, never model features.
"""
from contextlib import nullcontext
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from downstream.v7_native import sha, input_fingerprint

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'data/downstream/ab_antigen_sequence_v2_raw_rows/raw/chimera'
DATA = ROOT / 'data/downstream/chimera_h3_epitope_group_v1'
OFFICIAL = ROOT / 'downstream/benchmark/baselines/chimera_bench_official'
COMMIT = 'c0ada2ff045e1622ef39e4a25db34724813f378a'
PROTOCOL = 'chimera_epitope_group_h3_sequence_v1'
AA = 'ACDEFGHIKLMNPQRSTVWY'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def official_modules():
    """Use the pinned author implementation; reject changed scoring source."""
    head = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        raise ValueError('CHIMERA official source revision changed')
    sources = ['chimera_bench/__init__.py', 'chimera_bench/data.py',
               'chimera_bench/metrics.py', 'chimera_bench/evaluate.py']
    for name in sources:
        original = subprocess.check_output(['git', '-C', str(OFFICIAL), 'show', COMMIT + ':' + name])
        if hashlib.sha256(original).hexdigest() != sha(OFFICIAL / name):
            raise ValueError('Modified official scorer: ' + name)
    sys.path.insert(0, str(OFFICIAL))
    modules = [importlib.import_module('chimera_bench.' + name) for name in ('data', 'metrics', 'evaluate')]
    for module in modules:
        if not Path(module.__file__).resolve().is_relative_to(OFFICIAL.resolve()):
            raise ValueError('Conflicting CHIMERA import')
    return (*modules, {name: sha(OFFICIAL / name) for name in sources})


def prepare(raw=RAW, destination=DATA, scheme='imgt'):
    """Keep every official row; record input incompatibilities instead of deleting IDs."""
    raw, destination = Path(raw), Path(destination)
    data, metrics, _, sources = official_modules()
    split_path = raw / 'splits/epitope_group.json'
    split = data.load_split('epitope_group', data_root=raw)
    rows, feature_hashes = [], {}
    for subset in ('train', 'val', 'test'):
        for cid in split[subset]:
            path = raw / 'complex_features' / (cid + '.pt')
            feat = data.load_complex(cid, data_root=raw)
            feature_hashes[cid] = sha(path)
            indices = data.cdr_indices(feat, 'H3', scheme).tolist()
            heavy = feat['heavy_sequence']
            errors = []
            if not indices or indices != list(range(indices[0], indices[-1] + 1)):
                errors.append('missing_or_noncontiguous_official_h3')
            if indices and indices[-1] >= len(heavy):
                raise ValueError('Official H3 outside sequence: ' + cid)
            # Exactly the CA contact definition used by released baseline full evaluation.
            pairs = metrics.compute_ca_contacts(np.asarray(feat['heavy_ca_coords'])[indices],
                                                np.asarray(feat['antigen_ca_coords']), cutoff=8.0)
            contacts = sorted({int(i) for i, _ in pairs})
            bounds, variable = {}, {}
            for chain in ('heavy', 'light'):
                full = feat[chain + '_sequence']
                fv = ''.join(e[2] for e in feat['numbering']['imgt'][chain] if e[2] != '-')
                start = full.find(fv) if fv else -1
                if start < 0 or full.find(fv, start + 1) >= 0:
                    errors.append('unverified_' + chain + '_fv_mapping')
                bounds[chain] = [start, start + len(fv)]
                variable[chain] = fv
            hstart, hend = bounds['heavy']
            if indices and not hstart <= indices[0] < indices[-1] + 1 <= hend:
                errors.append('h3_outside_fv')
            rows.append(dict(complex_id=cid, split=subset, antigen=feat['antigen_sequence'],
                             heavy=variable['heavy'], light=variable['light'], fv_bounds=bounds,
                             h3_start=indices[0] - hstart if indices else None,
                             h3_end=indices[-1] + 1 - hstart if indices else None,
                             h3_indices_raw=indices, native_h3=''.join(heavy[i] for i in indices),
                             h3_contact_indices=contacts, input_errors=errors))
    ids = [row['complex_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate official complex IDs; refusing ambiguous predictions')
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (scheme + '.jsonl')
    content = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows)
    if path.exists() and path.read_text() != content:
        raise ValueError('Prepared CHIMERA data changed; choose a new versioned directory')
    path.write_text(content)
    manifest = dict(protocol=PROTOCOL, split='epitope_group', numbering_scheme=scheme, cdr='H3',
                    source_root=str(raw.resolve()), split_sha256=sha(split_path),
                    official_commit=COMMIT, official_source_sha256=sources, feature_sha256=feature_hashes,
                    counts={s: len(split[s]) for s in split}, removed_rows=0, deduplicated_rows=0,
                    input_errors=[{'complex_id': r['complex_id'], 'errors': r['input_errors']}
                                  for r in rows if r['input_errors']],
                    dataset_sha256=sha(path), contact_definition='official_CA_CA_8A',
                    builder_sha256=sha(__file__),
                    default_candidates=1, candidate_budget_basis='released_MEAN_and_DiffAb_inference_one_per_complex',
                    paper_candidate_budget_verified=False)
    write(destination / (scheme + '.manifest.json'), manifest)
    return rows, manifest


def load_dataset(directory=DATA, scheme='imgt', subset='test'):
    directory = Path(directory)
    if scheme not in ('imgt', 'chothia') or subset not in ('train', 'val', 'test'):
        raise ValueError('Unknown CHIMERA scheme/subset')
    path = directory / (scheme + '.jsonl')
    manifest = json.loads((directory / (scheme + '.manifest.json')).read_text())
    if manifest['protocol'] != PROTOCOL or manifest['dataset_sha256'] != sha(path):
        raise ValueError('CHIMERA prepared data identity mismatch')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [r for r in rows if r['split'] == subset]
    if len(selected) != manifest['counts'][subset]:
        raise ValueError('Incomplete official split')
    return selected, manifest


def model_inputs(model, row):
    """Expose only antigen, visible Fv scaffold and known H3 span to the denoiser."""
    from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
    from downstream.v5_layout import cdr_inputs
    if row.get('input_errors'):
        raise ValueError(','.join(row['input_errors']))
    start, end = row['h3_start'], row['h3_end']
    heavy = row['heavy']
    if not 0 <= start < end <= len(heavy):
        raise ValueError('Invalid H3 span')
    heavy = heavy[:start] + 'A' * (end - start) + heavy[end:]
    record = BioSeqRecord([BioSeqChain(row['antigen'], 'antigen'),
                          BioSeqChain(heavy, 'antibody_heavy'),
                          BioSeqChain(row['light'], 'antibody_light')],
                         task_type='antibody_antigen', source=PROTOCOL)
    if getattr(model.config, 'layout_contract', None) == 'immune.fixed_chain.f011.v1':
        from dllm.pipelines.immune_llada.fixed_chain.layout import FixedChainCollator
        from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
        batch = FixedChainCollator.from_config(model.config)([record])
        features = model.feature_args(batch)
        mask = ((features['role_ids'] == CHAIN_ROLE_TO_ID['antibody_heavy'])
                & (features['position_ids'] >= start + 1) & (features['position_ids'] <= end)
                & features['target_slot_mask'].bool() & features['attention_mask'].bool())
        features['input_ids'] = features['input_ids'].masked_fill(mask, model.tokenizer.mask_token_id)
    else:
        features, mask = cdr_inputs(model, record, 'antibody_heavy', start, end)
    if int(mask.sum()) != end - start:
        raise ValueError('H3 span truncated by model layout')
    return features, mask


def sample_h3(model, features, mask, seed, steps=32, temperature=1.0):
    """Confidence-ordered sampling within the fixed H3 span; never change scaffold."""
    if steps < 1 or not np.isfinite(temperature) or temperature < 0:
        raise ValueError('Invalid decoding settings')
    device = next(model.parameters()).device
    features = {k: v.to(device) for k, v in features.items()}
    mask = mask.to(device)
    initial = int(mask.sum())
    if not initial or not features['input_ids'][mask].eq(model.tokenizer.mask_token_id).all():
        raise ValueError('Every H3 residue must start masked')
    ids = features['input_ids'].clone()
    allowed = torch.tensor(model.tokenizer.encode_residues(AA), device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    steps = min(steps, initial)
    precision = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with torch.inference_mode(), precision:
        for step in range(steps):
            logits, _ = model.denoise(**{**features, 'input_ids': ids})
            positions = (mask & ids.eq(model.tokenizer.mask_token_id))[0].nonzero().flatten()
            scores = logits[0, positions][:, allowed].float()
            if not torch.isfinite(scores).all():
                raise ValueError('Nonfinite H3 logits')
            probs = (scores / (temperature or 1.0)).softmax(-1)
            proposal = probs.argmax(-1) if temperature == 0 else torch.multinomial(probs, 1, generator=rng).flatten()
            confidence = probs.gather(1, proposal[:, None]).flatten()
            count = len(positions) - initial * (steps - step - 1) // steps
            chosen = confidence.topk(count).indices
            ids[0, positions[chosen]] = allowed[proposal[chosen]]
    if not torch.equal(ids[~mask], features['input_ids'][~mask]):
        raise AssertionError('Scaffold changed during H3 sampling')
    sequence = ''.join(model.tokenizer.token(i) for i in ids[mask].tolist())
    if len(sequence) != initial or not set(sequence) <= set(AA):
        raise ValueError('Invalid H3 output')
    return sequence, dict(actual_forward_input_sha256=input_fingerprint(features), forward_calls=steps,
                         labels_provided=False, epitope_provided=False, coordinates_provided=False)


def score(rows, predictions, candidates=1):
    """Use author AAR/CAAR/best-of-K aggregation, enforcing complete ID/budget coverage."""
    if type(candidates) is not int or candidates < 1:
        raise ValueError('candidates must be positive')
    _, _, evaluator, sources = official_modules()
    expected = {(r['complex_id'], k) for r in rows for k in range(candidates)}
    observed = [(p['complex_id'], p['candidate']) for p in predictions]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError('Missing/duplicate/unexpected complex or candidate IDs')
    if len({r['complex_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate reference IDs')
    natives, designs, failed = {}, {}, []
    for row in rows:
        cid = row['complex_id']
        if not row['native_h3']:
            raise ValueError('Official H3 reference missing; cannot score: ' + cid)
        contact_mask = np.zeros(len(row['native_h3']), dtype=bool)
        contact_mask[row['h3_contact_indices']] = True
        natives[cid] = evaluator.NativeComplex(cid, row['native_h3'], None, None, None, contact_mask, set())
        designs[cid] = []
    for pred in predictions:
        cid = pred['complex_id']
        sequence = pred['pred_sequence']
        if pred['status'] != 'ok':
            if sequence:
                raise ValueError('Failed attempts must have an empty sequence')
            failed.append(dict(complex_id=cid, candidate=pred['candidate'], reason=pred.get('failure_reason')))
        elif len(sequence) != len(natives[cid].true_sequence) or not set(sequence) <= set(AA):
            raise ValueError('Malformed successful H3 prediction')
        designs[cid].append(evaluator.DesignResult(cid, 'H3', sequence))
    result = evaluator.evaluate_benchmark(designs, natives)
    # Official generic evaluator fills absent structural composites with zero; those are not measurements.
    keep = {'complex_id', 'n_designs', 'best_aar', 'best_caar', 'median_aar', 'median_caar',
            'seq_diversity', 'mean_liabilities'}
    result['per_complex'] = [{k: v for k, v in row.items() if k in keep} for row in result['per_complex']]
    result['summary'] = {k: v for k, v in result['summary'].items() if k in keep}
    result.update(protocol=PROTOCOL, n_complexes=len(rows), candidates=candidates,
                  n_attempts=len(predictions), failed_attempts=failed,
                  aggregation='per_complex_independent_best_of_K_then_macro_mean_std_bootstrap95',
                  official_commit=COMMIT, official_source_sha256=sources,
                  structural_metrics=None, epitope_metrics=None, deduplicated_rows=0)
    return result


def run(out, model, identity, directory=DATA, scheme='imgt', candidates=1, seed=42, steps=32, temperature=1.0):
    """Frozen native checkpoint evaluation. This performs no CHIMERA fine-tuning."""
    out = Path(out)
    if type(candidates) is not int or candidates < 1:
        raise ValueError('candidates must be positive')
    rows, manifest = load_dataset(directory, scheme)
    _, _, _, sources = official_modules()
    if sources != manifest['official_source_sha256']:
        raise ValueError('Official scorer differs from dataset preparation')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'predictions.jsonl').exists():
        raise FileExistsError('Use a new CHIMERA H3 output directory')
    write(out / 'provenance.json', dict(protocol=PROTOCOL, native_identity=identity, dataset=manifest,
          scope='full', subset='test', candidates=candidates, seed=seed, steps=steps, temperature=temperature,
          conditioning='antigen_sequence_and_fixed_Fv_scaffold_known_H3_span',
          epitope_input=False, structure_input=False, training='frozen_zero_shot',
          pretraining_overlap='not_cleared', code_sha256=sha(__file__),
          layout_sha256=sha(ROOT / 'downstream/v5_layout.py')))
    predictions = []
    model.eval().requires_grad_(False)
    with (out / 'predictions.jsonl').open('x') as stream:
        for row in rows:
            # Known unsupported inputs remain failed attempts in the official full denominator.
            try:
                features, mask = model_inputs(model, row)
                error = None
            except ValueError as exc:
                error = str(exc)
            for k in range(candidates):
                attempt_seed = seed + int(hashlib.sha256(row['complex_id'].encode()).hexdigest()[:8], 16) + k
                pred = dict(complex_id=row['complex_id'], candidate=k, seed=attempt_seed,
                            pred_sequence='', status='input_incompatible', failure_reason=error)
                if error is None:
                    sequence, audit = sample_h3(model, features, mask, attempt_seed, steps, temperature)
                    pred.update(pred_sequence=sequence, status='ok', audit=audit)
                predictions.append(pred)
                stream.write(json.dumps(pred, allow_nan=False) + '\n'); stream.flush()
            print('CHIMERA H3', row['complex_id'], pred['status'], flush=True)
    result = score(rows, predictions, candidates)
    result.update(native_identity=identity, numbering_scheme=scheme, split='epitope_group', subset='test',
                  status='complete', dataset_sha256=manifest['dataset_sha256'])
    write(out / 'metrics.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'preflight', 'run', 'score'])
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--scheme', choices=['imgt', 'chothia'], default='imgt')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--candidates', type=int, default=1)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'prepare':
        _, manifest = prepare(args.raw, args.data, args.scheme)
        print(json.dumps({k: manifest[k] for k in ('counts', 'removed_rows', 'input_errors', 'dataset_sha256')}))
    elif args.command == 'preflight':
        # Layout-only validation; no checkpoint or model performance is evaluated.
        from types import SimpleNamespace
        from dllm.pipelines.immune_llada.v3.model import ImmuneConfig, ImmuneModel
        from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
        from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
        rows, manifest = load_dataset(args.data, args.scheme)
        validation = dict(protocol=PROTOCOL, dataset_sha256=manifest['dataset_sha256'],
                          model_inference=False, test_rows=len(rows), layouts={})
        for name, config in [('object302', ImmuneConfig(ablation='V5', full_capacity=302)),
                             ('fixed_chain_187_154', FixedChainConfig())]:
            stub = SimpleNamespace(config=config, tokenizer=ImmuneCollator.from_config(config).tokenizer,
                                   feature_args=ImmuneModel.feature_args)
            errors = []
            for row in rows:
                try:
                    features, mask = model_inputs(stub, row)
                    changed = {**row, 'heavy': row['heavy'][:row['h3_start']] +
                               'G' * len(row['native_h3']) + row['heavy'][row['h3_end']:],
                               'native_h3': 'changed_reference', 'h3_contact_indices': []}
                    other, _ = model_inputs(stub, changed)
                    if any(not torch.equal(features[k], other[k]) for k in features):
                        raise AssertionError('Reference leaked into features: ' + row['complex_id'])
                except ValueError as exc:
                    errors.append(dict(complex_id=row['complex_id'], reason=str(exc)))
            validation['layouts'][name] = dict(compatible=len(rows) - len(errors), input_errors=errors)
        oracle = [dict(complex_id=r['complex_id'], candidate=0, pred_sequence=r['native_h3'], status='ok') for r in rows]
        result = score(rows, oracle)
        if result['summary']['best_aar']['mean'] != 1.0:
            raise AssertionError('Native sequence recovery oracle failed')
        expected_caar = sum(bool(r['h3_contact_indices']) for r in rows) / len(rows)
        if not np.isclose(result['summary']['best_caar']['mean'], expected_caar):
            raise AssertionError('Official empty-contact convention changed')
        validation.update(native_oracle_passed=True, zero_contact_complexes=sum(not r['h3_contact_indices'] for r in rows))
        write((args.out_dir or args.data) / (args.scheme + '.preflight.json'), validation)
        print(json.dumps(validation))
    elif args.command == 'run':
        if args.checkpoint is None or args.out_dir is None:
            parser.error('run requires --checkpoint and --out-dir')
        if not torch.cuda.is_available():
            raise RuntimeError('Actual checkpoint evaluation requires a Volc GPU job')
        from downstream.v5_native_eval import load
        model, identity = load(args.checkpoint)
        run(args.out_dir, model, identity, args.data, args.scheme, args.candidates,
            args.seed, args.steps, args.temperature)
    else:
        if args.predictions is None or args.out_dir is None:
            parser.error('score requires --predictions and --out-dir')
        rows, manifest = load_dataset(args.data, args.scheme)
        predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
        result = score(rows, predictions, args.candidates)
        result.update(dataset_sha256=manifest['dataset_sha256'], predictions_sha256=sha(args.predictions),
                      numbering_scheme=args.scheme, split='epitope_group', subset='test')
        write(args.out_dir / 'metrics.json', result)


if __name__ == '__main__':
    main()
