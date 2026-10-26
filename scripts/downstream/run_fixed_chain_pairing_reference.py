"""Known-length pairing control; activate pllm then protenix_abtcr.

Run: python -B scripts/downstream/run_fixed_chain_pairing_reference.py --run ABS --check --device cpu
Full evaluation: same command without --check, with --device cuda.
Default decoder: AirGen Gumbel + noisy-margin confidence; requires a fresh v2 plan.
Only reference LENGTH is passed to the adapter, never reference residues.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))

from downstream.fixed_chain_pairing import decode, validate_plan


def write(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_input(model, heavy, light_length):
    """Keep native geometry; reveal EOS at L+1, mask only L residue targets."""
    import torch
    from dllm.pipelines.immune_llada.data.records import BioSeqChain, CHAIN_ROLE_TO_ID
    from dllm.pipelines.immune_llada.v3.layout import GenerationRequest
    from dllm.pipelines.immune_llada.fixed_chain.layout import FixedChainCollator
    if type(light_length) is not int or not 0 < light_length < model.config.second_chain_capacity:
        raise ValueError('Reference length must fit residues plus EOS')
    request = GenerationRequest((BioSeqChain(heavy, 'antibody_heavy'),), (('antibody_light', 'full'),))
    batch = FixedChainCollator(model.config, mode='generation')([request])
    features = model.feature_args(batch)
    positions = ((features['role_ids'][0] == CHAIN_ROLE_TO_ID['antibody_light'])
                 & features['attention_mask'][0]).nonzero().flatten()
    assert len(positions) == model.config.second_chain_capacity
    assert features['input_ids'][0, positions].eq(model.tokenizer.mask_token_id).all()
    features['input_ids'][0, positions[light_length]] = model.tokenizer.eos_token_id
    target = torch.zeros_like(features['attention_mask'])
    target[0, positions[:light_length]] = True
    return features, target, positions


def decode_native(model, features, target, positions, *, seed, steps=16):
    """Native multinomial/top-k schedule restricted to the known residue span.

Retain AA+EOS vocabulary and native first-EOS termination (early EOS possible).
Tail MASKs and the supplied EOS remain fixed. No crop/pad repair is applied.
"""
    import torch
    from dllm.pipelines.immune_llada.fixed_chain.generation import parse_chain
    device = next(model.parameters()).device
    features = {k: v.to(device) for k, v in features.items()}
    target, positions = target.to(device), positions.to(device)
    ids = features['input_ids'].clone()
    length = int(target.sum())
    assert length > 0 and ids[target].eq(model.tokenizer.mask_token_id).all()
    rng = torch.Generator(device=device).manual_seed(seed)
    precision = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with torch.inference_mode(), precision:
        for step in range(steps):
            logits, _ = model.denoise(**{**features, 'input_ids': ids})
            pending = (target[0] & ids[0].eq(model.tokenizer.mask_token_id)).nonzero().flatten()
            scores = logits[0, pending].float()
            allowed = model.allowed_target
            assert torch.isfinite(scores[:, allowed]).all()
            probs = scores.masked_fill(~allowed, float('-inf')).softmax(-1)
            prediction = torch.multinomial(probs, 1, generator=rng).squeeze(1)
            confidence = probs.gather(1, prediction[:, None]).squeeze(1)
            remaining = length * (steps - step - 1) // steps
            count = max(0, len(pending) - remaining)
            chosen = confidence.topk(count).indices
            ids[0, pending[chosen]] = prediction[chosen]
            if parse_chain(model.tokenizer, ids[0, positions])['status'] == 'ok':
                break
            if not (target & ids.eq(model.tokenizer.mask_token_id)).any():
                break
    assert torch.equal(ids[~target], features['input_ids'][~target]), 'Condition changed'
    parsed = parse_chain(model.tokenizer, ids[0, positions])
    return dict(**parsed, steps=step + 1, reference_length=length,
                early_eos=ids[0, positions].tolist().index(model.tokenizer.eos_token_id) < length,
                raw_light_token_ids=ids[0, positions].tolist(), conditions_unchanged=True)


def check(model, frame, run, device):
    """Verify all real layouts and deterministic actual forward without hidden residues."""
    import torch
    from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
    for row in frame.itertuples():
        heavy, light = row.h_sequence.replace('-', ''), row.l_sequence.replace('-', '')
        f, mask, pos = reference_input(model, heavy, len(light))
        alternate, alt_mask, _ = reference_input(model, heavy, len('G' * len(light)))
        assert all(torch.equal(v, alternate[k]) for k, v in f.items()) and torch.equal(mask, alt_mask)
        assert mask.sum().item() == len(light)
        h = f['role_ids'][0].eq(CHAIN_ROLE_TO_ID['antibody_heavy']) & f['position_ids'][0].le(len(heavy))
        assert f['input_ids'][0, h].tolist() == model.tokenizer.encode_residues(heavy)
        assert f['input_ids'][0, pos[len(light)]].item() == model.tokenizer.eos_token_id
        assert f['input_ids'][0, pos[len(light)+1:]].eq(model.tokenizer.mask_token_id).all()
        assert f['attention_mask'][0, pos].all()
    row = frame.iloc[0]
    f, target, pos = reference_input(model, row.h_sequence.replace('-', ''), len(row.l_sequence.replace('-', '')))
    a = decode(model, f, target, pos, seed=42)
    b = decode(model, f, target, pos, seed=42)
    assert a == b
    assert a['status'] == 'ok' and 0 < a['length'] <= a['reference_length']
    native_control = None
    if device == 'cuda':
        from downstream.fixed_chain_eval import generate
        from dllm.pipelines.immune_llada.data.records import BioSeqChain
        result = generate(model, [BioSeqChain(row.h_sequence.replace('-', ''), 'antibody_heavy')],
                          ['antibody_light'], 42, steps=16)
        plan = json.loads((run/'plan.json').read_text())
        with Path(plan['baseline_predictions']).open() as stream:
            base = json.loads(stream.readline())
        assert result['chains'][1] == base['gen_l_sequence'], 'Native baseline reproduction differs'
        native_control = 'first native EOS candidate reproduced exactly'
    report = dict(status='passed', device=device, records_checked=len(frame),
                  hidden_reference_letters_invariant=True, heavy_and_tail_preserved=True,
                  deterministic_actual_forward=True, native_control=native_control,
                  smoke_output=a)
    write(run/f'{device}_checks.json', report)
    print('CHECK', device, 'passed', flush=True)


def publish(run, metrics):
    """Append the completed control to both task tables under the shared lock."""
    name = (f"Ours fixed-chain 75.6M / {metrics['native_identity']['step']}, "
            'p0 / reference / iter=16 / 对齐V7采样器')
    values = [metrics[k] for k in ['gen_immunomatch_mean', 'gen_valid_rate', 'overall_chain_match_rate', 'diversity_mean']]
    with (ROOT/'output/downstream_generation/v5_native_best_20261008/results_update.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for filename in ['RESULTS.md', 'RESULTS_SUMMARY.md']:
            path = ROOT/'downstream/benchmark'/filename
            text = path.read_text()
            if name in text:
                continue
            label = name if filename == 'RESULTS_SUMMARY.md' else f'[{name}]({run}/metrics.json)'
            row = '| ' + label + ' | ' + ' | '.join(f'{x:.6f}' for x in values) + ' |\n'
            start = text.index('AB Light-chain pairing')
            table = text.index('\n|', start) + 1
            end = text.index('\n\n', table)
            text = text[:end] + '\n' + row.rstrip() + text[end:]
            temp = path.with_suffix('.reference.tmp'); temp.write_text(text); temp.replace(path)
        path = ROOT/'downstream/benchmark/PROGRESS.md'
        with path.open('a') as f:
            f.write(f'\n\n{datetime.now(timezone.utc).isoformat()} {name} 完成，4000候选评分后已核对结果表。[原件]({run}/metrics.json)。\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--check', action='store_true')
    args = p.parse_args(); run = args.run.resolve()
    plan = json.loads((run/'plan.json').read_text())
    validate_plan(plan)
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected, 'Source changed: ' + path
    assert sha(plan['benchmark']['path']) == plan['benchmark']['sha256']
    import pandas as pd
    from downstream.fixed_chain_eval import load
    model, identity = load(plan['checkpoint'], args.device)
    assert identity['step'] == plan['checkpoint_step'] and identity['model_sha256'] == plan['model_sha256']
    frame = pd.read_csv(plan['benchmark']['path'], keep_default_na=False)
    assert len(frame) == 500
    check(model, frame, run, args.device)
    if args.check:
        return
    assert args.device == 'cuda'
    write(run/'status.json', dict(status='running', started_utc=datetime.now(timezone.utc).isoformat()))
    rows = []
    try:
        with (run/'predictions.jsonl').open('x') as out:
            for i, row in frame.iterrows():
                heavy, ref = row.h_sequence.replace('-', ''), row.l_sequence.replace('-', '')
                features, target, positions = reference_input(model, heavy, len(ref))
                for variant in range(8):
                    result = decode(model, features, target, positions, seed=42 + 8*i + variant)
                    item = dict(row_index=int(i), variant_idx=variant, h_sequence=heavy,
                                raw_l_sequence=ref, gen_l_sequence=result.get('sequence') or '', raw=result)
                    rows.append(item); out.write(json.dumps(item) + '\n'); out.flush()
                write(run/'progress.json', dict(heavy_completed=int(i)+1, candidate_completed=len(rows),
                      updated_utc=datetime.now(timezone.utc).isoformat()))
                print('pairing reference', i+1, '/ 500', flush=True)
        assert len(rows) == 4000 and len({(r['row_index'], r['variant_idx']) for r in rows}) == 4000
        tok = model.tokenizer
        for r in rows:
            ids = r['raw']['raw_light_token_ids']; end = ids.index(tok.eos_token_id)
            assert ids[len(r['raw_l_sequence'])] == tok.eos_token_id
            assert len(ids) == model.config.second_chain_capacity
            assert all(x == tok.mask_token_id for x in ids[len(r['raw_l_sequence'])+1:])
            if r['raw']['status'] == 'ok':
                assert r['gen_l_sequence'] == ''.join(tok.token(x) for x in ids[:end])
        csv = run/'generated.csv'
        pd.DataFrame([{k: v for k, v in r.items() if k != 'raw'} for r in rows]).to_csv(csv, index=False)
        write(run/'generation_audit.json', dict(status='passed', candidates=4000, all_candidates_retained=True,
              hidden_reference_sequence_never_passed=True, supplied_end_and_tail_preserved=True,
              early_eos_candidates=sum(r['raw']['early_eos'] for r in rows),
              predictions_sha256=sha(run/'predictions.jsonl')))
        del model
        import torch
        torch.cuda.empty_cache()
        from downstream.grammar.light_chain_pairing import run_comp_chain_eval
        metrics = run_comp_chain_eval(csv, 8)
        assert metrics['total_pairs'] == 4000
        metrics.update(native_identity=identity, evaluation_protocol=plan['protocol'],
                       candidate_attempts=4000, length_condition='reference; visible EOS at L+1; no light residues supplied')
        write(run/'metrics.json', metrics)
        base = json.loads(Path(plan['baseline_metrics']).read_text())
        keys = ['gen_immunomatch_mean', 'gen_valid_rate', 'overall_chain_match_rate', 'diversity_mean', 'gen_length_mean', 'ref_immunomatch_mean']
        write(run/'comparison.json', {k: dict(native_eos=base[k], reference=metrics[k], difference=metrics[k]-base[k]) for k in keys})
        publish(run, metrics)
        write(run/'status.json', dict(status='success', completed_utc=datetime.now(timezone.utc).isoformat()))
        print('COMPLETE', json.dumps({k: metrics[k] for k in keys}), flush=True)
    except BaseException as e:
        write(run/'status.json', dict(status='failed', error=repr(e)))
        raise


if __name__ == '__main__':
    main()
