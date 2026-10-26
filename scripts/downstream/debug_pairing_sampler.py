"""Activate pllm/protenix_abtcr; python -B SCRIPT --help.

Controlled pairing diagnostic: fixed original inputs/checkpoints, p0/reference,
16 steps, 8 candidates, original row-based seeds. Only the sampler is exchanged.
Every attempt uses a fresh output folder; historical runs are read-only.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
GEN = ROOT / 'output/downstream_generation'
CHECKPOINTS = {
    'fixed50000': GEN / 'fixed_chain_75m_step50000_20261013/checkpoint',
    'fixed112000': GEN / 'fixed_chain_75m_best112000_20261014/checkpoint',
    'v7': ROOT / 'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/run/checkpoint-50082',
}
ORIGINALS = {
    'fixed50000': GEN / 'fixed_chain_75m_step50000_pairing_reference_20261013/predictions.jsonl',
    'fixed112000': GEN / 'fixed_chain_75m_best112000_20261014/pairing-reference/predictions.jsonl',
    'v7': GEN / 'v7_f011_step50082_pairing_local_gpu_20261006/predictions.jsonl',
}
BENCHMARK = ROOT / 'data/oas_previous_clean/splits/holdout_select_500_eval_OAS.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def model_input(model, name, heavy, length):
    if name == 'v7':
        from downstream.v7_pairing import pairing_input
        return pairing_input(model, heavy, length)[:2]
    # Import only after the checkpoint-bound implementation has been loaded.
    from scripts.downstream.run_fixed_chain_pairing_reference import reference_input
    return reference_input(model, heavy, length)[:2]


def sample(model, features, target, mode, seed, trace=False):
    """Compare legacy noisy-margin with sampled-probability commitment.

    gumbel_prob uses the legacy proposal algorithm and random-number stream,
    but ranks commits by unperturbed model probability. Proposals can diverge
    after changed commitments alter the next input. native uses pending-only
    multinomial.
    All modes finish the same L target positions; early terminals are parsed
    afterwards and retained. No reference sequence is accepted by this API.
    """
    import torch
    from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import _sample_tokens, _confidence_decoding
    device = next(model.parameters()).device
    native = {k: v.to(device).clone() for k, v in features.items()}
    target = target.to(device)
    ids, pending = native['input_ids'].clone(), target.clone()
    total = int(target.sum())
    assert total and ids[target].eq(model.tokenizer.mask_token_id).all()
    rng = torch.Generator(device=device).manual_seed(seed)
    devices = [] if device.type == 'cpu' else [device.index or 0]
    history = []
    precision = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with torch.random.fork_rng(devices=devices), torch.inference_mode(), precision:
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        for step in range(1, 17):
            logits, _ = model.denoise(**{**native, 'input_ids': ids})
            probabilities = logits.float().softmax(-1)
            before = pending.clone()
            if mode == 'native':
                positions = pending[0].nonzero().flatten()
                probs = probabilities[0, positions]
                prediction = torch.multinomial(probs, 1, generator=rng).squeeze(1)
                confidence = probs.gather(1, prediction[:, None]).squeeze(1)
                remaining = total * (16-step) // 16
                chosen = confidence.topk(len(positions)-remaining).indices
                ids[0, positions[chosen]] = prediction[chosen]
                pending = target & ids.eq(model.tokenizer.mask_token_id)
            else:
                proposal, confidence = _sample_tokens(logits, 'gumbel_argmax', 1.)
                if mode == 'gumbel_prob':
                    confidence = probabilities.gather(-1, proposal[..., None]).squeeze(-1)
                pending, ids, _ = _confidence_decoding(ids, torch.zeros_like(ids, dtype=torch.float32),
                    proposal, confidence, 'confidence-deterministic-linear', pending, target, step, 16)
            committed = before & ~pending
            assert model.allowed_target[ids[committed]].all()
            assert torch.equal(ids[~target], native['input_ids'][~target])
            if trace:
                p = probabilities[committed]
                chosen_ids = ids[committed]
                history.append(dict(step=step, committed_positions=committed[0].nonzero().flatten().tolist(),
                    token_ids=chosen_ids.tolist(), pending=int(pending.sum()),
                    committed_argmax_fraction=float(chosen_ids.eq(p.argmax(-1)).float().mean()),
                    committed_mean_probability=float(p.gather(-1, chosen_ids[:, None]).mean())))
            if not pending.any():
                break
    assert not pending.any()
    tokens = ids[target].tolist()
    terminal = model.terminal_id
    end = tokens.index(terminal) if terminal in tokens else len(tokens)
    return dict(sequence=''.join(model.tokenizer.token(t) for t in tokens[:end]),
                raw_light_token_ids=tokens, length=end, early_terminal=end < len(tokens), trace=history)


def checks(model, name, frame, device):
    """Compare old and new code on identical logits/inputs before quality runs."""
    import torch
    from downstream.v7_pairing import pairing_decode
    row = frame.iloc[0]
    heavy, light = row.h_sequence.replace('-', ''), row.l_sequence.replace('-', '')
    features, target = model_input(model, name, heavy, len(light))
    alternate, other_target = model_input(model, name, heavy, len('G'*len(light)))
    assert all(torch.equal(v, alternate[k]) for k, v in features.items())
    assert torch.equal(target, other_target)
    if name == 'v7':
        old_ids, _ = pairing_decode(model, features, target, seed=42)
        old = old_ids[target.to(old_ids.device)].tolist()
        expected = sample(model, features, target, 'legacy', 42, trace=True)
        assert old == expected['raw_light_token_ids'], 'Legacy decoder not reproduced'
    else:
        from scripts.downstream.run_fixed_chain_pairing_reference import reference_input, decode_native
        f, t, positions = reference_input(model, heavy, len(light))
        old = decode_native(model, f, t, positions, seed=42)
        expected = sample(model, features, target, 'native', 42, trace=True)
        assert old['sequence'] == expected['sequence'], 'Native decoder not reproduced'
    original = json.loads(ORIGINALS[name].open().readline())
    historical = original['sequence'] if name == 'v7' else original['gen_l_sequence']
    gpu_reproduction = expected['sequence'] == historical
    if device == 'cuda':
        assert gpu_reproduction, 'GPU first historical candidate does not reproduce'
    return dict(status='passed', hidden_reference_invariant=True,
                original_decoder_exact=True, historical_gpu_sequence_exact=gpu_reproduction,
                historical_gpu_check_required=device == 'cuda', sample=expected,
                input_shape=list(features['input_ids'].shape), target_count=int(target.sum()))


def score(output):
    import generation_eval as evaluator
    observed = {}
    original = evaluator.compute_immunomatch_score
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        observed[kwargs['lseq_col']] = [float(v) for v in result]
        return result
    evaluator.compute_immunomatch_score = capture
    try:
        metrics = evaluator.main(str(output / 'generated.csv'), 'gen_l_sequence', 'raw_l_sequence',
                                 detailed_comparison=True, heavy_col_name='h_sequence', expected_count=8)
    finally:
        evaluator.compute_immunomatch_score = original
    assert metrics['total_pairs'] == len(observed['gen_l_sequence'])
    for column, key in [('gen_l_sequence', 'gen_immunomatch_mean'), ('raw_l_sequence', 'ref_immunomatch_mean')]:
        values = observed[column]
        assert len(values) == metrics['total_pairs'] and all(0 <= v <= 1 for v in values)
        assert abs(sum(values)/len(values)-metrics[key]) < 1e-10
    write(output/'immunomatch_scores.json', observed)
    write(output/'metrics.json', metrics)
    print('SCORED', json.dumps({k:metrics[k] for k in ['gen_immunomatch_mean','diversity_mean','gen_valid_rate']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=CHECKPOINTS, required=True)
    p.add_argument('--sampler', choices=['legacy','native','gumbel_prob'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--limit', type=int, default=500, help='First N original heavies; diagnostic only if N<500')
    p.add_argument('--expected-code-sha')
    args = p.parse_args()
    if args.expected_code_sha:
        assert sha(__file__) == args.expected_code_sha, 'Diagnostic runner changed'
    assert 1 <= args.limit <= 500
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(status='loading', model=args.model, sampler=args.sampler,
                 device=args.device, started_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/'status.json', state)
    try:
        import torch
        import pandas as pd
        torch.set_num_threads(4)
        if args.model == 'v7':
            from downstream.v7_native import load_native_checkpoint
            model, identity = load_native_checkpoint(CHECKPOINTS[args.model], args.device,
                                                    allow_snapshot_relocations=True)
        else:
            from downstream.fixed_chain_eval import load
            model, identity = load(CHECKPOINTS[args.model], args.device)
        frame = pd.read_csv(BENCHMARK, keep_default_na=False)
        assert len(frame) == 500
        from dllm.pipelines.bioseq.ophiuchus import sampling
        from dllm.pipelines.qwen3_vl_arch import sampling_bioseq
        sources = [Path(__file__), Path(sampling.__file__), Path(sampling_bioseq.__file__),
                   ROOT/'downstream/v7_pairing.py', ROOT/'scripts/downstream/run_fixed_chain_pairing_reference.py',
                   ROOT/'downstream/fixed_chain_eval.py', ROOT/'downstream/v7_native.py',
                   ROOT/'downstream/fixed_chain_pairing.py']
        scorer_root = ROOT/'downstream/comp_chain/eval_scripts'
        sources += [scorer_root / n for n in ['generation_eval.py','immunomatch_score.py','parse_anarci.py']]
        write(args.output/'manifest.json', dict(**state, checkpoint=identity, benchmark_sha256=sha(BENCHMARK),
              sources={str(f):sha(f) for f in sources}, count=args.limit*8,
              protocol=dict(prefix=0, reference_length=True, steps=16, temperature=1., cfg=0.,
                            seed='42+8*original_row+variant', finish_all_reference_positions=True)))
        write(args.output/'checks.json', checks(model, args.model, frame, args.device))
        if args.check_only:
            state['status']='checks_passed'
        else:
            rows=[];started=time.monotonic()
            with (args.output/'predictions.jsonl').open('x') as stream:
                for i, row in frame.iloc[:args.limit].iterrows():
                    heavy, ref = row.h_sequence.replace('-', ''), row.l_sequence.replace('-', '')
                    features, target = model_input(model, args.model, heavy, len(ref))
                    for variant in range(8):
                        result = sample(model, features, target, args.sampler, 42+8*i+variant, trace=i<2)
                        record=dict(row_index=int(i), variant_idx=variant, h_sequence=heavy,
                                    raw_l_sequence=ref, gen_l_sequence=result['sequence'], raw=result)
                        stream.write(json.dumps(record)+'\n');stream.flush();rows.append(record)
                    write(args.output/'progress.json',dict(heavy_completed=int(i)+1, candidates=len(rows),
                          elapsed_seconds=time.monotonic()-started))
                    print('GENERATED',args.model,args.sampler,i+1,'/',args.limit,flush=True)
            assert len(rows) == args.limit*8
            pd.DataFrame([{k:v for k,v in r.items() if k!='raw'} for r in rows]).to_csv(args.output/'generated.csv',index=False)
            del model
            if args.device=='cuda':torch.cuda.empty_cache()
            sys.path.insert(0,str(scorer_root))
            os.chdir(args.output)
            score(args.output)
            state['status']='success'
        state['completed_utc']=datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        state.update(status='failed', error=repr(error));raise
    finally:
        write(args.output/'status.json',state)


if __name__ == '__main__':
    main()
