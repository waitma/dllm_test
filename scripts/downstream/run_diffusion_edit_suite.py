"""Evaluate an immutable diffusion/edit checkpoint on existing AB/TCR tasks.

Activate pllm/protenix_abtcr; python -B THIS_FILE --run ABS --check --device cpu
or --task TASK. Use run_local_diffusion_edit.py for sequential local execution.
Generation and representation use separate processes and retain raw results.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
from downstream.diffusion_edit_eval import (
    DiffusionBackbone, check, load, sequence_vocabulary, sha)
from scripts.downstream.run_bert_representation_suite import report, verify, write


def invoke(module, *arguments):
    original = sys.argv
    try:
        sys.argv = [module.__file__, *arguments]
        module.main()
    finally:
        sys.argv = original


def evaluator(run, model, identity):
    """Use the unchanged AirGen evaluator with native 167/135 coordinates."""
    path = run / 'airgen_evaluate.py'
    spec = importlib.util.spec_from_file_location('_diffedit_cdr', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from downstream import fixed_chain_eval, v5_native_eval
    v5_native_eval.load = lambda *args, **kwargs: (model, identity)
    module.inputs = fixed_chain_eval.airgen_inputs
    module.CKPT, module.OUT = run / 'checkpoint', run / 'cdr'
    module.report = lambda: None
    return module


def pairing_gate(model, frame, out, device):
    """Check all layouts and actual sampler parity, without a quality threshold."""
    import torch
    from scripts.downstream.run_fixed_chain_pairing_reference import reference_input
    from downstream.fixed_chain_pairing import decode
    from scripts.downstream.debug_pairing_sampler import sample
    from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
    for row in frame.itertuples():
        heavy, light = row.h_sequence.replace('-', ''), row.l_sequence.replace('-', '')
        features, target, positions = reference_input(model, heavy, len(light))
        alternate, target2, _ = reference_input(model, heavy, len('G' * len(light)))
        assert all(torch.equal(value, alternate[key]) for key, value in features.items())
        assert torch.equal(target, target2) and int(target.sum()) == len(light)
        h = (features['role_ids'][0].eq(CHAIN_ROLE_TO_ID['antibody_heavy'])
             & features['position_ids'][0].le(len(heavy)))
        assert features['input_ids'][0, h].tolist() == model.tokenizer.encode_residues(heavy)
        assert len(positions) == model.config.second_chain_capacity
    row = frame.iloc[0]
    f, target, positions = reference_input(
        model, row.h_sequence.replace('-', ''), len(row.l_sequence.replace('-', '')))
    actual = decode(model, f, target, positions, seed=42)
    expected = sample(model, f, target, 'legacy', 42)
    assert actual['raw_light_token_ids'][:int(target.sum())] == expected['raw_light_token_ids']
    assert (actual.get('sequence') or '') == expected['sequence']
    assert actual == decode(model, f, target, positions, seed=42)
    write(out / f'{device}_sampler_alignment.json', dict(
        status='passed', device=device, records=len(frame), actual_checkpoint=True,
        matches_validated_debug_sampler=True, deterministic=True,
        hidden_reference_letters_invariant=True, sample_status=actual['status']))


def generation_checks(run, device):
    import pandas as pd
    import torch
    from downstream.fixed_chain_eval import generate
    from dllm.pipelines.immune_llada.data.records import BioSeqChain
    plan = verify(run)
    model, identity = load(run / 'checkpoint', device)
    before = model.allowed_target.clone()
    with sequence_vocabulary(model):
        module = evaluator(run, model, identity)
        if device == 'cpu':
            invoke(module, '--audit')
            # Correct the inherited model-description text, not the data/sampler.
            audit = json.loads((run / 'cdr/protocol_audit.json').read_text())
            audit['differences'][0] = 'native fixed-chain 167/135 layout'
            write(run / 'cdr/protocol_audit.json', audit)
            invoke(module, '--check')
        else:
            # Real CUDA CDR forward, with the same no-answer/context invariants.
            rows, _ = module.records('sab23')
            features, batch, _, target, maps = module.inputs(model, rows[:2])
            features = {k: v.to(device) for k, v in features.items()}
            batch = {k: v.to(device) for k, v in batch.items()}
            target = target.to(device)
            bridge = module.decoder(model, features, maps)
            with torch.inference_mode():
                tokens, _ = bridge.generate(
                    batch, max_iter=2, temperature=1.,
                    partial_masks=~target & batch['input_ids'].ne(bridge.bos_id),
                    sampling_strategy='argmax', cfg_scale=0.)
            assert torch.equal(tokens[~target], batch['input_ids'][~target])
            assert not tokens[target].eq(model.tokenizer.mask_token_id).any()
        frame = pd.read_csv(plan['benchmark']['path'], keep_default_na=False)
        assert len(frame) == 500
        pairing_gate(model, frame, run / 'pairing', device)
        entries = json.loads(Path(plan['tcr']['data']).read_text())
        entry = next(iter(entries.values()))
        conditions = [BioSeqChain(entry['mhc_pseudo'], 'mhc'),
                      BioSeqChain(entry['epitope'], 'peptide')]
        result = generate(model, conditions, ['tcr_beta'], 42, steps=32, single_chain=True)
        assert len(result['chains']) == 1
        assert result == generate(model, conditions, ['tcr_beta'], 42, steps=32, single_chain=True)
    assert torch.equal(before, model.allowed_target)
    write(run / f'{device}_generation_checks.json', dict(
        status='passed', model=identity, generation_vocabulary_restored=True,
        tcr_reference_not_supplied=True, tcr_deterministic=True,
        tcr_sample_status=result['status'], pairing_records=500,
        cdr_forward=True, device=device))


def representation(run, task):
    """Install the explicit diffusion loader into the existing task runner."""
    plan = verify(run)
    model, identity = load(run / 'checkpoint')
    assert identity['model_sha256'] == plan['model_sha256']
    from downstream import v5_native_eval as native
    native.load = lambda *args, **kwargs: (model, identity)
    native.NativeBackbone = DiffusionBackbone
    from scripts.downstream import report_v5_suite, run_v5_suite_task as runner
    report_v5_suite.update = report
    old_install, old_probes = runner.install, runner.probes

    def install(backbone):
        binding = old_install(backbone)
        binding.OURS_METHOD = 'Ours diffusion/edit V7 75.6M frozen observed-mean + MLP'
        return binding

    def probes(name, checkpoint, out, backbone):
        old_probes(name, checkpoint, out, backbone)
        manifest = json.loads((out / 'run_manifest.json').read_text())
        manifest['model'] = plan['model']
        write(out / 'run_manifest.json', manifest)
        metrics = json.loads((out / 'metrics.json').read_text())
        metrics.update(model=plan['model'], run_manifest_sha256=sha(out / 'run_manifest.json'))
        write(out / 'metrics.json', metrics)

    runner.install, runner.probes = install, probes
    invoke(runner, '--run', str(run), '--task', task)


def tcr_generation(run, task, model, identity):
    """Preserve the established native full-beta protocol and every candidate."""
    from downstream.fixed_chain_eval import generate
    from dllm.pipelines.immune_llada.data.records import BioSeqChain
    from downstream.benchmark.tcr_generation_bench.scoring import (
        extract_cdr3b_anarci, is_valid_cdr3b)
    from downstream.v7_tcr_scoring import score_saved_candidates
    plan = verify(run)
    category = task.removeprefix('tcr-')
    entries = {k: v for k, v in json.loads(Path(plan['tcr']['data']).read_text()).items()
               if v['category'] == category}
    assert len(entries) == (14 if category == 'benchmark14' else 20)
    out = run / task
    designs, greedy = {}, {}
    with (out / 'designs.jsonl').open('x') as stream:
        for key, entry in entries.items():
            if (run / 'STOP').exists():
                raise RuntimeError('Stop requested')
            conditions = [BioSeqChain(entry['mhc_pseudo'], 'mhc'),
                          BioSeqChain(entry['epitope'], 'peptide')]
            raw = []
            for n in range(101):
                result = generate(model, conditions, ['tcr_beta'], 42 + n,
                                  steps=32, greedy=n == 100, single_chain=True)
                result['protocol'] = 'diffedit_fixed167_beta_eos_native32'
                result.update(seed=42 + n, candidate_index=n)
                raw.append(result)
            sequences = [r['chains'][0] if r['status'] == 'ok' else '' for r in raw]
            extracted = extract_cdr3b_anarci(sequences)
            for item, sequence in zip(raw, extracted):
                valid = bool(sequence and is_valid_cdr3b(sequence))
                item.update(sequence=sequence if valid else '', valid=valid)
            designs[key] = [r['sequence'] for r in raw[:100]]
            greedy[key] = raw[100]['sequence']
            stream.write(json.dumps(dict(
                pmhc=key, sequences=designs[key], greedy=greedy[key],
                candidates=raw[:100], greedy_raw=raw[100], n_requested=100,
                n_valid=sum(r['valid'] for r in raw[:100]))) + '\n')
            stream.flush()
            write(out / 'progress.json', dict(targets_completed=len(designs),
                                               targets=len(entries), candidates=len(designs) * 101))
            print('TCR', category, len(designs), '/', len(entries), flush=True)
    saved = [json.loads(line) for line in (out / 'designs.jsonl').read_text().splitlines()]
    assert [row['pmhc'] for row in saved] == list(entries)
    for row in saved:
        assert len(row['candidates']) == 100
        assert row['sequences'] == [r['sequence'] for r in row['candidates']]
        assert row['greedy'] == row['greedy_raw']['sequence']
    result = score_saved_candidates(designs, greedy, entries)
    result.update(identity=identity, dataset=category, protocol=plan['tcr'],
                  generation_validity={k: dict(requested=100, valid=sum(bool(s) for s in seqs))
                                       for k, seqs in designs.items()},
                  pretraining_overlap_audit='not established for this diffusion checkpoint')
    write(out / 'metrics.json', result)
    write(out / 'verification.json', dict(
        status='passed', raw_candidates=len(entries) * 101, all_attempts_retained=True,
        designs_sha256=sha(out / 'designs.jsonl'),
        nullable_metrics_have_explicit_denominators=True))


def generation(run, task):
    from scripts.downstream import run_oas_generation_eval as existing
    from downstream import fixed_chain_eval
    plan = verify(run)
    out = run / task
    if (out / 'status.json').exists():
        raise FileExistsError('Task already attempted: ' + task)
    write(out / 'status.json', dict(status='running', task=task,
                                     started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        model, identity = load(run / 'checkpoint')
        assert identity['model_sha256'] == plan['model_sha256']
        write(out / 'model_identity.json', identity)
        with sequence_vocabulary(model):
            if task == 'cdr':
                module = evaluator(run, model, identity)
                for dataset in ('kong', 'sab23'):
                    invoke(module, '--dataset', dataset)
                existing.audit_cdr(run, module, model.tokenizer)
            elif task == 'pairing':
                fixed_chain_eval.load = lambda *args, **kwargs: (model, identity)
                existing.pairing_gate = pairing_gate
                # Reuse generation, annotation, ImmunoMatch and independent aggregation.
                existing.pairing(run, plan)
            else:
                tcr_generation(run, task, model, identity)
        write(out / 'status.json', dict(status='success', task=task,
                                         finished_utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as error:
        write(out / 'status.json', dict(status='failed', task=task, error=repr(error),
                                         traceback=traceback.format_exc()))
        raise
    finally:
        report(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--task')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    plan = verify(run)
    if args.report:
        print(json.dumps(report(run))); return
    import torch
    torch.set_num_threads(4)
    if args.check:
        result = check(run / 'checkpoint', args.device)
        write(run / f'{args.device}_adapter_checks.json', result)
        generation_checks(run, args.device)
        print('DIFFUSION_EDIT_CHECKS_PASSED', args.device, flush=True); return
    if args.task not in plan['tasks']:
        parser.error('Choose a task in plan.json')
    for device in ('cpu', 'cuda'):
        for suffix in ('adapter_checks', 'generation_checks'):
            assert json.loads((run / f'{device}_{suffix}.json').read_text())['status'] == 'passed'
    if args.task in plan['generation_tasks']:
        generation(run, args.task)
    else:
        representation(run, args.task)


if __name__ == '__main__':
    main()
