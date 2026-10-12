"""Bounded whole-model synthetic capacity/throughput probe; never a data run.

Activate conda pllm. CPU only: python -B /absolute/repo/scripts/research/
profile_v7_large_batch.py --self-test. Future single-GPU invocation:
python -I -B /absolute/profile_v7_large_batch.py --source-root /absolute/frozen
--output-dir /absolute/new-profile. Defaults: f000/f111/cv_segment_global,
micro4/16/32, global32, two warmup + six measured updates per candidate (72
updates total, hard cap100), per-candidate180s, total1800s. Each candidate is a
fresh subprocess on the inherited physical GPU. No data, sequences, logits,
checkpoints, queue changes or platform submission. OOM is a capacity result;
other errors/timeouts fail with preserved evidence, without automatic retry.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

VARIANTS = ('f000', 'f111', 'cv_segment_global')
MICROS = (4, 16, 32)
CONFIG = dict(ablation='A4', d_model=768, n_layers=8, n_heads=12, mlp_hidden_size=3072,
              max_length=2048, full_capacity=256, fragment_capacity=64)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--micro-batches', nargs='+', type=int, choices=MICROS, default=list(MICROS))
    parser.add_argument('--global-batch', type=int, choices=(32,), default=32)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--measure-steps', type=int, default=6)
    parser.add_argument('--candidate-seconds', type=int, default=180)
    parser.add_argument('--max-seconds', type=int, default=1800)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--candidate-variant', choices=VARIANTS)
    parser.add_argument('--candidate-micro', type=int, choices=MICROS)
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    require(all(path is not None and path.is_absolute() and path == path.resolve()
                for path in (args.source_root, args.output_dir)), 'Canonical absolute source/output paths required')
    require(len(set(args.variants)) == len(args.variants) and len(set(args.micro_batches)) == len(args.micro_batches), 'Duplicate candidates forbidden')
    require(args.warmup_steps >= 2 and args.measure_steps >= 2 and args.seed >= 0, 'At least two warmup/measured updates required')
    require(len(args.variants)*len(args.micro_batches)*(args.warmup_steps+args.measure_steps) <= 100, 'Total optimizer update cap is 100')
    require(0 < args.candidate_seconds <= args.max_seconds <= 1800, 'Finite total budget is at most 1800 seconds')
    require((args.candidate_variant is None) == (args.candidate_micro is None), 'Candidate identity must be complete')
    require(args.candidate_variant is None or (args.candidate_variant in args.variants and args.candidate_micro in args.micro_batches),
            'Candidate must belong to the declared finite sweep')
    return args


@contextmanager
def interruption_handlers():
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(number, frame):
        raise InterruptedError('Profile interrupted by signal '+str(number))
    for number in previous:
        signal.signal(number, interrupted)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def cleanup_process(process):
    """Bounded owned-session cleanup; a second signal cannot strand the child."""
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in previous:
        signal.signal(number, signal.SIG_IGN)
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def integer_batch(micro, allowed_ids, terminal_id, roles=(1, 2, 3), device='cpu'):
    """2048 attended integer positions, long visible context, two full256 slots."""
    import torch
    length, prefix, capacity = CONFIG['max_length'], 8, CONFIG['full_capacity']
    context_end = length-2*capacity
    ids = allowed_ids.to(device)[torch.arange(micro*length, device=device).reshape(micro, length) % len(allowed_ids)]
    chain = torch.full_like(ids, -1)
    chain[:, prefix:context_end] = 0
    chain[:, context_end:context_end+capacity] = 1
    chain[:, context_end+capacity:] = 2
    positions = torch.zeros_like(ids)
    positions[:, prefix:context_end] = torch.arange(context_end-prefix, device=device)
    positions[:, context_end:context_end+capacity] = torch.arange(capacity, device=device)
    positions[:, context_end+capacity:] = torch.arange(capacity, device=device)
    role = torch.zeros_like(ids)
    for chain_id, role_id in enumerate(roles):
        role[chain == chain_id] = role_id
    targets = chain >= 1
    terminal = torch.zeros_like(targets)
    terminal[:, context_end+capacity-1] = True
    terminal[:, -1] = True
    ids[terminal] = terminal_id
    labels = ids.clone().masked_fill(~targets, -100)
    return dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(targets), role_ids=role,
        scope_ids=torch.zeros_like(ids), chain_ids=chain, position_ids=positions,
        target_slot_mask=targets, eligible_mask=targets.clone(), residue_mask=(chain >= 0) & ~terminal,
        terminal_mask=terminal, token_t=torch.ones_like(ids, dtype=torch.float32))


def source_hashes(root):
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(root):
                paths.add(path)
    return {str(path): sha(path) for path in sorted(paths)}


def candidate(args):
    sys.path.insert(0, str(args.source_root))
    import torch
    from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
    from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
    from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
    from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one inherited GPU required')
    require(not torch.distributed.is_initialized(), 'This is a single-GPU model probe')
    require(Path(sys.modules[FactorialModel.__module__].__file__).resolve().is_relative_to(args.source_root), 'Model source root differs')
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    properties = torch.cuda.get_device_properties(0)
    require('A100' in properties.name and 75 <= properties.total_memory/2**30 <= 85, 'This profile contract requires A100 80GB')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    micro, accum = args.candidate_micro, args.global_batch//args.candidate_micro
    result = dict(kind='synthetic_whole_model_capacity', variant=args.candidate_variant, micro_batch=micro,
        grad_accum=accum, global_batch=32, config=CONFIG, gpu=properties.name, total_gib=properties.total_memory/2**30,
        precision='fp32_parameters_adam_bf16_autocast', activation_checkpointing='whole_layer', seed=args.seed,
        max_grad_norm=1.0, optimizer='AdamW', learning_rate=1e-4,
        shape=[micro, 2048], attended_positions_per_record=2048, target_positions_per_record=512,
        target_terminal_positions_per_record=2, corruption='fixed100percent_target_mask',
        biological_quality_claim=False, actual_corpus_test=False, warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps, completed_optimizer_steps=0, measured_losses=[], phase='construction')
    model = optimizer = batch = None
    started = time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sources = source_hashes(args.source_root)
    result['free_gib_before'] = torch.cuda.mem_get_info()[0]/2**30
    try:
        model = FactorialModel(ImmuneConfig(**CONFIG), args.candidate_variant).to('cuda').train()
        result['config'] = asdict(model.config)
        model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
        result['model_parameters'] = sum(parameter.numel() for parameter in model.parameters())
        require(all(parameter.dtype == torch.float32 for parameter in model.parameters()), 'Parameter precision differs')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        allowed = model.allowed_target.nonzero().flatten()
        allowed = allowed[allowed != model.terminal_id]
        batch = integer_batch(micro, allowed, model.terminal_id,
            tuple(CHAIN_ROLE_TO_ID[name] for name in ('antigen', 'antibody_heavy', 'antibody_light')), 'cuda')
        def update():
            require(time.monotonic()-started < args.candidate_seconds, 'Candidate cooperative time budget expired')
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for _ in range(accum):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs['loss']/accum
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite synthetic loss')
                loss.backward()
                loss_sum += float(loss.detach())
                del outputs, loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            result['completed_optimizer_steps'] += 1
            return loss_sum
        result['phase'] = 'warmup'
        for _ in range(args.warmup_steps):
            update()
        require(bool(optimizer.state), 'AdamW state was not established')
        result['optimizer_state_tensor_bytes'] = sum(value.numel()*value.element_size()
            for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))
        result['warmup_peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
        result['warmup_peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
        torch.cuda.reset_peak_memory_stats()
        result['phase'] = 'measurement'
        measured_start = time.monotonic()
        for _ in range(args.measure_steps):
            result['measured_losses'].append(update())
        seconds = time.monotonic()-measured_start
        result.update(status='passed', measured_seconds=seconds,
            input_tokens_per_second=32*2048*args.measure_steps/seconds,
            target_tokens_per_second=32*512*args.measure_steps/seconds,
            records_per_second=32*args.measure_steps/seconds,
            throughput_scope='preallocated integer tensors; forward/backward/AdamW/sync; excludes collation and H2D')
    except torch.cuda.OutOfMemoryError as error:
        result.update(status='oom', expected_capacity_failure=True, error=str(error))
    except BaseException:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        result['peak_allocated_gib'] = max(result.get('warmup_peak_allocated_gib', 0), torch.cuda.max_memory_allocated()/2**30)
        result['peak_reserved_gib'] = max(result.get('warmup_peak_reserved_gib', 0), torch.cuda.max_memory_reserved()/2**30)
        result['elapsed_seconds'] = time.monotonic()-started
        result['source_sha256'] = source_hashes(args.source_root)
        require(all(result['source_sha256'].get(path) == value for path, value in sources.items()), 'Source changed during model profile')
        batch = optimizer = model = None
        gc.collect()
        torch.cuda.empty_cache()
        result['allocated_gib_after_cleanup'] = torch.cuda.memory_allocated()/2**30
        write_json(args.output_dir/'measurement.json', result)
        print(json.dumps(result, allow_nan=False), flush=True)
    return result


def recommendations(results, variants):
    per_variant = {}
    for variant in variants:
        passed = [item for item in results if item['variant'] == variant and item['status'] == 'passed'
            and item['peak_allocated_gib'] <= .85*item['total_gib'] and item['peak_reserved_gib'] <= .85*item['total_gib']]
        fastest = max(passed, key=lambda item: item['input_tokens_per_second']) if passed else None
        per_variant[variant] = None if fastest is None else {
            'micro_batch': fastest['micro_batch'], 'grad_accum': fastest['grad_accum'],
            'synthetic_input_tokens_per_second': fastest['input_tokens_per_second']}
    common = [micro for micro in MICROS if all(any(item['variant'] == variant and item['micro_batch'] == micro
        and item['status'] == 'passed' and item['peak_allocated_gib'] <= .85*item['total_gib']
        and item['peak_reserved_gib'] <= .85*item['total_gib'] for item in results) for variant in variants)]
    return dict(per_variant_fastest_with_15percent_memory_headroom=per_variant,
        largest_common_synthetic_capacity_micro=max(common) if common else None,
        observed_oom_micro_batches={variant: [item['micro_batch'] for item in results
            if item['variant'] == variant and item['status'] == 'oom'] for variant in variants},
        synthetic_only=True, actual_data_batch_change_authorized=False,
        note='Capacity/timing on one artificial full-length shape, not corpus learning or biological performance.')


def run(args):
    require(args.candidate_variant is None, 'Orchestrator cannot masquerade as a candidate')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require(all(path.name in ('train.log', 'worker.log', 'stdout.log', 'stderr.log') and path.is_file()
                and not path.is_symlink() for path in args.output_dir.iterdir()), 'Refusing to overwrite prior profile evidence')
    planned = len(args.variants)*len(args.micro_batches)*(args.warmup_steps+args.measure_steps)
    summary = dict(kind='synthetic_whole_model_capacity_sweep', status='running', planned_optimizer_updates=planned,
        optimizer_update_hard_cap=100, global_batch=32, config=CONFIG, measurements=[], biological_quality_claim=False,
        source_root=str(args.source_root), profile_script_sha256=sha(__file__), max_seconds=args.max_seconds,
        candidate_seconds=args.candidate_seconds, started_at_unix=time.time())
    write_json(args.output_dir/'summary.json', summary)
    started = time.monotonic()
    try:
        for variant in args.variants:
            for micro in args.micro_batches:
                directory = args.output_dir/f'{variant}_micro{micro}'
                directory.mkdir(exist_ok=False)
                remaining = args.max_seconds-(time.monotonic()-started)
                require(remaining > 0, 'Total profile wall-time budget expired')
                command = [sys.executable, '-I', '-B', '-u', str(Path(__file__).resolve()),
                    '--source-root', str(args.source_root), '--output-dir', str(directory),
                    '--variants', *args.variants, '--micro-batches', *map(str, args.micro_batches),
                    '--candidate-variant', variant, '--candidate-micro', str(micro), '--seed', str(args.seed),
                    '--warmup-steps', str(args.warmup_steps), '--measure-steps', str(args.measure_steps),
                    '--candidate-seconds', str(args.candidate_seconds), '--max-seconds', str(args.max_seconds)]
                environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', USE_TF='0', OMP_NUM_THREADS='1')
                for key in tuple(environment):
                    if key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT') or key.startswith(('ACCELERATE_', 'TORCHELASTIC_')):
                        environment.pop(key)
                with (directory/'candidate.log').open('x') as log:
                    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = process.wait(timeout=min(args.candidate_seconds, remaining))
                    except subprocess.TimeoutExpired:
                        summary['measurements'].append(dict(variant=variant, micro_batch=micro,
                            status='timeout', peak_allocated_gib=None, peak_reserved_gib=None,
                            reason='Child reached the finite wall-time limit; unavailable metrics are not invented'))
                        raise
                    finally:
                        cleanup_process(process)
                require(code == 0 and (directory/'measurement.json').is_file(), 'Candidate failed; inspect preserved log')
                measured = json.loads((directory/'measurement.json').read_text())
                require(measured['status'] in ('passed', 'oom'), 'Unexpected candidate result')
                summary['measurements'].append(measured)
                write_json(args.output_dir/'summary.json', summary)
        summary.update(status='completed', elapsed_seconds=time.monotonic()-started,
            recommendation=recommendations(summary['measurements'], args.variants))
        write_json(args.output_dir/'summary.json', summary)
        write_json(args.output_dir/'COMPLETE.json', dict(kind='synthetic_capacity_profile_only',
            status='completed', summary_sha256=sha(args.output_dir/'summary.json'), training_completed=False,
            actual_corpus_test=False, checkpoint_created=False))
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir/'summary.json', summary)
        raise
    return summary


def self_test():
    import torch
    args = parse_args(['--source-root', '/tmp/frozen', '--output-dir', '/tmp/artificial-profile'])
    require(len(args.variants)*len(args.micro_batches)*(args.warmup_steps+args.measure_steps) == 72, 'Default update budget differs')
    try:
        parse_args(['--source-root', '/tmp/frozen', '--output-dir', '/tmp/artificial-profile', '--measure-steps', '10'])
    except ValueError:
        pass
    else:
        raise AssertionError('Must reject more than100 planned updates')
    for micro in MICROS:
        batch = integer_batch(micro, torch.arange(4, 24), 25)
        require(all(value.shape == (micro, 2048) for value in batch.values()), 'Integer shape differs')
        require(batch['attention_mask'].all().item() and batch['eligible_mask'].sum().item() == micro*512, 'Attention/target extent differs')
        require(batch['terminal_mask'].sum().item() == micro*2 and (batch['labels'][~batch['eligible_mask']] == -100).all().item(), 'Context/end supervision differs')
        require(32//micro*micro == 32 and batch['token_t'].eq(1).all().item(), 'Accumulation/mask contract differs')
    require(recommendations([], VARIANTS)['largest_common_synthetic_capacity_micro'] is None, 'No evidence cannot recommend a batch')
    observed = [dict(variant=variant, micro_batch=micro, grad_accum=32//micro, status='passed',
        peak_allocated_gib=20., peak_reserved_gib=30., total_gib=80., input_tokens_per_second=float(micro))
        for variant in VARIANTS for micro in MICROS]
    require(recommendations(observed, VARIANTS)['largest_common_synthetic_capacity_micro'] == 32, 'Artificial capacity aggregation differs')
    observed[-1].update(status='oom')
    require(recommendations(observed, VARIANTS)['largest_common_synthetic_capacity_micro'] == 16, 'OOM must exclude the common32 recommendation')
    print('CPU checks passed: 72-update budget, cap100 rejection, micro4/16/32 integer2048 shapes, context/END masks, no-data recommendation. No model/GPU run.', flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    with interruption_handlers():
        return candidate(args) if args.candidate_variant else run(args)


if __name__ == '__main__':
    main()
