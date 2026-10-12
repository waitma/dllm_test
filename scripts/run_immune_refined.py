"""Run refined LLaDA-backbone profiles, expanded diagnostics, or certified pilots.

After activating pllm: USE_TF=0 python scripts/run_immune_refined.py --variant r20_refined_control --mode synthetic --cpu-test --steps 2 --output-dir /tmp/refined-smoke
Long artificial diagnostics use --mode train --data-dir <verified synthetic fixture
manifest directory>. They are always labelled synthetic, never biological pilots.
Real prepared data still requires both existing cleaning certificates.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts import run_immune_research as base
from scripts import run_immune_architecture as previous
from dllm.pipelines.immune_llada.research_refined.model import RefinedModel, VARIANTS
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig

_REAL_TRAIN = base.train
_CONFIG_FOR = base.config_for
_LEGACY_SYNTHETIC = base.synthetic


def config_for(args):
    if args.cpu_test:
        return ImmuneConfig(ablation='A4', d_model=32, n_layers=2, n_heads=4,
            mlp_hidden_size=64, max_length=768, full_capacity=64, fragment_capacity=32)
    return _CONFIG_FOR(args)


def fingerprints():
    hashes = dict(previous.fingerprints())
    files = [Path(__file__), ROOT/'scripts/architecture_prototypes/expanded_fixtures.py']
    files += list((ROOT/'dllm/pipelines/immune_llada/research_refined').rglob('*.py'))
    for path in sorted(files):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def profile_unlocked(args):
    """Measure each candidate in a clean process using this exact release."""
    directory = args.profile_dir
    if (directory/'COMPLETE.json').exists():
        return base.verify_profile(directory, args.variant)
    directory.mkdir(parents=True, exist_ok=True)
    sources, results = fingerprints(), []
    for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        output = directory/f'bs{batch}'
        output.mkdir(exist_ok=True)
        command = [sys.executable, str(Path(__file__)), '--mode', 'profile-candidate',
            '--variant', args.variant, '--batch-size', str(batch), '--output-dir', str(output)]
        with (output/'profile.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        if completed.returncode:
            if 'out of memory' not in (output/'profile.log').read_text()[-16000:].lower():
                raise RuntimeError(f'Profile failed: {output}/profile.log')
            results.append(dict(status='oom', batch_size=batch))
            break
        measured = json.loads((output/'measurement.json').read_text())
        results.append(measured)
        print(json.dumps(measured), flush=True)
        if measured['max_allocated_gib']/measured['total_gib'] >= .8:
            break
    passed = [item for item in results if item['status'] == 'passed'
              and item['max_allocated_gib']/item['total_gib'] <= .85]
    if not passed or sources != fingerprints():
        raise RuntimeError('No passing profile or source changed')
    chosen = passed[-1]
    base.write_json(directory/'summary.json', dict(variant=args.variant,
        config=asdict(ImmuneConfig(ablation='A4')), measurements=results, gpu=chosen['gpu'],
        recommended_micro_batch=chosen['batch_size'], global_batch=256))
    base.write_json(directory/'COMPLETE.json', dict(status='passed', variant=args.variant,
        source_sha256=sources, summary_sha256=base.sha(directory/'summary.json')))
    return chosen['batch_size']


def combine_metrics(items):
    """Pool numerators/counts, not unweighted means of uneven minibatches."""
    if not items:
        raise ValueError('Empty diagnostic evaluation')
    levels = {}
    count_for = {'loss': 'selected_count', 'aa_loss': 'aa_count', 'aa_accuracy': 'aa_count',
                 'terminal_loss': 'terminal_count', 'terminal_accuracy': 'terminal_count'}
    for name, ratio in previous.MASK_LEVELS:
        portions = [item['mask_levels'][name] for item in items]
        level = dict(mask_ratio=ratio)
        for count in ('selected_count', 'aa_count', 'terminal_count'):
            level[count] = sum(part[count] for part in portions)
        for metric, count in count_for.items():
            level[metric] = (sum(part[metric]*part[count] for part in portions if part[count])
                             /level[count]) if level[count] else None
        levels[name] = level
    mean = {}
    for metric in previous.METRIC_KEYS:
        defined = [level[metric] for level in levels.values() if level[metric] is not None]
        mean[metric] = sum(defined)/len(defined) if defined else None
    return {**{metric: levels['mask100'][metric] for metric in previous.METRIC_KEYS},
        'mask_levels': levels, 'mask_mean': mean,
        'mask_protocol': 'fixed_per_record_selected_tokens_v1'}


def build_fixtures(config, cpu=False):
    from scripts.architecture_prototypes.expanded_fixtures import build_fixture_split
    return build_fixture_split(seed=42, train_size=36 if cpu else 576,
        valid_size=18 if cpu else 72, full_capacity=config.full_capacity,
        fragment_capacity=config.fragment_capacity, max_length=config.max_length)


def expanded_diagnostic(args, fixtures=None):
    """Finite artificial learning diagnostic with an untouched heldout set."""
    sources = fingerprints()
    model = base.make_model(args)
    fixtures = build_fixtures(model.config, args.cpu_test) if fixtures is None else fixtures
    train_rows, valid_rows = fixtures.train_records, fixtures.valid_records
    collate = base.collator_for(model.config)
    device = next(model.parameters()).device
    batch_size = args.batch_size
    accumulation = 1 if args.cpu_test else 256//batch_size
    def transfer(rows):
        return {key: value.to(device) for key, value in collate(rows).items()}
    def evaluate(rows):
        grouped = {}
        for row in rows:
            key = row.metadata['diagnostic_group']+'/'+row.metadata['length_bin']
            grouped.setdefault(key, []).append(row)
        strata, all_parts = {}, []
        with base.precision(args.cpu_test):
            for key, group in sorted(grouped.items()):
                parts = [previous.measure(model, transfer(group[start:start+min(batch_size, 4)]))
                         for start in range(0, len(group), min(batch_size, 4))]
                strata[key] = combine_metrics(parts)
                all_parts.extend(parts)
        return {**combine_metrics(all_parts), 'strata': strata, 'records': len(rows)}
    report = dict(variant=args.variant, options=model.options, config=asdict(model.config),
        kind='synthetic_generalization_diagnostic', biological_quality_claim=False,
        dataset_kind='deterministic_artificial_sequences', fixture_manifest=fixtures.manifest,
        source_sha256=sources, seed=42, micro_batch=batch_size, grad_accum=accumulation,
        global_batch=batch_size*accumulation, steps=args.steps, trace=[],
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        training_schedule='constant_lr1e-4', validation_uses_test=False,
        initial=evaluate(valid_rows))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    order, epoch = [], 0
    def take_rows(count):
        nonlocal epoch
        selected = []
        while len(selected) < count:
            if not order:
                order.extend(random.Random(42+epoch).sample(range(len(train_rows)), len(train_rows)))
                epoch += 1
            selected.append(train_rows[order.pop()])
        return selected
    start = time.monotonic()
    if not args.cpu_test:
        torch.cuda.reset_peak_memory_stats()
    base.write_json(args.output_dir/'summary.json', report)
    for step in range(1, args.steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            with base.precision(args.cpu_test):
                result = model(**transfer(take_rows(batch_size)))
                loss = result['loss']/accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite artificial training loss at step {step}')
            loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0 or step == args.steps:
            point = dict(step=step, gradient_norm=float(gradient_norm),
                         auxiliary_loss=float(result['auxiliary_loss']), **evaluate(valid_rows))
            report['trace'].append(point)
            base.write_json(args.output_dir/'summary.json', report)
            print(json.dumps(dict(step=step, gradient_norm=point['gradient_norm'],
                                 mask_mean=point['mask_mean'])), flush=True)
    report.update(final=evaluate(valid_rows), train_probe=evaluate(train_rows[:min(len(train_rows), 72)]),
        elapsed_seconds=time.monotonic()-start, status='completed',
        max_allocated_gib=None if args.cpu_test else torch.cuda.max_memory_allocated()/2**30)
    if sources != fingerprints():
        raise RuntimeError('Source changed during diagnostic')
    report['loss_decreased'] = report['final']['mask_mean']['loss'] < report['initial']['mask_mean']['loss']
    base.write_json(args.output_dir/'summary.json', report)
    base.write_json(args.output_dir/'COMPLETE.json', dict(status='completed',
        kind=report['kind'], biological_quality_claim=False, global_step=args.steps,
        summary_sha256=base.sha(args.output_dir/'summary.json')))


def train(args):
    """An explicit artificial manifest selects diagnostics; real data remains gated."""
    fixture_file = args.data_dir/'synthetic_fixture_manifest.json'
    if fixture_file.exists():
        from scripts.architecture_prototypes.expanded_fixtures import load_verified_fixture_manifest
        if args.resume_from_checkpoint or args.steps > 1000:
            raise ValueError('Artificial diagnostic: no checkpoint resume, maximum 1000 steps')
        fixtures = load_verified_fixture_manifest(args.data_dir)
        expected = build_fixtures(config_for(args), args.cpu_test)
        if fixtures.manifest != expected.manifest:
            raise ValueError('Fixture manifest differs from the frozen production protocol')
        args.batch_size = base.verify_profile(args.profile_dir, args.variant)
        return expanded_diagnostic(args, fixtures)
    return _REAL_TRAIN(args)


def configure():
    previous.configure()
    base.ResearchModel = RefinedModel
    base.VARIANTS = VARIANTS
    base.config_for = config_for
    base.fingerprints = fingerprints
    base.profile_unlocked = profile_unlocked
    base.synthetic = expanded_diagnostic
    base.train = train


def main():
    configure()
    base.main()


if __name__ == '__main__':
    main()
