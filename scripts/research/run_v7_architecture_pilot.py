"""Compare existing backbones on a fixed, explicitly exploratory V7 subset.

Activate conda pllm, then run this absolute file path with --variant r20 --seed 42
--data-dir /absolute/subset --output-dir /absolute/new-run. Defaults are 3125
updates, microbatch 4, accumulation 8. --cpu-smoke --steps 2 uses artificial
records and a tiny model only; it never reads --data-dir. No sequence generation,
platform submission, formal-ready change, or automatic failure retry occurs.
Tests save metrics and logs by default. --save-checkpoints explicitly enables
model files; --resume additionally requires that option and an existing file.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from dllm.pipelines.immune_llada.data.preprocessing.validators import validate_prepared_row
from dllm.pipelines.immune_llada.research_refined.model import RefinedModel
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from scripts.run_immune_architecture import MASK_LEVELS, METRIC_KEYS, measure
from scripts.run_immune_refined import combine_metrics
from scripts.run_immune_research import toy_rows
from scripts.research.result_retention import record_observation

VARIANTS = {'r20': 'r20_refined_control', 'r21': 'r21_refined_head_gate',
            'r24': 'r24_refined_fixed_scale', 'r26': 'r26_refined_local_global'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checked_metadata(item):
    path = Path(item['path'])
    raw = path.read_bytes()
    if not path.is_absolute() or hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError('Metadata path or SHA mismatch')
    return json.loads(raw)


def load_subset(directory):
    """Read only the subset and its already-existing parent metadata."""
    manifest_path = directory / 'sample_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'v7.exploratory_uncertified.v1'
            or manifest.get('status') != 'exploratory_uncertified'
            or manifest.get('formal_training_ready') is not False):
        raise ValueError('Requires an explicitly exploratory subset manifest')
    parent = manifest['parent']
    original = checked_metadata(parent['manifest'])
    if original.get('training_ready') is not True or parent['manifest'].get('training_ready') is not True:
        raise ValueError('Parent dataset must already be ready; pilot never changes it')
    for name in ('audit_certificate', 'downstream_scope_certificate'):
        entry = parent['certificates'][name]
        certificate = checked_metadata(entry)
        binding = 'manifest_sha256' if name == 'audit_certificate' else 'dataset_manifest_sha256'
        if (certificate.get('status') != 'passed' or certificate.get('pending') != []
                or entry.get('status') != certificate['status']
                or entry.get('pending') != certificate['pending']
                or certificate.get(binding) != parent['manifest']['sha256']
                or certificate.get('forbidden_overlap_rows') != 0 or certificate.get('family_split_overlap') != 0):
            raise ValueError(f'Parent {name} is not passed/current')
    splits, groups = {}, {}
    for split in ('train', 'valid'):
        item = manifest['files'][split]
        path = Path(item['path'])
        if (not path.is_absolute() or path.resolve().parent != directory.resolve()
                or path.is_symlink()):
            raise ValueError(f'Subset {split} path or SHA mismatch')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError(f'Subset {split} SHA mismatch')
        lines = raw.splitlines()
        if len(lines) != item['records'] or not lines or any(not line.strip() for line in lines):
            raise ValueError(f'Subset {split} row count differs')
        if split == 'valid' and len(lines) > 1024:
            raise ValueError('Pilot validation budget is at most 1024 records')
        source_groups = [None] * len(lines)
        for span in item['source_ranges']:
            start, end = span['start_line'] - 1, span['end_line']
            source = span['source_group']
            if not isinstance(source, str) or not source or not 0 <= start < end <= len(lines):
                raise ValueError('Invalid source range')
            if any(value is not None for value in source_groups[start:end]):
                raise ValueError('Overlapping source ranges')
            source_groups[start:end] = [source] * (end - start)
        if None in source_groups or dict(Counter(source_groups)) != item['source_counts']:
            raise ValueError('Source ranges/counts do not cover this subset')
        rows = [json.loads(line) for line in lines]
        if any(row.get('split') not in (None, split) for row in rows):
            raise ValueError('Subset row has a different split')
        splits[split] = [validate_prepared_row(row) for row in rows]
        groups[split] = source_groups
    return splits, groups, {'path': str(manifest_path), 'sha256': sha(manifest_path),
                            'manifest': manifest}


def source_hashes():
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT) and path.is_file():
                paths.add(path)
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def initialize(config, variant, seed, model_factory=RefinedModel):
    """Copy every common baseline parameter; preserve candidate-only modules."""
    torch.manual_seed(seed)
    baseline = RefinedModel(config, VARIANTS['r20'])
    common = baseline.state_dict()
    digest = hashlib.sha256()
    for name, value in sorted(common.items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    if model_factory is RefinedModel and variant == VARIANTS['r20']:
        model = baseline
    else:
        torch.manual_seed(seed)
        model = model_factory(config, variant)
        result = model.load_state_dict(common, strict=False)
        if result.unexpected_keys or any(not key.endswith('.gate.weight') for key in result.missing_keys):
            raise ValueError('Unexpected common-parameter mismatch')
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    return model, digest.hexdigest()


def evaluate(model, collate, rows, sources, device, micro, cpu):
    """Persist aggregate numerators/counts only, never predicted sequences."""
    by_source = {}
    for row, source in zip(rows, sources):
        by_source.setdefault(source, []).append(row)
    portions, strata = [], {}
    with nullcontext() if cpu else torch.autocast('cuda', dtype=torch.bfloat16):
        for source, records in sorted(by_source.items()):
            parts = []
            for start in range(0, len(records), micro):
                batch = {k: v.to(device) for k, v in collate(records[start:start + micro]).items()}
                parts.append(measure(model, batch))
            strata[source] = combine_metrics(parts)
            strata[source]['records'] = len(records)
            portions.extend(parts)
    pooled = combine_metrics(portions)
    macro = {}
    for level, _ in MASK_LEVELS:
        macro[level] = {}
        for metric in METRIC_KEYS:
            values = [value['mask_levels'][level][metric] for value in strata.values()
                      if value['mask_levels'][level][metric] is not None]
            macro[level][metric] = sum(values) / len(values) if values else None
    # Counts plus sums make pooling transparent without saving sequence/logit payloads.
    for result in [pooled, *strata.values()]:
        for level in result['mask_levels'].values():
            for metric, count in [('loss', 'selected_count'), ('aa_loss', 'aa_count'),
                                  ('terminal_loss', 'terminal_count'), ('aa_accuracy', 'aa_count'),
                                  ('terminal_accuracy', 'terminal_count')]:
                level[metric + '_sum'] = level[metric] * level[count] if level[count] else 0.
    return dict(records=len(rows), token_pooled=pooled, source_macro=macro, sources=strata)


def run(args, *, horizon=3125, model_factory=RefinedModel, evaluator=evaluate,
        study=None, expected_train_records=None):
    save_checkpoints = getattr(args, 'save_checkpoints', False)
    if args.resume and not save_checkpoints:
        raise ValueError('Metrics-only tests cannot resume; checkpoint saving must be explicitly enabled')
    cpu = args.cpu_smoke
    config = (ImmuneConfig(d_model=32, n_layers=2, n_heads=4, mlp_hidden_size=64,
                           max_length=128, full_capacity=24, fragment_capacity=16)
              if cpu else ImmuneConfig())
    if cpu:
        rows = toy_rows(config=config)
        splits = {'train': rows, 'valid': rows}
        groups = {split: [row.source for row in rows] for split in splits}
        data = {'kind': 'artificial_cpu_functionality_only', 'heldout': False}
    else:
        splits, groups, data = load_subset(args.data_dir)
        if expected_train_records is not None and len(splits['train']) != expected_train_records:
            raise ValueError('Training subset size differs from the registered study')
        if horizon <= 100 or args.steps > horizon:
            raise ValueError('Invalid study schedule horizon or cumulative stop step')
    model, common_sha = initialize(config, args.variant, args.seed, model_factory)
    device = torch.device('cpu' if cpu else 'cuda:0')
    model.to(device)
    collate = ImmuneCollator('A4', max_length=config.max_length,
                            full_capacity=config.full_capacity, fragment_capacity=config.fragment_capacity)
    provenance = dict(kind='cpu_smoke' if cpu else 'exploratory_architecture_reconstruction',
        formal_training=False, biological_quality_claim=False, generation=False,
        checkpoint_saving=save_checkpoints,
        config=asdict(config), variant=args.variant, seed=args.seed, data=data,
        source_sha256=source_hashes(), common_initialization_sha256=common_sha,
        micro_batch=args.micro_batch, grad_accum=args.grad_accum,
        global_batch=args.micro_batch * args.grad_accum, scheduler_horizon=2 if cpu else horizon,
        evaluation_interval=args.eval_every, validation_sampling_claim='subset_only',
        warmup=0 if cpu else 100, learning_rate=1e-4, loss='existing_micro_mean',
        mask_protocol='independent_seed_per_global_microstep', torch_version=torch.__version__,
        parameters=sum(p.numel() for p in model.parameters()))
    if study is not None:
        provenance['study'] = study
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    def schedule(step):
        if cpu:
            return 1.
        return step / 100 if step < 100 else .5 * (1 + math.cos(math.pi * (step - 100) / (horizon - 100)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    state = dict(step=0, epoch=0, position=0, records=0, input_tokens=0,
                 eligible_tokens=0, selected_aa_terminal_tokens=0, train_seconds=0.)
    latest = args.output_dir / 'latest.pt'
    if args.resume:
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['provenance'] != provenance:
            raise ValueError('Resume data/source/model/seed/schedule binding differs')
        state = saved['state']
        if not state['step'] < args.steps:
            raise ValueError('Resume requires a larger cumulative stop step')
        curve = [json.loads(line) for line in (args.output_dir / 'curve.jsonl').read_text().splitlines()]
        if not curve or curve[-1]['step'] != state['step']:
            raise ValueError('Curve/checkpoint transaction incomplete; preserve evidence for review')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['cpu_rng'])
        if not cpu:
            torch.cuda.set_rng_state(saved['cuda_rng'], device)
    else:
        write_json(args.output_dir / 'run_provenance.json', provenance)
    if not cpu:
        torch.cuda.reset_peak_memory_stats(device)
    order_epoch, order = -1, []
    def take_rows():
        nonlocal order_epoch, order
        picked = []
        while len(picked) < args.micro_batch:
            if order_epoch != state['epoch']:
                order = list(range(len(splits['train'])))
                random.Random(args.seed + state['epoch']).shuffle(order)
                order_epoch = state['epoch']
            picked.append(splits['train'][order[state['position']]])
            state['position'] += 1
            if state['position'] == len(order):
                state['epoch'] += 1
                state['position'] = 0
        return picked
    def checkpoint(point):
        def write_checkpoint():
            temporary = latest.with_suffix('.pt.tmp')
            torch.save(dict(provenance=provenance, state=dict(state), model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                cpu_rng=torch.get_rng_state(), cuda_rng=None if cpu else torch.cuda.get_rng_state(device)), temporary)
            temporary.replace(latest)
        record_observation(args.output_dir, point, save_checkpoint=save_checkpoints,
                           checkpoint_writer=write_checkpoint)
    def observation(**extra):
        started = time.monotonic()
        validation = evaluator(model, collate, splits['valid'], groups['valid'], device, args.micro_batch, cpu)
        return dict(**state, **extra, validation=validation, evaluation_seconds=time.monotonic() - started,
                    unique_train_records=min(state['records'], len(splits['train'])),
                    subset_train_records=len(splits['train']), record_exposures=state['records'],
                    train_tokens_per_second=state['input_tokens'] / max(state['train_seconds'], 1e-9),
                    peak_allocated_gib=None if cpu else torch.cuda.max_memory_allocated(device) / 2**30)
    if not args.resume:
        checkpoint(observation())
    for step in range(state['step'] + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started, loss_sum = time.monotonic(), 0.
        lr = optimizer.param_groups[0]['lr']
        for micro in range(args.grad_accum):
            batch = {k: v.to(device) for k, v in collate(take_rows()).items()}
            # Model initialization and validation cannot consume this corruption stream.
            torch.manual_seed(args.seed * 1000003 + (step - 1) * args.grad_accum + micro)
            with nullcontext() if cpu else torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(**batch)
                loss = result['loss'] / args.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            loss_sum += float(loss.detach())
            state['records'] += len(batch['input_ids'])
            state['input_tokens'] += int(batch['attention_mask'].sum())
            state['eligible_tokens'] += int(batch['eligible_mask'].sum())
            state['selected_aa_terminal_tokens'] += int(result['aa_count'] + result['terminal_count'])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if not cpu:
            torch.cuda.synchronize(device)
        state['train_seconds'] += time.monotonic() - started
        state['step'] = step
        if step % 25 == 0 or step == args.steps:
            progress = dict(step=step, loss=loss_sum, records=state['records'],
                train_seconds=state['train_seconds'], input_tokens=state['input_tokens'],
                peak_allocated_gib=None if cpu else torch.cuda.max_memory_allocated(device) / 2**30)
            write_json(args.output_dir / 'progress.json', progress)
            print(json.dumps(progress, allow_nan=False), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            point = observation(train_micro_mean_loss=loss_sum, gradient_norm=float(norm), learning_rate=lr)
            checkpoint(point)
            print(json.dumps({'step': step, 'validation': point['validation']['token_pooled']['mask_mean']}), flush=True)
    if provenance['source_sha256'] != source_hashes():
        raise RuntimeError('Loaded project source changed during this run')
    if not cpu and expected_train_records is not None and args.steps == horizon:
        if (state['records'] != expected_train_records or state['epoch'] != 1 or state['position'] != 0):
            raise RuntimeError('Registered single-epoch run did not consume every row exactly once')
    marker = 'COMPLETE.json' if cpu or args.steps == horizon else f'SEGMENT_{args.steps}.json'
    write_json(args.output_dir / marker, dict(status='completed', kind=provenance['kind'],
        global_step=state['step'], formal_training=False, biological_quality_claim=False,
        summary_sha256=sha(args.output_dir / 'summary.json'), provenance_sha256=sha(args.output_dir / 'run_provenance.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=tuple(VARIANTS) + tuple(VARIANTS.values()))
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=3125)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--cpu-smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-checkpoints', action='store_true',
                        help='Explicitly retain model/optimizer files; tests default to metrics only')
    args = parser.parse_args()
    args.variant = VARIANTS.get(args.variant, args.variant)
    if not args.output_dir.is_absolute() or (args.data_dir is not None and not args.data_dir.is_absolute()):
        parser.error('All directory paths must be absolute')
    if (not 0 <= args.seed < 2**31 or not 1 <= args.steps <= 3125 or args.eval_every != 250
            or args.micro_batch != 4 or args.grad_accum != 8):
        parser.error('Require valid seed, 1..3125 steps, evaluation interval 250, microbatch 4 and accumulation 8')
    if args.cpu_smoke and (args.steps != 2 or args.data_dir is not None):
        parser.error('--cpu-smoke requires --steps 2 and no --data-dir')
    if not args.cpu_smoke and args.data_dir is None:
        parser.error('GPU pilot requires --data-dir')
    if args.resume:
        if not args.output_dir.is_dir() or (args.output_dir / 'COMPLETE.json').exists():
            parser.error('Resume requires an existing incomplete run')
    else:
        if args.output_dir.exists():
            worker_logs = {'train.log', 'worker.log', 'stdout.log', 'stderr.log'}
            if (not args.output_dir.is_dir() or args.output_dir.is_symlink()
                    or any(path.name not in worker_logs or not path.is_file() or path.is_symlink()
                           for path in args.output_dir.iterdir())):
                parser.error('Fresh output may contain only worker logs; existing run evidence is preserved')
        else:
            args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException:
        failure = args.output_dir / f'failure_{time.time_ns()}.txt'
        failure.write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
