"""Run isolated R12--R19 architecture experiments after activating pllm.

USE_TF=0 python scripts/run_immune_architecture.py --variant r12_architecture_control --mode synthetic --cpu-test --steps 2 --output-dir /tmp/new-architecture-run
GPU profile/train arguments follow run_immune_research.py; this script never submits jobs.
Protocol v2 resets training noise RNG after initialization; compare against R12,
not the old R00 trajectory whose initialization consumed a different RNG stream.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F

from scripts import run_immune_research as base
from dllm.pipelines.immune_llada.research_next.model import ArchitectureModel, VARIANTS


# Capture before configure() replaces base.fingerprints; repeated configure is safe.
_BASE_FINGERPRINTS = base.fingerprints
_BASE_MAKE_MODEL = base.make_model
MASK_LEVELS = (("mask25", .25), ("mask50", .5), ("mask75", .75), ("mask100", 1.))
METRIC_KEYS = ("loss", "aa_loss", "terminal_loss", "aa_accuracy", "terminal_accuracy")


def make_model(args):
    """Protocol v2: initialize normally, then align CPU/CUDA training-noise RNG."""
    model = _BASE_MAKE_MODEL(args)
    torch.manual_seed(42)
    return model


def fingerprints():
    """Extend the complete original source proof with this isolated implementation."""
    hashes = dict(_BASE_FINGERPRINTS())
    files = [Path(__file__).resolve()]
    files += list((ROOT/'dllm/pipelines/immune_llada/research_next').rglob('*.py'))
    for path in sorted(files):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def profile_unlocked(args):
    """Keep the original sweep protocol, launching this runner in every child."""
    directory = args.profile_dir
    if (directory/'COMPLETE.json').exists():
        return base.verify_profile(directory, args.variant)
    directory.mkdir(parents=True, exist_ok=True)
    sources = base.fingerprints()
    results = []
    candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if base.VARIANTS[args.variant].get('vicreg'):
        candidates = candidates[1:]
    for batch in candidates:
        out = directory/f'bs{batch}'
        command = [sys.executable, str(Path(__file__).resolve()), '--mode', 'profile-candidate',
                   '--variant', args.variant, '--batch-size', str(batch), '--output-dir', str(out)]
        out.mkdir(exist_ok=True)
        with (out/'profile.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        if completed.returncode:
            tail = (out/'profile.log').read_text()[-16000:]
            if 'out of memory' not in tail.lower():
                raise RuntimeError(f'Profile failed; inspect {out}/profile.log')
            results.append(dict(status='oom', batch_size=batch))
            break
        item = json.loads((out/'measurement.json').read_text())
        results.append(item)
        print(json.dumps(item), flush=True)
        if item['max_allocated_gib']/item['total_gib'] >= .8:
            break
    passed = [r for r in results if r['status'] == 'passed' and r['max_allocated_gib']/r['total_gib'] <= .85]
    if not passed:
        raise RuntimeError('No feasible measured batch with memory headroom')
    chosen = passed[-1]
    if sources != base.fingerprints():
        raise RuntimeError('Source changed during profiling')
    summary = dict(variant=args.variant, config=asdict(base.ImmuneConfig(ablation='A4')),
                   measurements=results, gpu=chosen['gpu'], recommended_micro_batch=chosen['batch_size'],
                   global_batch=256)
    base.write_json(directory/'summary.json', summary)
    base.write_json(directory/'COMPLETE.json', dict(status='passed', variant=args.variant,
        source_sha256=sources, summary_sha256=base.sha(directory/'summary.json')))
    return chosen['batch_size']


def _average(values, selected):
    count = int(selected.sum())
    return float(values[selected].mean()) if count else None


@torch.no_grad()
def measure(model, batch):
    """Evaluate four fixed masks without leaking labels into features or RNG state.

    Top-level metrics retain the original full-mask semantics. mask_mean is the
    unweighted mean across the four levels (undefined empty-subset values excluded).
    Fully masked toy records can have identical features and conflicting labels;
    this diagnostic must not be interpreted as an attainable 100% accuracy test.
    """
    device = batch['input_ids'].device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    was_training = model.training
    levels = {}
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            for name, ratio in MASK_LEVELS:
                selected = (base.fixed_validation_mask(batch, ratio).bool()
                            & batch['eligible_mask'].bool() & batch['attention_mask'].bool())
                if not torch.equal(batch['input_ids'][selected], batch['labels'][selected]):
                    raise ValueError('Selected labels must match the clean input tokens')
                features = model.feature_args(batch)
                features['input_ids'] = batch['input_ids'].masked_fill(selected, model.tokenizer.mask_token_id)
                logits = model.denoise(**features)[0]
                labels = batch['labels'].masked_fill(~selected, -100)
                ce = F.cross_entropy(logits.float().transpose(1, 2), labels, reduction='none', ignore_index=-100)
                aa = selected & batch['residue_mask'].bool()
                terminal = selected & batch['terminal_mask'].bool()
                correct = logits.argmax(-1).eq(batch['labels']).float()
                levels[name] = dict(
                    mask_ratio=ratio, selected_count=int(selected.sum()), aa_count=int(aa.sum()),
                    terminal_count=int(terminal.sum()), loss=_average(ce, selected),
                    aa_loss=_average(ce, aa), terminal_loss=_average(ce, terminal),
                    aa_accuracy=_average(correct, aa), terminal_accuracy=_average(correct, terminal))
    finally:
        model.train(was_training)
    averages = {}
    for key in METRIC_KEYS:
        available = [level[key] for level in levels.values() if level[key] is not None]
        averages[key] = sum(available)/len(available) if available else None
    return {**{key: levels['mask100'][key] for key in METRIC_KEYS},
            'mask_levels': levels, 'mask_mean': averages,
            'mask_protocol': 'fixed_per_record_selected_tokens_v1'}


def configure():
    """Install overrides in this process only; importing the module changes nothing."""
    base.ResearchModel = ArchitectureModel
    base.VARIANTS = VARIANTS
    base.make_model = make_model
    base.fingerprints = fingerprints
    base.profile_unlocked = profile_unlocked
    base.measure = measure


def main():
    configure()
    base.main()


if __name__ == '__main__':
    main()
