"""Profile, smoke-test, or train isolated scratch-model research variants.

After activating pllm:
python scripts/run_immune_research.py --variant r04_qk_norm --mode synthetic --cpu-test --steps 2 --output-dir /tmp/new-run
GPU jobs: generated YAMLs in train_jobs/immune_research_queue012. Never submits jobs.
"""

# Original-study ablations after actual training and generation completion.
if __name__ == '__main__':
    import sys as _abl_sys
    import runpy as _abl_runpy
    import hashlib as _abl_hashlib
    import json as _abl_json
    import time as _abl_time
    from pathlib import Path as _AblPath
    _abl_selected = [_abl_sys.argv[i+1] for i, value in enumerate(_abl_sys.argv[:-1]) if value == '--variant']
    if len(_abl_selected) == 1 and _abl_selected[0] in ('p04_v7_f010_seed43', 'p04_v7_f101_seed43', 'p04_v7_f110_seed42', 'p04_v7_f110_seed43', 'p04_v7_f111_seed42', 'p04_v7_f111_seed43', 'p04_v7_cv_segment_global_seed42', 'p04_v7_cv_segment_global_seed43', 'p04_v7_cv_late_fusion_seed42', 'p04_v7_cv_late_fusion_seed43', 'p04_v7_fixed_gate_control_seed42', 'p04_v7_fixed_gate_control_seed43'):
        _abl_entry = _AblPath('/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/source_v1/pool_metrics_only_ablation_v2_dispatch.py')
        if _abl_hashlib.sha256(_abl_entry.read_bytes()).hexdigest() != 'c40bfbb48541dc11cfe35baaa53b02d4f4fdc07dd4257e807aa8c7a1155dca9d':
            raise RuntimeError('Published post-generation ablation entry changed')
        _abl_receipt = _AblPath('/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/enqueue_receipt.json')
        _abl_deadline = _abl_time.monotonic()+60
        while True:
            try:
                _abl_batch = _abl_json.loads(_abl_receipt.read_text())
            except FileNotFoundError:
                _abl_batch = {}
            if _abl_batch.get('state') == 'enqueued':
                if _abl_batch.get('catalog') != {'path': '/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/catalog.json', 'sha256': '50040676c11ab06acb90f65a6cef2090510ff5907ba8fadcc0769940f0fa2ed8'}:
                    raise RuntimeError('Post-generation batch binding changed')
                break
            if _abl_time.monotonic() >= _abl_deadline:
                raise RuntimeError('Post-generation publication did not finish; no automatic retry')
            _abl_time.sleep(0.1)
        raise SystemExit(_abl_runpy.run_path(str(_abl_entry))['main'](_abl_sys.argv[1:],
            catalog_path=_AblPath('/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/catalog.json'), catalog_sha256='50040676c11ab06acb90f65a6cef2090510ff5907ba8fadcc0769940f0fa2ed8',
            ledger_path=_AblPath('/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/guarded_dispatch_v1/claims_post_generation_pool_v2.json'), ledger_sha256='8aea0b22d8dba21b06d2b31ce8c99cadad41efb70617ec74704f4c97c8a476ec'))

# Post-training CDR entry; no model or CUDA imports before dispatch.
if __name__ == '__main__':
    import sys as _cdr_sys
    import runpy as _cdr_runpy
    import hashlib as _cdr_hashlib
    from pathlib import Path as _CdrPath
    _cdr_selected = [_cdr_sys.argv[i+1] for i, value in enumerate(_cdr_sys.argv[:-1]) if value == '--variant']
    if len(_cdr_selected) == 1 and _cdr_selected[0] in ('p07_v7_cdr_kong_cdrh1', 'p07_v7_cdr_kong_cdrh2', 'p07_v7_cdr_kong_cdrh3', 'p07_v7_cdr_sab23_cdrl1', 'p07_v7_cdr_sab23_cdrl2', 'p07_v7_cdr_sab23_cdrl3', 'p07_v7_cdr_sab23_cdrh1', 'p07_v7_cdr_sab23_cdrh2', 'p07_v7_cdr_sab23_cdrh3'):
        _cdr_entry = _CdrPath('/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_pool_v1/source_v1/pool_v7_generation_dispatch.py')
        if _cdr_hashlib.sha256(_cdr_entry.read_bytes()).hexdigest() != '51eb045fd69b876a125990e2893613fbdbe44a3b3260f665affd36e48adaf703':
            raise RuntimeError('Published CDR evaluator changed')
        raise SystemExit(_cdr_runpy.run_path(str(_cdr_entry))['main'](_cdr_sys.argv[1:]))
if __name__ == '__main__':
    # Route before any ML import; the selected release starts in a fresh process.
    import runpy as _route_runpy
    from pathlib import Path as _RoutePath
    import hashlib as _route_hashlib
    _route_observer = _RoutePath('/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/pool_parent_source_v1/pool_parent_runtime_observation.py')
    if _route_hashlib.sha256(_route_observer.read_bytes()).hexdigest() != '8d13c32c6c8e0b6ee65df4573a14e784947e98602e50c7ef10cf2459393e79c8':
        raise RuntimeError('Frozen pool parent observer changed')
    _route_runpy.run_path(str(_route_observer))['observe']()
    _route_dispatcher = _RoutePath('/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_20261005/pool_formal_20261005_dispatch.py')
    if _route_hashlib.sha256(_route_dispatcher.read_bytes()).hexdigest() != '87002bb6ba57f837d72943a3816c0254e7fca3224a7dde46052b17af7e6cf492':
        raise RuntimeError('Frozen large-batch-v2 entry changed')
    raise SystemExit(_route_runpy.run_path(str(_route_dispatcher))['main']())

import argparse
import fcntl
from contextlib import nullcontext
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
from torch.utils.data import ConcatDataset, Subset, Sampler
from transformers import TrainingArguments
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
from dllm.pipelines.immune_llada.data.balanced_sampling import FullCoverageSourceSampler
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.immune_llada.research.model import ResearchModel, VARIANTS
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from examples.llada.immune_pretrain_v3 import ImmuneTrainer, PilotStop, assert_ready, sha
from scripts.profile_immune_v3 import code_hashes
from scripts.smoke_immune_v3_cpu import measure


def fingerprints():
    hashes = code_hashes()
    files = list((ROOT/'dllm/pipelines/immune_llada/research').glob('*.py'))
    files += [Path(__file__), ROOT/'scripts/smoke_immune_v3_cpu.py',
              ROOT/'scripts/data/tcr_scope_gate.py']
    files += list((ROOT/'dllm/pipelines/immune_llada/data').rglob('*.py'))
    for path in sorted(files):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def fixed_validation_mask(batch, ratio):
    """Stable per-record masks across different batching, padding and RNG states."""
    if ratio not in (.25, .5, .75, 1.):
        raise ValueError('Use one of the frozen validation noise levels')
    ids = batch['input_ids'].detach().cpu()
    attention = batch['attention_mask'].detach().cpu().bool()
    eligible = batch['eligible_mask'].detach().cpu().bool()
    result = torch.zeros_like(ids, dtype=torch.float)
    for row in range(ids.shape[0]):
        visible = ids[row, attention[row]].tolist()
        digest = hashlib.sha256(json.dumps(visible).encode()).digest()
        seed = int.from_bytes(digest[:8], 'little') % (2**63-1)
        generator = torch.Generator().manual_seed(seed)
        positions = eligible[row].nonzero().flatten()
        draws = torch.rand(len(positions), generator=generator)
        chosen = draws < ratio
        if len(positions) and not chosen.any():
            chosen[draws.argmin()] = True
        result[row, positions[chosen]] = 1
    return result.to(batch['input_ids'].device)


def validate_research_resume(checkpoint, provenance, requested_step):
    """Extend stop step only; preserve data, method, optimizer and LR horizon."""
    required = ['run_provenance.json', 'research_config.json', 'model.safetensors',
                'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']
    missing = [name for name in required if not (checkpoint/name).is_file()]
    if missing:
        raise ValueError(f'Incomplete resumable checkpoint: {missing}')
    saved = json.loads((checkpoint/'run_provenance.json').read_text())
    previous = {k: v for k, v in saved.items() if k != 'pilot_stop_step'}
    current = {k: v for k, v in provenance.items() if k != 'pilot_stop_step'}
    if previous != current:
        raise ValueError('Resume data/method/source/batch/schedule provenance differs')
    completed = int(json.loads((checkpoint/'trainer_state.json').read_text())['global_step'])
    if not 0 < completed < requested_step <= provenance['max_steps']:
        raise ValueError('Resume requires a strictly larger cumulative stop step within the same horizon')
    return completed


def toy_rows(long_context=False, config=None):
    """Artificial varied strings; never read a benchmark or biological corpus."""
    if long_context:
        # A4 uses capacity tails even for short targets; maximize total input length.
        size = config.max_length-2*config.full_capacity-6
        return [BioSeqRecord([BioSeqChain(letter*size, 'antigen'),
                BioSeqChain('C'+letter*120+'W', 'antibody_heavy'),
                BioSeqChain('C'+letter*100+'F', 'antibody_light')], 'antibody_antigen', 'synthetic_long')
                for letter in ('A', 'G')]
    rows = []
    # Different family frequencies allow the equal-source sampler to be exercised.
    for i, letter in enumerate('AGST'):
        rows.append(BioSeqRecord([BioSeqChain('C'+letter*(5+i)+'F', 'tcr_beta'),
                                  BioSeqChain('C'+letter*(3+i)+'W', 'tcr_alpha')], 'tcr', 'toy_tcr'))
    rows.append(BioSeqRecord([BioSeqChain('AAGGSS', 'antigen'), BioSeqChain('CAAGGSSW', 'antibody_heavy'),
                              BioSeqChain('CGGAAF', 'antibody_light')], 'antibody_antigen', 'toy_ab'))
    rows.append(BioSeqRecord([BioSeqChain('CSSGGAAW', 'nanobody_vhh')], 'nanobody', 'toy_nb'))
    return rows


def config_for(args):
    if args.cpu_test:
        return ImmuneConfig(ablation='A4', d_model=64, n_layers=2, n_heads=4,
                            mlp_hidden_size=128, max_length=512, full_capacity=24, fragment_capacity=16)
    return ImmuneConfig(ablation='A4')


def collator_for(config):
    return ImmuneCollator('A4', max_length=config.max_length, full_capacity=config.full_capacity,
                          fragment_capacity=config.fragment_capacity)


def make_model(args):
    torch.manual_seed(42)
    model = ResearchModel(config_for(args), args.variant)
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    return model.to('cpu' if args.cpu_test else 'cuda')


def precision(cpu=False):
    return nullcontext() if cpu else torch.autocast('cuda', dtype=torch.bfloat16)


def verify_profile(directory, variant):
    record = json.loads((directory/'COMPLETE.json').read_text())
    if record['status'] != 'passed' or record['variant'] != variant:
        raise ValueError('No passing profile for this variant')
    if record['source_sha256'] != fingerprints():
        raise ValueError('Code changed since profile; use a fresh profile directory')
    if record['summary_sha256'] != sha(directory/'summary.json'):
        raise ValueError('Profile summary changed')
    summary = json.loads((directory/'summary.json').read_text())
    if summary['config'] != asdict(ImmuneConfig(ablation='A4')):
        raise ValueError('Production configuration changed')
    if summary['gpu'] != torch.cuda.get_device_name():
        raise ValueError('GPU differs from measured profile')
    batch = summary['recommended_micro_batch']
    passed = [r for r in summary['measurements'] if r.get('status') == 'passed' and r['batch_size'] == batch]
    if (len(passed) != 1 or batch < 1 or 256 % batch or
            passed[0]['precision'] != 'fp32_parameters_adam_bf16_autocast' or
            passed[0]['config'] != summary['config'] or
            passed[0]['max_allocated_gib']/passed[0]['total_gib'] > .85):
        raise ValueError('Invalid measured microbatch')
    return batch


def profile_candidate(args):
    model = make_model(args).train()
    rows = toy_rows(True, model.config)
    data = collator_for(model.config)([rows[i % len(rows)] for i in range(args.batch_size)])
    batch = {k: v.cuda() for k, v in data.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with precision():
            result = model(**batch)
        if not torch.isfinite(result['loss']):
            raise FloatingPointError('Nonfinite profile loss')
        result['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    write_json(args.output_dir/'measurement.json', dict(status='passed', batch_size=args.batch_size,
        variant=args.variant, config=asdict(model.config), actual_length=batch['input_ids'].shape[1],
        internal_length=batch['input_ids'].shape[1]+model.options.get('registers', 0),
        max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
        gpu=torch.cuda.get_device_name(), precision='fp32_parameters_adam_bf16_autocast',
        elapsed_seconds=time.monotonic()-started))


def profile_unlocked(args):
    directory = args.profile_dir
    if (directory/'COMPLETE.json').exists():
        return verify_profile(directory, args.variant)
    directory.mkdir(parents=True, exist_ok=True)
    sources = fingerprints()
    results = []
    # VICReg needs a fixed nontrivial statistical batch. All variants remain global 256.
    candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if VARIANTS[args.variant].get('vicreg'):
        candidates = candidates[1:]
    for batch in candidates:
        out = directory/f'bs{batch}'
        # Fresh child process after each OOM; training only starts after measurement.
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
    if sources != fingerprints():
        raise RuntimeError('Source changed during profiling')
    summary = dict(variant=args.variant, config=asdict(ImmuneConfig(ablation='A4')), measurements=results,
                   gpu=chosen['gpu'], recommended_micro_batch=chosen['batch_size'], global_batch=256)
    write_json(directory/'summary.json', summary)
    write_json(directory/'COMPLETE.json', dict(status='passed', variant=args.variant, source_sha256=sources,
                                             summary_sha256=sha(directory/'summary.json')))
    return chosen['batch_size']


def profile(args):
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    with (args.profile_dir/'profile.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This method is already being profiled; do not submit duplicate jobs') from exc
        return profile_unlocked(args)


def synthetic(args):
    sources = fingerprints()
    model = make_model(args)
    collate = collator_for(model.config)
    rows = toy_rows()
    device = next(model.parameters()).device
    def transfer(items):
        return {k: v.to(device) for k, v in collate(items).items()}
    evaluation = transfer(rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch_size = args.batch_size
    accumulation = 1 if args.cpu_test else 256//batch_size
    def evaluate():
        with precision(args.cpu_test):
            return measure(model, evaluation)
    source_order = [0, 4, 5, 1, 4, 5, 2, 4, 5, 3, 4, 5] if model.options.get('equal_source') else list(range(len(rows)))
    report = dict(variant=args.variant, options=model.options, config=asdict(model.config),
                  kind='synthetic_functional_diagnostic', biological_quality_claim=False,
                  source_sha256=sources, seed=42, micro_batch=batch_size, grad_accum=accumulation,
                  global_batch=batch_size*accumulation, initial=evaluate(), trace=[], steps=args.steps,
                  parameters=sum(p.numel() for p in model.parameters()))
    start = time.monotonic()
    if not args.cpu_test:
        torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            offset = (step*accumulation+micro)*batch_size
            batch = transfer([rows[source_order[(offset+i) % len(source_order)]] for i in range(batch_size)])
            with precision(args.cpu_test):
                result = model(**batch)
                loss = result['loss']/accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at {step}')
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step+1) % 5 == 0 or step+1 == args.steps:
            point = dict(step=step+1, gradient_norm=float(norm), auxiliary_loss=float(result['auxiliary_loss']), **evaluate())
            report['trace'].append(point)
            write_json(args.output_dir/'summary.json', report)
            print(json.dumps(point), flush=True)
    report.update(final=evaluate(), elapsed_seconds=time.monotonic()-start, status='completed',
                  max_allocated_gib=None if args.cpu_test else torch.cuda.max_memory_allocated()/2**30)
    if sources != fingerprints():
        raise RuntimeError('Source changed during diagnostic')
    report['loss_decreased'] = report['final']['loss'] < report['initial']['loss']
    write_json(args.output_dir/'summary.json', report)
    write_json(args.output_dir/'COMPLETE.json', dict(status='completed', biological_quality_claim=False))


class EpochRandomSampler(Sampler):
    """Random-access epoch seeding makes natural sampling resumable across epochs."""
    def __init__(self, dataset, seed=42):
        self.size, self.seed, self.epoch = len(dataset), seed, 0

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch*1_000_003)
        yield from torch.randperm(self.size, generator=generator).tolist()


class ResearchTrainer(ImmuneTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not model.training:
            inputs = {**inputs, 'token_t': fixed_validation_mask(inputs, getattr(self, '_validation_ratio', 1.))}
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if not isinstance(dataset, dict):
            return super().evaluate(dataset, ignore_keys, metric_key_prefix)
        metrics = {}
        # Evaluation must not change the subsequent training corruption stream.
        devices = [self.args.device.index or 0] if self.args.device.type == 'cuda' else []
        try:
            with torch.random.fork_rng(devices=devices):
                for ratio in (.25, .5, .75, 1.):
                    self._validation_ratio = ratio
                    prefix = f'{metric_key_prefix}_mask{int(ratio*100)}'
                    metrics.update(super().evaluate(dataset, ignore_keys, prefix))
            for kind in ('loss', 'aa_loss', 'terminal_loss'):
                metrics[f'{metric_key_prefix}_{kind}'] = metrics[f'{metric_key_prefix}_mask100_{kind}']
            self.log({k: v for k, v in metrics.items() if k in {f'{metric_key_prefix}_{s}' for s in ('loss', 'aa_loss', 'terminal_loss')}})
            return metrics
        finally:
            self._validation_ratio = 1.

    def _get_train_sampler(self, train_dataset=None):
        if VARIANTS[self.model.variant].get('equal_source'):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            return FullCoverageSourceSampler([len(d) for d in dataset.datasets], effective_batch_size=256, seed=42)
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return EpochRandomSampler(dataset, self.args.data_seed)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        self.accelerator.unwrap_model(self.model).save_research_config(Path(output_dir or self.args.output_dir))


def train(args):
    # Intentionally before model construction: fail closed on current uncleared data.
    manifest, _ = assert_ready(args.data_dir)
    batch = verify_profile(args.profile_dir, args.variant)
    source_hashes = fingerprints()
    model = make_model(args)
    source_names = sorted({s['source'] for s in manifest['splits']['train']['shards']})
    train_data = ConcatDataset([PreparedImmuneDataset(args.data_dir, split='train', source=s) for s in source_names])
    validation = {}
    for source in sorted({s['source'] for s in manifest['splits']['valid']['shards']}):
        data = PreparedImmuneDataset(args.data_dir, split='valid', source=source)
        indices = random.Random(42).sample(range(len(data)), min(len(data), 256))
        validation[source] = Subset(data, indices)
    provenance = dict(kind='research_pilot', variant=args.variant, options=model.options, config=asdict(model.config),
        source_sha256=source_hashes, seed=42, data_manifest_sha256=sha(args.data_dir/'dataset_manifest.json'),
        certificate_sha256=sha(args.data_dir/'audit_certificate.json'),
        scope_certificate_sha256=sha(args.data_dir/'downstream_scope_certificate.json'),
        profile_complete_sha256=sha(args.profile_dir/'COMPLETE.json'), micro_batch=batch, grad_accum=256//batch,
        global_batch=256, max_steps=10000, pilot_stop_step=args.steps, learning_rate=1e-4, warmup_steps=100,
        eval_steps=100, save_steps=100,
        validation='seed42 up to 256 rows/source; per-record fixed masks25/50/75/100; not downstream test',
        source_names=source_names, training_sampling='equal_source' if model.options.get('equal_source') else 'natural')
    if args.resume_from_checkpoint:
        validate_research_resume(args.resume_from_checkpoint, provenance, args.steps)
    settings = TrainingArguments(output_dir=str(args.output_dir), per_device_train_batch_size=batch,
        per_device_eval_batch_size=min(batch, 16), gradient_accumulation_steps=256//batch,
        max_steps=10000, learning_rate=1e-4, warmup_steps=100, lr_scheduler_type='cosine', seed=42, data_seed=42,
        bf16=True, remove_unused_columns=False, label_names=['labels'], eval_strategy='steps', eval_steps=100,
        save_strategy='steps', save_steps=100, save_total_limit=3, logging_steps=10, report_to='none',
        prediction_loss_only=True, dataloader_num_workers=2)
    trainer = ResearchTrainer(model=model, args=settings, train_dataset=train_data, eval_dataset=validation,
                              data_collator=collator_for(model.config))
    trainer.run_provenance = provenance
    trainer.add_callback(PilotStop(args.steps))
    write_json(args.output_dir/'run_provenance.json', provenance)
    write_json(args.output_dir/'execution_request.json', dict(
        resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
        requested_cumulative_step=args.steps, validation_uses_test=False))
    if not args.resume_from_checkpoint:
        trainer.save_metrics('initial_eval', trainer.evaluate())
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_metrics('train', result.metrics)
    trainer.save_model()
    if trainer.state.global_step != args.steps:
        raise RuntimeError('Requested cumulative step budget was not reached')
    if source_hashes != fingerprints():
        raise RuntimeError('Source changed during training; do not use run for comparison')
    write_json(args.output_dir/'COMPLETE.json', dict(status='completed', variant=args.variant,
                                                   global_step=trainer.state.global_step))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=sorted(VARIANTS), required=True)
    parser.add_argument('--mode', choices=['profile', 'profile-candidate', 'profile-synthetic', 'synthetic', 'train'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile-dir', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/prepared/immune_v7_mhc_sequence_v1')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--resume-from-checkpoint', type=Path)
    parser.add_argument('--cpu-test', action='store_true')
    args = parser.parse_args()
    if args.cpu_test and (args.mode != 'synthetic' or args.steps > 3 or args.batch_size != 2):
        raise ValueError('Local smoke only: synthetic, <=3 steps, batch 2')
    if not 1 <= args.steps <= (5000 if args.mode == 'train' else 100):
        raise ValueError('Bounded pilot/smoke budget exceeded')
    if args.resume_from_checkpoint and args.mode != 'train':
        raise ValueError('Checkpoint resume is supported only for real-data training')
    if not args.cpu_test and not torch.cuda.is_available():
        raise ValueError('GPU modes require CUDA; use --cpu-test for local smoke')
    torch.set_num_threads(2 if args.cpu_test else 4)
    if args.mode != 'profile-candidate':
        args.output_dir.mkdir(parents=True, exist_ok=False)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {'profile', 'profile-synthetic', 'train'} and args.profile_dir is None:
        raise ValueError('--profile-dir is required')
    if args.mode == 'profile-candidate':
        profile_candidate(args)
    elif args.mode in {'profile', 'profile-synthetic'}:
        args.batch_size = profile(args)
        if args.mode == 'profile-synthetic':
            synthetic(args)
    elif args.mode == 'synthetic':
        if not args.cpu_test:
            if args.profile_dir is None:
                raise ValueError('GPU smoke requires measured profile')
            args.batch_size = verify_profile(args.profile_dir, args.variant)
        synthetic(args)
    else:
        train(args)


# CLI dispatch occurs above, before Torch/model imports.
