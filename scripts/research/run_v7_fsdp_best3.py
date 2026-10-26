"""Independent 2-node x 4-GPU FSDP1 runner with full best3+latest retention.

Activate conda pllm; inspect with python -B /project_c20250601/251105016/project/dllm_test/scripts/research/run_v7_fsdp_best3.py --help.
Use the matching Volc YAML for synthetic save/resume admission then corpus training.
This never resumes DDP checkpoints, edits the original pool, or creates data audits.
"""
import argparse
from dataclasses import asdict
from datetime import timedelta
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
BASE = ROOT/'output/v7_formal_eight_gpu_20261002/source_snapshot_v3'
PARENT = ROOT/'data/prepared/immune_v7_test_scope_final_20261002'
ADAPTER = ROOT/'scripts/research/v7_fsdp_trainer.py'
GATE = ROOT/'scripts/research/check_v7_formal_parent.py'
RETENTION = ROOT/'scripts/research/native_best3_latest.py'
DATA_SHA = '494592e33d3bea3f16a5bd363d53c6b2f276372f92cb3a84b21fcd65f07c28e9'
SIDECAR = 'fsdp_checkpoint.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['synthetic', 'corpus', 'cleanup-synthetic'], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--source-manifest-sha256', required=True)
    p.add_argument('--admission', type=Path)
    p.add_argument('--resume-from-checkpoint', type=Path)
    p.add_argument('--micro-batch', type=int, default=32)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=25042)
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--trace-steps', type=int, default=4)
    args = p.parse_args(argv)
    require(args.micro_batch > 0 and args.grad_accum > 0 and args.micro_batch*args.grad_accum*8 == 256,
            'Global batch must stay 256 = micro_batch * accumulation * 8')
    require(args.max_steps > args.warmup_steps >= 0 and args.eval_every > 0 and args.save_every > 0,
            'Invalid schedule')
    require(0 <= args.stop_after <= args.max_steps and args.trace_steps >= 0, 'Invalid stop/trace window')
    for name in ('output_dir', 'source_manifest', 'admission', 'resume_from_checkpoint'):
        path = getattr(args, name)
        require(path is None or path.is_absolute() and path == path.resolve(), 'Canonical absolute path required: '+name)
    require(not args.output_dir.is_relative_to(ROOT/'output/v7_initial_training_20261005'),
            'Original formal output is protected')
    return args


def verify_sources(args, loaded=False):
    require(sha(args.source_manifest) == args.source_manifest_sha256, 'Source manifest changed')
    files = json.loads(args.source_manifest.read_text())['files']
    for path in (Path(__file__).resolve(), ADAPTER, GATE, RETENTION):
        require(str(path) in files, 'Missing integration source binding')
    for filename, expected in files.items():
        path = Path(filename)
        require(path.is_absolute() and path == path.resolve() and not path.is_symlink()
                and sha(path) == expected, 'Bound source changed: '+filename)
    if loaded:
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, '__file__', None)
            if filename and name.split('.')[0] in ('dllm', 'scripts', 'examples'):
                path = Path(filename).resolve()
                require(path.is_relative_to(BASE) and files.get(str(path)) == sha(path),
                        'Training import escaped frozen source: '+str(path))
    return files


def retain_checkpoint(checkpoint, binding):
    """Called on rank zero only after the native eight-rank save receipt exists."""
    spec = importlib.util.spec_from_file_location('_v7_fsdp_native_retention', RETENTION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.retain({str(checkpoint.parent): module.binding_digest(binding)},
                         checkpoint, sidecar_name=SIDECAR)


def parent_gate(args):
    env = {k: v for k, v in os.environ.items() if k not in
           ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'PYTHONPATH')}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-I', '-B', str(GATE), '--repo-root', str(ROOT),
        '--data-dir', str(PARENT), '--source-manifest', str(args.source_manifest),
        '--source-manifest-sha256', args.source_manifest_sha256], env=env,
        cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, 'Original read-only data gate failed: '+result.stderr[-3000:])
    receipt = json.loads(result.stdout)
    require(receipt['status'] == 'passed' and receipt['metadata']['dataset_manifest.json'] == DATA_SHA,
            'Unexpected parent admission')
    require(json.loads((PARENT/'dataset_manifest.json').read_text()).get('training_ready') is True,
            'training_ready is false')
    return receipt


def synthetic_row(tokenizer, length=2048, index=0):
    """Numeric fabricated tensor fixture; no corpus, annotation or sequence sampling."""
    import torch
    aa = tokenizer.encode_residues('AGST')[index % 4]
    end = tokenizer.special_id('<protd>')
    ids = torch.full((length,), aa, dtype=torch.long)
    chain = torch.full((length,), 2, dtype=torch.long)
    positions = torch.arange(length)
    slots, residue, terminal = [torch.zeros(length, dtype=torch.bool) for _ in range(3)]
    for uid, start in enumerate((2, 260)):
        stop = min(start+256, length)
        if stop <= start+9:
            continue
        chain[start:stop], positions[start:stop] = uid, torch.arange(stop-start)
        slots[start:stop] = True
        ids[start:stop] = tokenizer.mask_token_id
        ids[start:start+8], residue[start:start+8] = aa, True
        ids[start+8], terminal[start+8] = end, True
    chain[:2] = -1
    eligible = residue | terminal
    return dict(input_ids=ids, labels=ids.masked_fill(~eligible, -100), chain_ids=chain,
        position_ids=positions, role_ids=torch.ones(length, dtype=torch.long),
        scope_ids=torch.ones(length, dtype=torch.long), attention_mask=torch.ones(length, dtype=torch.bool),
        target_slot_mask=slots, residue_mask=residue, terminal_mask=terminal, eligible_mask=eligible)


def full_inventory(checkpoint):
    return {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(checkpoint.iterdir()) if p.is_file() and p.name != SIDECAR}


def validate_checkpoint(checkpoint, binding):
    import torch
    required = {'model.safetensors', 'pytorch_model_fsdp.bin', 'optimizer.bin', 'scheduler.pt',
        'trainer_state.json', 'training_args.bin', 'immune_config.json', 'research_config.json',
        'vocabulary.json', 'run_provenance.json', *(f'rng_state_{r}.pth' for r in range(8))}
    require(required <= {p.name for p in checkpoint.iterdir()}, 'Incomplete full FSDP checkpoint')
    require(json.loads((checkpoint/'run_provenance.json').read_text()) == binding, 'Resume binding differs')
    step = json.loads((checkpoint/'trainer_state.json').read_text())['global_step']
    require(0 < step <= binding['schedule']['max_steps'], 'Invalid checkpoint step')
    optim = torch.load(checkpoint/'optimizer.bin', map_location='cpu', weights_only=False)
    require(bool(optim.get('state')) and bool(optim.get('param_groups')), 'Empty optimizer checkpoint')
    require(torch.load(checkpoint/'scheduler.pt', map_location='cpu', weights_only=False)['last_epoch'] == step,
            'Saved scheduler step differs')
    for rank in range(8):
        require({'python', 'numpy', 'cpu', 'cuda'} <= torch.load(checkpoint/f'rng_state_{rank}.pth',
                map_location='cpu', weights_only=False).keys(), 'Incomplete RNG checkpoint')
    return {'step': step, 'files': full_inventory(checkpoint)}


def cleanup_synthetic(args):
    binding = json.loads((args.output_dir/'run_provenance.json').read_text())
    receipt = json.loads((args.output_dir/'ADMISSION.json').read_text())
    require(binding['phase'] == 'synthetic' and receipt['status'] == 'passed'
            and receipt['source_manifest_sha256'] == args.source_manifest_sha256,
            'Only this completed synthetic run may be cleaned')
    removed = []
    for path in args.output_dir.glob('checkpoint-*'):
        require(path.name.removeprefix('checkpoint-').isdigit() and path.is_dir() and not path.is_symlink(),
                'Unexpected cleanup path')
        sidecar = path/SIDECAR
        if sidecar.exists():
            shutil.copyfile(sidecar, args.output_dir/(path.name+'_evidence.json'))
        shutil.rmtree(path)
        removed.append(str(path))
    write_json(args.output_dir/'CLEANUP.json', {'removed_own_synthetic_checkpoints': removed})


def main(argv=None):
    args = parse_args(argv)
    verify_sources(args)
    if args.phase == 'cleanup-synthetic':
        cleanup_synthetic(args)
        return
    sys.path[:] = [str(BASE)] + [p for p in sys.path if p and not Path(p).resolve().is_relative_to(ROOT)]
    import torch
    import transformers
    import accelerate
    from torch.utils.data import Dataset, Subset
    from transformers import TrainingArguments, TrainerCallback, set_seed
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
    from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
    from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
    spec = importlib.util.spec_from_file_location('_v7_fsdp_adapter', ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    require(torch.__version__.split('+')[0] == '2.8.0' and transformers.__version__ == '4.48.1'
            and accelerate.__version__ == '1.14.0', 'Use the pinned training environment')
    require(os.environ.get('WORLD_SIZE') == '8' and os.environ.get('LOCAL_WORLD_SIZE') == '4',
            'Requires torchrun with 2 nodes, 4 local workers each')
    local_rank = int(os.environ['LOCAL_RANK'])
    require(torch.cuda.device_count() == 4 and 0 <= local_rank < 4, 'Four visible GPUs per node required')
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=30))
    distributed, rank = torch.distributed, torch.distributed.get_rank()
    group = distributed.new_group(backend='gloo', timeout=timedelta(minutes=45))
    lock = None

    def phase(operation):
        try:
            packet = {'value': operation(), 'error': None}
        except Exception as exc:
            packet = {'value': None, 'error': repr(exc)}
        packets = [None]*8
        distributed.all_gather_object(packets, packet, group=group)
        require(not any(p['error'] for p in packets), 'Rank phase failed: '+str(packets))
        return [p['value'] for p in packets]

    try:
        topology = phase(lambda: dict(rank=rank, local_rank=local_rank, host=socket.gethostname(),
            gpu=torch.cuda.get_device_name(), total_memory=torch.cuda.get_device_properties(local_rank).total_memory))
        hosts = {x['host'] for x in topology}
        require(len(hosts) == 2 and all(sorted(x['local_rank'] for x in topology if x['host'] == h) == [0, 1, 2, 3]
                for h in hosts), 'Actual topology is not two distinct 4-GPU hosts')
        def claim():
            nonlocal lock
            if rank == 0:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                lock = (args.output_dir/'.run.lock').open('a')
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.resume_from_checkpoint is None:
                    require({p.name for p in args.output_dir.iterdir()} == {'.run.lock'},
                            'Refusing to overwrite an existing run')
        phase(claim)
        audit = phase(lambda: parent_gate(args) if rank == 0 and args.phase == 'corpus' else None)[0]
        config = ImmuneConfig(ablation='A4')
        set_seed(42)
        model = adapter.ForwardMetricModel(config, 'f011')
        # FSDP applies nonreentrant activation checkpointing inside each wrapped block.
        # The backbone's outer checkpoint path stays off to avoid duplicate all-gathers.
        require(model.backbone.activation_checkpointing_strategy is None, 'Unexpected outer checkpointing')
        if args.phase == 'synthetic':
            class NumericDataset(Dataset):
                def __init__(self, size): self.size = size
                def __len__(self): return self.size
                def __getitem__(self, index): return synthetic_row(model.tokenizer, 2048, index)
            train, valid = NumericDataset(256*12), {'numeric': NumericDataset(19)}
            def collator(rows): return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
            selection = {'numeric': list(range(19))}
        else:
            require(args.admission is not None, 'Synthetic save/resume admission is required')
            admission = json.loads(args.admission.read_text())
            require(admission['status'] == 'passed' and admission['source_manifest_sha256'] == args.source_manifest_sha256
                and admission['micro_batch'] == args.micro_batch and admission['grad_accum'] == args.grad_accum
                and admission['config'] == asdict(config) and admission['topology'] == topology,
                'Synthetic admission does not match this run/hardware')
            local_metadata = {n: sha(PARENT/n) for n in audit['metadata']}
            phase(lambda: require(local_metadata == audit['metadata'], 'Per-node audited metadata differs'))
            manifest = json.loads((PARENT/'dataset_manifest.json').read_text())
            train = PreparedImmuneDataset(PARENT, split='train')
            valid, selection = {}, {}
            for source in sorted({s['source'] for s in manifest['splits']['valid']['shards']}):
                data = PreparedImmuneDataset(PARENT, split='valid', source=source)
                indices = random.Random(42).sample(range(len(data)), min(256, len(data)))
                valid[source], selection[source] = Subset(data, indices), indices
            collator = ImmuneCollator('A4', max_length=2048, full_capacity=256, fragment_capacity=64)
        fsdp_config = dict(transformer_layer_cls_to_wrap=['ArchitectureBlock'], use_orig_params=True,
            sync_module_states=True, cpu_ram_efficient_loading=False, backward_prefetch='backward_pre',
            forward_prefetch=False, activation_checkpointing=True)
        os.environ.update(FSDP_VERSION='1', FSDP_STATE_DICT_TYPE='FULL_STATE_DICT', FSDP_OFFLOAD_PARAMS='false')
        training = TrainingArguments(output_dir=str(args.output_dir),
            per_device_train_batch_size=args.micro_batch, per_device_eval_batch_size=args.micro_batch,
            gradient_accumulation_steps=args.grad_accum, max_steps=args.max_steps,
            warmup_steps=args.warmup_steps, learning_rate=1e-4, lr_scheduler_type='cosine',
            seed=42, data_seed=42, bf16=True, remove_unused_columns=False, label_names=['labels'],
            eval_strategy='steps', eval_steps=args.eval_every, save_strategy='steps', save_steps=args.save_every,
            save_total_limit=None, save_only_model=False, save_on_each_node=False, save_safetensors=True,
            logging_steps=25 if args.phase == 'corpus' else 1, report_to='none', prediction_loss_only=True,
            logging_nan_inf_filter=False,
            disable_tqdm=True, dataloader_num_workers=0, dataloader_drop_last=True,
            optim='adamw_torch', local_rank=local_rank, fsdp='full_shard auto_wrap', fsdp_config=fsdp_config)
        trainer = adapter.FSDPTrainer(model=model, args=training, train_dataset=train,
            eval_dataset=valid, data_collator=collator)
        require(trainer.is_fsdp_enabled and not trainer.is_deepspeed_enabled, 'FSDP1 required')
        plugin = trainer.accelerator.state.fsdp_plugin
        require(plugin.fsdp_version == 1 and plugin.sharding_strategy == ShardingStrategy.FULL_SHARD
                and str(plugin.state_dict_type).endswith('FULL_STATE_DICT') and plugin.activation_checkpointing,
                'Actual FSDP settings differ')
        sources = verify_sources(args, loaded=True)
        binding = dict(schema='v7.fsdp.formal.v1',
            checkpoint_policy=dict(metric='eval_loss', mode='min', best_k=3, latest=True, full_state=True),
            phase=args.phase, variant='f011', config=asdict(config),
            options=model.options, source_manifest_sha256=args.source_manifest_sha256, source_files=sources,
            parent_metadata=audit['metadata'] if audit else None, data_dir=str(PARENT) if audit else None,
            world_size=8, nodes=2, local_world_size=4, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
            global_batch=256, seed=42, validation_indices=selection, train_records=len(train), fsdp_config=fsdp_config,
            schedule=dict(max_steps=args.max_steps, warmup_steps=args.warmup_steps, learning_rate=1e-4,
                scheduler='cosine', eval_every=args.eval_every, save_every=args.save_every),
            versions=dict(torch=torch.__version__, transformers=transformers.__version__, accelerate=accelerate.__version__))
        bindings = phase(lambda: binding)
        require(all(b == binding for b in bindings), 'Rank bindings differ')
        trainer.run_provenance = binding
        if args.resume_from_checkpoint:
            require(args.resume_from_checkpoint.parent == args.output_dir, 'Only same-run FSDP resumes allowed')
            def check_resume():
                if rank != 0: return None
                saved = json.loads((args.resume_from_checkpoint/SIDECAR).read_text())
                actual = validate_checkpoint(args.resume_from_checkpoint, binding)
                require(saved['schema'] == 'v7.fsdp.checkpoint.v1' and saved['rank_acknowledgments'] == list(range(8))
                    and saved['step'] == actual['step'] and saved['files'] == actual['files'], 'Resume inventory differs')
                require(actual['step'] < (args.stop_after or args.max_steps), 'No remaining updates')
                return actual['step']
            prior_step = phase(check_resume)[0]
        else:
            prior_step = 0
        phase(lambda: write_json(args.output_dir/'run_provenance.json', binding) if rank == 0 else None)

        class NativeCheckpoint(TrainerCallback):
            def on_evaluate(self, args_, state, control, metrics=None, **kwargs):
                if args.phase == 'synthetic':
                    expected = {'eval_records_per_mask': 19, 'eval_record_evaluations': 76,
                        'eval_mask100_numeric_selected_count': 342,
                        'eval_mask100_numeric_aa_count': 304,
                        'eval_mask100_numeric_terminal_count': 38,
                        'eval_mask100_numeric_chain_count': 38}
                    require(all((metrics or {}).get(k) == v for k, v in expected.items()),
                            'Synthetic distributed validation counts/tail deduplication differ')

            def on_log(self, args_, state, control, logs=None, **kwargs):
                require(all(not isinstance(v, float) or math.isfinite(v) for v in (logs or {}).values()),
                        'Nonfinite training or validation metric')

            def on_train_begin(self, args_, state, control, **kwargs):
                wrapped = trainer.model_wrapped
                units = [m for m in wrapped.modules() if isinstance(m, FSDP)]
                phase(lambda: require(len(units) == config.n_layers+1, 'Expected root plus one FSDP unit per ArchitectureBlock'))

            def on_step_end(self, args_, state, control, **kwargs):
                if state.global_step >= (args.stop_after or args.max_steps):
                    control.should_training_stop = control.should_save = control.should_evaluate = True
                return control

            def on_save(self, args_, state, control, **kwargs):
                with trainer.phase('checkpoint_verification_host'):
                    distributed.barrier(group=group)
                    checkpoint = args.output_dir/f'checkpoint-{state.global_step}'
                    acknowledgments = phase(lambda: dict(rank=rank,
                        rng_sha256=sha(checkpoint/f'rng_state_{rank}.pth')))
                    receipt = phase(lambda: validate_checkpoint(checkpoint, binding) if rank == 0 else None)[0]
                    receipt.update(schema='v7.fsdp.checkpoint.v1', world_size=8,
                        rank_acknowledgments=[item['rank'] for item in acknowledgments],
                        saved_rank_rng=acknowledgments, writer_rank=0)
                    phase(lambda: write_json(checkpoint/SIDECAR, receipt) if rank == 0 else None)
                    phase(lambda: retain_checkpoint(checkpoint, binding) if rank == 0 else None)

        trainer.add_callback(NativeCheckpoint())
        trainer.add_callback(adapter.BoundedTrace(trainer, args.trace_steps))
        result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if prior_step else None)
        require(math.isfinite(result.metrics['train_loss']), 'Nonfinite aggregate training loss')
        final = trainer.state.global_step
        restore = phase(lambda: dict(rank=rank, optimizer=getattr(trainer, 'native_optimizer_restore', None),
            rng=getattr(trainer, 'native_rng_restore', False), peak_gib=torch.cuda.max_memory_allocated()/2**30))
        if prior_step:
            require(all(x['rng'] and x['optimizer']['step'] == prior_step for x in restore), 'Incomplete native resume evidence')
        def finish():
            if rank != 0: return
            trainer.save_metrics('train', result.metrics)
            write_json(args.output_dir/f'SEGMENT_{prior_step}_TO_{final}.json', dict(final_step=final,
                prior_step=prior_step, actual_native_resume=restore, topology=topology))
            if final == args.max_steps:
                write_json(args.output_dir/'COMPLETE.json', dict(step=final, status='completed', phase=args.phase))
                if args.phase == 'synthetic':
                    require(prior_step > 0, 'Synthetic admission requires actual native resume')
                    write_json(args.output_dir/'ADMISSION.json', dict(status='passed', config=asdict(config),
                        source_manifest_sha256=args.source_manifest_sha256, micro_batch=args.micro_batch,
                        grad_accum=args.grad_accum, topology=topology, restored_ranks=restore,
                        note='Synthetic engineering/memory admission only; no biological quality claim.'))
        phase(finish)
    finally:
        if lock: lock.close()
        distributed.destroy_process_group(group)
        distributed.destroy_process_group()


if __name__ == '__main__':
    main()
