"""Native Trainer entry for one model on eight externally initialized ranks.

Activate conda pllm. Inspect without importing torch: python -B /absolute/path/
run_v7_formal.py --help. The launcher must initialize NCCL world8, expose one
physical GPU per process, set RANK/WORLD_SIZE/LOCAL_RANK=0 and call main(argv)
in that same process. A shell invocation alone cannot initialize this entry.
Pass --source-root (frozen code), --repo-root (real audit resources), --data-dir,
--output-dir, --variant, --seed, --micro-batch,
--grad-accum, --global-batch, --max-steps, --source-manifest and its SHA256.
The source manifest is {"files": {"/absolute/source.py": "sha256", ...}} and
must cover this entry, both reused adapters and every imported repository .py.
User-authorized epoch-two DDP continuation with explicit scheduler/source provenance,
resumed sampler epoch preservation, and best3+latest retention. Run only from the
independent frozen source path recorded in the epoch-two source manifest.
No launcher, certificate creation, sequence generation or winner selection.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

WORLD = 8
VARIANTS = {'r20': 'r20_refined_control', 'r21': 'r21_refined_head_gate',
            'r24': 'r24_refined_fixed_scale', 'r26': 'r26_refined_local_global'}
FACTORIAL = ('f000', 'f001', 'f010', 'f011', 'f100', 'f101', 'f110', 'f111',
             'cv_segment_global', 'cv_late_fusion', 'fixed_gate_control')
GATE_HELPER = 'scripts/research/check_v7_formal_parent.py'
FINAL_MANIFEST = '28e3cbf8fb860c13b514a11d600579257a233fefe9f70ea070b042d04f8a424c'
SIDECAR = 'formal_checkpoint.json'
ADAPTERS = ('output/generic_eight_rank_trainer_integration_20261002/eight_rank_trainer_adapter_v1.py',
            'output/generic_eight_worker_group_integration_20261002/eight_rank_checkpoint_adapter_v2.py')
CONTRACT = 'output/generic_eight_rank_integration_20261002/eight_rank_contracts_v1.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'repo-root', 'data-dir', 'output-dir', 'source-manifest'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--variant', choices=(*VARIANTS, *FACTORIAL), required=True)
    for name in ('seed', 'micro-batch', 'grad-accum', 'global-batch', 'max-steps'):
        parser.add_argument('--'+name, type=int, required=True)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--valid-per-source', type=int, default=256)
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--resume-from-checkpoint', type=Path)
    parser.add_argument('--transition', type=Path, required=True)
    parser.add_argument('--transition-sha256', required=True)
    args = parser.parse_args(argv)
    for key in ('source_root', 'repo_root', 'data_dir', 'output_dir', 'source_manifest', 'resume_from_checkpoint', 'transition'):
        path = getattr(args, key)
        require(path is None or (path.is_absolute() and path == path.resolve()), key+' must be canonical absolute')
    require(args.seed >= 0, 'seed must be nonnegative')
    require(all(getattr(args, key) > 0 for key in ('micro_batch', 'grad_accum', 'global_batch',
        'max_steps', 'eval_every', 'save_every', 'valid_per_source')), 'Positive budgets required')
    require(args.micro_batch * args.grad_accum * WORLD == args.global_batch,
            'global_batch must equal micro_batch * grad_accum * 8')
    require(0 <= args.warmup_steps < args.max_steps and 0 < args.learning_rate < 1,
            'Invalid fixed scheduler parameters')
    require(0 <= args.stop_after <= args.max_steps, 'stop-after must be within the fixed horizon')
    return args


def verify_sources(args, loaded=False):
    require(sha(args.source_manifest) == args.source_manifest_sha256, 'Source manifest SHA differs')
    files = json.loads(args.source_manifest.read_text())['files']
    required = {str(Path(__file__).resolve()), str(args.repo_root/GATE_HELPER), str(args.repo_root/CONTRACT),
                *(str(args.source_root/name) for name in ADAPTERS)}
    if loaded:
        for module_name, module in tuple(sys.modules.items()):
            name = getattr(module, '__file__', None)
            if name:
                path = Path(name).resolve()
                if module_name.split('.')[0] in ('dllm', 'scripts', 'examples') and path.suffix == '.py':
                    require(path.is_relative_to(args.source_root), 'Previously imported training module uses another source root')
                if path.suffix == '.py' and path.is_relative_to(args.source_root):
                    required.add(str(path))
    require(required <= files.keys(), 'Source manifest omits executable sources: '+str(sorted(required-files.keys())))
    for name, expected in files.items():
        path = Path(name)
        require(path.is_absolute() and path == path.resolve() and not path.is_symlink()
                and sha(path) == expected, 'Bound source changed: '+name)
    return files


def isolate_training_sources(args):
    """Reject cached modules before importing any training code; never reload a mix."""
    require(Path(__file__).resolve() == args.source_root/'scripts/research/run_v7_formal.py',
            'The driver itself must execute from source-root')
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and name.split('.')[0] in ('dllm', 'scripts', 'examples'):
            require(Path(filename).resolve().is_relative_to(args.source_root), 'Mixed training module cache')
    sys.path[:] = [str(args.source_root)] + [entry for entry in sys.path if entry
        and not Path(entry).resolve().is_relative_to(args.repo_root)
        and Path(entry).resolve() != args.source_root]


def check_parent(args):
    files = verify_sources(args)
    environment = {key: value for key, value in os.environ.items() if key not in
        ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'PYTHONPATH')}
    environment.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-I', '-B', str(args.repo_root/GATE_HELPER),
        '--repo-root', str(args.repo_root), '--data-dir', str(args.data_dir),
        '--source-manifest', str(args.source_manifest), '--source-manifest-sha256', args.source_manifest_sha256],
        cwd=args.repo_root, env=environment, text=True, capture_output=True, check=False)
    require(result.returncode == 0, 'Read-only parent gate failed: '+result.stderr[-4000:])
    receipt = json.loads(result.stdout)
    require(receipt.get('schema') == 'v7.formal.read_only_parent_check.v1' and receipt.get('status') == 'passed'
        and receipt.get('repo_root') == str(args.repo_root) and receipt.get('data_dir') == str(args.data_dir)
        and receipt.get('source_manifest_sha256') == args.source_manifest_sha256, 'Audit helper identity differs')
    executed = receipt.get('executed_sources', {})
    require(str(args.repo_root/GATE_HELPER) in executed and all(files.get(path) == value
        and sha(path) == value for path, value in executed.items()), 'Actual gate source binding differs')
    require(receipt['metadata']['dataset_manifest.json'] == FINAL_MANIFEST, 'Wrong final parent')
    return receipt


def parameter_digest(model, checkpoint=None):
    """Hash this rank's actual parameter bytes, including names/shapes/dtypes."""
    import torch
    digestor = hashlib.sha256()
    count = 0
    from safetensors import safe_open
    context = safe_open(str(checkpoint/'model.safetensors'), framework='pt', device='cpu') if checkpoint else nullcontext(None)
    with context as archive:
        for name, parameter in sorted(model.named_parameters()):
            value = (archive.get_tensor(name) if archive else parameter.detach()).cpu().contiguous()
            digestor.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
            digestor.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            count += value.numel()
    return {'sha256': digestor.hexdigest(), 'numel': count}


def check_launcher(distributed, cuda, environment):
    require(distributed.is_initialized() and distributed.get_world_size() == WORLD,
            'External launcher must initialize the eight-rank process group first')
    rank = distributed.get_rank()
    require(distributed.get_backend() == 'nccl' and cuda.device_count() == 1
            and cuda.current_device() == 0, 'One visible CUDA device/rank and NCCL required')
    require(environment.get('RANK') == str(rank) and environment.get('WORLD_SIZE') == '8'
            and environment.get('LOCAL_RANK') == '0', 'Launcher rank environment differs')
    return rank


def load_adapter(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collective(distributed, group, operation):
    """Propagate a local exception before any rank proceeds to its next phase."""
    try:
        packet = {'value': operation(), 'error': None}
    except Exception as exc:
        packet = {'value': None, 'error': repr(exc)}
    packets = [None] * WORLD
    distributed.all_gather_object(packets, packet, group=group)
    require(not any(item['error'] for item in packets), 'Rank phase failed: '+str(packets))
    return [item['value'] for item in packets]


def inventory(checkpoint, helper):
    return {key: value for key, value in helper._files(checkpoint).items() if key != SIDECAR}


def validate_checkpoint(checkpoint, binding, helper, torch, expected_step=None):
    """Validate full native state; model/RNG restoration remains HF's job."""
    files = inventory(checkpoint, helper)
    required = {'model.safetensors', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json',
                'training_args.bin', 'immune_config.json', 'research_config.json',
                'vocabulary.json', 'run_provenance.json', *(f'rng_state_{rank}.pth' for rank in range(WORLD))}
    require(required <= files.keys(), 'Incomplete native Trainer checkpoint')
    require(json.loads((checkpoint/'run_provenance.json').read_text()) == binding,
            'Checkpoint model/data/source/schedule binding differs')
    state = json.loads((checkpoint/'trainer_state.json').read_text())
    step = state['global_step']
    require(type(step) is int and 0 < step <= binding['schedule']['max_steps'], 'Invalid saved step')
    require(expected_step is None or step == expected_step, 'Checkpoint step differs')
    # Only locally created checkpoints admitted by the bound inventory are read.
    optimizer = torch.load(checkpoint/'optimizer.pt', map_location='cpu', weights_only=False)
    scheduler = torch.load(checkpoint/'scheduler.pt', map_location='cpu', weights_only=False)
    require(optimizer.get('state') and optimizer.get('param_groups'), 'Optimizer state is absent')
    require(scheduler.get('last_epoch') == step, 'Scheduler step differs')
    for rank in range(WORLD):
        rng = torch.load(checkpoint/f'rng_state_{rank}.pth', map_location='cpu', weights_only=False)
        require({'python', 'numpy', 'cpu', 'cuda'} <= rng.keys(), 'Missing rank RNG components')
    require(inventory(checkpoint, helper) == files, 'Checkpoint changed during validation')
    return {'step': step, 'files': files, 'binding_sha256': digest(binding)}


def checkpoint_callback(trainer, binding, args, helper, torch, group):
    from transformers import TrainerCallback
    distributed, rank = torch.distributed, torch.distributed.get_rank()

    class CompleteCheckpoint(TrainerCallback):
        def on_train_begin(self, training_args, state, control, **kwargs):
            self.previous_time = time.monotonic()
            self.previous_save = True
            return control

        def on_step_end(self, training_args, state, control, **kwargs):
            stamp = time.monotonic()
            if rank == 0:
                with (args.output_dir/'step_timing.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=state.global_step, time_unix=time.time(),
                        interval_seconds=stamp-self.previous_time,
                        first_or_after_save=self.previous_save,
                        scope='rank0_training_loop_wall_interval_including_data_and_DDP'))+'\n')
            self.previous_time, self.previous_save = stamp, False
            if state.global_step >= (args.stop_after or args.max_steps):
                control.should_training_stop = True
                control.should_save = True
                control.should_evaluate = True
            return control

        def on_save(self, training_args, state, control, **kwargs):
            # Native Trainer has written optimizer/scheduler and every rank RNG.
            distributed.monitored_barrier(group=group, timeout=timedelta(seconds=300), wait_all_ranks=True)
            checkpoint = args.output_dir / f'checkpoint-{state.global_step}'
            checks = collective(distributed, group, lambda: verify_sources(args, loaded=True))
            require(all(item == checks[0] for item in checks), 'Source binding differs across ranks')
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            parameters = collective(distributed, group, lambda: parameter_digest(unwrapped))
            require(all(item == parameters[0] for item in parameters), 'Actual model parameters differ across ranks')
            saved = collective(distributed, group, lambda: parameter_digest(unwrapped, checkpoint) if rank == 0 else None)
            require(saved[0] == parameters[0], 'Saved model differs from the actual rank parameters')
            receipts = collective(distributed, group, lambda: validate_checkpoint(
                checkpoint, binding, helper, torch, state.global_step))
            require(all(item == receipts[0] for item in receipts), 'Checkpoint bytes differ across ranks')
            collective(distributed, group, lambda: helper._atomic_json_once(checkpoint/SIDECAR,
                {'schema': 'v7.formal.native_checkpoint.v1', **receipts[0], 'world_size': WORLD,
                 'writer_rank': 0, 'rank_acknowledgments': list(range(WORLD)),
                 'rank_parameter_digests': [{'rank': index, **value} for index, value in enumerate(parameters)],
                 'saved_parameter_digest': saved[0]}) if rank == 0 else None)
            from scripts.research.native_best3_latest import retain
            from scripts.research.v7_epoch_transition import transition
            def rotate():
                spec, previous_binding, _ = transition(args)
                owners = {str(args.output_dir): digest(binding), **spec['retention_owners']}
                return retain(owners, checkpoint)
            collective(distributed, group, lambda: rotate() if rank == 0 else None)
            self.previous_time, self.previous_save = time.monotonic(), True
            return control

    return CompleteCheckpoint()


def main(argv=None):
    args = parse_args(argv)
    verify_sources(args)
    isolate_training_sources(args)
    import torch
    import transformers
    import accelerate
    from torch.utils.data import Subset
    from transformers import TrainingArguments, set_seed
    from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
    from dllm.pipelines.immune_llada.research_refined.model import RefinedModel
    from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel, VARIANTS as FACTORIAL_VARIANTS
    from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
    from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
    from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
    from scripts.architecture_prototypes import validation_metrics as metrics
    from scripts.run_immune_research import EpochRandomSampler
    from scripts.research.formal_resume_observation import ResumeObservation, complete_receipt
    from scripts.research.v7_epoch_transition import transition, POLICY, LR_EXTENSION
    from scripts.research.v7_epoch_schedule import epoch_restart_factor, preserve_resume_epoch
    transition_spec, old_binding, anchor = transition(args)
    distributed = torch.distributed
    rank = check_launcher(distributed, torch.cuda, os.environ)
    require(set(FACTORIAL_VARIANTS) == set(FACTORIAL), 'Factorial registry differs from the explicit supported set')
    factory = load_adapter(args.source_root / ADAPTERS[0])
    helper = load_adapter(args.source_root / ADAPTERS[1])
    require(factory.CONTRACTS == args.repo_root/CONTRACT, 'Legacy immutable metadata contract path differs')
    verify_sources(args, loaded=True)
    group = distributed.new_group(backend='gloo', timeout=timedelta(minutes=30))
    try:
        # Full parent audit executes once; all ranks receive its result or error.
        audit_receipt = collective(distributed, group, lambda: check_parent(args) if rank == 0 else None)[0]
        def local_metadata():
            metadata = {name: sha(args.data_dir/name) for name in audit_receipt['metadata']}
            require(metadata == audit_receipt['metadata'], 'Local parent metadata differs from checked parent')
            return metadata
        audits = collective(distributed, group, local_metadata)
        manifest = json.loads((args.data_dir/'dataset_manifest.json').read_text())
        def datasets():
            train = PreparedImmuneDataset(args.data_dir, split='train')
            valid, selection = {}, {}
            for source in sorted({item['source'] for item in manifest['splits']['valid']['shards']}):
                data = PreparedImmuneDataset(args.data_dir, split='valid', source=source)
                indices = random.Random(42).sample(range(len(data)), min(len(data), args.valid_per_source))
                valid[source], selection[source] = Subset(data, indices), indices
            return train, valid, selection
        prepared = []
        def prepare_data():
            prepared.append(datasets())
            train, valid, selection = prepared[0]
            return {'train_records': len(train), 'validation': selection}
        data_receipts = collective(distributed, group, prepare_data)
        require(all(item == data_receipts[0] for item in data_receipts), 'Rank dataset indices differ')
        train, valid, selection = prepared[0]
        config = ImmuneConfig(ablation='A4', d_model=768, n_layers=8, n_heads=12,
                              mlp_hidden_size=3072, max_length=2048, full_capacity=256, fragment_capacity=64)
        set_seed(args.seed)
        model = FactorialModel(config, args.variant) if args.variant in FACTORIAL else RefinedModel(config, VARIANTS[args.variant])
        model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
        collator = ImmuneCollator('A4', max_length=2048, full_capacity=256, fragment_capacity=64)
        training = TrainingArguments(output_dir=str(args.output_dir),
            per_device_train_batch_size=args.micro_batch, per_device_eval_batch_size=args.micro_batch,
            gradient_accumulation_steps=args.grad_accum, max_steps=args.max_steps,
            learning_rate=args.learning_rate, warmup_steps=args.warmup_steps, lr_scheduler_type='cosine',
            seed=args.seed, data_seed=args.seed, bf16=True, remove_unused_columns=False,
            label_names=['labels'], eval_strategy='steps', eval_steps=args.eval_every,
            save_strategy='steps', save_steps=args.save_every, save_total_limit=None,
            save_only_model=False, save_on_each_node=False, save_safetensors=True,
            logging_steps=25, report_to='none', prediction_loss_only=True, disable_tqdm=True,
            dataloader_num_workers=0, dataloader_drop_last=True, ddp_find_unused_parameters=False,
            optim='adamw_torch', local_rank=0)
        protocol = {key: getattr(metrics, key) for key in factory.PROTOCOL_KEYS}
        resume_observer = None
        class FormalBaseTrainer(metrics.TokenValidationTrainer):
            def create_scheduler(self, num_training_steps, optimizer=None):
                require(num_training_steps == args.max_steps == LR_EXTENSION['total_steps'],
                        'Actual scheduler horizon differs from epoch extension')
                if self.lr_scheduler is None:
                    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        self.optimizer if optimizer is None else optimizer,
                        partial(epoch_restart_factor,
                            first_phase_steps=LR_EXTENSION['first_phase_steps'],
                            total_steps=LR_EXTENSION['total_steps'],
                            warmup_steps=LR_EXTENSION['warmup_steps']))
                    self._created_lr_scheduler = True
                return self.lr_scheduler

            def _load_optimizer_and_scheduler(self, checkpoint):
                native = super()._load_optimizer_and_scheduler
                result = native(checkpoint)
                if checkpoint is not None:
                    collective(distributed, group, lambda: resume_observer.observe(
                        'optimizer_scheduler', self, checkpoint, native))
                return result

            def _load_rng_state(self, checkpoint):
                native = super()._load_rng_state
                result = native(checkpoint)
                if checkpoint is not None:
                    collective(distributed, group, lambda: resume_observer.observe(
                        'rng', self, checkpoint, native))
                return result

            def _get_train_sampler(self, train_dataset=None):
                dataset = self.train_dataset if train_dataset is None else train_dataset
                return EpochRandomSampler(dataset, self.args.data_seed)

            def get_eval_dataloader(self, eval_dataset=None):
                # HF shares this argument between train and eval loaders. Keep
                # training's full windows, but let Accelerate pad/trim eval tails.
                drop_last = self.args.dataloader_drop_last
                try:
                    self.args.dataloader_drop_last = False
                    return super().get_eval_dataloader(eval_dataset)
                finally:
                    self.args.dataloader_drop_last = drop_last
        cls = factory.make_distributed_trainer_class(FormalBaseTrainer, protocol)
        trainer = cls(model=model, args=training, train_dataset=train, eval_dataset=valid, data_collator=collator)
        trainer._evaluation_rank()
        require(not trainer.is_fsdp_enabled and not trainer.is_deepspeed_enabled,
                'This entry supports native DDP only, including launcher environment settings')
        require(training.should_save is (rank == 0), 'Only rank0 may persist shared state')
        sources = verify_sources(args, loaded=True)
        binding = {'schema': 'v7.formal.driver.v1', 'variant': model.variant, 'config': asdict(config),
            'options': model.options, 'seed': args.seed, 'data_seed': args.seed, 'world_size': WORLD,
            'micro_batch': args.micro_batch, 'grad_accum': args.grad_accum, 'global_batch': args.global_batch,
            'data_dir': str(args.data_dir), 'parent_metadata': audits[0], 'train_records': len(train),
            'validation_seed': 42, 'validation_indices': selection, 'source_files': sources,
            'source_root': str(args.source_root), 'repo_root': str(args.repo_root), 'audit_execution': audit_receipt,
            'source_manifest_sha256': args.source_manifest_sha256,
            'schedule': {'max_steps': args.max_steps, 'warmup_steps': args.warmup_steps,
                'learning_rate': args.learning_rate, 'scheduler': 'cosine', 'eval_every': args.eval_every,
                'save_every': args.save_every}, 'optimizer': 'adamw_torch', 'precision': 'fp32_parameters_bf16_autocast',
            'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'accelerate': accelerate.__version__},
            'sampling': 'epoch_random_natural; drop_last=True; accelerator rank partition; no source balancing',
            'global_batch_scope': 'full accumulation windows; epoch-tail windows may contain fewer records',
            'objective': 'existing A4 masked reconstruction; no generation or binary-negative augmentation'}
        binding['checkpoint_policy'] = POLICY
        binding['lr_extension'] = LR_EXTENSION
        binding['transition'] = {'path': str(args.transition), 'sha256': args.transition_sha256}
        collective(distributed, group, lambda: transition(args, binding)[0])
        is_migration = str(args.resume_from_checkpoint) == transition_spec['anchor_checkpoint']
        resume_binding = old_binding if is_migration else binding
        signatures = collective(distributed, group, lambda: digest(binding))
        require(len(set(signatures)) == 1, 'Ranks disagree on full training binding')
        trainer.run_provenance = binding
        def prepare_output():
            if rank != 0:
                return
            args.output_dir.mkdir(parents=True, exist_ok=True)
            if args.resume_from_checkpoint is None or is_migration:
                require(all(path.name in ('train.log', 'worker.log', 'stdout.log', 'stderr.log')
                    and path.is_file() and not path.is_symlink() for path in args.output_dir.iterdir()),
                    'Refusing to overwrite an existing run')
            else:
                require(json.loads((args.output_dir/'run_provenance.json').read_text()) == binding,
                        'Resume output provenance differs')
                require(not (args.output_dir/'COMPLETE.json').exists(), 'Completed run cannot resume')
            helper._atomic_json_once(args.output_dir/'run_provenance.json', binding, allow_identical=True)
        collective(distributed, group, prepare_output)
        trainer.add_callback(checkpoint_callback(trainer, binding, args, helper, torch, group))
        if args.resume_from_checkpoint:
            def resume_gate():
                checkpoint = args.resume_from_checkpoint
                expected_parent = Path(transition_spec['old_output_dir']) if is_migration else args.output_dir
                require(checkpoint.parent == expected_parent, 'Resume owner differs')
                if is_migration:
                    require(sha(checkpoint/SIDECAR) == transition_spec['anchor_sidecar']['sha256'],
                            'Migration anchor changed')
                sidecar = json.loads((checkpoint/SIDECAR).read_text())
                require(sidecar.get('schema') == 'v7.formal.native_checkpoint.v1'
                    and sidecar.get('world_size') == WORLD and sidecar.get('writer_rank') == 0
                    and sidecar.get('rank_acknowledgments') == list(range(WORLD)), 'Resume sidecar differs')
                require(inventory(checkpoint, helper) == sidecar['files'], 'Resume bytes changed')
                saved = parameter_digest(model, checkpoint)
                require(saved == sidecar.get('saved_parameter_digest') and sidecar.get('rank_parameter_digests') ==
                    [{'rank': index, **saved} for index in range(WORLD)], 'Prior actual-rank parameter evidence differs')
                result = validate_checkpoint(checkpoint, resume_binding, helper, torch)
                require(all(sidecar[key] == result[key] for key in result), 'Resume state differs')
                require(result['step'] < (args.stop_after or args.max_steps), 'Resume has no remaining updates')
                return result
            receipts = collective(distributed, group, resume_gate)
            require(all(item == receipts[0] for item in receipts), 'Rank resume states differ')
            resume_observer = ResumeObservation(args.resume_from_checkpoint,
                sha(args.resume_from_checkpoint/SIDECAR), rank, digest(resume_binding),
                {'optimizer_scheduler': transformers.Trainer._load_optimizer_and_scheduler,
                 'rng': transformers.Trainer._load_rng_state})
        with preserve_resume_epoch():
            result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
        if trainer.state.global_step == args.max_steps:
            require(abs(trainer.state.epoch - 2.0) < 1e-9, 'Final native epoch is not exactly two')
        collective(distributed, group, lambda: None)
        if resume_observer is not None:
            restored = collective(distributed, group, lambda: resume_observer.receipt(trainer.state.global_step))
            evidence = complete_receipt(restored)
            if is_migration:
                collective(distributed, group, lambda: helper._atomic_json_once(
                    args.output_dir/'MIGRATION_RESTORE.json', dict(schema='v7.epoch2.native_migration.v1',
                        transition=binding['transition'], old_binding_sha256=digest(old_binding),
                        new_binding_sha256=digest(binding), native_resume=evidence))
                    if rank == 0 else None)
            collective(distributed, group, lambda: helper._atomic_json_once(
                args.output_dir/f'RESUME_{evidence["prior_step"]}_TO_{evidence["final_step"]}.json',
                evidence) if rank == 0 else None)
        if rank == 0:
            trainer.save_metrics('train', result.metrics)
            name = 'COMPLETE.json' if trainer.state.global_step == args.max_steps else f'SEGMENT_{trainer.state.global_step}.json'
            helper._atomic_json_once(args.output_dir/name, {'step': trainer.state.global_step,
                'binding_sha256': digest(binding), 'checkpoint': f'checkpoint-{trainer.state.global_step}',
                'world_size': WORLD, 'status': 'completed' if name == 'COMPLETE.json' else 'paused'})
    finally:
        distributed.destroy_process_group(group)


if __name__ == '__main__':
    main()
