"""Resume OAS-only fixed-chain training from a complete native checkpoint.

Activate conda pllm. CPU check:
python -I -B THIS_FILE --spec ABS --check-only
The submitted YAML profiles eight allocated GPUs, then continues the original
OAS run. Frozen source and hyperparameters stay the parent scratch bundle.
W&B uses the current worker credentials and an explicit project; it does not
attach a previous run.
"""
import argparse
from dataclasses import asdict, is_dataclass
from datetime import timedelta
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def spec_digest(spec):
    payload = {key: value for key, value in spec.items() if key != 'spec_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    require(spec.get('schema') == 'immune.oas_resume_launch.v1', 'Unexpected resume spec')
    require(spec_digest(spec) == spec['spec_sha256'], 'Resume spec changed')
    return spec


def metadata(spec):
    require(sha(Path(__file__)) == spec['launcher_sha256'], 'Resume launcher changed')
    parent = Path(spec['parent_bundle']).resolve()
    launch = json.loads((parent / 'direct_launch.json').read_text())
    require(sha(parent / 'launch_training_args.json') == launch['arguments_sha256'],
            'Parent launch arguments changed')
    require(sha(parent / 'source/scripts/research/launch_v5_direct.py') == launch['launcher_sha256'],
            'Parent scratch launcher changed')
    argv = json.loads((parent / 'launch_training_args.json').read_text())
    checkpoint = Path(spec['resume_from_checkpoint']).resolve()
    require(checkpoint.name == 'checkpoint-11000' and checkpoint.parent == Path(spec['output_dir']).resolve(),
            'Resume checkpoint is not this run\'s latest complete save')
    require(sha(checkpoint / 'formal_checkpoint.json') == spec['resume_sidecar_sha256'],
            'Resume sidecar changed')
    argv = list(argv) + ['--resume-from-checkpoint', str(checkpoint)]
    source = Path(argv[argv.index('--source-root') + 1])
    parent_launcher = runpy.run_path(str(source / 'scripts/research/launch_v5_direct.py'),
                                     run_name='_oas_parent_launcher')
    driver = runpy.run_path(str(source / 'scripts/research/run_v7_formal.py'),
                            run_name='_oas_resume_metadata')
    args = driver['parse_args'](argv)
    require(args.resume_from_checkpoint == checkpoint and args.stop_after == 0,
            'Resume arguments differ')
    require(args.input_layout == 'V5' and args.variant == 'fixed_chain_full_global',
            'Unexpected training variant')
    require(args.output_dir == Path(spec['output_dir']).resolve(), 'Resume output differs')
    require(not (args.output_dir / 'COMPLETE.json').exists(), 'Completed run cannot resume')
    driver['verify_sources'](args)
    sidecar = json.loads((checkpoint / 'formal_checkpoint.json').read_text())
    require(sidecar['schema'] == 'v7.formal.native_checkpoint.v1' and sidecar['step'] == 11000
            and sidecar['world_size'] == 8 and sidecar['rank_acknowledgments'] == list(range(8)),
            'Resume sidecar incomplete')
    provenance = json.loads((args.output_dir / 'run_provenance.json').read_text())
    require(provenance['source_root'] == str(args.source_root)
            and provenance['source_manifest_sha256'] == args.source_manifest_sha256
            and provenance['data_selection_sha256'] == args.data_selection_sha256,
            'Existing run provenance is not this OAS bundle')
    manifest = json.loads((args.data_dir / 'dataset_manifest.json').read_text())
    require(manifest.get('training_ready') is not False, 'Explicit training_ready=false stop remains in effect')
    require(args.micro_batch * args.grad_accum * 8 == args.global_batch == 512,
            'Batch configuration differs')
    return launch, argv, driver, args, parent_launcher


def install_wandb_config_adapter(source_root):
    """Adapt the loaded frozen config, without changing its source or fields."""
    from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig

    module = sys.modules[FixedChainConfig.__module__]
    expected = Path(source_root).resolve() / 'dllm/pipelines/immune_llada/fixed_chain/model.py'
    require(Path(module.__file__).resolve() == expected,
            'W&B config adapter must target the bound training source')
    require(is_dataclass(FixedChainConfig), 'Expected a dataclass training config')
    if not hasattr(FixedChainConfig, 'to_dict'):
        FixedChainConfig.to_dict = asdict
    config = FixedChainConfig()
    require(config.to_dict() == asdict(config), 'W&B config serialization differs')
    json.dumps(config.to_dict(), allow_nan=False)
    return dict(config_source=str(expected), config_source_sha256=sha(expected),
                serialization='dataclasses.asdict')


def enable_wandb(spec, source_root):
    wandb = spec['wandb']
    require(wandb['enabled'] is True and wandb['resume_old_run'] is False,
            'This resume must start a new W&B run')
    require(os.environ.get('WANDB_PROJECT') == wandb['project'],
            'WANDB_PROJECT must be set in the worker Entrypoint to this job project')
    require(os.environ.get('WANDB_ENTITY') == wandb['entity'],
            'WANDB_ENTITY must be the current account default entity')
    require(bool(os.environ.get('WANDB_API_KEY')),
            'WANDB_API_KEY missing; use the current waitma secret file')
    require(not os.environ.get('WANDB_RUN_ID'),
            'Do not attach a previous W&B run id after the account switch')
    require(os.environ.get('WANDB_RESUME', 'never') in ('', 'never', 'allow'),
            'Do not force-resume a previous W&B run')
    adapter = install_wandb_config_adapter(source_root)
    os.environ['WANDB_RESUME'] = 'never'
    from transformers import TrainingArguments
    original = TrainingArguments.__init__
    run_name = wandb['run_name']

    def patched(self, *a, **kwargs):
        if kwargs.get('report_to') == 'none':
            kwargs['report_to'] = ['wandb']
            kwargs['run_name'] = run_name
        return original(self, *a, **kwargs)

    TrainingArguments.__init__ = patched
    return adapter


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    options = parser.parse_args()
    spec = load_spec(options.spec)
    launch, argv, driver, args, parent_launcher = metadata(spec)
    if options.check_only:
        driver['isolate_training_sources'](args)
        adapter = install_wandb_config_adapter(args.source_root)
        driver['verify_sources'](args, loaded=True)
        print(json.dumps(dict(status='metadata_passed', initialization='resume_11000',
                             model_parameters_frozen=False, global_batch=args.global_batch,
                             resume_from_checkpoint=str(args.resume_from_checkpoint),
                             wandb_project=spec['wandb']['project'],
                             wandb_entity=spec['wandb']['entity'],
                             wandb_run_name=spec['wandb']['run_name'],
                             wandb_config_adapter=adapter)))
        return
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    require(os.environ['WORLD_SIZE'] == '8' and 0 <= rank < 8 and rank == local, 'One-node world8 required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    visible = visible if visible != [''] else list(map(str, range(8)))
    require(len(visible) == 8 and len(set(visible)) == 8, 'Exactly eight assigned CUDA devices required')
    require(sys.flags.isolated, 'Launch every rank with Python -I, matching the accepted runtime')
    os.environ.update(CUDA_VISIBLE_DEVICES=visible[local], LOCAL_RANK='0', LOCAL_WORLD_SIZE='1')
    driver['isolate_training_sources'](args)
    import torch
    import torch.utils.checkpoint
    import accelerate
    import safetensors.torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, PreTrainedModel
    require(not torch.cuda.is_initialized(), 'Cold imports unexpectedly initialized CUDA')
    adapter = enable_wandb(spec, args.source_root)
    require(not torch.cuda.is_initialized(), 'W&B config adapter unexpectedly initialized CUDA')
    torch.set_num_threads(1)
    directory = Path(spec['runtime_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    write_once(directory / f'rank{rank}_runtime.json', dict(
        rank=rank, torch=torch.__version__, torch_path=torch.__file__,
        accelerate=accelerate.__version__, isolated=bool(sys.flags.isolated),
        visible_gpu=visible[local], nccl_api_version=torch.cuda.nccl.version(),
        wandb_config_adapter=adapter,
        loaded_libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                 if any(name in line for name in ('libnccl', 'libcudart', 'libtorch_cuda'))})))
    if rank == 0:
        write_once(directory / 'STARTED.json', dict(
            time_unix=time.time(), initialization='resume_11000',
            all_parameters_trainable=True, args=argv,
            wandb=spec['wandb']))
    torch.cuda.set_device(0)
    require(torch.cuda.device_count() == 1, 'Rank CUDA isolation failed')
    parent_launcher['profile'](args, directory, rank)
    write_once(directory / f'rank{rank}_before_nccl.json', dict(rank=rank, time_unix=time.time()))
    torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=torch.device('cuda:0'))
    try:
        torch.distributed.barrier()
        write_once(directory / f'rank{rank}_nccl_ready.json', dict(rank=rank, time_unix=time.time()))
        driver['main'](argv)
        write_once(directory / f'rank{rank}_success.json', dict(rank=rank, time_unix=time.time()))
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
