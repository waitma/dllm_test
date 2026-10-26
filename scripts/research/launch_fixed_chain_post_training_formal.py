"""Launch source-bound eight-GPU post-training or native checkpoint continuation.

Activate pllm/protenix_abtcr; use the retained worker pool with --bundle ABS.
--check-only validates metadata; real launch profiles all GPUs before NCCL/training.
The parent model and final confirmation decision must match the frozen launch binding.
"""
import argparse
from datetime import timedelta
import faulthandler
import gc
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


def metadata(bundle):
    bundle = bundle.resolve()
    launch = json.loads((bundle/'direct_launch.json').read_text())
    require(sha(bundle/'launch_training_args.json') == launch['arguments_sha256'], 'Launch arguments changed')
    require(sha(__file__) == launch['launcher_sha256'], 'Launcher changed')
    argv = json.loads((bundle/'launch_training_args.json').read_text())
    source = Path(argv[argv.index('--source-root')+1])
    driver = runpy.run_path(str(source/'scripts/research/run_fixed_chain_post_training_formal.py'), run_name='_direct_metadata')
    args = driver['parse_args'](argv)
    require(sha(args.init_from_checkpoint/'model.safetensors') == args.init_model_sha256,
            'Parent initialization changed')
    require(sha(launch['decision']) == launch['decision_sha256'], 'Final decision changed')
    decision = json.loads(Path(launch['decision']).read_text())
    require(decision['selected_family'] == args.post_recipe, 'Recipe differs from confirmed decision')
    require(sha(args.post_data_index) == args.post_data_index_sha256, 'Full-data index changed')
    require(sha(args.pilot_manifest) == args.pilot_manifest_sha256, 'Pilot validation binding changed')
    require(args.input_layout == 'V5' and args.variant == 'fixed_chain_full_global', 'Unexpected training variant')
    driver['verify_sources'](args)
    if args.resume_from_checkpoint is None:
        require(not args.output_dir.exists(), 'Output exists; refusing duplicate initialization')
    else:
        require((args.resume_from_checkpoint/'formal_checkpoint.json').exists(), 'Complete native resume checkpoint missing')
        if args.continuation_plan:
            require(sha(args.continuation_plan) == args.continuation_plan_sha256, 'Continuation plan changed')
            continuation = json.loads(args.continuation_plan.read_text())
            require(str(args.output_dir) == continuation['output_dir'], 'Continuation output differs')
            if str(args.resume_from_checkpoint) == continuation['parent_checkpoint']:
                require(not args.output_dir.exists(), 'Continuation output already exists')
                require(sha(args.resume_from_checkpoint/'formal_checkpoint.json') == continuation['parent_sidecar_sha256'],
                        'Continuation parent changed')
            else:
                require(args.resume_from_checkpoint.parent == args.output_dir, 'Unexpected continuation checkpoint')
        else:
            require(args.resume_from_checkpoint.parent == args.output_dir, 'Resume checkpoint belongs to another run')
    manifest = json.loads((args.data_dir/'dataset_manifest.json').read_text())
    require(manifest.get('training_ready') is not False, 'Explicit training_ready=false stop remains in effect')
    require(args.micro_batch * args.grad_accum * 8 == args.global_batch == 512, 'Batch configuration differs')
    return launch, argv, driver, args


def profile(args, directory, rank):
    """Whole-model synthetic memory/gradient check; contains no corpus records."""
    import torch
    from transformers import set_seed
    from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
    from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig
    from dllm.pipelines.immune_llada.fixed_chain.post_training_data import PostTrainingCollator
    from scripts.research.run_fixed_chain_post_training import load_model
    from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
    from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
    from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
    from scripts.research.fixed_chain_validation import selected_token_metrics, summarize
    set_seed(args.seed)
    config = FixedChainConfig(ablation='V5', d_model=768, n_layers=8, n_heads=12,
                          mlp_hidden_size=3072, full_capacity=343, max_length=2048)
    collator = PostTrainingCollator(config)
    def row(context_length):
        return BioSeqRecord([BioSeqChain('A'*context_length, 'antigen'),
                             BioSeqChain('A'*166, 'antibody_heavy'),
                             BioSeqChain('A'*134, 'antibody_light')],
                            'antibody_antigen', 'synthetic_runtime_profile')
    # 1136 is the audited fixed-chain full-V5 maximum; fail if that audit changes.
    length = 1136
    base_length = collator([row(1)])['input_ids'].shape[1]
    batch = {k: v.cuda() for k, v in collator([row(length-base_length+1)]*args.micro_batch).items()}
    require(batch['input_ids'].shape[1] == length, 'Synthetic profile length differs')
    require(not (batch['eligible_mask'].bool() & ~batch['target_slot_mask'].bool()).any(),
            'Synthetic context unexpectedly supervised')
    model = load_model(args.init_from_checkpoint, args.post_recipe).cuda().train()
    require(any(p.requires_grad for p in model.parameters()), 'No trainable parameters')
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=.01)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**batch)['loss']
        require(torch.isfinite(loss).item(), 'Nonfinite synthetic loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        scores = summarize(selected_token_metrics(model, {k:v[:1] for k,v in batch.items()}, 1.)['sample_totals'])
    require(scores['split_count'] == 0 and scores['terminal_count'] == 2 and scores['chain_count'] == 2, 'Fixed-chain validation failed')
    require(model.recipe == args.post_recipe, 'Profile recipe differs')
    torch.cuda.synchronize()
    report = dict(status='passed', rank=rank, synthetic=True, steps=3, loss=losses,
                  sequence_length=length, micro_batch=args.micro_batch,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  all_parameters_trainable=(args.post_recipe != 'lora'), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
                  gpu=torch.cuda.get_device_name(0), validation=scores, checkpoint_saved=False)
    write_once(directory/f'profile_rank{rank}.json', report)
    del optimizer, model, batch, loss
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return report


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    options = parser.parse_args()
    launch, argv, driver, args = metadata(options.bundle)
    if options.check_only:
        print(json.dumps(dict(status='metadata_passed', initialization='parent_weights_or_native_resume',
                             model_parameters_frozen=(args.post_recipe == 'lora'), global_batch=args.global_batch)))
        return
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    require(os.environ['WORLD_SIZE'] == '8' and 0 <= rank < 8 and rank == local, 'One-node world8 required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    visible = visible if visible != [''] else list(map(str, range(8)))
    require(len(visible) == 8 and len(set(visible)) == 8, 'Exactly eight assigned CUDA devices required')
    require(sys.flags.isolated, 'Launch every rank with Python -I, matching the accepted runtime')
    os.environ.update(CUDA_VISIBLE_DEVICES=visible[local], LOCAL_RANK='0', LOCAL_WORLD_SIZE='1')
    driver['isolate_training_sources'](args)
    # Match the successful previous launcher: cold library imports precede CUDA
    # initialization and communication threads. No model is created here.
    import torch
    import torch.utils.checkpoint
    import accelerate
    import safetensors.torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, PreTrainedModel
    require(not torch.cuda.is_initialized(), 'Cold imports unexpectedly initialized CUDA')
    torch.set_num_threads(1)
    directory = Path(launch['runtime_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    write_once(directory/f'rank{rank}_runtime.json', dict(rank=rank,
        torch=torch.__version__, torch_path=torch.__file__, accelerate=accelerate.__version__,
        isolated=bool(sys.flags.isolated), visible_gpu=visible[local],
        nccl_api_version=torch.cuda.nccl.version(),
        loaded_libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                if any(name in line for name in ('libnccl','libcudart','libtorch_cuda'))})))
    if rank == 0:
        write_once(directory/'STARTED.json', dict(time_unix=time.time(), initialization='parent_weights_or_native_resume',
                   all_parameters_trainable=(args.post_recipe != 'lora'), args=argv))
    torch.cuda.set_device(0)
    require(torch.cuda.device_count() == 1, 'Rank CUDA isolation failed')
    profile(args, directory, rank)
    write_once(directory/f'rank{rank}_before_nccl.json', dict(rank=rank, time_unix=time.time()))
    torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=torch.device('cuda:0'))
    try:
        torch.distributed.barrier()
        write_once(directory/f'rank{rank}_nccl_ready.json', dict(rank=rank, time_unix=time.time()))
        # The native driver verifies all sources and all V5 shard hashes again,
        # resets the seed and initializes parent weights or restores full native state.
        driver['main'](argv)
        write_once(directory/f'rank{rank}_success.json', dict(rank=rank, time_unix=time.time()))
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
