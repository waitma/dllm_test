"""Bounded local V11 performance comparison, separate from formal training.

Activate pllm, preload the bundle's NCCL library, select idle CUDA GPU0, then:
python -B /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/benchmark_v11_microbatch.py --output ABS_NEW_DIRECTORY
Uses frozen V11 modules and unchanged real V8 inputs; never writes checkpoints.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore')
RUN = ROOT/'output/v11_a5_75m_300k_20261105'
BUNDLE = RUN/'bundle'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main(output, compare_checkpointing=False, checkpoint=None):
    sys.path.insert(0, str(BUNDLE/'source'))
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from v11_a5.model import V11A5Model, install_trainer_adapter
    from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
    from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator, corrupt
    from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
    from scripts.research.bert_mlm_data import IndexedPreparedDataset, CompleteEpochBatchSampler

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    trainer = install_trainer_adapter()
    plan = json.loads((BUNDLE/'plan.json').read_text())
    if checkpoint is None:
        latest = json.loads((RUN/'train/latest.json').read_text())
        checkpoint = Path(latest['checkpoint'])
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        raise ValueError('Checkpoint must be absolute')
    # Load once: checkpoint retention may rotate files during this diagnostic.
    initial_model = V11A5Model.from_pretrained(checkpoint)
    initial = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}
    collator = VFamilyCollator(initial_model.config, family_spec=initial_model.family_spec)
    dataset = IndexedPreparedDataset(plan['data_index'])
    sampler = CompleteEpochBatchSampler(len(dataset), 512, 0, 8, 42, 0,
                                        start_batch=10000, max_batches=4)
    noise = torch.Generator().manual_seed(10042)
    batches, inputs = [], []
    for indices in sampler:
        batch = corrupt(collator([dataset[index] for index in indices]), initial_model.tokenizer, noise)
        digest = hashlib.sha256()
        for key, value in sorted(batch.items()):
            digest.update(key.encode())
            digest.update(value.contiguous().numpy().tobytes())
        inputs.append(dict(indices=indices, shape=list(batch['input_ids'].shape), sha256=digest.hexdigest()))
        batches.append({key: value.pin_memory() for key, value in batch.items()})
    del initial_model, dataset
    report = dict(started_at=datetime.now(timezone.utc).isoformat(),
        scope='Single-GPU native DDP update on cached real rank0 batches; no eight-rank communication, data loading, validation or saving.',
        formal_training_changed=False, gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()), torch=torch.__version__,
        cuda=torch.version.cuda, parameters=sum(value.numel() for value in initial.values()),
        checkpoint=str(checkpoint), source=str(BUNDLE/'source'),
        plan_sha256=hashlib.sha256((BUNDLE/'plan.json').read_bytes()).hexdigest(),
        local_records_per_update=64, equivalent_formal_global_batch=512,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bf16=True, comparison='checkpointing' if compare_checkpointing else 'micro_batch',
        warmup_updates=4, measured_updates=16,
        inputs=inputs, cases=[])
    write(output/'result.json', report)
    with tempfile.TemporaryDirectory(dir=output, prefix='rendezvous-') as temporary:
        torch.distributed.init_process_group('nccl', init_method='file://'+str(Path(temporary)/'store'),
                                            rank=0, world_size=1, device_id=torch.device('cuda:0'))
        try:
            cases = ([('checkpoint_on',16,True),('checkpoint_off',16,False),('checkpoint_on_repeat',16,True)]
                     if compare_checkpointing else
                     [('micro16',16,True),('micro32',32,True),('micro64',64,True),('micro16_repeat',16,True)])
            for name, micro, checkpointing in cases:
                torch.manual_seed(42)
                model = V11A5Model(FixedChainConfig(**plan['model']), family_spec=plan['family_prediction'])
                model.load_state_dict(initial, strict=True)
                model.to('cuda').train()
                model.backbone.set_activation_checkpointing(
                    ActivationCheckpointingStrategy.whole_layer if checkpointing else None)
                ddp = DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False,
                                               gradient_as_bucket_view=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9,.999),
                                               eps=1e-8, weight_decay=.01)
                scheduler = trainer.make_scheduler(optimizer,300000,15000)
                torch.cuda.reset_peak_memory_stats()
                measurements, metrics = [], []
                print(json.dumps(dict(event='case_start',name=name,micro=micro)),flush=True)
                for index in range(20):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    value = trainer.update(ddp,batches[index%len(batches)],optimizer,scheduler,micro,1)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    if index>=4:
                        measurements.append(elapsed)
                        metrics.append(value)
                case = dict(name=name,micro_batch=micro,gradient_accumulation=64//micro,
                    whole_layer_checkpointing=checkpointing,
                    final_parameter_sha256=trainer.parameter_digest(model),
                    seconds_per_update=measurements,mean_seconds=statistics.mean(measurements),
                    median_seconds=statistics.median(measurements),metrics=metrics,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                if name=='micro16':
                    # Instrumentation is excluded from the timed measurements.
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        trainer.update(ddp,batches[0],optimizer,scheduler,micro,1)
                        torch.cuda.synchronize()
                    events = [dict(name=event.key,count=event.count,cpu_total_us=event.cpu_time_total,
                                   self_cpu_us=event.self_cpu_time_total,
                                   device_total_us=event.device_time_total,
                                   self_device_us=event.self_device_time_total)
                              for event in profile.key_averages()]
                    write(output/'micro16_profile_events.json',events)
                report['cases'].append(case)
                write(output/'result.json',report)
                print(json.dumps({key:value for key,value in case.items()
                                  if key not in ('metrics','seconds_per_update')}),flush=True)
                del scheduler, optimizer, ddp, model
                gc.collect()
                torch.cuda.empty_cache()
            report['status']='complete'
            report['finished_at']=datetime.now(timezone.utc).isoformat()
            write(output/'result.json',report)
        finally:
            torch.distributed.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare-checkpointing',action='store_true',
                        help='Compare whole-layer checkpointing on/off/on, at the unchanged micro16.')
    parser.add_argument('--checkpoint',type=Path,
                        help='Read-only checkpoint; default is the formal latest checkpoint at startup.')
    args=parser.parse_args()
    if not args.output.is_absolute():
        parser.error('--output must be absolute')
    with (ROOT/'output/arch70m_local_gpu0.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.output.mkdir(parents=True,exist_ok=False)
        main(args.output,args.compare_checkpointing,args.checkpoint)
