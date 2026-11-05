"""Three maximum-length V11 updates without activation checkpointing, local only.

Activate pllm and preload the frozen NCCL library, then select idle CUDA GPU0:
python -B /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/profile_v11_without_checkpointing.py --output ABS_NEW_DIRECTORY
Reuses the frozen synthetic capacity probe and native trainer; saves no weights.
"""
import argparse
import fcntl
from functools import partial
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT=Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore')
BUNDLE=ROOT/'output/v11_a5_75m_300k_20261105/bundle'


def main(output):
    sys.path.insert(0,str(BUNDLE/'source'))
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from v11_a5.model import V11A5Model, install_trainer_adapter
    from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
    from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator, corrupt
    from scripts.research import profile_bert_mlm as memory
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    plan=json.loads((BUNDLE/'plan.json').read_text())
    trainer=install_trainer_adapter()
    config=FixedChainConfig(**plan['model'])
    model=V11A5Model(config,family_spec=plan['family_prediction']).cuda().train()
    model.backbone.set_activation_checkpointing(None)
    memory.BertMLMCollator=partial(VFamilyCollator,family_spec=plan['family_prediction'])
    clean,tokenizer=memory.synthetic_batch(config,64,config.max_length)
    batch=corrupt(clean,tokenizer,torch.Generator().manual_seed(10042))
    assert tuple(batch['input_ids'].shape)==(64,2048)
    with tempfile.TemporaryDirectory(dir=output,prefix='rendezvous-') as temporary:
        torch.distributed.init_process_group('nccl',init_method='file://'+str(Path(temporary)/'store'),
            rank=0,world_size=1,device_id=torch.device('cuda:0'))
        try:
            ddp=DistributedDataParallel(model,device_ids=[0],broadcast_buffers=False,gradient_as_bucket_view=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
            scheduler=trainer.make_scheduler(optimizer,300000,15000)
            torch.cuda.reset_peak_memory_stats()
            updates=[trainer.update(ddp,batch,optimizer,scheduler,16,1) for _ in range(3)]
            torch.cuda.synchronize()
            free,total=torch.cuda.mem_get_info()
            allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
            external=max(0,total-free-torch.cuda.memory_reserved())
            report=dict(status='passed' if reserved+external<=.85*total else 'insufficient_headroom',
                gpu=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source=str(BUNDLE/'source'),synthetic_capacity_probe=True,sequence_length=2048,
                world_size=1,local_records_per_update=64,micro_batch=16,gradient_accumulation=4,
                whole_layer_checkpointing=False,bf16=True,updates=updates,
                peak_allocated_gib=allocated/2**30,peak_reserved_gib=reserved/2**30,
                total_gib=total/2**30,external_gib=external/2**30,
                formal_training_changed=False,
                limitation='Local maximum-shape memory only; eight-rank performance and resume remain unqualified.')
            (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
        finally:
            torch.distributed.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.output.is_absolute():parser.error('--output must be absolute')
    with (ROOT/'output/arch70m_local_gpu0.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.output.mkdir(parents=True,exist_ok=False)
        main(args.output)
