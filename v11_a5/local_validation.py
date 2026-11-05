"""Bounded one-GPU validation of V11, never its formal train directory.

Run with NCCL2.31.2 preloaded, CUDA_VISIBLE_DEVICES=0 and --bundle ABS.
Two updates on 16 real records verify inherited DDP update/checkpoint/resume.
"""
import argparse
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
import tempfile
import torch
from torch.nn.parallel import DistributedDataParallel
from safetensors.torch import load_file
from v11_a5.release import checked_plan,write
from v11_a5.model import V11A5Model,install_trainer_adapter
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator,corrupt
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from scripts.research.bert_mlm_data import IndexedPreparedDataset


def main(bundle, output):
    plan,_=checked_plan(bundle);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
    device=torch.device('cuda:0');torch.manual_seed(42)
    trainer=install_trainer_adapter()
    dataset=IndexedPreparedDataset(plan['data_index'])
    indices=plan['smoke_indices'][:8]+plan['smoke_indices'][-8:]
    model=V11A5Model(FixedChainConfig(**plan['model']),family_spec=plan['family_prediction']).to(device)
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    collate=VFamilyCollator(model.config,family_spec=model.family_spec)
    clean=collate([dataset[i] for i in indices]);noise=torch.Generator().manual_seed(10042)
    with tempfile.TemporaryDirectory(dir=output,prefix='ddp-') as rendezvous:
        torch.distributed.init_process_group('nccl',init_method='file://'+str(Path(rendezvous)/'store'),rank=0,world_size=1,device_id=device)
        try:
            ddp=DistributedDataParallel(model,device_ids=[0],broadcast_buffers=False,gradient_as_bucket_view=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
            scheduler=trainer.make_scheduler(optimizer,300000,15000)
            torch.cuda.reset_peak_memory_stats()
            first=trainer.update(ddp,corrupt(clean,model.tokenizer,noise),optimizer,scheduler,16,1)
            state=dict(step=1,epoch=0,next_batch=1,seen_records=16,best_score=-1.,best_step=0)
            provenance=dict(version='V11',validation_only=True,plan_sha256=__import__('hashlib').sha256((Path(bundle)/'plan.json').read_bytes()).hexdigest())
            cp=trainer.save_checkpoint(output,model,optimizer,scheduler,noise,state,provenance,0,1)
            before=trainer.parameter_digest(model)
            second=trainer.update(ddp,corrupt(clean,model.tokenizer,noise),optimizer,scheduler,16,1)
            expected=trainer.parameter_digest(model)
            restored=trainer.load_checkpoint(cp,model,optimizer,scheduler,noise,provenance,0)
            assert restored==state and trainer.parameter_digest(model)==before
            assert isinstance(V11A5Model.from_pretrained(cp),V11A5Model)
            # Replay using identical saved corruption/RNG and the unchanged parent updater.
            replay=trainer.update(ddp,corrupt(clean,model.tokenizer,noise),optimizer,scheduler,16,1)
            replay_digest=trainer.parameter_digest(model)
            report=dict(status='passed',version='V11',parameters=sum(p.numel() for p in model.parameters()),
                actual_gpu=torch.cuda.get_device_name(),world_size=1,records=len(indices),micro_batch=16,
                formal_training_started=False,global_batch_validation_only=16,formal_global_batch=512,
                family_targets=int(clean['family_labels'].ne(-100).sum()),updates=2,replay_updates=1,
                first=first,second=second,replay=replay,exact_replay_parameters=expected==replay_digest,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                limitation='One-GPU validation does not qualify eight-rank execution or worst-case max-length memory.')
            assert all(torch.isfinite(p).all() for p in model.parameters())
            write(output/'result.json',report);print(json.dumps(report),flush=True)
        finally:torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with (Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_local_gpu0.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(a.bundle,a.output)
