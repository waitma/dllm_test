"""Run frozen V11 training; invoke the absolute launch_v11_a5_75m.sh path.

--verify-only performs offline CPU source/data/model checks; --resume ABS uses
an explicitly selected V11 checkpoint with the original immutable run settings.
"""
import argparse
from datetime import timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--resume',type=Path);args=p.parse_args()
    from v11_a5.release import checked_plan
    plan,plan_sha=checked_plan(args.bundle)
    if Path(__file__).resolve()!=Path(plan['source_root'])/'v11_a5/launch.py':raise ValueError('Execute frozen V11 source')
    import torch
    from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
    from v11_a5.model import V11A5Model,install_trainer_adapter
    from scripts.research.bert_mlm_data import write_json
    from scripts.research.bert_1b_runtime import configure_wandb
    trainer=install_trainer_adapter()
    versions={n:importlib.metadata.version(n) for n in plan['environment']['packages']}
    if versions!=plan['environment']['packages']:raise ValueError('Runtime packages differ from the 1B environment')
    if args.verify_only:
        model=V11A5Model(FixedChainConfig(**plan['model']),family_spec=plan['family_prediction'])
        count=sum(x.numel() for x in model.parameters())
        if count!=plan['expected_parameters']:raise ValueError('V11 parameter count differs')
        report=dict(status='passed',version='V11',parameters=count,plan_sha256=plan_sha,
            architecture='8 all-global layers',training_identical_to_1b=True,training=plan['training'],
            data=trainer.data_gate(plan),wandb='not_contacted',gpu_training='not_started')
        write_json(Path(plan['run_root'])/'verification.json',report);print(json.dumps(report),flush=True);return
    rank,world,local=(int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    if world!=8 or torch.cuda.device_count()!=8:raise ValueError('Default V11 requires exactly eight visible GPUs')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Set deterministic cuBLAS workspace before CUDA')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.distributed.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=device)
    try:
        out=Path(plan['run_root'])/'train'
        if args.resume:
            cp=args.resume.resolve()
            def check_resume():
                from v11_a5.model import SCHEMA
                if json.loads((cp/'research_config.json').read_text()).get('schema')!=SCHEMA:raise ValueError('Only V11 checkpoints can resume V11')
                if cp.parent!=out.resolve():raise ValueError('Resume checkpoint must belong to this run train directory')
                return str(cp)
            trainer.rank_zero(check_resume,rank)
            plan=dict(plan,resume_checkpoint=str(cp))
        def configure_logging():
            configure_wandb()
            return True
        trainer.rank_zero(configure_logging,rank)
        if rank==0:print(json.dumps(dict(stage='V11_training_start',plan_sha256=plan_sha,
            parameters=plan['expected_parameters'],world_size=world,training=plan['training'])),flush=True)
        trainer.main(dict(plan,phase='train',output_dir=str(out)),device,rank,world)
    finally:torch.distributed.destroy_process_group()


if __name__=='__main__':main()
