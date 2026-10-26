"""Resume the original frozen diffusion/edit implementation in a replacement pool.

Activate pllm. Copy this transport into a reviewed bundle, then enqueue its
ticket; the persistent supervisor runs eight torchrun ranks with --bundle ABS.
Use --verify-only on the development host before enqueuing the training ticket.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(bundle, checkpoint_files=False):
    launch = json.loads((bundle / 'direct_launch.json').read_text())
    if sha(bundle / 'plan.json') != launch['plan_sha256']:
        raise ValueError('Resume launch plan changed')
    if sha(__file__) != launch['launcher_sha256']:
        raise ValueError('Resume transport changed')
    plan = json.loads((bundle / 'plan.json').read_text())
    contract = plan['resume_contract']
    original_bundle = Path(contract['original_bundle'])
    original_launch = json.loads((original_bundle / 'direct_launch.json').read_text())
    if sha(original_bundle / 'plan.json') != original_launch['plan_sha256']:
        raise ValueError('Original frozen plan changed')
    if original_launch['plan_sha256'] != contract['original_plan_sha256']:
        raise ValueError('Original launch identity differs')
    original = json.loads((original_bundle / 'plan.json').read_text())
    allowed = {'stage', 'created_utc', 'resume_checkpoint', 'resume_contract'}
    if ({k: v for k, v in plan.items() if k not in allowed}
            != {k: v for k, v in original.items() if k not in allowed}):
        raise ValueError('Resume changed training/model/data/schedule/source/W&B settings')
    for section in ('sources', 'vendor_files'):
        for path, expected in plan.get(section, {}).items():
            if sha(path) != expected:
                raise ValueError('Frozen dependency changed: ' + path)
    for gate in plan['gates']:
        if sha(gate['path']) != gate['sha256']:
            raise ValueError('Original training gate changed')
    checkpoint = Path(plan['resume_checkpoint'])
    if sha(checkpoint / 'checkpoint_receipt.json') != contract['checkpoint_receipt_sha256']:
        raise ValueError('Resume checkpoint identity changed')
    receipt = json.loads((checkpoint / 'checkpoint_receipt.json').read_text())
    if (receipt['status'] != 'complete' or receipt['world_size'] != 8
            or len(receipt['rank_parameter_sha256']) != 8
            or len(set(receipt['rank_parameter_sha256'])) != 1
            or receipt['step'] != contract['checkpoint_step']):
        raise ValueError('Resume requires a complete eight-rank checkpoint')
    provenance = {key: plan[key] for key in (
        'model', 'training', 'recipe', 'source_root', 'sources',
        'data_index', 'data_index_sha256', 'parent_checkpoint', 'parent_model_sha256')}
    provenance.update(schema='immune.diffusion_edit.training.v1', world_size=8,
                      normalization='global record-view mean; one original record has two complementary views')
    if plan.get('extension'):
        provenance['extension'] = plan['extension']
    if json.loads((checkpoint / 'run_provenance.json').read_text()) != provenance:
        raise ValueError('Checkpoint provenance differs from the unchanged training plan')
    if checkpoint_files:
        for name, expected in receipt['files'].items():
            if sha(checkpoint / name) != expected:
                raise ValueError('Checkpoint state changed: ' + name)
    if (Path(plan['output_dir']) / 'STOP').exists():
        raise ValueError('Training STOP is present')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if Path(__file__).resolve() != bundle / 'source/scripts/research/launch_v5_direct.py':
        raise ValueError('Execute the bound transport copy')
    plan = verify(bundle, checkpoint_files=args.verify_only)
    source = Path(plan['source_root']).resolve()
    sys.path.insert(0, str(source))
    if plan.get('wandb_vendor'):
        sys.path.insert(0, plan['wandb_vendor'])
    os.environ.update(USE_TF='0', IMMUNE_SDPA_BACKEND='efficient', OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    from scripts.research import run_diffusion_continuation as runtime
    if not Path(runtime.__file__).resolve().is_relative_to(source):
        raise ValueError('Training runtime is not the original frozen source')
    if args.verify_only:
        data = runtime.data_gate(plan)
        parent = runtime.verify_parent(plan)
        result = dict(status='passed', checked_utc=datetime.now(timezone.utc).isoformat(),
                      checkpoint_step=plan['resume_contract']['checkpoint_step'],
                      target_steps=plan['training']['max_steps'],
                      training_computation_retained=True, full_checkpoint_hashes=True,
                      data_gate=data, parent_gate=parent, training_started=False)
        (bundle / 'data_preflight.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    import torch
    rank, world, local = [int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    if world != 8 or torch.cuda.device_count() != 8:
        raise ValueError('Eight allocated GPUs required')
    torch.cuda.set_device(local)
    torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=30))
    try:
        original_restore = runtime.load_checkpoint

        def audited_restore(checkpoint, model, optimizer, scheduler, noise, provenance, rank):
            # Original loader checks all tensors, Adam moments, scheduler and rank RNG.
            state = original_restore(checkpoint, model, optimizer, scheduler, noise, provenance, rank)
            assert state['step'] == plan['resume_contract']['checkpoint_step']
            assert state['seen_records'] == state['step'] * plan['training']['global_batch']
            runtime.write_json(bundle / f'restore_rank{rank}.json', dict(
                status='passed', rank=rank, world_size=world, restored_step=state['step'],
                seen_records=state['seen_records'], scheduler_step=scheduler.last_epoch,
                learning_rate=optimizer.param_groups[0]['lr'],
                model_optimizer_scheduler_rng_checked_by_original_loader=True,
                checked_utc=datetime.now(timezone.utc).isoformat()))
            return state

        runtime.load_checkpoint = audited_restore
        runtime.main(plan, torch.device('cuda', local), rank, world)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
