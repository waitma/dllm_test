"""Run frozen V8 profile, mixed-data DDP replay checks, then BERT continuation.

Run with eight torchrun ranks: THIS_FILE --bundle ABS --phase all
Separate phases profile/smoke/train are available; train requires matching
eight-rank profile and smoke receipts produced by this exact frozen bundle.
"""
import argparse
from copy import deepcopy
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import sys


def checked_plan(bundle):
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    launch = json.loads((bundle / 'direct_launch.json').read_text())
    if digest(bundle / 'plan.json') != launch['plan_sha256']:
        raise ValueError('V8 frozen plan changed')
    plan = json.loads((bundle / 'plan.json').read_text())
    if Path(__file__).resolve() != Path(plan['source_root']) / 'scripts/research/launch_bert_unbinding_v8.py':
        raise ValueError('Execute the frozen V8 launcher')
    for path, expected in {**plan['sources'], **plan['vendor_files']}.items():
        if digest(path) != expected:
            raise ValueError('Frozen V8 dependency changed: ' + path)
    return plan, launch['plan_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--phase', choices=('profile', 'smoke', 'train', 'all'), default='all')
    args = parser.parse_args()
    plan, plan_sha = checked_plan(args.bundle.resolve())
    sys.path.insert(0, plan['source_root'])
    sys.path.insert(0, plan['wandb_vendor'])
    os.environ.update(USE_TF='0', IMMUNE_SDPA_BACKEND='efficient', OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    from scripts.research.bert_mlm_data import write_json, sha
    from scripts.research.run_bert_mlm import main as train, rank_zero, data_gate, verify_initial_checkpoint
    from scripts.research.profile_bert_mlm import main as profile
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 8 or torch.cuda.device_count() != 8:
        raise ValueError('Formal V8 continuation requires one node with eight GPUs')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=30))
    root = Path(plan['run_root'])
    try:
        rank_zero(lambda: verify_initial_checkpoint(plan), rank)
        rank_zero(lambda: data_gate(plan), rank)
        if args.phase in ('all', 'profile'):
            phase_plan = dict(plan, phase='profile', output_dir=str(root / 'profile'))
            profile(phase_plan, device, rank, world)
            rank_zero(lambda: write_json(root / 'profile/qualification.json', dict(
                plan_sha256=plan_sha, profile_sha256=sha(root / 'profile/profile.json'))), rank)
        if args.phase == 'profile':
            return
        def profile_gate():
            receipt = json.loads((root / 'profile/qualification.json').read_text())
            profiled = json.loads((root / 'profile/profile.json').read_text())
            if (receipt['plan_sha256'] != plan_sha or receipt['profile_sha256'] != sha(root / 'profile/profile.json')
                    or profiled['status'] != 'passed' or profiled['world_size'] != world
                    or profiled['sequence_length'] != plan['model']['max_length']
                    or profiled['global_batch'] != plan['training']['global_batch']):
                raise ValueError('Missing/stale eight-rank V8 profile')
            return profiled
        profiled = rank_zero(profile_gate, rank)
        plan['training'].update(micro_batch=profiled['micro_batch'], grad_accum=profiled['grad_accum'])
        gc.collect()
        torch.cuda.empty_cache()
        if args.phase in ('all', 'smoke'):
            smoke = deepcopy(plan)
            smoke.update(phase='smoke', output_dir=str(root / 'smoke'))
            smoke['wandb']['run_id'] += 'smoke'
            smoke['wandb']['name'] += '-smoke'
            train(smoke, device, rank, world)
            def smoke_receipt():
                complete = json.loads((root / 'smoke/COMPLETE.json').read_text())
                if complete != dict(status='passed', world_size=world):
                    raise ValueError('Eight-rank V8 smoke incomplete')
                files = [root / 'smoke' / f'smoke_rank{r}.json' for r in range(world)]
                files += [root / 'smoke/run_provenance.json', root / 'smoke/COMPLETE.json']
                write_json(root / 'smoke/qualification.json', dict(plan_sha256=plan_sha,
                    profile_sha256=sha(root / 'profile/profile.json'), files={str(p): sha(p) for p in files}))
            rank_zero(smoke_receipt, rank)
        if args.phase == 'smoke':
            return
        def smoke_gate():
            receipt = json.loads((root / 'smoke/qualification.json').read_text())
            if (receipt['plan_sha256'] != plan_sha or receipt['profile_sha256'] != sha(root / 'profile/profile.json')
                    or any(sha(p) != h for p, h in receipt['files'].items())):
                raise ValueError('Eight-rank V8 smoke evidence changed')
            for r in range(world):
                evidence = json.loads((root / 'smoke' / f'smoke_rank{r}.json').read_text())
                if not (evidence['status'] == 'passed' and evidence['restored_state_exact']
                        and evidence['corruption_exact'] and evidence['math_oracle']['status'] == 'passed'):
                    raise ValueError('Eight-rank V8 replay checks failed')
            return True
        rank_zero(smoke_gate, rank)
        gc.collect()
        torch.cuda.empty_cache()
        train(dict(plan, phase='train', output_dir=str(root / 'train')), device, rank, world)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
