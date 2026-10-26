"""Freeze a runnable V8 continuation package, shell entrypoint and eight-GPU YAML.

Activate pllm; run USE_TF=0 python -B -m scripts.research.prepare_bert_unbinding_v8 --steps 13000.
Requires published/indexed V8 and the single-GPU qualification; does not submit.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

from dllm.pipelines.immune_llada.fixed_chain.unbinding_mlm import OBJECTIVE, RECIPE
from scripts.research.bert_mlm_data import sha, write_json, step_budget
from scripts.research.bert_mlm_v8_data import INDEX_DIR, release_binding
from scripts.data.prepare_immune_v8 import OUTPUT
from scripts.research.run_bert_mlm import data_gate, verify_initial_checkpoint

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/bert_unbinding_v8_from287000_20261026'
PARENT = ROOT / 'output/bert_full_token_v7_300k_20261017/train/checkpoint-287000'
PYTHON = '/project_c20250601/251105016/conda/envs/protenix_abtcr/bin/python'


def smoke_indices(index, total):
    positive, negative, offset = [], [], 0
    for shard in index['shards']:
        if shard['split'] != 'train':
            continue
        if Path(shard['path']).name.startswith('abexp_'):
            with Path(shard['path']).open() as stream:
                for i, line in enumerate(stream):
                    relation = json.loads(line).get('labels', {}).get('relation')
                    destination = negative if relation == 'nonbinding' else positive
                    if len(destination) < total // 2:
                        destination.append(offset + i)
                    if min(len(positive), len(negative)) == total // 2:
                        break
        else:
            destination = negative if Path(shard['path']).name.startswith('unbind_') else positive
            missing = total // 2 - len(destination)
            if missing > 0:
                destination.extend(range(offset, offset + min(missing, shard['records'])))
        offset += shard['records']
    if len(positive) != total // 2 or len(negative) != total // 2:
        raise ValueError('Smoke requires both negatives and MLM-eligible rows')
    # Blocks of eight make each rank alternate positive/negative records.
    return [value for i in range(0, total // 2, 8)
            for group in (positive[i:i+8], negative[i:i+8]) for value in group]


def prepare(steps, run=RUN, data_dir=OUTPUT, index_dir=INDEX_DIR, suffix=''):
    if steps < 1:
        raise ValueError('Continuation steps must be positive')
    if not re.fullmatch(r'[a-z0-9_]*', suffix):
        raise ValueError('Invalid launch suffix')
    bound = release_binding(data_dir)
    index_path = index_dir / 'index.json'
    index = json.loads(index_path.read_text())
    profile_path = run / 'qualification/single_gpu_profile.json'
    gpu_path = run / 'qualification/real_data_gpu_check.json'
    profile, gpu = json.loads(profile_path.read_text()), json.loads(gpu_path.read_text())
    if (profile['status'] != 'passed' or gpu['status'] != 'passed'
            or gpu['data_index_sha256'] != sha(index_path)
            or profile['objective_sha256'] != sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/unbinding_mlm.py')):
        raise ValueError('Missing/stale local qualification')
    parent = json.loads((PARENT / 'checkpoint_receipt.json').read_text())
    bundle = run / 'bundle'
    bundle.mkdir(exist_ok=False)
    source = bundle / 'source'
    for directory in ('dllm', 'scripts', 'examples', 'downstream'):
        for path in (ROOT / directory).rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    vendor = ROOT / 'output/bert_mlm_10epoch_20261016/vendor'
    budget = step_budget(bound['records']['train'], 512, steps)
    training = dict(budget_mode='steps', minimum_epochs=0, epochs=budget['epochs_started'], max_steps=steps,
        global_batch=512, micro_batch=profile['micro_batch'], grad_accum=512 // 8 // profile['micro_batch'],
        seed=42, learning_rate=1e-5, warmup_steps=min(500, max(1, steps // 10)), scheduler='cosine',
        weight_decay=.01, adam_betas=[.9, .999], adam_epsilon=1e-8, gradient_clip=1., bf16=True,
        eval_every=1000, log_every=25, loader_workers=4,
        sampler='complete_epoch_no_replacement_no_padding_no_drop_last_then_partial_final_epoch')
    plan = dict(schema='immune.bert_unbinding_v8.launch.v1', run_root=str(run), source_root=str(source),
        model=bound['model'], objective=OBJECTIVE, recipe=RECIPE, training=training, budget=budget,
        data_dir=str(data_dir), data_index=str(index_path), data_index_sha256=sha(index_path),
        parent_checkpoint=str(PARENT), parent_step=287000,
        parent_receipt_sha256=sha(PARENT / 'checkpoint_receipt.json'),
        parent_model_sha256=parent['files']['model.safetensors'],
        target_pretraining_step=287000 + steps,
        initialization='parent_weights; fresh_optimizer_scheduler_sampler_rng',
        smoke_indices=smoke_indices(index, 512 * 3),
        sources={str(p): sha(p) for p in source.rglob('*.py')},
        wandb_vendor=str(vendor), vendor_files={str(p): sha(p) for p in vendor.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'},
        wandb=dict(username='waitma', entity='pressure_jian', project='dllm-immune',
                   run_id='bertv8unbind20261026' + suffix, name='bert-v8-unbinding-from287000-20261026' + suffix,
                   credential_file='/project_c20250601/251105016/.secrets/wandb_api_key'),
        qualification={str(p): sha(p) for p in (profile_path, gpu_path)})
    if not plan['vendor_files']:
        raise ValueError('Missing isolated W&B SDK')
    verify_initial_checkpoint(plan)
    gate = data_gate(plan)
    write_json(bundle / 'plan.json', plan)
    write_json(bundle / 'direct_launch.json', dict(plan_sha256=sha(bundle / 'plan.json')))
    write_json(run / 'preparation.json', dict(status='passed', data_gate=gate,
        bundle=str(bundle), submitted=False, additional_steps=steps, target_pretraining_step=287000+steps))
    launch = run / 'launch.sh'
    launch.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 IMMUNE_SDPA_BACKEND=efficient
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec {PYTHON} -I -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 --no_python {PYTHON} -I -B -u {source}/scripts/research/launch_bert_unbinding_v8.py --bundle {bundle} "$@"
''')
    launch.chmod(0o755)
    peak = profile['reports'][-1]['reserved_gib']
    job = ROOT / f'train_jobs/bert_unbinding_v8_from287000_8gpu_20261026{suffix}.yml'
    job.write_text(f'''TaskName: bert-unbinding-v8-from287000-8gpu-20261026{suffix.replace('_', '-')}
Description: >-
  BERT287000 weights; V8 negatives relation-only, positives full-token MLM.
  Additional {steps} steps; global batch512 = {profile['micro_batch']} x GA{training['grad_accum']} x 8.
  A10080 single-GPU length2048 peak reserved {peak:.2f}GiB; eight-rank profile/replay runs before training.
Storages:
- Type: Vepfs
  MountPath: /project_c20250601/251105016
  VepfsId: vepfs-cnbj2c98dea54433
  SubPath: c20250601/251105016
Entrypoint: bash {launch} --phase all
ImageUrl: cr-mlp-cn-beijing.cr.volces.com/public/airgen:v1
Framework: Custom
ResourceQueueName: ZENITH_MLP2_TASK
TaskRoleSpecs:
- RoleName: worker
  RoleReplicas: 1
  Flavor: ml.pni2.28xlarge
ActiveDeadlineSeconds: 604800
DelayExitTimeSeconds: 0
AccessType: Private
Preemptible: false
Priority: 6
RetryOptions:
  EnableRetry: false
  MaxRetryTimes: 0
''')
    config = ROOT / f'configs/models/v8_bert_unbinding_from287000_20261026{suffix}'
    config.mkdir(exist_ok=False)
    write_json(config / 'immune_config.json', bound['model'])
    write_json(config / 'training_config.json', {k: v for k, v in plan.items()
               if k not in ('sources', 'vendor_files', 'smoke_indices')})
    print(json.dumps(dict(status='prepared', bundle=str(bundle), launch=str(launch), job=str(job),
                          additional_steps=steps, submitted=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=13000)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--data-dir', type=Path, default=OUTPUT)
    parser.add_argument('--index-dir', type=Path, default=INDEX_DIR)
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()
    prepare(args.steps, args.run_dir.resolve(), args.data_dir.resolve(), args.index_dir.resolve(), args.suffix)
