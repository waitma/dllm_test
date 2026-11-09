"""Prepare the requested V10 QK/SwiGLU/score-bias architecture; do not train.

Activate pllm, then run from the project root:
USE_TF=0 python -B -m scripts.research.prepare_bert_v10_qk_swiglu_bias --output ABS_NEW_RUN
Preserves frozen V10 data and training budget, removes the family auxiliary. Creates a separate
source bundle and a user-launched eight-GPU Shell entrypoint; old checkpoints,
bundles and training receipts are not modified or reused.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import json
import shlex
import shutil
import sys

from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.v10_airgen import ARCHITECTURE as ORIGINAL_ARCHITECTURE
from dllm.pipelines.immune_llada.fixed_chain.v10_airgen_qk_swiglu_bias import (
    ARCHITECTURE, OPTIONS, INPUT_FEATURES, OBJECTIVE, RECIPE, parameter_count, swiglu_hidden_size)
from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.launch_bert_unbinding_v8 import verify_initialization
from scripts.research.run_bert_mlm import data_model_for_plan

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
REFERENCE = ROOT/'output/bert_v10_airgen_75m_scratch_300k_20261101/bundle/plan.json'


def build_plan(reference, run, source, reference_path=REFERENCE):
    """Keep V10's data/targets/schedule; explicitly bind the new architecture."""
    if reference.get('architecture') != ORIGINAL_ARCHITECTURE:
        raise ValueError('Expected the original, frozen V10 AirGen plan')
    data_model_for_plan(reference)
    if not run.is_absolute() or not source.is_absolute():
        raise ValueError('Run and source paths must be absolute')
    cfg = replace(FixedChainConfig(**reference['model']), mlp_hidden_size=swiglu_hidden_size(1280))
    count = parameter_count(cfg)
    previous = reference['parameter_budget']['parameters']
    plan = deepcopy(reference)
    for key in ('family_prediction','family_spec_path','family_spec_sha256','smoke_family_records'):
        plan.pop(key, None)
    plan.update(schema='immune.bert_v10.airgen_qk_swiglu_bias_scratch.v7', architecture=ARCHITECTURE,
        objective=OBJECTIVE, recipe=RECIPE,
        input_features=list(INPUT_FEATURES),
        run_root=str(run), source_root=str(source), model=asdict(cfg),
        reference_plan=str(reference_path), reference_plan_sha256=sha(reference_path),
        parameter_budget=dict(parameters=count, reference_parameters=previous,
            relative_difference=count/previous-1,
            selection='Retain3x1280/20heads; ESMC SwiGLU ceil256(8d/3)=3584 with bias-free linears; '
                      'four gain-only QK LayerNorms/layer; independent generic_inter/Pair/Recognition bias per head/layer, 180 total; '
                      'token+role input without scope embedding or scope_ids; '
                      'one zero-initialized dynamic type-feature feedback gate per layer, 3 total; '
                      'no family classification head, labels or auxiliary loss; '
                      'exact ESMC FFN rounding supersedes the original 1pct parameter-match policy'),
        qualification_policy='Require fresh CPU/CUDA checks and eight-rank profile/replay for this exact source bundle; '
                             'never reuse original V10 qualification receipts')
    identity = hashlib.sha256(str(run).encode()).hexdigest()[:16]
    plan['wandb'].update(run_id='v10qkswb'+identity,name=run.name)
    unchanged = ['data_model','training','budget','data_dir','data_index',
                 'data_index_sha256','smoke_indices','initialization']
    for key in unchanged:
        if plan[key] != reference[key]:
            raise ValueError('Unrequested V10 change: '+key)
    data_model_for_plan(plan)
    verify_initialization(plan)
    return plan


def write_entrypoint(run, source):
    """A runnable entrypoint; the caller never executes or submits it."""
    python = shlex.quote(sys.executable)
    launcher = shlex.quote(str(source/'scripts/research/launch_bert_unbinding_v8.py'))
    bundle = shlex.quote(str(run/'bundle'))
    path = run/'launch.sh'
    path.write_text(f'''#!/usr/bin/env bash
# User-launched: fresh eight-GPU profile/replay gates precede training.
# CPU binding verification only: bash ABS_LAUNCH --verify-only
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /project_c20250601/251105016/conda/envs/pllm
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 IMMUNE_SDPA_BACKEND=efficient
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "${{1:-}}" == "--verify-only" ]]; then
  exec {python} -I -B -u {launcher} --bundle {bundle} --verify-only
fi
exec {python} -I -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 --no_python {python} -I -B -u {launcher} --bundle {bundle} "$@"
''')
    path.chmod(0o755)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    reference_path,run = args.reference,args.output
    if not reference_path.is_absolute() or not run.is_absolute():
        raise ValueError('Use absolute paths')
    if run.exists():
        raise FileExistsError(run)
    reference = json.loads(reference_path.read_text())
    receipt = json.loads((reference_path.parent/'direct_launch.json').read_text())
    if sha(reference_path) != receipt['plan_sha256']:
        raise ValueError('Original V10 plan hash changed')
    source = run/'bundle/source'
    plan = build_plan(reference,run,source,reference_path)
    # Verify the few inputs used for preparation before creating any output.
    if sha(plan['data_index']) != plan['data_index_sha256']:
        raise ValueError('V10 data index changed')
    for path,digest in plan['vendor_files'].items():
        if sha(path) != digest:
            raise ValueError('Frozen vendor changed: '+path)
    for directory in ['dllm','scripts','downstream']:
        for path in (ROOT/directory).rglob('*.py'):
            target = source/path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    plan['sources'] = {str(path):sha(path) for path in source.rglob('*.py')}
    write_json(run/'bundle/plan.json',plan)
    write_json(run/'bundle/direct_launch.json',dict(plan_sha256=sha(run/'bundle/plan.json')))
    entrypoint = write_entrypoint(run,source)
    write_json(run/'preparation.json',dict(status='prepared',architecture=ARCHITECTURE,
        architecture_options=OPTIONS,parameter_budget=plan['parameter_budget'],
        plan_sha256=sha(run/'bundle/plan.json'),launch=str(entrypoint),launch_sha256=sha(entrypoint),
        submitted=False,formal_training_started=False,fresh_eight_rank_qualification_required=True,
        initialization=verify_initialization(plan)))
    print(json.dumps(dict(status='prepared',run=str(run),parameters=plan['parameter_budget'],
                          formal_training_started=False)),flush=True)


if __name__=='__main__':
    main()
