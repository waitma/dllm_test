"""Freeze V10's AirGen architecture with the unchanged V9 scratch-training plan.

Activate pllm; run python -B -m scripts.research.prepare_bert_v10.
This prepares files only. The generated web entrypoint performs CPU verification,
fresh eight-GPU profiling and checkpoint replay before the user starts training.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.v10_airgen import ARCHITECTURE, OPTIONS, V10Model
from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.launch_bert_unbinding_v8 import verify_initialization
from scripts.research.run_bert_mlm import data_model_for_plan

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
REFERENCE = ROOT / 'output/bert_v9_75m_scratch_300k_20261029'
RUN = ROOT / 'output/bert_v10_airgen_75m_scratch_300k_20261101'
PYTHON = '/project_c20250601/251105016/conda/envs/protenix_abtcr/bin/python'


def build_plan(reference, run, source):
    """Change the architecture and run identity; preserve data/objective/budget."""
    plan = deepcopy(reference)
    config = replace(FixedChainConfig(**reference['model']), d_model=1280,
                     n_layers=3, n_heads=20, mlp_hidden_size=5120)
    model = V10Model(config, family_spec=reference['family_prediction'])
    parameters = sum(p.numel() for p in model.parameters())
    plan.update(schema='immune.bert_v10.airgen_scratch.v1', architecture=ARCHITECTURE,
        run_root=str(run), source_root=str(source), data_model=reference['model'],
        model=asdict(config), parameter_budget=dict(parameters=parameters,
            reference_parameters=75741903, relative_difference=parameters/75741903-1,
            selection='Keep AirGen width1280, heads20, FFN5120; reduce33 layers to3; nearest depth to V9 parameter budget'),
        reference_plan=str(REFERENCE / 'bundle/plan.json'),
        reference_plan_sha256=sha(REFERENCE / 'bundle/plan.json'),
        family_spec_path=str(run / 'family_labels/family_spec.json'),
        family_spec_sha256=reference['family_spec_sha256'],
        qualification_policy='New V10 CPU/CUDA checks and fresh eight-rank profile/replay; never reuse V9 GPU receipts')
    plan['wandb'].update(run_id='bertv10airgen75mscratch300k20261101',
                         name='bert-v10-airgen-75m-scratch-300k-20261101')
    for key in ('training','budget','objective','recipe','data_dir','data_index',
                'data_index_sha256','family_prediction','smoke_indices','initialization'):
        assert plan[key] == reference[key], key
    data_model_for_plan(plan)
    verify_initialization(plan)
    return plan


def write_entrypoints(run, source):
    launch = run / 'launch.sh'
    launch.write_text(f'''#!/usr/bin/env bash
# User-launched single-node eight-A100 job. --verify-only uses no GPU.
set -euo pipefail
source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate /project_c20250601/251105016/conda/envs/pllm
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 IMMUNE_SDPA_BACKEND=efficient
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "${{1:-}}" == "--verify-only" ]]; then
  exec {PYTHON} -I -B -u {source / 'scripts/research/launch_bert_unbinding_v8.py'} --bundle {run / 'bundle'} --verify-only
fi
{PYTHON} -I -B -u {source / 'scripts/research/launch_bert_unbinding_v8.py'} --bundle {run / 'bundle'} --verify-only
exec {PYTHON} -I -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 --no_python {PYTHON} -I -B -u {source / 'scripts/research/launch_bert_unbinding_v8.py'} --bundle {run / 'bundle'} "$@"
''')
    launch.chmod(0o755)
    directory = run / 'web_submission'
    directory.mkdir()
    template = (REFERENCE / 'web_submission/entrypoint.sh').read_text()
    template = template.replace(REFERENCE.name, run.name)
    old_digest = sha(REFERENCE / 'launch.sh')
    if old_digest not in template:
        raise ValueError('Reference entrypoint no longer binds its launch script')
    path = directory / 'entrypoint.sh'
    path.write_text(template.replace(old_digest, sha(launch)))
    path.chmod(0o755)
    write_json(directory / 'entrypoint_receipt.json', dict(status='prepared',
        entrypoint=str(path), entrypoint_sha256=sha(path), launch=str(launch),
        launch_sha256=sha(launch), command='bash '+str(path), submitted=False))


def main():
    reference_path = REFERENCE / 'bundle/plan.json'
    reference = json.loads(reference_path.read_text())
    if sha(reference_path) != json.loads((REFERENCE / 'bundle/direct_launch.json').read_text())['plan_sha256']:
        raise ValueError('Reference V9 plan changed')
    if (RUN / 'bundle').exists():
        raise ValueError('Never overwrite a frozen V10 bundle')
    source = RUN / 'bundle/source'
    plan = build_plan(reference, RUN, source)
    family = RUN / 'family_labels'
    family.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(reference['family_spec_path'], plan['family_spec_path'])
    if sha(plan['family_spec_path']) != plan['family_spec_sha256']:
        raise ValueError('V-family labels differ from V9')
    for directory in ('dllm','scripts','downstream'):
        for src in (ROOT / directory).rglob('*.py'):
            dest = source / src.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    plan['sources'] = {str(p):sha(p) for p in source.rglob('*.py')}
    for path,digest in plan['vendor_files'].items():
        if sha(path) != digest:
            raise ValueError('Reference vendor changed: '+path)
    write_json(RUN / 'bundle/plan.json', plan)
    write_json(RUN / 'bundle/direct_launch.json', dict(plan_sha256=sha(RUN / 'bundle/plan.json')))
    reference_files = [Path('/project_c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/modules/mint') / p
        for p in ('modules.py','multihead_attention.py','rotary_embedding.py','model/esm2.py')]
    write_json(RUN / 'architecture_comparison.json', dict(status='prepared',
        architecture=OPTIONS, parameter_budget=plan['parameter_budget'],
        reference_plan_sha256=sha(reference_path), reference_source={str(p):sha(p) for p in reference_files},
        unchanged_keys=['training','budget','objective','recipe','data_dir','data_index',
            'data_index_sha256','family_prediction','smoke_indices','initialization'],
        pretrained_weights_loaded=False, esm_alphabet_loaded=False,
        input_adapter='V9 token+role+scope, chain IDs, position IDs and attention mask retained; no ESM mask-embedding dropout',
        tail_and_eos='Unchanged V9 collator/loss; no repeated-EOS padding introduced',
        submitted=False))
    write_entrypoints(RUN, source)
    from scripts.research.recover_bert_v10_wandb import prepare as prepare_online_wrapper
    prepare_online_wrapper(RUN)
    write_json(RUN / 'preparation.json', dict(status='prepared',
        initialization=verify_initialization(plan), parameters=plan['parameter_budget'],
        plan_sha256=sha(RUN / 'bundle/plan.json'), submitted=False))
    print(json.dumps(dict(status='prepared', run=str(RUN), parameters=plan['parameter_budget'])),flush=True)


if __name__ == '__main__':
    main()
